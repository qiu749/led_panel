from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import winsound
except ImportError:  # pragma: no cover - non-Windows fallback.
    winsound = None

try:
    cv2.setLogLevel(0)
except Exception:
    pass

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - reported in the UI at runtime.
    raise RuntimeError("Pillow is required to run the desktop app. Install pillow first.") from exc

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


ROOT = Path(__file__).resolve().parent


def resource_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return ROOT


def runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return ROOT


RESOURCE_ROOT = resource_root()
RUNTIME_ROOT = runtime_root()

SCRIPTS_DIR = RESOURCE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from try_overexposed_point_detection import Params as TraditionalParams
    from try_overexposed_point_detection import detect_points as detect_traditional_points
    from try_overexposed_point_detection_v4_filtered import FilterParams
    from try_overexposed_point_detection_v4_filtered import filter_points

    TRADITIONAL_AVAILABLE = True
    TRADITIONAL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - shown in UI.
    TraditionalParams = None
    FilterParams = None
    detect_traditional_points = None
    filter_points = None
    TRADITIONAL_AVAILABLE = False
    TRADITIONAL_IMPORT_ERROR = str(exc)

try:
    import cv_pattern_probe as cvp
    import cv_template_probe as cv_template

    CV_TEMPLATE_AVAILABLE = True
    CV_TEMPLATE_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - shown in UI.
    cvp = None
    cv_template = None
    CV_TEMPLATE_AVAILABLE = False
    CV_TEMPLATE_IMPORT_ERROR = str(exc)


APP_TITLE = "组件点亮质检测试"
ROI_MODEL_DEFAULT = RESOURCE_ROOT / "models" / "roi_led_model_v2" / "weights" / "best.pt"
POINT_MODEL_CANDIDATES = [
    RESOURCE_ROOT / "models" / "led_panel_yolo11m" / "weights" / "best.pt",
    RESOURCE_ROOT / "models" / "led_panel" / "weights" / "best.pt",
]
CV_TEMPLATE_MODEL_DEFAULT = RESOURCE_ROOT / "models" / "cv_templates" / "templates.json"
MARQUEE_TEMPLATE_MODEL_DEFAULT = RESOURCE_ROOT / "models" / "marquee_templates" / "default_marquee.json"
OUTPUT_DEFAULT = RUNTIME_ROOT / "output" / "desktop_runs"
LOG_DIR = RUNTIME_ROOT / "logs"
LOG_FILE = LOG_DIR / "desktop_app.log"
CV_TEMPLATE_METHOD = "模板匹配CV"

ROI_EXPECTED_POINTS = {
    "ledgroup": 50,
    "ledgroup2": 26,
    "ledgroup3": 24,
}


@dataclass
class CameraInfo:
    key: str
    label: str
    kind: str
    index: int | None = None
    sdk_index: int | None = None
    tlayer_type: int | None = None


@dataclass
class DetectionBox:
    bbox: list[int]
    confidence: float
    class_id: int
    class_name: str


class LightController:
    """First-version no-op light controller, kept to preserve the final flow shape."""

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def set_scene(self, scene_name: str) -> None:
        return None

    def set_channel(self, channel: int, enabled: bool) -> None:
        return None

    def turn_on(self) -> None:
        return None

    def turn_off(self) -> None:
        return None

    def wait_stable(self, milliseconds: int = 200) -> None:
        time.sleep(max(milliseconds, 0) / 1000.0)

    def get_status(self) -> dict[str, Any]:
        return {"mode": "mock", "connected": True}


class ModelCache:
    def __init__(self, log: Callable[[str], None]) -> None:
        self._models: dict[str, Any] = {}
        self._log = log

    def get(self, model_path: str) -> Any:
        if YOLO is None:
            raise RuntimeError("未安装 ultralytics，无法加载 YOLO 模型。")
        path = str(Path(model_path).expanduser())
        if not Path(path).exists():
            raise FileNotFoundError(f"模型不存在: {path}")
        if path not in self._models:
            self._log(f"加载模型: {path}")
            self._models[path] = YOLO(path)
        return self._models[path]


class OpenCVCameraStream:
    def __init__(self, camera: CameraInfo, log: Callable[[str], None]) -> None:
        self.camera = camera
        self.log = log
        self.cap: cv2.VideoCapture | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_time = 0.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self.camera.kind != "opencv" or self.camera.index is None:
            raise RuntimeError("第一版仅支持 OpenCV 摄像头预览；海康 SDK 会在后续接入。")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="camera-preview", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def _loop(self) -> None:
        index = int(self.camera.index or 0)
        self.log(f"打开摄像头: {self.camera.label}")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            self.log(f"无法打开摄像头: {self.camera.label}")
            return

        self.cap = cap
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            with self._lock:
                self.latest_frame = frame.copy()
                self.latest_frame_time = time.time()
            time.sleep(0.005)
        cap.release()
        self.cap = None
        self.log("预览已停止")


_HIK_MODULE: Any | None = None
_HIK_IMPORT_ERROR = ""
_HIK_SDK_INITIALIZED = False
_HIK_LOCK = threading.Lock()
_HIK_DLL_DIRS: list[Path] = []


def append_log_file(message: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LOG_FILE.open("a", encoding="utf-8").write(f"[{stamp}] {message}\n")
    except Exception:
        pass


def hik_import_dirs() -> list[Path]:
    candidates: list[Path] = []
    candidates.append(RESOURCE_ROOT / "MvImport")
    env_root = os.getenv("MVCAM_COMMON_RUNENV")
    if env_root:
        candidates.append(Path(env_root) / "Samples" / "Python" / "MvImport")
    candidates.extend(
        [
            Path(r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"),
            Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
        ]
    )
    return candidates


def add_hik_dll_dirs() -> None:
    global _HIK_DLL_DIRS
    installed_candidates = [
        Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"),
        Path(r"C:\Program Files\Common Files\MVS\Runtime\Win64_x64"),
    ]
    bundled_candidates = [
        RUNTIME_ROOT,
        RUNTIME_ROOT / "mvs_runtime",
        RESOURCE_ROOT / "mvs_runtime",
        RESOURCE_ROOT,
    ]
    candidates = [path for path in installed_candidates if (path / "MvCameraControl.dll").exists()]
    if not candidates:
        candidates = [path for path in bundled_candidates if (path / "MvCameraControl.dll").exists()]

    _HIK_DLL_DIRS = candidates
    for dll_dir in candidates:
        if not dll_dir.exists():
            continue
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(dll_dir))
        except OSError:
            pass
        current_path = os.environ.get("PATH", "")
        if str(dll_dir) not in current_path:
            os.environ["PATH"] = str(dll_dir) + os.pathsep + current_path
    append_log_file("MVS DLL dirs: " + ("; ".join(str(path) for path in _HIK_DLL_DIRS) or "none"))


def load_hik_module() -> Any | None:
    global _HIK_MODULE, _HIK_IMPORT_ERROR
    if _HIK_MODULE is not None:
        return _HIK_MODULE
    add_hik_dll_dirs()
    for import_dir in hik_import_dirs():
        if import_dir.exists() and str(import_dir) not in sys.path:
            sys.path.insert(0, str(import_dir))
    try:
        _HIK_MODULE = import_module("MvCameraControl_class")
        _HIK_IMPORT_ERROR = ""
        append_log_file(f"MVS Python module: {getattr(_HIK_MODULE, '__file__', 'unknown')}")
        return _HIK_MODULE
    except Exception as exc:
        _HIK_IMPORT_ERROR = str(exc)
        append_log_file(f"MVS Python module import failed: {_HIK_IMPORT_ERROR}")
        return None


def initialize_hik_sdk() -> Any | None:
    global _HIK_SDK_INITIALIZED
    module = load_hik_module()
    if module is None:
        return None
    with _HIK_LOCK:
        if not _HIK_SDK_INITIALIZED:
            ret = module.MvCamera.MV_CC_Initialize()
            if ret != 0:
                append_log_file(f"MVS SDK initialize failed: 0x{ret:x}")
                raise RuntimeError(f"海康 MVS SDK 初始化失败: 0x{ret:x}")
            _HIK_SDK_INITIALIZED = True
            try:
                version = module.MvCamera.MV_CC_GetSDKVersion()
                append_log_file(f"MVS SDK version: 0x{version:x}")
            except Exception as exc:
                append_log_file(f"MVS SDK version read failed: {exc}")
    return module


def hik_tlayer_type(module: Any) -> int:
    names = [
        "MV_GIGE_DEVICE",
        "MV_USB_DEVICE",
        "MV_GENTL_GIGE_DEVICE",
        "MV_GENTL_CAMERALINK_DEVICE",
        "MV_GENTL_CXP_DEVICE",
        "MV_GENTL_XOF_DEVICE",
    ]
    value = 0
    for name in names:
        value |= int(getattr(module, name, 0))
    return value


def hik_tlayer_scan_groups(module: Any) -> list[tuple[str, int]]:
    groups: list[tuple[str, int]] = []
    classic = int(getattr(module, "MV_GIGE_DEVICE", 0)) | int(getattr(module, "MV_USB_DEVICE", 0))
    if classic:
        groups.append(("GigE/USB", classic))
    for name in (
        "MV_GENTL_GIGE_DEVICE",
        "MV_GENTL_CAMERALINK_DEVICE",
        "MV_GENTL_CXP_DEVICE",
        "MV_GENTL_XOF_DEVICE",
    ):
        value = int(getattr(module, name, 0))
        if value:
            groups.append((name, value))
    return groups


def decode_ctypes_string(value: Any) -> str:
    try:
        raw = memoryview(value).tobytes()
    except TypeError:
        raw = bytes(bytearray(value))
    raw = raw.split(b"\x00", 1)[0]
    for encoding in ("gbk", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def hik_is_gige(module: Any, layer_type: int) -> bool:
    return layer_type in {
        int(getattr(module, "MV_GIGE_DEVICE", -1)),
        int(getattr(module, "MV_GENTL_GIGE_DEVICE", -2)),
    }


def hik_device_label(module: Any, device_info: Any, sdk_index: int) -> str:
    layer_type = int(device_info.nTLayerType)
    if hik_is_gige(module, layer_type):
        info = device_info.SpecialInfo.stGigEInfo
        model = decode_ctypes_string(info.chModelName)
        user_name = decode_ctypes_string(getattr(info, "chUserDefinedName", b""))
        ip = ".".join(
            str((int(info.nCurrentIp) >> shift) & 0xFF)
            for shift in (24, 16, 8, 0)
        )
        name = user_name or model or "GigE"
        return f"海康 GigE {sdk_index}: {name} {ip}"
    if layer_type == int(getattr(module, "MV_USB_DEVICE", -1)):
        info = device_info.SpecialInfo.stUsb3VInfo
        model = decode_ctypes_string(info.chModelName)
        serial = decode_ctypes_string(info.chSerialNumber)
        user_name = decode_ctypes_string(getattr(info, "chUserDefinedName", b""))
        name = user_name or model or "USB"
        suffix = f" SN:{serial}" if serial else ""
        return f"海康 USB {sdk_index}: {name}{suffix}"
    return f"海康相机 {sdk_index}: layer={layer_type}"


def enumerate_hikvision_cameras() -> list[CameraInfo]:
    module = initialize_hik_sdk()
    if module is None:
        append_log_file(f"MVS enumerate skipped: module unavailable: {_HIK_IMPORT_ERROR}")
        return []

    cameras: list[CameraInfo] = []
    errors: list[str] = []
    for group_name, tlayer_type in hik_tlayer_scan_groups(module):
        device_list = module.MV_CC_DEVICE_INFO_LIST()
        ret = module.MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        count = int(device_list.nDeviceNum)
        append_log_file(
            f"MVS enum group={group_name}, ret=0x{ret:x}, tlayer=0x{tlayer_type:x}, count={count}"
        )
        if ret != 0:
            errors.append(f"{group_name}=0x{ret:x}")
            continue
        for sdk_index in range(count):
            device_info = cast(device_list.pDeviceInfo[sdk_index], POINTER(module.MV_CC_DEVICE_INFO)).contents
            label = hik_device_label(module, device_info, sdk_index)
            append_log_file(f"MVS device[{len(cameras)}] sdk_index={sdk_index}: {label}")
            cameras.append(
                CameraInfo(
                    key=f"hik:{len(cameras)}",
                    label=label,
                    kind="hikvision",
                    sdk_index=sdk_index,
                    tlayer_type=tlayer_type,
                )
            )
    if not cameras and errors:
        raise RuntimeError(f"海康相机枚举失败: {'; '.join(errors)}")
    return cameras


def hik_is_hb_pixel_format(module: Any, pixel_type: int) -> bool:
    names = [
        "PixelType_Gvsp_HB_Mono8",
        "PixelType_Gvsp_HB_Mono10",
        "PixelType_Gvsp_HB_Mono10_Packed",
        "PixelType_Gvsp_HB_Mono12",
        "PixelType_Gvsp_HB_Mono12_Packed",
        "PixelType_Gvsp_HB_Mono16",
        "PixelType_Gvsp_HB_RGB8_Packed",
        "PixelType_Gvsp_HB_BGR8_Packed",
        "PixelType_Gvsp_HB_RGBA8_Packed",
        "PixelType_Gvsp_HB_BGRA8_Packed",
        "PixelType_Gvsp_HB_RGB16_Packed",
        "PixelType_Gvsp_HB_BGR16_Packed",
        "PixelType_Gvsp_HB_RGBA16_Packed",
        "PixelType_Gvsp_HB_BGRA16_Packed",
        "PixelType_Gvsp_HB_YUV422_Packed",
        "PixelType_Gvsp_HB_YUV422_YUYV_Packed",
        "PixelType_Gvsp_HB_BayerGR8",
        "PixelType_Gvsp_HB_BayerRG8",
        "PixelType_Gvsp_HB_BayerGB8",
        "PixelType_Gvsp_HB_BayerBG8",
        "PixelType_Gvsp_HB_BayerRBGG8",
        "PixelType_Gvsp_HB_BayerGB10",
        "PixelType_Gvsp_HB_BayerGB10_Packed",
        "PixelType_Gvsp_HB_BayerBG10",
        "PixelType_Gvsp_HB_BayerBG10_Packed",
        "PixelType_Gvsp_HB_BayerRG10",
        "PixelType_Gvsp_HB_BayerRG10_Packed",
        "PixelType_Gvsp_HB_BayerGR10",
        "PixelType_Gvsp_HB_BayerGR10_Packed",
        "PixelType_Gvsp_HB_BayerGB12",
        "PixelType_Gvsp_HB_BayerGB12_Packed",
        "PixelType_Gvsp_HB_BayerBG12",
        "PixelType_Gvsp_HB_BayerBG12_Packed",
        "PixelType_Gvsp_HB_BayerRG12",
        "PixelType_Gvsp_HB_BayerRG12_Packed",
        "PixelType_Gvsp_HB_BayerGR12",
        "PixelType_Gvsp_HB_BayerGR12_Packed",
    ]
    return pixel_type in {int(getattr(module, name)) for name in names if hasattr(module, name)}


def hik_is_mono_pixel_format(module: Any, pixel_type: int) -> bool:
    names = [
        "PixelType_Gvsp_Mono8",
        "PixelType_Gvsp_Mono10",
        "PixelType_Gvsp_Mono10_Packed",
        "PixelType_Gvsp_Mono12",
        "PixelType_Gvsp_Mono12_Packed",
        "PixelType_Gvsp_Mono14",
        "PixelType_Gvsp_Mono16",
    ]
    return pixel_type in {int(getattr(module, name)) for name in names if hasattr(module, name)}


class HikvisionCameraStream:
    def __init__(self, camera: CameraInfo, log: Callable[[str], None]) -> None:
        self.camera = camera
        self.log = log
        self.module: Any | None = None
        self.cam: Any | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_time = 0.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._is_grabbing = False
        self._is_open = False
        self._has_handle = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="hikvision-preview", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._thread = None
        self._close_device()

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def _loop(self) -> None:
        try:
            self._open_device()
            self.log(f"打开海康相机: {self.camera.label}")
            while not self._stop_event.is_set():
                frame = self._grab_bgr_frame(timeout_ms=1000)
                if frame is None:
                    continue
                with self._lock:
                    self.latest_frame = frame
                    self.latest_frame_time = time.time()
        except Exception as exc:
            self.log(f"海康相机取流失败: {exc}")
        finally:
            self._close_device()
            self.log("海康预览已停止")

    def _open_device(self) -> None:
        module = initialize_hik_sdk()
        if module is None:
            raise RuntimeError(f"无法加载海康 MVS Python 绑定: {_HIK_IMPORT_ERROR}")
        self.module = module

        device_list = module.MV_CC_DEVICE_INFO_LIST()
        tlayer_type = int(self.camera.tlayer_type or hik_tlayer_type(module))
        ret = module.MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != 0:
            raise RuntimeError(f"海康相机枚举失败: 0x{ret:x}, tlayer=0x{tlayer_type:x}")
        sdk_index = int(self.camera.sdk_index or 0)
        if sdk_index >= int(device_list.nDeviceNum):
            raise RuntimeError(f"海康相机序号不存在: {sdk_index}")

        device_info = cast(device_list.pDeviceInfo[sdk_index], POINTER(module.MV_CC_DEVICE_INFO)).contents
        cam = module.MvCamera()
        ret = cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            raise RuntimeError(f"创建海康相机句柄失败: 0x{ret:x}")
        self.cam = cam
        self._has_handle = True

        ret = cam.MV_CC_OpenDevice(module.MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"打开海康相机失败: 0x{ret:x}")
        self._is_open = True

        if hik_is_gige(module, int(device_info.nTLayerType)):
            packet_size = cam.MV_CC_GetOptimalPacketSize()
            if int(packet_size) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
                if ret != 0:
                    self.log(f"设置 GigE 包大小失败，继续取流: 0x{ret:x}")

        ret = cam.MV_CC_SetEnumValue("TriggerMode", module.MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise RuntimeError(f"关闭触发模式失败: 0x{ret:x}")

        try:
            ret = cam.MV_CC_SetBayerCvtQuality(1)
            if ret != 0:
                self.log(f"设置 Bayer 转换质量失败，继续取流: 0x{ret:x}")
        except Exception:
            pass

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"海康相机开始取流失败: 0x{ret:x}")
        self._is_grabbing = True
        self._auto_exposure_once()

    def _auto_exposure_once(self) -> None:
        module = self.module
        cam = self.cam
        if module is None or cam is None:
            return

        errors: list[str] = []
        try:
            setter = getattr(cam, "MV_CC_SetEnumValueByString", None)
            if setter is not None:
                ret = setter("ExposureAuto", "Once")
                if ret == 0:
                    self.log("海康相机已触发自动曝光一次: ExposureAuto=Once")
                    time.sleep(0.45)
                    return
                errors.append(f"SetEnumValueByString=0x{ret:x}")
        except Exception as exc:
            errors.append(f"SetEnumValueByString={exc}")

        try:
            ret = cam.MV_CC_SetEnumValue("ExposureAuto", 1)
            if ret == 0:
                self.log("海康相机已触发自动曝光一次: ExposureAuto=1")
                time.sleep(0.45)
                return
            errors.append(f"SetEnumValue=0x{ret:x}")
        except Exception as exc:
            errors.append(f"SetEnumValue={exc}")

        self.log("海康相机自动曝光一次设置失败，继续取流: " + "; ".join(errors))

    def _grab_bgr_frame(self, timeout_ms: int) -> np.ndarray | None:
        module = self.module
        cam = self.cam
        if module is None or cam is None:
            return None

        out_frame = module.MV_FRAME_OUT()
        memset(byref(out_frame), 0, sizeof(out_frame))
        ret = cam.MV_CC_GetImageBuffer(out_frame, timeout_ms)
        if ret != 0 or out_frame.pBufAddr is None:
            return None
        try:
            return self._convert_frame_to_bgr(out_frame)
        finally:
            cam.MV_CC_FreeImageBuffer(out_frame)

    def _convert_frame_to_bgr(self, out_frame: Any) -> np.ndarray:
        module = self.module
        cam = self.cam
        if module is None or cam is None:
            raise RuntimeError("海康相机未打开")

        width = int(out_frame.stFrameInfo.nWidth)
        height = int(out_frame.stFrameInfo.nHeight)
        src_pixel_type = int(out_frame.stFrameInfo.enPixelType)
        src_data = out_frame.pBufAddr
        src_len = int(out_frame.stFrameInfo.nFrameLen)

        decode_buffer = None
        if hik_is_hb_pixel_format(module, src_pixel_type):
            decode_len = width * height * 3
            decode_buffer = (c_ubyte * decode_len)()
            decode_param = module.MV_CC_HB_DECODE_PARAM()
            memset(byref(decode_param), 0, sizeof(decode_param))
            decode_param.pSrcBuf = out_frame.pBufAddr
            decode_param.nSrcLen = src_len
            decode_param.pDstBuf = decode_buffer
            decode_param.nDstBufSize = decode_len
            ret = cam.MV_CC_HBDecode(decode_param)
            if ret != 0:
                raise RuntimeError(f"HB 图像解码失败: 0x{ret:x}")
            src_data = decode_param.pDstBuf
            src_len = int(decode_param.nDstBufLen)
            src_pixel_type = int(decode_param.enDstPixelType)

        is_mono = hik_is_mono_pixel_format(module, src_pixel_type)
        channels = 1 if is_mono else 3
        dst_type = (
            module.PixelType_Gvsp_Mono8
            if is_mono
            else getattr(module, "PixelType_Gvsp_BGR8_Packed", module.PixelType_Gvsp_RGB8_Packed)
        )
        dst_is_rgb = (not is_mono) and dst_type == module.PixelType_Gvsp_RGB8_Packed
        dst_len = width * height * channels
        dst_buffer = (c_ubyte * dst_len)()

        convert_param = module.MV_CC_PIXEL_CONVERT_PARAM_EX()
        memset(byref(convert_param), 0, sizeof(convert_param))
        convert_param.nWidth = width
        convert_param.nHeight = height
        convert_param.pSrcData = src_data
        convert_param.nSrcDataLen = src_len
        convert_param.enSrcPixelType = src_pixel_type
        convert_param.enDstPixelType = dst_type
        convert_param.pDstBuffer = dst_buffer
        convert_param.nDstBufferSize = dst_len

        ret = cam.MV_CC_ConvertPixelTypeEx(convert_param)
        if ret != 0:
            raise RuntimeError(f"海康像素格式转换失败: 0x{ret:x}")

        if is_mono:
            gray = np.frombuffer(dst_buffer, dtype=np.uint8, count=dst_len).reshape(height, width)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).copy()
        image = np.frombuffer(dst_buffer, dtype=np.uint8, count=dst_len).reshape(height, width, 3)
        if dst_is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()

    def _close_device(self) -> None:
        cam = self.cam
        if cam is None:
            return
        if self._is_grabbing:
            try:
                cam.MV_CC_StopGrabbing()
            except Exception:
                pass
            self._is_grabbing = False
        if self._is_open:
            try:
                cam.MV_CC_CloseDevice()
            except Exception:
                pass
            self._is_open = False
        if self._has_handle:
            try:
                cam.MV_CC_DestroyHandle()
            except Exception:
                pass
            self._has_handle = False
        self.cam = None


def enumerate_opencv_cameras(max_index: int = 8) -> list[CameraInfo]:
    cameras: list[CameraInfo] = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        ok = False
        if cap.isOpened():
            ok, _ = cap.read()
        cap.release()
        if ok:
            cameras.append(
                CameraInfo(
                    key=f"opencv:{index}",
                    label=f"OpenCV 摄像头 {index}",
                    kind="opencv",
                    index=index,
                )
            )
    return cameras


def detect_hikvision_sdk_hint() -> str:
    module = load_hik_module()
    if module is None:
        return f"未检测到海康 MVS Python 模块；当前使用 OpenCV 摄像头通道。{_HIK_IMPORT_ERROR}"
    dll_dirs = "; ".join(str(path) for path in _HIK_DLL_DIRS) or "未找到"
    return f"已检测到海康 MVS Python 模块，MVS运行库: {dll_dirs}"


def choose_default_point_model() -> str:
    for candidate in POINT_MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return str(POINT_MODEL_CANDIDATES[0])


def run_yolo_detect(
    model: Any,
    image: np.ndarray,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> list[DetectionBox]:
    results = model.predict(
        image,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
        verbose=False,
    )
    if not results:
        return []
    result = results[0]
    if result.boxes is None:
        return []

    detections: list[DetectionBox] = []
    names = result.names or {}
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
        confidence = float(box.conf.cpu().numpy()[0])
        class_id = int(box.cls.cpu().numpy()[0])
        class_name = str(names.get(class_id, f"class_{class_id}"))
        detections.append(
            DetectionBox(
                bbox=[int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            )
        )
    return detections


def sort_rois(rois: list[DetectionBox]) -> list[DetectionBox]:
    if not rois:
        return []
    heights = [max(1, roi.bbox[3] - roi.bbox[1]) for roi in rois]
    row_tolerance = max(20.0, float(np.median(heights)) * 0.45)
    entries = []
    for roi in rois:
        x1, y1, x2, y2 = roi.bbox
        entries.append({"roi": roi, "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0})
    entries.sort(key=lambda item: item["cy"])

    rows: list[list[dict[str, Any]]] = []
    for item in entries:
        if not rows:
            rows.append([item])
            continue
        center = float(np.median([row_item["cy"] for row_item in rows[-1]]))
        if abs(item["cy"] - center) <= row_tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])

    sorted_rois: list[DetectionBox] = []
    for row in rows:
        row.sort(key=lambda item: item["cx"])
        sorted_rois.extend([item["roi"] for item in row])
    return sorted_rois


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return [x1, y1, x2, y2]


def resize_for_max_width(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if max_width <= 0 or width <= max_width:
        return image, 1.0
    scale = max_width / max(width, 1)
    target_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA), scale


def scale_detections(detections: list[DetectionBox], inv_scale: float, width: int, height: int) -> list[DetectionBox]:
    if abs(inv_scale - 1.0) < 1e-6:
        return detections
    scaled: list[DetectionBox] = []
    for det in detections:
        bbox = [int(round(value * inv_scale)) for value in det.bbox]
        scaled.append(
            DetectionBox(
                bbox=clamp_bbox(bbox, width, height),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
            )
        )
    return scaled


def map_traditional_points_to_crop(
    points: list[dict[str, float | int]],
    crop_width: int,
    crop_height: int,
    resized_width: int,
    resized_height: int,
) -> list[dict[str, Any]]:
    scale_x = crop_width / max(1, resized_width)
    scale_y = crop_height / max(1, resized_height)
    mapped: list[dict[str, Any]] = []
    for point in points:
        mapped.append(
            {
                "x": float(point["x"]) * scale_x,
                "y": float(point["y"]) * scale_y,
                "response": float(point.get("response", 0.0)),
                "brightness": float(point.get("brightness", 0.0)),
            }
        )
    return mapped


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"无法编码图片: {path}")
    buffer.tofile(str(path))


class DesktopInspectionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x860")
        self.minsize(1000, 720)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.cameras: list[CameraInfo] = []
        self.camera_stream: OpenCVCameraStream | HikvisionCameraStream | None = None
        self.model_cache = ModelCache(self.log)
        self.template_cache: dict[str, Any] = {}
        self.light_controller = LightController()
        self.detection_thread: threading.Thread | None = None
        self.inference_lock = threading.Lock()
        self.live_overlay_enabled = False
        self.live_overlay_thread: threading.Thread | None = None
        self.live_overlay_frame: np.ndarray | None = None
        self.live_overlay_lock = threading.Lock()
        self.display_image: ImageTk.PhotoImage | None = None
        self.last_display_frame: np.ndarray | None = None
        self.result_hold_until = 0.0

        self.camera_var = tk.StringVar()
        self.roi_model_var = tk.StringVar(value=str(ROI_MODEL_DEFAULT))
        self.point_model_var = tk.StringVar(value=choose_default_point_model())
        self.point_method_var = tk.StringVar(value="YOLO")
        self.output_dir_var = tk.StringVar(value=str(OUTPUT_DEFAULT))
        self.preview_fps_var = tk.IntVar(value=15)
        self.live_interval_ms_var = tk.IntVar(value=800)
        self.live_max_width_var = tk.IntVar(value=1280)
        self.multi_frame_count_var = tk.IntVar(value=7)
        self.multi_frame_interval_ms_var = tk.IntVar(value=45)
        self.alarm_enabled_var = tk.BooleanVar(value=True)
        self.rotate_180_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="就绪")

        self._build_ui()
        self.refresh_cameras()
        self.log(detect_hikvision_sdk_hint())
        if not TRADITIONAL_AVAILABLE:
            self.log(f"传统方法不可用: {TRADITIONAL_IMPORT_ERROR}")
        if not CV_TEMPLATE_AVAILABLE:
            self.log(f"模板匹配CV不可用: {CV_TEMPLATE_IMPORT_ERROR}")

        self.after(66, self._refresh_preview)
        self.after(100, self._drain_log_queue)
        self.bind_all("<space>", self.on_template_shortcut)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(12, 10, 12, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        header.columnconfigure(4, weight=1)

        ttk.Label(header, text="相机").grid(row=0, column=0, padx=(0, 6), sticky="w")
        self.camera_combo = ttk.Combobox(header, textvariable=self.camera_var, state="readonly", width=28)
        self.camera_combo.grid(row=0, column=1, padx=(0, 8), sticky="ew")
        ttk.Button(header, text="刷新", command=self.refresh_cameras).grid(row=0, column=2, padx=4)
        ttk.Button(header, text="开始预览", command=self.start_preview).grid(row=0, column=3, padx=4)
        ttk.Button(header, text="停止预览", command=self.stop_preview).grid(row=0, column=4, padx=4, sticky="w")

        ttk.Label(header, text="ROI模型").grid(row=1, column=0, padx=(0, 6), pady=(8, 0), sticky="w")
        ttk.Entry(header, textvariable=self.roi_model_var).grid(row=1, column=1, columnspan=3, padx=(0, 8), pady=(8, 0), sticky="ew")
        ttk.Button(header, text="选择", command=lambda: self.choose_file(self.roi_model_var)).grid(row=1, column=4, pady=(8, 0), sticky="w")

        ttk.Label(header, text="灯点模型").grid(row=2, column=0, padx=(0, 6), pady=(8, 0), sticky="w")
        ttk.Entry(header, textvariable=self.point_model_var).grid(row=2, column=1, columnspan=3, padx=(0, 8), pady=(8, 0), sticky="ew")
        ttk.Button(header, text="选择", command=lambda: self.choose_file(self.point_model_var)).grid(row=2, column=4, pady=(8, 0), sticky="w")

        method_frame = ttk.Frame(header)
        method_frame.grid(row=3, column=0, columnspan=5, pady=(10, 0), sticky="ew")
        ttk.Label(method_frame, text="灯点方法").pack(side="left", padx=(0, 8))
        ttk.Radiobutton(method_frame, text="YOLO", value="YOLO", variable=self.point_method_var).pack(side="left")
        ttk.Radiobutton(method_frame, text="传统方法", value="传统方法", variable=self.point_method_var).pack(side="left", padx=(10, 10))
        ttk.Radiobutton(method_frame, text="模板匹配CV", value=CV_TEMPLATE_METHOD, variable=self.point_method_var).pack(side="left", padx=(0, 16))
        ttk.Button(method_frame, text="检测当前帧", command=self.detect_current_frame).pack(side="left", padx=(0, 12))
        ttk.Button(method_frame, text="正式检测(空格)", command=self.detect_template_multiframe).pack(side="left", padx=(0, 12))
        self.live_overlay_button = ttk.Button(method_frame, text="开始实时框点", command=self.toggle_live_overlay)
        self.live_overlay_button.pack(side="left", padx=(0, 12))
        ttk.Label(method_frame, text="保存目录").pack(side="left", padx=(0, 8))
        ttk.Entry(method_frame, textvariable=self.output_dir_var, width=54).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(method_frame, text="选择", command=self.choose_output_dir).pack(side="left")

        live_frame = ttk.Frame(header)
        live_frame.grid(row=4, column=0, columnspan=5, pady=(8, 0), sticky="ew")
        ttk.Label(live_frame, text="预览FPS").pack(side="left", padx=(0, 6))
        ttk.Spinbox(live_frame, from_=5, to=30, increment=1, textvariable=self.preview_fps_var, width=6).pack(side="left", padx=(0, 14))
        ttk.Label(live_frame, text="实时推理间隔ms").pack(side="left", padx=(0, 6))
        ttk.Spinbox(live_frame, from_=200, to=3000, increment=100, textvariable=self.live_interval_ms_var, width=7).pack(side="left", padx=(0, 14))
        ttk.Label(live_frame, text="实时推理最大宽").pack(side="left", padx=(0, 6))
        ttk.Spinbox(live_frame, from_=640, to=4096, increment=160, textvariable=self.live_max_width_var, width=7).pack(side="left", padx=(0, 14))
        ttk.Label(live_frame, text="多帧数").pack(side="left", padx=(0, 6))
        ttk.Spinbox(live_frame, from_=3, to=15, increment=2, textvariable=self.multi_frame_count_var, width=6).pack(side="left", padx=(0, 14))
        ttk.Label(live_frame, text="多帧间隔ms").pack(side="left", padx=(0, 6))
        ttk.Spinbox(live_frame, from_=20, to=300, increment=5, textvariable=self.multi_frame_interval_ms_var, width=7).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(live_frame, text="NG报警", variable=self.alarm_enabled_var).pack(side="left", padx=(0, 14))
        ttk.Checkbutton(live_frame, text="画面旋转180", variable=self.rotate_180_var).pack(side="left", padx=(0, 14))

        image_frame = ttk.Frame(self, padding=(12, 0, 12, 8))
        image_frame.grid(row=1, column=0, sticky="nsew")
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
        self.image_label = ttk.Label(image_frame, anchor="center", background="#111111")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        bottom = ttk.Frame(self, padding=(12, 0, 12, 12))
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.log_text = tk.Text(bottom, height=9, wrap="word")
        self.log_text.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.log_text.configure(state="disabled")

    def choose_file(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch 模型", "*.pt"), ("所有文件", "*.*")],
        )
        if path:
            variable.set(path)

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="选择保存目录")
        if path:
            self.output_dir_var.set(path)

    @staticmethod
    def bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError, tk.TclError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    def refresh_cameras(self) -> None:
        self.status_var.set("正在枚举摄像头...")
        self.update_idletasks()
        try:
            cameras = []
            try:
                hik_cameras = enumerate_hikvision_cameras()
                cameras.extend(hik_cameras)
                self.log(f"海康相机数量: {len(hik_cameras)}")
            except Exception as exc:
                self.log(f"枚举海康相机失败: {exc}")
            cameras.extend(enumerate_opencv_cameras())
        except Exception as exc:
            cameras = []
            self.log(f"枚举摄像头失败: {exc}")
        self.cameras = cameras
        labels = [camera.label for camera in cameras]
        self.camera_combo["values"] = labels
        if labels:
            self.camera_var.set(labels[0])
            self.status_var.set(f"找到 {len(labels)} 个摄像头")
            self.log(f"找到摄像头: {', '.join(labels)}")
        else:
            self.camera_var.set("")
            self.status_var.set("未找到可用摄像头")
            self.log(f"未找到可打开的海康或 OpenCV 摄像头。现场日志: {LOG_FILE}")

    def selected_camera(self) -> CameraInfo | None:
        label = self.camera_var.get()
        for camera in self.cameras:
            if camera.label == label:
                return camera
        return None

    def start_preview(self) -> None:
        camera = self.selected_camera()
        if camera is None:
            messagebox.showwarning(APP_TITLE, "请先选择可用相机。")
            return
        self.stop_preview()
        if camera.kind == "hikvision":
            self.camera_stream = HikvisionCameraStream(camera, self.log)
        else:
            self.camera_stream = OpenCVCameraStream(camera, self.log)
        try:
            self.camera_stream.start()
            self.status_var.set(f"预览中: {camera.label}")
            if self.rotate_180_var.get():
                self.log("画面方向校正已启用: 预览/检测统一旋转180度")
        except Exception as exc:
            self.camera_stream = None
            self.log(f"开始预览失败: {exc}")
            messagebox.showerror(APP_TITLE, str(exc))

    def stop_preview(self) -> None:
        self.stop_live_overlay()
        if self.camera_stream is not None:
            self.camera_stream.stop()
            self.camera_stream = None
        self.status_var.set("预览已停止")

    def detect_current_frame(self) -> None:
        if self.detection_thread and self.detection_thread.is_alive():
            self.log("检测正在运行，请等待当前检测完成。")
            return
        frame = self.get_latest_oriented_frame()
        if frame is None:
            messagebox.showwarning(APP_TITLE, "没有可检测的实时帧，请先开始预览。")
            return

        roi_model = self.roi_model_var.get().strip()
        point_model = self.point_model_var.get().strip()
        method = self.point_method_var.get()
        output_dir = self.output_dir_var.get().strip()

        self.status_var.set("检测中...")
        self.detection_thread = threading.Thread(
            target=self._run_detection_safe,
            args=(frame, roi_model, point_model, method, output_dir),
            name="inspection",
            daemon=True,
        )
        self.detection_thread.start()

    def on_template_shortcut(self, event: tk.Event) -> None:
        if isinstance(event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox, ttk.Combobox)):
            return
        self.detect_template_multiframe()

    def detect_template_multiframe(self) -> None:
        if self.detection_thread and self.detection_thread.is_alive():
            self.log("检测正在运行，请等待当前检测完成。")
            return
        if self.get_latest_oriented_frame() is None:
            messagebox.showwarning(APP_TITLE, "没有可检测的实时帧，请先开始预览。")
            return
        if not CV_TEMPLATE_AVAILABLE:
            messagebox.showerror(APP_TITLE, f"模板匹配CV不可用: {CV_TEMPLATE_IMPORT_ERROR}")
            return

        output_dir = self.output_dir_var.get().strip()
        frame_count = self.bounded_int(self.multi_frame_count_var.get(), 7, 1, 31)
        if frame_count % 2 == 0:
            frame_count += 1
        interval_ms = self.bounded_int(self.multi_frame_interval_ms_var.get(), 45, 5, 1000)

        self.point_method_var.set(CV_TEMPLATE_METHOD)
        self.status_var.set(f"正式检测中: 采集 {frame_count} 帧...")
        self.detection_thread = threading.Thread(
            target=self._run_template_multiframe_safe,
            args=(output_dir, frame_count, interval_ms),
            name="template-multiframe-inspection",
            daemon=True,
        )
        self.detection_thread.start()

    def toggle_live_overlay(self) -> None:
        if self.live_overlay_enabled:
            self.stop_live_overlay()
        else:
            self.start_live_overlay()

    def start_live_overlay(self) -> None:
        if self.get_latest_oriented_frame() is None:
            messagebox.showwarning(APP_TITLE, "请先开始预览，等画面出来后再开启实时框点。")
            return
        if self.live_overlay_thread and self.live_overlay_thread.is_alive():
            return

        roi_model = self.roi_model_var.get().strip()
        point_model = self.point_model_var.get().strip()
        method = self.point_method_var.get()
        self.live_overlay_enabled = True
        with self.live_overlay_lock:
            self.live_overlay_frame = None
        self.live_overlay_button.configure(text="停止实时框点")
        preview_name = "实时评估" if method == CV_TEMPLATE_METHOD else "实时框点"
        self.status_var.set(f"{preview_name}中: {method}")
        self.log(f"开始{preview_name}: {method}")
        self.live_overlay_thread = threading.Thread(
            target=self._live_overlay_loop,
            args=(roi_model, point_model, method),
            name="live-overlay",
            daemon=True,
        )
        self.live_overlay_thread.start()

    def stop_live_overlay(self) -> None:
        if not self.live_overlay_enabled and self.live_overlay_thread is None:
            return
        self.live_overlay_enabled = False
        if self.live_overlay_thread and self.live_overlay_thread.is_alive():
            self.live_overlay_thread.join(timeout=1.0)
        self.live_overlay_thread = None
        with self.live_overlay_lock:
            self.live_overlay_frame = None
        if hasattr(self, "live_overlay_button"):
            self.live_overlay_button.configure(text="开始实时框点")
        self.log("实时框点已停止")

    def _live_overlay_loop(self, roi_model_path: str, point_model_path: str, method: str) -> None:
        while self.live_overlay_enabled:
            started = time.monotonic()
            try:
                if method == CV_TEMPLATE_METHOD:
                    frame_count = self.bounded_int(self.multi_frame_count_var.get(), 7, 1, 15)
                    if frame_count % 2 == 0:
                        frame_count += 1
                    interval_ms = self.bounded_int(self.multi_frame_interval_ms_var.get(), 45, 5, 1000)
                    frames = self.capture_frame_sequence(frame_count, interval_ms)
                    if not frames:
                        time.sleep(0.1)
                        continue
                    with self.inference_lock:
                        templates = self.get_cv_templates()
                        frame_payloads = [self.analyze_template_frame(frame, templates) for frame in frames]
                        payload = self.fuse_template_frames(
                            frame_payloads,
                            templates,
                            requested_frame_count=frame_count,
                            frame_interval_ms=interval_ms,
                        )
                        base_frame = self.choose_template_base_frame(frames, frame_payloads)
                        overlay = self.draw_template_fused_result(base_frame, payload, templates, live=True)
                else:
                    frame = self.get_latest_oriented_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    max_width = self.bounded_int(self.live_max_width_var.get(), 1280, 320, 8192)
                    with self.inference_lock:
                        overlay, payload = self.build_live_overlay(frame, roi_model_path, point_model_path, method, max_width)
                with self.live_overlay_lock:
                    self.live_overlay_frame = overlay
                elapsed_ms = int((time.monotonic() - started) * 1000)
                prefix = "实时评估" if method == CV_TEMPLATE_METHOD else "实时框点"
                frames_text = ""
                if method == CV_TEMPLATE_METHOD:
                    frames_text = f" 帧={payload['valid_frame_count']}/{payload['frames_captured']}"
                summary = (
                    f"{prefix}: ROI={payload['roi_count']} "
                    f"class={payload['roi_class'] or 'mixed'} "
                    f"{'PASS' if payload['overall_ok'] else 'FAIL'}"
                    f"{frames_text} {elapsed_ms}ms"
                )
                self.after(0, lambda text=summary: self.status_var.set(text))
            except Exception as exc:
                detail = traceback.format_exc()
                self.log(f"实时框点失败: {exc}")
                self.log(detail)
                self.live_overlay_enabled = False
                self.after(0, lambda: self.live_overlay_button.configure(text="开始实时框点"))
                self.after(0, lambda: self.status_var.set("实时框点失败"))
                break
            interval = self.bounded_int(self.live_interval_ms_var.get(), 800, 100, 10000) / 1000.0
            elapsed = time.monotonic() - started
            time.sleep(max(0.02, interval - elapsed))

    def capture_frame_sequence(self, frame_count: int, interval_ms: int) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for index in range(frame_count):
            frame = self.get_latest_oriented_frame()
            if frame is not None:
                frames.append(frame)
            if index < frame_count - 1:
                time.sleep(max(0.001, interval_ms / 1000.0))
        return frames

    def _run_template_multiframe_safe(
        self,
        output_dir: str,
        frame_count: int,
        interval_ms: int,
    ) -> None:
        try:
            frames = self.capture_frame_sequence(frame_count, interval_ms)
            if not frames:
                raise RuntimeError("没有采集到有效帧。")

            self.after(0, lambda: self.status_var.set(f"正式检测中: 已采集 {len(frames)}/{frame_count} 帧，正在融合..."))
            with self.inference_lock:
                result_image, payload, run_dir = self.run_template_multiframe_inspection(
                    frames=frames,
                    output_dir=output_dir,
                    requested_frame_count=frame_count,
                    frame_interval_ms=interval_ms,
            )
            self.last_display_frame = result_image
            self.result_hold_until = time.monotonic() + 5.0
            self.log(f"正式模板检测完成: {'PASS' if payload['overall_ok'] else 'FAIL'}")
            self.log(f"结果已保存: {run_dir}")
            self.log(self.format_summary(payload))
            alarmed = self.trigger_detection_alarm(payload, "正式模板检测")
            status = (
                f"正式检测完成: {'PASS' if payload['overall_ok'] else 'FAIL'} "
                f"有效帧={payload['valid_frame_count']}/{payload['frames_captured']} "
                f"ROI={payload['roi_count']}/20 class={payload['roi_class'] or 'mixed'}"
                f"{' NG报警已触发' if alarmed else ''}"
            )
            self.after(0, lambda: self._show_frame(result_image))
            self.after(0, lambda text=status: self.status_var.set(text))
        except Exception as exc:
            detail = traceback.format_exc()
            self.log(f"正式模板检测失败: {exc}")
            self.log(detail)
            self.after(0, lambda: self.status_var.set("正式模板检测失败"))

    def _run_detection_safe(
        self,
        frame: np.ndarray,
        roi_model_path: str,
        point_model_path: str,
        method: str,
        output_dir: str,
    ) -> None:
        try:
            with self.inference_lock:
                result_image, payload, run_dir = self.run_inspection(
                    frame,
                    roi_model_path,
                    point_model_path,
                    method,
                    output_dir,
            )
            self.last_display_frame = result_image
            self.result_hold_until = time.monotonic() + 5.0
            self.log(f"检测完成: {'PASS' if payload['overall_ok'] else 'FAIL'}")
            self.log(f"结果已保存: {run_dir}")
            self.log(self.format_summary(payload))
            alarmed = self.trigger_detection_alarm(payload, "检测当前帧")
            self.after(0, lambda: self._show_frame(result_image))
            self.after(
                0,
                lambda: self.status_var.set(
                    f"检测完成: {'PASS' if payload['overall_ok'] else 'FAIL'}"
                    f"{' NG报警已触发' if alarmed else ''}"
                ),
            )
        except Exception as exc:
            detail = traceback.format_exc()
            self.log(f"检测失败: {exc}")
            self.log(detail)
            self.after(0, lambda: self.status_var.set("检测失败"))

    def trigger_detection_alarm(self, payload: dict[str, Any], source: str) -> bool:
        if payload.get("overall_ok", False):
            return False
        if hasattr(self, "alarm_enabled_var") and not self.alarm_enabled_var.get():
            return False
        self.log(f"{source} NG报警触发")
        threading.Thread(target=self._play_alarm_sequence, name="inspection-alarm", daemon=True).start()
        return True

    def _play_alarm_sequence(self) -> None:
        if winsound is not None:
            for _ in range(3):
                winsound.Beep(1800, 220)
                winsound.Beep(900, 180)
                time.sleep(0.08)
            return
        for _ in range(3):
            self.after(0, self.bell)
            time.sleep(0.25)

    def build_live_overlay(
        self,
        frame: np.ndarray,
        roi_model_path: str,
        point_model_path: str,
        method: str,
        max_width: int = 1280,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if method == CV_TEMPLATE_METHOD:
            templates = self.get_cv_templates()
            frame_items = self.analyze_template_frame(frame, templates)
            payload = self.fuse_template_frames([frame_items], templates, requested_frame_count=1, frame_interval_ms=0)
            result_image = self.draw_template_fused_result(frame, payload, templates, live=True)
            return result_image, payload

        height, width = frame.shape[:2]
        roi_input, roi_scale = resize_for_max_width(frame, max_width)
        roi_model = self.model_cache.get(roi_model_path)
        roi_detections = run_yolo_detect(
            roi_model,
            roi_input,
            conf=0.2,
            iou=0.45,
            max_det=50,
            device="cpu",
        )
        roi_detections = sort_rois(
            scale_detections(roi_detections, 1.0 / max(roi_scale, 1e-6), width, height)
        )

        result_image = frame.copy()
        class_names = [roi.class_name for roi in roi_detections]
        roi_count_ok = len(roi_detections) == 20
        roi_class_consistent = len(set(class_names)) == 1 if class_names else False
        roi_class = class_names[0] if roi_class_consistent else ""
        expected_points = ROI_EXPECTED_POINTS.get(roi_class)

        if method == "YOLO":
            point_model = self.model_cache.get(point_model_path)
        else:
            point_model = None
            if not TRADITIONAL_AVAILABLE:
                raise RuntimeError(f"传统方法不可用: {TRADITIONAL_IMPORT_ERROR}")

        rois_payload: list[dict[str, Any]] = []
        for index, roi in enumerate(roi_detections):
            x1, y1, x2, y2 = clamp_bbox(roi.bbox, width, height)
            crop = frame[y1:y2, x1:x2]
            current_expected = ROI_EXPECTED_POINTS.get(roi.class_name)
            point_count = 0

            if crop.size > 0:
                if method == "YOLO":
                    point_detections = run_yolo_detect(
                        point_model,
                        crop,
                        conf=0.2,
                        iou=0.45,
                        max_det=200,
                        device="cpu",
                    )
                    point_count = len(point_detections)
                    for det in point_detections:
                        px1, py1, px2, py2 = det.bbox
                        full_bbox = [px1 + x1, py1 + y1, px2 + x1, py2 + y1]
                        cv2.rectangle(
                            result_image,
                            (full_bbox[0], full_bbox[1]),
                            (full_bbox[2], full_bbox[3]),
                            (80, 255, 80),
                            1,
                        )
                else:
                    raw = detect_traditional_points(crop, TraditionalParams())
                    resized = raw["resized"]
                    filtered = filter_points(
                        raw["points"],
                        resized.shape[1],
                        resized.shape[0],
                        FilterParams(),
                    )
                    crop_points = map_traditional_points_to_crop(
                        filtered["neighbor_points"],
                        crop.shape[1],
                        crop.shape[0],
                        resized.shape[1],
                        resized.shape[0],
                    )
                    point_count = len(crop_points)
                    for point in crop_points:
                        full_x = int(round(point["x"] + x1))
                        full_y = int(round(point["y"] + y1))
                        cv2.circle(result_image, (full_x, full_y), 4, (80, 255, 80), 1, cv2.LINE_AA)
                        cv2.circle(result_image, (full_x, full_y), 1, (0, 255, 255), -1, cv2.LINE_AA)

            roi_ok = current_expected is not None and point_count == current_expected
            color = (0, 200, 0) if roi_ok else (0, 0, 255)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            label = f"{index + 1}:{roi.class_name} {point_count}/{current_expected or '?'}"
            self.draw_label(result_image, label, x1, max(16, y1 - 6), color)
            rois_payload.append(
                {
                    "index": index,
                    "class_name": roi.class_name,
                    "point_count": point_count,
                    "expected_count": current_expected,
                    "ok": roi_ok,
                }
            )

        overall_ok = bool(
            roi_count_ok
            and roi_class_consistent
            and expected_points is not None
            and all(item["ok"] for item in rois_payload)
        )
        header_color = (0, 200, 0) if overall_ok else (0, 0, 255)
        self.draw_label(
            result_image,
            f"LIVE {'PASS' if overall_ok else 'FAIL'} ROI={len(roi_detections)}/20 class={roi_class or 'mixed'} method={method}",
            12,
            28,
            header_color,
            scale=0.8,
            thickness=2,
        )

        payload = {
            "roi_count": len(roi_detections),
            "roi_count_ok": roi_count_ok,
            "roi_class_consistent": roi_class_consistent,
            "roi_class": roi_class,
            "expected_points_per_roi": expected_points,
            "overall_ok": overall_ok,
            "rois": rois_payload,
        }
        return result_image, payload

    def run_inspection(
        self,
        frame: np.ndarray,
        roi_model_path: str,
        point_model_path: str,
        method: str,
        output_dir: str,
    ) -> tuple[np.ndarray, dict[str, Any], Path]:
        if method == CV_TEMPLATE_METHOD:
            return self.run_template_multiframe_inspection(
                frames=[frame],
                output_dir=output_dir,
                requested_frame_count=1,
                frame_interval_ms=0,
                single_frame=True,
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / timestamp

        self.light_controller.set_scene("default")
        self.light_controller.wait_stable(50)

        roi_model = self.model_cache.get(roi_model_path)
        roi_detections = run_yolo_detect(
            roi_model,
            frame,
            conf=0.2,
            iou=0.45,
            max_det=50,
            device="cpu",
        )
        roi_detections = sort_rois(roi_detections)

        result_image = frame.copy()
        height, width = frame.shape[:2]
        class_names = [roi.class_name for roi in roi_detections]
        roi_count_ok = len(roi_detections) == 20
        roi_class_consistent = len(set(class_names)) == 1 if class_names else False
        roi_class = class_names[0] if roi_class_consistent else ""
        expected_points = ROI_EXPECTED_POINTS.get(roi_class)

        rois_payload: list[dict[str, Any]] = []
        if method == "YOLO":
            point_model = self.model_cache.get(point_model_path)
        else:
            point_model = None
            if not TRADITIONAL_AVAILABLE:
                raise RuntimeError(f"传统方法不可用: {TRADITIONAL_IMPORT_ERROR}")

        for index, roi in enumerate(roi_detections):
            x1, y1, x2, y2 = clamp_bbox(roi.bbox, width, height)
            crop = frame[y1:y2, x1:x2]
            current_expected = ROI_EXPECTED_POINTS.get(roi.class_name)

            color = (0, 180, 255)
            point_count = 0
            points: list[dict[str, Any]] = []
            detections_payload: list[dict[str, Any]] = []

            if crop.size > 0:
                if method == "YOLO":
                    point_detections = run_yolo_detect(
                        point_model,
                        crop,
                        conf=0.2,
                        iou=0.45,
                        max_det=200,
                        device="cpu",
                    )
                    point_count = len(point_detections)
                    for det in point_detections:
                        px1, py1, px2, py2 = det.bbox
                        full_bbox = [px1 + x1, py1 + y1, px2 + x1, py2 + y1]
                        cv2.rectangle(result_image, (full_bbox[0], full_bbox[1]), (full_bbox[2], full_bbox[3]), (80, 255, 80), 1)
                        detections_payload.append(
                            {
                                "bbox": full_bbox,
                                "confidence": det.confidence,
                                "class_id": det.class_id,
                                "class_name": det.class_name,
                            }
                        )
                else:
                    raw = detect_traditional_points(crop, TraditionalParams())
                    resized = raw["resized"]
                    filtered = filter_points(
                        raw["points"],
                        resized.shape[1],
                        resized.shape[0],
                        FilterParams(),
                    )
                    crop_points = map_traditional_points_to_crop(
                        filtered["neighbor_points"],
                        crop.shape[1],
                        crop.shape[0],
                        resized.shape[1],
                        resized.shape[0],
                    )
                    point_count = len(crop_points)
                    for point in crop_points:
                        full_x = int(round(point["x"] + x1))
                        full_y = int(round(point["y"] + y1))
                        cv2.circle(result_image, (full_x, full_y), 4, (80, 255, 80), 1, cv2.LINE_AA)
                        cv2.circle(result_image, (full_x, full_y), 1, (0, 255, 255), -1, cv2.LINE_AA)
                        points.append(
                            {
                                "x": full_x,
                                "y": full_y,
                                "crop_x": point["x"],
                                "crop_y": point["y"],
                                "response": point["response"],
                                "brightness": point["brightness"],
                            }
                        )

            roi_ok = current_expected is not None and point_count == current_expected
            color = (0, 200, 0) if roi_ok else (0, 0, 255)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            label = f"{index + 1}:{roi.class_name} {point_count}/{current_expected or '?'} {'OK' if roi_ok else 'NG'}"
            self.draw_label(result_image, label, x1, max(16, y1 - 6), color)

            rois_payload.append(
                {
                    "index": index,
                    "class_name": roi.class_name,
                    "class_id": roi.class_id,
                    "confidence": roi.confidence,
                    "bbox": [x1, y1, x2, y2],
                    "point_count": point_count,
                    "expected_count": current_expected,
                    "ok": roi_ok,
                    "points": points,
                    "detections": detections_payload,
                }
            )

        overall_ok = bool(
            roi_count_ok
            and roi_class_consistent
            and expected_points is not None
            and all(item["ok"] for item in rois_payload)
        )
        header_color = (0, 200, 0) if overall_ok else (0, 0, 255)
        self.draw_label(
            result_image,
            f"{'PASS' if overall_ok else 'FAIL'} ROI={len(roi_detections)}/20 class={roi_class or 'mixed'} method={method}",
            12,
            28,
            header_color,
            scale=0.8,
            thickness=2,
        )

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "camera": self.camera_var.get(),
            "roi_model": roi_model_path,
            "point_model": point_model_path if method == "YOLO" else "",
            "point_method": "yolo" if method == "YOLO" else "traditional",
            "roi_count": len(roi_detections),
            "roi_count_ok": roi_count_ok,
            "roi_class_consistent": roi_class_consistent,
            "roi_class": roi_class,
            "expected_points_per_roi": expected_points,
            "overall_ok": overall_ok,
            "light_controller": self.light_controller.get_status(),
            "rois": rois_payload,
        }

        write_image(run_dir / "original.jpg", frame)
        write_image(run_dir / "result.jpg", result_image)
        save_json(run_dir / "result.json", payload)
        return result_image, payload, run_dir

    def get_cv_templates(self) -> dict[str, dict[int, Any]]:
        if not CV_TEMPLATE_AVAILABLE:
            raise RuntimeError(f"模板匹配CV不可用: {CV_TEMPLATE_IMPORT_ERROR}")
        template_path = str(CV_TEMPLATE_MODEL_DEFAULT)
        if template_path not in self.template_cache:
            path = Path(template_path)
            if not path.exists():
                raise FileNotFoundError(f"模板文件不存在: {path}")
            self.log(f"加载模板: {path}")
            self.template_cache[template_path] = cv_template.load_templates(path)
        return self.template_cache[template_path]

    def analyze_template_frame(
        self,
        frame: np.ndarray,
        templates: dict[str, dict[int, Any]],
    ) -> dict[str, Any]:
        boxes, _ = cvp.detect_roi_boxes(frame)
        items: list[dict[str, Any]] = []
        height, width = frame.shape[:2]
        for roi_index, box in enumerate(boxes[:20]):
            x1, y1, x2, y2 = box.xyxy
            x1, y1, x2, y2 = clamp_bbox([x1, y1, x2, y2], width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame[y1:y2, x1:x2]
            result = cv_template.match_roi(
                roi=roi,
                roi_index=roi_index,
                templates=templates,
                slot_radius=9,
                slot_threshold=0.42,
                extra_penalty=0.65,
            )
            items.append(
                {
                    "roi_index": roi_index,
                    "box": cvp.RoiBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1),
                    "result": result,
                }
            )

        cv_template.apply_frame_majority(items)
        class_counts: dict[str, int] = {}
        abnormal_count = 0
        for item in items:
            result = item["result"]
            class_name = result["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if result["count_status"] != "ok":
                abnormal_count += 1
        return {
            "items": items,
            "roi_count": len(items),
            "class_counts": class_counts,
            "abnormal_count": abnormal_count,
            "valid": len(items) == 20,
        }

    def run_template_multiframe_inspection(
        self,
        frames: list[np.ndarray],
        output_dir: str,
        requested_frame_count: int,
        frame_interval_ms: int,
        single_frame: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any], Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "template_single" if single_frame else "template_multiframe"
        run_dir = Path(output_dir) / f"{timestamp}_{suffix}"

        self.light_controller.set_scene("default")
        self.light_controller.wait_stable(50)

        templates = self.get_cv_templates()
        frame_payloads = [self.analyze_template_frame(frame, templates) for frame in frames]
        payload = self.fuse_template_frames(
            frame_payloads,
            templates,
            requested_frame_count=requested_frame_count,
            frame_interval_ms=frame_interval_ms,
        )
        payload.update(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "camera": self.camera_var.get(),
                "roi_model": "",
                "point_model": "",
                "point_method": "cv_template_multiframe" if not single_frame else "cv_template",
                "light_controller": self.light_controller.get_status(),
                "template_path": str(CV_TEMPLATE_MODEL_DEFAULT),
            }
        )

        base_frame = self.choose_template_base_frame(frames, frame_payloads)
        result_image = self.draw_template_fused_result(base_frame, payload, templates, live=False)

        frames_dir = run_dir / "frames"
        for index, frame in enumerate(frames, start=1):
            write_image(frames_dir / f"frame_{index:02d}.jpg", frame)
        write_image(run_dir / "original.jpg", base_frame)
        write_image(run_dir / "result.jpg", result_image)
        save_json(run_dir / "result.json", payload)
        return result_image, payload, run_dir

    @staticmethod
    def choose_template_base_frame(
        frames: list[np.ndarray],
        frame_payloads: list[dict[str, Any]],
    ) -> np.ndarray:
        if not frames:
            raise RuntimeError("没有可显示的帧。")
        best_index = 0
        best_score = -1
        for index, payload in enumerate(frame_payloads):
            score = int(payload["valid"]) * 100 + int(payload["roi_count"]) - int(payload["abnormal_count"]) * 2
            if score > best_score:
                best_index = index
                best_score = score
        return frames[min(best_index, len(frames) - 1)].copy()

    def fuse_template_frames(
        self,
        frame_payloads: list[dict[str, Any]],
        templates: dict[str, dict[int, Any]],
        requested_frame_count: int,
        frame_interval_ms: int,
    ) -> dict[str, Any]:
        frames_captured = len(frame_payloads)
        valid_frame_count = sum(1 for frame_payload in frame_payloads if frame_payload["valid"])
        min_observations = max(1, min(3, (frames_captured + 1) // 2))
        items_by_roi: dict[int, list[dict[str, Any]]] = {index: [] for index in range(20)}
        frame_summaries: list[dict[str, Any]] = []

        for frame_index, frame_payload in enumerate(frame_payloads):
            frame_summaries.append(
                {
                    "frame_index": frame_index,
                    "roi_count": int(frame_payload["roi_count"]),
                    "valid": bool(frame_payload["valid"]),
                    "class_counts": frame_payload["class_counts"],
                    "abnormal_count": int(frame_payload["abnormal_count"]),
                }
            )
            for item in frame_payload["items"]:
                roi_index = int(item["roi_index"])
                if 0 <= roi_index < 20:
                    items_by_roi[roi_index].append(item)

        rois_payload: list[dict[str, Any]] = []
        for roi_index in range(20):
            observations = items_by_roi[roi_index]
            if not observations:
                rois_payload.append(
                    {
                        "index": roi_index,
                        "class_name": "unknown",
                        "class_id": -1,
                        "confidence": 0.0,
                        "bbox": [],
                        "point_count": 0,
                        "expected_count": None,
                        "ok": False,
                        "observations": 0,
                        "missing_slots": [],
                        "extra_slots": [],
                        "hits": [],
                        "slot_offset": [0, 0],
                        "class_scores": {},
                    }
                )
                continue

            class_votes: dict[str, float] = {}
            for item in observations:
                result = item["result"]
                for class_name, detail in result["class_details"].items():
                    score = max(0.01, float(detail["class_score"]))
                    if detail["count_status"] == "ok":
                        score += 0.15
                    class_votes[class_name] = class_votes.get(class_name, 0.0) + score
            class_name = max(class_votes.items(), key=lambda item: item[1])[0]
            template = templates[class_name][roi_index]
            expected = ROI_EXPECTED_POINTS.get(class_name)
            slot_weights = [0.0 for _ in template.points]
            total_weight = 0.0
            extra_votes: dict[int, float] = {}
            score_sum = 0.0

            for item in observations:
                result = item["result"]
                detail = result["class_details"].get(class_name, result)
                weight = max(0.05, float(detail["class_score"]))
                if detail["count_status"] == "ok":
                    weight += 0.10
                total_weight += weight
                score_sum += float(detail["class_score"])
                for slot_index, hit in enumerate(detail["hits"][: len(slot_weights)]):
                    if hit:
                        slot_weights[slot_index] += weight
                for extra_index in detail.get("extra_indices", []):
                    extra_votes[int(extra_index)] = extra_votes.get(int(extra_index), 0.0) + weight

            threshold = total_weight * 0.5
            hits = [weight >= threshold for weight in slot_weights]
            point_count = int(sum(hits))
            missing_slots = [index + 1 for index, hit in enumerate(hits) if not hit]
            extra_slots = [index + 1 for index, weight in extra_votes.items() if weight >= threshold]
            bboxes = [item["box"].xyxy for item in observations]
            bbox = [
                int(round(float(np.median([box[pos] for box in bboxes]))))
                for pos in range(4)
            ]
            enough_observations = len(observations) >= min_observations
            ok = bool(enough_observations and expected is not None and point_count == expected)
            rois_payload.append(
                {
                    "index": roi_index,
                    "class_name": class_name,
                    "class_id": int(cvp.CLASS_IDS.get(class_name, -1)),
                    "confidence": round(score_sum / max(1, len(observations)), 4),
                    "bbox": bbox,
                    "point_count": point_count,
                    "expected_count": expected,
                    "ok": ok,
                    "observations": len(observations),
                    "missing_slots": missing_slots,
                    "extra_slots": extra_slots,
                    "hits": hits,
                    "slot_offset": [0, 0],
                    "class_scores": {name: round(value, 4) for name, value in class_votes.items()},
                }
            )

        roi_count = sum(1 for roi in rois_payload if roi["observations"] > 0)
        class_names = [roi["class_name"] for roi in rois_payload if roi["class_name"] != "unknown"]
        roi_class_consistent = len(set(class_names)) == 1 and roi_count == 20
        roi_class = class_names[0] if roi_class_consistent else ""
        expected_points = ROI_EXPECTED_POINTS.get(roi_class)
        roi_count_ok = roi_count == 20
        enough_frames = frames_captured > 0 and valid_frame_count >= 1
        overall_ok = bool(
            enough_frames
            and roi_count_ok
            and roi_class_consistent
            and expected_points is not None
            and all(roi["ok"] for roi in rois_payload)
        )
        return {
            "frames_requested": requested_frame_count,
            "frames_captured": frames_captured,
            "frame_interval_ms": frame_interval_ms,
            "valid_frame_count": valid_frame_count,
            "min_observations_per_roi": min_observations,
            "frame_summaries": frame_summaries,
            "roi_count": roi_count,
            "roi_count_ok": roi_count_ok,
            "roi_class_consistent": roi_class_consistent,
            "roi_class": roi_class,
            "expected_points_per_roi": expected_points,
            "overall_ok": overall_ok,
            "rois": rois_payload,
        }

    def draw_template_fused_result(
        self,
        frame: np.ndarray,
        payload: dict[str, Any],
        templates: dict[str, dict[int, Any]],
        live: bool,
    ) -> np.ndarray:
        result_image = frame.copy()
        image_width = max(1, result_image.shape[1])
        box_thickness = max(3, int(round(image_width / 800)))
        label_scale = max(0.9, image_width / 2200)
        detail_scale = max(0.65, image_width / 3200)
        header_scale = max(1.15, image_width / 1700)
        label_thickness = max(2, int(round(image_width / 1200)))
        point_radius = max(4, int(round(image_width / 520)))
        missing_radius = max(point_radius + 2, int(round(image_width / 420)))

        for roi in payload["rois"]:
            bbox = roi.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = clamp_bbox(bbox, result_image.shape[1], result_image.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = roi["class_name"]
            template = templates.get(class_name, {}).get(int(roi["index"]))
            if template is None:
                continue
            color = (0, 210, 0) if roi["ok"] else (0, 0, 255)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, box_thickness)

            if not live:
                roi_w = max(1, x2 - x1)
                roi_h = max(1, y2 - y1)
                template_w, template_h = template.size
                offset = roi.get("slot_offset", [0, 0])
                offset_x = float(offset[0]) if len(offset) >= 1 else 0.0
                offset_y = float(offset[1]) if len(offset) >= 2 else 0.0
                for slot_index, (point, hit) in enumerate(zip(template.points, roi["hits"])):
                    px, py = point
                    cx = int(round(x1 + (px + offset_x) * roi_w / template_w))
                    cy = int(round(y1 + (py + offset_y) * roi_h / template_h))
                    if hit:
                        cv2.circle(result_image, (cx, cy), point_radius, (0, 255, 255), -1, cv2.LINE_AA)
                    else:
                        cv2.circle(result_image, (cx, cy), missing_radius, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(
                            result_image,
                            str(slot_index + 1),
                            (cx + missing_radius, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            detail_scale,
                            (0, 0, 255),
                            label_thickness,
                            cv2.LINE_AA,
                        )

            status = "OK" if roi["ok"] else "NG"
            label = (
                f"#{int(roi['index']) + 1:02d} {class_name} "
                f"{roi['point_count']}/{roi['expected_count'] or '?'} {status}"
            )
            self.draw_label(
                result_image,
                label,
                x1 + 4,
                y1 + int(34 * label_scale),
                color,
                scale=label_scale,
                thickness=label_thickness,
            )
            if not roi["ok"]:
                self.draw_label(
                    result_image,
                    f"NG obs={roi['observations']} miss={len(roi['missing_slots'])} extra={len(roi['extra_slots'])}",
                    x1,
                    min(result_image.shape[0] - 8, y2 + 18),
                    (0, 0, 255),
                    scale=detail_scale,
                    thickness=label_thickness,
                )

        header_color = (0, 200, 0) if payload["overall_ok"] else (0, 0, 255)
        prefix = "LIVE PREVIEW CV" if live else "FINAL CV"
        self.draw_label(
            result_image,
            (
                f"{prefix} {'PASS' if payload['overall_ok'] else 'FAIL'} "
                f"frames={payload['valid_frame_count']}/{payload['frames_captured']} "
                f"ROI={payload['roi_count']}/20 class={payload['roi_class'] or 'mixed'}"
            ),
            12,
            int(48 * header_scale),
            header_color,
            scale=header_scale,
            thickness=max(2, label_thickness + 1),
        )
        return result_image

    @staticmethod
    def draw_label(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        scale: float = 0.5,
        thickness: int = 1,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(0, min(image.shape[1] - tw - 4, x))
        y = max(th + 4, min(image.shape[0] - baseline - 4, y))
        cv2.rectangle(image, (x, y - th - 5), (x + tw + 4, y + baseline + 4), (20, 20, 20), -1)
        cv2.putText(image, text, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)

    def format_summary(self, payload: dict[str, Any]) -> str:
        lines = [
            f"摘要: ROI={payload['roi_count']} count_ok={payload['roi_count_ok']} class_ok={payload['roi_class_consistent']} overall={payload['overall_ok']}",
            f"ROI类型: {payload['roi_class'] or '不一致/无'} 期望点数: {payload['expected_points_per_roi']}",
        ]
        for roi in payload["rois"]:
            lines.append(
                f"  #{roi['index'] + 1:02d} {roi['class_name']} points={roi['point_count']}/{roi['expected_count']} {'PASS' if roi['ok'] else 'FAIL'}"
            )
        return "\n".join(lines)

    def orient_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.rotate_180_var.get():
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame.copy()

    def get_latest_oriented_frame(self) -> np.ndarray | None:
        if self.camera_stream is None:
            return None
        frame = self.camera_stream.get_latest_frame()
        if frame is None:
            return None
        return self.orient_frame(frame)

    def _refresh_preview(self) -> None:
        if self.camera_stream is not None:
            if self.result_hold_until > time.monotonic() and self.last_display_frame is not None:
                self._show_frame(self.last_display_frame)
            else:
                frame = self.get_latest_oriented_frame()
                if frame is not None:
                    self.last_display_frame = frame
                    display_frame = frame
                    if self.live_overlay_enabled:
                        with self.live_overlay_lock:
                            if self.live_overlay_frame is not None:
                                display_frame = self.live_overlay_frame.copy()
                    self._show_frame(display_frame)
        preview_fps = self.bounded_int(self.preview_fps_var.get(), 15, 1, 60)
        self.after(max(15, int(1000 / preview_fps)), self._refresh_preview)

    def _show_frame(self, frame: np.ndarray) -> None:
        label_width = max(320, self.image_label.winfo_width())
        label_height = max(240, self.image_label.winfo_height())
        height, width = frame.shape[:2]
        scale = min(label_width / max(width, 1), label_height / max(height, 1))
        target_w = max(1, int(width * scale))
        target_h = max(1, int(height * scale))
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        self.display_image = ImageTk.PhotoImage(image=pil_image)
        self.image_label.configure(image=self.display_image)

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        append_log_file(message)
        self.log_queue.put(f"[{stamp}] {message}")

    def _drain_log_queue(self) -> None:
        messages: list[str] = []
        while True:
            try:
                messages.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        if messages:
            self.log_text.configure(state="normal")
            for message in messages:
                self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.after(100, self._drain_log_queue)

    def on_close(self) -> None:
        self.stop_preview()
        self.destroy()


def main() -> None:
    app = DesktopInspectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
