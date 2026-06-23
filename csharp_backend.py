from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import cv2
import numpy as np

from desktop_app import (
    CV_TEMPLATE_MODEL_DEFAULT,
    MARQUEE_TEMPLATE_MODEL_DEFAULT,
    OUTPUT_DEFAULT,
    RESOURCE_ROOT,
    ROI_EXPECTED_POINTS,
    CameraInfo,
    HikvisionCameraStream,
    OpenCVCameraStream,
    clamp_bbox,
    cv_template,
    cvp,
    enumerate_hikvision_cameras,
    enumerate_opencv_cameras,
    save_json,
    write_image,
)

import marquee_inspection as marquee
import detect_dot_panel as dot_panel


# FS-GRMA-JC 白/蓝点阵面板模板（detect_dot_panel.py 的模型）。
DOT_PANEL_MODEL_DEFAULT = (
    RESOURCE_ROOT / "models" / "dot_panel" / "Video_20260529135514029" / "templates.json"
)
BRIGHTNESS_PANEL_MODEL_DEFAULT = (
    RESOURCE_ROOT / "models" / "brightness_panel" / "Video_20260610100027363" / "templates.json"
)

# 型号标识：ME70 走 cv_template 逻辑，FS-GRMA-JC 走 detect_dot_panel 逻辑。
MODEL_ME70 = "ME70"
MODEL_DOT_PANEL = "FS-GRMA-JC"
MODEL_BRIGHTNESS_PANEL = "BRIGHTNESS_PANEL"
DOT_PANEL_ALIASES = {"FS-GRMA-JC", "FS_GRMA_JC", "FSGRMAJC", "DOT_PANEL", "DOTPANEL"}
BRIGHTNESS_PANEL_ALIASES = {
    "BRIGHTNESS",
    "BRIGHTNESS_PANEL",
    "BRIGHTNESS-PANEL",
    "WHITE_PANEL",
    "WHITE-BRIGHTNESS",
    "WHITE_TEXT_PANEL",
    "LIGHT_CHECK",
    "亮度检测",
    "白字面板亮度",
}


def normalize_model(value: str) -> str:
    name = (value or "").strip().upper()
    if name in DOT_PANEL_ALIASES:
        return MODEL_DOT_PANEL
    if name in BRIGHTNESS_PANEL_ALIASES:
        return MODEL_BRIGHTNESS_PANEL
    return MODEL_ME70


def json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes, str]:
    return status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"


class VideoFileStream:
    def __init__(self, video_path: str, log: Callable[[str], None]) -> None:
        self.video_path = Path(video_path)
        self.log = log
        self.cap: cv2.VideoCapture | None = None
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_time = 0.0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.camera = CameraInfo(
            key="video_stream",
            label=f"视频文件: {self.video_path.name}",
            kind="video"
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="video-stream-preview", daemon=True)
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
        self.log(f"打开视频流: {self.video_path}")
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.log(f"无法打开视频文件: {self.video_path}")
            return
        
        self.cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / fps
        
        while not self._stop_event.is_set():
            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            with self._lock:
                self.latest_frame = frame.copy()
                self.latest_frame_time = time.time()
                
            elapsed = time.perf_counter() - started
            sleep_time = max(0.001, frame_delay - elapsed)
            time.sleep(sleep_time)
            
        cap.release()
        self.cap = None
        self.log("视频流预览已停止")


class InspectionBackend:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.cameras: list[CameraInfo] = []
        self.camera_stream: OpenCVCameraStream | HikvisionCameraStream | VideoFileStream | None = None
        self.template_cache: dict[str, Any] = {}
        self.dot_panel_model: dict[str, Any] | None = None
        self.brightness_panel_model: dict[str, Any] | None = None
        self.brightness_panel_reference: dict[str, np.ndarray] | None = None
        self.dot_panel_pose_cache: dot_panel.PanelPose | None = None
        self.dot_panel_pose_signature: np.ndarray | None = None
        self.dot_panel_pose_shape: tuple[int, int] | None = None
        self.logs: list[str] = []
        self.output_dir = OUTPUT_DEFAULT
        self.rotate_180 = False
        self.last_result_image: np.ndarray | None = None
        self.last_payload: dict[str, Any] | None = None
        self.last_run_dir: Path | None = None
        self.refresh_cameras()

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-300:]
        try:
            print(line, flush=True)
        except OSError:
            pass

    def refresh_cameras(self) -> list[dict[str, Any]]:
        cameras: list[CameraInfo] = []
        try:
            cameras.extend(enumerate_hikvision_cameras())
        except Exception as exc:
            self.log(f"枚举海康相机失败: {exc}")
        try:
            cameras.extend(enumerate_opencv_cameras())
        except Exception as exc:
            self.log(f"枚举 OpenCV 相机失败: {exc}")
        with self.lock:
            self.cameras = cameras
        self.log(f"摄像头数量: {len(cameras)}")
        return [self.camera_to_json(camera) for camera in cameras]

    @staticmethod
    def camera_to_json(camera: CameraInfo) -> dict[str, Any]:
        return {
            "key": camera.key,
            "label": camera.label,
            "kind": camera.kind,
            "index": camera.index,
            "sdk_index": camera.sdk_index,
            "tlayer_type": camera.tlayer_type,
        }

    def current_camera_label(self) -> str:
        stream = self.camera_stream
        return stream.camera.label if stream is not None else ""

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ok": True,
                "previewing": self.camera_stream is not None,
                "camera": self.current_camera_label(),
                "camera_count": len(self.cameras),
                "rotate_180": self.rotate_180,
                "last_payload": self.last_payload,
                "last_run_dir": str(self.last_run_dir) if self.last_run_dir else "",
                "logs": self.logs[-80:],
            }

    def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if "rotate_180" in payload:
                rotate_180 = bool(payload["rotate_180"])
                if rotate_180 != self.rotate_180:
                    self.rotate_180 = rotate_180
                    self.clear_dot_panel_pose_cache_locked()
            if "output_dir" in payload and str(payload["output_dir"]).strip():
                self.output_dir = Path(str(payload["output_dir"]).strip())
        return self.status()

    def start_preview(self, camera_label: str = "") -> dict[str, Any]:
        self.stop_preview()
        with self.lock:
            self.clear_dot_panel_pose_cache_locked()
        
        is_video = False
        if camera_label:
            path_check = Path(camera_label)
            if path_check.is_file() and path_check.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}:
                is_video = True
                
        if is_video:
            stream = VideoFileStream(camera_label, self.log)
        else:
            with self.lock:
                if not self.cameras:
                    self.refresh_cameras()
                camera = self.select_camera(camera_label)
            if camera is None:
                raise RuntimeError("没有可用相机")
            stream: OpenCVCameraStream | HikvisionCameraStream | VideoFileStream
            if camera.kind == "hikvision":
                stream = HikvisionCameraStream(camera, self.log)
            else:
                stream = OpenCVCameraStream(camera, self.log)
                
        stream.start()
        with self.lock:
            self.camera_stream = stream
        if is_video:
            self.log(f"开始视频流预览: {Path(camera_label).name}")
        else:
            self.log(f"开始预览: {camera.label}")
        return self.status()

    def select_camera(self, label: str) -> CameraInfo | None:
        if label:
            for camera in self.cameras:
                if camera.label == label:
                    return camera
        hikvision = [camera for camera in self.cameras if camera.kind == "hikvision"]
        return hikvision[0] if hikvision else (self.cameras[0] if self.cameras else None)

    def stop_preview(self) -> dict[str, Any]:
        stream = self.camera_stream
        if stream is not None:
            try:
                stream.stop()
            finally:
                with self.lock:
                    self.camera_stream = None
                    self.clear_dot_panel_pose_cache_locked()
            self.log("停止预览")
        return self.status()

    def clear_dot_panel_pose_cache_locked(self) -> None:
        self.dot_panel_pose_cache = None
        self.dot_panel_pose_signature = None
        self.dot_panel_pose_shape = None

    def orient_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.rotate_180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame.copy()

    def latest_frame(self) -> np.ndarray | None:
        stream = self.camera_stream
        if stream is None:
            return None
        frame = stream.get_latest_frame()
        if frame is None:
            return None
        return self.orient_frame(frame)

    def encode_jpeg(self, image: np.ndarray, quality: int = 82) -> bytes:
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("无法编码 JPEG")
        return buffer.tobytes()

    def get_frame_jpeg(self) -> bytes:
        frame = self.latest_frame()
        if frame is None:
            raise RuntimeError("没有实时帧")
        return self.encode_jpeg(frame)

    def get_result_jpeg(self) -> bytes:
        image = self.last_result_image
        if image is None:
            return self.get_frame_jpeg()
        return self.encode_jpeg(image, quality=90)

    def get_templates(self) -> dict[str, dict[int, Any]]:
        template_path = str(CV_TEMPLATE_MODEL_DEFAULT)
        if template_path not in self.template_cache:
            path = Path(template_path)
            if not path.exists():
                raise FileNotFoundError(f"模板文件不存在: {path}")
            self.log(f"加载模板: {path}")
            self.template_cache[template_path] = cv_template.load_templates(path)
        return self.template_cache[template_path]

    def get_dot_panel_model(self) -> dict[str, Any]:
        if self.dot_panel_model is None:
            path = DOT_PANEL_MODEL_DEFAULT
            if not path.exists():
                raise FileNotFoundError(f"FS-GRMA-JC 模板文件不存在: {path}")
            self.log(f"加载 FS-GRMA-JC 模板: {path}")
            self.dot_panel_model = json.loads(path.read_text(encoding="utf-8"))
        return self.dot_panel_model

    def get_brightness_panel_model(self) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        if self.brightness_panel_model is None or self.brightness_panel_reference is None:
            path = BRIGHTNESS_PANEL_MODEL_DEFAULT
            if not path.exists():
                raise FileNotFoundError(f"白字面板亮度模板不存在: {path}")
            model = json.loads(path.read_text(encoding="utf-8"))
            asset_dir = path.parent
            ref_image_path = asset_dir / str(model.get("assets", {}).get("reference_image", "reference.jpg"))
            ref_mask_path = asset_dir / str(model.get("assets", {}).get("reference_mask", "reference_mask.png"))
            ref_image = cv2.imread(str(ref_image_path), cv2.IMREAD_COLOR)
            ref_mask = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
            if ref_image is None:
                raise FileNotFoundError(f"白字面板参考图不存在: {ref_image_path}")
            if ref_mask is None:
                raise FileNotFoundError(f"白字面板参考mask不存在: {ref_mask_path}")
            if ref_mask.ndim == 3:
                ref_mask = ref_mask[:, :, 0]
            self.log(f"加载白字面板亮度模板: {path}")
            self.brightness_panel_model = model
            self.brightness_panel_reference = {
                "image": ref_image,
                "mask": (ref_mask > 0).astype(np.uint8) * 255,
            }
        return self.brightness_panel_model, self.brightness_panel_reference

    def capture_frame_sequence(self, frame_count: int, interval_ms: int) -> list[np.ndarray]:
        frames: list[np.ndarray] = []
        for index in range(frame_count):
            frame = self.latest_frame()
            if frame is not None:
                frames.append(frame)
            if index < frame_count - 1:
                time.sleep(max(0.001, interval_ms / 1000.0))
        return frames

    def capture_frame_sequence_for_duration(self, duration_seconds: float, interval_ms: int) -> tuple[list[np.ndarray], float]:
        interval_ms = max(20, min(1000, int(interval_ms)))
        duration_seconds = max(1.0, min(90.0, float(duration_seconds)))
        frames: list[np.ndarray] = []
        started = time.perf_counter()
        deadline = started + duration_seconds
        while time.perf_counter() < deadline:
            frame = self.latest_frame()
            if frame is not None:
                frames.append(frame)
            time.sleep(interval_ms / 1000.0)
        elapsed = max(0.001, time.perf_counter() - started)
        fps = len(frames) / elapsed if frames else 1000.0 / interval_ms
        return frames, fps

    def analyze_frame(self, frame: np.ndarray, templates: dict[str, dict[int, Any]]) -> dict[str, Any]:
        boxes, _ = cvp.detect_roi_boxes(frame)
        height, width = frame.shape[:2]
        items: list[dict[str, Any]] = []
        for roi_index, box in enumerate(boxes[:20]):
            x1, y1, x2, y2 = clamp_bbox(box.xyxy, width, height)
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
            items.append({"roi_index": roi_index, "box": cvp.RoiBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1), "result": result})

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

    def fuse_frames(
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

        for frame_payload in frame_payloads:
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
            extra_votes: dict[int, float] = {}
            total_weight = 0.0
            score_sum = 0.0
            offset_sum_x = 0.0
            offset_sum_y = 0.0

            for item in observations:
                result = item["result"]
                detail = result["class_details"].get(class_name, result)
                weight = max(0.05, float(detail["class_score"]))
                if detail["count_status"] == "ok":
                    weight += 0.10
                total_weight += weight
                score_sum += float(detail["class_score"])
                offset = detail.get("slot_offset", [0, 0])
                offset_sum_x += float(offset[0]) * weight
                offset_sum_y += float(offset[1]) * weight
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
            boxes = [item["box"].xyxy for item in observations]
            bbox = [int(round(float(np.median([box[pos] for box in boxes])))) for pos in range(4)]
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
                    "slot_offset": [
                        round(offset_sum_x / max(total_weight, 1e-6), 3),
                        round(offset_sum_y / max(total_weight, 1e-6), 3),
                    ],
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
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frames_requested": requested_frame_count,
            "frames_captured": frames_captured,
            "frame_interval_ms": frame_interval_ms,
            "valid_frame_count": valid_frame_count,
            "min_observations_per_roi": min_observations,
            "roi_count": roi_count,
            "roi_count_ok": roi_count_ok,
            "roi_class_consistent": roi_class_consistent,
            "roi_class": roi_class,
            "expected_points_per_roi": expected_points,
            "overall_ok": overall_ok,
            "ng_count": sum(1 for roi in rois_payload if not roi["ok"]),
            "rois": rois_payload,
        }

    @staticmethod
    def choose_base_frame(frames: list[np.ndarray], frame_payloads: list[dict[str, Any]]) -> np.ndarray:
        best_index = 0
        best_score = -1
        for index, payload in enumerate(frame_payloads):
            score = int(payload["valid"]) * 100 + int(payload["roi_count"]) - int(payload["abnormal_count"]) * 2
            if score > best_score:
                best_index = index
                best_score = score
        return frames[min(best_index, len(frames) - 1)].copy()

    def draw_operator_result(
        self,
        frame: np.ndarray,
        payload: dict[str, Any],
        templates: dict[str, dict[int, Any]],
    ) -> np.ndarray:
        image = frame.copy()
        image_width = max(1, image.shape[1])
        box_thickness = max(5, int(round(image_width / 520)))
        point_radius = max(7, int(round(image_width / 360)))
        label_scale = max(1.0, image_width / 1800)
        label_thickness = max(2, int(round(image_width / 900)))

        for roi in payload["rois"]:
            if roi["ok"]:
                continue
            bbox = roi.get("bbox") or []
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = clamp_bbox(bbox, image.shape[1], image.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue
            class_name = roi["class_name"]
            template = templates.get(class_name, {}).get(int(roi["index"]))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), box_thickness)
            self.draw_label(
                image,
                f"NG #{int(roi['index']) + 1:02d} {roi['point_count']}/{roi['expected_count'] or '?'}",
                x1 + 4,
                y1 + int(40 * label_scale),
                (0, 0, 255),
                scale=label_scale,
                thickness=label_thickness,
            )
            if template is None:
                continue
            roi_w = max(1, x2 - x1)
            roi_h = max(1, y2 - y1)
            template_w, template_h = template.size
            offset = roi.get("slot_offset", [0, 0])
            offset_x = float(offset[0]) if len(offset) >= 1 else 0.0
            offset_y = float(offset[1]) if len(offset) >= 2 else 0.0
            for slot_index, (point, hit) in enumerate(zip(template.points, roi["hits"])):
                if hit:
                    continue
                px, py = point
                cx = int(round(x1 + (px + offset_x) * roi_w / template_w))
                cy = int(round(y1 + (py + offset_y) * roi_h / template_h))
                cv2.circle(image, (cx, cy), point_radius, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.putText(
                    image,
                    str(slot_index + 1),
                    (cx + point_radius, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.65, label_scale * 0.7),
                    (0, 0, 255),
                    label_thickness,
                    cv2.LINE_AA,
                )

        header_color = (0, 180, 0) if payload["overall_ok"] else (0, 0, 255)
        header = (
            f"{'PASS' if payload['overall_ok'] else 'NG'}  "
            f"ROI={payload['roi_count']}/20  "
            f"class={payload['roi_class'] or 'mixed'}  "
            f"NG={payload['ng_count']}"
        )
        self.draw_label(image, header, 16, int(58 * label_scale), header_color, scale=max(1.4, label_scale * 1.3), thickness=label_thickness + 1)
        return image

    @staticmethod
    def draw_label(
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
        scale: float,
        thickness: int,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(0, min(image.shape[1] - tw - 8, x))
        y = max(th + 8, min(image.shape[0] - baseline - 8, y))
        cv2.rectangle(image, (x, y - th - 8), (x + tw + 8, y + baseline + 8), (18, 18, 18), -1)
        cv2.putText(image, text, (x + 4, y), font, scale, color, thickness, cv2.LINE_AA)

    def detect(self, frame_count: int = 7, interval_ms: int = 45) -> dict[str, Any]:
        if frame_count % 2 == 0:
            frame_count += 1
        frame_count = max(1, min(31, frame_count))
        interval_ms = max(5, min(1000, interval_ms))
        with self.inference_lock:
            frames = self.capture_frame_sequence(frame_count, interval_ms)
            if not frames:
                raise RuntimeError("没有采集到有效帧")
            return self.process_me70_frames(frames, frame_count, interval_ms, tag="csharp_station")

    def process_me70_frames(
        self,
        frames: list[np.ndarray],
        frame_count: int,
        interval_ms: int,
        tag: str = "csharp_station",
    ) -> dict[str, Any]:
        templates = self.get_templates()
        frame_payloads = [self.analyze_frame(frame, templates) for frame in frames]
        payload = self.fuse_frames(frame_payloads, templates, frame_count, interval_ms)
        base_frame = self.choose_base_frame(frames, frame_payloads)
        result_image = self.draw_operator_result(base_frame, payload, templates)

        run_dir = self.output_dir / datetime.now().strftime(f"%Y%m%d_%H%M%S_{tag}")
        frames_dir = run_dir / "frames"
        for index, frame in enumerate(frames, start=1):
            write_image(frames_dir / f"frame_{index:02d}.jpg", frame)
        write_image(run_dir / "original.jpg", base_frame)
        write_image(run_dir / "result.jpg", result_image)
        save_json(run_dir / "result.json", payload)

        with self.lock:
            self.last_result_image = result_image
            self.last_payload = payload
            self.last_run_dir = run_dir
        self.log(f"检测完成: {'PASS' if payload['overall_ok'] else 'NG'} ROI={payload['roi_count']}/20 NG={payload['ng_count']}")
        response = dict(payload)
        response["run_dir"] = str(run_dir)
        response["message"] = self.result_message(payload)
        return response

    def analyze_dot_panel_frame(
        self,
        frame: np.ndarray,
        model: dict[str, Any],
        pose: dot_panel.PanelPose | None = None,
    ) -> dict[str, Any]:
        params = model.get("detection_params", {})
        templates = model["templates"]
        if pose is None:
            pose = dot_panel.estimate_panel_pose([frame], model)

        panel = dot_panel.prepare_panel(frame, pose, model)
        detected = dot_panel.detect_all_leds(panel, params)

        best_template = ""
        best_result: dict[str, Any] | None = None
        best_score = -1e9
        for tmpl_key, tmpl_data in templates.items():
            tmpl_points = [{"x": p["x"], "y": p["y"]} for p in tmpl_data["points"]]
            result = dot_panel.match_template(detected, tmpl_points, match_radius=30.0)
            score = result["matched_count"] - result["missing_count"] * 2 - result["extra_count"]
            if score > best_score:
                best_score = score
                best_template = tmpl_key
                best_result = result

        is_dark = (not best_template) or len(detected) < 3
        return {
            "panel": panel,
            "detected": detected,
            "best_template": best_template,
            "best_result": best_result,
            "is_dark": is_dark,
            "score": best_score,
            "pose": pose,
        }

    def draw_dot_panel_result(
        self,
        panel: np.ndarray,
        detected: list[dict[str, float]],
        best_result: dict[str, Any],
        tmpl_data: dict[str, Any],
        ok: bool,
    ) -> np.ndarray:
        image = panel.copy()
        image_width = max(1, image.shape[1])
        scale = max(0.7, image_width / 1600)
        thickness = max(2, int(round(image_width / 900)))
        expected = tmpl_data["expected_count"]

        # 模板期望位置（蓝圈）
        for pt in tmpl_data["points"]:
            cv2.circle(image, (int(pt["x"]), int(pt["y"])), 14, (180, 130, 0), 1, cv2.LINE_AA)
        # 检测到的点（绿圈）
        for pt in detected:
            cv2.circle(image, (int(pt["x"]), int(pt["y"])), 8, (0, 255, 0), 2, cv2.LINE_AA)
        # 缺点（黄叉）
        for pt in best_result["missing_points"]:
            cv2.drawMarker(image, (int(pt["x"]), int(pt["y"])), (0, 200, 255), cv2.MARKER_CROSS, 22, 3)
        # 多点（红圈）
        for pt in best_result["extra_points"]:
            cv2.circle(image, (int(pt["x"]), int(pt["y"])), 12, (0, 0, 255), 3, cv2.LINE_AA)

        header_color = (0, 180, 0) if ok else (0, 0, 255)
        header = (
            f"{'PASS' if ok else 'NG'}  [{tmpl_data['name']}]  "
            f"{best_result['matched_count']}/{expected}  "
            f"缺{best_result['missing_count']} 多{best_result['extra_count']}"
        )
        self.draw_label(image, header, 16, int(40 * scale), header_color, scale=max(0.9, scale), thickness=thickness)
        return image

    def detect_dot_panel(self, frame_count: int = 5, interval_ms: int = 60) -> dict[str, Any]:
        frame_count = max(1, min(15, frame_count))
        interval_ms = max(5, min(1000, interval_ms))
        with self.inference_lock:
            frames = self.capture_frame_sequence(frame_count, interval_ms)
            if not frames:
                raise RuntimeError("没有采集到有效帧")
            return self.process_dot_panel_frames(frames, tag="dotpanel_station")

    @staticmethod
    def dot_panel_scene_signature(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA).astype(np.int16)

    def dot_panel_scene_changed(self, signature: np.ndarray, frame_shape: tuple[int, int]) -> bool:
        if self.dot_panel_pose_cache is None or self.dot_panel_pose_signature is None:
            return True
        if self.dot_panel_pose_shape != frame_shape:
            return True
        diff = float(np.mean(np.abs(signature - self.dot_panel_pose_signature)))
        return diff > 12.0

    def get_dot_panel_pose(self, frames: list[np.ndarray], model: dict[str, Any]) -> dot_panel.PanelPose:
        projection = dot_panel.build_max_projection(frames)
        if projection is None:
            return dot_panel.estimate_panel_pose(frames, model, refine_template=False)

        signature = self.dot_panel_scene_signature(projection)
        frame_shape = projection.shape[:2]
        with self.lock:
            if not self.dot_panel_scene_changed(signature, frame_shape):
                cached = self.dot_panel_pose_cache
                if cached is not None:
                    return dot_panel.PanelPose(cached.roi, cached.angle, f"{cached.source}+cached")

        pose = dot_panel.estimate_panel_pose(frames, model, refine_template=False)
        self.update_dot_panel_pose_cache(pose, signature, frame_shape)
        return pose

    def update_dot_panel_pose_cache(
        self,
        pose: dot_panel.PanelPose,
        signature: np.ndarray | None,
        frame_shape: tuple[int, int] | None,
    ) -> None:
        with self.lock:
            self.dot_panel_pose_cache = pose
            self.dot_panel_pose_signature = signature.copy() if signature is not None else None
            self.dot_panel_pose_shape = frame_shape

    def process_dot_panel_frames(
        self,
        frames: list[np.ndarray],
        tag: str = "dotpanel_station",
    ) -> dict[str, Any]:
        model = self.get_dot_panel_model()
        pose = self.get_dot_panel_pose(frames, model)
        analyses = [self.analyze_dot_panel_frame(frame, model, pose) for frame in frames]
        # 优先选择非暗帧中得分最高的一帧；若全暗则取首帧。
        lit = [item for item in analyses if not item["is_dark"]]
        pool = lit if lit else analyses
        best_index = max(range(len(pool)), key=lambda i: pool[i]["score"])
        best = pool[best_index]

        if not best["is_dark"] and (
            best["best_result"]["missing_count"] > 0 or best["best_result"]["extra_count"] > 0
        ):
            projection = dot_panel.build_max_projection(frames)
            signature = self.dot_panel_scene_signature(projection) if projection is not None else None
            frame_shape = projection.shape[:2] if projection is not None else None
            refined_pose = dot_panel.estimate_panel_pose(frames, model, refine_template=True)
            refined_analyses = [self.analyze_dot_panel_frame(frame, model, refined_pose) for frame in frames]
            refined_lit = [item for item in refined_analyses if not item["is_dark"]]
            refined_pool = refined_lit if refined_lit else refined_analyses
            refined_index = max(range(len(refined_pool)), key=lambda i: refined_pool[i]["score"])
            refined_best = refined_pool[refined_index]
            if refined_best["score"] >= best["score"]:
                pose = refined_pose
                analyses = refined_analyses
                lit = refined_lit
                pool = refined_pool
                best = refined_best
                self.update_dot_panel_pose_cache(refined_pose, signature, frame_shape)

        templates = model["templates"]
        if best["is_dark"]:
            payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": MODEL_DOT_PANEL,
                "overall_ok": False,
                "status": "DARK",
                "detected_count": len(best["detected"]),
                "matched_template": "",
                "template_name": "暗帧",
                "expected_count": 0,
                "matched_count": 0,
                "missing_count": 0,
                "extra_count": 0,
                "ng_count": 1,
                "frames_captured": len(frames),
                "pose_source": pose.source,
                "pose_roi": {"x": pose.roi.x, "y": pose.roi.y, "w": pose.roi.w, "h": pose.roi.h},
                "pose_angle": float(pose.angle),
            }
            result_image = best["panel"].copy()
            self.draw_label(result_image, "DARK: 暗帧/未点亮", 16, 40, (0, 200, 255), scale=1.0, thickness=2)
        else:
            tmpl_data = templates[best["best_template"]]
            best_result = best["best_result"]
            ok = best_result["missing_count"] == 0 and best_result["extra_count"] == 0
            payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": MODEL_DOT_PANEL,
                "overall_ok": ok,
                "status": "PASS" if ok else "NG",
                "detected_count": len(best["detected"]),
                "matched_template": best["best_template"],
                "template_name": tmpl_data["name"],
                "expected_count": tmpl_data["expected_count"],
                "matched_count": best_result["matched_count"],
                "missing_count": best_result["missing_count"],
                "extra_count": best_result["extra_count"],
                "ng_count": 0 if ok else 1,
                "frames_captured": len(frames),
                "pose_source": pose.source,
                "pose_roi": {"x": pose.roi.x, "y": pose.roi.y, "w": pose.roi.w, "h": pose.roi.h},
                "pose_angle": float(pose.angle),
            }
            result_image = self.draw_dot_panel_result(
                best["panel"], best["detected"], best_result, tmpl_data, ok
            )

        run_dir = self.output_dir / datetime.now().strftime(f"%Y%m%d_%H%M%S_{tag}")
        frames_dir = run_dir / "frames"
        for index, frame in enumerate(frames, start=1):
            write_image(frames_dir / f"frame_{index:02d}.jpg", frame)
        write_image(run_dir / "result.jpg", result_image)
        save_json(run_dir / "result.json", payload)

        with self.lock:
            self.last_result_image = result_image
            self.last_payload = payload
            self.last_run_dir = run_dir
        self.log(
            f"FS-GRMA-JC 检测完成: {payload['status']} "
            f"匹配[{payload['template_name']}] {payload['matched_count']}/{payload['expected_count']} "
            f"缺{payload['missing_count']} 多{payload['extra_count']}"
        )
        response = dict(payload)
        response["run_dir"] = str(run_dir)
        response["message"] = self.dot_panel_result_message(payload)
        return response

    def detect_brightness_panel(self, frame_count: int = 7, interval_ms: int = 60) -> dict[str, Any]:
        frame_count = max(3, min(31, int(frame_count)))
        interval_ms = max(5, min(1000, int(interval_ms)))
        with self.inference_lock:
            frames = self.capture_frame_sequence(frame_count, interval_ms)
            if not frames:
                raise RuntimeError("没有采集到有效帧")
            return self.process_brightness_panel_frames(frames, tag="brightness_station")

    @staticmethod
    def max_project_frames(frames: list[np.ndarray]) -> np.ndarray:
        if len(frames) == 1:
            return frames[0].copy()
        return np.max(np.stack(frames, axis=0), axis=0).astype(np.uint8)

    @staticmethod
    def estimate_brightness_alignment(
        current_mask: np.ndarray,
        reference_mask: np.ndarray,
        search_px: int,
    ) -> tuple[int, int, float]:
        if current_mask.shape != reference_mask.shape:
            current_mask = cv2.resize(current_mask, (reference_mask.shape[1], reference_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        scale = 0.25
        ref_small = cv2.resize(reference_mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cur_small = cv2.resize(current_mask, (ref_small.shape[1], ref_small.shape[0]), interpolation=cv2.INTER_NEAREST)
        search = max(1, int(round(search_px * scale)))
        padded = cv2.copyMakeBorder(cur_small, search, search, search, search, cv2.BORDER_CONSTANT, value=0)
        result = cv2.matchTemplate(padded, ref_small, cv2.TM_CCORR_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        dx = int(round((loc[0] - search) / scale))
        dy = int(round((loc[1] - search) / scale))
        return dx, dy, float(score)

    @staticmethod
    def projection_runs(
        projection: np.ndarray,
        threshold_ratio: float,
        min_length: int,
        blur_kernel: int,
    ) -> list[tuple[int, int, float]]:
        if projection.size == 0:
            return []
        kernel = max(3, int(blur_kernel))
        if kernel % 2 == 0:
            kernel += 1
        data = projection.astype(np.float32)
        if data.ndim == 1:
            smooth = cv2.GaussianBlur(data.reshape(1, -1), (kernel, 1), 0).ravel()
        else:
            smooth = cv2.GaussianBlur(data, (kernel, 1), 0).ravel()
        peak = float(np.max(smooth)) if smooth.size else 0.0
        if peak <= 0:
            return []
        threshold = max(1.0, peak * float(threshold_ratio))
        active = smooth > threshold
        runs: list[tuple[int, int, float]] = []
        start: int | None = None
        for index, value in enumerate(active):
            if bool(value) and start is None:
                start = index
            is_last = index == len(active) - 1
            if start is not None and (not bool(value) or is_last):
                end = index if not bool(value) else index + 1
                if end - start >= min_length:
                    runs.append((start, end, float(np.max(smooth[start:end]))))
                start = None
        return runs

    @staticmethod
    def vertical_projection_runs(
        projection: np.ndarray,
        threshold_ratio: float,
        min_length: int,
        blur_kernel: int,
    ) -> list[tuple[int, int, float]]:
        if projection.size == 0:
            return []
        kernel = max(3, int(blur_kernel))
        if kernel % 2 == 0:
            kernel += 1
        smooth = cv2.GaussianBlur(projection.astype(np.float32).reshape(-1, 1), (1, kernel), 0).ravel()
        peak = float(np.max(smooth)) if smooth.size else 0.0
        if peak <= 0:
            return []
        threshold = max(1.0, peak * float(threshold_ratio))
        active = smooth > threshold
        runs: list[tuple[int, int, float]] = []
        start: int | None = None
        for index, value in enumerate(active):
            if bool(value) and start is None:
                start = index
            is_last = index == len(active) - 1
            if start is not None and (not bool(value) or is_last):
                end = index if not bool(value) else index + 1
                if end - start >= min_length:
                    runs.append((start, end, float(np.max(smooth[start:end]))))
                start = None
        return runs

    def warp_brightness_panel_by_grid(
        self,
        projection: np.ndarray,
        model: dict[str, Any],
        reference: dict[str, np.ndarray],
        ref_mask: np.ndarray,
        detect_gray: float,
    ) -> tuple[np.ndarray, dict[str, Any]] | None:
        thresholds = model.get("thresholds", {})
        if not bool(thresholds.get("auto_grid_roi", True)):
            return None

        gray = cv2.cvtColor(projection, cv2.COLOR_BGR2GRAY)
        mask = gray > float(detect_gray)
        x_projection = np.count_nonzero(mask, axis=0)
        column_runs = self.projection_runs(
            x_projection,
            float(thresholds.get("auto_roi_x_threshold_ratio", 0.12)),
            int(thresholds.get("auto_roi_min_column_width", 40)),
            int(thresholds.get("auto_roi_x_blur", 25)),
        )
        if len(column_runs) < 6:
            return None
        column_runs = sorted(column_runs, key=lambda run: run[2], reverse=True)[:6]
        column_runs = sorted(column_runs, key=lambda run: run[0])
        median_column_width = float(np.median([max(1, run[1] - run[0]) for run in column_runs]))

        cell_templates = self.build_brightness_cell_templates(model, reference, ref_mask)
        ref_centers = {
            (str(cell["group"]), int(cell["row"])): (
                float(cell["bbox"][0]) + float(cell["bbox"][2]) / 2.0,
                float(cell["bbox"][1]) + float(cell["bbox"][3]) / 2.0,
            )
            for cell in cell_templates
        }

        source_points: list[list[float]] = []
        target_points: list[list[float]] = []
        row_counts: list[int] = []
        detected_cells: list[dict[str, Any]] = []
        detected_groups: list[dict[str, Any]] = []
        column_angles: list[float] = []
        pad_x = int(thresholds.get("auto_roi_column_pad_px", 10))
        cell_pad_x = int(thresholds.get("auto_roi_cell_pad_x", 8))
        cell_pad_y = int(thresholds.get("auto_roi_cell_pad_y", 8))
        raw_columns: list[dict[str, Any]] = []
        for column_index, (x1, x2, strength) in enumerate(column_runs, start=1):
            col_x1 = max(0, x1 - pad_x)
            col_x2 = min(mask.shape[1], x2 + pad_x)
            column_mask = mask[:, col_x1:col_x2]
            y_projection = np.count_nonzero(column_mask, axis=1)
            row_runs = self.vertical_projection_runs(
                y_projection,
                float(thresholds.get("auto_roi_y_threshold_ratio", 0.25)),
                int(thresholds.get("auto_roi_min_row_height", 20)),
                int(thresholds.get("auto_roi_y_blur", 17)),
            )
            row_runs = row_runs[:13]
            raw_columns.append({
                "column_index": column_index,
                "x1": int(x1),
                "x2": int(x2),
                "strength": float(strength),
                "col_x1": int(col_x1),
                "col_x2": int(col_x2),
                "row_runs": row_runs,
            })

        full_columns = [column for column in raw_columns if len(column["row_runs"]) == 13]
        reference_row_centers: list[float] | None = None
        synthetic_row_height = float(thresholds.get("auto_roi_synthetic_row_height", 0.0))
        row_pitch = float(thresholds.get("auto_roi_row_pitch", 0.0))
        if full_columns:
            reference_row_centers = [
                float(np.median([
                    (float(column["row_runs"][row_index][0]) + float(column["row_runs"][row_index][1])) / 2.0
                    for column in full_columns
                ]))
                for row_index in range(13)
            ]
            row_heights = [
                max(1.0, float(y2) - float(y1))
                for column in full_columns
                for y1, y2, _ in column["row_runs"]
            ]
            if synthetic_row_height <= 0:
                synthetic_row_height = float(np.median(row_heights)) if row_heights else 32.0
            diffs = np.diff(np.array(reference_row_centers, dtype=np.float32))
            if row_pitch <= 0 and diffs.size:
                row_pitch = float(np.median(diffs))

        for column in raw_columns:
            column_index = int(column["column_index"])
            x1 = int(column["x1"])
            x2 = int(column["x2"])
            col_x1 = int(column["col_x1"])
            col_x2 = int(column["col_x2"])
            row_runs = list(column["row_runs"])
            row_counts.append(len(row_runs))
            group_y1: int | None = None
            group_y2: int | None = None
            group_x1: int | None = None
            group_x2: int | None = None
            row_infos: list[dict[str, Any]] = []
            row_records: list[dict[str, Any]] = []
            if reference_row_centers is not None and len(row_runs) < 13:
                assigned: dict[int, dict[str, Any]] = {}
                max_row_match_distance = max(
                    synthetic_row_height * 1.1,
                    (row_pitch * 0.48) if row_pitch > 0 else synthetic_row_height * 1.1,
                )
                for y1, y2, strength in row_runs:
                    y_center = (float(y1) + float(y2)) / 2.0
                    nearest_index = int(np.argmin([abs(y_center - center) for center in reference_row_centers]))
                    distance = abs(y_center - reference_row_centers[nearest_index])
                    if distance > max_row_match_distance:
                        continue
                    current = assigned.get(nearest_index)
                    if current is None or distance < float(current["distance"]):
                        assigned[nearest_index] = {
                            "row_index": nearest_index + 1,
                            "y1": int(y1),
                            "y2": int(y2),
                            "strength": float(strength),
                            "synthetic": False,
                            "distance": float(distance),
                        }
                half_height = max(8.0, synthetic_row_height / 2.0)
                for row_index, y_center in enumerate(reference_row_centers, start=1):
                    record = assigned.get(row_index - 1)
                    if record is None:
                        record = {
                            "row_index": row_index,
                            "y1": int(round(y_center - half_height)),
                            "y2": int(round(y_center + half_height)),
                            "strength": 0.0,
                            "synthetic": True,
                            "distance": 0.0,
                        }
                    row_records.append(record)
            else:
                for row_index, (y1, y2, strength) in enumerate(row_runs, start=1):
                    row_records.append({
                        "row_index": row_index,
                        "y1": int(y1),
                        "y2": int(y2),
                        "strength": float(strength),
                        "synthetic": False,
                        "distance": 0.0,
                    })

            for record in row_records:
                row_index = int(record["row_index"])
                y1 = int(record["y1"])
                y2 = int(record["y2"])
                target = ref_centers.get((f"C{column_index}", row_index))
                if target is None:
                    continue
                ref_cell = next(
                    (
                        cell for cell in cell_templates
                        if str(cell["group"]) == f"C{column_index}" and int(cell["row"]) == row_index
                    ),
                    None,
                )
                if ref_cell is None:
                    continue
                row_mask = mask[max(0, int(y1)):min(mask.shape[0], int(y2)), col_x1:col_x2]
                row_x_center = (float(x1) + float(x2)) / 2.0
                if row_mask.size:
                    lit_y, lit_x = np.where(row_mask)
                    if lit_x.size >= 8:
                        row_x_center = float(col_x1) + (float(np.min(lit_x)) + float(np.max(lit_x))) / 2.0
                row_infos.append({
                    "row_index": row_index,
                    "y1": int(y1),
                    "y2": int(y2),
                    "target": target,
                    "ref_cell": ref_cell,
                    "x_center": row_x_center,
                    "y_center": (float(y1) + float(y2)) / 2.0,
                    "synthetic": bool(record.get("synthetic", False)),
                })
            x_fit: np.ndarray | None = None
            real_row_infos = [info for info in row_infos if not bool(info.get("synthetic", False))]
            if len(real_row_infos) >= 4:
                fit_y = np.array([float(info["y_center"]) for info in real_row_infos], dtype=np.float32)
                fit_x = np.array([float(info["x_center"]) for info in real_row_infos], dtype=np.float32)
                x_fit = np.polyfit(fit_y, fit_x, 1)
                column_angles.append(float(np.degrees(np.arctan(float(x_fit[0])))))
            column_angle = float(np.degrees(np.arctan(float(x_fit[0])))) if x_fit is not None else 0.0
            column_width = max(1.0, float(x2 - x1))
            if column_width < median_column_width * 0.90:
                column_width = median_column_width
            nominal_cell_w = max(1, int(round(column_width)) + cell_pad_x * 2)
            for info in row_infos:
                center_x = float(info["x_center"])
                if x_fit is not None:
                    center_x = float(x_fit[0]) * float(info["y_center"]) + float(x_fit[1])
                center_y = float(info["y_center"])
                target = info["target"]
                if not bool(info.get("synthetic", False)):
                    source_points.append([center_x, center_y])
                    target_points.append([target[0], target[1]])
                ref_cell = info["ref_cell"]
                bx1 = max(0, int(round(center_x - nominal_cell_w / 2.0)))
                bx2 = min(mask.shape[1], bx1 + nominal_cell_w)
                bx1 = max(0, bx2 - nominal_cell_w)
                by1 = max(0, int(info["y1"]) - cell_pad_y)
                by2 = min(mask.shape[0], int(info["y2"]) + cell_pad_y)
                detected_cells.append({
                    "id": ref_cell["id"],
                    "group": ref_cell["group"],
                    "row": ref_cell["row"],
                    "bbox": [bx1, by1, max(1, bx2 - bx1), max(1, by2 - by1)],
                    "ref_bbox": ref_cell["bbox"],
                    "center": [round(float(center_x), 3), round(float(center_y), 3)],
                    "size": [max(1, bx2 - bx1), max(1, by2 - by1)],
                    "angle_degrees": round(column_angle, 3),
                })
                group_y1 = by1 if group_y1 is None else min(group_y1, by1)
                group_y2 = by2 if group_y2 is None else max(group_y2, by2)
                group_x1 = bx1 if group_x1 is None else min(group_x1, bx1)
                group_x2 = bx2 if group_x2 is None else max(group_x2, bx2)
            if group_y1 is not None and group_y2 is not None:
                gx1 = max(0, int(group_x1 if group_x1 is not None else int(x1) - cell_pad_x))
                gx2 = min(mask.shape[1], int(group_x2 if group_x2 is not None else int(x2) + cell_pad_x))
                detected_groups.append({
                    "name": f"C{column_index}",
                    "bbox": [gx1, group_y1, max(1, gx2 - gx1), max(1, group_y2 - group_y1)],
                })

        min_points = int(thresholds.get("auto_roi_min_grid_points", 60))
        if len(source_points) < min_points:
            return None
        source = np.array(source_points, dtype=np.float32)
        target = np.array(target_points, dtype=np.float32)
        transform, inliers = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
        if transform is None:
            return None
        inlier_count = int(np.count_nonzero(inliers)) if inliers is not None else len(source_points)
        min_inliers = int(thresholds.get("auto_roi_min_grid_inliers", 55))
        if inlier_count < min_inliers:
            return None

        warped = cv2.warpPerspective(
            projection,
            transform,
            (ref_mask.shape[1], ref_mask.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        x_min = int(min(run[0] for run in column_runs))
        x_max = int(max(run[1] for run in column_runs))
        y_values = [point[1] for point in source_points]
        geometry = {
            "method": "auto_grid",
            "x": x_min,
            "y": int(min(y_values)) if y_values else 0,
            "w": x_max - x_min,
            "h": int(max(y_values) - min(y_values)) if y_values else 0,
            "column_count": len(column_runs),
            "row_counts": row_counts,
            "grid_points": len(source_points),
            "grid_inliers": inlier_count,
            "grid_score": round(float(inlier_count) / max(1, len(source_points)), 4),
            "column_angle_degrees": round(float(np.median(column_angles)), 3) if column_angles else 0.0,
            "cells": detected_cells,
            "groups": detected_groups,
        }
        return warped, geometry

    @staticmethod
    def shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
        matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)

    @staticmethod
    def build_brightness_cell_templates(
        model: dict[str, Any],
        reference: dict[str, np.ndarray],
        ref_mask: np.ndarray,
    ) -> list[dict[str, Any]]:
        thresholds = model.get("thresholds", {})
        row_gap_px = float(thresholds.get("cell_row_gap_px", 55))
        pad = int(thresholds.get("cell_roi_pad_px", 8))
        ref_gray = cv2.cvtColor(reference["image"], cv2.COLOR_BGR2GRAY)
        mask_bool = ref_mask > 0
        height, width = mask_bool.shape[:2]
        cells: list[dict[str, Any]] = []

        for group in model.get("groups", []):
            group_name = str(group["name"])
            comps = sorted(
                [comp for comp in model.get("components", []) if str(comp.get("group", "")) == group_name],
                key=lambda comp: float(comp["y"]) + float(comp["h"]) / 2.0,
            )
            clusters: list[dict[str, Any]] = []
            for comp in comps:
                cy = float(comp["y"]) + float(comp["h"]) / 2.0
                if not clusters or cy - float(clusters[-1]["last_y"]) > row_gap_px:
                    clusters.append({"last_y": cy, "components": [comp]})
                else:
                    clusters[-1]["last_y"] = cy
                    clusters[-1]["components"].append(comp)

            for row_index, cluster in enumerate(clusters, start=1):
                row_components = cluster["components"]
                x1 = max(0, int(min(float(comp["x"]) for comp in row_components) - pad))
                y1 = max(0, int(min(float(comp["y"]) for comp in row_components) - pad))
                x2 = min(width, int(max(float(comp["x"]) + float(comp["w"]) for comp in row_components) + pad))
                y2 = min(height, int(max(float(comp["y"]) + float(comp["h"]) for comp in row_components) + pad))
                if x2 <= x1 or y2 <= y1:
                    continue
                cell_mask = mask_bool[y1:y2, x1:x2]
                expected_area = int(np.count_nonzero(cell_mask))
                if expected_area <= 0:
                    continue
                ref_values = ref_gray[y1:y2, x1:x2][cell_mask]
                cells.append({
                    "id": f"{group_name}R{row_index:02d}",
                    "group": group_name,
                    "row": row_index,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "expected_area": expected_area,
                    "ref_mean": float(ref_values.mean()) if ref_values.size else 1.0,
                    "component_count": len(row_components),
                })
        return cells

    @staticmethod
    def build_brightness_unit_part_templates(
        model: dict[str, Any],
        reference: dict[str, np.ndarray],
        ref_mask: np.ndarray,
        cells: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ref_gray = cv2.cvtColor(reference["image"], cv2.COLOR_BGR2GRAY)
        mask_bool = ref_mask > 0
        parts: list[dict[str, Any]] = []

        for cell in cells:
            x, y, w, h = [int(value) for value in cell["bbox"]]
            components = []
            for comp in model.get("components", []):
                cx, cy, cw, ch = int(comp["x"]), int(comp["y"]), int(comp["w"]), int(comp["h"])
                center_x = cx + cw / 2.0
                center_y = cy + ch / 2.0
                if x <= center_x <= x + w and y <= center_y <= y + h:
                    components.append({"x": cx, "y": cy, "w": cw, "h": ch})
            if not components:
                continue

            components = sorted(components, key=lambda comp: comp["x"])
            battery_components = components[:1]
            value_components = components[1:]
            for part_name, part_components in (("battery", battery_components), ("value", value_components)):
                if not part_components:
                    continue
                x1 = max(0, min(comp["x"] for comp in part_components))
                y1 = max(0, min(comp["y"] for comp in part_components))
                x2 = min(mask_bool.shape[1], max(comp["x"] + comp["w"] for comp in part_components))
                y2 = min(mask_bool.shape[0], max(comp["y"] + comp["h"] for comp in part_components))
                if x2 <= x1 or y2 <= y1:
                    continue
                part_mask = mask_bool[y1:y2, x1:x2]
                expected_area = int(np.count_nonzero(part_mask))
                if expected_area <= 0:
                    continue
                ref_values = ref_gray[y1:y2, x1:x2][part_mask]
                parts.append({
                    "id": f"{cell['id']}_{part_name}",
                    "cell_id": cell["id"],
                    "group": cell["group"],
                    "row": cell["row"],
                    "part": part_name,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "expected_area": expected_area,
                    "ref_mean": float(ref_values.mean()) if ref_values.size else 1.0,
                    "component_count": len(part_components),
                })
        return parts

    def process_brightness_panel_detected_grid(
        self,
        frames: list[np.ndarray],
        projection: np.ndarray,
        model: dict[str, Any],
        reference: dict[str, np.ndarray],
        ref_mask: np.ndarray,
        geometry_payload: dict[str, Any],
        tag: str,
    ) -> dict[str, Any]:
        thresholds = model.get("thresholds", {})
        lit_gray = float(thresholds.get("lit_gray", thresholds.get("active_gray", 115.0)))
        min_group_area_ratio = float(thresholds.get("min_group_area_ratio", 0.72))
        min_group_mean_ratio = float(thresholds.get("min_group_mean_ratio", 0.68))
        min_relative_group_area_ratio = float(thresholds.get("min_relative_group_area_ratio", 0.0))
        min_relative_group_mean_ratio = float(thresholds.get("min_relative_group_mean_ratio", 0.0))
        min_component_mean_ratio = float(thresholds.get("min_component_mean_ratio", 0.62))
        min_component_lit_ratio = float(thresholds.get("min_component_lit_ratio", 0.55))
        max_ng_components_per_group = int(thresholds.get("max_ng_components_per_group", 2))
        min_cell_area_ratio = float(thresholds.get("min_cell_area_ratio", 0.0))
        min_cell_mean_ratio = float(thresholds.get("min_cell_mean_ratio", 0.0))
        min_relative_cell_area_ratio = float(thresholds.get("min_relative_cell_area_ratio", 0.0))
        min_relative_cell_mean_ratio = float(thresholds.get("min_relative_cell_mean_ratio", 0.0))
        max_ng_cells_per_group = int(thresholds.get("max_ng_cells_per_group", 0))
        min_panel_median_gray = float(thresholds.get("min_panel_median_gray", 0.0))
        cell_acceptance = thresholds.get("cell_acceptance", {})
        calibrated_cell_mode = isinstance(cell_acceptance, dict) and bool(cell_acceptance)
        cell_local_align_search_px = int(thresholds.get("cell_local_align_search_px", 10))
        strict_component_shape_min_panel_gray = float(thresholds.get("strict_component_shape_min_panel_gray", 0.0))
        strict_component_shape_max_angle_degrees = float(thresholds.get("strict_component_shape_max_angle_degrees", 0.0))
        strict_component_shape_threshold = float(thresholds.get("strict_component_shape_threshold", 0.35))
        min_strict_component_precision = float(thresholds.get("min_strict_component_precision", 0.0))
        min_strict_component_recall = float(thresholds.get("min_strict_component_recall", 0.0))
        min_strict_component_mean_ratio = float(thresholds.get("min_strict_component_mean_ratio", 0.0))
        strict_component_min_area = int(thresholds.get("strict_component_min_area", 0))
        strict_component_acceptance = thresholds.get("strict_component_acceptance", {})
        if not isinstance(strict_component_acceptance, dict):
            strict_component_acceptance = {}
        strict_segment_acceptance = thresholds.get("strict_digit_segment_acceptance", {})
        if not isinstance(strict_segment_acceptance, dict):
            strict_segment_acceptance = {}
        strict_segment_min_failed = int(thresholds.get("strict_digit_segment_min_failed_per_digit", 2))
        strict_segment_min_failed_per_cell = int(thresholds.get("strict_digit_segment_min_failed_per_cell", 0))
        strict_segment_per_cell_max_relative_mean = float(thresholds.get("strict_digit_segment_per_cell_max_relative_mean", 0.0))
        strict_segment_max_cell_relative_mean = float(thresholds.get("strict_digit_segment_max_cell_relative_mean", 0.0))
        strict_segment_isolated_mild_mean = float(thresholds.get("strict_digit_segment_isolated_mild_relative_mean", 0.0))
        strict_segment_isolated_min_group_cells = int(thresholds.get("strict_digit_segment_isolated_min_group_cells", 0))
        strict_segment_two_failure_min_relative_mean = float(thresholds.get("strict_digit_segment_two_failure_min_relative_mean", 0.0))
        strict_segment_two_failure_ignore_segments = thresholds.get("strict_digit_segment_two_failure_ignore_segments", [])
        if not isinstance(strict_segment_two_failure_ignore_segments, list):
            strict_segment_two_failure_ignore_segments = []
        strict_segment_two_failure_ignore_segments = {str(item) for item in strict_segment_two_failure_ignore_segments}
        strict_percent_ignore_min_relative_mean = float(thresholds.get("strict_percent_component_ignore_min_relative_mean", 0.0))
        strict_percent_ignore_min_align_score = float(thresholds.get("strict_percent_component_ignore_min_align_score", 0.0))
        tilt_degrees = abs(float(geometry_payload.get("column_angle_degrees", 0.0)))
        tilt_start = float(thresholds.get("tilted_cell_acceptance_start_degrees", 2.0))
        tilt_full = max(tilt_start + 0.01, float(thresholds.get("tilted_cell_acceptance_full_degrees", 4.0)))
        tilt_min_factor = float(thresholds.get("tilted_cell_acceptance_min_factor", 0.90))
        tilt_ratio = min(1.0, max(0.0, (tilt_degrees - tilt_start) / (tilt_full - tilt_start)))
        tilted_acceptance_factor = 1.0 - tilt_ratio * (1.0 - tilt_min_factor)

        gray = cv2.cvtColor(projection, cv2.COLOR_BGR2GRAY)
        cell_templates = self.build_brightness_cell_templates(model, reference, ref_mask)
        cell_template_by_id = {str(cell["id"]): cell for cell in cell_templates}
        detected_cells = list(geometry_payload.get("cells", []))
        detected_groups = {str(group.get("name", "")): group for group in geometry_payload.get("groups", [])}

        groups_payload: list[dict[str, Any]] = []
        for group in model.get("groups", []):
            name = str(group["name"])
            detected_group = detected_groups.get(name, {})
            bbox = detected_group.get("bbox", [0, 0, 1, 1])
            groups_payload.append({
                "name": name,
                "x": int(bbox[0]),
                "w": int(bbox[2]),
                "bbox": [int(v) for v in bbox],
                "expected_area": 0,
                "lit_area": 0,
                "area_ratio": 0.0,
                "mean": 0.0,
                "ref_mean": round(float(group.get("ref_mean", 1.0)), 3),
                "mean_ratio": 0.0,
                "p10": 0.0,
                "ref_p10": round(float(group.get("ref_p10", 1.0)), 3),
                "p10_ratio": 1.0,
                "ok": True,
                "issues": [],
                "warnings": [],
                "ng_component_count": 0,
                "ng_cell_count": 0,
            })
        groups_by_name = {str(group["name"]): group for group in groups_payload}

        cell_payloads: list[dict[str, Any]] = []
        resized_cells: dict[str, np.ndarray] = {}
        for detected in detected_cells:
            cell_id = str(detected.get("id", ""))
            template = cell_template_by_id.get(cell_id)
            if template is None:
                continue
            x, y, w, h = [int(v) for v in detected["bbox"]]
            rx, ry, rw, rh = [int(v) for v in template["bbox"]]
            if x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
                continue
            crop = gray[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
            cell_ref_mask = ref_mask[ry:ry + rh, rx:rx + rw] > 0
            local_dx = 0
            local_dy = 0
            local_score = 0.0
            if cell_local_align_search_px > 0 and np.any(cell_ref_mask):
                resized_mask = (resized > lit_gray).astype(np.uint8) * 255
                ref_cell_mask = cell_ref_mask.astype(np.uint8) * 255
                local_dx, local_dy, local_score = self.estimate_brightness_alignment(
                    resized_mask,
                    ref_cell_mask,
                    cell_local_align_search_px,
                )
                resized = self.shift_image(resized, local_dx, local_dy)
            resized_cells[cell_id] = resized
            values = resized[cell_ref_mask]
            expected_area = max(1, int(template["expected_area"]))
            lit_area = int(np.count_nonzero(values > lit_gray))
            area_ratio = float(lit_area) / float(expected_area)
            mean_value = float(values.mean()) if values.size else 0.0
            ref_mean = max(1.0, float(template["ref_mean"]))
            mean_ratio = mean_value / ref_mean
            cell_payloads.append({
                "id": cell_id,
                "group": template["group"],
                "row": template["row"],
                "bbox": [x, y, w, h],
                "ref_bbox": template["bbox"],
                "expected_area": expected_area,
                "lit_area": lit_area,
                "area_ratio": round(area_ratio, 4),
                "mean": round(mean_value, 3),
                "ref_mean": round(ref_mean, 3),
                "mean_ratio": round(mean_ratio, 4),
                "extra_area": 0,
                "extra_ratio": 0.0,
                "component_count": template.get("component_count", 0),
                "local_dx": int(local_dx),
                "local_dy": int(local_dy),
                "local_align_score": round(float(local_score), 4),
                "angle_degrees": round(float(detected.get("angle_degrees", 0.0)), 3),
                "ok": True,
                "issues": [],
            })

        median_cell_area_ratio = float(np.median([float(cell["area_ratio"]) for cell in cell_payloads])) if cell_payloads else 0.0
        median_cell_mean_ratio = float(np.median([float(cell["mean_ratio"]) for cell in cell_payloads])) if cell_payloads else 0.0
        median_cell_gray = float(np.median([float(cell["mean"]) for cell in cell_payloads])) if cell_payloads else 0.0
        bad_cells: list[dict[str, Any]] = []
        bad_cell_ids: set[str] = set()
        for cell in cell_payloads:
            area_ratio = float(cell["area_ratio"])
            mean_ratio = float(cell["mean_ratio"])
            relative_area_ratio = area_ratio / max(median_cell_area_ratio, 1e-6)
            relative_mean_ratio = mean_ratio / max(median_cell_mean_ratio, 1e-6)
            cell["relative_area_ratio"] = round(relative_area_ratio, 4)
            cell["relative_mean_ratio"] = round(relative_mean_ratio, 4)
            issues: list[str] = []
            acceptance = cell_acceptance.get(str(cell["id"]), {}) if isinstance(cell_acceptance, dict) else {}
            if isinstance(acceptance, dict) and acceptance:
                min_area = float(acceptance.get("min_area_ratio", min_cell_area_ratio)) * tilted_acceptance_factor
                min_mean = float(acceptance.get("min_mean_ratio", min_cell_mean_ratio)) * tilted_acceptance_factor
                cell["acceptance_factor"] = round(float(tilted_acceptance_factor), 4)
                cell["min_area_ratio"] = round(float(min_area), 4)
                cell["min_mean_ratio"] = round(float(min_mean), 4)
                if area_ratio < min_area:
                    issues.append(f"低于位置基线亮区 {area_ratio:.0%}")
                if mean_ratio < min_mean:
                    issues.append(f"低于位置基线亮度 {mean_ratio:.0%}")
            else:
                if min_cell_area_ratio > 0 and area_ratio < min_cell_area_ratio:
                    issues.append(f"ROI亮区不足 {area_ratio:.0%}")
                if min_cell_mean_ratio > 0 and mean_ratio < min_cell_mean_ratio:
                    issues.append(f"ROI亮度偏低 {mean_ratio:.0%}")
                if min_relative_cell_area_ratio > 0 and relative_area_ratio < min_relative_cell_area_ratio:
                    issues.append(f"相对亮区不足 {relative_area_ratio:.0%}")
                if min_relative_cell_mean_ratio > 0 and relative_mean_ratio < min_relative_cell_mean_ratio:
                    issues.append(f"相对亮度偏低 {relative_mean_ratio:.0%}")
            cell["issues"] = issues
            cell["ok"] = not issues
            group_payload = groups_by_name.get(str(cell["group"]))
            if group_payload is not None:
                group_payload["expected_area"] = int(group_payload["expected_area"]) + int(cell["expected_area"])
                group_payload["lit_area"] = int(group_payload["lit_area"]) + int(cell["lit_area"])
            if issues:
                bad_cells.append(cell)
                bad_cell_ids.add(str(cell["id"]))
                if group_payload is not None:
                    group_payload["ng_cell_count"] = int(group_payload.get("ng_cell_count", 0)) + 1

        for group in groups_payload:
            cells = [cell for cell in cell_payloads if str(cell["group"]) == str(group["name"])]
            if cells:
                group["area_ratio"] = round(float(sum(float(cell["area_ratio"]) for cell in cells) / len(cells)), 4)
                group["mean"] = round(float(sum(float(cell["mean"]) for cell in cells) / len(cells)), 3)
                group["mean_ratio"] = round(float(sum(float(cell["mean_ratio"]) for cell in cells) / len(cells)), 4)

        median_area_ratio = float(np.median([float(group["area_ratio"]) for group in groups_payload])) if groups_payload else 0.0
        median_mean_ratio = float(np.median([float(group["mean_ratio"]) for group in groups_payload])) if groups_payload else 0.0
        ng_groups: list[str] = []
        for group in groups_payload:
            area_ratio = float(group["area_ratio"])
            mean_ratio = float(group["mean_ratio"])
            relative_area_ratio = area_ratio / max(median_area_ratio, 1e-6)
            relative_mean_ratio = mean_ratio / max(median_mean_ratio, 1e-6)
            group["relative_area_ratio"] = round(relative_area_ratio, 4)
            group["relative_mean_ratio"] = round(relative_mean_ratio, 4)
            issues: list[str] = []
            if area_ratio < min_group_area_ratio:
                issues.append(f"亮区面积不足 {area_ratio:.0%}")
            if mean_ratio < min_group_mean_ratio:
                issues.append(f"平均亮度偏低 {mean_ratio:.0%}")
            if min_relative_group_area_ratio > 0 and relative_area_ratio < min_relative_group_area_ratio:
                issues.append(f"相对亮区不足 {relative_area_ratio:.0%}")
            if min_relative_group_mean_ratio > 0 and relative_mean_ratio < min_relative_group_mean_ratio:
                issues.append(f"相对亮度偏低 {relative_mean_ratio:.0%}")
            group["issues"] = issues
            group["ok"] = not issues
            if issues:
                ng_groups.append(str(group["name"]))

        bad_components: list[dict[str, Any]] = []
        unit_part_templates = self.build_brightness_unit_part_templates(model, reference, ref_mask, cell_templates)
        cell_payload_by_id = {str(cell["id"]): cell for cell in cell_payloads}
        strict_shape_enabled = bool(
            (
                strict_component_acceptance
                or min_strict_component_precision > 0
                or min_strict_component_recall > 0
                or min_strict_component_mean_ratio > 0
            )
            and median_cell_gray >= strict_component_shape_min_panel_gray
            and (strict_component_shape_max_angle_degrees <= 0 or tilt_degrees <= strict_component_shape_max_angle_degrees)
        )
        original_component_templates: list[dict[str, Any]] = []
        if strict_shape_enabled:
            ref_gray = cv2.cvtColor(reference["image"], cv2.COLOR_BGR2GRAY)
            for comp in model.get("components", []):
                px, py, pw, ph = int(comp["x"]), int(comp["y"]), int(comp["w"]), int(comp["h"])
                center_x = px + pw / 2.0
                center_y = py + ph / 2.0
                owner = None
                for cell_template in cell_templates:
                    rx, ry, rw, rh = [int(v) for v in cell_template["bbox"]]
                    if rx <= center_x <= rx + rw and ry <= center_y <= ry + rh:
                        owner = cell_template
                        break
                if owner is None:
                    continue
                x1 = max(0, px)
                y1 = max(0, py)
                x2 = min(ref_mask.shape[1], px + pw)
                y2 = min(ref_mask.shape[0], py + ph)
                if x2 <= x1 or y2 <= y1:
                    continue
                comp_mask = ref_mask[y1:y2, x1:x2] > 0
                expected_area = int(np.count_nonzero(comp_mask))
                if expected_area <= max(0, strict_component_min_area):
                    continue
                ref_values = ref_gray[y1:y2, x1:x2][comp_mask]
                original_component_templates.append({
                    "id": str(comp.get("id", "")),
                    "cell_id": str(owner["id"]),
                    "group": owner["group"],
                    "row": owner["row"],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "expected_area": expected_area,
                    "ref_mean": float(ref_values.mean()) if ref_values.size else 1.0,
                })
        for part in unit_part_templates:
            cell_id = str(part["cell_id"])
            cell = cell_payload_by_id.get(cell_id)
            resized = resized_cells.get(cell_id)
            if cell is None or resized is None:
                continue
            rx, ry, rw, rh = [int(v) for v in cell["ref_bbox"]]
            px, py, pw, ph = [int(v) for v in part["bbox"]]
            local_x = max(0, px - rx)
            local_y = max(0, py - ry)
            local_w = min(pw, rw - local_x)
            local_h = min(ph, rh - local_y)
            if local_w <= 0 or local_h <= 0:
                continue
            comp_mask = ref_mask[py:py + ph, px:px + pw] > 0
            comp_mask = comp_mask[:local_h, :local_w]
            values = resized[local_y:local_y + local_h, local_x:local_x + local_w][comp_mask]
            expected_area = max(1, int(part["expected_area"]))
            lit_ratio = float(np.count_nonzero(values > lit_gray)) / float(expected_area)
            mean_value = float(values.mean()) if values.size else 0.0
            ref_mean = max(1.0, float(part.get("ref_mean", 1.0)))
            mean_ratio = mean_value / ref_mean
            issues: list[str] = []
            if mean_ratio < min_component_mean_ratio:
                issues.append(f"亮度{mean_ratio:.0%}")
            if lit_ratio < min_component_lit_ratio:
                issues.append(f"点亮{lit_ratio:.0%}")
            cx, cy, cw, ch = [int(v) for v in cell["bbox"]]
            draw_bbox = [
                int(round(cx + local_x * cw / max(1, rw))),
                int(round(cy + local_y * ch / max(1, rh))),
                int(round(local_w * cw / max(1, rw))),
                int(round(local_h * ch / max(1, rh))),
            ]
            if issues:
                payload = {
                    "id": part["id"],
                    "cell_id": cell_id,
                    "part": part["part"],
                    "group": part.get("group", ""),
                    "row": part.get("row", 0),
                    "bbox": draw_bbox,
                    "expected_area": expected_area,
                    "lit_area": int(np.count_nonzero(values > lit_gray)),
                    "mean": round(mean_value, 3),
                    "ref_mean": round(ref_mean, 3),
                    "mean_ratio": round(mean_ratio, 4),
                    "lit_ratio": round(lit_ratio, 4),
                    "component_count": part.get("component_count", 0),
                    "ok": False,
                    "issues": issues,
                }
                if calibrated_cell_mode and cell.get("ok"):
                    continue
                bad_components.append(payload)
                group_payload = groups_by_name.get(str(part.get("group", "")))
                if group_payload is not None:
                    group_payload["ng_component_count"] = int(group_payload["ng_component_count"]) + 1

        for part in original_component_templates:
            cell_id = str(part["cell_id"])
            cell = cell_payload_by_id.get(cell_id)
            resized = resized_cells.get(cell_id)
            if cell is None or resized is None:
                continue
            rx, ry, rw, rh = [int(v) for v in cell["ref_bbox"]]
            px, py, pw, ph = [int(v) for v in part["bbox"]]
            x1 = max(px, rx)
            y1 = max(py, ry)
            x2 = min(px + pw, rx + rw, ref_mask.shape[1])
            y2 = min(py + ph, ry + rh, ref_mask.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue
            local_x = x1 - rx
            local_y = y1 - ry
            local_w = x2 - x1
            local_h = y2 - y1
            comp_mask = ref_mask[y1:y2, x1:x2] > 0
            if not np.any(comp_mask):
                continue
            cell_ref_mask = ref_mask[ry:ry + rh, rx:rx + rw] > 0
            cell_values = resized[cell_ref_mask]
            norm_base = float(np.percentile(cell_values, 90)) if cell_values.size else 0.0
            if norm_base <= 1.0:
                continue
            component_crop = resized[local_y:local_y + local_h, local_x:local_x + local_w]
            binary_crop = (component_crop.astype(np.float32) / norm_base) > strict_component_shape_threshold
            true_positive = int(np.count_nonzero(binary_crop & comp_mask))
            false_positive = int(np.count_nonzero(binary_crop & ~comp_mask))
            expected_area = max(1, int(part["expected_area"]))
            precision = true_positive / max(1, true_positive + false_positive)
            recall = true_positive / float(expected_area)
            values = component_crop[comp_mask]
            mean_value = float(values.mean()) if values.size else 0.0
            ref_mean = max(1.0, float(part.get("ref_mean", 1.0)))
            mean_ratio = mean_value / ref_mean
            strict_issues: list[str] = []
            component_acceptance = strict_component_acceptance.get(str(part["id"]), {})
            display_part = "槽位"
            if isinstance(component_acceptance, dict) and component_acceptance:
                expected_cell_id = str(component_acceptance.get("cell_id", ""))
                display_part = str(component_acceptance.get("part", display_part))
                min_recall = float(component_acceptance.get("min_recall", 0.0))
                min_mean = float(component_acceptance.get("min_mean_ratio", 0.0))
                min_precision = float(component_acceptance.get("min_precision", 0.0))
                if expected_cell_id and expected_cell_id != cell_id:
                    continue
                if min_precision > 0 and precision < min_precision:
                    strict_issues.append(f"{display_part}模板不符")
                if min_recall > 0 and recall < min_recall:
                    strict_issues.append(f"{display_part}模板不符")
                if min_mean > 0 and mean_ratio < min_mean:
                    strict_issues.append(f"{display_part}亮度不足")
            else:
                if min_strict_component_precision > 0 and precision < min_strict_component_precision:
                    strict_issues.append(f"{display_part}模板不符")
                if (
                    min_strict_component_recall > 0
                    and min_strict_component_mean_ratio > 0
                    and recall < min_strict_component_recall
                    and mean_ratio < min_strict_component_mean_ratio
                ):
                    strict_issues.append(f"{display_part}亮度不足")
            strict_issues = list(dict.fromkeys(strict_issues))
            if (
                display_part == "%"
                and strict_percent_ignore_min_relative_mean > 0
                and float(cell.get("relative_mean_ratio", 1.0)) >= strict_percent_ignore_min_relative_mean
                and float(cell.get("local_align_score", 0.0)) >= strict_percent_ignore_min_align_score
            ):
                strict_issues = []
            if not strict_issues:
                continue
            cx, cy, cw, ch = [int(v) for v in cell["bbox"]]
            draw_bbox = [
                int(round(cx + local_x * cw / max(1, rw))),
                int(round(cy + local_y * ch / max(1, rh))),
                int(round(local_w * cw / max(1, rw))),
                int(round(local_h * ch / max(1, rh))),
            ]
            payload = {
                "id": f"{cell_id}_{part['id']}",
                "cell_id": cell_id,
                "part": display_part,
                "group": part.get("group", ""),
                "row": part.get("row", 0),
                "bbox": draw_bbox,
                "expected_area": expected_area,
                "lit_area": true_positive,
                "mean": round(mean_value, 3),
                "ref_mean": round(ref_mean, 3),
                "mean_ratio": round(mean_ratio, 4),
                "lit_ratio": round(recall, 4),
                "shape_precision": round(float(precision), 4),
                "shape_threshold": round(float(strict_component_shape_threshold), 4),
                "component_count": 1,
                "ok": False,
                "issues": strict_issues,
            }
            bad_components.append(payload)
            if cell_id not in bad_cell_ids:
                cell.setdefault("issues", []).append(strict_issues[0])
                cell["ok"] = False
                bad_cells.append(cell)
                bad_cell_ids.add(cell_id)
                group_payload = groups_by_name.get(str(part.get("group", "")))
                if group_payload is not None:
                    group_payload["ng_cell_count"] = int(group_payload.get("ng_cell_count", 0)) + 1
            group_payload = groups_by_name.get(str(part.get("group", "")))
            if group_payload is not None:
                group_payload["ng_component_count"] = int(group_payload.get("ng_component_count", 0)) + 1

        segment_failures_by_digit: dict[tuple[str, str], list[dict[str, Any]]] = {}
        if strict_shape_enabled and strict_segment_acceptance:
            for segment_id, segment in strict_segment_acceptance.items():
                if not isinstance(segment, dict):
                    continue
                cell_id = str(segment.get("cell_id", ""))
                cell = cell_payload_by_id.get(cell_id)
                resized = resized_cells.get(cell_id)
                if cell is None or resized is None:
                    continue
                if (
                    strict_segment_max_cell_relative_mean > 0
                    and float(cell.get("relative_mean_ratio", 1.0)) >= strict_segment_max_cell_relative_mean
                ):
                    continue
                rx, ry, rw, rh = [int(v) for v in cell["ref_bbox"]]
                bbox = segment.get("bbox", [])
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                sx, sy, sw, sh = [int(v) for v in bbox]
                x1 = max(sx, rx)
                y1 = max(sy, ry)
                x2 = min(sx + sw, rx + rw, ref_mask.shape[1])
                y2 = min(sy + sh, ry + rh, ref_mask.shape[0])
                if x2 <= x1 or y2 <= y1:
                    continue
                local_x = x1 - rx
                local_y = y1 - ry
                local_w = x2 - x1
                local_h = y2 - y1
                segment_mask = ref_mask[y1:y2, x1:x2] > 0
                expected_area = int(np.count_nonzero(segment_mask))
                if expected_area <= 0:
                    continue
                cell_ref_mask = ref_mask[ry:ry + rh, rx:rx + rw] > 0
                cell_values = resized[cell_ref_mask]
                norm_base = float(np.percentile(cell_values, 90)) if cell_values.size else 0.0
                if norm_base <= 1.0:
                    continue
                segment_crop = resized[local_y:local_y + local_h, local_x:local_x + local_w]
                binary_crop = (segment_crop.astype(np.float32) / norm_base) > strict_component_shape_threshold
                lit_area = int(np.count_nonzero(binary_crop & segment_mask))
                recall = lit_area / max(1, expected_area)
                min_recall = float(segment.get("min_recall", 0.0))
                if min_recall <= 0 or recall >= min_recall:
                    continue
                cx, cy, cw, ch = [int(v) for v in cell["bbox"]]
                draw_bbox = [
                    int(round(cx + local_x * cw / max(1, rw))),
                    int(round(cy + local_y * ch / max(1, rh))),
                    int(round(local_w * cw / max(1, rw))),
                    int(round(local_h * ch / max(1, rh))),
                ]
                digit = str(segment.get("digit", "188"))
                segment_payload = {
                    "id": f"{cell_id}_{segment_id}",
                    "cell_id": cell_id,
                    "part": "188",
                    "digit": digit,
                    "segment": str(segment.get("segment", "")),
                    "group": cell.get("group", ""),
                    "row": cell.get("row", 0),
                    "bbox": draw_bbox,
                    "expected_area": expected_area,
                    "lit_area": lit_area,
                    "mean": round(float(segment_crop[segment_mask].mean()) if np.any(segment_mask) else 0.0, 3),
                    "ref_mean": 0.0,
                    "mean_ratio": 0.0,
                    "lit_ratio": round(float(recall), 4),
                    "min_lit_ratio": round(float(min_recall), 4),
                    "shape_threshold": round(float(strict_component_shape_threshold), 4),
                    "component_count": 1,
                    "ok": False,
                    "issues": ["188段缺失"],
                }
                segment_failures_by_digit.setdefault((cell_id, digit), []).append(segment_payload)

            accepted_segment_failures: dict[tuple[str, str], list[dict[str, Any]]] = {}
            candidate_cells_by_group: dict[str, set[str]] = {}
            segment_failures_by_cell: dict[str, list[dict[str, Any]]] = {}
            for (cell_id, digit), failures in segment_failures_by_digit.items():
                segment_failures_by_cell.setdefault(cell_id, []).extend(failures)
                if len(failures) < max(1, strict_segment_min_failed):
                    continue
                cell = cell_payload_by_id.get(cell_id)
                if cell is None:
                    continue
                if (
                    len(failures) == 2
                    and strict_segment_two_failure_min_relative_mean > 0
                    and float(cell.get("relative_mean_ratio", 1.0)) >= strict_segment_two_failure_min_relative_mean
                    and strict_segment_two_failure_ignore_segments
                    and all(str(failure.get("segment", "")) in strict_segment_two_failure_ignore_segments for failure in failures)
                ):
                    continue
                accepted_segment_failures[(cell_id, digit)] = failures
                candidate_cells_by_group.setdefault(str(cell.get("group", "")), set()).add(cell_id)

            accepted_cells = {cell_id for cell_id, _digit in accepted_segment_failures}
            if strict_segment_min_failed_per_cell > 0:
                for cell_id, failures in segment_failures_by_cell.items():
                    if cell_id in accepted_cells or len(failures) < strict_segment_min_failed_per_cell:
                        continue
                    cell = cell_payload_by_id.get(cell_id)
                    if cell is None:
                        continue
                    if (
                        strict_segment_per_cell_max_relative_mean > 0
                        and float(cell.get("relative_mean_ratio", 1.0)) > strict_segment_per_cell_max_relative_mean
                    ):
                        continue
                    accepted_segment_failures[(cell_id, "cell")] = failures
                    candidate_cells_by_group.setdefault(str(cell.get("group", "")), set()).add(cell_id)

            for (cell_id, _digit), failures in accepted_segment_failures.items():
                cell = cell_payload_by_id.get(cell_id)
                if cell is None:
                    continue
                if (
                    strict_segment_isolated_mild_mean > 0
                    and strict_segment_isolated_min_group_cells > 1
                    and float(cell.get("relative_mean_ratio", 1.0)) >= strict_segment_isolated_mild_mean
                    and len(candidate_cells_by_group.get(str(cell.get("group", "")), set())) < strict_segment_isolated_min_group_cells
                ):
                    continue
                bad_components.extend(failures)
                if cell_id not in bad_cell_ids:
                    cell.setdefault("issues", []).append("188段缺失")
                    cell["ok"] = False
                    bad_cells.append(cell)
                    bad_cell_ids.add(cell_id)
                    group_payload = groups_by_name.get(str(cell.get("group", "")))
                    if group_payload is not None:
                        group_payload["ng_cell_count"] = int(group_payload.get("ng_cell_count", 0)) + 1
                group_payload = groups_by_name.get(str(cell.get("group", "")))
                if group_payload is not None:
                    group_payload["ng_component_count"] = int(group_payload.get("ng_component_count", 0)) + len(failures)

        ng_component_groups = [
            group["name"]
            for group in groups_payload
            if int(group.get("ng_component_count", 0)) > max_ng_components_per_group
        ]
        ng_cell_groups = [
            group["name"]
            for group in groups_payload
            if int(group.get("ng_cell_count", 0)) > max_ng_cells_per_group
        ]
        panel_brightness_ok = bool(min_panel_median_gray <= 0 or median_cell_gray >= min_panel_median_gray)
        if not panel_brightness_ok:
            ng_groups = sorted(set(ng_groups + [str(group["name"]) for group in groups_payload]))
        row_counts = [int(value) for value in geometry_payload.get("row_counts", [])]
        layout_ok = bool(
            int(geometry_payload.get("column_count", 0)) == 6
            and len(row_counts) == 6
            and all(count == 13 for count in row_counts)
            and len(detected_cells) == 78
        )
        alignment_ok = bool(
            layout_ok
            and int(geometry_payload.get("grid_inliers", 0)) >= int(thresholds.get("auto_roi_min_grid_inliers", 55))
        )
        overall_ok = bool(alignment_ok and panel_brightness_ok and not ng_groups and not ng_component_groups and not ng_cell_groups)

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL_BRIGHTNESS_PANEL,
            "overall_ok": overall_ok,
            "status": "PASS" if overall_ok else "NG",
            "frames_captured": len(frames),
            "roi": {
                "x": int(geometry_payload.get("x", 0)),
                "y": int(geometry_payload.get("y", 0)),
                "w": int(geometry_payload.get("w", projection.shape[1])),
                "h": int(geometry_payload.get("h", projection.shape[0])),
                "method": "auto_grid_direct",
            },
            "geometry": {key: value for key, value in geometry_payload.items() if key not in {"cells", "groups"}},
            "alignment": {
                "dx": 0,
                "dy": 0,
                "score": round(float(geometry_payload.get("grid_score", 0.0)), 4),
                "ok": alignment_ok,
                "layout_ok": layout_ok,
                "method": "auto_grid_direct",
            },
            "group_count": len(groups_payload),
            "ng_count": len(ng_groups) + len(ng_component_groups) + len(ng_cell_groups) + (0 if alignment_ok else 1),
            "ng_groups": sorted(set(ng_groups + ng_component_groups + ng_cell_groups)),
            "cell_count": len(cell_payloads),
            "bad_cell_count": len(bad_cells),
            "bad_cells": bad_cells[:80],
            "cells": cell_payloads,
            "unit_part_count": len(unit_part_templates),
            "bad_component_count": len(bad_components),
            "median_cell_gray": round(median_cell_gray, 3),
            "panel_brightness_ok": panel_brightness_ok,
            "groups": groups_payload,
            "bad_components": bad_components[:80],
            "thresholds": thresholds,
        }

        result_image = self.draw_brightness_panel_result(projection.copy(), payload, ref_mask)
        run_dir = self.output_dir / datetime.now().strftime(f"%Y%m%d_%H%M%S_{tag}")
        frames_dir = run_dir / "frames"
        for index, frame in enumerate(frames, start=1):
            write_image(frames_dir / f"frame_{index:02d}.jpg", frame)
        write_image(run_dir / "panel_aligned.jpg", projection)
        write_image(run_dir / "current_mask.png", (gray > lit_gray).astype(np.uint8) * 255)
        write_image(run_dir / "result.jpg", result_image)
        save_json(run_dir / "result.json", payload)

        with self.lock:
            self.last_result_image = result_image
            self.last_payload = payload
            self.last_run_dir = run_dir
        self.log(
            f"白字面板亮度检测完成: {payload['status']} "
            f"grid={payload['alignment']['score']:.2f} NG列={','.join(payload['ng_groups']) or '-'} "
            f"暗亮区={payload['bad_component_count']}"
        )
        response = dict(payload)
        response["run_dir"] = str(run_dir)
        response["message"] = self.brightness_panel_result_message(payload)
        return response

    def process_brightness_panel_frames(
        self,
        frames: list[np.ndarray],
        tag: str = "brightness_station",
    ) -> dict[str, Any]:
        model, reference = self.get_brightness_panel_model()
        roi = model["roi"]
        thresholds = model.get("thresholds", {})
        x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
        projection = self.max_project_frames(frames)
        height, width = projection.shape[:2]
        ref_mask = reference["mask"]
        active_gray = float(thresholds.get("active_gray", 115.0))
        lit_gray = float(thresholds.get("lit_gray", thresholds.get("active_gray", 115.0)))
        geometry_payload: dict[str, Any] = {"method": "fixed_roi"}
        warped = self.warp_brightness_panel_by_grid(projection, model, reference, ref_mask, active_gray)
        if warped is not None:
            panel, geometry_payload = warped
            if geometry_payload.get("cells") and geometry_payload.get("groups"):
                return self.process_brightness_panel_detected_grid(
                    frames,
                    projection,
                    model,
                    reference,
                    ref_mask,
                    geometry_payload,
                    tag,
                )
            x = int(geometry_payload.get("x", x))
            y = int(geometry_payload.get("y", y))
            w = int(ref_mask.shape[1])
            h = int(ref_mask.shape[0])
        else:
            if x < 0 or y < 0 or x + w > width or y + h > height:
                raise RuntimeError(f"白字面板ROI超出画面: roi=({x},{y},{w},{h}) frame=({width},{height})")
            panel = projection[y:y + h, x:x + w].copy()

        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        current_mask = (gray > active_gray).astype(np.uint8) * 255
        search_px = int(thresholds.get("align_search_px", 24))
        dx, dy, align_score = self.estimate_brightness_alignment(current_mask, ref_mask, search_px)
        aligned_gray = self.shift_image(gray, dx, dy)
        aligned_panel = self.shift_image(panel, dx, dy)
        aligned_mask = (aligned_gray > lit_gray).astype(np.uint8) * 255

        min_group_area_ratio = float(thresholds.get("min_group_area_ratio", 0.72))
        min_group_mean_ratio = float(thresholds.get("min_group_mean_ratio", 0.68))
        min_relative_group_area_ratio = float(thresholds.get("min_relative_group_area_ratio", 0.0))
        min_relative_group_mean_ratio = float(thresholds.get("min_relative_group_mean_ratio", 0.0))
        min_group_p10_ratio = float(thresholds.get("min_group_p10_ratio", 0.50))
        min_component_mean_ratio = float(thresholds.get("min_component_mean_ratio", 0.62))
        min_component_lit_ratio = float(thresholds.get("min_component_lit_ratio", 0.55))
        max_ng_components_per_group = int(thresholds.get("max_ng_components_per_group", 2))
        min_cell_area_ratio = float(thresholds.get("min_cell_area_ratio", 0.0))
        min_cell_mean_ratio = float(thresholds.get("min_cell_mean_ratio", 0.0))
        min_relative_cell_area_ratio = float(thresholds.get("min_relative_cell_area_ratio", 0.0))
        min_relative_cell_mean_ratio = float(thresholds.get("min_relative_cell_mean_ratio", 0.0))
        max_cell_extra_ratio = float(thresholds.get("max_cell_extra_ratio", 1.0))
        max_ng_cells_per_group = int(thresholds.get("max_ng_cells_per_group", 0))
        min_alignment_overlap = float(thresholds.get("min_alignment_overlap", 0.65))
        min_panel_median_gray = float(thresholds.get("min_panel_median_gray", 0.0))

        groups_payload: list[dict[str, Any]] = []
        for group in model.get("groups", []):
            gx, gw = int(group["x"]), int(group["w"])
            ref_group_mask = ref_mask[:, gx:gx + gw] > 0
            group_gray = aligned_gray[:, gx:gx + gw]
            expected_area = max(1, int(group.get("expected_area", np.count_nonzero(ref_group_mask))))
            expected_values = group_gray[ref_group_mask]
            lit_values = expected_values[expected_values > lit_gray]
            area_ratio = float(len(lit_values)) / float(expected_area)
            mean_value = float(expected_values.mean()) if expected_values.size else 0.0
            p10_value = float(np.percentile(expected_values, 10)) if expected_values.size else 0.0
            ref_mean = max(1.0, float(group.get("ref_mean", 1.0)))
            ref_p10 = max(1.0, float(group.get("ref_p10", 1.0)))
            mean_ratio = mean_value / ref_mean
            p10_ratio = p10_value / ref_p10
            warnings: list[str] = []
            if p10_ratio < min_group_p10_ratio:
                warnings.append(f"暗部亮度偏低 {p10_ratio:.0%}")
            groups_payload.append({
                "name": group["name"],
                "x": gx,
                "w": gw,
                "expected_area": expected_area,
                "lit_area": int(len(lit_values)),
                "area_ratio": round(area_ratio, 4),
                "mean": round(mean_value, 3),
                "ref_mean": round(ref_mean, 3),
                "mean_ratio": round(mean_ratio, 4),
                "p10": round(p10_value, 3),
                "ref_p10": round(ref_p10, 3),
                "p10_ratio": round(p10_ratio, 4),
                "ok": True,
                "issues": [],
                "warnings": warnings,
                "ng_component_count": 0,
                "ng_cell_count": 0,
            })

        median_area_ratio = float(np.median([float(group["area_ratio"]) for group in groups_payload])) if groups_payload else 0.0
        median_mean_ratio = float(np.median([float(group["mean_ratio"]) for group in groups_payload])) if groups_payload else 0.0
        ng_groups: list[str] = []
        for group in groups_payload:
            area_ratio = float(group["area_ratio"])
            mean_ratio = float(group["mean_ratio"])
            relative_area_ratio = area_ratio / max(median_area_ratio, 1e-6)
            relative_mean_ratio = mean_ratio / max(median_mean_ratio, 1e-6)
            group["relative_area_ratio"] = round(relative_area_ratio, 4)
            group["relative_mean_ratio"] = round(relative_mean_ratio, 4)
            issues: list[str] = []
            if area_ratio < min_group_area_ratio:
                issues.append(f"亮区面积不足 {area_ratio:.0%}")
            if mean_ratio < min_group_mean_ratio:
                issues.append(f"平均亮度偏低 {mean_ratio:.0%}")
            if min_relative_group_area_ratio > 0 and relative_area_ratio < min_relative_group_area_ratio:
                issues.append(f"相对亮区不足 {relative_area_ratio:.0%}")
            if min_relative_group_mean_ratio > 0 and relative_mean_ratio < min_relative_group_mean_ratio:
                issues.append(f"相对亮度偏低 {relative_mean_ratio:.0%}")
            group["issues"] = issues
            group["ok"] = not issues
            if issues:
                ng_groups.append(str(group["name"]))

        groups_by_name = {str(group["name"]): group for group in groups_payload}
        cell_payloads: list[dict[str, Any]] = []
        cell_templates = self.build_brightness_cell_templates(model, reference, ref_mask)
        ref_kernel = np.ones((5, 5), np.uint8)
        for cell in cell_templates:
            cx, cy, cw, ch = [int(value) for value in cell["bbox"]]
            cell_ref_mask = ref_mask[cy:cy + ch, cx:cx + cw] > 0
            cell_current_mask = aligned_mask[cy:cy + ch, cx:cx + cw] > 0
            values = aligned_gray[cy:cy + ch, cx:cx + cw][cell_ref_mask]
            expected_area = max(1, int(cell["expected_area"]))
            lit_area = int(np.count_nonzero(values > lit_gray))
            area_ratio = float(lit_area) / float(expected_area)
            mean_value = float(values.mean()) if values.size else 0.0
            ref_mean = max(1.0, float(cell["ref_mean"]))
            mean_ratio = mean_value / ref_mean
            dilated_ref = cv2.dilate(cell_ref_mask.astype(np.uint8), ref_kernel, iterations=1) > 0
            extra_area = int(np.count_nonzero(cell_current_mask & ~dilated_ref))
            extra_ratio = float(extra_area) / float(expected_area)
            cell_payloads.append({
                "id": cell["id"],
                "group": cell["group"],
                "row": cell["row"],
                "bbox": cell["bbox"],
                "expected_area": expected_area,
                "lit_area": lit_area,
                "area_ratio": round(area_ratio, 4),
                "mean": round(mean_value, 3),
                "ref_mean": round(ref_mean, 3),
                "mean_ratio": round(mean_ratio, 4),
                "extra_area": extra_area,
                "extra_ratio": round(extra_ratio, 4),
                "component_count": cell.get("component_count", 0),
                "ok": True,
                "issues": [],
            })

        median_cell_area_ratio = float(np.median([float(cell["area_ratio"]) for cell in cell_payloads])) if cell_payloads else 0.0
        median_cell_mean_ratio = float(np.median([float(cell["mean_ratio"]) for cell in cell_payloads])) if cell_payloads else 0.0
        median_cell_gray = float(np.median([float(cell["mean"]) for cell in cell_payloads])) if cell_payloads else 0.0
        bad_cells: list[dict[str, Any]] = []
        for cell in cell_payloads:
            area_ratio = float(cell["area_ratio"])
            mean_ratio = float(cell["mean_ratio"])
            extra_ratio = float(cell["extra_ratio"])
            relative_area_ratio = area_ratio / max(median_cell_area_ratio, 1e-6)
            relative_mean_ratio = mean_ratio / max(median_cell_mean_ratio, 1e-6)
            cell["relative_area_ratio"] = round(relative_area_ratio, 4)
            cell["relative_mean_ratio"] = round(relative_mean_ratio, 4)
            issues: list[str] = []
            if min_cell_area_ratio > 0 and area_ratio < min_cell_area_ratio:
                issues.append(f"ROI亮区不足 {area_ratio:.0%}")
            if min_cell_mean_ratio > 0 and mean_ratio < min_cell_mean_ratio:
                issues.append(f"ROI亮度偏低 {mean_ratio:.0%}")
            if min_relative_cell_area_ratio > 0 and relative_area_ratio < min_relative_cell_area_ratio:
                issues.append(f"相对亮区不足 {relative_area_ratio:.0%}")
            if min_relative_cell_mean_ratio > 0 and relative_mean_ratio < min_relative_cell_mean_ratio:
                issues.append(f"相对亮度偏低 {relative_mean_ratio:.0%}")
            if extra_ratio > max_cell_extra_ratio:
                issues.append(f"ROI多亮区 {extra_ratio:.0%}")
            cell["issues"] = issues
            cell["ok"] = not issues
            if issues:
                bad_cells.append(cell)
                group_payload = groups_by_name.get(str(cell.get("group", "")))
                if group_payload is not None:
                    group_payload["ng_cell_count"] = int(group_payload.get("ng_cell_count", 0)) + 1

        bad_components: list[dict[str, Any]] = []
        unit_part_templates = self.build_brightness_unit_part_templates(model, reference, ref_mask, cell_templates)
        for part in unit_part_templates:
            cx, cy, cw, ch = [int(value) for value in part["bbox"]]
            if cx < 0 or cy < 0 or cx + cw > ref_mask.shape[1] or cy + ch > ref_mask.shape[0]:
                continue
            comp_mask = ref_mask[cy:cy + ch, cx:cx + cw] > 0
            if not np.any(comp_mask):
                continue
            values = aligned_gray[cy:cy + ch, cx:cx + cw][comp_mask]
            expected_area = max(1, int(part["expected_area"]))
            lit_ratio = float(np.count_nonzero(values > lit_gray)) / float(expected_area)
            mean_value = float(values.mean()) if values.size else 0.0
            ref_mean = max(1.0, float(part.get("ref_mean", 1.0)))
            mean_ratio = mean_value / ref_mean
            issues: list[str] = []
            if mean_ratio < min_component_mean_ratio:
                issues.append(f"亮度{mean_ratio:.0%}")
            if lit_ratio < min_component_lit_ratio:
                issues.append(f"点亮{lit_ratio:.0%}")
            ok = not issues
            payload = {
                "id": part["id"],
                "cell_id": part["cell_id"],
                "part": part["part"],
                "group": part.get("group", ""),
                "row": part.get("row", 0),
                "bbox": [cx, cy, cw, ch],
                "expected_area": expected_area,
                "lit_area": int(np.count_nonzero(values > lit_gray)),
                "mean": round(mean_value, 3),
                "ref_mean": round(ref_mean, 3),
                "mean_ratio": round(mean_ratio, 4),
                "lit_ratio": round(lit_ratio, 4),
                "component_count": part.get("component_count", 0),
                "ok": ok,
                "issues": issues,
            }
            if not ok:
                bad_components.append(payload)
                group_payload = groups_by_name.get(str(part.get("group", "")))
            if group_payload is not None:
                group_payload["ng_component_count"] = int(group_payload["ng_component_count"]) + 1

        ng_component_groups = [
            group["name"]
            for group in groups_payload
            if int(group.get("ng_component_count", 0)) > max_ng_components_per_group
        ]
        ng_cell_groups = [
            group["name"]
            for group in groups_payload
            if int(group.get("ng_cell_count", 0)) > max_ng_cells_per_group
        ]
        grid_ok = (
            str(geometry_payload.get("method", "")) == "auto_grid"
            and int(geometry_payload.get("grid_inliers", 0)) >= int(thresholds.get("auto_roi_min_grid_inliers", 55))
        )
        alignment_ok = bool(align_score >= min_alignment_overlap or grid_ok)
        panel_brightness_ok = bool(min_panel_median_gray <= 0 or median_cell_gray >= min_panel_median_gray)
        if not panel_brightness_ok:
            ng_groups = sorted(set(ng_groups + [str(group["name"]) for group in groups_payload]))
        overall_ok = bool(alignment_ok and panel_brightness_ok and not ng_groups and not ng_component_groups and not ng_cell_groups)

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL_BRIGHTNESS_PANEL,
            "overall_ok": overall_ok,
            "status": "PASS" if overall_ok else "NG",
            "frames_captured": len(frames),
            "roi": {"x": x, "y": y, "w": w, "h": h, "method": geometry_payload.get("method", "fixed_roi")},
            "geometry": geometry_payload,
            "alignment": {
                "dx": dx,
                "dy": dy,
                "score": round(align_score, 4),
                "ok": alignment_ok,
                "method": geometry_payload.get("method", "fixed_roi"),
            },
            "group_count": len(groups_payload),
            "ng_count": len(ng_groups) + len(ng_component_groups) + len(ng_cell_groups) + (0 if alignment_ok else 1),
            "ng_groups": sorted(set(ng_groups + ng_component_groups + ng_cell_groups)),
            "cell_count": len(cell_payloads),
            "bad_cell_count": len(bad_cells),
            "bad_cells": bad_cells[:80],
            "cells": cell_payloads,
            "unit_part_count": len(unit_part_templates),
            "bad_component_count": len(bad_components),
            "median_cell_gray": round(median_cell_gray, 3),
            "panel_brightness_ok": panel_brightness_ok,
            "groups": groups_payload,
            "bad_components": bad_components[:80],
            "thresholds": thresholds,
        }

        result_image = self.draw_brightness_panel_result(aligned_panel, payload, ref_mask)
        run_dir = self.output_dir / datetime.now().strftime(f"%Y%m%d_%H%M%S_{tag}")
        frames_dir = run_dir / "frames"
        for index, frame in enumerate(frames, start=1):
            write_image(frames_dir / f"frame_{index:02d}.jpg", frame)
        write_image(run_dir / "panel_aligned.jpg", aligned_panel)
        write_image(run_dir / "current_mask.png", aligned_mask)
        write_image(run_dir / "result.jpg", result_image)
        save_json(run_dir / "result.json", payload)

        with self.lock:
            self.last_result_image = result_image
            self.last_payload = payload
            self.last_run_dir = run_dir
        self.log(
            f"白字面板亮度检测完成: {payload['status']} "
            f"align={align_score:.2f} NG列={','.join(payload['ng_groups']) or '-'} "
            f"暗亮区={payload['bad_component_count']}"
        )
        response = dict(payload)
        response["run_dir"] = str(run_dir)
        response["message"] = self.brightness_panel_result_message(payload)
        return response

    def draw_brightness_panel_result(
        self,
        panel: np.ndarray,
        payload: dict[str, Any],
        ref_mask: np.ndarray,
    ) -> np.ndarray:
        image = panel.copy()
        image_width = max(1, image.shape[1])
        scale = max(0.85, image_width / 1800)
        thickness = max(2, int(round(image_width / 900)))
        if ref_mask.shape[:2] == image.shape[:2]:
            overlay = image.copy()
            overlay[ref_mask > 0] = (overlay[ref_mask > 0] * 0.65 + np.array([0, 120, 0]) * 0.35).astype(np.uint8)
            image = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)
        thresholds = payload.get("thresholds", {})
        max_ng_components_per_group = int(thresholds.get("max_ng_components_per_group", 2))

        for group in payload.get("groups", []):
            if "bbox" in group:
                gx, gy, gw, gh = [int(v) for v in group["bbox"]]
            else:
                gx, gy, gw, gh = int(group["x"]), 0, int(group["w"]), image.shape[0] - 1
            color = (
                (0, 180, 0)
                if group.get("ok")
                and int(group.get("ng_component_count", 0)) <= max_ng_components_per_group
                and int(group.get("ng_cell_count", 0)) == 0
                else (0, 0, 255)
            )
            cv2.rectangle(image, (gx, gy), (gx + gw, gy + gh), color, 4)
            label = (
                f"{group['name']} "
                f"A{float(group['area_ratio']):.0%} "
                f"L{float(group['mean_ratio']):.0%}"
            )
            self.draw_label(image, label, gx + 8, max(42, gy + 34), color, scale=max(0.8, scale * 0.78), thickness=thickness)

        for cell in payload.get("bad_cells", [])[:40]:
            cx, cy, cw, ch = [int(v) for v in cell["bbox"]]
            cv2.rectangle(image, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 3)
            self.draw_label(
                image,
                str(cell.get("id", "")),
                cx + 2,
                max(28, cy - 6),
                (0, 0, 255),
                scale=max(0.55, scale * 0.58),
                thickness=max(1, thickness - 1),
            )

        if bool(thresholds.get("draw_component_debug_boxes", False)):
            ng_groups = set(str(group) for group in payload.get("ng_groups", []))
            visible_bad_components = [
                comp for comp in payload.get("bad_components", [])
                if str(comp.get("group", "")) in ng_groups
            ]
            for comp in visible_bad_components[:60]:
                cx, cy, cw, ch = [int(v) for v in comp["bbox"]]
                cv2.rectangle(image, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)

        header_color = (0, 180, 0) if payload["overall_ok"] else (0, 0, 255)
        ng_groups = ",".join(payload.get("ng_groups", [])) or "-"
        align = payload.get("alignment", {})
        header = (
            f"{'PASS' if payload['overall_ok'] else 'NG'}  BRIGHTNESS_PANEL  "
            f"align={float(align.get('score', 0.0)):.2f}  "
            f"NG={ng_groups}  roi={payload.get('bad_cell_count', 0)}  dim={payload.get('bad_component_count', 0)}"
        )
        self.draw_label(image, header, 16, int(92 * scale), header_color, scale=max(1.0, scale), thickness=thickness)
        return image

    def detect_video(self, path: str, model: str, max_frames: int = 60, stride: int = 5) -> dict[str, Any]:
        video_path = Path(path)
        if not path.strip():
            raise FileNotFoundError("视频文件路径为空")
        if not video_path.is_file():
            raise FileNotFoundError(f"视频文件不存在或不是有效文件: {video_path}")
        frames = self.read_video_frames(video_path, max_frames=max_frames, stride=stride)
        if not frames:
            raise RuntimeError(f"无法从视频读取有效帧: {video_path}")
        self.log(f"视频测试[{model}]: {video_path.name} 抽取 {len(frames)} 帧")
        with self.inference_lock:
            normalized_model = normalize_model(model)
            if normalized_model == MODEL_DOT_PANEL:
                response = self.process_dot_panel_frames(frames, tag="dotpanel_video")
            elif normalized_model == MODEL_BRIGHTNESS_PANEL:
                response = self.process_brightness_panel_frames(frames, tag="brightness_video")
            else:
                frame_count = len(frames) if len(frames) % 2 == 1 else max(1, len(frames) - 1)
                response = self.process_me70_frames(frames, frame_count, stride, tag="me70_video")
        response["source"] = "video"
        response["video"] = str(video_path)
        return response

    @staticmethod
    def read_video_frames(video_path: Path, max_frames: int = 60, stride: int = 5) -> list[np.ndarray]:
        stride = max(1, int(stride))
        max_frames = max(1, int(max_frames))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        frames: list[np.ndarray] = []
        index = 0
        try:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % stride == 0:
                    frames.append(frame)
                index += 1
        finally:
            cap.release()
        return frames

    def detect_marquee(self, duration_seconds: float = 22.0, interval_ms: int = 66) -> dict[str, Any]:
        with self.inference_lock:
            template_path = Path(MARQUEE_TEMPLATE_MODEL_DEFAULT)
            if not template_path.exists():
                raise FileNotFoundError(f"跑马灯模板不存在: {template_path}")
            frames, fps = self.capture_frame_sequence_for_duration(duration_seconds, interval_ms)
            if len(frames) < 5:
                raise RuntimeError("跑马灯检测没有采集到足够帧")

            run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_marquee_station")
            frames_dir = run_dir / "frames"
            for index, frame in enumerate(frames, start=1):
                if index <= 20 or index == len(frames):
                    write_image(frames_dir / f"frame_{index:04d}.jpg", frame)

            payload = marquee.detect_frames(
                frames=frames,
                fps=fps,
                template_path=template_path,
                output_dir=run_dir,
                source="camera",
            )
            result_image = frames[-1].copy()
            write_image(run_dir / "result.jpg", result_image)
            with self.lock:
                self.last_result_image = result_image
                self.last_payload = payload
                self.last_run_dir = run_dir
            self.log(
                f"跑马灯检测完成: {'PASS' if payload['overall_ok'] else 'NG'} "
                f"score={payload.get('order_score')} NG={payload.get('ng_count')}"
            )
            response = dict(payload)
            response["run_dir"] = str(run_dir)
            response["message"] = self.marquee_result_message(payload)
            return response

    @staticmethod
    def result_message(payload: dict[str, Any]) -> str:
        if payload.get("overall_ok"):
            return f"PASS: ROI={payload.get('roi_count')}/20 {payload.get('roi_class') or ''}"
        bad = [roi for roi in payload.get("rois", []) if not roi.get("ok")]
        if not bad:
            return "NG: 检测结果异常"
        first = bad[0]
        return f"NG: {len(bad)} 个组件异常，首个 #{int(first['index']) + 1:02d} 缺点 {len(first.get('missing_slots', []))} 个"

    @staticmethod
    def dot_panel_result_message(payload: dict[str, Any]) -> str:
        if payload.get("status") == "DARK":
            return f"DARK: 暗帧/未点亮（检测 {payload.get('detected_count', 0)} 点）"
        name = payload.get("template_name", "")
        matched = payload.get("matched_count", 0)
        expected = payload.get("expected_count", 0)
        if payload.get("overall_ok"):
            return f"PASS: [{name}] {matched}/{expected} 全亮正常"
        return (
            f"NG: [{name}] {matched}/{expected} "
            f"缺{payload.get('missing_count', 0)} 多{payload.get('extra_count', 0)}"
        )

    @staticmethod
    def brightness_panel_result_message(payload: dict[str, Any]) -> str:
        align = payload.get("alignment", {})
        align_score = float(align.get("score", 0.0))
        if payload.get("overall_ok"):
            return (
                f"PASS: 白字面板亮度合格 align={align_score:.2f} "
                f"异常ROI={payload.get('bad_cell_count', 0)} 暗亮区={payload.get('bad_component_count', 0)}"
            )
        ng_groups = ",".join(payload.get("ng_groups", [])) or "-"
        if (
            not align.get("ok", True)
            and ng_groups == "-"
            and int(payload.get("bad_cell_count", 0)) == 0
            and int(payload.get("bad_component_count", 0)) == 0
        ):
            return f"NG: 白字面板位置匹配不足 align={align_score:.2f}"
        bad_cells = payload.get("bad_cells", [])
        first_cells = ",".join(str(cell.get("id", "")) for cell in bad_cells[:5]) if isinstance(bad_cells, list) else ""
        roi_text = f" 异常ROI={payload.get('bad_cell_count', 0)}"
        if first_cells:
            roi_text += f"({first_cells})"
        return (
            f"NG: 白字面板亮度异常 NG列={ng_groups}{roi_text} "
            f"暗亮区={payload.get('bad_component_count', 0)} align={align_score:.2f}"
        )

    @staticmethod
    def marquee_result_message(payload: dict[str, Any]) -> str:
        score = float(payload.get("order_score", 0.0))
        if payload.get("overall_ok"):
            return f"PASS: 跑马顺序正常 score={score:.2f} events={payload.get('observed_event_count', 0)}"
        issues = payload.get("issues", [])
        if not issues:
            return f"NG: 跑马灯检测异常 score={score:.2f}"
        first = issues[0]
        return f"NG: {len(issues)} 项异常，{first.get('message', '跑马顺序异常')}"


BACKEND = InspectionBackend()


class Handler(BaseHTTPRequestHandler):
    server_version = "LedInspectionBackend/1.0"

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/status":
                self.write_response(*json_bytes(BACKEND.status()))
            elif path == "/cameras":
                self.write_response(*json_bytes({"cameras": [BACKEND.camera_to_json(camera) for camera in BACKEND.cameras]}))
            elif path == "/frame.jpg":
                self.write_response(200, BACKEND.get_frame_jpeg(), "image/jpeg")
            elif path == "/result.jpg":
                self.write_response(200, BACKEND.get_result_jpeg(), "image/jpeg")
            else:
                self.write_response(*json_bytes({"error": "not found"}, status=404))
        except Exception as exc:
            self.write_error(exc)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            payload = self.read_json()
            BACKEND.log(f"POST {path} payload: {payload}")
            if path == "/refresh_cameras":
                self.write_response(*json_bytes({"cameras": BACKEND.refresh_cameras()}))
            elif path == "/configure":
                self.write_response(*json_bytes(BACKEND.configure(payload)))
            elif path == "/start_preview":
                self.write_response(*json_bytes(BACKEND.start_preview(str(payload.get("camera", "")))))
            elif path == "/stop_preview":
                self.write_response(*json_bytes(BACKEND.stop_preview()))
            elif path == "/detect":
                model = normalize_model(str(payload.get("model", MODEL_ME70)))
                if model == MODEL_DOT_PANEL:
                    self.write_response(*json_bytes(BACKEND.detect_dot_panel(int(payload.get("frame_count", 5)), int(payload.get("interval_ms", 60)))))
                elif model == MODEL_BRIGHTNESS_PANEL:
                    self.write_response(*json_bytes(BACKEND.detect_brightness_panel(int(payload.get("frame_count", 7)), int(payload.get("interval_ms", 60)))))
                else:
                    self.write_response(*json_bytes(BACKEND.detect(int(payload.get("frame_count", 7)), int(payload.get("interval_ms", 45)))))
            elif path == "/detect_video":
                self.write_response(*json_bytes(BACKEND.detect_video(
                    str(payload.get("path", "")),
                    str(payload.get("model", MODEL_ME70)),
                    int(payload.get("max_frames", 60)),
                    int(payload.get("stride", 5)),
                )))
            else:
                self.write_response(*json_bytes({"error": "not found"}, status=404))
        except Exception as exc:
            self.write_error(exc)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8"))

    def write_error(self, exc: Exception) -> None:
        BACKEND.log(f"请求失败: {exc}\n{traceback.format_exc()}")
        self.write_response(*json_bytes({"error": str(exc)}, status=500))

    def write_response(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="LED inspection local backend for the C# station UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        print(f"LED inspection backend listening on http://{args.host}:{args.port}", flush=True)
    except OSError:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        BACKEND.stop_preview()
        server.server_close()


if __name__ == "__main__":
    main()
