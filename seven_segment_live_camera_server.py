from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from desktop_app import (
    CameraInfo,
    HikvisionCameraStream,
    OpenCVCameraStream,
    RESOURCE_ROOT,
    RUNTIME_ROOT,
    enumerate_hikvision_cameras,
    enumerate_opencv_cameras,
)
from scripts.detect_scan_frame_sequence import (
    Template,
    bit_string_to_vector,
    dilate_vector,
    estimate_image_alignment,
    find_best_template,
    jaccard_similarity,
    load_model,
    mask_to_vector,
    warp_to_reference,
)
from scripts.create_sevenseg_live_camera_demo import HTML as CLIENT_CAMERA_HTML
from scripts.create_sevenseg_live_camera_demo import build_model_data as build_client_camera_model_data
from scripts.extract_scan_frame_templates import make_mask, mask_to_color_bits, color_similarity
from scripts.scan_refresh_seven_segment_probe import Roi, crop_roi, detect_display_roi


ROOT = Path(__file__).resolve().parent
RESOURCE_BASE = RESOURCE_ROOT
RUNTIME_BASE = RUNTIME_ROOT
DEFAULT_MODEL = RESOURCE_BASE / "models" / "seven_segment" / "Video_20260520165824830_fine_color" / "templates.json"
OUTPUT_ROOT = RUNTIME_BASE / "output"


def json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes, str]:
    return status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"


def jpeg_bytes(image: np.ndarray, quality: int = 78) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buffer.tobytes()


@dataclass
class DetectionState:
    cursor: int = 0
    ordered_hits: int = 0
    unknown_segments: int = 0
    last_template: str = ""
    last_status: str = "dark"
    frames: int = 0
    result: str = "WAIT"


class SevenSegmentLiveBackend:
    def __init__(self, model_path: Path) -> None:
        self.lock = threading.RLock()
        self.detector_lock = threading.Lock()
        self.logs: list[str] = []
        self.cameras: list[CameraInfo] = []
        self.stream: OpenCVCameraStream | HikvisionCameraStream | None = None
        self.detector_thread: threading.Thread | None = None
        self.detector_stop = threading.Event()
        self.video_thread: threading.Thread | None = None
        self.video_stop = threading.Event()
        self.video_source = ""
        self.sequence_only_video = False
        self.detecting = False
        self.frozen = False
        self.model_path = model_path
        self.templates, self.sequence, self.model_summary = load_model(model_path, 1, 0.0)
        # 不再过滤过渡模板，每一个模板都要匹配
        self.log(f"已加载全部模板: {len(self.templates)} 个, 序列长度: {len(self.sequence)}")
        self.auto_roi_enabled = True  # 启用自动ROI检测
        self.auto_roi_frames = 300  # 用前N帧做投影来检测ROI (覆盖开头黑帧)
        self.auto_roi_projection_threshold = 40  # 投影阈值
        self.auto_roi_pad_ratio = 0.06  # ROI边距比例
        self.templates_by_id = {template.id: template for template in self.templates}
        self.template_mask_cache: dict[tuple[str, tuple[int, int]], np.ndarray] = {}
        self.template_image_cache: dict[tuple[str, tuple[int, int]], np.ndarray] = {}
        self.grid_cols = 96
        self.grid_rows = 64
        self.active_threshold = 55
        self.cell_threshold = 28
        self.match_threshold = 0.68
        self.tolerant_threshold = 0.58
        self.tolerant_dilate = 4
        self.aligned_expected_threshold = 0.52
        self.aligned_expected_tolerant_threshold = 0.60
        self.alignment_accept_min_score = 0.12
        self.expected_min_recall = 0.70
        self.expected_max_extra_ratio_small = 1.8
        self.expected_max_extra_ratio_large = 0.22
        self.expected_small_template_cells = 80
        self.color_check_enabled = True
        self.expected_min_color_recall = 0.70
        self.global_alignment_enabled = True
        self.global_alignment_min_active_cells = 400
        self.global_alignment_min_score = 0.75
        self.min_ordered_coverage = 0.90
        self.start_sync_window = 120
        self.max_unknown_segments = 3
        self.min_active_cells = 1
        self.quality_enabled = True
        self.quality_missing_ratio_review = 0.006
        self.quality_missing_ratio_ng = 0.025
        self.quality_extra_ratio_review = 0.030
        self.quality_extra_ratio_ng = 0.090
        self.quality_min_missing_blob_area = 120
        self.quality_ng_missing_blob_area = 450
        self.quality_tolerance_px = 10
        self.quality_min_alignment_score = 0.80
        self.quality_min_trusted_alignment_score = 0.55
        self.quality_frame_stride = 5
        self.roi = self.load_default_roi()
        self.template_matrix = np.vstack([bit_string_to_vector(t.bits) for t in self.templates])
        self.template_on_counts = self.template_matrix.sum(axis=1)
        self.tolerant_template_matrix = np.vstack([
            dilate_vector(bit_string_to_vector(t.bits), self.grid_cols, self.grid_rows, self.tolerant_dilate)
            for t in self.templates
        ])
        self.tolerant_template_on_counts = self.tolerant_template_matrix.sum(axis=1)
        self.state = DetectionState()
        self.last_payload: dict[str, Any] = {}
        self.last_frame: np.ndarray | None = None
        self.last_roi: np.ndarray | None = None
        self.last_mask_preview: np.ndarray | None = None
        self.last_diff_preview: np.ndarray | None = None
        self.last_template_preview: np.ndarray | None = None
        self.display_snapshot: dict[str, np.ndarray | None] = {}
        self.display_snapshot_until = 0.0
        self.display_snapshots_by_key: dict[str, tuple[float, dict[str, np.ndarray | None]]] = {}
        self.last_quality_overlay: np.ndarray | None = None
        self.last_bits: np.ndarray | None = None
        self.global_warp: np.ndarray | None = None
        self.global_warp_score = 0.0
        self.global_warp_template_id = ""
        self.capture_fps = 0.0
        self.detect_fps = 0.0
        self._quality_matched_counter = 0
        self._last_quality_template_id = ""
        self._detect_frames_window = 0
        self._detect_window_started = time.perf_counter()
        self.refresh_cameras()

    def filter_unstable_keyframes(self, templates: list[Template], sequence: list[str]) -> tuple[list[Template], list[str], list[str]]:
        """过滤不稳定模板和过渡帧。
        
        两步过滤:
        1. 移除极弱帧 (active_cells<3 或 active_pixels<250)
        2. 合并序列中与相邻模板形状+颜色都高度相似的过渡帧
           注意: 形状相同但颜色不同的帧(如RED→GREEN→CYAN)必须保留
        """
        stable: list[Template] = []
        removed: list[str] = []
        for template in templates:
            active_cells = int(np.count_nonzero(bit_string_to_vector(template.bits)))
            if active_cells < 3 or float(template.active_pixels_median) < 250.0:
                removed.append(template.id)
                continue
            stable.append(template)
        stable_ids = {template.id for template in stable}
        # 第一步: 去除不稳定模板后压缩序列
        filtered_sequence: list[str] = []
        previous = ""
        for template_id in sequence:
            if template_id not in stable_ids:
                continue
            if template_id == previous:
                continue
            filtered_sequence.append(template_id)
            previous = template_id

        # 第二步: 合并过渡帧 — 形状AND颜色都与邻居高度相似才视为过渡
        # 注意：只对小模板(cells<200)做过渡帧检测，大帧是正式测试阶段不能过滤
        bits_by_id = {t.id: bit_string_to_vector(t.bits) for t in stable}
        cells_by_id = {t.id: int(np.count_nonzero(bit_string_to_vector(t.bits))) for t in stable}
        color_by_id = {t.id: str(getattr(t, "color_bits", "") or "") for t in stable}
        transition_ids: set[str] = set()
        if len(filtered_sequence) >= 3:
            for i in range(1, len(filtered_sequence) - 1):
                cur_id = filtered_sequence[i]
                # 大模板（全亮/半亮帧）是正式测试阶段，跳过
                if cells_by_id.get(cur_id, 0) >= 200:
                    continue
                prev_id = filtered_sequence[i - 1]
                next_id = filtered_sequence[i + 1]
                cur_bits = bits_by_id.get(cur_id)
                prev_bits = bits_by_id.get(prev_id)
                next_bits = bits_by_id.get(next_id)
                if cur_bits is None or prev_bits is None or next_bits is None:
                    continue
                sim_prev = jaccard_similarity(cur_bits, prev_bits)
                sim_next = jaccard_similarity(cur_bits, next_bits)
                # 只有形状相似时才进一步检查
                if sim_prev <= 0.80 and sim_next <= 0.80:
                    continue
                # 检查颜色: 如果颜色明显不同，不是过渡帧，应保留
                c_cur = color_by_id.get(cur_id, "")
                c_prev = color_by_id.get(prev_id, "")
                c_next = color_by_id.get(next_id, "")
                is_color_different = False
                if sim_prev > 0.80 and c_cur and c_prev:
                    if color_similarity(c_cur, c_prev) < 0.70:
                        is_color_different = True
                if sim_next > 0.80 and c_cur and c_next:
                    if color_similarity(c_cur, c_next) < 0.70:
                        is_color_different = True
                if is_color_different:
                    continue  # 颜色不同，保留该模板
                # 形状和颜色都相似 → 真正的过渡帧
                transition_ids.add(cur_id)

        if transition_ids:
            # 从序列中移除过渡帧并再次去重
            deduped: list[str] = []
            previous = ""
            for tid in filtered_sequence:
                if tid in transition_ids:
                    continue
                if tid == previous:
                    continue
                deduped.append(tid)
                previous = tid
            removed.extend(transition_ids)
            # 从模板列表中也移除（避免匹配时干扰）
            stable = [t for t in stable if t.id not in transition_ids]
            filtered_sequence = deduped

        # 第三步: 消除大帧振荡 — 如果同一个ID在短窗口内重复出现，只保留首次
        # 例如序列 T003→T004→T005→T004→T003 变为 T003→T004→T005
        seen_recent: list[str] = []
        no_oscillation: list[str] = []
        window = 6  # 看前6步以内是否出现过
        for tid in filtered_sequence:
            if tid in seen_recent[-window:]:
                continue  # 短窗口内重复，跳过
            no_oscillation.append(tid)
            seen_recent.append(tid)
        filtered_sequence = no_oscillation

        return stable, filtered_sequence, removed

    def load_default_roi(self) -> Roi:
        roi = self.model_summary.get("roi") or {"x": 770, "y": 350, "w": 1126, "h": 817}
        return Roi(int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"]))

    def detect_roi_from_video(self, video_path: str) -> Roi | None:
        """从视频前N帧自动检测显示面板ROI，替代固定坐标。
        
        原理：读取视频若干帧构建最大投影图，再用detect_display_roi检测亮区包络。
        如果前auto_roi_frames帧全黑，会继续读到有亮度为止（最多读总帧数的一半）。
        如果检测失败则回退到模型默认ROI。
        """
        if not self.auto_roi_enabled:
            return None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            max_read = max(self.auto_roi_frames, min(total_frames // 2, 600))
            max_proj: np.ndarray | None = None
            frames_read = 0
            has_bright = False
            while frames_read < max_read:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_proj is None:
                    max_proj = frame.copy()
                else:
                    np.maximum(max_proj, frame, out=max_proj)
                frames_read += 1
                # 检查是否已有足够亮度来做ROI检测
                if not has_bright and frames_read >= self.auto_roi_frames:
                    gray = cv2.cvtColor(max_proj, cv2.COLOR_BGR2GRAY)
                    if int(gray.max()) > self.auto_roi_projection_threshold:
                        has_bright = True
                        # 再多读一点确保完整
                        max_read = min(frames_read + 60, max_read)
            cap.release()
            if max_proj is None or frames_read < 5:
                return None
            roi = detect_display_roi(max_proj, self.auto_roi_projection_threshold, self.auto_roi_pad_ratio)
            # 合理性检查：ROI不能太小也不能占满整个画面
            h, w = max_proj.shape[:2]
            if roi.w < 100 or roi.h < 100:
                self.log(f"自动ROI太小({roi.w}x{roi.h}), 回退模型ROI")
                return None
            if roi.w > w * 0.95 and roi.h > h * 0.95:
                self.log(f"自动ROI覆盖全画面({roi.w}x{roi.h}), 回退模型ROI")
                return None
            self.log(f"自动ROI检测成功: x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h} (从前{frames_read}帧)")
            return roi
        except Exception as exc:
            self.log(f"自动ROI检测异常: {exc}, 回退模型ROI")
            return None

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-120:]
        print(line, flush=True)

    def camera_to_json(self, camera: CameraInfo) -> dict[str, Any]:
        return {
            "label": camera.label,
            "kind": camera.kind,
            "index": camera.index,
            "sdk_index": camera.sdk_index,
            "tlayer_type": camera.tlayer_type,
        }

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
        self.log(f"相机数量: {len(cameras)}")
        return [self.camera_to_json(c) for c in cameras]

    def select_camera(self, label: str = "") -> CameraInfo | None:
        if label:
            for camera in self.cameras:
                if camera.label == label:
                    return camera
        hikvision = [camera for camera in self.cameras if camera.kind == "hikvision"]
        return hikvision[0] if hikvision else (self.cameras[0] if self.cameras else None)

    def start(self, label: str = "") -> dict[str, Any]:
        self.stop()
        if not self.cameras:
            self.refresh_cameras()
        camera = self.select_camera(label)
        if camera is None:
            raise RuntimeError("没有可用相机")
        stream: OpenCVCameraStream | HikvisionCameraStream
        if camera.kind == "hikvision":
            stream = HikvisionCameraStream(camera, self.log)
        else:
            stream = OpenCVCameraStream(camera, self.log)
        stream.start()
        with self.lock:
            self.stream = stream
            self.video_source = ""
            self.sequence_only_video = True
            self.detecting = True
            self.frozen = False
            self.state = DetectionState(result="RUNNING")
            self.global_warp = None
            self.global_warp_score = 0.0
            self.global_warp_template_id = ""
        self.detector_stop.clear()
        self.detector_thread = threading.Thread(target=self.detection_loop, name="sevenseg-detector", daemon=True)
        self.detector_thread.start()
        self.log(f"开始采集检测: {camera.label}")
        return self.status()

    def stop(self) -> dict[str, Any]:
        self.video_stop.set()
        video_thread = self.video_thread
        if video_thread is not None and video_thread.is_alive():
            video_thread.join(timeout=1.5)
        self.video_thread = None
        self.video_source = ""
        self.sequence_only_video = False
        self.detector_stop.set()
        thread = self.detector_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        self.detector_thread = None
        stream = self.stream
        if stream is not None:
            stream.stop()
            with self.lock:
                self.stream = None
                self.detecting = False
                self.frozen = False
            self.log("停止采集检测")
        elif self.detecting:
            with self.lock:
                self.detecting = False
                self.frozen = False
            self.log("停止视频检测")
        return self.status()

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.state = DetectionState(result="RUNNING" if self.stream is not None else "WAIT")
            self.last_payload = {}
            self.last_bits = None
            self._quality_matched_counter = 0
            self._last_quality_template_id = ""
            self.global_warp = None
            self.global_warp_score = 0.0
            self.global_warp_template_id = ""
            self.frozen = False
        self.log("检测状态已重置")
        return self.status()

    def ack(self) -> dict[str, Any]:
        with self.lock:
            self.frozen = False
            if (self.stream is not None or self.video_thread is not None) and self.state.result in {"REVIEW", "NG"}:
                self.state.result = "RUNNING"
                self.last_payload = {**self.last_payload, "result": "RUNNING", "message": "已确认，继续检测"}
        self.log("异常已确认，继续检测")
        return self.status()

    def status(self) -> dict[str, Any]:
        stream = self.stream
        video_active = self.video_thread is not None and self.video_thread.is_alive()
        last = dict(self.last_payload)
        return {
            "ok": True,
            "previewing": stream is not None or video_active,
            "detecting": self.detecting,
            "frozen": self.frozen,
            "camera": stream.camera.label if stream else (f"视频: {Path(self.video_source).name}" if self.video_source else ""),
            "video_source": self.video_source,
            "video_active": video_active,
            "cameras": [self.camera_to_json(c) for c in self.cameras],
            "roi": {"x": self.roi.x, "y": self.roi.y, "w": self.roi.w, "h": self.roi.h},
            "sequence_count": len(self.sequence),
            "template_count": len(self.templates),
            "state": self.state.__dict__,
            "last": last,
            "result": last.get("result", self.state.result),
            "status": last.get("status", self.state.last_status),
            "message": last.get("message", ""),
            "template_id": last.get("template_id", ""),
            "similarity": last.get("similarity", 0.0),
            "coverage": last.get("coverage", 0.0),
            "ordered_hits": last.get("ordered_hits", self.state.ordered_hits),
            "unknown_segments": last.get("unknown_segments", self.state.unknown_segments),
            "quality": last.get("quality", {}),
            "quality_status": last.get("quality_status", "unchecked"),
            "report_dir": last.get("report_dir", ""),
            "capture_fps": round(self.capture_fps, 1),
            "detect_fps": round(self.detect_fps, 1),
            "logs": self.logs[-80:],
        }

    def latest_frame(self) -> np.ndarray | None:
        stream = self.stream
        if stream is None:
            return None
        return stream.get_latest_frame()

    def set_roi_from_query(self, query: dict[str, list[str]]) -> None:
        if not all(k in query for k in ("x", "y", "w", "h")):
            return
        try:
            self.roi = Roi(
                int(float(query["x"][0])),
                int(float(query["y"][0])),
                int(float(query["w"][0])),
                int(float(query["h"][0])),
            )
        except Exception:
            return

    def match_bits(self, bits: np.ndarray) -> tuple[Template | None, float, str]:
        tmpl, sim = find_best_template(bits, self.templates, self.template_matrix, self.template_on_counts)
        if tmpl and sim >= self.match_threshold:
            return tmpl, sim, "exact"
        tolerant = dilate_vector(bits, self.grid_cols, self.grid_rows, self.tolerant_dilate)
        tmpl2, sim2 = find_best_template(tolerant, self.templates, self.tolerant_template_matrix, self.tolerant_template_on_counts)
        if tmpl2 and sim2 >= self.tolerant_threshold:
            return tmpl2, sim2, "tolerant"
        if sim2 > sim:
            return tmpl2, sim2, "below_threshold"
        return tmpl, sim, "below_threshold"

    def maybe_establish_global_alignment(self, crop: np.ndarray, bits: np.ndarray, active_cells: int) -> None:
        if not self.global_alignment_enabled or self.global_warp is not None:
            return
        if active_cells < self.global_alignment_min_active_cells:
            return
        tmpl, sim = find_best_template(bits, self.templates, self.template_matrix, self.template_on_counts)
        if tmpl is None:
            return
        reference = self.load_template_image(tmpl.id, crop.shape[:2])
        if reference is None:
            return
        try:
            warp, score = self.estimate_quality_alignment(reference, crop)
        except Exception:
            return
        if score >= self.global_alignment_min_score:
            self.global_warp = warp
            self.global_warp_score = float(score)
            self.global_warp_template_id = tmpl.id
            self.log(f"全局内容对齐已建立: {tmpl.id} sim={sim:.2f} align={score:.2f}")

    def _orb_alignment(self, reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, float]:
        """用ORB特征点匹配估计仿射变换，作为ECC的后备方案。"""
        gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
        gray_mov = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY) if moving.ndim == 3 else moving
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray_ref, None)
        kp2, des2 = orb.detectAndCompute(gray_mov, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return np.eye(2, 3, dtype=np.float32), 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)
        # 取前30%或至少10个最佳匹配
        n_good = max(10, len(matches) // 3)
        good = matches[:n_good]
        if len(good) < 4:
            return np.eye(2, 3, dtype=np.float32), 0.0
        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_mov = np.float32([kp2[m.trainIdx].pt for m in good])
        # 估计Euclidean变换(仅旋转+平移)
        warp, inliers = cv2.estimateAffinePartial2D(pts_mov, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if warp is None:
            return np.eye(2, 3, dtype=np.float32), 0.0
        # 计算inlier比例作为score
        inlier_ratio = float(inliers.sum()) / len(inliers) if inliers is not None else 0.0
        # 检查变换合理性：平移不超过crop尺寸的10%，旋转不超过5度
        tx, ty = warp[0, 2], warp[1, 2]
        h, w = moving.shape[:2]
        if abs(tx) > w * 0.10 or abs(ty) > h * 0.10:
            return np.eye(2, 3, dtype=np.float32), 0.0
        # 取2x3的前2x2部分检查旋转角度
        cos_angle = (warp[0, 0] + warp[1, 1]) / 2.0
        if cos_angle < 0.996:  # >5度
            return np.eye(2, 3, dtype=np.float32), 0.0
        # 转为2x3格式（与ECC输出兼容）
        warp_2x3 = warp[:2, :].astype(np.float32)
        return warp_2x3, inlier_ratio

    def apply_global_alignment(self, crop: np.ndarray) -> np.ndarray:
        if self.global_warp is None:
            return crop
        try:
            return warp_to_reference(crop, self.global_warp)
        except Exception:
            return crop

    @staticmethod
    def color_compatible(expected: str, actual: str) -> bool:
        if expected == "0" or actual == "0":
            return False
        if expected == actual:
            return True
        cool = {"W", "C", "B"}
        warm = {"R", "Y"}
        return (expected in cool and actual in cool) or (expected in warm and actual in warm)

    def color_recall(self, actual_color_bits: str, template: Template, actual_bits: np.ndarray) -> float | None:
        expected_color_bits = str(getattr(template, "color_bits", "") or "")
        if not self.color_check_enabled or not expected_color_bits or not actual_color_bits:
            return None
        template_bits = str(getattr(template, "bits", "") or "")
        n = min(len(expected_color_bits), len(actual_color_bits), len(actual_bits), len(template_bits))
        total = 0
        matched = 0
        for i in range(n):
            if template_bits[i] != "1" or not bool(actual_bits[i]):
                continue
            expected = expected_color_bits[i]
            if expected == "0":
                continue
            actual = actual_color_bits[i]
            if actual == "0":
                continue
            total += 1
            if self.color_compatible(expected, actual):
                matched += 1
        if total <= 0:
            return None
        return matched / total

    def expected_match(self, bits: np.ndarray, expected_id: str, color_bits: str = "", active_pixels: int = 0) -> tuple[str, float, str]:
        if not expected_id:
            return "", 0.0, ""
        template = self.templates_by_id.get(expected_id)
        if template is None:
            return "", 0.0, ""

        # 亮度自适应检查 (Active pixels validation)
        # 对较小的模板进行亮度校验，防止因为亮度强弱混淆形状相同的帧 (例如 T017 vs T018)
        if active_pixels > 0 and template.active_pixels_median > 0:
            if template.active_pixels_median < 4000:
                ratio = active_pixels / template.active_pixels_median
                if ratio < 0.50 or ratio > 2.0:
                    return "", 0.0, f"expected_reject brightness_ratio={ratio:.2f}"

        expected = bit_string_to_vector(template.bits)
        actual_tolerant = dilate_vector(bits, self.grid_cols, self.grid_rows, self.tolerant_dilate)
        expected_tolerant = dilate_vector(expected, self.grid_cols, self.grid_rows, self.tolerant_dilate)
        expected_count = max(1, int(np.count_nonzero(expected)))
        actual_count = max(1, int(np.count_nonzero(bits)))
        hit_count = int(np.logical_and(actual_tolerant, expected).sum())
        extra_count = int(np.logical_and(bits, ~expected_tolerant).sum())
        recall = hit_count / expected_count
        extra_ratio = extra_count / expected_count
        precision = hit_count / actual_count
        score = min(recall, precision) if expected_count < self.expected_small_template_cells else jaccard_similarity(actual_tolerant, expected_tolerant)
        max_extra = self.expected_max_extra_ratio_small if expected_count < self.expected_small_template_cells else self.expected_max_extra_ratio_large
        if recall >= self.expected_min_recall and extra_ratio <= max_extra:
            color_score = self.color_recall(color_bits, template, bits)
            if color_score is not None and color_score < self.expected_min_color_recall:
                return "", float(min(score, color_score)), f"expected_reject recall={recall:.2f} extra={extra_ratio:.2f} color={color_score:.2f}"
            if color_score is not None:
                return expected_id, float(min(score, color_score)), f"expected_aligned recall={recall:.2f} extra={extra_ratio:.2f} color={color_score:.2f}"
            return expected_id, float(score), f"expected_aligned recall={recall:.2f} extra={extra_ratio:.2f}"
        # 位置无关匹配：对小模板，当固定位置匹配失败时，用滑动窗口形状匹配
        if expected_count < self.expected_small_template_cells and actual_count > 0:
            shape_score, shape_mode = self._position_invariant_match(expected, bits, expected_count, actual_count)
            if shape_score >= 0.60:
                return expected_id, float(shape_score), shape_mode
        return "", float(score), f"expected_reject recall={recall:.2f} extra={extra_ratio:.2f}"

    def _position_invariant_match(self, expected: np.ndarray, actual: np.ndarray,
                                   expected_count: int, actual_count: int) -> tuple[float, str]:
        """对小模板进行位置无关的形状匹配：提取模板和实际帧的亮点 bounding box，滑动比较"""
        eg = expected.reshape(self.grid_rows, self.grid_cols)
        ag = actual.reshape(self.grid_rows, self.grid_cols)
        # 提取模板亮点的 bounding box 区域
        e_rows, e_cols = np.where(eg)
        if len(e_rows) == 0:
            return 0.0, "shape_no_template"
        e_r0, e_r1 = int(e_rows.min()), int(e_rows.max()) + 1
        e_c0, e_c1 = int(e_cols.min()), int(e_cols.max()) + 1
        e_patch = eg[e_r0:e_r1, e_c0:e_c1]
        ph, pw = e_patch.shape
        # 提取实际帧亮点的 bounding box（带 padding）
        a_rows, a_cols = np.where(ag)
        if len(a_rows) == 0:
            return 0.0, "shape_no_actual"
        a_r0 = max(0, int(a_rows.min()) - ph)
        a_r1 = min(self.grid_rows, int(a_rows.max()) + 1 + ph)
        a_c0 = max(0, int(a_cols.min()) - pw)
        a_c1 = min(self.grid_cols, int(a_cols.max()) + 1 + pw)
        # 限制搜索范围：如果实际帧亮点数量和模板差太多，不太可能匹配
        if actual_count > expected_count * 5 or actual_count < expected_count * 0.3:
            return 0.0, "shape_count_mismatch"
        # 滑动窗口匹配
        best_score = 0.0
        best_dr, best_dc = 0, 0
        search_region = ag[a_r0:a_r1, a_c0:a_c1]
        sh, sw = search_region.shape
        if sh < ph or sw < pw:
            return 0.0, "shape_region_too_small"
        for dr in range(sh - ph + 1):
            for dc in range(sw - pw + 1):
                window = search_region[dr:dr + ph, dc:dc + pw]
                intersection = int(np.logical_and(e_patch, window).sum())
                union = int(np.logical_or(e_patch, window).sum())
                if union > 0:
                    iou = intersection / union
                    if iou > best_score:
                        best_score = iou
                        best_dr, best_dc = dr + a_r0, dc + a_c0
        if best_score >= 0.60:
            return best_score, f"shape_match iou={best_score:.2f} at=({best_dr},{best_dc})"
        return best_score, f"shape_reject iou={best_score:.2f}"

    def accept_template(self, template_id: str) -> str:
        state = self.state
        if not template_id or template_id == state.last_template:
            return "重复帧"
        expected = self.next_expected_template_id()
        if template_id != expected:
            state.last_template = template_id
            return f"非当前目标: 等待{expected} 检出{template_id}"
        state.ordered_hits += 1
        state.cursor += 1
        state.last_template = template_id
        return "关键模板出现，顺序正确"

    def accept_start_sync_template(self, template_id: str) -> str:
        state = self.state
        if state.cursor != 0 or not template_id:
            return self.accept_template(template_id)
        try:
            index = self.sequence.index(template_id, 0, min(len(self.sequence), self.start_sync_window))
        except ValueError:
            return self.accept_template(template_id)
        state.ordered_hits = index + 1
        state.cursor = index + 1
        state.last_template = template_id
        return f"起点同步到{template_id}"

    def detection_loop(self) -> None:
        self.log("实时检测线程已启动")
        while not self.detector_stop.is_set():
            if self.frozen:
                time.sleep(0.05)
                continue
            try:
                self.process_latest_frame()
            except Exception as exc:
                self.log(f"实时检测失败: {exc}")
                with self.lock:
                    self.last_payload = {"ok": False, "result": "ERROR", "message": str(exc)}
                    self.state.result = "ERROR"
                    self.frozen = True
                time.sleep(0.2)
            time.sleep(0.001)
        self.log("实时检测线程已停止")

    def detect_once(self, query: dict[str, list[str]]) -> dict[str, Any]:
        self.set_roi_from_query(query)
        return self.process_latest_frame()

    def process_latest_frame(self) -> dict[str, Any]:
        with self.detector_lock:
            return self._process_latest_frame()

    def _process_latest_frame(self) -> dict[str, Any]:
        frame = self.latest_frame()
        if frame is None:
            return {"ok": False, "message": "没有实时帧", **self.status()}
        return self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        roi = self.roi
        h, w = frame.shape[:2]
        roi = Roi(max(0, min(roi.x, w - 1)), max(0, min(roi.y, h - 1)), max(10, min(roi.w, w - roi.x)), max(10, min(roi.h, h - roi.y)))
        self.roi = roi
        crop = crop_roi(frame, roi)
        mask = make_mask(crop, self.active_threshold)
        bits = mask_to_vector(mask, self.grid_cols, self.grid_rows, self.cell_threshold)
        active_cells = int(np.count_nonzero(bits))
        self.maybe_establish_global_alignment(crop, bits, active_cells)
        aligned_crop = self.apply_global_alignment(crop)
        aligned_mask = make_mask(aligned_crop, self.active_threshold)
        aligned_bits = mask_to_vector(aligned_mask, self.grid_cols, self.grid_rows, self.cell_threshold)
        aligned_color_bits, aligned_color_counts = mask_to_color_bits(aligned_crop, aligned_mask, self.grid_cols, self.grid_rows, self.cell_threshold)
        aligned_active_cells = int(np.count_nonzero(aligned_bits))
        aligned_active_pixels = int(np.count_nonzero(aligned_mask))
        frame_preview = self.draw_roi(frame.copy(), roi)
        roi_preview = aligned_crop.copy()
        mask_preview = self.actual_mask_overlay(aligned_crop, aligned_mask, status="mask")

        state = self.state
        state.frames += 1
        expected_before_id = self.next_expected_template_id()
        expected_before_index = state.cursor + 1 if expected_before_id else 0
        template_id = ""
        similarity = 0.0
        mode = ""
        message = "黑帧/弱亮"
        status = "dark"
        best_template_id = ""
        best_similarity = 0.0
        best_mode = ""
        if aligned_active_cells >= self.min_active_cells:
            best_tmpl, best_similarity, best_mode = self.match_bits(aligned_bits)
            best_template_id = best_tmpl.id if best_tmpl is not None else ""
            if self.sequence_only_video:
                template_id, similarity, mode = self.expected_match(aligned_bits, expected_before_id, aligned_color_bits, aligned_active_pixels)
                if not template_id and state.cursor == 0 and self.sequence:
                    best_sync: tuple[str, float, str] = ("", 0.0, "")
                    for candidate_id in self.sequence[: self.start_sync_window]:
                        cand_id, cand_sim, cand_mode = self.expected_match(aligned_bits, candidate_id, aligned_color_bits, aligned_active_pixels)
                        if cand_id and cand_sim > best_sync[1]:
                            best_sync = (cand_id, cand_sim, cand_mode)
                    if best_sync[0]:
                        template_id, similarity, mode = best_sync
            else:
                template_id, similarity, mode = best_template_id, best_similarity, best_mode
            if template_id and not mode.endswith("below_threshold"):
                status = "matched"
                message = self.accept_start_sync_template(template_id) if self.sequence_only_video else self.accept_template(template_id)
            else:
                status = "unknown"
                if state.last_status != "unknown":
                    state.unknown_segments += 1
                state.last_template = ""
                if self.sequence_only_video and expected_before_id:
                    message = f"等待{expected_before_id}: 当前最佳{best_template_id or '-'} sim={best_similarity:.2f}"
                else:
                    message = "亮区未匹配模板"
        else:
            state.last_template = ""
        state.last_status = status
        self.last_quality_overlay = None
        quality = self.empty_quality("unchecked")
        should_check_quality = status == "matched" and template_id and message != "重复帧"
        if should_check_quality:
            self._quality_matched_counter += 1
            quality = self.inspect_lit_quality(aligned_crop, aligned_mask, template_id)
            self._last_quality_template_id = template_id
        coverage = state.ordered_hits / max(len(self.sequence), 1)
        sequence_only_video = self.sequence_only_video
        if sequence_only_video:
            if status == "unknown":
                message = "非关键帧，等待下一个关键模板"
            elif not self.video_source and coverage >= self.min_ordered_coverage and status == "matched":
                message = "关键帧时序已覆盖，继续播放复核"
            if self.video_source:
                state.result = "RUNNING"
                self.frozen = False
            elif quality["status"] == "ng":
                state.result = "NG"
                self.frozen = False
                message = quality["message"]
            elif quality["status"] == "review":
                state.result = "REVIEW"
                self.frozen = False
                message = quality["message"]
            elif coverage >= self.min_ordered_coverage:
                state.result = "PASS"
                self.frozen = False
            else:
                state.result = "RUNNING"
                self.frozen = False
        elif quality["status"] == "ng":
            state.result = "NG"
            self.frozen = True
            message = quality["message"]
        elif quality["status"] == "review":
            state.result = "REVIEW"
            self.frozen = True
            message = quality["message"]
        elif coverage >= self.min_ordered_coverage:
            state.result = "PASS"
        elif status == "unknown" and state.unknown_segments > self.max_unknown_segments:
            state.result = "REVIEW"
            self.frozen = True
        else:
            state.result = "RUNNING"
        diff_preview = self.last_quality_overlay if self.last_quality_overlay is not None else self.actual_mask_overlay(aligned_crop, aligned_mask, status=status)
        expected_after_id = self.next_expected_template_id()
        display_template_id = template_id if status == "matched" and template_id else expected_before_id
        template_preview = self.render_template_image(display_template_id) if display_template_id else None
        self.update_detection_fps()

        payload = {
            "ok": True,
            "status": status,
            "template_id": template_id,
            "display_template_id": display_template_id,
            "expected_template_id": expected_before_id,
            "expected_sequence_index": expected_before_index,
            "next_expected_template_id": expected_after_id,
            "similarity": similarity,
            "match_mode": mode,
            "message": message,
            "active_cells": active_cells,
            "aligned_active_cells": aligned_active_cells,
            "color_counts": aligned_color_counts,
            "best_template_id": best_template_id,
            "best_similarity": best_similarity,
            "best_match_mode": best_mode,
            "global_alignment": {
                "enabled": self.global_warp is not None,
                "score": round(self.global_warp_score, 4),
                "template_id": self.global_warp_template_id,
            },
            "ordered_hits": state.ordered_hits,
            "sequence_count": len(self.sequence),
            "coverage": coverage,
            "unknown_segments": state.unknown_segments,
            "quality_status": quality["status"],
            "quality": quality,
            "result": state.result,
            "frozen": self.frozen,
            "cursor": state.cursor,
            "capture_fps": round(self.capture_fps, 1),
            "detect_fps": round(self.detect_fps, 1),
            "roi": {"x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h},
            "logs": self.logs[-60:],
        }
        with self.lock:
            self.last_frame = frame_preview
            self.last_roi = roi_preview
            self.last_mask_preview = mask_preview
            self.last_diff_preview = diff_preview
            self.last_template_preview = template_preview
            self.last_bits = bits.copy()
            self.last_payload = payload
        return payload

    def update_detection_fps(self) -> None:
        self._detect_frames_window += 1
        now = time.perf_counter()
        elapsed = now - self._detect_window_started
        if elapsed >= 1.0:
            self.detect_fps = self._detect_frames_window / elapsed
            self.capture_fps = self.detect_fps
            self._detect_frames_window = 0
            self._detect_window_started = now

    def draw_roi(self, frame: np.ndarray, roi: Roi) -> np.ndarray:
        cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 0), 4)
        cv2.putText(frame, "ROI", (roi.x, max(24, roi.y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def mask_preview(self, bits: np.ndarray) -> np.ndarray:
        img = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.uint8)
        img[bits.reshape(self.grid_rows, self.grid_cols)] = (60, 220, 110)
        return cv2.resize(img, (720, 480), interpolation=cv2.INTER_NEAREST)

    def actual_mask_overlay(self, crop: np.ndarray, mask: np.ndarray, status: str = "mask") -> np.ndarray:
        base = cv2.resize(crop, (720, 480), interpolation=cv2.INTER_AREA)
        h, w = crop.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        color = (70, 230, 90)
        if status == "unknown":
            color = (255, 180, 70) if self.sequence_only_video else (0, 215, 255)
        elif status == "dark":
            color = (130, 140, 150)
        overlay[mask.astype(bool)] = color
        overlay = cv2.resize(overlay, (720, 480), interpolation=cv2.INTER_NEAREST)
        out = cv2.addWeighted(base, 0.55, overlay, 0.85, 0)
        label = "ACTUAL MASK"
        if status == "matched":
            label = "ACTUAL MASK MATCHED"
        elif status == "unknown":
            label = "WAITING NEXT KEYFRAME" if self.sequence_only_video else "ACTUAL MASK UNMATCHED"
        elif status == "dark":
            label = "ACTUAL MASK DARK"
        cv2.putText(out, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        return out

    @staticmethod
    def empty_quality(status: str) -> dict[str, Any]:
        return {
            "status": status,
            "message": "",
            "missing_ratio": 0.0,
            "extra_ratio": 0.0,
            "max_missing_blob_area": 0,
            "max_extra_blob_area": 0,
            "missing_blob_count": 0,
            "extra_blob_count": 0,
        }

    def load_template_mask(self, template_id: str, shape: tuple[int, int]) -> np.ndarray | None:
        cache_key = (template_id, shape)
        if cache_key in self.template_mask_cache:
            return self.template_mask_cache[cache_key]
        template = self.templates_by_id.get(template_id)
        if template is None or not template.mask_image:
            return None
        target = (self.model_path.parent / template.mask_image).resolve()
        model_root = self.model_path.parent.resolve()
        if model_root not in target.parents and target != model_root:
            return None
        image = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        resized = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = resized > 20
        self.template_mask_cache[cache_key] = mask
        return mask

    def load_template_image(self, template_id: str, shape: tuple[int, int]) -> np.ndarray | None:
        cache_key = (template_id, shape)
        if cache_key in self.template_image_cache:
            return self.template_image_cache[cache_key]
        template = self.templates_by_id.get(template_id)
        if template is None or not template.image:
            return None
        target = (self.model_path.parent / template.image).resolve()
        model_root = self.model_path.parent.resolve()
        if model_root not in target.parents and target != model_root:
            return None
        image = cv2.imread(str(target))
        if image is None:
            return None
        resized = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        self.template_image_cache[cache_key] = resized
        return resized

    @staticmethod
    def component_stats(mask: np.ndarray, min_area: int) -> tuple[int, int]:
        if not np.any(mask):
            return 0, 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        kept = 0
        max_area = 0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                kept += 1
                max_area = max(max_area, area)
        return kept, max_area

    def inspect_lit_quality(self, crop: np.ndarray, actual_mask: np.ndarray, template_id: str) -> dict[str, Any]:
        if not self.quality_enabled:
            return self.empty_quality("off")
        expected_mask = self.load_template_mask(template_id, actual_mask.shape)
        reference_image = self.load_template_image(template_id, actual_mask.shape)
        if expected_mask is None or not np.any(expected_mask):
            return self.empty_quality("unchecked")
        if reference_image is not None:
            warp, alignment_score = self.estimate_quality_alignment(reference_image, crop)
            if alignment_score < self.quality_min_trusted_alignment_score:
                quality = self.empty_quality("unchecked")
                quality["message"] = f"质量未检: 配准分数低 {alignment_score:.2f}"
                self.last_quality_overlay = self.quality_overlay(crop, expected_mask, actual_mask, np.zeros_like(actual_mask), np.zeros_like(actual_mask), "unchecked")
                return quality
            if alignment_score >= self.quality_min_alignment_score:
                crop = warp_to_reference(crop, warp)
                actual_mask = make_mask(crop, self.active_threshold)
        else:
            alignment_score = 0.0

        kernel = np.ones((self.quality_tolerance_px * 2 + 1, self.quality_tolerance_px * 2 + 1), np.uint8)
        actual_u8 = actual_mask.astype(np.uint8)
        expected_u8 = expected_mask.astype(np.uint8)
        actual_tolerant = cv2.dilate(actual_u8, kernel) > 0
        expected_tolerant = cv2.dilate(expected_u8, kernel) > 0

        missing = expected_mask & ~actual_tolerant
        extra = actual_mask & ~expected_tolerant
        expected_area = max(1, int(np.count_nonzero(expected_mask)))
        actual_area = max(1, int(np.count_nonzero(actual_mask)))
        missing_count, max_missing = self.component_stats(missing, self.quality_min_missing_blob_area)
        extra_count, max_extra = self.component_stats(extra, self.quality_min_missing_blob_area)
        missing_ratio = float(np.count_nonzero(missing)) / expected_area
        extra_ratio = float(np.count_nonzero(extra)) / actual_area
        significant_missing = missing_count > 0 or max_missing >= self.quality_min_missing_blob_area
        significant_extra = extra_count > 0 or max_extra >= self.quality_min_missing_blob_area

        status = "ok"
        message = "亮段质量正常"
        if (significant_missing and missing_ratio >= self.quality_missing_ratio_ng) or max_missing >= self.quality_ng_missing_blob_area:
            status = "ng"
            message = f"亮段缺失 NG: 缺失{missing_ratio:.1%} 最大缺口{max_missing}px"
        elif (significant_extra and extra_ratio >= self.quality_extra_ratio_ng) or max_extra >= self.quality_ng_missing_blob_area * 2:
            status = "ng"
            message = f"多亮/反光 NG: 多亮{extra_ratio:.1%} 最大块{max_extra}px"
        elif (
            significant_missing
            or significant_extra
            or max_missing >= self.quality_min_missing_blob_area
        ):
            status = "review"
            message = f"亮段质量复核: 缺失{missing_ratio:.1%} 最大缺口{max_missing}px"

        self.last_quality_overlay = self.quality_overlay(crop, expected_mask, actual_mask, missing, extra, status)
        return {
            "status": status,
            "message": message,
            "missing_ratio": round(missing_ratio, 5),
            "extra_ratio": round(extra_ratio, 5),
            "max_missing_blob_area": max_missing,
            "max_extra_blob_area": max_extra,
            "missing_blob_count": missing_count,
            "extra_blob_count": extra_count,
            "alignment_score": round(float(alignment_score), 4),
        }

    def estimate_quality_alignment(self, reference_image: np.ndarray, crop: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = crop.shape[:2]
        scale = min(1.0, 320.0 / max(h, w))
        if scale < 1.0:
            size = (max(16, int(round(w * scale))), max(16, int(round(h * scale))))
            ref_small = cv2.resize(reference_image, size, interpolation=cv2.INTER_AREA)
            crop_small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            warp, _, score = estimate_image_alignment(ref_small, crop_small, self.active_threshold, "euclidean")
            warp = warp.copy()
            warp[0, 2] /= scale
            warp[1, 2] /= scale
            return warp, score
        warp, _, score = estimate_image_alignment(reference_image, crop, self.active_threshold, "euclidean")
        return warp, score

    def quality_overlay(
        self,
        crop: np.ndarray,
        expected_mask: np.ndarray,
        actual_mask: np.ndarray,
        missing: np.ndarray,
        extra: np.ndarray,
        status: str,
    ) -> np.ndarray:
        base = cv2.resize(crop, (720, 480), interpolation=cv2.INTER_AREA)
        h, w = crop.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[actual_mask & expected_mask] = (50, 210, 90)
        overlay[extra] = (30, 70, 230)
        overlay[missing] = (0, 215, 255)
        overlay = cv2.resize(overlay, (720, 480), interpolation=cv2.INTER_NEAREST)
        out = cv2.addWeighted(base, 0.58, overlay, 0.78, 0)
        color = (0, 220, 0) if status == "ok" else ((0, 215, 255) if status == "review" else (0, 0, 255))
        cv2.putText(out, f"QUALITY {status.upper()}", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(out, "green=match red=extra yellow=missing", (16, 466), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        return out

    def diff_preview(self, bits: np.ndarray, template_id: str) -> np.ndarray:
        template = self.templates_by_id.get(template_id)
        if template is None:
            cursor = max(0, min(self.state.cursor, len(self.sequence) - 1))
            template = self.templates_by_id.get(self.sequence[cursor]) if self.sequence else None
        expected = bit_string_to_vector(template.bits) if template is not None else np.zeros_like(bits)
        actual_on = bits.reshape(self.grid_rows, self.grid_cols).astype(bool)
        expected_on = expected.reshape(self.grid_rows, self.grid_cols).astype(bool)
        img = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.uint8)
        img[:, :] = (28, 33, 42)
        img[actual_on & expected_on] = (60, 220, 110)
        img[actual_on & ~expected_on] = (40, 70, 230)
        img[~actual_on & expected_on] = (0, 210, 255)
        return cv2.resize(img, (720, 480), interpolation=cv2.INTER_NEAREST)

    def start_video_detection(self, video_path: str, realtime: bool = True) -> dict[str, Any]:
        raw_path = video_path.strip().strip('"')
        if not raw_path:
            raise ValueError("请先选择视频文件")
        video = Path(raw_path)
        if not video.exists():
            raise FileNotFoundError(f"视频不存在: {video}")
        if not video.is_file():
            raise ValueError(f"视频路径不是文件: {video}")

        self.stop()
        # 自动检测该视频的ROI，解决不同视频面板位置不同的问题
        auto_roi = self.detect_roi_from_video(str(video))
        with self.lock:
            if auto_roi is not None:
                self.roi = auto_roi
            else:
                self.roi = self.load_default_roi()
            self.video_source = str(video)
            self.sequence_only_video = True
            self.detecting = True
            self.frozen = False
            self.state = DetectionState(result="RUNNING")
            self.last_payload = {
                "ok": True,
                "result": "RUNNING",
                "message": "正在播放视频并实时检测",
                "coverage": 0.0,
                "ordered_hits": 0,
                "sequence_count": len(self.sequence),
                "quality_status": "unchecked",
                "quality": {},
            }
            self.last_bits = None
            self.capture_fps = 0.0
            self.detect_fps = 0.0
            self._quality_matched_counter = 0
            self._last_quality_template_id = ""
            self.global_warp = None
            self.global_warp_score = 0.0
            self.global_warp_template_id = ""
        self.video_stop.clear()
        self.video_thread = threading.Thread(
            target=self.video_detection_loop,
            args=(str(video), realtime),
            name="sevenseg-video-detector",
            daemon=True,
        )
        self.video_thread.start()
        self.log(f"开始视频实时检测: {video}")
        return self.status()

    def video_detection_loop(self, video_path: str, realtime: bool = True) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with self.lock:
                self.state.result = "ERROR"
                self.last_payload = {**self.last_payload, "result": "ERROR", "message": f"无法打开视频: {video_path}"}
                self.detecting = False
            self.log(f"无法打开视频: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 1 or fps > 240:
            fps = 25.0
        frame_interval = 1.0 / fps
        next_frame_due = time.perf_counter()
        frame_count = 0
        processed = 0
        matched_frames = 0
        unknown_frames = 0
        quality_counts = {"ok": 0, "review": 0, "ng": 0, "unchecked": 0, "off": 0}
        first_quality_issue: dict[str, Any] | None = None
        frame_records: list[dict[str, Any]] = []
        sequence_records: list[dict[str, Any]] = []
        template_stats: list[dict[str, Any]] = [
            {
                "template_id": tid,
                "sequence_index": index + 1,
                "hit": False,
                "first_hit_frame": None,
                "first_hit_time_sec": None,
                "max_expected_similarity": 0.0,
                "max_best_similarity": 0.0,
                "best_seen_as": "",
                "reject_count": 0,
                "last_reject_reason": "",
            }
            for index, tid in enumerate(self.sequence)
        ]
        stats_by_index = {index + 1: stat for index, stat in enumerate(template_stats)}
        started = time.time()
        capture_window_started = time.perf_counter()
        capture_window_frames = 0

        try:
            while not self.video_stop.is_set():
                if self.frozen:
                    time.sleep(0.05)
                    continue
                loop_started = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                capture_window_frames += 1
                if loop_started - capture_window_started >= 1.0:
                    self.capture_fps = capture_window_frames / max(loop_started - capture_window_started, 0.001)
                    capture_window_started = loop_started
                    capture_window_frames = 0

                with self.detector_lock:
                    payload = self._process_frame(frame)
                processed += 1
                expected_id = str(payload.get("expected_template_id", ""))
                detected_id = str(payload.get("template_id", ""))
                record = {
                    "frame": frame_count,
                    "time_sec": round(frame_count / fps, 3),
                    "cursor": int(payload.get("cursor", 0)),
                    "expected_sequence_index": int(payload.get("expected_sequence_index", 0)),
                    "expected_template_id": expected_id,
                    "detected_template_id": detected_id,
                    "display_template_id": payload.get("display_template_id", ""),
                    "next_expected_template_id": payload.get("next_expected_template_id", ""),
                    "status": payload.get("status", ""),
                    "similarity": round(float(payload.get("similarity", 0.0)), 5),
                    "match_mode": payload.get("match_mode", ""),
                    "best_template_id": payload.get("best_template_id", ""),
                    "best_similarity": round(float(payload.get("best_similarity", 0.0)), 5),
                    "best_match_mode": payload.get("best_match_mode", ""),
                    "active_cells": int(payload.get("active_cells", 0)),
                    "aligned_active_cells": int(payload.get("aligned_active_cells", 0)),
                    "ordered_hits": int(payload.get("ordered_hits", 0)),
                    "coverage": round(float(payload.get("coverage", 0.0)), 5),
                    "quality_status": payload.get("quality_status", "unchecked"),
                    "quality_message": payload.get("quality", {}).get("message", ""),
                    "quality_missing_ratio": payload.get("quality", {}).get("missing_ratio", 0.0),
                    "quality_extra_ratio": payload.get("quality", {}).get("extra_ratio", 0.0),
                    "quality_max_missing_blob_area": payload.get("quality", {}).get("max_missing_blob_area", 0),
                    "quality_max_extra_blob_area": payload.get("quality", {}).get("max_extra_blob_area", 0),
                    "quality_alignment_score": payload.get("quality", {}).get("alignment_score", 0.0),
                    "alignment_enabled": bool(payload.get("global_alignment", {}).get("enabled", False)),
                    "alignment_template_id": payload.get("global_alignment", {}).get("template_id", ""),
                    "alignment_score": payload.get("global_alignment", {}).get("score", 0.0),
                    "message": payload.get("message", ""),
                }
                frame_records.append(record)
                sequence_index = int(payload.get("expected_sequence_index", int(payload.get("cursor", 0)) + 1))
                stat = stats_by_index.get(sequence_index)
                if stat is not None and expected_id == stat["template_id"]:
                    stat["max_expected_similarity"] = max(stat["max_expected_similarity"], record["similarity"])
                    stat["max_best_similarity"] = max(stat["max_best_similarity"], record["best_similarity"])
                    if record["best_template_id"]:
                        stat["best_seen_as"] = record["best_template_id"]
                    if detected_id == expected_id:
                        if not stat["hit"]:
                            stat["first_hit_frame"] = frame_count
                            stat["first_hit_time_sec"] = round(frame_count / fps, 3)
                            sequence_records.append({
                                "kind": "hit",
                                "sequence_index": sequence_index,
                                "template_id": expected_id,
                                "frame": frame_count,
                                "time_sec": round(frame_count / fps, 3),
                                "status": record["status"],
                                "similarity": record["similarity"],
                                "best_template_id": record["best_template_id"],
                                "best_similarity": record["best_similarity"],
                                "message": record["message"],
                                "roi": None if self.last_roi is None else self.last_roi.copy(),
                                "mask": None if self.last_mask_preview is None else self.last_mask_preview.copy(),
                                "template": None if self.last_template_preview is None else self.last_template_preview.copy(),
                            })
                        stat["hit"] = True
                    elif record["status"] in {"unknown", "dark"}:
                        stat["reject_count"] += 1
                        stat["last_reject_reason"] = str(record["match_mode"] or record["message"])
                        best_reject = stat.get("_best_reject_similarity", -1.0)
                        if record["similarity"] >= best_reject:
                            stat["_best_reject_similarity"] = record["similarity"]
                            stat["_best_reject_visual"] = {
                                "kind": "best_reject",
                                "sequence_index": sequence_index,
                                "template_id": expected_id,
                                "frame": frame_count,
                                "time_sec": round(frame_count / fps, 3),
                                "status": record["status"],
                                "similarity": record["similarity"],
                                "best_template_id": record["best_template_id"],
                                "best_similarity": record["best_similarity"],
                                "message": record["message"],
                                "roi": None if self.last_roi is None else self.last_roi.copy(),
                                "mask": None if self.last_mask_preview is None else self.last_mask_preview.copy(),
                                "template": None if self.last_template_preview is None else self.last_template_preview.copy(),
                            }
                if payload.get("status") == "matched":
                    matched_frames += 1
                elif payload.get("status") == "unknown":
                    unknown_frames += 1
                quality_status = str(payload.get("quality_status", "unchecked"))
                quality_counts[quality_status] = quality_counts.get(quality_status, 0) + 1
                if quality_status in {"review", "ng"} and first_quality_issue is None:
                    first_quality_issue = {
                        "frame": frame_count,
                        "quality_status": quality_status,
                        "template_id": payload.get("template_id", ""),
                        "quality": payload.get("quality", {}),
                    }

                if realtime:
                    next_frame_due += frame_interval
                    now = time.perf_counter()
                    if now < next_frame_due:
                        time.sleep(next_frame_due - now)
                    elif now - next_frame_due > frame_interval:
                        skipped = 0
                        while now - next_frame_due > frame_interval and skipped < 3:
                            ok_skip, _ = cap.read()
                            if not ok_skip:
                                break
                            frame_count += 1
                            skipped += 1
                            next_frame_due += frame_interval

            elapsed_total = time.time() - started
            coverage = self.state.ordered_hits / max(len(self.sequence), 1)
            has_quality_issue = quality_counts.get("review", 0) > 0 or quality_counts.get("ng", 0) > 0
            quality_status = "NG" if quality_counts.get("ng", 0) > 0 else ("REVIEW" if quality_counts.get("review", 0) > 0 else "PASS")
            sequence_pass = self.state.ordered_hits >= len(self.sequence) * self.min_ordered_coverage
            overall_status = "PASS" if sequence_pass and not has_quality_issue else "NG"
            sequence_status = "PASS" if sequence_pass else "NG"
            summary = {
                "source": video_path,
                "overall_status": overall_status,
                "sequence_status": sequence_status,
                "quality_status": quality_status,
                "frame_count_read": frame_count,
                "frames_processed": processed,
                "matched_frames": matched_frames,
                "unknown_frames": unknown_frames,
                "quality_counts": quality_counts,
                "first_quality_issue": first_quality_issue,
                "sequence_check": {
                    "ordered_hits": self.state.ordered_hits,
                    "sequence_count": len(self.sequence),
                    "ordered_coverage": coverage,
                    "unknown_segments": self.state.unknown_segments,
                },
                "elapsed_sec": elapsed_total,
            }
            run_name = f"live_video_{Path(video_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_dir = OUTPUT_ROOT / "sevenseg_station" / run_name
            out_dir.mkdir(parents=True, exist_ok=True)
            visual_dir = out_dir / "template_debug_images"
            visual_dir.mkdir(parents=True, exist_ok=True)
            for stat in template_stats:
                reject_visual = stat.pop("_best_reject_visual", None)
                stat.pop("_best_reject_similarity", None)
                if not stat["hit"]:
                    if reject_visual is not None:
                        sequence_records.append(reject_visual)
                    else:
                        sequence_records.append({
                            "kind": "missing",
                            "sequence_index": stat["sequence_index"],
                            "template_id": stat["template_id"],
                            "frame": 0,
                            "time_sec": None,
                            "status": "missing",
                            "similarity": stat["max_expected_similarity"],
                            "best_template_id": stat["best_seen_as"],
                            "best_similarity": stat["max_best_similarity"],
                            "message": stat["last_reject_reason"],
                            "roi": None,
                            "mask": None,
                            "template": self.render_template_image(str(stat["template_id"])),
                        })
            visual_manifest: list[dict[str, Any]] = []
            for record in sorted(sequence_records, key=lambda item: int(item.get("sequence_index", 0))):
                tid = str(record["template_id"])
                seq_index = int(record.get("sequence_index", 0))
                frame_no = int(record["frame"])
                kind = str(record["kind"])
                prefix = f"{seq_index:03d}_{tid}_{kind}_f{frame_no:04d}"
                item = {k: v for k, v in record.items() if k not in {"roi", "mask", "template"}}
                roi_img = record.get("roi")
                mask_img = record.get("mask")
                template_img = record.get("template")
                panels = [img for img in (template_img, roi_img, mask_img) if img is not None]
                if panels:
                    height = 260
                    normalized = []
                    for img in panels:
                        h, w = img.shape[:2]
                        width = max(1, int(round(w * height / max(h, 1))))
                        normalized.append(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))
                    composite = cv2.hconcat(normalized)
                    path = visual_dir / f"{prefix}.jpg"
                    cv2.imwrite(str(path), composite)
                    item["image"] = path.relative_to(out_dir).as_posix()
                visual_manifest.append(item)
            (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "frame_matches.json").write_text(json.dumps(frame_records, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "template_summary.json").write_text(json.dumps(template_stats, ensure_ascii=False, indent=2), encoding="utf-8")
            (out_dir / "visual_manifest.json").write_text(json.dumps(visual_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if frame_records:
                with (out_dir / "frame_matches.csv").open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=list(frame_records[0].keys()))
                    writer.writeheader()
                    writer.writerows(frame_records)
            if template_stats:
                rows = template_stats
                with (out_dir / "template_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            with self.lock:
                self.state.result = overall_status
                self.frozen = overall_status != "PASS"
                self.detecting = False
                self.last_payload = {
                    **self.last_payload,
                    "result": overall_status,
                    "message": f"视频检测完成: {overall_status} 时序{sequence_status} 质量{quality_status} 覆盖率 {coverage:.1%}",
                    "coverage": coverage,
                    "ordered_hits": self.state.ordered_hits,
                    "sequence_count": len(self.sequence),
                    "unknown_segments": self.state.unknown_segments,
                    "summary": summary,
                    "report_dir": str(out_dir),
                }
            self.log(f"视频实时检测完成: {overall_status} coverage={coverage:.1%} {elapsed_total:.1f}s")
        except Exception as exc:
            with self.lock:
                self.state.result = "ERROR"
                self.detecting = False
                self.frozen = True
                self.last_payload = {**self.last_payload, "result": "ERROR", "message": str(exc)}
            self.log(f"视频实时检测失败: {exc}")
        finally:
            cap.release()

    def run_video_detection(self, video_path: str, quick: bool = False) -> dict[str, Any]:
        raw_path = video_path.strip().strip('"')
        if not raw_path:
            raise ValueError("请先选择视频文件")
        video = Path(raw_path)
        if not video.exists():
            raise FileNotFoundError(f"视频不存在: {video}")
        if not video.is_file():
            raise ValueError(f"视频路径不是文件: {video}")
        # 自动检测该视频的ROI
        auto_roi = self.detect_roi_from_video(str(video))
        if auto_roi is not None:
            self.roi = auto_roi
        else:
            self.roi = self.load_default_roi()
        run_name = f"live_video_{video.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_dir = OUTPUT_ROOT / "sevenseg_station" / run_name
        self.log(f"开始导入视频检测: {video}")
        started = time.time()
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video}")
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_count = 0
        processed = 0
        matched_frames = 0
        unknown_frames = 0
        quality_counts = {"ok": 0, "review": 0, "ng": 0, "unchecked": 0, "off": 0}
        first_quality_issue: dict[str, Any] | None = None
        last_payload: dict[str, Any] = {}
        with self.detector_lock:
            old_state = self.state
            old_frozen = self.frozen
            old_quality_counter = self._quality_matched_counter
            old_quality_template = self._last_quality_template_id
            old_sequence_only = self.sequence_only_video
            old_video_source = self.video_source
            self.state = DetectionState(result="RUNNING")
            self.frozen = False
            self.sequence_only_video = True
            self.video_source = str(video)
            self._quality_matched_counter = 0
            self._last_quality_template_id = ""
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1
                if quick and frame_count % 2 == 0:
                    continue
                last_payload = self._process_frame(frame)
                processed += 1
                if last_payload.get("status") == "matched":
                    matched_frames += 1
                elif last_payload.get("status") == "unknown":
                    unknown_frames += 1
                quality_status = str(last_payload.get("quality_status", "unchecked"))
                quality_counts[quality_status] = quality_counts.get(quality_status, 0) + 1
                if quality_status in {"review", "ng"} and first_quality_issue is None:
                    first_quality_issue = {
                        "frame": frame_count,
                        "quality_status": quality_status,
                        "template_id": last_payload.get("template_id", ""),
                        "quality": last_payload.get("quality", {}),
                    }
            summary_payload = dict(last_payload)
            final_state = self.state
            final_frozen = self.frozen
            self.state = old_state
            self.frozen = old_frozen
            self.sequence_only_video = old_sequence_only
            self.video_source = old_video_source
            self._quality_matched_counter = old_quality_counter
            self._last_quality_template_id = old_quality_template
        cap.release()
        elapsed = time.time() - started
        coverage = final_state.ordered_hits / max(len(self.sequence), 1)
        has_quality_issue = quality_counts.get("review", 0) > 0 or quality_counts.get("ng", 0) > 0
        quality_status = "NG" if quality_counts.get("ng", 0) > 0 else ("REVIEW" if quality_counts.get("review", 0) > 0 else "PASS")
        sequence_pass = final_state.ordered_hits >= len(self.sequence) * self.min_ordered_coverage
        overall_status = "PASS" if sequence_pass and not has_quality_issue else "NG"
        sequence_status = "PASS" if sequence_pass else "NG"
        summary = {
            "source": str(video),
            "overall_status": overall_status,
            "sequence_status": sequence_status,
            "quality_status": quality_status,
            "frame_count_read": frame_count,
            "frames_processed": processed,
            "matched_frames": matched_frames,
            "unknown_frames": unknown_frames,
            "quality_counts": quality_counts,
            "first_quality_issue": first_quality_issue,
            "sequence_check": {
                "ordered_hits": final_state.ordered_hits,
                "sequence_count": len(self.sequence),
                "ordered_coverage": coverage,
                "unknown_segments": final_state.unknown_segments,
            },
            "elapsed_sec": elapsed,
            "last": summary_payload,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.last_frame is not None:
            cv2.imwrite(str(out_dir / "last_frame.jpg"), self.last_frame)
        if self.last_mask_preview is not None:
            cv2.imwrite(str(out_dir / "last_mask.jpg"), self.last_mask_preview)
        if self.last_diff_preview is not None:
            cv2.imwrite(str(out_dir / "last_diff.jpg"), self.last_diff_preview)
        rel_dir = out_dir.relative_to(OUTPUT_ROOT).as_posix()
        result = {
            "ok": True,
            "elapsed_sec": elapsed,
            "out_dir": str(out_dir),
            "summary": summary,
            "review_url": f"/reports/{rel_dir}/summary.json",
            "player_url": f"/reports/{rel_dir}/summary.json",
            "summary_url": f"/reports/{rel_dir}/summary.json",
            "stdout": "",
        }
        self.log(
            f"视频检测完成: {overall_status} coverage={coverage:.1%} "
            f"{elapsed:.1f}s"
        )
        return result

    def next_expected_template_id(self) -> str:
        if not self.sequence:
            return ""
        if self.state.cursor < 0 or self.state.cursor >= len(self.sequence):
            return ""
        return self.sequence[self.state.cursor]

    def render_template_image(self, template_id: str) -> np.ndarray | None:
        template = self.templates_by_id.get(template_id)
        if template is None or not template.image:
            return None
        target = (self.model_path.parent / template.image).resolve()
        model_root = self.model_path.parent.resolve()
        if model_root not in target.parents and target != model_root:
            return None
        image = cv2.imread(str(target))
        if image is None:
            return None
        return image

    @staticmethod
    def clone_snapshot(snapshot: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
        return {
            key: None if value is None else value.copy()
            for key, value in snapshot.items()
        }

    def build_display_snapshot_locked(self) -> dict[str, np.ndarray | None]:
        return {
            "frame": None if self.last_frame is None else self.last_frame.copy(),
            "roi": None if self.last_roi is None else self.last_roi.copy(),
            "mask": None if self.last_mask_preview is None else self.last_mask_preview.copy(),
            "diff": None if self.last_diff_preview is None else self.last_diff_preview.copy(),
            "template": None if self.last_template_preview is None else self.last_template_preview.copy(),
        }

    def get_display_snapshot(self, key: str = "") -> dict[str, np.ndarray | None]:
        now = time.perf_counter()
        with self.lock:
            if key:
                expired = [item_key for item_key, (expires_at, _) in self.display_snapshots_by_key.items() if expires_at < now]
                for item_key in expired:
                    self.display_snapshots_by_key.pop(item_key, None)
                if key in self.display_snapshots_by_key:
                    return self.clone_snapshot(self.display_snapshots_by_key[key][1])
                snapshot = self.build_display_snapshot_locked()
                self.display_snapshots_by_key[key] = (now + 2.0, snapshot)
                return self.clone_snapshot(snapshot)
            if self.display_snapshot and now < self.display_snapshot_until:
                return self.clone_snapshot(self.display_snapshot)
            snapshot = self.build_display_snapshot_locked()
            self.display_snapshot = snapshot
            self.display_snapshot_until = now + 0.15
            return self.clone_snapshot(snapshot)

    def current_template_image(self) -> np.ndarray:
        image = self.get_display_snapshot().get("template")
        if image is not None:
            return image
        template_id = self.next_expected_template_id()
        image = self.render_template_image(template_id)
        if image is None:
            raise RuntimeError("没有可显示的标准模板")
        return image


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>七段管海康实时展示</title>
  <style>
    body { margin: 0; background: #0f172a; color: #e5e7eb; font-family: Arial, "Microsoft YaHei", sans-serif; }
    header { padding: 18px 24px; background: #111827; border-bottom: 1px solid #334155; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    main { padding: 18px 24px 28px; }
    .bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 12px 0; }
    select, input { background: #111827; color: #e5e7eb; border: 1px solid #475569; border-radius: 6px; padding: 8px 10px; }
    button { border: 0; border-radius: 6px; padding: 9px 14px; font-weight: 700; color: white; background: #2563eb; cursor: pointer; }
    button.secondary { background: #475569; }
    button.warn { background: #b45309; }
    .pill { background: #334155; padding: 8px 12px; border-radius: 6px; font-weight: 700; }
    .pass { background: #15803d; } .ng { background: #b91c1c; } .wait { background: #475569; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0; }
    .metric { background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; }
    .metric b { display: block; color: #94a3b8; font-size: 12px; margin-bottom: 4px; }
    .metric span { font-size: 23px; font-weight: 800; color: #f8fafc; }
    .stage { display: grid; grid-template-columns: 1.3fr 1fr 1fr; gap: 14px; }
    .panel { background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 12px; min-width: 0; }
    .panel h2 { margin: 0 0 10px; font-size: 18px; }
    img { width: 100%; background: #020617; border-radius: 8px; display: block; }
    .roi { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
    .roi label { display: grid; gap: 4px; color: #cbd5e1; font-size: 12px; }
    .log { height: 130px; overflow: auto; white-space: pre-wrap; background: #020617; border: 1px solid #334155; border-radius: 8px; padding: 10px; color: #cbd5e1; font-size: 13px; }
    .videoDetect { display: grid; grid-template-columns: 1fr auto auto; gap: 10px; align-items: center; }
    .videoResult { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-top: 10px; }
    a { color: #93c5fd; font-weight: 700; }
    @media (max-width: 1100px) { .stage { grid-template-columns: 1fr; } }
    @media (max-width: 760px) { .videoDetect { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<header>
  <h1>七段管海康实时展示</h1>
  <div>本地 Python 服务连接海康/OpenCV 相机，网页实时显示预览和关键模板时序检测。</div>
</header>
<main>
  <div class="bar">
    <select id="camera"></select>
    <button onclick="refreshCameras()" class="secondary">刷新相机</button>
    <button onclick="startPreview()">开始预览</button>
    <button onclick="stopPreview()" class="secondary">停止</button>
    <button onclick="resetDetect()" class="warn">重置检测</button>
    <button onclick="window.open('/client_camera.html','_blank')" class="secondary">远端本机摄像头模式</button>
    <span id="status" class="pill wait">未开始</span>
  </div>
  <div class="metrics">
    <div class="metric"><b>当前模板</b><span id="tmpl">-</span></div>
    <div class="metric"><b>相似度</b><span id="sim">0.00</span></div>
    <div class="metric"><b>有序命中</b><span id="hits">0 / 0</span></div>
    <div class="metric"><b>覆盖率</b><span id="coverage">0%</span></div>
    <div class="metric"><b>未知亮段</b><span id="unknown">0</span></div>
    <div class="metric"><b>判定</b><span id="result">WAIT</span></div>
  </div>
  <div class="stage">
    <section class="panel">
      <h2>相机预览</h2>
      <img id="frame" src="/frame.jpg">
      <div class="roi">
        <label>X<input id="x" type="number"></label>
        <label>Y<input id="y" type="number"></label>
        <label>W<input id="w" type="number"></label>
        <label>H<input id="h" type="number"></label>
      </div>
    </section>
    <section class="panel">
      <h2>当前 ROI</h2>
      <img id="roi" src="/roi.jpg">
    </section>
    <section class="panel">
      <h2>抽象亮区</h2>
      <img id="mask" src="/mask.jpg">
    </section>
  </div>
  <section class="panel" style="margin-top:14px;">
    <h2>导入视频对比检测</h2>
    <div class="videoDetect">
      <input id="videoPath" value="C:\\Users\\Administrator\\Videos\\518.mp4" placeholder="输入本机视频路径，例如 C:\\Users\\Administrator\\Videos\\518.mp4">
      <label><input id="quickVideo" type="checkbox"> 快速模式</label>
      <button onclick="runVideoDetect()">导入视频检测</button>
    </div>
    <div id="videoStatus" class="bar"><span class="pill wait">未检测</span></div>
    <div id="videoResult" class="videoResult"></div>
  </section>
  <section class="panel" style="margin-top:14px;">
    <h2>日志</h2>
    <div id="log" class="log"></div>
  </section>
</main>
<script>
let loop = null;
async function api(path, options) {
  const r = await fetch(path, options);
  const data = await r.json();
  if (!r.ok || data.ok === false) throw new Error(data.message || data.error || r.statusText);
  return data;
}
function roiQuery() {
  return `x=${x.value || 0}&y=${y.value || 0}&w=${w.value || 0}&h=${h.value || 0}`;
}
function updateImages() {
  const t = Date.now();
  frame.src = `/frame.jpg?${roiQuery()}&t=${t}`;
  roi.src = `/roi.jpg?t=${t}`;
  mask.src = `/mask.jpg?t=${t}`;
}
function render(data) {
  if (data.roi) {
    x.value = data.roi.x; y.value = data.roi.y; w.value = data.roi.w; h.value = data.roi.h;
  }
  tmpl.textContent = data.template_id || "-";
  sim.textContent = Number(data.similarity || 0).toFixed(2);
  hits.textContent = `${data.ordered_hits || 0} / ${data.sequence_count || 0}`;
  coverage.textContent = `${Math.round((data.coverage || 0) * 100)}%`;
  unknown.textContent = data.unknown_segments || 0;
  result.textContent = data.result || "WAIT";
  status.textContent = data.message || data.result || "运行中";
  status.className = "pill " + (data.result === "PASS" ? "pass" : (data.status === "unknown" ? "ng" : "wait"));
  log.textContent = (data.logs || []).slice().reverse().join("\\n");
}
async function refreshCameras() {
  const data = await api("/api/cameras");
  camera.innerHTML = "";
  for (const c of data.cameras) {
    const option = document.createElement("option");
    option.value = c.label;
    option.textContent = `${c.kind === "hikvision" ? "海康" : "OpenCV"} - ${c.label}`;
    camera.appendChild(option);
  }
  render(data);
}
async function startPreview() {
  const data = await api("/api/start", {method:"POST", body: JSON.stringify({camera: camera.value})});
  render(data);
  if (loop) clearInterval(loop);
  loop = setInterval(tick, 220);
}
async function stopPreview() {
  if (loop) clearInterval(loop);
  loop = null;
  render(await api("/api/stop", {method:"POST"}));
}
async function resetDetect() {
  render(await api("/api/reset", {method:"POST"}));
}
async function runVideoDetect() {
  const statusBox = document.getElementById("videoStatus");
  const resultBox = document.getElementById("videoResult");
  statusBox.innerHTML = '<span class="pill wait">视频检测中...</span>';
  resultBox.innerHTML = "";
  try {
    const data = await api("/api/video_detect", {
      method: "POST",
      body: JSON.stringify({ path: videoPath.value, quick: quickVideo.checked })
    });
    const s = data.summary || {};
    const seq = s.sequence_check || {};
    statusBox.innerHTML = `<span class="pill ${s.overall_status === "PASS" ? "pass" : "ng"}">${s.overall_status || "-"}</span>`;
    resultBox.innerHTML = `
      <div class="metric"><b>时序结果</b><span>${s.sequence_status || "-"}</span></div>
      <div class="metric"><b>覆盖率</b><span>${Math.round((seq.ordered_coverage || 0) * 100)}%</span></div>
      <div class="metric"><b>匹配帧</b><span>${s.matched_frames || 0}</span></div>
      <div class="metric"><b>未知亮帧</b><span>${s.unknown_frames || 0}</span></div>
      <div class="metric"><b>耗时</b><span>${data.elapsed_sec.toFixed(1)}s</span></div>
      <div class="metric"><b>报告</b><span><a href="${data.review_url}" target="_blank">打开</a> / <a href="${data.player_url}" target="_blank">播放器</a></span></div>
    `;
  } catch (e) {
    statusBox.innerHTML = `<span class="pill ng">${e.message}</span>`;
  }
}
async function tick() {
  try {
    const data = await api(`/api/detect?${roiQuery()}`);
    render(data);
    updateImages();
  } catch (e) {
    status.textContent = e.message;
    status.className = "pill ng";
  }
}
refreshCameras().catch(e => { status.textContent = e.message; status.className = "pill ng"; });
</script>
</body>
</html>
"""


BACKEND: SevenSegmentLiveBackend


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/index.html"):
                self.write(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/client_camera.html":
                model_data = build_client_camera_model_data()
                html = CLIENT_CAMERA_HTML.replace("__MODEL_DATA__", json.dumps(model_data, ensure_ascii=False))
                self.write(200, html.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path in ("/api/cameras", "/cameras"):
                self.write(*json_bytes({"ok": True, "cameras": BACKEND.refresh_cameras(), **BACKEND.status()}))
            elif parsed.path in ("/api/status", "/status"):
                self.write(*json_bytes(BACKEND.status()))
            elif parsed.path == "/api/detect":
                self.write(*json_bytes(BACKEND.detect_once(query)))
            elif parsed.path == "/frame.jpg":
                BACKEND.set_roi_from_query(query)
                snapshot = BACKEND.get_display_snapshot(query.get("t", [""])[0])
                frame = snapshot.get("frame")
                if frame is None:
                    frame = BACKEND.latest_frame()
                if frame is None:
                    raise RuntimeError("没有实时帧")
                if snapshot.get("frame") is None:
                    frame = BACKEND.draw_roi(frame.copy(), BACKEND.roi)
                self.write(200, jpeg_bytes(frame), "image/jpeg")
            elif parsed.path == "/roi.jpg":
                roi = BACKEND.get_display_snapshot(query.get("t", [""])[0]).get("roi")
                if roi is None:
                    raise RuntimeError("没有 ROI 图")
                self.write(200, jpeg_bytes(roi), "image/jpeg")
            elif parsed.path == "/mask.jpg":
                mask = BACKEND.get_display_snapshot(query.get("t", [""])[0]).get("mask")
                if mask is None:
                    raise RuntimeError("没有抽象图")
                self.write(200, jpeg_bytes(mask), "image/jpeg")
            elif parsed.path == "/diff.jpg":
                diff = BACKEND.get_display_snapshot(query.get("t", [""])[0]).get("diff")
                if diff is None:
                    raise RuntimeError("没有差异图")
                self.write(200, jpeg_bytes(diff), "image/jpeg")
            elif parsed.path == "/template.jpg":
                snapshot = BACKEND.get_display_snapshot(query.get("t", [""])[0])
                template = snapshot.get("template")
                if template is None:
                    template = BACKEND.current_template_image()
                self.write(200, jpeg_bytes(template, quality=88), "image/jpeg")
            elif parsed.path.startswith("/reports/"):
                self.serve_report_file(parsed.path.removeprefix("/reports/"))
            else:
                self.write(*json_bytes({"ok": False, "error": "not found"}, 404))
        except Exception as exc:
            self.write(*json_bytes({"ok": False, "message": str(exc)}, 500))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if parsed.path in ("/api/start", "/start"):
                self.write(*json_bytes(BACKEND.start(str(payload.get("camera", "")))))
            elif parsed.path in ("/api/stop", "/stop"):
                self.write(*json_bytes(BACKEND.stop()))
            elif parsed.path in ("/api/reset", "/reset"):
                self.write(*json_bytes(BACKEND.reset()))
            elif parsed.path in ("/api/ack", "/ack"):
                self.write(*json_bytes(BACKEND.ack()))
            elif parsed.path in ("/api/video_start", "/video_start"):
                self.write(*json_bytes(BACKEND.start_video_detection(str(payload.get("path", "")), bool(payload.get("realtime", True)))))
            elif parsed.path in ("/api/video_detect", "/video_detect"):
                self.write(*json_bytes(BACKEND.run_video_detection(str(payload.get("path", "")), bool(payload.get("quick", False)))))
            else:
                self.write(*json_bytes({"ok": False, "error": "not found"}, 404))
        except Exception as exc:
            self.write(*json_bytes({"ok": False, "message": str(exc)}, 500))

    def log_message(self, format: str, *args: Any) -> None:
        return

    def serve_report_file(self, rel_path: str) -> None:
        target = (OUTPUT_ROOT / rel_path).resolve()
        output_root = OUTPUT_ROOT.resolve()
        if output_root not in target.parents and target != output_root:
            self.write(*json_bytes({"ok": False, "error": "bad path"}, 403))
            return
        if not target.exists() or not target.is_file():
            self.write(*json_bytes({"ok": False, "error": "not found"}, 404))
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self.write(200, target.read_bytes(), content_type)

    def write(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    global BACKEND
    BACKEND = SevenSegmentLiveBackend(args.model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"七段管实时展示: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        BACKEND.stop()


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
