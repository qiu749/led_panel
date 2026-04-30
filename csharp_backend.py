from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from desktop_app import (
    CV_TEMPLATE_MODEL_DEFAULT,
    MARQUEE_TEMPLATE_MODEL_DEFAULT,
    OUTPUT_DEFAULT,
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


def json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes, str]:
    return status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"


class InspectionBackend:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.cameras: list[CameraInfo] = []
        self.camera_stream: OpenCVCameraStream | HikvisionCameraStream | None = None
        self.template_cache: dict[str, Any] = {}
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
        print(line, flush=True)

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
                self.rotate_180 = bool(payload["rotate_180"])
            if "output_dir" in payload and str(payload["output_dir"]).strip():
                self.output_dir = Path(str(payload["output_dir"]).strip())
        return self.status()

    def start_preview(self, camera_label: str = "") -> dict[str, Any]:
        self.stop_preview()
        with self.lock:
            if not self.cameras:
                self.refresh_cameras()
            camera = self.select_camera(camera_label)
        if camera is None:
            raise RuntimeError("没有可用相机")
        stream: OpenCVCameraStream | HikvisionCameraStream
        if camera.kind == "hikvision":
            stream = HikvisionCameraStream(camera, self.log)
        else:
            stream = OpenCVCameraStream(camera, self.log)
        stream.start()
        with self.lock:
            self.camera_stream = stream
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
            self.log("停止预览")
        return self.status()

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
            templates = self.get_templates()
            frame_payloads = [self.analyze_frame(frame, templates) for frame in frames]
            payload = self.fuse_frames(frame_payloads, templates, frame_count, interval_ms)
            base_frame = self.choose_base_frame(frames, frame_payloads)
            result_image = self.draw_operator_result(base_frame, payload, templates)

            run_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_csharp_station")
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
            if path == "/refresh_cameras":
                self.write_response(*json_bytes({"cameras": BACKEND.refresh_cameras()}))
            elif path == "/configure":
                self.write_response(*json_bytes(BACKEND.configure(payload)))
            elif path == "/start_preview":
                self.write_response(*json_bytes(BACKEND.start_preview(str(payload.get("camera", "")))))
            elif path == "/stop_preview":
                self.write_response(*json_bytes(BACKEND.stop_preview()))
            elif path == "/detect":
                self.write_response(*json_bytes(BACKEND.detect(int(payload.get("frame_count", 7)), int(payload.get("interval_ms", 45)))))
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
    print(f"LED inspection backend listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        BACKEND.stop_preview()
        server.server_close()


if __name__ == "__main__":
    main()
