from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_direct_point_templates import detect_points as detect_led_points
import marquee_inspection_v2 as grid_probe


DEFAULT_OUTPUT_DIR = Path("output") / "new_model_12state_probe"
DEFAULT_LONG_SIDE = 720


@dataclass
class SampleFrame:
    sample_index: int
    frame_index: int
    time_sec: float
    frame: np.ndarray


@dataclass
class StateSegment:
    segment_id: int
    color: str
    start_sample: int
    end_sample: int
    start_time_sec: float
    end_time_sec: float
    frame_count: int
    pattern_id: int = 0
    state_key: str = ""
    prototype_similarity: float = 0.0


@dataclass
class TrackedMeasurements:
    brightness: np.ndarray
    colors: np.ndarray
    centers: np.ndarray


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in value.replace(";", ",").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox 格式应为 x,y,w,h")
    return tuple(int(float(part)) for part in parts)  # type: ignore[return-value]


def video_meta(video_path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frame_count <= 0 or fps <= 0:
        raise RuntimeError(f"视频元数据异常: {video_path}")
    return frame_count, fps, width, height


def sample_video_frames(
    video_path: Path,
    start_sec: float,
    duration_sec: float,
    sample_fps: float,
) -> tuple[list[SampleFrame], float, dict[str, Any]]:
    frame_count, fps, width, height = video_meta(video_path)
    cap = cv2.VideoCapture(str(video_path))
    start_frame = max(0, min(frame_count - 1, int(round(start_sec * fps))))
    end_frame = frame_count - 1
    if duration_sec > 0:
        end_frame = min(frame_count - 1, int(round((start_sec + duration_sec) * fps)))
    step_frames = max(1, int(round(fps / max(sample_fps, 0.1))))

    frames: list[SampleFrame] = []
    sample_index = 0
    for frame_index in range(start_frame, end_frame + 1, step_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(
            SampleFrame(
                sample_index=sample_index,
                frame_index=frame_index,
                time_sec=frame_index / fps,
                frame=frame,
            )
        )
        sample_index += 1
    cap.release()
    if not frames:
        raise RuntimeError("没有采到任何有效帧")
    meta = {
        "video": str(video_path),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "sample_start_sec": start_sec,
        "sample_duration_sec": duration_sec,
        "sample_fps": sample_fps,
        "sampled_frames": len(frames),
        "sample_step_frames": step_frames,
    }
    return frames, fps, meta


def detect_matrix_bbox(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    threshold = max(20.0, float(np.percentile(blurred, 99.5)) * 0.45)
    binary = (blurred >= threshold).astype(np.uint8) * 255
    kernel_size = max(15, int(round(min(frame.shape[:2]) * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    close_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilate_kernel = np.ones((max(5, kernel_size // 3), max(5, kernel_size // 3)), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    binary = cv2.dilate(binary, dilate_kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    height, width = frame.shape[:2]
    frame_area = float(width * height)
    center_x = width * 0.5
    center_y = height * 0.5
    best_score = -1.0
    best_bbox: tuple[int, int, int, int] | None = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < frame_area * 0.01 or area > frame_area * 0.30:
            continue
        aspect = w / max(h, 1)
        if not 0.5 <= aspect <= 3.0:
            continue
        extent = cv2.contourArea(contour) / max(area, 1.0)
        cx = x + w * 0.5
        cy = y + h * 0.5
        centrality = 1.0 - min(1.0, math.hypot((cx - center_x) / width, (cy - center_y) / height) * 2.0)
        score = area / frame_area + extent * 0.8 + centrality * 0.3
        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)
    if best_bbox is None:
        return None
    x, y, w, h = best_bbox
    pad_x = max(8, int(round(w * 0.08)))
    pad_y = max(8, int(round(h * 0.08)))
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(width - x, w + pad_x * 2)
    h = min(height - y, h + pad_y * 2)
    return x, y, w, h


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = pts[:, 1] - pts[:, 0]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def quad_center(quad: np.ndarray) -> tuple[float, float]:
    ordered = order_corners(quad)
    center = ordered.mean(axis=0)
    return float(center[0]), float(center[1])


def expand_quad(quad: np.ndarray, frame_shape: tuple[int, int], pad_px: float) -> np.ndarray:
    ordered = order_corners(quad)
    center = ordered.mean(axis=0)
    expanded = ordered.copy()
    for index, point in enumerate(ordered):
        vector = point - center
        length = float(np.linalg.norm(vector))
        if length <= 1e-3:
            continue
        expanded[index] = center + vector * ((length + pad_px) / length)

    width = frame_shape[1]
    height = frame_shape[0]
    expanded[:, 0] = np.clip(expanded[:, 0], 0.0, float(width - 1))
    expanded[:, 1] = np.clip(expanded[:, 1], 0.0, float(height - 1))
    return expanded.astype(np.float32)


def padded_quad(quad: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    _, _, width, height = quad_bbox(quad)
    pad_px = float(np.clip(min(width, height) * 0.018, 6.0, 14.0))
    return expand_quad(quad, frame_shape, pad_px)


def quad_iou(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    ax, ay, aw, ah = quad_bbox(order_corners(quad_a))
    bx, by, bw, bh = quad_bbox(order_corners(quad_b))
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = float(inter_w * inter_h)
    area_a = float(max(1, aw * ah))
    area_b = float(max(1, bw * bh))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def sort_quads_reading_order(quads: list[np.ndarray]) -> list[np.ndarray]:
    if len(quads) <= 1:
        return [order_corners(quad) for quad in quads]
    centers = [quad_center(quad) for quad in quads]
    heights = [quad_bbox(quad)[3] for quad in quads]
    row_band = max(24.0, float(np.median(np.asarray(heights, dtype=np.float32))) * 0.6)
    ranked = sorted(zip(quads, centers), key=lambda item: (item[1][1], item[1][0]))
    rows: list[list[tuple[np.ndarray, tuple[float, float]]]] = []
    for quad, center in ranked:
        placed = False
        for row in rows:
            row_y = float(np.mean([item[1][1] for item in row]))
            if abs(center[1] - row_y) <= row_band:
                row.append((quad, center))
                placed = True
                break
        if not placed:
            rows.append([(quad, center)])
    ordered: list[np.ndarray] = []
    for row in rows:
        row.sort(key=lambda item: item[1][0])
        ordered.extend(order_corners(item[0]) for item in row)
    return ordered


def panel_cluster_kernel(frame_shape: tuple[int, int]) -> int:
    kernel_size = int(round(min(frame_shape) * 0.028))
    kernel_size = max(25, min(55, kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def find_matrix_quad_candidates_from_points(frame: np.ndarray) -> list[tuple[float, np.ndarray]]:
    points, _ = detect_led_points(frame)
    if len(points) < 120:
        return []

    height, width = frame.shape[:2]
    frame_area = float(width * height)
    center_x = width * 0.5
    center_y = height * 0.5

    point_mask = np.zeros((height, width), dtype=np.uint8)
    point_array = np.asarray(points, dtype=np.float32)
    point_pixels = np.round(point_array).astype(np.int32)
    point_pixels[:, 0] = np.clip(point_pixels[:, 0], 0, width - 1)
    point_pixels[:, 1] = np.clip(point_pixels[:, 1], 0, height - 1)
    for px, py in point_pixels:
        cv2.circle(point_mask, (int(px), int(py)), 2, 255, -1)

    kernel_size = panel_cluster_kernel((height, width))
    cluster_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    merged = cv2.dilate(point_mask, cluster_kernel, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, cluster_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    point_labels = labels[point_pixels[:, 1], point_pixels[:, 0]]

    candidates: list[tuple[float, np.ndarray]] = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < frame_area * 0.01:
            continue

        cluster_points = point_array[point_labels == index]
        point_count = int(len(cluster_points))
        if point_count < 120:
            continue

        rect = cv2.minAreaRect(cluster_points.astype(np.float32))
        (_, _), (rw, rh), _ = rect
        bbox_area = max(float(rw * rh), 1.0)
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        if bbox_area < frame_area * 0.01 or bbox_area > frame_area * 0.42:
            continue
        if not 1.05 <= aspect <= 4.2:
            continue

        component_mask = np.zeros((h, w), dtype=np.uint8)
        component_mask[labels[y : y + h, x : x + w] == index] = 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = max((cv2.contourArea(contour) for contour in contours), default=0.0)

        cx, cy = quad_center(cv2.boxPoints(rect))
        extent = contour_area / bbox_area
        density = point_count / max(1.0, bbox_area / 900.0)
        centrality = 1.0 - min(1.0, math.hypot((cx - center_x) / width, (cy - center_y) / height) * 2.0)
        score = bbox_area / frame_area + extent * 0.45 + density * 0.08 + centrality * 0.12
        candidates.append((float(score), order_corners(cv2.boxPoints(rect))))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def find_matrix_quad_candidates_from_contours(frame: np.ndarray) -> list[tuple[float, np.ndarray]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    threshold = max(20.0, float(np.percentile(blurred, 99.5)) * 0.45)
    binary = (blurred >= threshold).astype(np.uint8) * 255
    kernel_size = max(15, int(round(min(frame.shape[:2]) * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    close_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilate_kernel = np.ones((max(5, kernel_size // 3), max(5, kernel_size // 3)), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    binary = cv2.dilate(binary, dilate_kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    height, width = frame.shape[:2]
    frame_area = float(width * height)
    center_x = width * 0.5
    center_y = height * 0.5
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < frame_area * 0.008:
            continue
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        box = cv2.boxPoints(rect)
        bbox_area = max(rw * rh, 1.0)
        aspect = max(rw, rh) / max(min(rw, rh), 1.0)
        if bbox_area < frame_area * 0.01 or bbox_area > frame_area * 0.32:
            continue
        if not 1.1 <= aspect <= 3.5:
            continue
        extent = area / bbox_area
        centrality = 1.0 - min(1.0, math.hypot((cx - center_x) / width, (cy - center_y) / height) * 2.0)
        score = bbox_area / frame_area + extent * 0.7 + centrality * 0.3
        candidates.append((float(score), order_corners(box)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def find_matrix_quad_candidates(frame: np.ndarray) -> list[tuple[float, np.ndarray]]:
    merged: list[tuple[float, np.ndarray]] = []
    for score, quad in find_matrix_quad_candidates_from_points(frame):
        merged.append((score + 0.2, quad))
    for score, quad in find_matrix_quad_candidates_from_contours(frame):
        if any(quad_iou(quad, existing_quad) >= 0.28 for _, existing_quad in merged):
            continue
        merged.append((score, quad))
    merged.sort(key=lambda item: item[0], reverse=True)
    return merged


def detect_matrix_quad(frame: np.ndarray) -> np.ndarray | None:
    candidates = find_matrix_quad_candidates(frame)
    if not candidates:
        return None
    return padded_quad(candidates[0][1], frame.shape[:2])


def detect_matrix_quads(frame: np.ndarray, max_quads: int = 4, iou_threshold: float = 0.28) -> list[np.ndarray]:
    selected: list[np.ndarray] = []
    for _, quad in find_matrix_quad_candidates(frame):
        if any(quad_iou(quad, existing) >= iou_threshold for existing in selected):
            continue
        selected.append(quad)
        if len(selected) >= max_quads:
            break
    padded = [padded_quad(quad, frame.shape[:2]) for quad in selected]
    return sort_quads_reading_order(padded)


def quad_bbox(quad: np.ndarray) -> tuple[int, int, int, int]:
    xs = quad[:, 0]
    ys = quad[:, 1]
    x1 = int(round(float(np.min(xs))))
    y1 = int(round(float(np.min(ys))))
    x2 = int(round(float(np.max(xs))))
    y2 = int(round(float(np.max(ys))))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def median_quad(quads: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack([order_corners(quad) for quad in quads], axis=0)
    return np.median(stacked, axis=0).astype(np.float32)


def quad_size(quad: np.ndarray, long_side: int = DEFAULT_LONG_SIDE) -> tuple[int, int]:
    ordered = order_corners(quad)
    top = float(np.linalg.norm(ordered[1] - ordered[0]))
    bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    left = float(np.linalg.norm(ordered[3] - ordered[0]))
    right = float(np.linalg.norm(ordered[2] - ordered[1]))
    width = max(top, bottom, 1.0)
    height = max(left, right, 1.0)
    if width >= height:
        return long_side, max(64, int(round(long_side * height / width)))
    return max(64, int(round(long_side * width / height))), long_side


def warp_panel_quad(frame: np.ndarray, quad: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    ordered = order_corners(quad)
    pw, ph = size
    dst = np.array(
        [
            [0.0, 0.0],
            [float(pw - 1), 0.0],
            [float(pw - 1), float(ph - 1)],
            [0.0, float(ph - 1)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(frame, matrix, (pw, ph), flags=cv2.INTER_LINEAR)


def median_bbox(boxes: list[tuple[int, int, int, int]], fallback_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    if not boxes:
        width, height = fallback_shape
        margin_x = int(width * 0.28)
        margin_y = int(height * 0.28)
        return margin_x, margin_y, width - margin_x * 2, height - margin_y * 2
    arr = np.asarray(boxes, dtype=np.float32)
    x, y, w, h = np.median(arr, axis=0)
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def standard_size(bbox: tuple[int, int, int, int], long_side: int = DEFAULT_LONG_SIDE) -> tuple[int, int]:
    _, _, width, height = bbox
    if width <= 0 or height <= 0:
        return (long_side, long_side)
    if width >= height:
        return long_side, max(64, int(round(long_side * height / width)))
    return max(64, int(round(long_side * width / height))), long_side


def build_activity_region(panels: list[np.ndarray]) -> tuple[int, int, int, int]:
    accum: np.ndarray | None = None
    for panel in panels:
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        threshold = max(25.0, float(np.percentile(gray, 99.2)) * 0.55)
        binary = (gray >= threshold).astype(np.uint8)
        if accum is None:
            accum = binary.astype(np.float32)
        else:
            accum += binary.astype(np.float32)
    if accum is None:
        raise RuntimeError("无法构建点阵活跃区域")
    mask = accum >= max(2.0, len(panels) * 0.04)
    binary = (mask.astype(np.uint8) * 255)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, 0, panels[0].shape[1], panels[0].shape[0]
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    pad_x = max(6, int(round(w * 0.05)))
    pad_y = max(6, int(round(h * 0.05)))
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(panels[0].shape[1] - x, w + pad_x * 2)
    h = min(panels[0].shape[0] - y, h + pad_y * 2)
    return x, y, w, h


def infer_axis_gap(values: list[float], fallback: float) -> float:
    if len(values) < 2:
        return fallback
    diffs = np.diff(np.sort(np.asarray(values, dtype=np.float32)))
    diffs = diffs[diffs > 0.8]
    if len(diffs) == 0:
        return fallback
    pitch = float(np.median(diffs))
    return max(3.0, min(18.0, pitch * 0.4))


def cluster_axis(values: list[float], gap: float) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)
    groups: list[list[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(sum(group) / len(group)) for group in groups]


def fit_series(values: list[float], degree: int = 1) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) <= 1:
        return arr.copy()
    deg = min(degree, len(arr) - 1)
    xs = np.arange(len(arr), dtype=np.float32)
    coeffs = np.polyfit(xs, arr, deg=deg)
    return np.polyval(coeffs, xs).astype(np.float32)


def detect_led_layout(
    panels: list[np.ndarray],
    expected_rows: int | None,
    expected_cols: int | None,
    matrix_bbox: tuple[int, int, int, int],
) -> tuple[list[grid_probe.Slot], dict[str, Any], np.ndarray, dict[int, tuple[int, int]]]:
    x, y, w, h = matrix_bbox
    panel_crops: list[np.ndarray] = []
    panel_scores: list[float] = []
    for panel in panels:
        crop = panel[y : y + h, x : x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        panel_crops.append(gray)
        panel_scores.append(float(np.percentile(gray, 99.5)))

    if not panel_crops:
        raise RuntimeError("没有可用于建模的面板帧")

    best_index = int(np.argmax(panel_scores))
    heatmap = panel_crops[best_index]

    peak_threshold = max(22.0, float(np.percentile(heatmap, 99.35)) * 0.42)
    binary = (heatmap >= peak_threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num_labels, _, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_area = max(4, int(round((w * h) / 18000)))
    max_area = max(min_area + 4, int(round((w * h) / 500)))
    led_points: list[tuple[float, float, int]] = []
    for index in range(1, num_labels):
        bx, by, bw, bh, area = stats[index]
        aspect = bw / max(bh, 1)
        if area < min_area or area > max_area:
            continue
        if not 0.45 <= aspect <= 2.2:
            continue
        led_points.append((float(centers[index][0]) + x, float(centers[index][1]) + y, int(area)))

    if len(led_points) < 30:
        raise RuntimeError(f"有效灯珠中心过少: {len(led_points)}")

    y_gap = infer_axis_gap([point[1] for point in led_points], fallback=max(4.0, h / 24.0))
    row_centers = cluster_axis([point[1] for point in led_points], y_gap)
    if expected_rows and len(row_centers) != expected_rows:
        # 如果用户给了行数，就按最近行中心重新压缩到指定数量
        row_edges = np.linspace(min(point[1] for point in led_points), max(point[1] for point in led_points), expected_rows)
        row_centers = [float(v) for v in row_edges]

    row_groups: list[list[tuple[float, float]]] = [[] for _ in row_centers]
    for px, py, _ in led_points:
        row_index = int(np.argmin([abs(py - center) for center in row_centers]))
        row_groups[row_index].append((px, py))
    row_groups = [sorted(group, key=lambda item: item[0]) for group in row_groups if len(group) >= 4]
    if not row_groups:
        raise RuntimeError("无法稳定分组灯珠行")
    row_centers = [float(np.mean([point[1] for point in group])) for group in row_groups]

    x_diffs: list[float] = []
    for group in row_groups:
        xs = [point[0] for point in group]
        x_diffs.extend([diff for diff in np.diff(xs) if diff > 1.0])
    x_pitch = float(np.median(x_diffs)) if x_diffs else max(6.0, w / 40.0)
    y_pitch = float(np.median(np.diff(row_centers))) if len(row_centers) > 1 else max(6.0, h / 12.0)
    slot_w = max(6.0, x_pitch * 0.72)
    slot_h = max(6.0, y_pitch * 0.72)

    row_count = len(row_groups)
    row_lengths = [len(group) for group in row_groups]
    if expected_cols is not None:
        row_lengths = [min(length, expected_cols) for length in row_lengths]
    col_count = max(row_lengths)

    slots: list[grid_probe.Slot] = []
    slot_layout: dict[int, tuple[int, int]] = {}
    for row_index, group in enumerate(row_groups, start=1):
        target_len = row_lengths[row_index - 1]
        if target_len <= 0:
            continue
        # Dense LED matrices are not perfectly affine: rows are staggered and lens/perspective
        # distortion bends local centers. Use the detected component centers directly instead
        # of fabricating an ideal linspace row, otherwise previews drift between neighboring LEDs.
        row_points = group
        if expected_cols is not None and len(row_points) > target_len:
            # Keep the points closest to an estimated regular row only when there are extras.
            # This removes small glare/noise components without replacing valid centers.
            left = float(row_points[0][0])
            right = float(row_points[-1][0])
            ideal_xs = np.linspace(left, right, target_len, dtype=np.float32)
            candidates: list[tuple[float, int, tuple[float, float]]] = []
            for point_index, point in enumerate(row_points):
                distance = float(np.min(np.abs(ideal_xs - float(point[0]))))
                candidates.append((distance, point_index, point))
            keep_indices = {item[1] for item in sorted(candidates, key=lambda item: item[0])[:target_len]}
            row_points = [point for point_index, point in enumerate(row_points) if point_index in keep_indices]
        for col_index, (px, py) in enumerate(row_points, start=1):
            slot_id = len(slots) + 1
            slots.append(
                grid_probe.Slot(
                    id=slot_id,
                    x=float(px),
                    y=float(py),
                    w=slot_w,
                    h=slot_h,
                    threshold=0.25,
                )
            )
            slot_layout[slot_id] = (row_index, col_index)

    scores = grid_probe.compute_slot_scores(panels, slots)
    filtered: list[grid_probe.Slot] = []
    filtered_layout: dict[int, tuple[int, int]] = {}
    for slot_index, slot in enumerate(slots):
        series = scores[:, slot_index]
        peak = float(np.max(series))
        background = float(np.percentile(series, 20))
        if peak < 0.20:
            continue
        threshold = max(0.16, min(0.80, background + (peak - background) * 0.42))
        duty = float(np.mean(series >= threshold))
        new_slot = grid_probe.Slot(
            id=len(filtered) + 1,
            x=slot.x,
            y=slot.y,
            w=slot.w,
            h=slot.h,
            threshold=threshold,
            peak=peak,
            duty=duty,
        )
        filtered.append(new_slot)
        filtered_layout[new_slot.id] = slot_layout[slot.id]
    slots = filtered
    slot_layout = filtered_layout
    scores = grid_probe.compute_slot_scores(panels, slots)
    thresholds = np.array([slot.threshold for slot in slots], dtype=np.float32)
    active_counts = np.sum(scores >= thresholds[None, :], axis=1)
    metadata = {
        "rows": row_count,
        "cols": col_count,
        "matrix_bbox": list(matrix_bbox),
        "avg_active_slots_per_frame": round(float(np.mean(active_counts)) if len(active_counts) else 0.0, 2),
        "detected_points": len(led_points),
        "filtered_slots": len(slots),
        "layout_mode": "detected_centers",
    }
    return slots, metadata, heatmap, slot_layout


def top_pixel_mean(roi: np.ndarray) -> tuple[float, float, float]:
    if roi.size == 0:
        return 0.0, 0.0, 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    threshold = max(5.0, float(np.percentile(gray, 85)))
    mask = gray >= threshold
    if int(mask.sum()) < 3:
        mean = roi.reshape(-1, 3).mean(axis=0)
    else:
        mean = roi[mask].mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def local_peak_center(
    gray: np.ndarray,
    nominal_x: float,
    nominal_y: float,
    search_radius: int,
    refine_radius: int,
    max_shift: float,
) -> tuple[float, float]:
    h, w = gray.shape
    x1 = max(0, int(round(nominal_x)) - search_radius)
    y1 = max(0, int(round(nominal_y)) - search_radius)
    x2 = min(w, int(round(nominal_x)) + search_radius + 1)
    y2 = min(h, int(round(nominal_y)) + search_radius + 1)
    patch = gray[y1:y2, x1:x2]
    if patch.size == 0:
        return nominal_x, nominal_y

    blurred = cv2.GaussianBlur(patch, (0, 0), sigmaX=1.2)
    max_index = int(np.argmax(blurred))
    peak_y, peak_x = np.unravel_index(max_index, blurred.shape)
    peak_x += x1
    peak_y += y1

    nominal_px = int(round(nominal_x))
    nominal_py = int(round(nominal_y))
    center_patch = gray[max(0, nominal_py - 1) : min(h, nominal_py + 2), max(0, nominal_px - 1) : min(w, nominal_px + 2)]
    peak_patch = gray[max(0, peak_y - 1) : min(h, peak_y + 2), max(0, peak_x - 1) : min(w, peak_x + 2)]
    center_strength = float(np.max(center_patch)) if center_patch.size else 0.0
    peak_strength = float(np.max(peak_patch)) if peak_patch.size else 0.0
    shift = math.hypot(float(peak_x) - nominal_x, float(peak_y) - nominal_y)
    if shift > max_shift and peak_strength < center_strength * 1.12:
        return nominal_x, nominal_y
    if peak_strength < max(center_strength * 1.03, 0.20):
        return nominal_x, nominal_y

    rx1 = max(0, peak_x - refine_radius)
    ry1 = max(0, peak_y - refine_radius)
    rx2 = min(w, peak_x + refine_radius + 1)
    ry2 = min(h, peak_y + refine_radius + 1)
    refine = gray[ry1:ry2, rx1:rx2]
    if refine.size == 0:
        return float(peak_x), float(peak_y)

    peak_value = float(np.max(refine))
    cutoff = max(peak_value * 0.78, float(np.percentile(refine, 90)))
    mask = refine >= cutoff
    if int(mask.sum()) < 3:
        return float(peak_x), float(peak_y)

    weights = refine[mask].astype(np.float32) + 1e-3
    ys, xs = np.nonzero(mask)
    cx = float(np.sum((xs + rx1) * weights) / np.sum(weights))
    cy = float(np.sum((ys + ry1) * weights) / np.sum(weights))
    return cx, cy


def measure_slots_with_local_search(panels: list[np.ndarray], slots: list[grid_probe.Slot]) -> TrackedMeasurements:
    brightness = np.zeros((len(panels), len(slots)), dtype=np.float32)
    colors = np.zeros((len(panels), len(slots), 3), dtype=np.float32)
    centers = np.zeros((len(panels), len(slots), 2), dtype=np.float32)
    if not panels or not slots:
        return TrackedMeasurements(brightness=brightness, colors=colors, centers=centers)

    for frame_index, panel in enumerate(panels):
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        for slot_index, slot in enumerate(slots):
            # After per-frame panel warping the template centers should stay stable.
            # Local peak search is unsafe on dense matrices: inactive slots can be
            # pulled onto neighboring lit LEDs/reflections and corrupt both preview
            # and brightness statistics.
            cx, cy = float(slot.x), float(slot.y)
            centers[frame_index, slot_index] = (cx, cy)

            half_w = max(3, int(round(slot.w * 0.42)))
            half_h = max(3, int(round(slot.h * 0.42)))
            x1 = max(0, int(round(cx)) - half_w)
            y1 = max(0, int(round(cy)) - half_h)
            x2 = min(gray.shape[1], int(round(cx)) + half_w + 1)
            y2 = min(gray.shape[0], int(round(cy)) + half_h + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            roi_gray = gray[y1:y2, x1:x2]
            brightness[frame_index, slot_index] = float(np.percentile(roi_gray, 98))
            colors[frame_index, slot_index] = top_pixel_mean(panel[y1:y2, x1:x2])
    return TrackedMeasurements(brightness=brightness, colors=colors, centers=centers)


def classify_frame_color(brightness_row: np.ndarray, color_row: np.ndarray, thresholds: np.ndarray) -> str:
    active = brightness_row >= thresholds
    if int(active.sum()) == 0:
        return "unknown"
    mean_b, mean_g, mean_r = color_row[active].mean(axis=0)
    channel_max = max(mean_r, mean_g, mean_b, 1e-6)
    if mean_r >= mean_g * 1.12 and mean_r >= mean_b * 1.12:
        return "red"
    if mean_g >= mean_r * 1.08 and mean_g >= mean_b * 1.08:
        return "green"
    if mean_b >= mean_r * 1.08 and mean_b >= mean_g * 1.08:
        return "blue"
    # 颜色混合时，回到最强通道，避免把有效状态丢成 unknown
    if channel_max == mean_r:
        return "red"
    if channel_max == mean_g:
        return "green"
    return "blue"


def smooth_labels(labels: list[str], window: int = 2) -> list[str]:
    if not labels:
        return labels
    smoothed: list[str] = []
    for index in range(len(labels)):
        left = max(0, index - window)
        right = min(len(labels), index + window + 1)
        counter = Counter(label for label in labels[left:right] if label != "unknown")
        if not counter:
            smoothed.append(labels[index])
            continue
        smoothed.append(counter.most_common(1)[0][0])
    return smoothed


def contiguous_segments(frames: list[SampleFrame], labels: list[str], min_length: int = 2) -> list[StateSegment]:
    segments: list[StateSegment] = []
    if not frames:
        return segments
    start = 0
    current = labels[0]
    for index in range(1, len(labels) + 1):
        value = labels[index] if index < len(labels) else None
        if value == current:
            continue
        if current != "unknown" and index - start >= min_length:
            start_frame = frames[start]
            end_frame = frames[index - 1]
            segments.append(
                StateSegment(
                    segment_id=len(segments) + 1,
                    color=current,
                    start_sample=start,
                    end_sample=index - 1,
                    start_time_sec=start_frame.time_sec,
                    end_time_sec=end_frame.time_sec,
                    frame_count=index - start,
                )
            )
        start = index
        current = value if value is not None else "unknown"
    return segments


def binary_signature(brightness: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return (brightness >= thresholds).astype(np.uint8)


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_bool = left.astype(bool)
    right_bool = right.astype(bool)
    union = np.logical_or(left_bool, right_bool).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(left_bool, right_bool).sum()
    return float(intersection / union)


def cluster_segments(
    segments: list[StateSegment],
    brightness: np.ndarray,
    thresholds: np.ndarray,
    max_patterns: int = 4,
) -> tuple[list[StateSegment], dict[tuple[str, int], np.ndarray]]:
    by_color: dict[str, list[StateSegment]] = defaultdict(list)
    for segment in segments:
        by_color[segment.color].append(segment)

    prototypes: dict[tuple[str, int], np.ndarray] = {}
    for color, color_segments in by_color.items():
        segment_signatures: list[tuple[StateSegment, np.ndarray]] = []
        for segment in color_segments:
            sample_slice = slice(segment.start_sample, segment.end_sample + 1)
            mean_scores = np.mean(brightness[sample_slice], axis=0)
            segment_signatures.append((segment, binary_signature(mean_scores, thresholds)))

        if not segment_signatures:
            continue

        vectors = np.stack([sig.astype(np.float32) for _, sig in segment_signatures], axis=0)
        cluster_count = min(max_patterns, len(segment_signatures))
        seed_indices = [0]
        while len(seed_indices) < cluster_count:
            distances = []
            for vec in vectors:
                nearest = min(float(np.mean((vec - vectors[index]) ** 2)) for index in seed_indices)
                distances.append(nearest)
            candidate = int(np.argmax(distances))
            if candidate in seed_indices:
                break
            seed_indices.append(candidate)
        prototypes_arr = vectors[seed_indices].copy()
        assignments = np.zeros(len(segment_signatures), dtype=np.int32)
        for _ in range(8):
            changed = False
            for index, vec in enumerate(vectors):
                distances = np.mean((prototypes_arr - vec[None, :]) ** 2, axis=1)
                new_assignment = int(np.argmin(distances))
                if assignments[index] != new_assignment:
                    assignments[index] = new_assignment
                    changed = True
            for cluster_index in range(len(prototypes_arr)):
                members = vectors[assignments == cluster_index]
                if len(members):
                    prototypes_arr[cluster_index] = np.mean(members, axis=0)
            if not changed:
                break

        assigned: list[tuple[StateSegment, int, float]] = []
        for (segment, sig), cluster_index in zip(segment_signatures, assignments.tolist()):
            proto_sig = (prototypes_arr[cluster_index] >= 0.5).astype(np.uint8)
            assigned.append((segment, cluster_index + 1, similarity(sig, proto_sig)))

        order = sorted(
            {pattern_id for _, pattern_id, _ in assigned},
            key=lambda pid: next(seg.start_time_sec for seg, p, _ in assigned if p == pid),
        )
        remap = {old: index + 1 for index, old in enumerate(order)}
        cluster_accum: dict[int, list[np.ndarray]] = defaultdict(list)
        for segment, pattern_id, proto_similarity in assigned:
            new_pattern = remap[pattern_id]
            segment.pattern_id = new_pattern
            segment.state_key = f"{color}_{new_pattern:02d}"
            segment.prototype_similarity = round(proto_similarity, 4)
            segment.segment_id = segment.segment_id
            sample_slice = slice(segment.start_sample, segment.end_sample + 1)
            cluster_accum[new_pattern].append(np.mean(brightness[sample_slice], axis=0))

        for pattern_id, vectors in cluster_accum.items():
            prototypes[(color, pattern_id)] = np.mean(np.stack(vectors, axis=0), axis=0)

    return segments, prototypes


def state_statistics(
    segments: list[StateSegment],
    frames: list[SampleFrame],
    panels: list[np.ndarray],
    slots: list[grid_probe.Slot],
    brightness: np.ndarray,
    colors: np.ndarray,
    slot_layout: dict[int, tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[int]]]:
    thresholds = np.array([slot.threshold for slot in slots], dtype=np.float32)
    by_state: dict[str, list[int]] = defaultdict(list)
    state_segments: dict[str, list[StateSegment]] = defaultdict(list)
    for segment in segments:
        state_segments[segment.state_key].append(segment)
        for sample_index in range(segment.start_sample, segment.end_sample + 1):
            by_state[segment.state_key].append(sample_index)

    summary_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    issue_map: dict[str, list[int]] = defaultdict(list)
    for state_key, sample_indices in sorted(by_state.items()):
        state_color = state_segments[state_key][0].color
        mean_brightness = np.mean(brightness[sample_indices], axis=0)
        peak_brightness = np.max(brightness[sample_indices], axis=0)
        active_ratio = np.mean(brightness[sample_indices] >= thresholds[None, :], axis=0)
        mean_color = np.mean(colors[sample_indices], axis=0)
        expected_on = active_ratio >= 0.55
        if int(expected_on.sum()) == 0:
            expected_on = peak_brightness >= np.percentile(peak_brightness, 75)

        on_values = mean_brightness[expected_on]
        on_median = float(np.median(on_values)) if len(on_values) else float(np.median(mean_brightness))
        issue_count = 0
        for slot_index, slot in enumerate(slots):
            b_val, g_val, r_val = mean_color[slot_index]
            flags: list[str] = []
            if expected_on[slot_index] and peak_brightness[slot_index] < on_median * 0.45:
                flags.append("missing")
            elif expected_on[slot_index] and mean_brightness[slot_index] < on_median * 0.65:
                flags.append("dim")
            if not expected_on[slot_index] and active_ratio[slot_index] > 0.30 and peak_brightness[slot_index] > max(0.15, on_median * 0.35):
                flags.append("stray")
            if expected_on[slot_index]:
                if state_color == "red" and not (r_val >= g_val * 1.08 and r_val >= b_val * 1.08):
                    flags.append("color")
                elif state_color == "green" and not (g_val >= r_val * 1.05 and g_val >= b_val * 1.05):
                    flags.append("color")
                elif state_color == "blue" and not (b_val >= r_val * 1.05 and b_val >= g_val * 1.05):
                    flags.append("color")
            if flags:
                issue_count += 1
                issue_map[state_key].append(slot.id)
            point_rows.append(
                {
                    "state_key": state_key,
                    "color": state_color,
                    "slot_id": slot.id,
                    "row": slot_layout.get(slot.id, (0, 0))[0],
                    "col": slot_layout.get(slot.id, (0, 0))[1],
                    "x": round(float(slot.x), 2),
                    "y": round(float(slot.y), 2),
                    "mean_brightness": round(float(mean_brightness[slot_index]), 4),
                    "peak_brightness": round(float(peak_brightness[slot_index]), 4),
                    "active_ratio": round(float(active_ratio[slot_index]), 4),
                    "mean_b": round(float(b_val), 2),
                    "mean_g": round(float(g_val), 2),
                    "mean_r": round(float(r_val), 2),
                    "expected_on": int(expected_on[slot_index]),
                    "flags": "|".join(flags),
                }
            )
        summary_rows.append(
            {
                "state_key": state_key,
                "color": state_color,
                "pattern_id": state_segments[state_key][0].pattern_id,
                "segment_count": len(state_segments[state_key]),
                "sample_frame_count": len(sample_indices),
                "start_time_sec": round(min(segment.start_time_sec for segment in state_segments[state_key]), 3),
                "end_time_sec": round(max(segment.end_time_sec for segment in state_segments[state_key]), 3),
                "total_slots": len(slots),
                "expected_on_slots": int(expected_on.sum()),
                "mean_on_brightness": round(float(on_median), 4),
                "suspect_slot_count": issue_count,
            }
        )
    return summary_rows, point_rows, issue_map


def build_state_templates(
    summary_rows: list[dict[str, Any]],
    point_rows: list[dict[str, Any]],
    total_slots: int,
) -> list[dict[str, Any]]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary_by_state = {str(row["state_key"]): row for row in summary_rows}
    for row in point_rows:
        by_state[str(row["state_key"])].append(row)

    templates: list[dict[str, Any]] = []
    for state_key in sorted(by_state):
        rows = sorted(by_state[state_key], key=lambda item: int(item["slot_id"]))
        mask = [0] * total_slots
        on_slots: list[int] = []
        suspect_slots: list[int] = []
        for row in rows:
            slot_id = int(row["slot_id"])
            if int(row["expected_on"]):
                mask[slot_id - 1] = 1
                on_slots.append(slot_id)
            if str(row.get("flags", "")):
                suspect_slots.append(slot_id)
        summary = summary_by_state.get(state_key, {})
        templates.append(
            {
                "state_key": state_key,
                "color": summary.get("color", rows[0]["color"] if rows else ""),
                "pattern_id": int(summary.get("pattern_id", 0) or 0),
                "expected_on_count": len(on_slots),
                "expected_on_slot_ids": on_slots,
                "expected_on_mask": mask,
                "suspect_slot_ids": suspect_slots,
            }
        )
    return templates


def draw_state_overlay(
    panel: np.ndarray,
    slots: list[grid_probe.Slot],
    brightness_row: np.ndarray,
    issue_slots: set[int],
    centers_row: np.ndarray | None,
    output_path: Path,
) -> None:
    vis = panel.copy()
    for slot_index, slot in enumerate(slots):
        if centers_row is not None and slot_index < len(centers_row):
            cx = int(round(float(centers_row[slot_index][0])))
            cy = int(round(float(centers_row[slot_index][1])))
        else:
            cx = int(round(slot.x))
            cy = int(round(slot.y))
        active = brightness_row[slot_index] >= slot.threshold
        if slot.id in issue_slots:
            color = (0, 0, 255)
        elif active:
            color = (0, 255, 0)
        else:
            color = (160, 160, 160)
        cv2.circle(vis, (cx, cy), 6 if active else 4, color, 2)
        cv2.putText(vis, str(slot.id), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def draw_state_heatmap(
    slots: list[grid_probe.Slot],
    values: np.ndarray,
    output_path: Path,
) -> None:
    if not slots:
        return
    width = int(max(slot.x + slot.w for slot in slots) + 24)
    height = int(max(slot.y + slot.h for slot in slots) + 24)
    canvas = np.zeros((height, width), dtype=np.float32)
    for slot, value in zip(slots, values.tolist()):
        cv2.circle(canvas, (int(round(slot.x)), int(round(slot.y))), max(3, int(round(min(slot.w, slot.h) * 0.35))), float(value), -1)
    image = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), color)


def draw_template_overlay(
    panel: np.ndarray,
    matrix_bbox: tuple[int, int, int, int],
    slots: list[grid_probe.Slot],
    output_path: Path,
    label_mode: str,
) -> None:
    vis = panel.copy()
    x, y, w, h = matrix_bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 180, 0), 2)
    for slot in slots:
        cx = int(round(float(slot.x)))
        cy = int(round(float(slot.y)))
        cv2.circle(vis, (cx, cy), 5, (0, 0, 0), 2)
        cv2.circle(vis, (cx, cy), 4, (255, 0, 255), 1)
        draw_label = label_mode == "all"
        if label_mode == "sparse":
            draw_label = slot.id % 10 == 0 or slot.id == 1
        if draw_label:
            origin = (cx + 4, cy - 4)
            cv2.putText(vis, str(slot.id), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, str(slot.id), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 0, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def save_grid_overlay(
    frame: np.ndarray,
    matrix_bbox: tuple[int, int, int, int],
    slots: list[grid_probe.Slot],
    output_path: Path,
) -> None:
    vis = frame.copy()
    x, y, w, h = matrix_bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 255), 2)
    scale_x = w / max(1.0, float(max(slot.x for slot in slots) + max(slot.w for slot in slots) * 0.5))
    scale_y = h / max(1.0, float(max(slot.y for slot in slots) + max(slot.h for slot in slots) * 0.5))
    # Slots are in aligned coordinates; draw on the aligned preview instead of original if scale becomes unstable.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def representative_panel(
    sample_indices: list[int],
    panels: list[np.ndarray],
    brightness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    best_index = sample_indices[0]
    if len(sample_indices) > 1:
        totals = np.sum(brightness[sample_indices], axis=1)
        best_index = sample_indices[int(np.argmax(totals))]
    return panels[best_index], brightness[best_index], best_index


def run_probe(
    video_path: Path,
    output_dir: Path,
    start_sec: float,
    duration_sec: float,
    sample_fps: float,
    panel_size_long_side: int,
    expected_rows: int | None,
    expected_cols: int | None,
    roi: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_frames, fps, meta = sample_video_frames(video_path, start_sec, duration_sec, sample_fps)
    write_json(output_dir / "video_meta.json", meta)

    matrix_quad_src: np.ndarray | None = None
    if roi is None:
        quads = [quad for quad in (detect_matrix_quad(item.frame) for item in sample_frames) if quad is not None]
        if quads:
            matrix_quad_src = median_quad(quads)
            matrix_bbox_src = quad_bbox(matrix_quad_src)
        else:
            boxes = [box for box in (detect_matrix_bbox(item.frame) for item in sample_frames) if box is not None]
            matrix_bbox_src = median_bbox(boxes, (meta["width"], meta["height"]))
    else:
        matrix_bbox_src = roi

    if matrix_quad_src is not None:
        target_size = quad_size(matrix_quad_src, panel_size_long_side)
        per_frame_quads = [detect_matrix_quad(item.frame) for item in sample_frames]
        aligned_panels = [
            warp_panel_quad(item.frame, quad if quad is not None else matrix_quad_src, target_size)
            for item, quad in zip(sample_frames, per_frame_quads)
        ]
    else:
        target_size = standard_size(matrix_bbox_src, panel_size_long_side)
        aligned_panels = [grid_probe.align_panel(item.frame, matrix_bbox_src, target_size) for item in sample_frames]
    matrix_bbox_panel = build_activity_region(aligned_panels)
    slots, grid_info, heatmap, slot_layout = detect_led_layout(aligned_panels, expected_rows, expected_cols, matrix_bbox_panel)
    tracked = measure_slots_with_local_search(aligned_panels, slots)
    brightness = tracked.brightness
    colors = tracked.colors
    thresholds = np.array([slot.threshold for slot in slots], dtype=np.float32)
    for slot_index, slot in enumerate(slots):
        series = brightness[:, slot_index]
        peak = float(np.max(series))
        background = float(np.percentile(series, 20))
        slot.peak = peak
        slot.duty = float(np.mean(series >= slot.threshold))
        slot.threshold = max(0.16, min(0.82, background + (peak - background) * 0.42))
    thresholds = np.array([slot.threshold for slot in slots], dtype=np.float32)

    raw_labels = [classify_frame_color(brightness[index], colors[index], thresholds) for index in range(len(sample_frames))]
    color_labels = smooth_labels(raw_labels)
    segments = contiguous_segments(sample_frames, color_labels, min_length=2)
    segments, prototypes = cluster_segments(segments, brightness, thresholds, max_patterns=4)
    summary_rows, point_rows, issue_map = state_statistics(segments, sample_frames, aligned_panels, slots, brightness, colors, slot_layout)
    state_templates = build_state_templates(summary_rows, point_rows, len(slots))

    sample_dir = output_dir / "sample_frames"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for item in sample_frames[: min(24, len(sample_frames))]:
        cv2.imwrite(str(sample_dir / f"sample_{item.sample_index:04d}_{item.frame_index}.jpg"), item.frame)

    overlay_index = len(aligned_panels) // 2
    overlay_panel = aligned_panels[overlay_index].copy()
    draw_template_overlay(
        overlay_panel,
        matrix_bbox_panel,
        slots,
        output_dir / "overlay_grid.jpg",
        label_mode="all",
    )
    draw_template_overlay(
        overlay_panel,
        matrix_bbox_panel,
        slots,
        output_dir / "overlay_grid_dots.jpg",
        label_mode="none",
    )
    draw_template_overlay(
        overlay_panel,
        matrix_bbox_panel,
        slots,
        output_dir / "overlay_grid_sparse_ids.jpg",
        label_mode="sparse",
    )
    grid_probe.save_heatmap_image(
        heatmap,
        np.array([slot.y - matrix_bbox_panel[1] for slot in slots], dtype=np.int32),
        np.array([slot.x - matrix_bbox_panel[0] for slot in slots], dtype=np.int32),
        output_dir / "grid_heatmap.png",
    )

    segments_payload = []
    by_state: dict[str, list[int]] = defaultdict(list)
    for segment in segments:
        sample_indices = list(range(segment.start_sample, segment.end_sample + 1))
        by_state[segment.state_key].extend(sample_indices)
        segments_payload.append(
            {
                "segment_id": segment.segment_id,
                "state_key": segment.state_key,
                "color": segment.color,
                "pattern_id": segment.pattern_id,
                "start_time_sec": round(segment.start_time_sec, 3),
                "end_time_sec": round(segment.end_time_sec, 3),
                "frame_count": segment.frame_count,
                "prototype_similarity": segment.prototype_similarity,
                "sample_start": segment.start_sample,
                "sample_end": segment.end_sample,
            }
        )
    write_json(output_dir / "state_segments.json", {"segments": segments_payload})
    write_csv(
        output_dir / "state_summary.csv",
        summary_rows,
        [
            "state_key",
            "color",
            "pattern_id",
            "segment_count",
            "sample_frame_count",
            "start_time_sec",
            "end_time_sec",
            "total_slots",
            "expected_on_slots",
            "mean_on_brightness",
            "suspect_slot_count",
        ],
    )
    write_csv(
        output_dir / "points_by_state.csv",
        point_rows,
        [
            "state_key",
            "color",
            "slot_id",
            "row",
            "col",
            "x",
            "y",
            "mean_brightness",
            "peak_brightness",
            "active_ratio",
            "mean_b",
            "mean_g",
            "mean_r",
            "expected_on",
            "flags",
        ],
    )

    for state_key, sample_indices in sorted(by_state.items()):
        panel, brightness_row, sample_index = representative_panel(sample_indices, aligned_panels, brightness)
        issue_slots = set(issue_map.get(state_key, []))
        draw_state_overlay(
            panel,
            slots,
            brightness_row,
            issue_slots,
            tracked.centers[sample_index],
            output_dir / f"{state_key}_overlay.jpg",
        )
        mean_values = np.mean(brightness[sample_indices], axis=0)
        draw_state_heatmap(slots, mean_values, output_dir / f"{state_key}_heatmap.png")

    template_payload = {
        "type": "matrix_12state_template",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_video": str(video_path),
        "sample_window": {
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "sample_fps": sample_fps,
        },
        "source_frame_size": [meta["width"], meta["height"]],
        "matrix_bbox_source": list(matrix_bbox_src),
        "matrix_quad_source": matrix_quad_src.tolist() if matrix_quad_src is not None else None,
        "aligned_panel_size": list(target_size),
        "matrix_bbox_aligned": list(matrix_bbox_panel),
        "grid_info": {
            "rows": int(grid_info.get("rows", 0)),
            "cols": int(grid_info.get("cols", 0)),
            "total_slots": len(slots),
        },
        "slots": [
            {
                "id": slot.id,
                "row": slot_layout.get(slot.id, (0, 0))[0],
                "col": slot_layout.get(slot.id, (0, 0))[1],
                "x": round(float(slot.x), 3),
                "y": round(float(slot.y), 3),
                "w": round(float(slot.w), 3),
                "h": round(float(slot.h), 3),
                "threshold": round(float(slot.threshold), 4),
                "peak": round(float(slot.peak), 4),
                "duty": round(float(slot.duty), 4),
            }
            for slot in slots
        ],
        "state_catalog": [
            {
                "state_key": row["state_key"],
                "color": row["color"],
                "pattern_id": row["pattern_id"],
                "segment_count": row["segment_count"],
                "sample_frame_count": row["sample_frame_count"],
                "suspect_slot_count": row["suspect_slot_count"],
            }
            for row in summary_rows
        ],
        "state_templates": state_templates,
    }
    write_json(output_dir / "matrix_template.json", template_payload)

    result = {
        "video": str(video_path),
        "output_dir": str(output_dir),
        "sampled_frames": len(sample_frames),
        "grid": f"{grid_info.get('rows', '?')}x{grid_info.get('cols', '?')}",
        "slots": len(slots),
        "states_detected": len(summary_rows),
        "colors_detected": sorted({row["color"] for row in summary_rows}),
        "state_keys": [row["state_key"] for row in summary_rows],
    }
    write_json(output_dir / "probe_summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="新机型 12 状态点阵脚本验证")
    parser.add_argument("--video", type=Path, required=True, help="输入视频路径")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=180.0, help="采样时长，0 表示全视频")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="视频抽样帧率")
    parser.add_argument("--panel-long-side", type=int, default=DEFAULT_LONG_SIDE)
    parser.add_argument("--rows", type=int, default=None, help="可选：固定网格行数")
    parser.add_argument("--cols", type=int, default=None, help="可选：固定网格列数")
    parser.add_argument("--roi", type=parse_bbox, default=None, help="可选：原图坐标 roi x,y,w,h")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_probe(
        video_path=args.video,
        output_dir=args.output_dir,
        start_sec=float(args.start_sec),
        duration_sec=float(args.duration_sec),
        sample_fps=float(args.sample_fps),
        panel_size_long_side=int(args.panel_long_side),
        expected_rows=args.rows,
        expected_cols=args.cols,
        roi=args.roi,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
