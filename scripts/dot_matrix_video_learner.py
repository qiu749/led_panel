from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_direct_point_templates import detect_points
from demo_led_slot_array import (
    compute_grid_fit,
    decide_layout,
    jsonable,
    make_grouped_slots,
    make_regular_slots,
    points_to_array,
    write_json,
)


@dataclass
class FrameItem:
    index: int
    time_sec: float
    image: np.ndarray
    source: str
    roi_offset: tuple[int, int] = (0, 0)


@dataclass
class Segment:
    start: int
    end: int
    mask: np.ndarray
    frame_count: int


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_frames(
    video: Path | None,
    frames_dir: Path | None,
    sample_fps: float,
    max_frames: int,
) -> tuple[list[FrameItem], dict[str, Any]]:
    if video is None and frames_dir is None:
        raise ValueError("Use --video or --frames-dir")
    if video is not None and frames_dir is not None:
        raise ValueError("Use only one of --video or --frames-dir")

    frames: list[FrameItem] = []
    if frames_dir is not None:
        paths: list[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(frames_dir.glob(pattern))
        paths = sorted(paths)
        if not paths:
            raise FileNotFoundError(f"No images found in {frames_dir}")
        stride = 1
        if max_frames > 0 and len(paths) > max_frames:
            stride = max(1, int(math.floor(len(paths) / max_frames)))
        for out_index, path in enumerate(paths[::stride]):
            image = cv2.imread(str(path))
            if image is None:
                continue
            frames.append(FrameItem(out_index, float(out_index), image, str(path)))
            if max_frames > 0 and len(frames) >= max_frames:
                break
        if not frames:
            raise RuntimeError(f"No readable images found in {frames_dir}")
        meta = {
            "source_type": "frames_dir",
            "source": str(frames_dir),
            "source_fps": 0.0,
            "sample_fps": 0.0,
            "frame_count": len(frames),
            "width": int(frames[0].image.shape[1]),
            "height": int(frames[0].image.shape[0]),
        }
        return frames, meta

    assert video is not None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, int(round(source_fps / max(sample_fps, 0.1))))
    idx = 0
    while True:
        ok, image = cap.read()
        if not ok or image is None:
            break
        if idx % stride == 0:
            frames.append(FrameItem(idx, idx / max(source_fps, 1e-6), image, str(video)))
            if max_frames > 0 and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames sampled from {video}")
    meta = {
        "source_type": "video",
        "source": str(video),
        "source_fps": source_fps,
        "sample_fps": float(sample_fps),
        "frame_count": total,
        "sampled_count": len(frames),
        "width": width,
        "height": height,
        "sample_stride": stride,
    }
    return frames, meta


def detect_panel_bbox(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    bright = (blur >= max(20, int(np.percentile(blur, 99.2)))).astype(np.uint8) * 255
    min_side = min(gray.shape[:2])
    close_size = max(7, int(round(min_side / 55)))
    dilate_size = max(9, int(round(min_side / 75)))
    if close_size % 2 == 0:
        close_size += 1
    if dilate_size % 2 == 0:
        dilate_size += 1
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    bright = cv2.dilate(bright, np.ones((dilate_size, dilate_size), np.uint8), iterations=1)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    height, width = gray.shape
    frame_area = float(width * height)
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < frame_area * 0.001 or area > frame_area * 0.5:
            continue
        aspect = w / max(h, 1)
        if not 0.35 <= aspect <= 4.5:
            continue
        extent = cv2.contourArea(contour) / max(area, 1.0)
        score = area * max(extent, 0.2)
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))
    if best is not None:
        return best[1]

    ys, xs = np.where(bright > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)
    area = float(w * h)
    aspect = w / max(h, 1)
    if frame_area * 0.001 <= area <= frame_area * 0.5 and 0.35 <= aspect <= 4.5:
        return x1, y1, w, h
    return None


def median_panel_bbox(frames: list[FrameItem], padding_ratio: float) -> tuple[int, int, int, int] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for frame in frames:
        bbox = detect_panel_bbox(frame.image)
        if bbox is not None:
            boxes.append(bbox)
    if not boxes:
        return None
    arr = np.asarray(boxes, dtype=np.float32)
    x, y, w, h = np.median(arr, axis=0)
    img_h, img_w = frames[0].image.shape[:2]
    pad_x = int(round(w * padding_ratio))
    pad_y = int(round(h * padding_ratio))
    x1 = max(0, int(round(x)) - pad_x)
    y1 = max(0, int(round(y)) - pad_y)
    x2 = min(img_w, int(round(x + w)) + pad_x)
    y2 = min(img_h, int(round(y + h)) + pad_y)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def crop_frames_to_bbox(frames: list[FrameItem], bbox: tuple[int, int, int, int]) -> list[FrameItem]:
    x, y, w, h = bbox
    cropped: list[FrameItem] = []
    for frame in frames:
        crop = frame.image[y:y + h, x:x + w]
        cropped.append(FrameItem(frame.index, frame.time_sec, crop, frame.source, (x, y)))
    return cropped


def alignment_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.6)
    floor = float(np.percentile(gray, 25))
    ceil = float(np.percentile(gray, 99))
    if ceil <= floor + 1:
        return np.zeros_like(gray, dtype=np.float32)
    norm = np.clip((gray - floor) / (ceil - floor), 0.0, 1.0)
    return norm.astype(np.float32)


def build_projection_image(frames: list[FrameItem]) -> np.ndarray:
    if not frames:
        raise ValueError("No frames for projection")
    stack = np.stack([frame.image for frame in frames], axis=0)
    return np.max(stack, axis=0).astype(np.uint8)


def find_profile_peaks(profile: np.ndarray, min_distance: int, threshold_ratio: float) -> list[int]:
    if len(profile) < 3:
        return []
    values = profile.astype(np.float32)
    values = values - float(values.min())
    peak = float(values.max())
    if peak <= 1e-6:
        return []
    values = values / peak
    kernel_size = max(3, int(round(min_distance * 0.55)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size,), dtype=np.float32) / float(kernel_size)
    smooth = np.convolve(values, kernel, mode="same")
    peaks: list[int] = []
    for idx in range(1, len(smooth) - 1):
        if smooth[idx] < threshold_ratio:
            continue
        if smooth[idx] < smooth[idx - 1] or smooth[idx] < smooth[idx + 1]:
            continue
        if peaks and idx - peaks[-1] < min_distance:
            if smooth[idx] > smooth[peaks[-1]]:
                peaks[-1] = idx
        else:
            peaks.append(idx)
    return peaks


def projection_grid_points(image: np.ndarray, threshold_ratio: float) -> tuple[np.ndarray, dict[str, Any]]:
    value = np.max(image, axis=2).astype(np.float32)
    bright_threshold = max(20.0, float(np.percentile(value, 99)) * 0.35)
    mask = value >= bright_threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32), {"reason": "no_bright_pixels"}

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    crop = value[y1:y2 + 1, x1:x2 + 1]
    crop_mask = mask[y1:y2 + 1, x1:x2 + 1]
    weighted = crop * crop_mask.astype(np.float32)
    x_profile = weighted.mean(axis=0)
    y_profile = weighted.mean(axis=1)

    min_x_distance = max(5, int(round(crop.shape[1] / 85)))
    min_y_distance = max(5, int(round(crop.shape[0] / 85)))
    x_peaks = find_profile_peaks(x_profile, min_x_distance, threshold_ratio)
    y_peaks = find_profile_peaks(y_profile, min_y_distance, threshold_ratio)
    if len(x_peaks) < 2 or len(y_peaks) < 2:
        return np.empty((0, 2), dtype=np.float32), {
            "reason": "not_enough_peaks",
            "x_peak_count": len(x_peaks),
            "y_peak_count": len(y_peaks),
        }

    points = np.asarray(
        [(float(x1 + x), float(y1 + y)) for y in y_peaks for x in x_peaks],
        dtype=np.float32,
    )
    return points, {
        "bbox": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
        "x_peak_count": len(x_peaks),
        "y_peak_count": len(y_peaks),
        "point_count": int(len(points)),
        "bright_threshold": round(float(bright_threshold), 3),
        "threshold_ratio": float(threshold_ratio),
    }


def estimate_translation_ncc(image: np.ndarray, reference: np.ndarray, max_shift: int) -> tuple[int, int, float]:
    height, width = image.shape[:2]
    best: tuple[int, int, float] = (0, 0, -1.0)
    for dy in range(-max_shift, max_shift + 1):
        out_y1 = max(0, dy)
        out_y2 = min(height, height + dy)
        src_y1 = out_y1 - dy
        src_y2 = out_y2 - dy
        if out_y2 - out_y1 < max(16, height // 3):
            continue
        for dx in range(-max_shift, max_shift + 1):
            out_x1 = max(0, dx)
            out_x2 = min(width, width + dx)
            src_x1 = out_x1 - dx
            src_x2 = out_x2 - dx
            if out_x2 - out_x1 < max(16, width // 3):
                continue
            a = image[src_y1:src_y2, src_x1:src_x2]
            b = reference[out_y1:out_y2, out_x1:out_x2]
            a_std = float(a.std())
            b_std = float(b.std())
            if a_std < 1e-4 or b_std < 1e-4:
                continue
            score = float(np.mean(((a - float(a.mean())) / a_std) * ((b - float(b.mean())) / b_std)))
            if score > best[2]:
                best = (dx, dy, score)
    return best


def estimate_translation_phase(image: np.ndarray, reference: np.ndarray, max_shift: int) -> tuple[int, int, float]:
    if image.shape != reference.shape:
        return 0, 0, 0.0
    window = cv2.createHanningWindow((image.shape[1], image.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(reference * window, image * window)
    dx = int(round(-float(shift[0])))
    dy = int(round(-float(shift[1])))
    dx = max(-max_shift, min(max_shift, dx))
    dy = max(-max_shift, min(max_shift, dy))
    return dx, dy, float(response)


def shift_frame(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return frame.copy()
    matrix = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def align_frames_translation(
    frames: list[FrameItem],
    max_shift: int,
    reference_image: np.ndarray | None = None,
    method: str = "phase",
) -> tuple[list[FrameItem], list[dict[str, Any]], np.ndarray]:
    if not frames or max_shift <= 0:
        reference = build_projection_image(frames) if frames else np.empty((0, 0, 3), dtype=np.uint8)
        return [FrameItem(f.index, f.time_sec, f.image.copy(), f.source, f.roi_offset) for f in frames], [], reference

    if reference_image is None:
        reference_image = build_projection_image(frames)
    reference = alignment_image(reference_image)
    aligned: list[FrameItem] = []
    rows: list[dict[str, Any]] = []
    for frame in frames:
        current = alignment_image(frame.image)
        if method == "ncc":
            dx, dy, score = estimate_translation_ncc(current, reference, max_shift)
        else:
            dx, dy, score = estimate_translation_phase(current, reference, max_shift)
        shifted = shift_frame(frame.image, dx, dy)
        aligned.append(FrameItem(frame.index, frame.time_sec, shifted, frame.source, frame.roi_offset))
        rows.append({"frame_index": frame.index, "time_sec": round(frame.time_sec, 4), "dx": dx, "dy": dy, "score": round(score, 5)})
    return aligned, rows, reference_image


def detect_points_scaled(image: np.ndarray, scale: float) -> tuple[np.ndarray, dict[str, float]]:
    if scale <= 1.0:
        points, stats = detect_points(image)
        return points_to_array(points), stats
    scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    points, stats = detect_points(scaled)
    arr = points_to_array(points)
    if len(arr):
        arr = arr / float(scale)
    stats = dict(stats)
    stats["point_detect_scale"] = float(scale)
    return arr, stats


def detect_blob_points_scaled(image: np.ndarray, scale: float) -> tuple[np.ndarray, dict[str, float]]:
    work = image
    if scale > 1.0:
        work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    value = np.max(work, axis=2).astype(np.uint8)
    blur = cv2.GaussianBlur(value, (0, 0), max(0.8, 0.45 * scale))
    otsu_threshold, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(18, int(round(float(otsu_threshold))), int(round(float(np.percentile(blur, 92)) * 0.80)))
    mask = (blur >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)

    components = stats[1:] if count > 1 else np.empty((0, 5), dtype=np.int32)
    valid_areas = components[:, cv2.CC_STAT_AREA].astype(np.float32) if len(components) else np.empty((0,), dtype=np.float32)
    valid_areas = valid_areas[(valid_areas >= max(2.0, scale * scale * 0.4)) & (valid_areas <= scale * scale * 500.0)]
    if len(valid_areas):
        median_area = float(np.median(valid_areas))
        min_area = max(1.0, median_area * 0.18)
        max_area = max(min_area + 1.0, median_area * 4.0)
    else:
        min_area = max(1.0, scale * scale * 0.4)
        max_area = scale * scale * 500.0

    max_dim = max(5.0, scale * 42.0)
    points: list[tuple[float, float]] = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < min_area or area > max_area:
            continue
        if max(w, h) > max_dim:
            continue
        points.append((float(centers[index][0]) / max(scale, 1.0), float(centers[index][1]) / max(scale, 1.0)))
    return points_to_array(points), {
        "threshold": float(threshold),
        "raw_count": float(len(points)),
        "min_area": float(min_area),
        "max_area": float(max_area),
        "point_detect_scale": float(scale),
    }


def detect_slot_points(image: np.ndarray, detector: str, scale: float) -> tuple[np.ndarray, dict[str, float]]:
    if detector == "blob":
        return detect_blob_points_scaled(image, scale)
    return detect_points_scaled(image, scale)


def cluster_union_points(points_by_frame: list[np.ndarray], merge_radius: float, min_observations: int) -> np.ndarray:
    all_points = [p for points in points_by_frame for p in points]
    if not all_points:
        return np.empty((0, 2), dtype=np.float32)

    centers: list[np.ndarray] = []
    counts: list[int] = []
    grid: dict[tuple[int, int], list[int]] = {}
    cell_size = max(float(merge_radius), 1.0)

    def cell_for(point: np.ndarray) -> tuple[int, int]:
        return int(math.floor(float(point[0]) / cell_size)), int(math.floor(float(point[1]) / cell_size))

    for point in sorted(all_points, key=lambda p: (float(p[1]), float(p[0]))):
        p = np.asarray(point, dtype=np.float32)
        if not centers:
            centers.append(p.copy())
            counts.append(1)
            grid.setdefault(cell_for(p), []).append(0)
            continue

        cx, cy = cell_for(p)
        candidates: list[int] = []
        for gx in range(cx - 1, cx + 2):
            for gy in range(cy - 1, cy + 2):
                candidates.extend(grid.get((gx, gy), []))

        idx = -1
        best_distance = float("inf")
        for candidate in candidates:
            distance = float(np.linalg.norm(centers[candidate] - p))
            if distance < best_distance:
                best_distance = distance
                idx = candidate

        if idx >= 0 and best_distance <= merge_radius:
            old_cell = cell_for(centers[idx])
            counts[idx] += 1
            centers[idx] = centers[idx] + (p - centers[idx]) / counts[idx]
            new_cell = cell_for(centers[idx])
            if new_cell != old_cell:
                if old_cell in grid and idx in grid[old_cell]:
                    grid[old_cell].remove(idx)
                grid.setdefault(new_cell, []).append(idx)
        else:
            centers.append(p.copy())
            counts.append(1)
            grid.setdefault(cell_for(p), []).append(len(centers) - 1)
    kept = [center for center, count in zip(centers, counts) if count >= min_observations]
    return np.asarray(kept, dtype=np.float32)


def build_slots_from_points(points: np.ndarray, width: int, height: int) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fit = compute_grid_fit(points, width, height)
    layout_mode = decide_layout(fit, "auto")
    if layout_mode in {"regular_grid", "staggered_grid"}:
        slots = make_regular_slots(points, fit)
        groups = [
            {
                "id": layout_mode,
                "slot_count": len(slots),
                "rows": len(fit.y_axis.centers),
                "cols": len(fit.x_axis.centers),
            }
        ]
    else:
        slots, groups = make_grouped_slots(points, fit)
    return layout_mode, slots, groups, {
        "regularity_score": round(float(fit.regularity_score), 5),
        "fill_ratio": round(float(fit.fill_ratio), 5),
        "occupied_count": int(fit.occupied_count),
        "expected_count": int(fit.expected_count),
        "median_residual": round(float(fit.median_residual), 5),
        "x": asdict(fit.x_axis),
        "y": asdict(fit.y_axis),
    }


def offset_slots(slots: list[dict[str, Any]], dx: int, dy: int) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for slot in slots:
        item = dict(slot)
        item["x"] = round(float(item["x"]) + dx, 3)
        item["y"] = round(float(item["y"]) + dy, 3)
        shifted.append(item)
    return shifted


def slot_score_image(image: np.ndarray, channel: str) -> np.ndarray:
    if channel == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channel == "red":
        return image[:, :, 2]
    if channel == "green":
        return image[:, :, 1]
    if channel == "blue":
        return image[:, :, 0]
    return np.max(image, axis=2).astype(np.uint8)


def slot_scores(image: np.ndarray, slots: list[dict[str, Any]], radius: int, channel: str = "max") -> np.ndarray:
    score_img = slot_score_image(image, channel)
    scores = np.zeros((len(slots),), dtype=np.float32)
    for index, slot in enumerate(slots):
        x = int(round(float(slot["x"])))
        y = int(round(float(slot["y"])))
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(score_img.shape[1], x + radius + 1)
        y2 = min(score_img.shape[0], y + radius + 1)
        roi = score_img[y1:y2, x1:x2]
        scores[index] = float(np.percentile(roi, 95)) if roi.size else 0.0
    return scores


def score_matrix(frames: list[FrameItem], slots: list[dict[str, Any]], radius: int, channel: str = "max") -> np.ndarray:
    return np.vstack([slot_scores(frame.image, slots, radius, channel) for frame in frames]).astype(np.float32)


def learn_slot_thresholds(scores: np.ndarray, min_peak_delta: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if len(scores) == 0:
        return np.empty((0,), dtype=np.float32), []
    thresholds = np.zeros((scores.shape[1],), dtype=np.float32)
    stats: list[dict[str, Any]] = []
    for idx in range(scores.shape[1]):
        series = scores[:, idx]
        bg = float(np.percentile(series, 15))
        peak = float(np.percentile(series, 95))
        delta = max(peak - bg, min_peak_delta)
        threshold = bg + delta * 0.45
        thresholds[idx] = float(threshold)
        stats.append(
            {
                "slot_id": idx + 1,
                "background": round(bg, 3),
                "peak": round(peak, 3),
                "threshold": round(float(threshold), 3),
            }
        )
    return thresholds, stats


def masks_from_scores(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return np.empty((0, 0), dtype=bool)
    return scores >= thresholds[None, :]


def fuse_masks(masks: np.ndarray, window_size: int, min_on_ratio: float) -> np.ndarray:
    if len(masks) == 0 or window_size <= 1:
        return masks.astype(bool)
    radius = window_size // 2
    fused = np.zeros_like(masks, dtype=bool)
    for idx in range(len(masks)):
        start = max(0, idx - radius)
        end = min(len(masks), idx + radius + 1)
        fused[idx] = masks[start:end].mean(axis=0) >= min_on_ratio
    return fused


def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b):
        return 0.0
    if len(a) == 0:
        return 1.0
    return 1.0 - float(np.count_nonzero(a != b)) / float(len(a))


def segment_masks(masks: np.ndarray, dedupe_similarity: float, min_segment_frames: int) -> list[Segment]:
    if len(masks) == 0:
        return []
    raw: list[Segment] = []
    start = 0
    current = masks[0].copy()
    members = [masks[0]]
    for idx in range(1, len(masks)):
        if hamming_similarity(current, masks[idx]) >= dedupe_similarity:
            members.append(masks[idx])
            current = np.asarray(members).mean(axis=0) >= 0.5
            continue
        segment_mask = np.asarray(members).mean(axis=0) >= 0.5
        raw.append(Segment(start, idx - 1, segment_mask, idx - start))
        start = idx
        current = masks[idx].copy()
        members = [masks[idx]]
    raw.append(Segment(start, len(masks) - 1, np.asarray(members).mean(axis=0) >= 0.5, len(masks) - start))
    return [segment for segment in raw if segment.frame_count >= min_segment_frames]


def assign_key_states(segments: list[Segment], state_similarity: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prototypes: list[np.ndarray] = []
    states: list[dict[str, Any]] = []
    sequence: list[dict[str, Any]] = []
    durations: dict[str, list[int]] = {}
    for segment in segments:
        best_idx = -1
        best_score = -1.0
        for idx, proto in enumerate(prototypes):
            sim = hamming_similarity(segment.mask, proto)
            if sim > best_score:
                best_idx = idx
                best_score = sim
        if best_idx >= 0 and best_score >= state_similarity:
            sid = states[best_idx]["id"]
            # Incremental prototype update so mild flicker does not create duplicate states.
            prototypes[best_idx] = ((prototypes[best_idx].astype(np.float32) + segment.mask.astype(np.float32)) / 2.0) >= 0.5
        else:
            sid = f"K{len(states):03d}"
            prototypes.append(segment.mask.copy())
            states.append({"id": sid, "index": len(states), "on_slot_ids": [], "on_count": 0})
        durations.setdefault(sid, []).append(segment.frame_count)
        if not sequence or sequence[-1]["state_id"] != sid:
            sequence.append(
                {
                    "state_id": sid,
                    "start_segment": len(sequence),
                    "first_frame": int(segment.start),
                    "last_frame": int(segment.end),
                    "duration_frames": int(segment.frame_count),
                }
            )
        else:
            sequence[-1]["last_frame"] = int(segment.end)
            sequence[-1]["duration_frames"] += int(segment.frame_count)

    for idx, proto in enumerate(prototypes):
        on_ids = (np.flatnonzero(proto) + 1).astype(int).tolist()
        sid = states[idx]["id"]
        ds = durations.get(sid, [1])
        states[idx].update(
            {
                "on_slot_ids": on_ids,
                "on_count": len(on_ids),
                "median_duration_frames": float(np.median(ds)),
                "min_duration_frames": int(np.min(ds)),
                "max_duration_frames": int(np.max(ds)),
            }
        )
    return states, sequence


def state_to_mask(state: dict[str, Any], slot_count: int) -> np.ndarray:
    mask = np.zeros((slot_count,), dtype=bool)
    ids = np.asarray(state.get("on_slot_ids", []), dtype=np.int32)
    ids = ids[(ids >= 1) & (ids <= slot_count)]
    mask[ids - 1] = True
    return mask


def build_logical_states(
    states: list[dict[str, Any]],
    sequence: list[dict[str, Any]],
    slot_count: int,
    window_states: int,
    similarity: float,
    fill_dense_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not states or not sequence or window_states <= 1:
        return [], []

    state_by_id = {str(state.get("id", "")): state for state in states}
    prototypes: list[np.ndarray] = []
    logical_states: list[dict[str, Any]] = []
    logical_sequence: list[dict[str, Any]] = []

    for start in range(0, len(sequence)):
        members = sequence[start:start + window_states]
        if len(members) < window_states:
            break
        union = np.zeros((slot_count,), dtype=bool)
        member_ids: list[str] = []
        for item in members:
            sid = str(item.get("state_id", ""))
            state = state_by_id.get(sid)
            if state is None:
                continue
            union |= state_to_mask(state, slot_count)
            member_ids.append(sid)
        if not member_ids:
            continue
        if fill_dense_ratio > 0 and float(np.mean(union)) >= fill_dense_ratio:
            union[:] = True

        best_idx = -1
        best_score = -1.0
        for idx, proto in enumerate(prototypes):
            score = hamming_similarity(union, proto)
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx >= 0 and best_score >= similarity:
            logical_id = logical_states[best_idx]["id"]
            prototypes[best_idx] = ((prototypes[best_idx].astype(np.float32) + union.astype(np.float32)) / 2.0) >= 0.5
        else:
            logical_id = f"L{len(logical_states):03d}"
            prototypes.append(union.copy())
            logical_states.append({"id": logical_id, "index": len(logical_states), "scan_members": member_ids})

        if not logical_sequence or logical_sequence[-1]["state_id"] != logical_id:
            logical_sequence.append(
                {
                    "state_id": logical_id,
                    "first_sequence_index": start,
                    "last_sequence_index": start + window_states - 1,
                    "scan_members": member_ids,
                }
            )
        else:
            logical_sequence[-1]["last_sequence_index"] = start + window_states - 1

    for idx, proto in enumerate(prototypes):
        on_ids = (np.flatnonzero(proto) + 1).astype(int).tolist()
        logical_states[idx].update({"on_slot_ids": on_ids, "on_count": len(on_ids)})

    return logical_states, logical_sequence


def bgr_to_lab(color: np.ndarray) -> np.ndarray:
    sample = np.asarray(color, dtype=np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(sample, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)


def slot_color_matrix(frames: list[FrameItem], slots: list[dict[str, Any]], radius: int) -> np.ndarray:
    colors = np.zeros((len(frames), len(slots), 3), dtype=np.float32)
    for frame_index, frame in enumerate(frames):
        image = frame.image
        value = np.max(image, axis=2)
        for slot_index, slot in enumerate(slots):
            x = int(round(float(slot["x"])))
            y = int(round(float(slot["y"])))
            x1 = max(0, x - radius)
            y1 = max(0, y - radius)
            x2 = min(image.shape[1], x + radius + 1)
            y2 = min(image.shape[0], y + radius + 1)
            roi = image[y1:y2, x1:x2]
            roi_value = value[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            cutoff = float(np.percentile(roi_value, 80))
            pixels = roi[roi_value >= cutoff]
            if len(pixels) == 0:
                pixels = roi.reshape(-1, 3)
            colors[frame_index, slot_index] = pixels.astype(np.float32).mean(axis=0)
    return colors


def assign_frame_color_labels(
    frame_colors: np.ndarray,
    max_clusters: int,
    distance_threshold: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    labels = np.full((len(frame_colors),), -1, dtype=np.int32)
    centers_lab: list[np.ndarray] = []
    centers_bgr: list[np.ndarray] = []
    counts: list[int] = []

    for idx, color in enumerate(frame_colors):
        lab = bgr_to_lab(color)
        if not centers_lab:
            centers_lab.append(lab)
            centers_bgr.append(color.astype(np.float32))
            counts.append(1)
            labels[idx] = 0
            continue

        distances = [float(np.linalg.norm(lab - center)) for center in centers_lab]
        best = int(np.argmin(distances))
        if distances[best] > distance_threshold and len(centers_lab) < max_clusters:
            centers_lab.append(lab)
            centers_bgr.append(color.astype(np.float32))
            counts.append(1)
            labels[idx] = len(centers_lab) - 1
        else:
            labels[idx] = best
            counts[best] += 1
            centers_bgr[best] = centers_bgr[best] + (color.astype(np.float32) - centers_bgr[best]) / counts[best]
            centers_lab[best] = bgr_to_lab(centers_bgr[best])
    return labels, centers_bgr


def color_segments(labels: np.ndarray, min_frames: int) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    current = int(labels[0])
    for idx in range(1, len(labels)):
        label = int(labels[idx])
        if label == current:
            continue
        if current >= 0 and idx - start >= min_frames:
            segments.append((start, idx - 1, current))
        start = idx
        current = label
    if current >= 0 and len(labels) - start >= min_frames:
        segments.append((start, len(labels) - 1, current))
    return segments


def build_display_states(
    frames: list[FrameItem],
    slots: list[dict[str, Any]],
    masks: np.ndarray,
    score_radius: int,
    max_color_states: int,
    color_distance: float,
    min_segment_frames: int,
    fill_dense_ratio: float,
    state_similarity: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(frames) == 0 or len(slots) == 0 or len(masks) == 0:
        return [], []

    colors = slot_color_matrix(frames, slots, score_radius)
    frame_colors = np.zeros((len(frames), 3), dtype=np.float32)
    for idx, mask in enumerate(masks):
        active = np.flatnonzero(mask)
        if len(active):
            frame_colors[idx] = colors[idx, active].mean(axis=0)
        else:
            frame_colors[idx] = colors[idx].mean(axis=0)

    labels, _ = assign_frame_color_labels(frame_colors, max_color_states, color_distance)
    segments = color_segments(labels, min_segment_frames)

    prototypes: list[dict[str, Any]] = []
    display_states: list[dict[str, Any]] = []
    display_sequence: list[dict[str, Any]] = []

    for start, end, label in segments:
        segment_masks = masks[start:end + 1]
        union = segment_masks.any(axis=0)
        dense_filled = False
        if fill_dense_ratio > 0 and float(np.mean(union)) >= fill_dense_ratio:
            union[:] = True
            dense_filled = True

        segment_colors = colors[start:end + 1]
        slot_colors = np.zeros((len(slots), 3), dtype=np.float32)
        segment_mean_color = frame_colors[start:end + 1].mean(axis=0)
        for slot_index in range(len(slots)):
            active = segment_masks[:, slot_index]
            if np.any(active):
                slot_colors[slot_index] = segment_colors[active, slot_index].mean(axis=0)
            else:
                slot_colors[slot_index] = segment_mean_color

        active_slot_colors = slot_colors[union]
        mean_bgr = active_slot_colors.mean(axis=0) if len(active_slot_colors) else segment_mean_color
        mean_lab = bgr_to_lab(mean_bgr)

        best_idx = -1
        best_score = -1.0
        for idx, proto in enumerate(prototypes):
            mask_score = hamming_similarity(union, proto["mask"])
            color_score = float(np.linalg.norm(mean_lab - proto["lab"]))
            merged_mask = np.logical_or(union, proto["mask"])
            merged_dense = fill_dense_ratio > 0 and float(np.mean(merged_mask)) >= fill_dense_ratio
            if color_score <= color_distance and (mask_score >= state_similarity or merged_dense) and mask_score > best_score:
                best_idx = idx
                best_score = mask_score

        if best_idx >= 0:
            display_id = display_states[best_idx]["id"]
            merged_mask = np.logical_or(prototypes[best_idx]["mask"], union)
            if fill_dense_ratio > 0 and float(np.mean(merged_mask)) >= fill_dense_ratio:
                merged_mask[:] = True
                display_states[best_idx]["dense_filled"] = True
            prototypes[best_idx]["mask"] = merged_mask
            prototypes[best_idx]["bgr"] = (prototypes[best_idx]["bgr"] + mean_bgr) / 2.0
            prototypes[best_idx]["lab"] = bgr_to_lab(prototypes[best_idx]["bgr"])
            on_ids = (np.flatnonzero(merged_mask) + 1).astype(int).tolist()
            display_states[best_idx]["on_slot_ids"] = on_ids
            display_states[best_idx]["on_count"] = len(on_ids)
            merged_bgr = prototypes[best_idx]["bgr"]
            display_states[best_idx]["mean_rgb"] = [
                int(round(float(merged_bgr[2]))),
                int(round(float(merged_bgr[1]))),
                int(round(float(merged_bgr[0]))),
            ]
        else:
            display_id = f"D{len(display_states):03d}"
            prototypes.append({"mask": union.copy(), "bgr": mean_bgr.astype(np.float32), "lab": mean_lab})
            on_ids = (np.flatnonzero(union) + 1).astype(int).tolist()
            display_states.append(
                {
                    "id": display_id,
                    "index": len(display_states),
                    "on_slot_ids": on_ids,
                    "on_count": len(on_ids),
                    "mean_rgb": [int(round(float(mean_bgr[2]))), int(round(float(mean_bgr[1]))), int(round(float(mean_bgr[0])))],
                    "dense_filled": dense_filled,
                    "slot_colors": [
                        {
                            "id": int(slot_index + 1),
                            "rgb": [
                                int(round(float(slot_colors[slot_index, 2]))),
                                int(round(float(slot_colors[slot_index, 1]))),
                                int(round(float(slot_colors[slot_index, 0]))),
                            ],
                        }
                        for slot_index in np.flatnonzero(union)
                    ],
                }
            )

        if not display_sequence or display_sequence[-1]["state_id"] != display_id:
            display_sequence.append(
                {
                    "state_id": display_id,
                    "first_frame": int(frames[start].index),
                    "last_frame": int(frames[end].index),
                    "duration_frames": int(end - start + 1),
                    "color_label": int(label),
                }
            )
        else:
            display_sequence[-1]["last_frame"] = int(frames[end].index)
            display_sequence[-1]["duration_frames"] += int(end - start + 1)

    return display_states, display_sequence


def build_scan_phase_templates(slots: list[dict[str, Any]], states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not slots:
        return []
    groups: dict[int, list[int]] = {0: [], 1: []}
    for slot in slots:
        row = int(slot.get("row", 0))
        col = int(slot.get("col", 0))
        groups[(row + col) % 2].append(int(slot["id"]))

    templates: list[dict[str, Any]] = []
    for parity in (0, 1):
        ids = sorted(groups[parity])
        id_set = set(ids)
        best_state = ""
        best_recall = 0.0
        best_precision = 0.0
        for state in states:
            state_ids = set(int(v) for v in state.get("on_slot_ids", []))
            if not state_ids:
                continue
            hit = len(state_ids & id_set)
            recall = hit / max(len(id_set), 1)
            precision = hit / max(len(state_ids), 1)
            if recall * precision > best_recall * best_precision:
                best_state = str(state.get("id", ""))
                best_recall = recall
                best_precision = precision
        templates.append(
            {
                "id": f"P{parity:03d}",
                "name": f"parity_{parity}",
                "slot_count": len(ids),
                "on_count": len(ids),
                "on_slot_ids": ids,
                "rule": "(row + col) % 2",
                "parity": parity,
                "best_raw_state": best_state,
                "best_raw_recall": round(float(best_recall), 5),
                "best_raw_precision": round(float(best_precision), 5),
            }
        )
    return templates


def build_scan_phase_color_templates(
    scan_phase_templates: list[dict[str, Any]],
    display_states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for display_state in display_states:
        display_id = str(display_state.get("id", ""))
        mean_rgb = display_state.get("mean_rgb", [255, 176, 46])
        color_by_id = {
            int(item["id"]): item.get("rgb", mean_rgb)
            for item in display_state.get("slot_colors", [])
            if "id" in item
        }
        for phase in scan_phase_templates:
            phase_id = str(phase.get("id", ""))
            ids = [int(v) for v in phase.get("on_slot_ids", [])]
            templates.append(
                {
                    "id": f"{display_id}_{phase_id}",
                    "display_state_id": display_id,
                    "phase_template_id": phase_id,
                    "slot_count": len(ids),
                    "on_count": len(ids),
                    "on_slot_ids": ids,
                    "mean_rgb": mean_rgb,
                    "slot_colors": [
                        {"id": sid, "rgb": color_by_id.get(sid, mean_rgb)}
                        for sid in ids
                    ],
                }
            )
    return templates


def draw_slots_overlay(image: np.ndarray, slots: list[dict[str, Any]], output_path: Path, title: str) -> None:
    vis = image.copy()
    for slot in slots:
        x = int(round(float(slot["x"])))
        y = int(round(float(slot["y"])))
        cv2.circle(vis, (x, y), 3, (0, 220, 0), -1)
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, 28), (0, 0, 0), -1)
    cv2.putText(vis, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def draw_state_preview(slots: list[dict[str, Any]], on_ids: set[int], image_size: tuple[int, int], output_path: Path, title: str) -> None:
    width, height = image_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for slot in slots:
        sid = int(slot["id"])
        x = int(round(float(slot["x"])))
        y = int(round(float(slot["y"])))
        color = (0, 190, 255) if sid in on_ids else (45, 45, 45)
        cv2.circle(canvas, (x, y), 3, color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 28), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def build_state_mask_from_ids(slot_count: int, ids: list[Any]) -> np.ndarray:
    mask = np.zeros((slot_count,), dtype=bool)
    arr = np.asarray(ids, dtype=np.int32)
    arr = arr[(arr >= 1) & (arr <= slot_count)]
    mask[arr - 1] = True
    return mask


def draw_frame_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], scale: float = 0.55) -> None:
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_detection_review_frame(
    frame: np.ndarray,
    slots: list[dict[str, Any]],
    expected_mask: np.ndarray,
    actual_mask: np.ndarray,
    trace_row: dict[str, Any],
    events_for_frame: list[dict[str, Any]],
) -> np.ndarray:
    canvas = frame.copy()
    panel = np.zeros((110, canvas.shape[1], 3), dtype=np.uint8)
    panel[:] = (12, 12, 12)
    info_lines = [
        f"frame={trace_row.get('frame_index', '')}  t={trace_row.get('time_sec', '')}s  sim={trace_row.get('similarity', '')}",
        f"expected={trace_row.get('expected_state', '')}  matched={trace_row.get('matched_state', '')}  transition={trace_row.get('transition', '')}",
        f"event={trace_row.get('event_type', '') or 'none'}",
    ]
    if events_for_frame:
        event_text = "; ".join(str(item.get("type", "")) for item in events_for_frame[:3])
        info_lines.append(f"events={event_text}")
    else:
        info_lines.append("events=none")
    for idx, line in enumerate(info_lines):
        color = (220, 220, 220)
        if idx == 2 and trace_row.get("event_type"):
            color = (80, 120, 255)
        draw_frame_label(panel, line, 12, 26 + idx * 24, color, scale=0.6)

    overlay = canvas.copy()
    for idx, slot in enumerate(slots):
        x = int(round(float(slot["x"])))
        y = int(round(float(slot["y"])))
        actual = bool(actual_mask[idx]) if idx < len(actual_mask) else False
        expected = bool(expected_mask[idx]) if idx < len(expected_mask) else False
        if actual and expected:
            color = (0, 210, 0)
            radius = 3
            thickness = -1
        elif actual and not expected:
            color = (0, 140, 255)
            radius = 3
            thickness = -1
        elif expected and not actual:
            color = (0, 0, 255)
            radius = 3
            thickness = 1
        else:
            color = (90, 90, 90)
            radius = 2
            thickness = 1
        cv2.circle(overlay, (x, y), radius, color, thickness, cv2.LINE_AA)

    blended = cv2.addWeighted(canvas, 0.82, overlay, 0.78, 0.0)
    review = np.vstack([panel, blended])
    legend_y = panel.shape[0] + 22
    draw_frame_label(review, "green=expected+actual  orange=unexpected_on  red=missing_on", 12, legend_y, (200, 200, 200), scale=0.5)
    return review


def write_detection_review_video(
    output_path: Path,
    frames: list[FrameItem],
    slots: list[dict[str, Any]],
    states: list[dict[str, Any]],
    masks: np.ndarray,
    trace_rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    fps: float,
) -> None:
    if not frames or len(trace_rows) != len(frames):
        return
    slot_count = len(slots)
    state_masks = {
        str(state.get("id", "")): build_state_mask_from_ids(slot_count, state.get("on_slot_ids", []))
        for state in states
    }
    events_by_frame: dict[int, list[dict[str, Any]]] = {}
    for item in events:
        try:
            key = int(item.get("frame_index", -1))
        except Exception:
            continue
        events_by_frame.setdefault(key, []).append(item)

    first = draw_detection_review_frame(
        frames[0].image,
        slots,
        state_masks.get(str(trace_rows[0].get("expected_state", "")), np.zeros((slot_count,), dtype=bool)),
        masks[0],
        trace_rows[0],
        events_by_frame.get(int(trace_rows[0].get("frame_index", -1)), []),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, max(float(fps), 1.0), (first.shape[1], first.shape[0]))
    if not writer.isOpened():
        return
    try:
        writer.write(first)
        for idx in range(1, len(frames)):
            row = trace_rows[idx]
            expected_mask = state_masks.get(str(row.get("expected_state", "")), np.zeros((slot_count,), dtype=bool))
            review_frame = draw_detection_review_frame(
                frames[idx].image,
                slots,
                expected_mask,
                masks[idx],
                row,
                events_by_frame.get(int(row.get("frame_index", -1)), []),
            )
            writer.write(review_frame)
    finally:
        writer.release()


def crop_image(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    if bbox is None:
        return image
    x, y, w, h = bbox
    return image[y:y + h, x:x + w]


def load_preview_frame(
    video: Path | None,
    frames_dir: Path | None,
    frames: list[FrameItem],
    source_meta: dict[str, Any],
    panel_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if frames_dir is not None:
        sample = frames[len(frames) // 2]
        if panel_bbox is None:
            return sample.image.copy()
        original = cv2.imread(sample.source)
        return original if original is not None else sample.image.copy()
    if video is None:
        return frames[len(frames) // 2].image.copy()
    cap = cv2.VideoCapture(str(video))
    mid_index = int(frames[len(frames) // 2].index)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_index)
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        return frame
    return frames[len(frames) // 2].image.copy()


def find_best_state(mask: np.ndarray, states: list[dict[str, Any]], slot_count: int) -> tuple[int, float]:
    best_idx = -1
    best_score = -1.0
    for idx, state in enumerate(states):
        proto = np.zeros((slot_count,), dtype=bool)
        ids = np.asarray(state.get("on_slot_ids", []), dtype=np.int32)
        ids = ids[(ids >= 1) & (ids <= slot_count)]
        proto[ids - 1] = True
        score = hamming_similarity(mask, proto)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


def color_distance(a: list[Any], b: list[Any]) -> float:
    if len(a) < 3 or len(b) < 3:
        return 999.0
    return float(math.sqrt(sum((float(a[idx]) - float(b[idx])) ** 2 for idx in range(3))))


def build_candidate_diagnostics(recipe: dict[str, Any]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    slot_count = int(recipe.get("slot_count", len(recipe.get("slots", []))) or 0)
    panel_bbox = recipe.get("panel_bbox")
    scan_phases = recipe.get("scan_phase_templates", [])
    display_states = recipe.get("display_states", [])
    display_sequence = recipe.get("display_sequence", [])
    grid_completion = recipe.get("thresholds", {}).get("grid_completion", {})

    def add(code: str, severity: str, message: str, data: dict[str, Any] | None = None) -> None:
        item: dict[str, Any] = {"code": code, "severity": severity, "message": message}
        if data:
            item["data"] = data
        issues.append(item)

    if not panel_bbox:
        add("roi_missing", "block", "没有稳定识别到矩阵 ROI，不能作为产线配方。")
    if slot_count <= 0:
        add("slot_count_unstable", "block", "没有得到有效物理槽位。")
    elif slot_count < 100:
        add("slot_count_unstable", "review", f"物理槽位数只有 {slot_count}，需要人工确认是否漏检。")

    if grid_completion:
        x_count = int(grid_completion.get("x_peak_count", 0) or grid_completion.get("cols", 0) or 0)
        y_count = int(grid_completion.get("y_peak_count", 0) or grid_completion.get("rows", 0) or 0)
        if x_count and y_count and x_count * y_count != slot_count:
            add(
                "slot_count_unstable",
                "review",
                "投影行列数与最终槽位数不一致，需要确认是否有裁剪或补点问题。",
                {"x_count": x_count, "y_count": y_count, "slot_count": slot_count},
            )

    if len(scan_phases) != 2:
        add(
            "scan_phase_count_review_required",
            "review",
            f"扫描相位数量为 {len(scan_phases)}，不是默认期望的 2 套，需要人工确认。",
            {"scan_phase_count": len(scan_phases)},
        )

    if not display_states:
        add("display_state_missing", "block", "没有生成显示模板，不能冻结。")
    for state in display_states:
        on_count = int(state.get("on_count", len(state.get("on_slot_ids", []))) or 0)
        if slot_count and on_count < int(slot_count * 0.9):
            add(
                "low_coverage_display_state",
                "review",
                f"{state.get('id', '')} 亮点数 {on_count} 低于物理槽位 {slot_count} 的 90%，可能是过渡态、扫描态或漏灯。",
                {"state_id": state.get("id", ""), "on_count": on_count, "slot_count": slot_count},
            )

    if len(display_states) > 8:
        add(
            "color_state_review_required",
            "review",
            f"显示颜色/状态数量达到 {len(display_states)} 个，可能存在颜色过分裂，需要人工确认。",
            {"display_state_count": len(display_states)},
        )

    close_pairs: list[dict[str, Any]] = []
    for idx, left in enumerate(display_states):
        for right in display_states[idx + 1:]:
            dist = color_distance(left.get("mean_rgb", []), right.get("mean_rgb", []))
            if dist < 25.0:
                close_pairs.append({"a": left.get("id", ""), "b": right.get("id", ""), "rgb_distance": round(dist, 2)})
    if close_pairs:
        add("color_state_review_required", "review", "存在颜色非常接近的显示模板，需要确认是否应合并。", {"pairs": close_pairs[:10]})

    short_steps = [
        {"state_id": item.get("state_id", ""), "duration_frames": item.get("duration_frames", 0)}
        for item in display_sequence
        if int(item.get("duration_frames", 0) or 0) < 3
    ]
    if short_steps:
        add("sequence_review_required", "review", "显示时序中存在极短状态，可能是过渡帧或状态抖动。", {"short_steps": short_steps[:10]})

    severities = {str(item.get("severity", "")) for item in issues}
    verdict = "PASS"
    if "block" in severities:
        verdict = "BLOCK"
    elif "review" in severities:
        verdict = "REVIEW"
    return {
        "verdict": verdict,
        "requires_approval": True,
        "issues": issues,
        "summary": {
            "slot_count": slot_count,
            "scan_phase_count": len(scan_phases),
            "display_state_count": len(display_states),
            "display_sequence_length": len(display_sequence),
        },
    }


def make_approval_template(recipe: dict[str, Any]) -> dict[str, Any]:
    slot_count = int(recipe.get("slot_count", len(recipe.get("slots", []))) or 0)
    display_states = recipe.get("display_states", [])
    accepted: list[str] = []
    ignored: list[str] = []
    for state in display_states:
        sid = str(state.get("id", ""))
        on_count = int(state.get("on_count", len(state.get("on_slot_ids", []))) or 0)
        if slot_count and on_count < int(slot_count * 0.9):
            ignored.append(sid)
        else:
            accepted.append(sid)
    return {
        "approval_version": 1,
        "expected_slot_count": slot_count,
        "expected_scan_phase_count": len(recipe.get("scan_phase_templates", [])),
        "accepted_display_states": accepted,
        "merged_display_states": {},
        "ignored_states": ignored,
        "expected_sequence": [],
        "override_diagnostics": False,
        "approval_notes": "人工确认后，把需要保留的显示态放入 accepted_display_states；过渡态/误分组放入 ignored_states。diagnostics 不是 PASS 时，如确认可用，需把 override_diagnostics 改为 true。",
    }


def merge_display_states(display_states: list[dict[str, Any]], approval: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {str(state.get("id", "")): state for state in display_states}
    ignored = {str(item) for item in approval.get("ignored_states", [])}
    accepted = [str(item) for item in approval.get("accepted_display_states", []) if str(item) not in ignored]
    merged = approval.get("merged_display_states", {}) or {}
    result: list[dict[str, Any]] = []
    consumed: set[str] = set()

    for new_id, source_ids_raw in merged.items():
        source_ids = [str(item) for item in source_ids_raw if str(item) in by_id and str(item) not in ignored]
        if not source_ids:
            continue
        union_ids: set[int] = set()
        colors: list[list[int]] = []
        for sid in source_ids:
            consumed.add(sid)
            union_ids.update(int(v) for v in by_id[sid].get("on_slot_ids", []))
            if by_id[sid].get("mean_rgb"):
                colors.append([int(v) for v in by_id[sid].get("mean_rgb", [])[:3]])
        mean_rgb = [255, 176, 46]
        if colors:
            mean_rgb = [int(round(float(np.mean([rgb[idx] for rgb in colors])))) for idx in range(3)]
        result.append(
            {
                "id": str(new_id),
                "index": len(result),
                "source_state_ids": source_ids,
                "on_slot_ids": sorted(union_ids),
                "on_count": len(union_ids),
                "mean_rgb": mean_rgb,
                "merged_by_approval": True,
            }
        )

    for sid in accepted:
        if sid in consumed or sid not in by_id:
            continue
        state = json.loads(json.dumps(by_id[sid], ensure_ascii=False))
        state["index"] = len(result)
        result.append(state)
    return result


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    candidate = load_recipe(args.candidate)
    approval = load_recipe(args.approval)
    status = str(candidate.get("status") or candidate.get("recipe_stage") or "")
    if status and status != "candidate":
        raise ValueError(f"freeze 只能读取 candidate recipe，当前状态是 {status}")

    expected_slot_count = int(approval.get("expected_slot_count", -1))
    candidate_slot_count = int(candidate.get("slot_count", len(candidate.get("slots", []))) or 0)
    if expected_slot_count != candidate_slot_count:
        raise ValueError(f"expected_slot_count={expected_slot_count} 与候选槽位数 {candidate_slot_count} 不一致，不能冻结")

    expected_phase_count = int(approval.get("expected_scan_phase_count", -1))
    candidate_phase_count = len(candidate.get("scan_phase_templates", []))
    if expected_phase_count != candidate_phase_count:
        raise ValueError(f"expected_scan_phase_count={expected_phase_count} 与候选扫描相位数 {candidate_phase_count} 不一致，不能冻结")

    diagnostics = candidate.get("diagnostics", {})
    verdict = str(diagnostics.get("verdict", "BLOCK"))
    if verdict != "PASS" and not bool(approval.get("override_diagnostics", False)):
        raise ValueError(f"候选诊断 verdict={verdict}，approval.json 未显式 override_diagnostics=true，不能冻结")

    frozen = json.loads(json.dumps(candidate, ensure_ascii=False))
    frozen["status"] = "frozen"
    frozen["recipe_stage"] = "frozen"
    frozen["frozen_at"] = datetime.now().isoformat(timespec="seconds")
    frozen["source_candidate"] = str(args.candidate)
    frozen["display_states"] = merge_display_states(candidate.get("display_states", []), approval)
    expected_sequence = [str(item) for item in approval.get("expected_sequence", [])]
    if expected_sequence:
        frozen["display_sequence"] = [{"state_id": sid, "duration_frames": 0, "approved": True} for sid in expected_sequence]
    else:
        accepted_ids = {str(state.get("id", "")) for state in frozen.get("display_states", [])}
        frozen["display_sequence"] = [item for item in candidate.get("display_sequence", []) if str(item.get("state_id", "")) in accepted_ids]
    frozen["approval_summary"] = {
        "approval": str(args.approval),
        "expected_slot_count": expected_slot_count,
        "expected_scan_phase_count": expected_phase_count,
        "accepted_display_states": approval.get("accepted_display_states", []),
        "ignored_states": approval.get("ignored_states", []),
        "merged_display_states": approval.get("merged_display_states", {}),
        "expected_sequence": expected_sequence,
        "override_diagnostics": bool(approval.get("override_diagnostics", False)),
        "approval_notes": str(approval.get("approval_notes", "")),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, frozen)
    summary = {
        "frozen_recipe": str(args.output),
        "source_candidate": str(args.candidate),
        "slot_count": candidate_slot_count,
        "scan_phase_count": candidate_phase_count,
        "display_state_count": len(frozen.get("display_states", [])),
        "diagnostics_verdict": verdict,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def event_category(event_type: str) -> str:
    if event_type in {"skipped_state", "reverse_state", "missing_state", "stuck_state"}:
        return "wrong_sequence"
    if event_type in {"no_states", "wrong_pattern", "unknown_state"}:
        return "wrong_display_state"
    if event_type in {"slot_dark", "slot_bright", "slot_abnormal"}:
        return "slot_defect"
    return "wrong_display_state"


def learn(args: argparse.Namespace) -> dict[str, Any]:
    frames, source_meta = load_frames(args.video, args.frames_dir, args.sample_fps, args.max_frames)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_recipe: dict[str, Any] | None = load_recipe(args.slots_recipe) if args.slots_recipe else None
    panel_bbox = None
    if seed_recipe and seed_recipe.get("panel_bbox"):
        panel_bbox = tuple(int(v) for v in seed_recipe["panel_bbox"][:4])
    elif args.auto_crop_panel:
        panel_bbox = median_panel_bbox(frames, args.panel_padding_ratio)
    raw_cropped_frames: list[FrameItem] | None = None
    if panel_bbox is not None:
        frames = crop_frames_to_bbox(frames, panel_bbox)
        raw_cropped_frames = [FrameItem(f.index, f.time_sec, f.image.copy(), f.source, f.roi_offset) for f in frames]

    alignment_rows: list[dict[str, Any]] = []
    alignment_reference_path: Path | None = None
    if args.align_roi:
        frames, alignment_rows, alignment_reference = align_frames_translation(frames, args.max_align_shift, method=args.alignment_method)
        alignment_reference_path = out_dir / "alignment_reference.jpg"
        cv2.imwrite(str(alignment_reference_path), alignment_reference)
        if alignment_rows:
            write_csv(out_dir / "alignment_shifts.csv", alignment_rows, ["frame_index", "time_sec", "dx", "dy", "score"])

    points_by_frame: list[np.ndarray] = []
    point_rows: list[dict[str, Any]] = []
    projection_image = build_projection_image(frames)
    projection_path = out_dir / "slot_source_projection.jpg"
    cv2.imwrite(str(projection_path), projection_image)

    if seed_recipe is not None:
        slots = seed_recipe["slots"]
        local_slots = offset_slots(slots, -panel_bbox[0], -panel_bbox[1]) if panel_bbox is not None else slots
        groups = seed_recipe.get("groups", [])
        fit = seed_recipe.get("fit", {})
        layout_mode = str(seed_recipe.get("layout_mode", "seeded_slots"))
        union_points = points_to_array([(float(slot["x"]) - (panel_bbox[0] if panel_bbox else 0), float(slot["y"]) - (panel_bbox[1] if panel_bbox else 0)) for slot in slots])
        point_rows.append({"frame_index": -1, "time_sec": 0.0, "point_count": len(union_points), "threshold": "seeded"})
    elif args.slot_source == "projection":
        union_points, projection_stats = detect_slot_points(projection_image, args.slot_detector, args.point_detect_scale)
        point_rows.append(
            {
                "frame_index": -1,
                "time_sec": 0.0,
                "point_count": len(union_points),
                "threshold": projection_stats.get("threshold", ""),
            }
        )
    else:
        for frame in frames:
            arr, stats = detect_slot_points(frame.image, args.slot_detector, args.point_detect_scale)
            points_by_frame.append(arr)
            point_rows.append({"frame_index": frame.index, "time_sec": round(frame.time_sec, 4), "point_count": len(arr), "threshold": stats.get("threshold", "")})
        union_points = cluster_union_points(points_by_frame, args.merge_radius, args.min_slot_observations)

    grid_completion: dict[str, Any] | None = None
    if seed_recipe is None and args.complete_grid_from_projection:
        grid_points, grid_completion = projection_grid_points(projection_image, args.grid_profile_threshold)
        if len(grid_points) >= max(args.min_slots, int(round(len(union_points) * 0.9))):
            union_points = grid_points

    if len(union_points) < args.min_slots:
        raise RuntimeError(f"Only {len(union_points)} union LED points found; need at least {args.min_slots}")

    height, width = frames[0].image.shape[:2]
    if seed_recipe is None:
        layout_mode, local_slots, groups, fit = build_slots_from_points(union_points, width, height)
    scores = score_matrix(frames, local_slots, args.score_radius, args.score_channel)
    thresholds, slot_stats = learn_slot_thresholds(scores, args.min_peak_delta)
    raw_masks = masks_from_scores(scores, thresholds)
    fused_masks = fuse_masks(raw_masks, args.fuse_window, args.fuse_on_ratio)
    segments = segment_masks(fused_masks, args.dedupe_similarity, args.min_state_frames)
    states, sequence = assign_key_states(segments, args.state_similarity)
    logical_states, logical_sequence = build_logical_states(
        states,
        sequence,
        len(local_slots),
        args.logical_window_states,
        args.logical_state_similarity,
        args.logical_fill_dense_ratio,
    )
    display_states, display_sequence = build_display_states(
        frames,
        local_slots,
        fused_masks,
        args.score_radius,
        args.display_max_color_states,
        args.display_color_distance,
        args.display_min_segment_frames,
        args.display_fill_dense_ratio,
        args.display_state_similarity,
    )
    scan_phase_templates = build_scan_phase_templates(local_slots, states)
    scan_phase_color_templates = build_scan_phase_color_templates(scan_phase_templates, display_states)
    slots = offset_slots(local_slots, panel_bbox[0], panel_bbox[1]) if panel_bbox is not None else local_slots

    recipe = {
        "type": "dot_matrix_video_recipe",
        "version": 1,
        "status": "candidate",
        "recipe_stage": "candidate",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source_meta,
        "image_size": [source_meta["width"], source_meta["height"]],
        "panel_bbox": list(panel_bbox) if panel_bbox is not None else None,
        "layout_mode": layout_mode,
        "slot_count": len(slots),
        "slots": slots,
        "groups": groups,
        "fit": fit,
        "slot_stats": slot_stats,
        "states": states,
        "sequence": sequence,
        "logical_states": logical_states,
        "logical_sequence": logical_sequence,
        "display_states": display_states,
        "display_sequence": display_sequence,
        "scan_phase_templates": scan_phase_templates,
        "scan_phase_color_templates": scan_phase_color_templates,
        "thresholds": {
            "slot_source": str(args.slot_source),
            "slot_detector": str(args.slot_detector),
            "slots_recipe": str(args.slots_recipe) if args.slots_recipe else "",
            "complete_grid_from_projection": bool(args.complete_grid_from_projection),
            "grid_profile_threshold": float(args.grid_profile_threshold),
            "grid_completion": grid_completion or {},
            "merge_radius": float(args.merge_radius),
            "min_slot_observations": int(args.min_slot_observations),
            "point_detect_scale": float(args.point_detect_scale),
            "align_roi": bool(args.align_roi),
            "alignment_method": str(args.alignment_method),
            "max_align_shift": int(args.max_align_shift),
            "alignment_reference_image": str(alignment_reference_path) if alignment_reference_path is not None else "",
            "score_radius": int(args.score_radius),
            "score_channel": str(args.score_channel),
            "min_peak_delta": float(args.min_peak_delta),
            "fuse_window": int(args.fuse_window),
            "fuse_on_ratio": float(args.fuse_on_ratio),
            "dedupe_similarity": float(args.dedupe_similarity),
            "state_similarity": float(args.state_similarity),
            "logical_window_states": int(args.logical_window_states),
            "logical_state_similarity": float(args.logical_state_similarity),
            "logical_fill_dense_ratio": float(args.logical_fill_dense_ratio),
            "display_max_color_states": int(args.display_max_color_states),
            "display_color_distance": float(args.display_color_distance),
            "display_min_segment_frames": int(args.display_min_segment_frames),
            "display_fill_dense_ratio": float(args.display_fill_dense_ratio),
            "display_state_similarity": float(args.display_state_similarity),
            "min_state_frames": int(args.min_state_frames),
            "detect_match_similarity": float(args.detect_match_similarity),
            "transition_tolerance_frames": int(args.transition_tolerance_frames),
            "hold_timeout_factor": float(args.hold_timeout_factor),
            "cyclic_sequence": bool(args.cyclic_sequence),
        },
    }
    recipe["diagnostics"] = build_candidate_diagnostics(recipe)

    recipe_path = out_dir / "dot_matrix_video_recipe.json"
    candidate_path = out_dir / "candidate_recipe.json"
    write_json(recipe_path, recipe)
    write_json(candidate_path, recipe)
    write_json(out_dir / "approval_template.json", make_approval_template(recipe))
    write_csv(out_dir / "frame_point_counts.csv", point_rows, ["frame_index", "time_sec", "point_count", "threshold"])
    original_preview = load_preview_frame(args.video, args.frames_dir, frames, source_meta, panel_bbox)
    draw_slots_overlay(original_preview, slots, out_dir / "slot_overlay.jpg", f"{layout_mode} slots={len(slots)} states={len(states)}")
    if panel_bbox is not None:
        if raw_cropped_frames:
            cv2.imwrite(str(out_dir / "roi_crop_before_align.jpg"), raw_cropped_frames[len(raw_cropped_frames) // 2].image)
        cv2.imwrite(str(out_dir / "roi_crop_after_align.jpg"), frames[len(frames) // 2].image)
        draw_slots_overlay(frames[len(frames) // 2].image, local_slots, out_dir / "slot_overlay_crop.jpg", f"{layout_mode} slots={len(slots)} states={len(states)}")
    preview_dir = out_dir / "state_previews"
    preview_slots = local_slots if panel_bbox is not None else slots
    preview_size = (panel_bbox[2], panel_bbox[3]) if panel_bbox is not None else (source_meta["width"], source_meta["height"])
    for state in states:
        draw_state_preview(
            preview_slots,
            set(int(v) for v in state["on_slot_ids"]),
            preview_size,
            preview_dir / f"{state['id']}.jpg",
            f"{state['id']} on={state['on_count']}",
        )
    logical_preview_dir = out_dir / "logical_state_previews"
    for state in logical_states:
        draw_state_preview(
            preview_slots,
            set(int(v) for v in state["on_slot_ids"]),
            preview_size,
            logical_preview_dir / f"{state['id']}.jpg",
            f"{state['id']} on={state['on_count']}",
        )
    display_preview_dir = out_dir / "display_state_previews"
    for state in display_states:
        draw_state_preview(
            preview_slots,
            set(int(v) for v in state["on_slot_ids"]),
            preview_size,
            display_preview_dir / f"{state['id']}.jpg",
            f"{state['id']} on={state['on_count']}",
        )
    phase_preview_dir = out_dir / "scan_phase_previews"
    for state in scan_phase_templates:
        draw_state_preview(
            preview_slots,
            set(int(v) for v in state["on_slot_ids"]),
            preview_size,
            phase_preview_dir / f"{state['id']}.jpg",
            f"{state['id']} on={state['on_count']}",
        )
    phase_color_preview_dir = out_dir / "scan_phase_color_previews"
    for state in scan_phase_color_templates:
        draw_state_preview(
            preview_slots,
            set(int(v) for v in state["on_slot_ids"]),
            preview_size,
            phase_color_preview_dir / f"{state['id']}.jpg",
            f"{state['id']} on={state['on_count']}",
        )
    summary = {
        "recipe": str(recipe_path),
        "candidate_recipe": str(candidate_path),
        "approval_template": str(out_dir / "approval_template.json"),
        "diagnostics_verdict": recipe["diagnostics"]["verdict"],
        "layout_mode": layout_mode,
        "slot_count": len(slots),
        "state_count": len(states),
        "logical_state_count": len(logical_states),
        "display_state_count": len(display_states),
        "sequence_length": len(sequence),
        "sampled_frames": len(frames),
        "slot_overlay": str(out_dir / "slot_overlay.jpg"),
        "state_previews": str(preview_dir),
    }
    write_json(out_dir / "learn_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def load_recipe(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def detect(args: argparse.Namespace) -> dict[str, Any]:
    recipe = load_recipe(args.recipe)
    recipe_status = str(recipe.get("status") or recipe.get("recipe_stage") or "")
    if recipe_status != "frozen":
        raise ValueError(f"产线 detect 只允许使用 frozen_recipe.json；当前配方状态是 {recipe_status or 'unknown'}")
    frames, source_meta = load_frames(args.video, args.frames_dir, args.sample_fps, args.max_frames)
    panel_bbox = tuple(int(v) for v in recipe.get("panel_bbox", [])[:4]) if recipe.get("panel_bbox") else None
    if panel_bbox is not None:
        frames = crop_frames_to_bbox(frames, panel_bbox)
    thresholds = recipe.get("thresholds", {})
    if bool(thresholds.get("align_roi", False)):
        reference_value = str(thresholds.get("alignment_reference_image", ""))
        reference_image = None
        if reference_value:
            reference_path = Path(reference_value)
            if not reference_path.exists():
                reference_path = args.recipe.parent / reference_path.name
            if reference_path.exists():
                reference_image = cv2.imread(str(reference_path))
        frames, _, _ = align_frames_translation(
            frames,
            int(thresholds.get("max_align_shift", 6)),
            reference_image,
            str(thresholds.get("alignment_method", "phase")),
        )
    slots = recipe["slots"]
    local_slots = offset_slots(slots, -panel_bbox[0], -panel_bbox[1]) if panel_bbox is not None else slots
    states = recipe["states"]
    learned_sequence = recipe.get("sequence", [])
    expected_sequence_ids = [str(item.get("state_id", "")) for item in learned_sequence if item.get("state_id")]
    if not expected_sequence_ids:
        expected_sequence_ids = [str(state.get("id", "")) for state in states]
    state_by_id = {str(state.get("id", "")): state for state in states}
    state_index_by_id = {str(state.get("id", "")): idx for idx, state in enumerate(states)}
    slot_count = len(slots)
    score_radius = int(args.score_radius or thresholds.get("score_radius", 4))
    score_channel = str(args.score_channel or thresholds.get("score_channel", "max"))
    fuse_window = int(args.fuse_window or thresholds.get("fuse_window", 3))
    fuse_on_ratio = float(args.fuse_on_ratio or thresholds.get("fuse_on_ratio", 0.5))
    match_similarity = float(args.match_similarity or thresholds.get("detect_match_similarity", 0.92))
    transition_tolerance = int(args.transition_tolerance_frames or thresholds.get("transition_tolerance_frames", 3))
    hold_timeout_factor = float(args.hold_timeout_factor or thresholds.get("hold_timeout_factor", 3.0))
    cyclic_sequence = bool(args.cyclic_sequence if args.cyclic_sequence is not None else thresholds.get("cyclic_sequence", True))

    slot_thresholds = np.asarray([float(item["threshold"]) for item in recipe["slot_stats"]], dtype=np.float32)
    scores = score_matrix(frames, local_slots, score_radius, score_channel)
    masks = fuse_masks(masks_from_scores(scores, slot_thresholds), fuse_window, fuse_on_ratio)

    expected_pos = 0
    if cyclic_sequence and len(masks) and expected_sequence_ids:
        first_idx, _ = find_best_state(masks[0], states, slot_count)
        first_id = states[first_idx]["id"] if first_idx >= 0 else ""
        first_positions = [pos for pos, sid in enumerate(expected_sequence_ids) if sid == first_id]
        if first_positions:
            expected_pos = first_positions[0]
    hold_frames = 0
    unknown_frames = 0
    events: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    visited_positions: set[int] = set()
    slot_expected_on_counts = np.zeros((slot_count,), dtype=np.int32)
    slot_missing_counts = np.zeros((slot_count,), dtype=np.int32)
    slot_expected_off_counts = np.zeros((slot_count,), dtype=np.int32)
    slot_unexpected_counts = np.zeros((slot_count,), dtype=np.int32)
    for idx, mask in enumerate(masks):
        frame = frames[idx]
        best_idx, best_score = find_best_state(mask, states, slot_count)
        expected_id = expected_sequence_ids[expected_pos] if expected_sequence_ids else ""
        if cyclic_sequence and expected_sequence_ids:
            next_pos = (expected_pos + 1) % len(expected_sequence_ids)
        else:
            next_pos = min(expected_pos + 1, max(len(expected_sequence_ids) - 1, 0))
        next_expected_id = expected_sequence_ids[next_pos] if expected_sequence_ids else ""
        current_state = state_by_id.get(expected_id, {"id": expected_id})
        matched_id = states[best_idx]["id"] if best_idx >= 0 else ""
        transition = "unknown"
        event_type = ""

        if not states:
            event_type = "no_states"
        elif matched_id == expected_id and best_score >= match_similarity:
            transition = "stay"
            hold_frames += 1
            unknown_frames = 0
            visited_positions.add(expected_pos)
        elif matched_id == next_expected_id and (cyclic_sequence or next_pos > expected_pos) and best_score >= match_similarity:
            expected_pos = next_pos
            transition = "advance"
            hold_frames = 1
            unknown_frames = 0
            visited_positions.add(expected_pos)
        elif best_score >= match_similarity and expected_sequence_ids:
            if cyclic_sequence:
                future_match = next(
                    (
                        (expected_pos + offset) % len(expected_sequence_ids)
                        for offset in range(2, len(expected_sequence_ids) + 1)
                        if expected_sequence_ids[(expected_pos + offset) % len(expected_sequence_ids)] == matched_id
                    ),
                    -1,
                )
                past_match = next(
                    (
                        (expected_pos - offset) % len(expected_sequence_ids)
                        for offset in range(1, len(expected_sequence_ids))
                        if expected_sequence_ids[(expected_pos - offset) % len(expected_sequence_ids)] == matched_id
                    ),
                    -1,
                )
            else:
                future_match = next(
                    (pos for pos in range(expected_pos + 2, len(expected_sequence_ids)) if expected_sequence_ids[pos] == matched_id),
                    -1,
                )
                past_match = next(
                    (pos for pos in range(expected_pos - 1, -1, -1) if expected_sequence_ids[pos] == matched_id),
                    -1,
                )
            if future_match >= 0:
                event_type = "skipped_state"
                events.append({"frame_index": frame.index, "time_sec": frame.time_sec, "type": event_type, "category": event_category(event_type), "expected": expected_id, "matched": matched_id, "score": best_score, "expected_pos": expected_pos, "matched_pos": future_match})
                expected_pos = future_match
                hold_frames = 1
                unknown_frames = 0
                transition = "advance_skip"
                visited_positions.add(expected_pos)
            elif past_match >= 0:
                event_type = "reverse_state"
                events.append({"frame_index": frame.index, "time_sec": frame.time_sec, "type": event_type, "category": event_category(event_type), "expected": expected_id, "matched": matched_id, "score": best_score, "expected_pos": expected_pos, "matched_pos": past_match})
                transition = "reverse"
            else:
                event_type = "wrong_pattern"
                events.append({"frame_index": frame.index, "time_sec": frame.time_sec, "type": event_type, "category": event_category(event_type), "expected": expected_id, "matched": matched_id, "score": best_score, "expected_pos": expected_pos})
                transition = "wrong_pattern"
        else:
            unknown_frames += 1
            transition = "unknown_transition"
            if unknown_frames > transition_tolerance:
                event_type = "unknown_state"
                events.append({"frame_index": frame.index, "time_sec": frame.time_sec, "type": event_type, "category": event_category(event_type), "expected": expected_id, "matched": matched_id, "score": best_score, "expected_pos": expected_pos})
                unknown_frames = 0

        median_duration = float(current_state.get("median_duration_frames", 0) or 0)
        if len(expected_sequence_ids) > 1 and transition == "stay" and median_duration > 0 and hold_frames > median_duration * hold_timeout_factor:
            event_type = "stuck_state"
            events.append({"frame_index": frame.index, "time_sec": frame.time_sec, "type": event_type, "category": event_category(event_type), "expected": expected_id, "matched": matched_id, "score": best_score})
            hold_frames = 0

        if matched_id == expected_id and best_score >= match_similarity and expected_id in state_by_id:
            expected_mask = np.zeros((slot_count,), dtype=bool)
            ids = np.asarray(state_by_id[expected_id].get("on_slot_ids", []), dtype=np.int32)
            ids = ids[(ids >= 1) & (ids <= slot_count)]
            expected_mask[ids - 1] = True
            slot_expected_on_counts += expected_mask.astype(np.int32)
            slot_missing_counts += np.logical_and(expected_mask, np.logical_not(mask)).astype(np.int32)
            slot_expected_off_counts += np.logical_not(expected_mask).astype(np.int32)
            slot_unexpected_counts += np.logical_and(np.logical_not(expected_mask), mask).astype(np.int32)

        trace_rows.append(
            {
                "frame_index": frame.index,
                "time_sec": round(frame.time_sec, 4),
                "expected_state": expected_id,
                "expected_pos": expected_pos,
                "matched_state": matched_id,
                "similarity": round(float(best_score), 5),
                "transition": transition,
                "event_type": event_type,
            }
        )

    for pos, sid in enumerate(expected_sequence_ids):
        if not cyclic_sequence and pos not in visited_positions:
            event_type = "missing_state"
            events.append({"frame_index": frames[-1].index, "time_sec": frames[-1].time_sec, "type": event_type, "category": event_category(event_type), "expected": sid, "matched": "", "score": 0.0, "expected_pos": pos})

    min_slot_samples = 5
    for slot_idx in range(slot_count):
        on_samples = int(slot_expected_on_counts[slot_idx])
        if on_samples >= min_slot_samples:
            ratio = float(slot_missing_counts[slot_idx]) / float(on_samples)
            if ratio >= 0.85:
                event_type = "slot_dark"
                events.append(
                    {
                        "frame_index": frames[-1].index,
                        "time_sec": frames[-1].time_sec,
                        "type": event_type,
                        "category": event_category(event_type),
                        "slot_id": slot_idx + 1,
                        "expected": "on",
                        "matched": "off",
                        "score": round(1.0 - ratio, 5),
                        "sample_count": on_samples,
                    }
                )
        off_samples = int(slot_expected_off_counts[slot_idx])
        if off_samples >= min_slot_samples:
            ratio = float(slot_unexpected_counts[slot_idx]) / float(off_samples)
            if ratio >= 0.85:
                event_type = "slot_bright"
                events.append(
                    {
                        "frame_index": frames[-1].index,
                        "time_sec": frames[-1].time_sec,
                        "type": event_type,
                        "category": event_category(event_type),
                        "slot_id": slot_idx + 1,
                        "expected": "off",
                        "matched": "on",
                        "score": round(1.0 - ratio, 5),
                        "sample_count": off_samples,
                    }
                )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "state_trace.csv", trace_rows, ["frame_index", "time_sec", "expected_state", "expected_pos", "matched_state", "similarity", "transition", "event_type"])
    category_counts: dict[str, int] = {}
    for item in events:
        category = str(item.get("category") or event_category(str(item.get("type", ""))))
        item["category"] = category
        category_counts[category] = category_counts.get(category, 0) + 1
    write_json(out_dir / "events.json", {"events": events, "event_categories": category_counts})
    review_video_path = out_dir / "review_video.mp4"
    write_detection_review_video(
        review_video_path,
        frames,
        local_slots,
        states,
        masks,
        trace_rows,
        events,
        float(source_meta.get("sample_fps", 0.0) or args.sample_fps or 12.0),
    )
    summary = {
        "type": "dot_matrix_video_detection_summary",
        "recipe": str(args.recipe),
        "recipe_status": recipe_status,
        "source": source_meta,
        "overall_ok": len(events) == 0,
        "event_count": len(events),
        "event_categories": category_counts,
        "events": events[:20],
        "trace_csv": str(out_dir / "state_trace.csv"),
        "events_json": str(out_dir / "events.json"),
        "review_video": str(review_video_path),
    }
    write_json(out_dir / "detection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def add_learn_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slots-recipe", type=Path, default=None)
    parser.add_argument("--sample-fps", type=float, default=12.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--slot-source", choices=["projection", "union"], default="projection")
    parser.add_argument("--slot-detector", choices=["blob", "peak"], default="blob")
    parser.add_argument("--complete-grid-from-projection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid-profile-threshold", type=float, default=0.25)
    parser.add_argument("--merge-radius", type=float, default=4.5)
    parser.add_argument("--min-slot-observations", type=int, default=2)
    parser.add_argument("--point-detect-scale", type=float, default=1.0)
    parser.add_argument("--score-radius", type=int, default=6)
    parser.add_argument("--score-channel", choices=["max", "gray", "red", "green", "blue"], default="max")
    parser.add_argument("--min-peak-delta", type=float, default=18.0)
    parser.add_argument("--fuse-window", type=int, default=3)
    parser.add_argument("--fuse-on-ratio", type=float, default=0.5)
    parser.add_argument("--dedupe-similarity", type=float, default=0.985)
    parser.add_argument("--state-similarity", type=float, default=0.94)
    parser.add_argument("--logical-window-states", type=int, default=2)
    parser.add_argument("--logical-state-similarity", type=float, default=0.97)
    parser.add_argument("--logical-fill-dense-ratio", type=float, default=0.95)
    parser.add_argument("--display-max-color-states", type=int, default=6)
    parser.add_argument("--display-color-distance", type=float, default=34.0)
    parser.add_argument("--display-min-segment-frames", type=int, default=3)
    parser.add_argument("--display-fill-dense-ratio", type=float, default=0.95)
    parser.add_argument("--display-state-similarity", type=float, default=0.96)
    parser.add_argument("--detect-match-similarity", type=float, default=0.92)
    parser.add_argument("--transition-tolerance-frames", type=int, default=3)
    parser.add_argument("--hold-timeout-factor", type=float, default=3.0)
    parser.add_argument("--cyclic-sequence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-state-frames", type=int, default=2)
    parser.add_argument("--min-slots", type=int, default=8)
    parser.add_argument("--auto-crop-panel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--panel-padding-ratio", type=float, default=0.18)
    parser.add_argument("--align-roi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alignment-method", choices=["phase", "ncc"], default="phase")
    parser.add_argument("--max-align-shift", type=int, default=6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Learn, freeze, and detect arbitrary dot-matrix LED video states.")
    sub = parser.add_subparsers(dest="command", required=True)

    learn_p = sub.add_parser("learn", help="Compatibility alias: learn a candidate recipe from a good video.")
    add_learn_arguments(learn_p)

    candidate_p = sub.add_parser("learn-candidate", help="Learn candidate_recipe.json and approval_template.json from a good video.")
    add_learn_arguments(candidate_p)

    freeze_p = sub.add_parser("freeze", help="Freeze a candidate recipe after manual approval.")
    freeze_p.add_argument("--candidate", type=Path, required=True)
    freeze_p.add_argument("--approval", type=Path, required=True)
    freeze_p.add_argument("--output", type=Path, required=True)

    detect_p = sub.add_parser("detect", help="Detect a video or frame directory using a frozen recipe.")
    detect_p.add_argument("--recipe", type=Path, required=True)
    detect_p.add_argument("--video", type=Path, default=None)
    detect_p.add_argument("--frames-dir", type=Path, default=None)
    detect_p.add_argument("--output-dir", type=Path, required=True)
    detect_p.add_argument("--sample-fps", type=float, default=12.0)
    detect_p.add_argument("--max-frames", type=int, default=0)
    detect_p.add_argument("--score-radius", type=int, default=0)
    detect_p.add_argument("--score-channel", choices=["max", "gray", "red", "green", "blue"], default="")
    detect_p.add_argument("--fuse-window", type=int, default=0)
    detect_p.add_argument("--fuse-on-ratio", type=float, default=0.0)
    detect_p.add_argument("--match-similarity", type=float, default=0.0)
    detect_p.add_argument("--transition-tolerance-frames", type=int, default=0)
    detect_p.add_argument("--hold-timeout-factor", type=float, default=0.0)
    detect_p.add_argument("--cyclic-sequence", action=argparse.BooleanOptionalAction, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"learn", "learn-candidate"}:
        learn(args)
    elif args.command == "freeze":
        freeze(args)
    elif args.command == "detect":
        detect(args)


if __name__ == "__main__":
    main()
