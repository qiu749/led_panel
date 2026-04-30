from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_PANEL_SIZE = (420, 720)
DEFAULT_OUTPUT_DIR = Path("output") / "marquee_inspection"


@dataclass
class PanelTrack:
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class Slot:
    id: int
    x: float
    y: float
    w: float
    h: float
    threshold: float
    peak: float
    duty: float
    hits: int


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


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, total_frames, fps, width, height


def read_video_frames(video_path: Path, stride: int = 1, max_frames: int = 0) -> tuple[list[np.ndarray], float]:
    cap, total_frames, fps, _, _ = open_video(video_path)
    frames: list[np.ndarray] = []
    stride = max(1, int(stride))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if index % stride == 0:
            frames.append(frame)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        index += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"视频无可读帧: {video_path}")
    return frames, fps / stride


def detect_panel_bbox(frame: np.ndarray) -> PanelTrack | None:
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    dark = (blurred < 92).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((23, 23), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: tuple[float, tuple[int, int, int, int]] | None = None
    frame_area = float(width * height)
    frame_cx = width * 0.5
    frame_cy = height * 0.5
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < frame_area * 0.04 or area > frame_area * 0.48:
            continue
        aspect = h / max(w, 1)
        if not 1.25 <= aspect <= 3.8:
            continue
        extent = cv2.contourArea(contour) / max(area, 1.0)
        cx = x + w * 0.5
        cy = y + h * 0.5
        centrality = 1.0 - min(1.0, math.hypot((cx - frame_cx) / width, (cy - frame_cy) / height) * 1.8)
        edge_penalty = 0.9 if x <= 4 or y <= 4 or x + w >= width - 4 or y + h >= height - 4 else 0.0
        score = area / frame_area + 0.45 * extent + 0.25 * centrality - edge_penalty
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))
    if best is None:
        return None

    x, y, w, h = refine_panel_bbox(frame, best[1])
    pad_x = int(round(w * 0.015))
    pad_y = int(round(h * 0.015))
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(width - x, w + pad_x * 2)
    h = min(height - y, h + pad_y * 2)
    return PanelTrack((x, y, w, h), float(best[0]))


def largest_true_run(flags: np.ndarray, min_length: int) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, flag in enumerate(flags.tolist() + [False]):
        if flag and start is None:
            start = index
        if not flag and start is not None:
            end = index
            if end - start >= min_length and (best is None or end - start > best[1] - best[0]):
                best = (start, end)
            start = None
    return best


def refine_panel_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    image_h, image_w = frame.shape[:2]
    x, y, w, h = bbox
    x = max(0, min(image_w - 1, x))
    y = max(0, min(image_h - 1, y))
    w = max(1, min(image_w - x, w))
    h = max(1, min(image_h - y, h))
    crop = frame[y : y + h, x : x + w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = gray < 92
    row_frac = dark.mean(axis=1)
    col_frac = dark.mean(axis=0)
    row_run = largest_true_run(row_frac > 0.34, max(80, int(h * 0.35)))
    col_run = largest_true_run(col_frac > 0.32, max(80, int(w * 0.35)))
    if row_run is None or col_run is None:
        return bbox
    y1, y2 = row_run
    x1, x2 = col_run
    pad_x = max(8, int((x2 - x1) * 0.04))
    pad_y = max(8, int((y2 - y1) * 0.04))
    nx = max(0, x + x1 - pad_x)
    ny = max(0, y + y1 - pad_y)
    nx2 = min(image_w, x + x2 + pad_x)
    ny2 = min(image_h, y + y2 + pad_y)
    if nx2 - nx < 80 or ny2 - ny < 120:
        return bbox
    return nx, ny, nx2 - nx, ny2 - ny


def median_panel_bbox(frames: list[np.ndarray], sample_count: int = 30) -> tuple[int, int, int, int]:
    if not frames:
        raise RuntimeError("没有帧可用于定位面板")
    indices = np.linspace(0, len(frames) - 1, min(sample_count, len(frames)), dtype=int)
    boxes: list[tuple[int, int, int, int]] = []
    for index in indices:
        track = detect_panel_bbox(frames[int(index)])
        if track is not None:
            boxes.append(track.bbox)
    if not boxes:
        height, width = frames[0].shape[:2]
        margin_x = int(width * 0.2)
        margin_y = int(height * 0.18)
        return margin_x, margin_y, width - margin_x * 2, height - margin_y * 2
    arr = np.asarray(boxes, dtype=np.float32)
    x, y, w, h = np.median(arr, axis=0)
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def align_panel(frame: np.ndarray, bbox: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    image_h, image_w = frame.shape[:2]
    x, y, w, h = bbox
    x = max(0, min(image_w - 1, int(x)))
    y = max(0, min(image_h - 1, int(y)))
    w = max(1, min(image_w - x, int(w)))
    h = max(1, min(image_h - y, int(h)))
    crop = frame[y : y + h, x : x + w]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)


def frame_bboxes(frames: list[np.ndarray], fallback_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    previous = fallback_bbox
    for frame in frames:
        track = detect_panel_bbox(frame)
        if track is None:
            boxes.append(previous)
        else:
            previous = track.bbox
            boxes.append(previous)
    return boxes


def inside_focus(component: dict[str, float], focus_bbox: tuple[int, int, int, int] | None) -> bool:
    if focus_bbox is None:
        return True
    x, y, w, h = focus_bbox
    return x <= component["x"] <= x + w and y <= component["y"] <= y + h


def bright_components(panel: np.ndarray, focus_bbox: tuple[int, int, int, int] | None = None) -> list[dict[str, float]]:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    local = cv2.GaussianBlur(gray, (31, 31), 0)
    contrast = cv2.subtract(gray, local)
    threshold = max(145, int(np.percentile(gray, 99.35) * 0.72))
    mask = ((gray >= threshold) | ((v >= 150) & (contrast >= 28) & (s >= 20))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    num, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    components: list[dict[str, float]] = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area < 4 or area > 2600:
            continue
        if w > panel.shape[1] * 0.35 or h > panel.shape[0] * 0.28:
            continue
        if max(w / max(h, 1), h / max(w, 1)) > 9.0 and area > 220:
            continue
        component = labels == idx
        score = float(np.percentile(gray[component], 95)) / 255.0
        if score < 0.45:
            continue
        component = {
            "x": float(centers[idx][0]),
            "y": float(centers[idx][1]),
            "w": float(w),
            "h": float(h),
            "area": float(area),
            "score": score,
        }
        if inside_focus(component, focus_bbox):
            components.append(component)
    return components


def cluster_components(
    panels: list[np.ndarray],
    cluster_radius: float = 18.0,
    min_hits: int = 2,
    focus_bbox: tuple[int, int, int, int] | None = None,
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    radius_sq = cluster_radius * cluster_radius
    for frame_index, panel in enumerate(panels):
        for component in bright_components(panel, focus_bbox=focus_bbox):
            best_index = -1
            best_distance = radius_sq
            for index, cluster in enumerate(clusters):
                dx = component["x"] - cluster["x"]
                dy = component["y"] - cluster["y"]
                dist = dx * dx + dy * dy
                if dist < best_distance:
                    best_distance = dist
                    best_index = index
            if best_index < 0:
                clusters.append(
                    {
                        "x": component["x"],
                        "y": component["y"],
                        "weights": component["score"],
                        "hits": 1,
                        "frames": {frame_index},
                        "widths": [component["w"]],
                        "heights": [component["h"]],
                    }
                )
                continue
            cluster = clusters[best_index]
            weight = component["score"]
            total = cluster["weights"] + weight
            cluster["x"] = (cluster["x"] * cluster["weights"] + component["x"] * weight) / total
            cluster["y"] = (cluster["y"] * cluster["weights"] + component["y"] * weight) / total
            cluster["weights"] = total
            cluster["hits"] += 1
            cluster["frames"].add(frame_index)
            cluster["widths"].append(component["w"])
            cluster["heights"].append(component["h"])

    filtered: list[dict[str, Any]] = []
    for cluster in clusters:
        if len(cluster["frames"]) < min_hits:
            continue
        filtered.append(cluster)
    return sorted(filtered, key=lambda item: (item["y"], item["x"]))


def slot_score(panel: np.ndarray, slot: Slot) -> float:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    x1 = max(0, int(round(slot.x - slot.w * 0.5)))
    y1 = max(0, int(round(slot.y - slot.h * 0.5)))
    x2 = min(gray.shape[1], int(round(slot.x + slot.w * 0.5)))
    y2 = min(gray.shape[0], int(round(slot.y + slot.h * 0.5)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = gray[y1:y2, x1:x2]
    return float(np.percentile(roi, 98)) / 255.0


def score_matrix(panels: list[np.ndarray], slots: list[Slot]) -> np.ndarray:
    scores = np.zeros((len(panels), len(slots)), dtype=np.float32)
    for frame_index, panel in enumerate(panels):
        for slot_index, slot in enumerate(slots):
            scores[frame_index, slot_index] = slot_score(panel, slot)
    return scores


def active_matrix(scores: np.ndarray, slots: list[Slot], dominance_ratio: float = 0.58) -> np.ndarray:
    if scores.size == 0:
        return np.zeros_like(scores, dtype=bool)
    thresholds = np.asarray([slot.threshold for slot in slots], dtype=np.float32)
    active = scores >= thresholds[None, :]
    frame_max = scores.max(axis=1)
    active &= scores >= (frame_max[:, None] * dominance_ratio)
    active[frame_max < 0.32, :] = False
    return active


def dominant_sequence(scores: np.ndarray, slots: list[Slot], min_event_frames: int = 2) -> list[dict[str, Any]]:
    active = active_matrix(scores, slots)
    raw: list[int] = []
    for frame_index in range(scores.shape[0]):
        indices = np.where(active[frame_index])[0]
        if len(indices) == 0:
            raw.append(-1)
            continue
        best = int(indices[np.argmax(scores[frame_index, indices])])
        raw.append(int(slots[best].id))

    events: list[dict[str, Any]] = []
    start = 0
    current = raw[0] if raw else -1
    for idx in range(1, len(raw) + 1):
        value = raw[idx] if idx < len(raw) else None
        if value == current:
            continue
        length = idx - start
        if current >= 0 and length >= min_event_frames:
            events.append({"slot_id": current, "start_frame": start, "end_frame": idx - 1, "duration_frames": length})
        start = idx
        current = value if value is not None else -1

    compressed: list[dict[str, Any]] = []
    for event in events:
        if compressed and compressed[-1]["slot_id"] == event["slot_id"]:
            compressed[-1]["end_frame"] = event["end_frame"]
            compressed[-1]["duration_frames"] += event["duration_frames"]
        else:
            compressed.append(event)
    return compressed


def build_slots_from_video(
    panels: list[np.ndarray],
    max_static_duty: float = 0.72,
    min_hits: int = 2,
    focus_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[list[Slot], np.ndarray, list[dict[str, Any]]]:
    clusters = cluster_components(panels, min_hits=min_hits, focus_bbox=focus_bbox)
    provisional: list[Slot] = []
    for idx, cluster in enumerate(clusters):
        w = float(np.clip(np.median(cluster["widths"]) * 2.2, 20, 76))
        h = float(np.clip(np.median(cluster["heights"]) * 2.2, 20, 76))
        provisional.append(
            Slot(
                id=idx + 1,
                x=float(cluster["x"]),
                y=float(cluster["y"]),
                w=w,
                h=h,
                threshold=0.48,
                peak=0.0,
                duty=0.0,
                hits=len(cluster["frames"]),
            )
        )

    if not provisional:
        raise RuntimeError("未能从视频中学习到跑马灯灯位")

    scores = score_matrix(panels, provisional)
    keep: list[Slot] = []
    for slot_index, slot in enumerate(provisional):
        series = scores[:, slot_index]
        peak = float(np.max(series))
        background = float(np.percentile(series, 35))
        threshold = max(0.34, min(0.78, background + (peak - background) * 0.48))
        duty = float(np.mean(series >= threshold))
        if peak < 0.42:
            continue
        if duty > max_static_duty:
            continue
        keep.append(
            Slot(
                id=len(keep) + 1,
                x=slot.x,
                y=slot.y,
                w=slot.w,
                h=slot.h,
                threshold=threshold,
                peak=peak,
                duty=duty,
                hits=slot.hits,
            )
        )
    if not keep:
        raise RuntimeError("学习到的亮区都像静态反光，未形成有效跑马灯灯位")
    scores = score_matrix(panels, keep)
    return keep, scores, clusters


def prune_slots_to_events(
    slots: list[Slot],
    scores: np.ndarray,
    events: list[dict[str, Any]],
    panels: list[np.ndarray],
    fps: float,
) -> tuple[list[Slot], np.ndarray, list[dict[str, Any]]]:
    active_ids = {int(event["slot_id"]) for event in events}
    if not active_ids:
        return slots, scores, events
    remap: dict[int, int] = {}
    pruned: list[Slot] = []
    for slot in slots:
        if slot.id not in active_ids:
            continue
        new_id = len(pruned) + 1
        remap[slot.id] = new_id
        pruned.append(
            Slot(
                id=new_id,
                x=slot.x,
                y=slot.y,
                w=slot.w,
                h=slot.h,
                threshold=slot.threshold,
                peak=slot.peak,
                duty=slot.duty,
                hits=slot.hits,
            )
        )
    if not pruned:
        return slots, scores, events
    pruned_scores = score_matrix(panels, pruned)
    pruned_events = dominant_sequence(pruned_scores, pruned, min_event_frames=max(1, int(round(fps * 0.04))))
    return pruned, pruned_scores, pruned_events


def load_template(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "marquee_sequence_template":
        raise RuntimeError(f"不是跑马灯模板: {path}")
    return payload


def slots_from_template(template: dict[str, Any]) -> list[Slot]:
    return [
        Slot(
            id=int(item["id"]),
            x=float(item["x"]),
            y=float(item["y"]),
            w=float(item["w"]),
            h=float(item["h"]),
            threshold=float(item["threshold"]),
            peak=float(item.get("peak", 1.0)),
            duty=float(item.get("duty", 0.0)),
            hits=int(item.get("hits", 0)),
        )
        for item in template.get("slots", [])
    ]


def lcs_length(left: list[int], right: list[int]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for lv in left:
        current = [0] * (len(right) + 1)
        for j, rv in enumerate(right, start=1):
            if lv == rv:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def best_order_score(expected: list[int], observed: list[int]) -> tuple[float, int, int]:
    if not expected:
        return 0.0, 0, 0
    if not observed:
        return 0.0, 0, 0
    best = (0.0, 0, 0)
    doubled = expected + expected
    max_window = min(len(doubled), max(len(expected), len(observed)) + 3)
    for offset in range(len(expected)):
        candidate = doubled[offset : offset + max_window]
        matched = lcs_length(candidate, observed)
        denom = max(len(expected), len(observed), 1)
        score = matched / denom
        if score > best[0]:
            best = (score, offset, matched)
    return best


def compare_sequence(
    template: dict[str, Any],
    slots: list[Slot],
    scores: np.ndarray,
    fps: float,
) -> dict[str, Any]:
    events = dominant_sequence(scores, slots, min_event_frames=max(1, int(round(fps * 0.04))))
    observed = [int(event["slot_id"]) for event in events]
    expected = [int(item) for item in template.get("expected_sequence", [])]
    score, offset, matched = best_order_score(expected, observed)
    observed_set = set(observed)
    expected_set = set(expected)

    missing_slots = sorted(expected_set - observed_set)
    extra_slots = sorted(observed_set - expected_set)
    constant_slots: list[int] = []
    dim_slots: list[int] = []
    slot_stats: list[dict[str, Any]] = []
    for slot_index, slot in enumerate(slots):
        series = scores[:, slot_index]
        active_ratio = float(np.mean(series >= slot.threshold))
        peak = float(np.max(series))
        template_peak = max(float(slot.peak), 1e-6)
        if active_ratio > 0.82:
            constant_slots.append(slot.id)
        if slot.id in expected_set and peak < template_peak * 0.62:
            dim_slots.append(slot.id)
        slot_stats.append(
            {
                "slot_id": slot.id,
                "peak": round(peak, 4),
                "threshold": round(slot.threshold, 4),
                "active_ratio": round(active_ratio, 4),
            }
        )

    event_durations = [event["duration_frames"] / max(fps, 1e-6) for event in events]
    expected_step = float(template.get("step_seconds", 0.0) or 0.0)
    timing_bad = False
    timing_message = ""
    if expected_step > 0 and event_durations:
        median_step = float(np.median(event_durations))
        if abs(median_step - expected_step) > max(0.12, expected_step * 0.55):
            timing_bad = True
            timing_message = f"节拍偏差: 实测{median_step:.2f}s 标准{expected_step:.2f}s"

    issues: list[dict[str, Any]] = []
    for slot_id in missing_slots:
        issues.append({"type": "missing", "slot_id": slot_id, "message": f"灯位 {slot_id} 未按周期亮起"})
    for slot_id in extra_slots:
        issues.append({"type": "extra", "slot_id": slot_id, "message": f"灯位 {slot_id} 不在标准顺序内但被点亮"})
    for slot_id in constant_slots:
        issues.append({"type": "constant", "slot_id": slot_id, "message": f"灯位 {slot_id} 疑似常亮或强反光"})
    for slot_id in dim_slots:
        issues.append({"type": "dim", "slot_id": slot_id, "message": f"灯位 {slot_id} 亮度偏低"})
    if score < float(template.get("min_order_score", 0.85)):
        issues.append({"type": "order", "slot_id": None, "message": f"跑马顺序不匹配，顺序得分 {score:.2f}"})
    if timing_bad:
        issues.append({"type": "timing", "slot_id": None, "message": timing_message})

    return {
        "overall_ok": not issues,
        "order_score": round(score, 4),
        "matched_events": matched,
        "expected_event_count": len(expected),
        "observed_event_count": len(observed),
        "best_offset": offset,
        "expected_sequence": expected,
        "observed_sequence": observed,
        "events": events,
        "slot_stats": slot_stats,
        "issues": issues,
        "ng_count": len(issues),
    }


def draw_timeline(path: Path, scores: np.ndarray, slots: list[Slot], events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if scores.size == 0:
        return
    normalized = np.clip(scores * 255, 0, 255).astype(np.uint8).T
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    row_height = 18
    scale_x = max(1, int(math.ceil(1100 / max(1, heat.shape[1]))))
    heat = cv2.resize(heat, (heat.shape[1] * scale_x, heat.shape[0] * row_height), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((heat.shape[0] + 34, heat.shape[1] + 70, 3), 28, dtype=np.uint8)
    canvas[24 : 24 + heat.shape[0], 62 : 62 + heat.shape[1]] = heat
    for index, slot in enumerate(slots):
        y = 24 + index * row_height + 13
        cv2.putText(canvas, f"{slot.id:02d}", (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    for event in events:
        x = 62 + int(event["start_frame"]) * scale_x
        cv2.line(canvas, (x, 18), (x, canvas.shape[0] - 4), (255, 255, 255), 1)
    cv2.imwrite(str(path), canvas)


def panel_to_frame_point(
    slot: Slot,
    bbox: tuple[int, int, int, int],
    panel_size: tuple[int, int],
) -> tuple[int, int]:
    panel_w, panel_h = panel_size
    x, y, w, h = bbox
    return int(round(x + slot.x * w / panel_w)), int(round(y + slot.y * h / panel_h))


def write_visual_video(
    output_path: Path,
    frames: list[np.ndarray],
    bboxes: list[tuple[int, int, int, int]],
    scores: np.ndarray,
    slots: list[Slot],
    fps: float,
    issues: list[dict[str, Any]] | None = None,
    panel_size: tuple[int, int] = DEFAULT_PANEL_SIZE,
    focus_bbox: tuple[int, int, int, int] | None = None,
) -> None:
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (width, height))
    issue_slots = {int(issue["slot_id"]) for issue in issues or [] if issue.get("slot_id") is not None}
    active = active_matrix(scores, slots)
    for frame_index, frame in enumerate(frames):
        vis = frame.copy()
        bbox = bboxes[min(frame_index, len(bboxes) - 1)]
        x, y, w, h = bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 255), 2)
        if focus_bbox is not None:
            fx, fy, fw, fh = focus_bbox
            panel_w, panel_h = panel_size
            fx1 = int(round(x + fx * w / panel_w))
            fy1 = int(round(y + fy * h / panel_h))
            fx2 = int(round(x + (fx + fw) * w / panel_w))
            fy2 = int(round(y + (fy + fh) * h / panel_h))
            cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (255, 180, 0), 2)
        if frame_index < active.shape[0]:
            active_ids = {slots[idx].id for idx in np.where(active[frame_index])[0]}
        else:
            active_ids = set()
        for slot_index, slot in enumerate(slots):
            cx, cy = panel_to_frame_point(slot, bbox, panel_size)
            if slot.id in issue_slots:
                color = (0, 0, 255)
                radius = 12
                thickness = 3
            elif slot.id in active_ids:
                color = (0, 255, 0)
                radius = 11
                thickness = 3
            else:
                color = (180, 180, 180)
                radius = 7
                thickness = 1
            cv2.circle(vis, (cx, cy), radius, color, thickness)
            cv2.putText(vis, str(slot.id), (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        writer.write(vis)
    writer.release()


def template_payload(
    video_path: Path,
    panel_bbox: tuple[int, int, int, int],
    panel_size: tuple[int, int],
    fps: float,
    slots: list[Slot],
    scores: np.ndarray,
    events: list[dict[str, Any]],
    focus_bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    sequence = [int(event["slot_id"]) for event in events]
    durations = [event["duration_frames"] / max(fps, 1e-6) for event in events]
    return {
        "type": "marquee_sequence_template",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_video": str(video_path),
        "panel_size": list(panel_size),
        "reference_panel_bbox": list(panel_bbox),
        "focus_bbox": list(focus_bbox) if focus_bbox is not None else None,
        "fps": fps,
        "expected_sequence": sequence,
        "step_seconds": round(float(np.median(durations)) if durations else 0.0, 4),
        "min_order_score": 0.85,
        "slots": [
            {
                "id": slot.id,
                "x": round(slot.x, 3),
                "y": round(slot.y, 3),
                "w": round(slot.w, 3),
                "h": round(slot.h, 3),
                "threshold": round(slot.threshold, 4),
                "peak": round(slot.peak, 4),
                "duty": round(slot.duty, 4),
                "hits": slot.hits,
            }
            for slot in slots
        ],
        "notes": "可人工修正 slots 坐标/阈值和 expected_sequence 顺序。",
    }


def learn_template(
    video_path: Path,
    output_template: Path,
    output_dir: Path,
    panel_size: tuple[int, int] = DEFAULT_PANEL_SIZE,
    stride: int = 1,
    focus_bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    frames, fps = read_video_frames(video_path, stride=stride)
    panel_bbox = median_panel_bbox(frames)
    bboxes = frame_bboxes(frames, panel_bbox)
    panels = [align_panel(frame, bbox, panel_size) for frame, bbox in zip(frames, bboxes)]
    min_hits = max(2, int(len(panels) * 0.006))
    slots, scores, _ = build_slots_from_video(panels, min_hits=min_hits, focus_bbox=focus_bbox)
    events = dominant_sequence(scores, slots, min_event_frames=max(1, int(round(fps * 0.04))))
    slots, scores, events = prune_slots_to_events(slots, scores, events, panels, fps)
    payload = template_payload(video_path, panel_bbox, panel_size, fps, slots, scores, events, focus_bbox=focus_bbox)
    write_json(output_template, payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    draw_timeline(output_dir / "learn_timeline.png", scores, slots, events)
    write_visual_video(output_dir / "learn_visual.mp4", frames, bboxes, scores, slots, fps, panel_size=panel_size, focus_bbox=focus_bbox)
    write_csv(
        output_dir / "learn_events.csv",
        events,
        ["slot_id", "start_frame", "end_frame", "duration_frames"],
    )
    write_json(
        output_dir / "learn_summary.json",
        {
            "template": str(output_template),
            "slot_count": len(slots),
            "event_count": len(events),
            "panel_bbox": panel_bbox,
            "fps": fps,
        },
    )
    return payload


def detect_video(
    video_path: Path,
    template_path: Path,
    output_dir: Path,
    stride: int = 1,
) -> dict[str, Any]:
    frames, fps = read_video_frames(video_path, stride=stride)
    return detect_frames(frames, fps, template_path, output_dir, source=str(video_path))


def detect_frames(
    frames: list[np.ndarray],
    fps: float,
    template_path: Path,
    output_dir: Path,
    source: str = "frames",
) -> dict[str, Any]:
    template = load_template(template_path)
    panel_size_payload = template.get("panel_size", list(DEFAULT_PANEL_SIZE))
    panel_size = (int(panel_size_payload[0]), int(panel_size_payload[1]))
    slots = slots_from_template(template)
    focus_payload = template.get("focus_bbox")
    focus_bbox = tuple(int(v) for v in focus_payload) if focus_payload else None
    if not slots:
        raise RuntimeError(f"模板没有灯位: {template_path}")

    reference_bbox = tuple(int(v) for v in template.get("reference_panel_bbox", median_panel_bbox(frames)))
    bboxes = frame_bboxes(frames, reference_bbox)
    panels = [align_panel(frame, bbox, panel_size) for frame, bbox in zip(frames, bboxes)]
    scores = score_matrix(panels, slots)
    result = compare_sequence(template, slots, scores, fps)
    result.update(
        {
            "video": source,
            "template": str(template_path),
            "fps": fps,
            "frame_count": len(frames),
            "panel_size": list(panel_size),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result)
    draw_timeline(output_dir / "detect_timeline.png", scores, slots, result["events"])
    write_visual_video(output_dir / "detect_visual.mp4", frames, bboxes, scores, slots, fps, result["issues"], panel_size, focus_bbox)
    write_csv(
        output_dir / "detect_events.csv",
        result["events"],
        ["slot_id", "start_frame", "end_frame", "duration_frames"],
    )
    score_rows: list[dict[str, Any]] = []
    for frame_index in range(scores.shape[0]):
        row: dict[str, Any] = {"frame": frame_index}
        for slot_index, slot in enumerate(slots):
            row[f"slot_{slot.id:02d}"] = round(float(scores[frame_index, slot_index]), 4)
        score_rows.append(row)
    write_csv(output_dir / "scores.csv", score_rows, ["frame"] + [f"slot_{slot.id:02d}" for slot in slots])
    return result


def parse_size(value: str) -> tuple[int, int]:
    if "x" not in value.lower():
        raise argparse.ArgumentTypeError("尺寸格式应为 WIDTHxHEIGHT，例如 420x720")
    left, right = value.lower().split("x", 1)
    return int(left), int(right)


def parse_bbox(value: str) -> tuple[int, int, int, int]:
    parts = [part.strip() for part in value.replace(";", ",").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox 格式应为 x,y,w,h")
    return tuple(int(float(part)) for part in parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="跑马灯时序质检：良品视频建模与视频检测。")
    sub = parser.add_subparsers(dest="command", required=True)

    learn = sub.add_parser("learn", help="从良品视频学习跑马灯模板")
    learn.add_argument("--video", type=Path, required=True)
    learn.add_argument("--output-template", type=Path, required=True)
    learn.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "learn")
    learn.add_argument("--panel-size", type=parse_size, default=DEFAULT_PANEL_SIZE)
    learn.add_argument("--stride", type=int, default=1)
    learn.add_argument("--focus-bbox", type=parse_bbox, default=None, help="标准面板坐标内的有效跑马灯区域: x,y,w,h")

    detect = sub.add_parser("detect", help="用跑马灯模板检测视频")
    detect.add_argument("--video", type=Path, required=True)
    detect.add_argument("--template", type=Path, required=True)
    detect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "detect")
    detect.add_argument("--stride", type=int, default=1)

    run = sub.add_parser("run", help="先建模再用同一视频回放验证")
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "run")
    run.add_argument("--panel-size", type=parse_size, default=DEFAULT_PANEL_SIZE)
    run.add_argument("--stride", type=int, default=1)
    run.add_argument("--focus-bbox", type=parse_bbox, default=None, help="标准面板坐标内的有效跑马灯区域: x,y,w,h")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "learn":
        payload = learn_template(args.video, args.output_template, args.output_dir, args.panel_size, args.stride, args.focus_bbox)
        print(json.dumps({"template": str(args.output_template), "slots": len(payload["slots"]), "events": len(payload["expected_sequence"])}, ensure_ascii=False))
    elif args.command == "detect":
        result = detect_video(args.video, args.template, args.output_dir, args.stride)
        print(json.dumps({"overall_ok": result["overall_ok"], "ng_count": result["ng_count"], "order_score": result["order_score"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    elif args.command == "run":
        output_dir = args.output_dir
        template_path = output_dir / "marquee_template.json"
        payload = learn_template(args.video, template_path, output_dir / "learn", args.panel_size, args.stride, args.focus_bbox)
        result = detect_video(args.video, template_path, output_dir / "detect", args.stride)
        print(
            json.dumps(
                {
                    "template": str(template_path),
                    "slots": len(payload["slots"]),
                    "events": len(payload["expected_sequence"]),
                    "overall_ok": result["overall_ok"],
                    "ng_count": result["ng_count"],
                    "order_score": result["order_score"],
                    "output_dir": str(output_dir),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
