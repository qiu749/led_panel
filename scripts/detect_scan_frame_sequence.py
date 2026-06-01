"""
detect_scan_frame_sequence

Detect a scan-refresh seven-segment video against a frame-template model.

The detector does not OCR. It matches each lit frame to a learned visual
template, compresses the matched frames into a sequence, compares that sequence
with the model sequence, and writes a non-technical visual review page.

Example:
  python scripts/detect_scan_frame_sequence.py ^
    --video D:/jiemi/七段管/Video_20260515112415516.avi ^
    --model output/sevenseg_frame_templates/Video_20260515112415516/templates.json ^
    --out output/sevenseg_detect/Video_20260515112415516_selfcheck
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.extract_scan_frame_templates import make_mask  # noqa: E402
from scripts.scan_refresh_seven_segment_probe import (  # noqa: E402
    Roi,
    crop_roi,
    detect_display_roi,
    first_pass_projection,
    write_csv,
    write_json,
)


@dataclass
class Template:
    id: str
    bits: str
    image: str
    mask_image: str = ""
    color_bits: str = ""
    color_counts: dict[str, int] | None = None
    count: int = 0
    active_pixels_median: float = 0.0


@dataclass
class MatchSegment:
    id: str
    template_id: str
    status: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    frames: int
    mean_similarity: float
    active_pixels_mean: float
    image: str = ""
    template_image: str = ""
    verdict: str = "ok"
    message: str = ""


@dataclass
class Event:
    level: str
    type: str
    frame: int
    time_sec: float
    template_id: str
    message: str
    image: str = ""
    template_image: str = ""


def vector_to_runs(vec: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    start = -1
    for i, value in enumerate(vec.tolist()):
        if value and start < 0:
            start = i
        elif not value and start >= 0:
            runs.append([start, i - start])
            start = -1
    if start >= 0:
        runs.append([start, len(vec) - start])
    return runs


def bits_to_runs(bits: str) -> list[list[int]]:
    return vector_to_runs(bit_string_to_vector(bits))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(model_path: Path, min_template_count: int, min_template_pixels: float) -> tuple[list[Template], list[str], dict[str, Any]]:
    payload = load_json(model_path)
    raw_templates = payload.get("templates", [])
    templates: list[Template] = []
    allowed_ids: set[str] = set()
    merge_map: dict[str, str] = {
        str(item.get("id")): str(item.get("merged_to"))
        for item in raw_templates
        if item.get("merged_to")
    }

    def canonical(template_id: str) -> str:
        seen: set[str] = set()
        current = template_id
        while current in merge_map and current not in seen:
            seen.add(current)
            current = merge_map[current]
        return current

    for item in raw_templates:
        if item.get("enabled") is False:
            continue
        count = int(item.get("count", 0))
        pixels = float(item.get("active_pixels_median", 0.0))
        if count < min_template_count or pixels < min_template_pixels:
            continue
        tid = str(item["id"])
        if merge_map.get(tid):
            continue
        allowed_ids.add(tid)
        templates.append(
            Template(
                id=tid,
                bits=str(item["bits"]),
                image=str(item.get("image", "")),
                mask_image=str(item.get("mask_image", "")),
                color_bits=str(item.get("color_bits", "")),
                color_counts=dict(item.get("color_counts", {})),
                count=count,
                active_pixels_median=pixels,
            )
        )

    expected = []
    previous = ""
    for item in payload.get("sequence", []):
        tid = canonical(str(item.get("template_id", "")))
        if tid in allowed_ids and tid != previous:
            expected.append(tid)
            previous = tid

    summary_path = model_path.parent / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    return templates, expected, summary


def load_roi(model_summary: dict[str, Any], fallback_projection: np.ndarray, args: argparse.Namespace) -> Roi:
    if args.roi:
        parts = [int(v.strip()) for v in args.roi.split(",")]
        if len(parts) != 4:
            raise ValueError("--roi must be x,y,w,h")
        return Roi(*parts)

    if args.roi_source == "model" and model_summary.get("roi"):
        r = model_summary["roi"]
        return Roi(int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))

    return detect_display_roi(fallback_projection, args.projection_threshold, args.roi_pad_ratio)


def bit_string_to_vector(bits: str) -> np.ndarray:
    return np.fromiter((ch == "1" for ch in bits), dtype=np.bool_, count=len(bits))


def mask_to_vector(mask: np.ndarray, cols: int, rows: int, cell_threshold: int) -> np.ndarray:
    small = cv2.resize(mask.astype(np.uint8) * 255, (cols, rows), interpolation=cv2.INTER_AREA)
    return small.reshape(-1) >= cell_threshold


def dilate_vector(vec: np.ndarray, cols: int, rows: int, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return vec
    mat = vec.reshape(rows, cols).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mat, kernel, iterations=1).astype(bool).reshape(-1)


def shift_vector(vec: np.ndarray, cols: int, rows: int, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return vec
    src = vec.reshape(rows, cols)
    dst = np.zeros_like(src)
    src_x0 = max(0, -dx)
    src_y0 = max(0, -dy)
    dst_x0 = max(0, dx)
    dst_y0 = max(0, dy)
    width = cols - abs(dx)
    height = rows - abs(dy)
    if width <= 0 or height <= 0:
        return dst.reshape(-1)
    dst[dst_y0:dst_y0 + height, dst_x0:dst_x0 + width] = src[src_y0:src_y0 + height, src_x0:src_x0 + width]
    return dst.reshape(-1)


def find_best_alignment_shift(
    actual_vec: np.ndarray,
    target_vec: np.ndarray,
    cols: int,
    rows: int,
    max_shift: int,
    dilate: int,
) -> tuple[int, int, float]:
    target_cmp = dilate_vector(target_vec, cols, rows, dilate)
    best_dx = 0
    best_dy = 0
    best_sim = jaccard_similarity(dilate_vector(actual_vec, cols, rows, dilate), target_cmp)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = shift_vector(actual_vec, cols, rows, dx, dy)
            sim = jaccard_similarity(dilate_vector(shifted, cols, rows, dilate), target_cmp)
            if sim > best_sim:
                best_dx = dx
                best_dy = dy
                best_sim = sim
    return best_dx, best_dy, best_sim


def find_best_template(
    frame_vec: np.ndarray,
    templates: list[Template],
    template_matrix: np.ndarray,
    template_on_counts: np.ndarray,
) -> tuple[Template | None, float]:
    if template_matrix.size == 0:
        return None, 0.0
    frame_on = int(np.count_nonzero(frame_vec))
    intersections = np.logical_and(template_matrix, frame_vec).sum(axis=1)
    unions = template_on_counts + frame_on - intersections
    sims = intersections / np.maximum(unions, 1)
    best_i = int(np.argmax(sims))
    return templates[best_i], float(sims[best_i])


def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return intersection / max(union, 1)


def shift_crop_image(crop: np.ndarray, dx_cells: int, dy_cells: int, cols: int, rows: int) -> np.ndarray:
    if dx_cells == 0 and dy_cells == 0:
        return crop
    h, w = crop.shape[:2]
    dx = int(round(dx_cells * w / max(cols, 1)))
    dy = int(round(dy_cells * h / max(rows, 1)))
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        crop,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def registration_image(crop: np.ndarray, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, max(10, threshold - 20), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask.astype(np.float32) / 255.0


def estimate_image_alignment(
    reference: np.ndarray | None,
    moving: np.ndarray,
    threshold: int,
    mode: str,
) -> tuple[np.ndarray, str, float]:
    warp = np.eye(2, 3, dtype=np.float32)
    if reference is None or mode == "off" or reference.shape[:2] != moving.shape[:2]:
        return warp, "off", 0.0

    motion_map = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    motion = motion_map.get(mode, cv2.MOTION_AFFINE)
    try:
        score, found = cv2.findTransformECC(
            registration_image(reference, threshold),
            registration_image(moving, threshold),
            warp,
            motion,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
            None,
            5,
        )
        return found.astype(np.float32), mode, float(score)
    except cv2.error:
        return warp, "failed", 0.0


def warp_to_reference(crop: np.ndarray, warp: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    return cv2.warpAffine(
        crop,
        warp,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def save_segment_image(
    out_dir: Path,
    segment_id: str,
    frame_index: int,
    crop: np.ndarray,
    label: str,
    color: tuple[int, int, int],
) -> str:
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    image_name = f"{segment_id}_f{frame_index:04d}.jpg"
    annotated = crop.copy()
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1] - 1, annotated.shape[0] - 1), color, 8)
    cv2.putText(
        annotated,
        label,
        (18, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(seg_dir / image_name), annotated)
    return f"segments/{image_name}"


def save_playback_frame_image(
    out_dir: Path,
    frame_index: int,
    crop: np.ndarray,
    status: str,
    template_id: str,
    similarity: float,
    thumb_width: int,
) -> str:
    frame_dir = out_dir / "playback_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    h, w = crop.shape[:2]
    scale = thumb_width / max(w, 1)
    thumb = cv2.resize(crop, (thumb_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

    colors = {
        "matched": (0, 190, 0),
        "unknown": (0, 0, 220),
        "dark": (150, 150, 150),
    }
    color = colors.get(status, (0, 190, 0))
    cv2.rectangle(thumb, (0, 0), (thumb.shape[1] - 1, thumb.shape[0] - 1), color, 5)
    label = f"f{frame_index}  {status.upper()}"
    if template_id:
        label += f"  {template_id}"
    if status != "dark":
        label += f"  sim={similarity:.2f}"
    cv2.putText(thumb, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    image_name = f"frame_{frame_index:05d}.jpg"
    cv2.imwrite(str(frame_dir / image_name), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return f"playback_frames/{image_name}"


def maybe_save_playback_frame_image(
    out_dir: Path,
    frame_index: int,
    crop: np.ndarray,
    status: str,
    template_id: str,
    similarity: float,
    thumb_width: int,
    enabled: bool,
) -> str:
    if not enabled:
        return ""
    return save_playback_frame_image(out_dir, frame_index, crop, status, template_id, similarity, thumb_width)


def copy_template_previews(model_dir: Path, out_dir: Path, templates: list[Template]) -> dict[str, str]:
    dest = out_dir / "model_templates"
    dest.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for tmpl in templates:
        if not tmpl.image:
            continue
        src = model_dir / tmpl.image
        if not src.exists():
            continue
        out_name = f"{tmpl.id}_{src.name}"
        shutil.copyfile(src, dest / out_name)
        copied[tmpl.id] = f"model_templates/{out_name}"
    return copied


def append_segment(
    segments: list[MatchSegment],
    template_id: str,
    status: str,
    frame_index: int,
    fps: float,
    similarity: float,
    active_pixels: int,
    max_gap_frames: int,
) -> None:
    if (
        segments
        and segments[-1].template_id == template_id
        and segments[-1].status == status
        and frame_index - segments[-1].end_frame <= max_gap_frames
    ):
        prev = segments[-1]
        total = prev.frames + 1
        prev.mean_similarity = ((prev.mean_similarity * prev.frames) + similarity) / total
        prev.active_pixels_mean = ((prev.active_pixels_mean * prev.frames) + active_pixels) / total
        prev.end_frame = frame_index
        prev.end_time = frame_index / fps if fps > 0 else 0.0
        prev.frames = total
        return

    seg_no = len(segments) + 1
    segments.append(
        MatchSegment(
            id=f"S{seg_no:03d}",
            template_id=template_id,
            status=status,
            start_frame=frame_index,
            end_frame=frame_index,
            start_time=frame_index / fps if fps > 0 else 0.0,
            end_time=frame_index / fps if fps > 0 else 0.0,
            frames=1,
            mean_similarity=similarity,
            active_pixels_mean=float(active_pixels),
        )
    )


def compare_sequence(segments: list[MatchSegment], expected: list[str], allowed_jitter_templates: int) -> list[Event]:
    events: list[Event] = []
    expected_pos = 0
    expected_len = len(expected)

    for seg in segments:
        if seg.status == "dark":
            seg.verdict = "ignored"
            seg.message = "黑帧/弱亮，已跳过"
            continue
        if seg.status == "unknown":
            seg.verdict = "bad"
            seg.message = "画面亮了，但没有匹配到标准模板"
            events.append(
                Event(
                    level="error",
                    type="unknown_template",
                    frame=seg.start_frame,
                    time_sec=seg.start_time,
                    template_id=seg.template_id,
                    message=seg.message,
                    image=seg.image,
                )
            )
            continue

        if expected_pos < expected_len and seg.template_id == expected[expected_pos]:
            seg.verdict = "ok"
            seg.message = "顺序正确"
            expected_pos += 1
            continue

        jitter_start = max(0, expected_pos - allowed_jitter_templates)
        if seg.template_id in expected[jitter_start:expected_pos]:
            seg.verdict = "ok"
            seg.message = "局部重复，已接受"
            continue

        later = -1
        for i in range(expected_pos + 1, expected_len):
            if expected[i] == seg.template_id:
                later = i
                break
        if later >= 0:
            skipped = expected[expected_pos:later]
            seg.verdict = "warn"
            seg.message = f"跳过 {len(skipped)} 个模板后匹配到 {seg.template_id}"
            for miss in skipped:
                events.append(
                    Event(
                        level="error",
                        type="missing_template",
                        frame=seg.start_frame,
                        time_sec=seg.start_time,
                        template_id=miss,
                        message=f"期望出现 {miss}，但检测时被跳过",
                    )
                )
            events.append(
                Event(
                    level="warning",
                    type="skipped_forward",
                    frame=seg.start_frame,
                    time_sec=seg.start_time,
                    template_id=seg.template_id,
                    message=seg.message,
                    image=seg.image,
                    template_image=seg.template_image,
                )
            )
            expected_pos = later + 1
            continue

        if seg.template_id in expected[:expected_pos]:
            seg.verdict = "warn"
            seg.message = f"{seg.template_id} 重复或回退出现"
            events.append(
                Event(
                    level="warning",
                    type="repeat_or_backtrack",
                    frame=seg.start_frame,
                    time_sec=seg.start_time,
                    template_id=seg.template_id,
                    message=seg.message,
                    image=seg.image,
                    template_image=seg.template_image,
                )
            )
        else:
            seg.verdict = "bad"
            seg.message = f"{seg.template_id} 不在标准时序中"
            events.append(
                Event(
                    level="error",
                    type="unexpected_template",
                    frame=seg.start_frame,
                    time_sec=seg.start_time,
                    template_id=seg.template_id,
                    message=seg.message,
                    image=seg.image,
                    template_image=seg.template_image,
                )
            )

    for miss in expected[expected_pos:]:
        events.append(
            Event(
                level="error",
                type="missing_template",
                frame=segments[-1].end_frame if segments else 0,
                time_sec=segments[-1].end_time if segments else 0.0,
                template_id=miss,
                message=f"检测结束前没有看到 {miss}",
            )
        )
    return events


def compare_ordered_presence(
    segments: list[MatchSegment],
    expected: list[str],
    min_coverage: float,
    max_unknown_segments: int,
) -> tuple[list[Event], dict[str, Any]]:
    events: list[Event] = []
    matched_segments = [seg for seg in segments if seg.status == "matched"]
    observed = [seg.template_id for seg in matched_segments]
    n = len(expected)
    m = len(observed)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if expected[i] == observed[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    i = 0
    j = 0
    matched_expected: list[str] = []
    matched_observed_indices: set[int] = set()
    while i < n and j < m:
        if expected[i] == observed[j] and dp[i][j] == dp[i + 1][j + 1] + 1:
            matched_expected.append(expected[i])
            matched_observed_indices.add(j)
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1

    matched_set = set(matched_expected)
    coverage = len(matched_expected) / max(len(expected), 1)
    unknown_segments = [seg for seg in segments if seg.status == "unknown"]
    passed = coverage >= min_coverage and len(unknown_segments) <= max_unknown_segments

    observed_i = 0
    for seg in segments:
        if seg.status == "dark":
            seg.verdict = "ignored"
            seg.message = "黑帧/弱亮，已跳过"
            continue
        if seg.status == "unknown":
            seg.verdict = "warn" if passed else "bad"
            seg.message = "未匹配亮帧；有序关键帧判定中作为复核点"
            if not passed:
                events.append(
                    Event(
                        level="error",
                        type="unknown_template",
                        frame=seg.start_frame,
                        time_sec=seg.start_time,
                        template_id=seg.template_id,
                        message=seg.message,
                        image=seg.image,
                    )
                )
            continue

        if observed_i in matched_observed_indices:
            seg.verdict = "ok"
            seg.message = "关键模板出现，顺序正确"
        else:
            seg.verdict = "warn"
            seg.message = "不影响关键时序的重复/预热帧，已忽略"
        observed_i += 1

    missing = [tid for tid in expected if tid not in matched_set]
    if passed:
        for tid in missing[:20]:
            events.append(
                Event(
                    level="warning",
                    type="template_not_sampled",
                    frame=segments[-1].end_frame if segments else 0,
                    time_sec=segments[-1].end_time if segments else 0.0,
                    template_id=tid,
                    message=f"{tid} 本次视频未采到；关键时序覆盖率已达标",
                )
            )
    else:
        for tid in missing:
            events.append(
                Event(
                    level="error",
                    type="missing_key_template",
                    frame=segments[-1].end_frame if segments else 0,
                    time_sec=segments[-1].end_time if segments else 0.0,
                    template_id=tid,
                    message=f"关键时序覆盖不足，未看到 {tid}",
                )
            )

    stats = {
        "mode": "ordered_presence",
        "expected_count": len(expected),
        "observed_segment_count": len(observed),
        "ordered_match_count": len(matched_expected),
        "ordered_coverage": coverage,
        "min_coverage": min_coverage,
        "unknown_segment_count": len(unknown_segments),
        "max_unknown_segments": max_unknown_segments,
        "status": "PASS" if passed else "FAIL",
        "missing_templates": missing,
    }
    return events, stats


def create_timeline_image(out_dir: Path, segments: list[MatchSegment], total_frames: int) -> str:
    width = 1600
    height = 170
    margin = 40
    bar_y = 64
    bar_h = 44
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(img, "Timeline", (margin, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2, cv2.LINE_AA)

    colors = {
        "ok": (72, 170, 72),
        "warn": (0, 170, 255),
        "bad": (62, 62, 210),
        "ignored": (178, 178, 178),
    }
    for seg in segments:
        x1 = margin + int((width - margin * 2) * seg.start_frame / max(total_frames, 1))
        x2 = margin + int((width - margin * 2) * (seg.end_frame + 1) / max(total_frames, 1))
        x2 = max(x2, x1 + 2)
        color = colors.get(seg.verdict, colors["bad"])
        cv2.rectangle(img, (x1, bar_y), (x2, bar_y + bar_h), color, -1)
        if x2 - x1 >= 38 and seg.status != "dark":
            cv2.putText(img, seg.template_id, (x1 + 3, bar_y + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    legend = [("OK", colors["ok"]), ("CHECK", colors["warn"]), ("NG", colors["bad"]), ("DARK", colors["ignored"])]
    lx = margin
    for label, color in legend:
        cv2.rectangle(img, (lx, 128), (lx + 28, 148), color, -1)
        cv2.putText(img, label, (lx + 36, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (55, 55, 55), 1, cv2.LINE_AA)
        lx += 150

    path = out_dir / "timeline_overview.jpg"
    cv2.imwrite(str(path), img)
    return "timeline_overview.jpg"


def status_label(verdict: str) -> str:
    return {
        "ok": "通过",
        "warn": "需复核",
        "bad": "异常",
        "ignored": "跳过",
    }.get(verdict, verdict)


def write_review_html(
    out_dir: Path,
    video_name: str,
    summary: dict[str, Any],
    segments: list[MatchSegment],
    events: list[Event],
    timeline_image: str,
) -> None:
    result_class = "pass" if summary["overall_status"] == "PASS" else "fail"
    visual_check = summary.get("overall_visual_check", {})
    alignment = summary.get("alignment", {})
    pass_basis_text = {
        "sequence": "按时序通过",
        "overall_visual": "按整体亮区通过",
        "none": "未通过",
    }.get(str(summary.get("pass_basis", "none")), str(summary.get("pass_basis", "none")))
    event_cards = []
    for event in events[:80]:
        image_html = f'<img src="{html.escape(event.image)}" alt="detected">' if event.image else ""
        tmpl_html = f'<img src="{html.escape(event.template_image)}" alt="template">' if event.template_image else ""
        pair = ""
        if image_html or tmpl_html:
            pair = f'<div class="pair"><div><b>检测画面</b>{image_html}</div><div><b>标准模板</b>{tmpl_html}</div></div>'
        event_cards.append(
            f"""
            <section class="event {html.escape(event.level)}">
              <div class="event-title">{html.escape(event.type)} · {html.escape(event.template_id)}</div>
              <div class="event-meta">f{event.frame} · {event.time_sec:.2f}s</div>
              <p>{html.escape(event.message)}</p>
              {pair}
            </section>
            """
        )
    if not event_cards:
        event_cards.append('<section class="event ok"><div class="event-title">未发现异常</div><p>检测时序与标准模板一致。</p></section>')

    segment_rows = []
    for seg in segments:
        segment_rows.append(
            f"""
            <tr class="{html.escape(seg.verdict)}">
              <td>{html.escape(status_label(seg.verdict))}</td>
              <td>{html.escape(seg.template_id)}</td>
              <td>f{seg.start_frame}-f{seg.end_frame}</td>
              <td>{seg.start_time:.2f}-{seg.end_time:.2f}s</td>
              <td>{seg.frames}</td>
              <td>{seg.mean_similarity:.2f}</td>
              <td>{html.escape(seg.message)}</td>
            </tr>
            """
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>模板时序检测结果</title>
  <style>
    :root {{ color-scheme: light; }}
    body {{ margin: 0; background: #f5f7fa; color: #1f2933; font-family: Arial, "Microsoft YaHei", sans-serif; }}
    header {{ padding: 28px 34px; background: #111827; color: white; }}
    main {{ padding: 24px 34px 50px; }}
    h1, h2 {{ margin: 0 0 14px; }}
    .result {{ display: inline-block; padding: 8px 16px; border-radius: 6px; font-size: 22px; font-weight: 700; }}
    .result.pass {{ background: #1f9d55; }}
    .result.fail {{ background: #d64545; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 16px; }}
    .num {{ font-size: 30px; font-weight: 700; margin-top: 6px; }}
    .timeline {{ background: white; border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; margin: 18px 0 26px; }}
    img {{ max-width: 100%; display: block; background: #000; }}
    .event {{ background: white; border-left: 8px solid #9aa5b1; border-radius: 8px; padding: 14px 16px; margin: 12px 0; box-shadow: 0 1px 2px rgba(15, 23, 42, .08); }}
    .event.error {{ border-left-color: #d64545; }}
    .event.warning {{ border-left-color: #f0b429; }}
    .event.ok {{ border-left-color: #1f9d55; }}
    .event-title {{ font-weight: 700; font-size: 17px; }}
    .event-meta {{ color: #6b7280; font-size: 13px; margin-top: 3px; }}
    .pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 10px; }}
    .pair b {{ display: block; margin-bottom: 5px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9e2ec; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf2f7; font-size: 13px; text-align: left; }}
    th {{ background: #edf2f7; }}
    tr.ok td:first-child {{ color: #1f9d55; font-weight: 700; }}
    tr.warn td:first-child {{ color: #b7791f; font-weight: 700; }}
    tr.bad td:first-child {{ color: #d64545; font-weight: 700; }}
    tr.ignored td:first-child {{ color: #7b8794; font-weight: 700; }}
    .scroll {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <header>
    <h1>模板时序检测结果</h1>
    <div class="result {result_class}">{summary["overall_status"]}</div>
    <div style="margin-top:10px;color:#cbd5e1;">{html.escape(video_name)}</div>
  </header>
  <main>
    <div class="cards">
      <div class="card">匹配帧<div class="num">{summary["matched_frames"]}</div></div>
      <div class="card">未知亮帧<div class="num">{summary["unknown_frames"]}</div></div>
      <div class="card">黑帧/弱亮<div class="num">{summary["dark_frames"]}</div></div>
      <div class="card">异常数<div class="num">{summary["event_count"]}</div></div>
      <div class="card">整体匹配率<div class="num">{summary["match_rate"]:.1%}</div></div>
      <div class="card">判定依据<div class="num" style="font-size:20px;">{html.escape(pass_basis_text)}</div></div>
      <div class="card">整体亮区<div class="num">{float(visual_check.get("similarity", 0.0)):.1%}</div></div>
      <div class="card">自动对齐<div class="num" style="font-size:20px;">dx {int(alignment.get("grid_dx", 0))}, dy {int(alignment.get("grid_dy", 0))}</div></div>
    </div>

    <h2>一眼看懂时间轴</h2>
    <div class="timeline"><img src="{html.escape(timeline_image)}" alt="timeline"></div>
    <p><a href="abstract_player.html" style="display:inline-block;background:#2563eb;color:white;padding:10px 14px;border-radius:6px;text-decoration:none;font-weight:700;">打开抽象模板播放器</a></p>

    <h2>异常和复核点</h2>
    {''.join(event_cards)}

    <h2>逐段明细</h2>
    <div class="scroll">
      <table>
        <thead><tr><th>结果</th><th>模板</th><th>帧号</th><th>时间</th><th>帧数</th><th>相似度</th><th>说明</th></tr></thead>
        <tbody>{''.join(segment_rows)}</tbody>
      </table>
    </div>
  </main>
</body>
</html>
"""
    (out_dir / "review.html").write_text(html_text, encoding="utf-8")


def write_abstract_player_html(out_dir: Path, video_name: str) -> None:
    html_text = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>抽象模板播放器</title>
  <style>
    body { margin: 0; background: #0f172a; color: #e5e7eb; font-family: Arial, "Microsoft YaHei", sans-serif; }
    header { padding: 20px 28px; background: #111827; border-bottom: 1px solid #334155; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    main { padding: 20px 28px 32px; }
    .status { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 14px 0; }
    .pill { padding: 7px 12px; border-radius: 6px; font-weight: 700; background: #334155; }
    .pill.ok { background: #15803d; }
    .pill.ng { background: #b91c1c; }
    .pill.dark { background: #64748b; }
    .stage { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; }
    .panel { background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 14px; }
    .panel h2 { margin: 0 0 10px; font-size: 18px; color: #f8fafc; }
    canvas { width: 100%; aspect-ratio: 3 / 2; background: #020617; border-radius: 8px; display: block; }
    .template-photo-wrap { width: 100%; aspect-ratio: 3 / 2; background: #020617; border-radius: 8px; display: grid; place-items: center; overflow: hidden; }
    .template-photo-wrap img { width: 100%; height: 100%; object-fit: contain; display: block; }
    .template-photo-wrap .empty { color: #94a3b8; font-weight: 700; }
    .controls { margin: 18px 0; display: grid; grid-template-columns: auto auto auto auto 1fr auto; gap: 10px; align-items: center; }
    button { border: 0; border-radius: 6px; padding: 9px 14px; font-weight: 700; background: #2563eb; color: white; cursor: pointer; }
    button.secondary { background: #475569; }
    input[type=range] { width: 100%; }
    .legend { display: flex; gap: 14px; flex-wrap: wrap; color: #cbd5e1; font-size: 14px; }
    .dot { display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 6px; vertical-align: -2px; }
    .note { color: #cbd5e1; line-height: 1.6; max-width: 1100px; }
    .sequence { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 14px 0; }
    .seq-card { background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; }
    .seq-card b { display: block; color: #94a3b8; font-size: 13px; margin-bottom: 4px; }
    .seq-card span { font-size: 20px; font-weight: 700; color: #f8fafc; }
    @media (max-width: 1300px) { .stage { grid-template-columns: repeat(2, 1fr); } }
    @media (max-width: 850px) { .stage { grid-template-columns: 1fr; } .controls { grid-template-columns: 1fr 1fr; } .sequence { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>抽象模板播放器</h1>
    <div id="videoName"></div>
  </header>
  <main>
    <p class="note">从左到右依次是标准模板真图、当前真实亮区、当前抽象亮区、差异图。差异图里绿色表示对上，红色表示当前多亮，黄色表示标准该亮但没亮。</p>
    <div class="status">
      <span id="resultPill" class="pill">加载中</span>
      <span id="timeText" class="pill">0.00s</span>
      <span id="targetText" class="pill">目标: -</span>
      <span id="detectedText" class="pill">检测: -</span>
      <span id="simText" class="pill">相似度: -</span>
    </div>
    <div class="sequence">
      <div class="seq-card"><b>上一步</b><span id="prevText">-</span></div>
      <div class="seq-card"><b>当前应匹配</b><span id="currentText">-</span></div>
      <div class="seq-card"><b>下一步</b><span id="nextText">-</span></div>
    </div>
    <div class="stage">
      <section class="panel">
        <h2>下一步标准模板真图</h2>
        <div class="template-photo-wrap"><img id="targetPhoto" alt="标准模板"><div id="targetEmpty" class="empty">无目标模板</div></div>
      </section>
      <section class="panel">
        <h2>当前真实亮区（已对齐）</h2>
        <div class="template-photo-wrap"><img id="realPhoto" alt="当前真实亮区"><div id="realEmpty" class="empty">无当前帧</div></div>
      </section>
      <section class="panel">
        <h2>视频当前抽象亮区</h2>
        <canvas id="actualCanvas"></canvas>
      </section>
      <section class="panel">
        <h2>差异图</h2>
        <canvas id="diffCanvas"></canvas>
      </section>
    </div>
    <div class="controls">
      <button id="playBtn">播放</button>
      <button id="stepBtn" class="secondary">单步</button>
      <button id="prevIssueBtn" class="secondary">上一个复核点</button>
      <button id="nextIssueBtn" class="secondary">下一个复核点</button>
      <input id="scrub" type="range" min="0" max="0" value="0">
      <label>速度 <input id="speed" type="range" min="1" max="10" value="4"></label>
    </div>
    <div class="legend">
      <span><i class="dot" style="background:#22c55e"></i>匹配</span>
      <span><i class="dot" style="background:#ef4444"></i>多亮/未知</span>
      <span><i class="dot" style="background:#facc15"></i>漏亮</span>
      <span><i class="dot" style="background:#64748b"></i>黑帧/弱亮</span>
      <span><i class="dot" style="background:#38bdf8"></i>左侧为标准模板真图</span>
    </div>
  </main>
<script src="playback_data.js"></script>
<script>
let data, frames, templates, playing = false, timer = null, index = 0;
const actualCanvas = document.getElementById("actualCanvas");
const diffCanvas = document.getElementById("diffCanvas");
const actualCtx = actualCanvas.getContext("2d");
const diffCtx = diffCanvas.getContext("2d");
const targetPhoto = document.getElementById("targetPhoto");
const targetEmpty = document.getElementById("targetEmpty");
const realPhoto = document.getElementById("realPhoto");
const realEmpty = document.getElementById("realEmpty");
const playBtn = document.getElementById("playBtn");
const stepBtn = document.getElementById("stepBtn");
const prevIssueBtn = document.getElementById("prevIssueBtn");
const nextIssueBtn = document.getElementById("nextIssueBtn");
const scrub = document.getElementById("scrub");
const speed = document.getElementById("speed");
let issueIndices = [];

function expandRuns(runs) {
  const values = [];
  for (const [start, len] of runs || []) {
    for (let i = 0; i < len; i++) values.push(start + i);
  }
  return values;
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(600, Math.round(rect.width * devicePixelRatio));
  canvas.height = Math.round(canvas.width * 2 / 3);
}

function drawGrid(ctx, canvas, runs, color, label) {
  resizeCanvas(canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#020617";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const cols = data.grid_cols, rows = data.grid_rows;
  const pad = 18 * devicePixelRatio;
  const cellW = (canvas.width - pad * 2) / cols;
  const cellH = (canvas.height - pad * 2) / rows;
  ctx.fillStyle = "rgba(30,41,59,.55)";
  for (let r = 0; r < rows; r += 4) {
    ctx.fillRect(pad, pad + r * cellH, canvas.width - pad * 2, Math.max(1, cellH * .12));
  }
  ctx.fillStyle = color;
  for (const idx of expandRuns(runs)) {
    const x = idx % cols;
    const y = Math.floor(idx / cols);
    ctx.fillRect(pad + x * cellW, pad + y * cellH, Math.max(1, cellW * .92), Math.max(1, cellH * .92));
  }
  ctx.fillStyle = "#cbd5e1";
  ctx.font = `${16 * devicePixelRatio}px Arial`;
  ctx.fillText(label || "", pad, canvas.height - 10 * devicePixelRatio);
}

function runsToSet(runs) {
  return new Set(expandRuns(runs));
}

function drawDiff(ctx, canvas, targetRuns, actualRuns) {
  resizeCanvas(canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#020617";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const cols = data.grid_cols, rows = data.grid_rows;
  const pad = 18 * devicePixelRatio;
  const cellW = (canvas.width - pad * 2) / cols;
  const cellH = (canvas.height - pad * 2) / rows;
  const target = runsToSet(targetRuns || []);
  const actual = runsToSet(actualRuns || []);
  const all = new Set([...target, ...actual]);
  for (const idx of all) {
    const inTarget = target.has(idx);
    const inActual = actual.has(idx);
    ctx.fillStyle = inTarget && inActual ? "#22c55e" : (inActual ? "#ef4444" : "#facc15");
    const x = idx % cols;
    const y = Math.floor(idx / cols);
    ctx.fillRect(pad + x * cellW, pad + y * cellH, Math.max(1, cellW * .92), Math.max(1, cellH * .92));
  }
  ctx.fillStyle = "#cbd5e1";
  ctx.font = `${16 * devicePixelRatio}px Arial`;
  ctx.fillText("green=match  red=extra  yellow=missing", pad, canvas.height - 10 * devicePixelRatio);
}

function isIssueFrame(f) {
  if (f.status === "dark") return false;
  return f.verdict === "bad";
}

function jumpIssue(direction) {
  if (!issueIndices.length) return;
  stop();
  const sorted = issueIndices;
  if (direction > 0) {
    const found = sorted.find(i => i > index);
    index = found ?? sorted[0];
  } else {
    const before = sorted.filter(i => i < index);
    index = before.length ? before[before.length - 1] : sorted[sorted.length - 1];
  }
  render();
}

function render() {
  const f = frames[index];
  const reviewTemplateId = f.review_template_id || f.expected_template_id || f.template_id || "";
  const target = reviewTemplateId && templates[reviewTemplateId] ? templates[reviewTemplateId] : null;
  const isDark = f.status === "dark";
  const ok = f.verdict === "ok" || f.verdict === "warn" || (f.status === "matched" && f.template_id === reviewTemplateId);
  const actualColor = isDark ? "#64748b" : (ok ? "#22c55e" : "#ef4444");
  if (target && target.image) {
    targetPhoto.style.display = "block";
    targetEmpty.style.display = "none";
    targetPhoto.src = target.image;
  } else {
    targetPhoto.style.display = "none";
    targetEmpty.style.display = "block";
  }
  if (f.real_image) {
    realPhoto.style.display = "block";
    realEmpty.style.display = "none";
    realPhoto.src = f.real_image;
  } else {
    realPhoto.style.display = "none";
    realEmpty.style.display = "block";
  }
  drawGrid(actualCtx, actualCanvas, f.runs || [], actualColor, f.status.toUpperCase());
  drawDiff(diffCtx, diffCanvas, target ? target.runs : [], f.runs || []);
  document.getElementById("timeText").textContent = `时间: ${f.time_sec.toFixed(2)}s / f${f.frame}`;
  document.getElementById("targetText").textContent = `复核模板: ${reviewTemplateId || "-"}`;
  document.getElementById("detectedText").textContent = `检测: ${f.template_id || "-"}`;
  document.getElementById("simText").textContent = `相似度: ${f.similarity.toFixed(2)}  微调: ${f.local_dx || 0},${f.local_dy || 0}`;
  document.getElementById("prevText").textContent = f.prev_template_id || "-";
  document.getElementById("currentText").textContent = f.expected_template_id || "-";
  document.getElementById("nextText").textContent = f.next_template_id || "-";
  const pill = document.getElementById("resultPill");
  pill.className = "pill " + (isDark ? "dark" : (f.verdict === "bad" ? "ng" : "ok"));
  pill.textContent = isDark ? "跳过黑帧" : (f.verdict === "bad" ? "需要复核" : (f.verdict === "warn" ? "已接受" : "匹配通过"));
  scrub.value = index;
}

function tick() {
  index = Math.min(frames.length - 1, index + 1);
  render();
  if (index >= frames.length - 1) stop();
}

function play() {
  if (playing) return;
  playing = true;
  playBtn.textContent = "暂停";
  timer = setInterval(tick, Math.max(35, 260 - Number(speed.value) * 22));
}

function stop() {
  playing = false;
  playBtn.textContent = "播放";
  if (timer) clearInterval(timer);
  timer = null;
}

function initPlayer(d) {
  data = d;
  frames = data.frames;
  templates = data.templates;
  issueIndices = frames.map((f, i) => isIssueFrame(f) ? i : -1).filter(i => i >= 0);
  document.getElementById("videoName").textContent = data.video;
  scrub.max = Math.max(0, frames.length - 1);
  render();
}

if (window.PLAYBACK_DATA) {
  initPlayer(window.PLAYBACK_DATA);
} else {
  document.getElementById("resultPill").textContent = "未加载到播放数据";
}

playBtn.onclick = () => playing ? stop() : play();
stepBtn.onclick = () => { stop(); index = Math.min(frames.length - 1, index + 1); render(); };
prevIssueBtn.onclick = () => jumpIssue(-1);
nextIssueBtn.onclick = () => jumpIssue(1);
scrub.oninput = () => { stop(); index = Number(scrub.value); render(); };
speed.oninput = () => { if (playing) { stop(); play(); } };
window.onresize = () => { if (frames) render(); };
</script>
</body>
</html>
"""
    (out_dir / "abstract_player.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect scan-refresh frame-template sequence")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="templates.json from extract_scan_frame_templates.py")
    parser.add_argument("--out", type=Path, default=Path("output") / "sevenseg_detect")
    parser.add_argument("--match-threshold", type=float, default=0.68)
    parser.add_argument("--active-threshold", type=int, default=55)
    parser.add_argument("--min-active-pixels", type=int, default=80)
    parser.add_argument("--grid-cols", type=int, default=96)
    parser.add_argument("--grid-rows", type=int, default=64)
    parser.add_argument("--cell-threshold", type=int, default=28)
    parser.add_argument("--tolerant-grid-dilate", type=int, default=4,
                        help="Fallback matching dilation on the downsampled mask; handles compressed MP4/anti-aliased edges")
    parser.add_argument("--tolerant-match-threshold", type=float, default=0.72,
                        help="Threshold used only by the tolerant fallback matcher")
    parser.add_argument("--overall-mode", choices=["sequence", "visual-or-sequence"], default="sequence",
                        help="sequence keeps strict timing; visual-or-sequence can pass compressed videos by full lit-area similarity")
    parser.add_argument("--sequence-mode", choices=["strict", "ordered-presence"], default="ordered-presence",
                        help="strict compares every template; ordered-presence only requires key templates to appear in order")
    parser.add_argument("--min-ordered-coverage", type=float, default=0.90,
                        help="Minimum expected-template coverage for ordered-presence mode")
    parser.add_argument("--max-unknown-segments", type=int, default=3,
                        help="Allowed unknown lit segments in ordered-presence mode")
    parser.add_argument("--overall-visual-threshold", type=float, default=0.80)
    parser.add_argument("--overall-visual-dilate", type=int, default=5)
    parser.add_argument("--align-grid-search", type=int, default=8,
                        help="Search +/- this many downsampled grid cells and align the video mask to the model")
    parser.add_argument("--align-grid-dilate", type=int, default=3)
    parser.add_argument("--local-align-grid-search", type=int, default=4,
                        help="After a template is selected, fine-align that frame to the selected template")
    parser.add_argument("--local-align-grid-dilate", type=int, default=2)
    parser.add_argument("--image-align-mode", choices=["off", "translation", "euclidean", "affine"], default="affine",
                        help="Register video ROI to the model reference image before matching")
    parser.add_argument("--max-gap-frames", type=int, default=2)
    parser.add_argument("--allowed-jitter-templates", type=int, default=3,
                        help="Allow short local repeat/backtrack caused by scan flicker")
    parser.add_argument("--min-template-count", type=int, default=1)
    parser.add_argument("--min-template-pixels", type=float, default=0.0)
    parser.add_argument("--playback-thumb-width", type=int, default=420)
    parser.add_argument("--no-playback-frames", action="store_true",
                        help="Skip per-frame real ROI thumbnails for faster batch checks")
    parser.add_argument("--roi-source", choices=["model", "auto"], default="model")
    parser.add_argument("--roi", type=str, default="", help="Override ROI as x,y,w,h")
    parser.add_argument("--projection-threshold", type=int, default=40)
    parser.add_argument("--roi-pad-ratio", type=float, default=0.06)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] Load model: {args.model}")
    templates, expected_sequence, model_summary = load_model(args.model, args.min_template_count, args.min_template_pixels)
    if not templates:
        raise RuntimeError("No templates available after filtering")
    template_matrix = np.vstack([bit_string_to_vector(t.bits) for t in templates])
    template_on_counts = template_matrix.sum(axis=1)
    template_vec_by_id = {tmpl.id: template_matrix[i] for i, tmpl in enumerate(templates)}
    tolerant_template_matrix = np.vstack([
        dilate_vector(bit_string_to_vector(t.bits), args.grid_cols, args.grid_rows, args.tolerant_grid_dilate)
        for t in templates
    ])
    tolerant_template_on_counts = tolerant_template_matrix.sum(axis=1)
    union_template_vec = np.any(template_matrix, axis=0)
    template_previews = copy_template_previews(args.model.parent, out_dir, templates)
    print(f"    templates: {len(templates)}, expected segments: {len(expected_sequence)}")

    print(f"[2] Build projection and choose ROI: {args.video}")
    max_projection, brightness_timeline, meta = first_pass_projection(args.video, args.active_threshold)
    roi = load_roi(model_summary, max_projection, args)
    raw_full_reconstruction = crop_roi(max_projection, roi)
    model_reference_path = args.model.parent / "full_reconstruction.jpg"
    model_reference = cv2.imread(str(model_reference_path)) if model_reference_path.exists() else None
    image_warp, image_align_mode, image_alignment_score = estimate_image_alignment(
        model_reference,
        raw_full_reconstruction,
        args.active_threshold,
        args.image_align_mode,
    )
    full_reconstruction = warp_to_reference(raw_full_reconstruction, image_warp)
    cv2.imwrite(str(out_dir / "full_reconstruction_raw.jpg"), raw_full_reconstruction)
    cv2.imwrite(str(out_dir / "full_reconstruction.jpg"), full_reconstruction)
    align_dx, align_dy, alignment_similarity = find_best_alignment_shift(
        mask_to_vector(make_mask(full_reconstruction, args.active_threshold), args.grid_cols, args.grid_rows, args.cell_threshold),
        union_template_vec,
        args.grid_cols,
        args.grid_rows,
        args.align_grid_search,
        args.align_grid_dilate,
    )
    full_vec = shift_vector(
        mask_to_vector(make_mask(full_reconstruction, args.active_threshold), args.grid_cols, args.grid_rows, args.cell_threshold),
        args.grid_cols,
        args.grid_rows,
        align_dx,
        align_dy,
    )
    aligned_full_vec = full_vec
    full_vec_cmp = dilate_vector(aligned_full_vec, args.grid_cols, args.grid_rows, args.overall_visual_dilate)
    union_vec_cmp = dilate_vector(union_template_vec, args.grid_cols, args.grid_rows, args.overall_visual_dilate)
    overall_visual_similarity = jaccard_similarity(full_vec_cmp, union_vec_cmp)
    overall_visual_pass = overall_visual_similarity >= args.overall_visual_threshold
    marked = max_projection.copy()
    cv2.rectangle(marked, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 255), 4)
    cv2.imwrite(str(out_dir / "roi_marked.jpg"), cv2.resize(marked, (612, 512), interpolation=cv2.INTER_AREA))
    print(f"    ROI: x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}")

    print("[3] Match frames")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = float(meta["fps"] or 0.0)
    segments: list[MatchSegment] = []
    frame_rows: list[dict[str, Any]] = []
    playback_frames: list[dict[str, Any]] = []
    expected_cursor = 0
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        crop = warp_to_reference(crop_roi(frame, roi), image_warp)
        mask = make_mask(crop, args.active_threshold)
        active_pixels = int(np.count_nonzero(mask))
        if active_pixels < args.min_active_pixels:
            expected_id = expected_sequence[expected_cursor] if expected_cursor < len(expected_sequence) else ""
            prev_id = expected_sequence[expected_cursor - 1] if expected_cursor > 0 else ""
            next_id = expected_sequence[expected_cursor + 1] if expected_cursor + 1 < len(expected_sequence) else ""
            display_crop = shift_crop_image(crop, align_dx, align_dy, args.grid_cols, args.grid_rows)
            real_image = maybe_save_playback_frame_image(
                out_dir, frame_index, display_crop, "dark", "", 0.0, args.playback_thumb_width, not args.no_playback_frames
            )
            append_segment(segments, "DARK", "dark", frame_index, fps, 0.0, active_pixels, args.max_gap_frames)
            frame_rows.append(
                {
                    "frame": frame_index,
                    "time_sec": frame_index / fps if fps > 0 else 0.0,
                    "status": "dark",
                    "template_id": "",
                "similarity": 0.0,
                "active_pixels": active_pixels,
                "match_mode": "",
                "local_dx": 0,
                "local_dy": 0,
            }
            )
            playback_frames.append(
                {
                    "frame": frame_index,
                    "time_sec": frame_index / fps if fps > 0 else 0.0,
                    "status": "dark",
                    "template_id": "",
                    "expected_template_id": expected_id,
                    "prev_template_id": prev_id,
                    "next_template_id": next_id,
                    "similarity": 0.0,
                    "active_pixels": active_pixels,
                    "real_image": real_image,
                "runs": [],
                "local_dx": 0,
                "local_dy": 0,
            }
            )
            frame_index += 1
            continue

        raw_frame_vec = mask_to_vector(mask, args.grid_cols, args.grid_rows, args.cell_threshold)
        frame_vec = shift_vector(raw_frame_vec, args.grid_cols, args.grid_rows, align_dx, align_dy)
        tmpl, sim = find_best_template(frame_vec, templates, template_matrix, template_on_counts)
        match_mode = "exact"
        local_dx = 0
        local_dy = 0
        local_alignment_similarity = sim
        if tmpl and sim >= args.match_threshold:
            template_id = tmpl.id
            status = "matched"
        else:
            tolerant_vec = dilate_vector(frame_vec, args.grid_cols, args.grid_rows, args.tolerant_grid_dilate)
            tolerant_tmpl, tolerant_sim = find_best_template(
                tolerant_vec,
                templates,
                tolerant_template_matrix,
                tolerant_template_on_counts,
            )
            if tolerant_tmpl and tolerant_sim >= args.tolerant_match_threshold:
                tmpl = tolerant_tmpl
                sim = tolerant_sim
                template_id = tmpl.id
                status = "matched"
                match_mode = f"tolerant_dilate_{args.tolerant_grid_dilate}"
            else:
                if tolerant_sim > sim:
                    tmpl = tolerant_tmpl
                    sim = tolerant_sim
                template_id = "UNKNOWN"
                status = "unknown"

        candidate_template_id = template_id if status == "matched" else (tmpl.id if tmpl else "")
        candidate_vec = template_vec_by_id.get(candidate_template_id)
        if candidate_vec is not None:
            local_dx, local_dy, local_alignment_similarity = find_best_alignment_shift(
                frame_vec,
                candidate_vec,
                args.grid_cols,
                args.grid_rows,
                args.local_align_grid_search,
                args.local_align_grid_dilate,
            )
            if local_dx or local_dy:
                frame_vec = shift_vector(frame_vec, args.grid_cols, args.grid_rows, local_dx, local_dy)
                exact_sim = jaccard_similarity(frame_vec, candidate_vec)
                tolerant_sim = jaccard_similarity(
                    dilate_vector(frame_vec, args.grid_cols, args.grid_rows, args.tolerant_grid_dilate),
                    dilate_vector(candidate_vec, args.grid_cols, args.grid_rows, args.tolerant_grid_dilate),
                )
                if exact_sim >= args.match_threshold:
                    template_id = candidate_template_id
                    status = "matched"
                    sim = exact_sim
                    match_mode = f"local_align_{local_dx}_{local_dy}"
                elif tolerant_sim >= args.tolerant_match_threshold:
                    template_id = candidate_template_id
                    status = "matched"
                    sim = tolerant_sim
                    match_mode = f"tolerant_local_align_{local_dx}_{local_dy}"
                else:
                    sim = max(sim, exact_sim, tolerant_sim, local_alignment_similarity)

        expected_id = expected_sequence[expected_cursor] if expected_cursor < len(expected_sequence) else ""
        prev_id = expected_sequence[expected_cursor - 1] if expected_cursor > 0 else ""
        next_id = expected_sequence[expected_cursor + 1] if expected_cursor + 1 < len(expected_sequence) else ""
        display_expected_id = expected_id
        if status == "matched":
            if template_id == expected_id:
                expected_cursor += 1
            else:
                jitter_start = max(0, expected_cursor - args.allowed_jitter_templates)
                if template_id in expected_sequence[jitter_start:expected_cursor]:
                    display_expected_id = template_id
                else:
                    later = -1
                    for i in range(expected_cursor + 1, len(expected_sequence)):
                        if expected_sequence[i] == template_id:
                            later = i
                            break
                    if later >= 0:
                        expected_cursor = later + 1

        display_crop = shift_crop_image(
            crop,
            align_dx + local_dx,
            align_dy + local_dy,
            args.grid_cols,
            args.grid_rows,
        )
        real_image = maybe_save_playback_frame_image(
            out_dir,
            frame_index,
            display_crop,
            status,
            template_id if status == "matched" else "",
            sim,
            args.playback_thumb_width,
            not args.no_playback_frames,
        )

        append_segment(segments, template_id, status, frame_index, fps, sim, active_pixels, args.max_gap_frames)
        frame_rows.append(
            {
                "frame": frame_index,
                "time_sec": frame_index / fps if fps > 0 else 0.0,
                "status": status,
                "template_id": template_id if status == "matched" else "",
                "similarity": sim,
                "active_pixels": active_pixels,
                "match_mode": match_mode if status == "matched" else "",
                "local_dx": local_dx,
                "local_dy": local_dy,
            }
        )
        playback_frames.append(
            {
                "frame": frame_index,
                "time_sec": frame_index / fps if fps > 0 else 0.0,
                "status": status,
                "template_id": template_id if status == "matched" else "",
                "expected_template_id": display_expected_id,
                "prev_template_id": prev_id,
                "next_template_id": next_id,
                "similarity": sim,
                "active_pixels": active_pixels,
                "real_image": real_image,
                "runs": vector_to_runs(frame_vec),
                "local_dx": local_dx,
                "local_dy": local_dy,
            }
        )
        frame_index += 1

    cap.release()

    print("[4] Save segment previews and compare sequence")
    frame_alignment = {
        int(row["frame"]): (int(row.get("local_dx", 0) or 0), int(row.get("local_dy", 0) or 0))
        for row in frame_rows
    }
    cap = cv2.VideoCapture(str(args.video))
    for seg in segments:
        if seg.status == "dark":
            seg.message = "黑帧/弱亮，已跳过"
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start_frame)
        ok, frame = cap.read()
        if not ok:
            continue
        color = (0, 200, 0) if seg.status == "matched" else (0, 0, 220)
        seg.image = save_segment_image(
            out_dir,
            seg.id,
            seg.start_frame,
            shift_crop_image(
                warp_to_reference(crop_roi(frame, roi), image_warp),
                align_dx + frame_alignment.get(seg.start_frame, (0, 0))[0],
                align_dy + frame_alignment.get(seg.start_frame, (0, 0))[1],
                args.grid_cols,
                args.grid_rows,
            ),
            f"{seg.status.upper()} {seg.template_id}",
            color,
        )
        seg.template_image = template_previews.get(seg.template_id, "")
    cap.release()

    if args.sequence_mode == "ordered-presence":
        events, sequence_check = compare_ordered_presence(
            segments,
            expected_sequence,
            args.min_ordered_coverage,
            args.max_unknown_segments,
        )
    else:
        events = compare_sequence(segments, expected_sequence, args.allowed_jitter_templates)
        sequence_check = {
            "mode": "strict",
            "status": "PASS" if not any(e.level == "error" for e in events) and not any(s.status == "unknown" for s in segments) else "FAIL",
        }
    timeline_image = create_timeline_image(out_dir, segments, int(meta["frame_count_read"]))
    segment_by_frame: dict[int, MatchSegment] = {}
    for seg in segments:
        for frame_no in range(seg.start_frame, seg.end_frame + 1):
            segment_by_frame[frame_no] = seg
    for frame_data in playback_frames:
        seg = segment_by_frame.get(int(frame_data["frame"]))
        if not seg:
            continue
        frame_data["verdict"] = seg.verdict
        frame_data["message"] = seg.message
        if args.sequence_mode == "ordered-presence" and seg.status == "matched":
            frame_data["review_template_id"] = seg.template_id
        else:
            frame_data["review_template_id"] = frame_data.get("expected_template_id", "")

    matched_frames = sum(r["status"] == "matched" for r in frame_rows)
    unknown_frames = sum(r["status"] == "unknown" for r in frame_rows)
    dark_frames = sum(r["status"] == "dark" for r in frame_rows)
    checked_frames = max(matched_frames + unknown_frames, 1)
    error_count = sum(1 for e in events if e.level == "error")
    warning_count = sum(1 for e in events if e.level == "warning")
    sequence_pass = sequence_check.get("status") == "PASS"
    visual_fallback_used = bool(args.overall_mode == "visual-or-sequence" and not sequence_pass and overall_visual_pass)
    overall = "PASS" if sequence_pass or visual_fallback_used else "FAIL"
    pass_basis = "sequence" if sequence_pass else ("overall_visual" if visual_fallback_used else "none")
    summary = {
        "overall_status": overall,
        "sequence_status": "PASS" if sequence_pass else "FAIL",
        "sequence_check": sequence_check,
        "pass_basis": pass_basis,
        "overall_visual_check": {
            "similarity": overall_visual_similarity,
            "threshold": args.overall_visual_threshold,
            "dilate": args.overall_visual_dilate,
            "status": "PASS" if overall_visual_pass else "FAIL",
        },
        "alignment": {
            "grid_dx": align_dx,
            "grid_dy": align_dy,
            "search": args.align_grid_search,
            "dilate": args.align_grid_dilate,
            "similarity": alignment_similarity,
        },
        "image_alignment": {
            "mode": image_align_mode,
            "score": image_alignment_score,
            "warp": image_warp.tolist(),
            "reference": str(model_reference_path),
        },
        "video": str(args.video),
        "model": str(args.model),
        "meta": meta,
        "roi": asdict(roi),
        "parameters": {
            "match_threshold": args.match_threshold,
            "active_threshold": args.active_threshold,
            "min_active_pixels": args.min_active_pixels,
            "grid_cols": args.grid_cols,
                "grid_rows": args.grid_rows,
                "cell_threshold": args.cell_threshold,
                "tolerant_grid_dilate": args.tolerant_grid_dilate,
                "tolerant_match_threshold": args.tolerant_match_threshold,
                "overall_mode": args.overall_mode,
                "sequence_mode": args.sequence_mode,
                "min_ordered_coverage": args.min_ordered_coverage,
                "max_unknown_segments": args.max_unknown_segments,
                "overall_visual_threshold": args.overall_visual_threshold,
                "overall_visual_dilate": args.overall_visual_dilate,
                "align_grid_search": args.align_grid_search,
                "align_grid_dilate": args.align_grid_dilate,
                "local_align_grid_search": args.local_align_grid_search,
                "local_align_grid_dilate": args.local_align_grid_dilate,
                "image_align_mode": args.image_align_mode,
                "max_gap_frames": args.max_gap_frames,
            "allowed_jitter_templates": args.allowed_jitter_templates,
            "min_template_count": args.min_template_count,
            "min_template_pixels": args.min_template_pixels,
            "playback_thumb_width": args.playback_thumb_width,
            "no_playback_frames": args.no_playback_frames,
            "roi_source": args.roi_source,
        },
        "template_count": len(templates),
        "expected_segment_count": len(expected_sequence),
        "detected_segment_count": len(segments),
        "matched_frames": matched_frames,
        "unknown_frames": unknown_frames,
        "dark_frames": dark_frames,
        "match_rate": matched_frames / checked_frames,
        "event_count": len(events),
        "error_count": error_count,
        "warning_count": warning_count,
        "outputs": {
            "review": "review.html",
            "abstract_player": "abstract_player.html",
            "playback_data": "playback_data.json",
            "playback_data_js": "playback_data.js",
            "timeline_overview": timeline_image,
            "segments": "segments.csv",
            "events": "events.json",
            "frame_matches": "frame_matches.csv",
        },
    }

    write_csv(
        out_dir / "frame_matches.csv",
        frame_rows,
        ["frame", "time_sec", "status", "template_id", "similarity", "active_pixels", "match_mode", "local_dx", "local_dy"],
    )
    write_csv(
        out_dir / "segments.csv",
        [asdict(s) for s in segments],
        [
            "id",
            "template_id",
            "status",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "frames",
            "mean_similarity",
            "active_pixels_mean",
            "image",
            "template_image",
            "verdict",
            "message",
        ],
    )
    write_csv(
        out_dir / "frame_brightness_timeline.csv",
        brightness_timeline,
        ["frame", "time_sec", "max_gray", "active_pixels", "active_ratio"],
    )
    write_json(out_dir / "events.json", {"events": [asdict(e) for e in events]})
    playback_payload = {
        "video": str(args.video),
        "fps": fps,
        "grid_cols": args.grid_cols,
        "grid_rows": args.grid_rows,
        "alignment": {
            "grid_dx": align_dx,
            "grid_dy": align_dy,
            "search": args.align_grid_search,
            "dilate": args.align_grid_dilate,
            "similarity": alignment_similarity,
        },
        "image_alignment": {
            "mode": image_align_mode,
            "score": image_alignment_score,
            "warp": image_warp.tolist(),
        },
        "templates": {
            tmpl.id: {
                "id": tmpl.id,
                "runs": bits_to_runs(tmpl.bits),
                "image": template_previews.get(tmpl.id, ""),
            }
            for tmpl in templates
        },
        "expected_sequence": expected_sequence,
        "frames": playback_frames,
    }
    write_json(out_dir / "playback_data.json", playback_payload)
    (out_dir / "playback_data.js").write_text(
        "window.PLAYBACK_DATA = " + json.dumps(playback_payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    write_json(out_dir / "summary.json", summary)
    write_review_html(out_dir, args.video.name, summary, segments, events, timeline_image)
    write_abstract_player_html(out_dir, args.video.name)

    print(f"[5] Done: {out_dir}")
    print(f"    result: {overall}")
    print(f"    matched frames: {matched_frames}, unknown frames: {unknown_frames}, dark frames: {dark_frames}")
    print(f"    events: {len(events)}")
    print(f"    review: {out_dir / 'review.html'}")


if __name__ == "__main__":
    main()
