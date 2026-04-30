from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cv_pattern_probe as cvp
import cv_template_probe as cv_template


DEFAULT_VIDEO = Path(r"D:\jiemi\Video_20260421163145108.avi")
DEFAULT_TEMPLATE = ROOT / "models" / "cv_templates" / "templates.json"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "video_template_multiframe_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CV template multi-frame probing on the middle clip of a video.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    parser.add_argument("--window-stride-ms", type=int, default=500)
    parser.add_argument("--multi-frame-count", type=int, default=7)
    parser.add_argument("--multi-frame-interval-ms", type=int, default=45)
    parser.add_argument("--slot-radius", type=int, default=9)
    parser.add_argument("--slot-threshold", type=float, default=0.42)
    parser.add_argument("--extra-penalty", type=float, default=0.65)
    parser.add_argument("--visual-max-width", type=int, default=1600)
    parser.add_argument("--contact-sheet-cols", type=int, default=4)
    parser.add_argument("--visual-mode", choices=("all", "anomaly"), default="all")
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(make_jsonable(value), ensure_ascii=False, separators=(",", ":"))


def make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resize_for_visual(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    target = (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale))))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    scale: float = 0.65,
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(image.shape[1] - tw - 4, x))
    y = max(th + 4, min(image.shape[0] - baseline - 4, y))
    cv2.rectangle(image, (x, y - th - 6), (x + tw + 6, y + baseline + 5), (18, 18, 18), -1)
    cv2.putText(image, text, (x + 3, y), font, scale, color, thickness, cv2.LINE_AA)


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def window_frame_indices(center_frame: int, fps: float, count: int, interval_ms: int, total_frames: int) -> list[int]:
    if count <= 1:
        return [min(total_frames - 1, max(0, center_frame))]
    interval_frames = max(1, int(round(fps * interval_ms / 1000.0)))
    half = (count - 1) / 2.0
    indices = []
    for offset in range(count):
        index = int(round(center_frame + (offset - half) * interval_frames))
        indices.append(min(total_frames - 1, max(0, index)))
    return sorted(set(indices))


def analyze_frame(
    frame: np.ndarray,
    templates: dict[str, dict[int, cv_template.SlotTemplate]],
    slot_radius: int,
    slot_threshold: float,
    extra_penalty: float,
) -> dict[str, Any]:
    boxes, _ = cvp.detect_roi_boxes(frame)
    height, width = frame.shape[:2]
    items: list[dict[str, Any]] = []

    for roi_index, box in enumerate(boxes[:20]):
        x1, y1, x2, y2 = box.xyxy
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        result = cv_template.match_roi(
            roi=roi,
            roi_index=roi_index,
            templates=templates,
            slot_radius=slot_radius,
            slot_threshold=slot_threshold,
            extra_penalty=extra_penalty,
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


def fuse_window(
    window_id: int,
    center_time_sec: float,
    center_frame: int,
    frame_payloads: list[dict[str, Any]],
    frame_indices: list[int],
    templates: dict[str, dict[int, cv_template.SlotTemplate]],
    requested_frame_count: int,
    interval_ms: int,
) -> dict[str, Any]:
    frames_captured = len(frame_payloads)
    valid_frame_count = sum(1 for payload in frame_payloads if payload["valid"])
    min_observations = max(1, min(3, (frames_captured + 1) // 2))
    items_by_roi: dict[int, list[dict[str, Any]]] = {index: [] for index in range(20)}

    for payload in frame_payloads:
        for item in payload["items"]:
            roi_index = int(item["roi_index"])
            if 0 <= roi_index < 20:
                items_by_roi[roi_index].append(item)

    rois: list[dict[str, Any]] = []
    for roi_index in range(20):
        observations = items_by_roi[roi_index]
        if not observations:
            rois.append(
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
                    "class_scores": {},
                    "per_frame_point_counts": [],
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
        expected = cvp.EXPECTED_POINT_COUNTS.get(class_name)
        slot_weights = [0.0 for _ in template.points]
        extra_votes: dict[int, float] = {}
        total_weight = 0.0
        score_sum = 0.0
        per_frame_point_counts: list[int] = []

        for item in observations:
            result = item["result"]
            detail = result["class_details"].get(class_name, result)
            weight = max(0.05, float(detail["class_score"]))
            if detail["count_status"] == "ok":
                weight += 0.10
            total_weight += weight
            score_sum += float(detail["class_score"])
            per_frame_point_counts.append(int(detail["point_count"]))
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
        rois.append(
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
                "class_scores": {name: round(value, 4) for name, value in class_votes.items()},
                "per_frame_point_counts": per_frame_point_counts,
            }
        )

    roi_count = sum(1 for roi in rois if roi["observations"] > 0)
    class_names = [roi["class_name"] for roi in rois if roi["class_name"] != "unknown"]
    roi_class_consistent = len(set(class_names)) == 1 and roi_count == 20
    roi_class = class_names[0] if roi_class_consistent else ""
    expected_points = cvp.EXPECTED_POINT_COUNTS.get(roi_class)
    abnormal_rois = [roi for roi in rois if not roi["ok"]]
    overall_ok = bool(
        valid_frame_count >= 1
        and roi_count == 20
        and roi_class_consistent
        and expected_points is not None
        and not abnormal_rois
    )
    return {
        "window_id": window_id,
        "center_time_sec": round(center_time_sec, 3),
        "center_frame": center_frame,
        "frame_indices": frame_indices,
        "frames_requested": requested_frame_count,
        "frames_captured": frames_captured,
        "frame_interval_ms": interval_ms,
        "valid_frame_count": valid_frame_count,
        "min_observations_per_roi": min_observations,
        "roi_count": roi_count,
        "roi_count_ok": roi_count == 20,
        "roi_class_consistent": roi_class_consistent,
        "roi_class": roi_class,
        "expected_points_per_roi": expected_points,
        "abnormal_roi_count": len(abnormal_rois),
        "missing_slot_total": sum(len(roi["missing_slots"]) for roi in abnormal_rois),
        "extra_slot_total": sum(len(roi["extra_slots"]) for roi in abnormal_rois),
        "overall_ok": overall_ok,
        "rois": rois,
    }


def choose_base_frame(frames: list[np.ndarray], frame_payloads: list[dict[str, Any]]) -> np.ndarray:
    best_index = 0
    best_score = -1
    for index, payload in enumerate(frame_payloads):
        score = int(payload["valid"]) * 100 + int(payload["roi_count"]) - int(payload["abnormal_count"]) * 2
        if score > best_score:
            best_index = index
            best_score = score
    return frames[min(best_index, len(frames) - 1)].copy()


def draw_fused_result(
    frame: np.ndarray,
    window: dict[str, Any],
    templates: dict[str, dict[int, cv_template.SlotTemplate]],
) -> np.ndarray:
    image = frame.copy()
    for roi in window["rois"]:
        bbox = roi.get("bbox") or []
        class_name = roi["class_name"]
        template = templates.get(class_name, {}).get(int(roi["index"]))
        if len(bbox) != 4 or template is None:
            continue
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(image.shape[1] - 1, x1))
        y1 = max(0, min(image.shape[0] - 1, y1))
        x2 = max(0, min(image.shape[1], x2))
        y2 = max(0, min(image.shape[0], y2))
        if x2 <= x1 or y2 <= y1:
            continue
        box = cvp.RoiBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
        draw_result = {
            "class_name": class_name,
            "point_count": int(roi["point_count"]),
            "expected_point_count": roi["expected_count"],
            "class_score": float(roi["confidence"]),
            "hits": roi["hits"],
            "slot_offset": (0, 0),
        }
        cv_template.draw_roi_slots(image, box, template, draw_result)
        if not roi["ok"]:
            draw_label(
                image,
                f"NG obs={roi['observations']} miss={len(roi['missing_slots'])} extra={len(roi['extra_slots'])}",
                x1,
                min(image.shape[0] - 8, y2 + 18),
                (0, 0, 255),
                scale=0.5,
            )

    header_color = (0, 200, 0) if window["overall_ok"] else (0, 0, 255)
    draw_label(
        image,
        (
            f"W{window['window_id']:03d} {'PASS' if window['overall_ok'] else 'FAIL'} "
            f"t={window['center_time_sec']:.2f}s frames={window['valid_frame_count']}/{window['frames_captured']} "
            f"ROI={window['roi_count']}/20 class={window['roi_class'] or 'mixed'}"
        ),
        12,
        32,
        header_color,
        scale=0.85,
        thickness=2,
    )
    return image


def make_contact_sheet(image_paths: list[Path], output_path: Path, cols: int, thumb_width: int = 420) -> None:
    if not image_paths:
        return
    thumbs: list[np.ndarray] = []
    for path in image_paths:
        image = cvp.imread_unicode(path)
        if image is None:
            continue
        scale = thumb_width / max(1, image.shape[1])
        thumb = cv2.resize(
            image,
            (thumb_width, max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        thumbs.append(thumb)
    if not thumbs:
        return

    cols = max(1, cols)
    rows = int(math.ceil(len(thumbs) / cols))
    cell_h = max(thumb.shape[0] for thumb in thumbs)
    sheet = np.full((rows * cell_h, cols * thumb_width, 3), 28, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row = index // cols
        col = index % cols
        y = row * cell_h
        x = col * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    cvp.imwrite_unicode(output_path, sheet, quality=90)


def write_annotated_video(image_paths: list[Path], output_path: Path, fps: float = 2.0) -> None:
    if not image_paths:
        return
    first = cvp.imread_unicode(image_paths[0])
    if first is None:
        return
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for path in image_paths:
            image = cvp.imread_unicode(path)
            if image is None:
                continue
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"video not found: {args.video}")
    if not args.template.exists():
        raise FileNotFoundError(f"template not found: {args.template}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or total_frames <= 0:
            raise RuntimeError("video fps/frame count is invalid")

        duration_sec = total_frames / fps
        clip_seconds = max(0.1, min(float(args.clip_seconds), duration_sec))
        clip_start_sec = max(0.0, duration_sec / 2.0 - clip_seconds / 2.0)
        clip_end_sec = min(duration_sec, clip_start_sec + clip_seconds)
        stride_sec = max(0.001, args.window_stride_ms / 1000.0)
        frame_count = max(1, int(args.multi_frame_count))
        if frame_count % 2 == 0:
            frame_count += 1

        run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        visual_dir = run_dir / "visualized"
        visual_dir.mkdir(parents=True, exist_ok=True)

        templates = cv_template.load_templates(args.template)
        window_times: list[float] = []
        current = clip_start_sec
        while current <= clip_end_sec + 1e-6:
            window_times.append(current)
            current += stride_sec
        if not window_times or window_times[-1] < clip_end_sec - stride_sec * 0.5:
            window_times.append(clip_end_sec)

        metadata = {
            "video": str(args.video),
            "template": str(args.template),
            "output_dir": str(run_dir),
            "fps": fps,
            "frame_count": total_frames,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
            "clip_start_sec": clip_start_sec,
            "clip_end_sec": clip_end_sec,
            "clip_seconds": clip_end_sec - clip_start_sec,
            "window_stride_ms": args.window_stride_ms,
            "window_count": len(window_times),
            "multi_frame_count": frame_count,
            "multi_frame_interval_ms": args.multi_frame_interval_ms,
            "slot_radius": args.slot_radius,
            "slot_threshold": args.slot_threshold,
            "extra_penalty": args.extra_penalty,
        }

        window_rows: list[dict[str, Any]] = []
        roi_rows: list[dict[str, Any]] = []
        frame_rows: list[dict[str, Any]] = []
        anomaly_rows: list[dict[str, Any]] = []
        window_payloads: list[dict[str, Any]] = []
        visual_paths: list[Path] = []

        print(f"video: {args.video}")
        print(f"fps={fps:.3f} frames={total_frames} size={width}x{height} duration={duration_sec:.3f}s")
        print(f"clip: {clip_start_sec:.3f}s -> {clip_end_sec:.3f}s, windows={len(window_times)}")
        print(f"output: {run_dir}")

        for window_id, center_time_sec in enumerate(window_times, start=1):
            center_frame = min(total_frames - 1, max(0, int(round(center_time_sec * fps))))
            indices = window_frame_indices(
                center_frame=center_frame,
                fps=fps,
                count=frame_count,
                interval_ms=args.multi_frame_interval_ms,
                total_frames=total_frames,
            )
            frames: list[np.ndarray] = []
            frame_payloads: list[dict[str, Any]] = []
            for sub_index, frame_index in enumerate(indices):
                frame = read_frame(cap, frame_index)
                if frame is None:
                    continue
                frames.append(frame)
                frame_payload = analyze_frame(
                    frame,
                    templates,
                    slot_radius=args.slot_radius,
                    slot_threshold=args.slot_threshold,
                    extra_penalty=args.extra_penalty,
                )
                frame_payloads.append(frame_payload)
                frame_rows.append(
                    {
                        "window_id": window_id,
                        "subframe_index": sub_index,
                        "frame_index": frame_index,
                        "timestamp_sec": round(frame_index / fps, 3),
                        "roi_count": frame_payload["roi_count"],
                        "valid": int(frame_payload["valid"]),
                        "class_counts": json_dumps(frame_payload["class_counts"]),
                        "abnormal_count": frame_payload["abnormal_count"],
                    }
                )

            if not frames:
                continue

            window = fuse_window(
                window_id=window_id,
                center_time_sec=center_time_sec,
                center_frame=center_frame,
                frame_payloads=frame_payloads,
                frame_indices=indices,
                templates=templates,
                requested_frame_count=frame_count,
                interval_ms=args.multi_frame_interval_ms,
            )
            window_payloads.append(window)

            window_rows.append(
                {
                    "window_id": window_id,
                    "center_time_sec": window["center_time_sec"],
                    "center_frame": center_frame,
                    "frame_indices": json_dumps(indices),
                    "frames_captured": window["frames_captured"],
                    "valid_frame_count": window["valid_frame_count"],
                    "roi_count": window["roi_count"],
                    "roi_count_ok": int(window["roi_count_ok"]),
                    "roi_class": window["roi_class"],
                    "roi_class_consistent": int(window["roi_class_consistent"]),
                    "expected_points_per_roi": window["expected_points_per_roi"],
                    "abnormal_roi_count": window["abnormal_roi_count"],
                    "missing_slot_total": window["missing_slot_total"],
                    "extra_slot_total": window["extra_slot_total"],
                    "overall_ok": int(window["overall_ok"]),
                }
            )

            for roi in window["rois"]:
                row = {
                    "window_id": window_id,
                    "center_time_sec": window["center_time_sec"],
                    "roi_index": roi["index"],
                    "class_name": roi["class_name"],
                    "point_count": roi["point_count"],
                    "expected_count": roi["expected_count"],
                    "ok": int(roi["ok"]),
                    "observations": roi["observations"],
                    "confidence": roi["confidence"],
                    "bbox": json_dumps(roi["bbox"]),
                    "missing_slots": json_dumps(roi["missing_slots"]),
                    "extra_slots": json_dumps(roi["extra_slots"]),
                    "class_scores": json_dumps(roi["class_scores"]),
                    "per_frame_point_counts": json_dumps(roi["per_frame_point_counts"]),
                }
                roi_rows.append(row)
                if not roi["ok"]:
                    anomaly_rows.append(row)

            save_visual = args.visual_mode == "all" or not window["overall_ok"]
            if save_visual:
                base_frame = choose_base_frame(frames, frame_payloads)
                visual = draw_fused_result(base_frame, window, templates)
                visual = resize_for_visual(visual, args.visual_max_width)
                visual_path = visual_dir / f"window_{window_id:03d}_{'PASS' if window['overall_ok'] else 'FAIL'}_{window['center_time_sec']:.2f}s.jpg"
                cvp.imwrite_unicode(visual_path, visual, quality=90)
                visual_paths.append(visual_path)

            print(
                f"[{window_id:03d}/{len(window_times):03d}] "
                f"t={center_time_sec:.2f}s "
                f"{'PASS' if window['overall_ok'] else 'FAIL'} "
                f"frames={window['valid_frame_count']}/{window['frames_captured']} "
                f"roi={window['roi_count']}/20 class={window['roi_class'] or 'mixed'} "
                f"abn={window['abnormal_roi_count']}"
            )

        metadata["processed_windows"] = len(window_rows)
        metadata["pass_windows"] = sum(1 for row in window_rows if row["overall_ok"])
        metadata["fail_windows"] = len(window_rows) - metadata["pass_windows"]
        metadata["anomaly_roi_rows"] = len(anomaly_rows)
        (run_dir / "metadata.json").write_text(json.dumps(make_jsonable(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "windows.json").write_text(json.dumps(make_jsonable(window_payloads), ensure_ascii=False, indent=2), encoding="utf-8")

        write_csv(
            run_dir / "window_summary.csv",
            window_rows,
            [
                "window_id",
                "center_time_sec",
                "center_frame",
                "frame_indices",
                "frames_captured",
                "valid_frame_count",
                "roi_count",
                "roi_count_ok",
                "roi_class",
                "roi_class_consistent",
                "expected_points_per_roi",
                "abnormal_roi_count",
                "missing_slot_total",
                "extra_slot_total",
                "overall_ok",
            ],
        )
        write_csv(
            run_dir / "roi_summary.csv",
            roi_rows,
            [
                "window_id",
                "center_time_sec",
                "roi_index",
                "class_name",
                "point_count",
                "expected_count",
                "ok",
                "observations",
                "confidence",
                "bbox",
                "missing_slots",
                "extra_slots",
                "class_scores",
                "per_frame_point_counts",
            ],
        )
        write_csv(
            run_dir / "frame_summary.csv",
            frame_rows,
            [
                "window_id",
                "subframe_index",
                "frame_index",
                "timestamp_sec",
                "roi_count",
                "valid",
                "class_counts",
                "abnormal_count",
            ],
        )
        write_csv(
            run_dir / "anomalies.csv",
            anomaly_rows,
            [
                "window_id",
                "center_time_sec",
                "roi_index",
                "class_name",
                "point_count",
                "expected_count",
                "ok",
                "observations",
                "confidence",
                "bbox",
                "missing_slots",
                "extra_slots",
                "class_scores",
                "per_frame_point_counts",
            ],
        )

        make_contact_sheet(visual_paths, run_dir / "contact_sheet.jpg", args.contact_sheet_cols)
        if not args.no_video:
            write_annotated_video(visual_paths, run_dir / "annotated_windows.mp4", fps=2.0)

        print("done")
        print(f"metadata: {run_dir / 'metadata.json'}")
        print(f"window summary: {run_dir / 'window_summary.csv'}")
        print(f"roi summary: {run_dir / 'roi_summary.csv'}")
        print(f"frame summary: {run_dir / 'frame_summary.csv'}")
        print(f"anomalies: {run_dir / 'anomalies.csv'}")
        print(f"visualized: {visual_dir}")
        print(f"contact sheet: {run_dir / 'contact_sheet.jpg'}")
        if not args.no_video:
            print(f"annotated video: {run_dir / 'annotated_windows.mp4'}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
