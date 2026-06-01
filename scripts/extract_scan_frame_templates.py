"""
extract_scan_frame_templates

Build a key-frame template library and compressed temporal sequence from a
scan-refresh seven-segment display video.

This is different from temporal composition: each source frame is treated as a
possible scan phase. Dark/weak frames are skipped, lit frames are clustered by
lit-mask overlap, and the resulting template sequence can be used for later
order checks.

Example:
  python scripts/extract_scan_frame_templates.py ^
    --video D:/jiemi/七段管/Video_20260515112415516.avi ^
    --out output/sevenseg_frame_templates/Video_20260515112415516
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.scan_refresh_seven_segment_probe import (  # noqa: E402
    Roi,
    crop_roi,
    detect_display_roi,
    first_pass_projection,
    make_contact_sheet,
    write_csv,
    write_json,
)


@dataclass
class FrameTemplate:
    id: str
    first_frame: int
    first_time: float
    last_frame: int
    last_time: float
    count: int
    active_pixels_median: float
    active_pixels_max: int
    bits: str
    color_bits: str
    color_counts: dict[str, int]
    image: str
    mask_image: str


@dataclass
class SequenceSegment:
    template_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    frames: int
    mean_similarity: float


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def bit_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    inter = 0
    union = 0
    for i in range(n):
        av = a[i] == "1"
        bv = b[i] == "1"
        inter += int(av and bv)
        union += int(av or bv)
    return inter / max(union, 1)


def mask_to_bits(mask: np.ndarray, cols: int, rows: int, cell_threshold: int) -> str:
    small = cv2.resize(mask.astype(np.uint8) * 255, (cols, rows), interpolation=cv2.INTER_AREA)
    return "".join("1" if int(v) >= cell_threshold else "0" for v in small.reshape(-1))


def classify_hsv_cell(hue: float, sat: float, val: float) -> str:
    if val < 45:
        return "0"
    if sat < 45:
        return "W"
    if hue < 10 or hue >= 165:
        return "R"
    if hue < 35:
        return "Y"
    if hue < 85:
        return "G"
    if hue < 105:
        return "C"
    if hue < 140:
        return "B"
    return "M"


def mask_to_color_bits(crop: np.ndarray, mask: np.ndarray, cols: int, rows: int, cell_threshold: int) -> tuple[str, dict[str, int]]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    height, width = mask.shape[:2]
    chars: list[str] = []
    counts: dict[str, int] = {}
    for row in range(rows):
        y1 = int(round(row * height / rows))
        y2 = int(round((row + 1) * height / rows))
        for col in range(cols):
            x1 = int(round(col * width / cols))
            x2 = int(round((col + 1) * width / cols))
            cell_mask = mask[y1:y2, x1:x2]
            if cell_mask.size == 0 or int(np.count_nonzero(cell_mask)) * 255 < cell_threshold * max(cell_mask.size, 1):
                chars.append("0")
                continue
            active = cell_mask > 0
            code = classify_hsv_cell(
                float(h[y1:y2, x1:x2][active].mean()),
                float(s[y1:y2, x1:x2][active].mean()),
                float(v[y1:y2, x1:x2][active].mean()),
            )
            chars.append(code)
            counts[code] = counts.get(code, 0) + 1
    return "".join(chars), counts


def make_mask(crop: np.ndarray, active_threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    mask = gray > active_threshold
    # Remove isolated sensor specks without erasing thin seven-segment strokes.
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask_u8 > 0


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    out[mask] = (255, 255, 255)
    return out


def color_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    total = 0
    matched = 0
    for i in range(n):
        av = a[i]
        bv = b[i]
        if av == "0" and bv == "0":
            continue
        total += 1
        matched += int(av == bv)
    return matched / max(total, 1)


def find_best_template(bits: str, color_bits: str, templates: list[FrameTemplate], color_threshold: float) -> tuple[int, float, float]:
    best_i = -1
    best_sim = 0.0
    best_color = 0.0
    for i, tmpl in enumerate(templates):
        sim = bit_jaccard(bits, tmpl.bits)
        c_sim = color_similarity(color_bits, tmpl.color_bits) if color_threshold > 0 else 1.0
        combined = min(sim, c_sim)
        if combined > best_sim:
            best_i = i
            best_sim = combined
            best_color = c_sim
    return best_i, best_sim, best_color


def save_template_assets(
    template_id: str,
    frame_index: int,
    crop: np.ndarray,
    mask: np.ndarray,
    templates_dir: Path,
) -> tuple[str, str]:
    image_name = f"{template_id}_f{frame_index:04d}.jpg"
    mask_name = f"{template_id}_f{frame_index:04d}_mask.jpg"
    annotated = crop.copy()
    cv2.putText(
        annotated,
        f"{template_id}  f{frame_index}",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(templates_dir / image_name), annotated)
    cv2.imwrite(str(templates_dir / mask_name), colorize_mask(mask))
    return f"templates/{image_name}", f"templates/{mask_name}"


def append_sequence_segment(
    segments: list[SequenceSegment],
    template_id: str,
    frame_index: int,
    fps: float,
    similarity: float,
    max_gap_frames: int,
) -> None:
    if (
        segments
        and segments[-1].template_id == template_id
        and frame_index - segments[-1].end_frame <= max_gap_frames
    ):
        prev = segments[-1]
        total_frames = prev.frames + 1
        prev.mean_similarity = ((prev.mean_similarity * prev.frames) + similarity) / total_frames
        prev.end_frame = frame_index
        prev.end_time = frame_index / fps if fps > 0 else 0.0
        prev.frames = total_frames
        return

    segments.append(
        SequenceSegment(
            template_id=template_id,
            start_frame=frame_index,
            end_frame=frame_index,
            start_time=frame_index / fps if fps > 0 else 0.0,
            end_time=frame_index / fps if fps > 0 else 0.0,
            frames=1,
            mean_similarity=similarity,
        )
    )


def write_review_html(out_dir: Path, meta: dict[str, Any], roi: Roi, templates: list[FrameTemplate]) -> None:
    cards = []
    for tmpl in templates:
        cards.append(
            f"""
            <figure>
              <img src="{tmpl.image}" loading="lazy">
              <figcaption>{tmpl.id} | first f{tmpl.first_frame} | count {tmpl.count}</figcaption>
            </figure>
            """
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Frame template sequence review</title>
  <style>
    body {{ margin: 0; background: #101010; color: #eee; font-family: Arial, "Microsoft YaHei", sans-serif; }}
    header {{ padding: 20px 28px; border-bottom: 1px solid #333; }}
    main {{ padding: 20px 28px 40px; }}
    img {{ max-width: 100%; display: block; background: #000; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; border: 1px solid #333; background: #050505; }}
    figcaption {{ padding: 8px 10px; color: #ccc; font-size: 13px; }}
    code {{ color: #9cdcfe; }}
    .meta {{ line-height: 1.7; color: #ccc; }}
  </style>
</head>
<body>
  <header>
    <h1>逐帧关键模板复核</h1>
    <div class="meta">
      视频: <code>{Path(str(meta['video'])).name}</code><br>
      帧率: {meta['fps']:.3f} FPS, 帧数: {meta['frame_count_read']}, 时长: {meta['duration_sec']:.2f}s<br>
      ROI: x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}<br>
      模板数: {len(templates)}
    </div>
  </header>
  <main>
    <h2>全视频最大投影参考</h2>
    <p><img src="full_reconstruction.jpg"></p>
    <h2>逐帧模板</h2>
    <div class="grid">{''.join(cards)}</div>
  </main>
</body>
</html>
"""
    (out_dir / "review.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract scan-refresh frame templates and sequence")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output") / "sevenseg_frame_templates")
    parser.add_argument("--projection-threshold", type=int, default=40)
    parser.add_argument("--active-threshold", type=int, default=55)
    parser.add_argument("--roi-pad-ratio", type=float, default=0.06)
    parser.add_argument("--stride-frames", type=int, default=1)
    parser.add_argument("--min-active-pixels", type=int, default=80)
    parser.add_argument("--grid-cols", type=int, default=96)
    parser.add_argument("--grid-rows", type=int, default=64)
    parser.add_argument("--cell-threshold", type=int, default=28)
    parser.add_argument("--template-threshold", type=float, default=0.72)
    parser.add_argument("--color-threshold", type=float, default=0.0)
    parser.add_argument("--max-gap-frames", type=int, default=2)
    parser.add_argument("--max-templates", type=int, default=260)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    templates_dir = out_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] Build projection and ROI: {args.video}")
    max_projection, frame_brightness, meta = first_pass_projection(args.video, args.active_threshold)
    roi = detect_display_roi(max_projection, args.projection_threshold, args.roi_pad_ratio)
    full_reconstruction = crop_roi(max_projection, roi)
    cv2.imwrite(str(out_dir / "full_reconstruction.jpg"), full_reconstruction)

    marked = max_projection.copy()
    cv2.rectangle(marked, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 255), 4)
    cv2.imwrite(str(out_dir / "roi_marked.jpg"), cv2.resize(marked, (612, 512), interpolation=cv2.INTER_AREA))
    print(f"    ROI: x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}")

    print("[2] Stream frames and cluster templates")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = float(meta["fps"] or 0.0)
    templates: list[FrameTemplate] = []
    active_pixel_samples: list[list[int]] = []
    frame_rows: list[dict[str, Any]] = []
    segments: list[SequenceSegment] = []

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(args.stride_frames, 1) != 0:
            frame_index += 1
            continue

        crop = crop_roi(frame, roi)
        mask = make_mask(crop, args.active_threshold)
        active_pixels = int(np.count_nonzero(mask))
        if active_pixels < args.min_active_pixels:
            frame_rows.append(
                {
                    "frame": frame_index,
                    "time_sec": frame_index / fps if fps > 0 else 0.0,
                    "active_pixels": active_pixels,
                    "template_id": "",
                    "similarity": 0.0,
                    "status": "dark",
                }
            )
            frame_index += 1
            continue

        bits = mask_to_bits(mask, args.grid_cols, args.grid_rows, args.cell_threshold)
        color_bits, color_counts = mask_to_color_bits(crop, mask, args.grid_cols, args.grid_rows, args.cell_threshold)
        best_i, best_sim, best_color = find_best_template(bits, color_bits, templates, args.color_threshold)

        if best_i >= 0 and best_sim >= args.template_threshold and best_color >= args.color_threshold:
            tmpl = templates[best_i]
            samples = active_pixel_samples[best_i]
            samples.append(active_pixels)
            tmpl.count += 1
            tmpl.last_frame = frame_index
            tmpl.last_time = frame_index / fps if fps > 0 else 0.0
            tmpl.active_pixels_median = float(np.median(samples))
            tmpl.active_pixels_max = max(tmpl.active_pixels_max, active_pixels)
            template_id = tmpl.id
            similarity = best_sim
        elif len(templates) < args.max_templates:
            template_id = f"T{len(templates) + 1:03d}"
            image, mask_image = save_template_assets(template_id, frame_index, crop, mask, templates_dir)
            templates.append(
                FrameTemplate(
                    id=template_id,
                    first_frame=frame_index,
                    first_time=frame_index / fps if fps > 0 else 0.0,
                    last_frame=frame_index,
                    last_time=frame_index / fps if fps > 0 else 0.0,
                    count=1,
                    active_pixels_median=float(active_pixels),
                    active_pixels_max=active_pixels,
                    bits=bits,
                    color_bits=color_bits,
                    color_counts=color_counts,
                    image=image,
                    mask_image=mask_image,
                )
            )
            active_pixel_samples.append([active_pixels])
            similarity = 1.0
        else:
            template_id = "UNMATCHED"
            similarity = best_sim

        append_sequence_segment(segments, template_id, frame_index, fps, similarity, args.max_gap_frames)
        frame_rows.append(
            {
                "frame": frame_index,
                "time_sec": frame_index / fps if fps > 0 else 0.0,
                "active_pixels": active_pixels,
                "template_id": template_id,
                "similarity": similarity,
                "status": "matched",
            }
        )
        frame_index += 1

    cap.release()

    templates_sorted = sorted(templates, key=lambda t: (t.first_frame, t.id))
    write_csv(
        out_dir / "frame_matches.csv",
        frame_rows,
        ["frame", "time_sec", "active_pixels", "template_id", "similarity", "status"],
    )
    write_csv(
        out_dir / "sequence.csv",
        [asdict(s) for s in segments],
        ["template_id", "start_frame", "end_frame", "start_time", "end_time", "frames", "mean_similarity"],
    )
    write_json(
        out_dir / "templates.json",
        {
            "templates": [asdict(t) for t in templates_sorted],
            "sequence": [asdict(s) for s in segments],
        },
    )

    template_images = [out_dir / t.image for t in templates_sorted]
    make_contact_sheet(template_images, out_dir / "template_contact.jpg", cols=4, thumb_w=300)

    summary = {
        "meta": meta,
        "roi": asdict(roi),
        "parameters": {
            "projection_threshold": args.projection_threshold,
            "active_threshold": args.active_threshold,
            "roi_pad_ratio": args.roi_pad_ratio,
            "stride_frames": args.stride_frames,
            "min_active_pixels": args.min_active_pixels,
            "grid_cols": args.grid_cols,
            "grid_rows": args.grid_rows,
            "cell_threshold": args.cell_threshold,
            "template_threshold": args.template_threshold,
            "color_threshold": args.color_threshold,
            "max_gap_frames": args.max_gap_frames,
            "max_templates": args.max_templates,
        },
        "processed_frames": len(frame_rows),
        "active_frames": sum(1 for r in frame_rows if r["status"] == "matched"),
        "dark_frames": sum(1 for r in frame_rows if r["status"] == "dark"),
        "template_count": len(templates_sorted),
        "sequence_segment_count": len(segments),
        "outputs": {
            "review": "review.html",
            "full_reconstruction": "full_reconstruction.jpg",
            "roi_marked": "roi_marked.jpg",
            "template_contact": "template_contact.jpg",
            "templates": "templates.json",
            "sequence": "sequence.csv",
            "frame_matches": "frame_matches.csv",
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_csv(
        out_dir / "frame_brightness_timeline.csv",
        frame_brightness,
        ["frame", "time_sec", "max_gray", "active_pixels", "active_ratio"],
    )
    write_review_html(out_dir, meta, roi, templates_sorted)

    print(f"[3] Done: {out_dir}")
    print(f"    templates: {len(templates_sorted)}")
    print(f"    sequence segments: {len(segments)}")
    print(f"    active frames: {summary['active_frames']}/{summary['processed_frames']}")
    print(f"    review: {out_dir / 'review.html'}")


if __name__ == "__main__":
    main()
