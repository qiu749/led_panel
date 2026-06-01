"""
scan_refresh_seven_segment_probe

Offline probe for seven-segment display videos with scan-refresh flicker.

This script is intentionally video-first rather than model-first:
  1. Build a full-video max projection to lock the display ROI.
  2. Rebuild readable frames with short temporal max windows.
  3. Emit review artifacts for this video: reconstruction, timeline, key windows.

Example:
  python scripts/scan_refresh_seven_segment_probe.py ^
    --video D:/jiemi/七段管/Video_20260515112415516.avi ^
    --out output/sevenseg_scan_refresh/Video_20260515112415516
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Roi:
    x: int
    y: int
    w: int
    h: int


@dataclass
class WindowRecord:
    index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    active_pixels: int
    active_ratio: float
    is_active: bool
    key_state_id: str
    image: str


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_pass_projection(video: Path, bright_threshold: int) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_projection: np.ndarray | None = None
    timeline: list[dict[str, Any]] = []
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if max_projection is None:
            max_projection = frame.copy()
        else:
            max_projection = np.maximum(max_projection, frame)

        active_pixels = int(np.count_nonzero(gray > bright_threshold))
        timeline.append(
            {
                "frame": frame_index,
                "time_sec": round(frame_index / fps, 4) if fps > 0 else 0.0,
                "max_gray": int(gray.max()),
                "active_pixels": active_pixels,
                "active_ratio": active_pixels / max(gray.size, 1),
            }
        )
        frame_index += 1

    cap.release()
    if max_projection is None:
        raise RuntimeError(f"No readable frames: {video}")

    meta = {
        "video": str(video),
        "frame_count_reported": total,
        "frame_count_read": frame_index,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": frame_index / fps if fps > 0 else 0.0,
    }
    return max_projection, timeline, meta


def detect_display_roi(max_projection: np.ndarray, threshold: int, pad_ratio: float) -> Roi:
    gray = cv2.cvtColor(max_projection, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if area >= 20 and w >= 2 and h >= 2:
            boxes.append((x, y, w, h, area))

    if not boxes:
        raise RuntimeError("Cannot detect display ROI from max projection")

    x1 = min(x for x, _, _, _, _ in boxes)
    y1 = min(y for _, y, _, _, _ in boxes)
    x2 = max(x + w for x, _, w, _, _ in boxes)
    y2 = max(y + h for _, y, _, h, _ in boxes)

    frame_h, frame_w = gray.shape[:2]
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)
    return Roi(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def crop_roi(frame: np.ndarray, roi: Roi) -> np.ndarray:
    return frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]


def compose_windows(
    video: Path,
    roi: Roi,
    fps: float,
    out_dir: Path,
    window_frames: int,
    stride_frames: int,
    active_threshold: int,
    min_active_pixels: int,
    max_saved_windows: int,
    key_jaccard_threshold: float,
) -> tuple[list[WindowRecord], list[dict[str, Any]]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    windows_dir = out_dir / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    records: list[WindowRecord] = []
    key_states: list[dict[str, Any]] = []
    saved_count = 0
    last_key_bits = ""
    key_id = 0

    for win_index, start in enumerate(range(0, total, stride_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        composite: np.ndarray | None = None
        actual_end = start - 1
        for offset in range(window_frames):
            ok, frame = cap.read()
            if not ok:
                break
            roi_frame = crop_roi(frame, roi)
            composite = roi_frame.copy() if composite is None else np.maximum(composite, roi_frame)
            actual_end = start + offset

        if composite is None:
            continue

        gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
        active_mask = gray > active_threshold
        active_pixels = int(np.count_nonzero(active_mask))
        active_ratio = active_pixels / max(gray.size, 1)
        is_active = active_pixels >= min_active_pixels

        image_name = ""
        key_state_id = ""
        if is_active and saved_count < max_saved_windows:
            image_name = f"win_{win_index:04d}_f{start:04d}_{actual_end:04d}.jpg"
            annotated = composite.copy()
            cv2.putText(
                annotated,
                f"f{start}-{actual_end}  t={start / fps:.2f}-{actual_end / fps:.2f}s",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(str(windows_dir / image_name), annotated)
            saved_count += 1

        if is_active:
            bits = downsample_bits(active_mask, cols=64, rows=40)
            similarity = bit_jaccard(bits, last_key_bits) if last_key_bits else 0.0
            if not last_key_bits or similarity < key_jaccard_threshold:
                key_id += 1
                key_state_id = f"S{key_id:03d}"
                last_key_bits = bits
                key_image = f"key_{key_id:03d}_f{start:04d}_{actual_end:04d}.jpg"
                cv2.imwrite(str(out_dir / "key_states" / key_image), composite)
                key_states.append(
                    {
                        "id": key_state_id,
                        "start_frame": start,
                        "end_frame": actual_end,
                        "start_time": start / fps,
                        "end_time": actual_end / fps,
                        "active_pixels": active_pixels,
                        "active_ratio": active_ratio,
                        "image": f"key_states/{key_image}",
                    }
                )

        records.append(
            WindowRecord(
                index=win_index,
                start_frame=start,
                end_frame=actual_end,
                start_time=start / fps if fps > 0 else 0.0,
                end_time=actual_end / fps if fps > 0 else 0.0,
                active_pixels=active_pixels,
                active_ratio=active_ratio,
                is_active=is_active,
                key_state_id=key_state_id,
                image=f"windows/{image_name}" if image_name else "",
            )
        )

    cap.release()
    return records, key_states


def downsample_bits(mask: np.ndarray, cols: int, rows: int) -> str:
    mask_u8 = mask.astype(np.uint8) * 255
    small = cv2.resize(mask_u8, (cols, rows), interpolation=cv2.INTER_AREA)
    return "".join("1" if v >= 20 else "0" for v in small.reshape(-1))


def bit_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / max(len(a), len(b))


def bit_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    intersection = 0
    union = 0
    for i in range(n):
        av = a[i] == "1"
        bv = b[i] == "1"
        intersection += int(av and bv)
        union += int(av or bv)
    return intersection / max(union, 1)


def make_contact_sheet(image_paths: list[Path], out_path: Path, cols: int, thumb_w: int) -> None:
    if not image_paths:
        return
    imgs: list[np.ndarray] = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        thumb_h = max(1, int(round(thumb_w * h / max(w, 1))))
        imgs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    if not imgs:
        return

    thumb_h = max(img.shape[0] for img in imgs)
    rows = math.ceil(len(imgs) / cols)
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        y = r * thumb_h
        x = c * thumb_w
        sheet[y : y + img.shape[0], x : x + img.shape[1]] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def write_review_html(out_dir: Path, meta: dict[str, Any], roi: Roi, key_states: list[dict[str, Any]]) -> None:
    cards = []
    for state in key_states:
        cards.append(
            f"""
            <figure>
              <img src="{state['image']}" loading="lazy">
              <figcaption>{state['id']} | f{state['start_frame']}-{state['end_frame']} |
                {state['start_time']:.2f}-{state['end_time']:.2f}s</figcaption>
            </figure>
            """
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Scan-refresh seven-segment review</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: Arial, "Microsoft YaHei", sans-serif; }}
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
    <h1>扫描刷新数码管复核</h1>
    <div class="meta">
      视频: <code>{Path(str(meta['video'])).name}</code><br>
      帧率: {meta['fps']:.3f} FPS, 帧数: {meta['frame_count_read']}, 时长: {meta['duration_sec']:.2f}s<br>
      ROI: x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}
    </div>
  </header>
  <main>
    <h2>全视频最大投影重建</h2>
    <p><img src="full_reconstruction.jpg"></p>
    <h2>去重后的关键合成窗</h2>
    <div class="grid">
      {''.join(cards)}
    </div>
  </main>
</body>
</html>
"""
    (out_dir / "review.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan-refresh seven-segment video probe")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output") / "sevenseg_scan_refresh")
    parser.add_argument("--projection-threshold", type=int, default=40)
    parser.add_argument("--active-threshold", type=int, default=55)
    parser.add_argument("--roi-pad-ratio", type=float, default=0.06)
    parser.add_argument("--window-frames", type=int, default=12, help="Temporal max window; 12 frames is about 0.2s at 60fps")
    parser.add_argument("--stride-frames", type=int, default=6)
    parser.add_argument("--min-active-pixels", type=int, default=250)
    parser.add_argument("--max-saved-windows", type=int, default=240)
    parser.add_argument("--key-jaccard-threshold", type=float, default=0.80,
                        help="New key state threshold based only on lit cells")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    (out_dir / "key_states").mkdir(parents=True, exist_ok=True)

    print(f"[1] Read video and build max projection: {args.video}")
    max_projection, frame_timeline, meta = first_pass_projection(args.video, args.active_threshold)
    roi = detect_display_roi(max_projection, args.projection_threshold, args.roi_pad_ratio)
    print(f"    ROI: x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}")

    full_reconstruction = crop_roi(max_projection, roi)
    cv2.imwrite(str(out_dir / "full_reconstruction.jpg"), full_reconstruction)

    marked = max_projection.copy()
    cv2.rectangle(marked, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 255), 4)
    cv2.imwrite(str(out_dir / "roi_marked.jpg"), cv2.resize(marked, (612, 512), interpolation=cv2.INTER_AREA))

    print(f"[2] Compose temporal windows: window={args.window_frames}, stride={args.stride_frames}")
    records, key_states = compose_windows(
        args.video,
        roi,
        float(meta["fps"]),
        out_dir,
        args.window_frames,
        args.stride_frames,
        args.active_threshold,
        args.min_active_pixels,
        args.max_saved_windows,
        args.key_jaccard_threshold,
    )

    write_csv(
        out_dir / "frame_brightness_timeline.csv",
        frame_timeline,
        ["frame", "time_sec", "max_gray", "active_pixels", "active_ratio"],
    )
    write_csv(
        out_dir / "window_timeline.csv",
        [asdict(r) for r in records],
        [
            "index",
            "start_frame",
            "end_frame",
            "start_time",
            "end_time",
            "active_pixels",
            "active_ratio",
            "is_active",
            "key_state_id",
            "image",
        ],
    )

    window_images = sorted((out_dir / "windows").glob("*.jpg"))
    key_images = sorted((out_dir / "key_states").glob("*.jpg"))
    make_contact_sheet(window_images[:120], out_dir / "active_window_contact.jpg", cols=4, thumb_w=300)
    make_contact_sheet(key_images, out_dir / "key_state_contact.jpg", cols=4, thumb_w=300)

    active_windows = [r for r in records if r.is_active]
    summary = {
        "meta": meta,
        "roi": asdict(roi),
        "parameters": {
            "projection_threshold": args.projection_threshold,
            "active_threshold": args.active_threshold,
            "roi_pad_ratio": args.roi_pad_ratio,
            "window_frames": args.window_frames,
            "stride_frames": args.stride_frames,
            "min_active_pixels": args.min_active_pixels,
            "key_jaccard_threshold": args.key_jaccard_threshold,
        },
        "active_frame_count": sum(1 for r in frame_timeline if r["active_pixels"] > 0),
        "window_count": len(records),
        "active_window_count": len(active_windows),
        "key_state_count": len(key_states),
        "key_states": key_states,
        "outputs": {
            "full_reconstruction": "full_reconstruction.jpg",
            "roi_marked": "roi_marked.jpg",
            "active_window_contact": "active_window_contact.jpg",
            "key_state_contact": "key_state_contact.jpg",
            "review": "review.html",
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_review_html(out_dir, meta, roi, key_states)

    print(f"[3] Done: {out_dir}")
    print(f"    active windows: {len(active_windows)}/{len(records)}")
    print(f"    key states: {len(key_states)}")
    print(f"    review: {out_dir / 'review.html'}")


if __name__ == "__main__":
    main()
