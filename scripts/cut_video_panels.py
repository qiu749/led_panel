from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2

import matrix_12state_probe as matrix_probe


def parse_fixed_panel_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("fixed panel size must be WIDTHxHEIGHT")
    width = int(parts[0].strip())
    height = int(parts[1].strip())
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("fixed panel size must be positive")
    return width, height


def load_image_size_from_template_library(path: Path) -> tuple[int, int]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        width, height = [int(v) for v in payload["image_size"]]
        return width, height

    if not path.is_dir():
        raise FileNotFoundError(f"Template library path not found: {path}")

    sizes: set[tuple[int, int]] = set()
    for panel_dir in sorted(path.glob("panel_*")):
        for candidate in (
            panel_dir / "clustered_templates.json",
            panel_dir / "template_library.json",
            panel_dir / "direct_point_templates.json",
        ):
            if not candidate.exists():
                continue
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            sizes.add(tuple(int(v) for v in payload["image_size"]))
            break
    if not sizes:
        raise RuntimeError(f"No template library JSON found under {path}")
    if len(sizes) > 1:
        raise RuntimeError(f"Inconsistent image_size across panel template libraries under {path}: {sorted(sizes)}")
    return next(iter(sizes))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix_probe.jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cut a video into raw frames and aligned LED panel images.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=750)
    parser.add_argument("--panel-long-side", type=int, default=720)
    parser.add_argument("--fixed-panel-size", type=parse_fixed_panel_size, default=None)
    parser.add_argument("--template-library", type=Path, default=None)
    parser.add_argument("--max-panels", type=int, default=4)
    return parser


def sample_frame_indices(total_frames: int, target_count: int) -> list[int]:
    count = max(1, min(total_frames, target_count))
    if count == 1:
        return [0]
    return sorted({int(round(i * (total_frames - 1) / (count - 1))) for i in range(count)})


def stabilize_quad_sequence(
    detected_quads: list[matrix_probe.np.ndarray | None],
    fallback_quad: matrix_probe.np.ndarray,
    max_corner_step: float = 4.0,
) -> list[matrix_probe.np.ndarray]:
    stable: list[matrix_probe.np.ndarray] = []
    prev: matrix_probe.np.ndarray | None = None
    for quad in detected_quads:
        if quad is None:
            current = fallback_quad.astype(matrix_probe.np.float32) if prev is None else prev.copy()
        else:
            ordered = matrix_probe.order_corners(quad).astype(matrix_probe.np.float32)
            if prev is None:
                current = ordered
            else:
                delta = ordered - prev
                if float(matrix_probe.np.max(matrix_probe.np.abs(delta))) <= max_corner_step * 2.0:
                    current = ordered
                else:
                    current = prev + matrix_probe.np.clip(delta, -max_corner_step, max_corner_step)
        stable.append(current)
        prev = current
    return stable


def build_panel_column_name(panel_index: int) -> str:
    return f"panel_{panel_index + 1:02d}_aligned"


def main() -> None:
    args = build_parser().parse_args()
    if args.fixed_panel_size is not None and args.template_library is not None:
        raise RuntimeError("Use either --fixed-panel-size or --template-library, not both")

    resolved_fixed_size = args.fixed_panel_size
    if resolved_fixed_size is None and args.template_library is not None:
        resolved_fixed_size = load_image_size_from_template_library(args.template_library)

    frame_count, fps, width, height = matrix_probe.video_meta(args.video)
    indices = sample_frame_indices(frame_count, args.target_count)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    frames: list[matrix_probe.SampleFrame] = []
    wanted = set(indices)
    current_index = 0
    sample_index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if current_index in wanted:
            frames.append(
                matrix_probe.SampleFrame(
                    sample_index=sample_index,
                    frame_index=current_index,
                    time_sec=current_index / fps,
                    frame=frame,
                )
            )
            sample_index += 1
        current_index += 1
    cap.release()
    if not frames:
        raise RuntimeError("No frames sampled")

    frame_quads = [matrix_probe.detect_matrix_quads(item.frame, max_quads=max(1, args.max_panels)) for item in frames]
    panel_tracks: list[list[matrix_probe.np.ndarray | None]] = []
    panel_source_quads: list[matrix_probe.np.ndarray] = []
    panel_target_sizes: list[tuple[int, int]] = []
    panel_stable_quads: list[list[matrix_probe.np.ndarray]] = []
    for panel_index in range(max(1, args.max_panels)):
        track = [quads[panel_index] if panel_index < len(quads) else None for quads in frame_quads]
        valid = [quad for quad in track if quad is not None]
        if not valid:
            continue
        source_quad = matrix_probe.median_quad(valid)
        panel_tracks.append(track)
        panel_source_quads.append(source_quad)
        if resolved_fixed_size is not None:
            panel_target_sizes.append(resolved_fixed_size)
        else:
            panel_target_sizes.append(matrix_probe.quad_size(source_quad, args.panel_long_side))
        panel_stable_quads.append(stabilize_quad_sequence(track, source_quad))
    if not panel_tracks:
        raise RuntimeError("No panel quads detected")

    raw_dir = args.output_dir / f"raw_frames_{len(frames)}"
    aligned_dirs = [args.output_dir / f"aligned_panels_panel{panel_index + 1:02d}_{len(frames)}" for panel_index in range(len(panel_tracks))]
    rows: list[dict[str, Any]] = []

    for frame_index, item in enumerate(frames):
        filename = f"{item.sample_index:04d}_t{item.time_sec:07.2f}s_f{item.frame_index:06d}.jpg"
        raw_path = raw_dir / filename
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(raw_path), item.frame)
        row: dict[str, Any] = {
            "sample_index": f"{item.sample_index:04d}",
            "time_sec": f"{item.time_sec:.3f}",
            "frame_index": item.frame_index,
            "detected_panel_count": len(frame_quads[frame_index]),
            "raw_frame": str(raw_path),
        }
        for panel_index, (track, stable_track, target_size, aligned_dir) in enumerate(
            zip(panel_tracks, panel_stable_quads, panel_target_sizes, aligned_dirs)
        ):
            raw_quad = track[frame_index]
            row[f"panel_{panel_index + 1:02d}_detected"] = int(raw_quad is not None)
            column_name = build_panel_column_name(panel_index)
            if raw_quad is None:
                row[column_name] = ""
                continue
            aligned = matrix_probe.warp_panel_quad(item.frame, stable_track[frame_index], target_size)
            aligned_path = aligned_dir / filename
            aligned_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(aligned_path), aligned)
            row[column_name] = str(aligned_path)
        rows.append(row)

    fieldnames = ["sample_index", "time_sec", "frame_index", "detected_panel_count", "raw_frame"]
    for panel_index in range(len(panel_tracks)):
        fieldnames.append(f"panel_{panel_index + 1:02d}_detected")
        fieldnames.append(build_panel_column_name(panel_index))
    write_csv(
        args.output_dir / "index.csv",
        rows,
        fieldnames,
    )
    write_json(
        args.output_dir / "meta.json",
        {
            "video": str(args.video),
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "target_count": args.target_count,
            "sampled_count": len(frames),
            "max_panels": args.max_panels,
            "panel_count": len(panel_tracks),
            "panel_source_quads": [quad.tolist() for quad in panel_source_quads],
            "stabilized_quad_max_corner_step": 4.0,
            "fixed_panel_size": list(resolved_fixed_size) if resolved_fixed_size is not None else None,
            "template_library": str(args.template_library) if args.template_library is not None else None,
            "aligned_panel_sizes": [list(size) for size in panel_target_sizes],
            "raw_dir": str(raw_dir),
            "aligned_dirs": [str(path) for path in aligned_dirs],
        },
    )
    print(
        json.dumps(
            {
                "video": str(args.video),
                "sampled_count": len(frames),
                "panel_count": len(panel_tracks),
                "output_dir": str(args.output_dir),
                "raw_dir": str(raw_dir),
                "aligned_dirs": [str(path) for path in aligned_dirs],
                "index_csv": str(args.output_dir / "index.csv"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
