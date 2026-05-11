from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import build_direct_point_templates as direct_tpl
import matrix_12state_probe as matrix_probe
import run_direct_point_templates as direct_run


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(direct_tpl.jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dense-cut, classify, intra-class cluster, and refine direct point templates.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--template-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--panel-long-side", type=int, default=720)
    parser.add_argument("--match-radius", type=float, default=5.5)
    parser.add_argument("--align-radius", type=int, default=8)
    parser.add_argument("--stable-per-class", type=int, default=60)
    parser.add_argument("--cluster-min-size", type=int, default=12)
    return parser


def save_dense_panels(
    video_path: Path,
    output_dir: Path,
    sample_fps: float,
    panel_long_side: int,
) -> tuple[Path, Path, dict[str, Any]]:
    frame_count, fps, width, height = matrix_probe.video_meta(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    step_frames = max(1, int(round(fps / max(sample_fps, 0.1))))
    frames: list[matrix_probe.SampleFrame] = []
    frame_index = 0
    sample_index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_index % step_frames == 0:
            frames.append(
                matrix_probe.SampleFrame(
                    sample_index=sample_index,
                    frame_index=frame_index,
                    time_sec=frame_index / fps,
                    frame=frame,
                )
            )
            sample_index += 1
        frame_index += 1
    cap.release()
    if not frames:
        raise RuntimeError("Dense cut failed: no frames sampled")
    meta = {
        "video": str(video_path),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "sample_start_sec": 0.0,
        "sample_duration_sec": 0.0,
        "sample_fps": sample_fps,
        "sampled_frames": len(frames),
        "sample_step_frames": step_frames,
    }
    quads = [quad for quad in (matrix_probe.detect_matrix_quad(item.frame) for item in frames) if quad is not None]
    if not quads:
        raise RuntimeError("Dense cut failed: no panel quads detected")

    source_quad = matrix_probe.median_quad(quads)
    target_size = matrix_probe.quad_size(source_quad, panel_long_side)
    per_frame_quads = [matrix_probe.detect_matrix_quad(item.frame) for item in frames]

    aligned_dir = output_dir / f"aligned_panels_{sample_fps:g}fps"
    raw_dir = output_dir / f"raw_frames_{sample_fps:g}fps"
    rows: list[dict[str, Any]] = []

    for item, quad in zip(frames, per_frame_quads):
        used_quad = quad if quad is not None else source_quad
        aligned = matrix_probe.warp_panel_quad(item.frame, used_quad, target_size)
        filename = f"{item.sample_index:04d}_t{item.time_sec:07.2f}s_f{item.frame_index:06d}.jpg"
        raw_path = raw_dir / filename
        aligned_path = aligned_dir / filename
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(raw_path), item.frame)
        cv2.imwrite(str(aligned_path), aligned)
        rows.append(
            {
                "sample_index": f"{item.sample_index:04d}",
                "time_sec": f"{item.time_sec:.3f}",
                "frame_index": item.frame_index,
                "raw_frame": str(raw_path),
                "aligned_panel": str(aligned_path),
            }
        )

    write_csv(output_dir / "index.csv", rows, ["sample_index", "time_sec", "frame_index", "raw_frame", "aligned_panel"])
    meta.update(
        {
            "source_quad": source_quad.tolist(),
            "aligned_panel_size": list(target_size),
            "aligned_dir": str(aligned_dir),
            "raw_dir": str(raw_dir),
        }
    )
    write_json(output_dir / "meta.json", meta)
    return aligned_dir, output_dir / "index.csv", meta


def render_mask(points: np.ndarray, width: int, height: int, radius: int = 2, blur_sigma: float = 1.2) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y in points:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        if 0 <= cx < width and 0 <= cy < height:
            cv2.circle(mask, (cx, cy), radius, 255, -1)
    if blur_sigma > 0:
        return cv2.GaussianBlur(mask, (0, 0), blur_sigma)
    return mask


def feature_from_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = render_mask(points, width, height, radius=2, blur_sigma=1.0)
    small = cv2.resize(mask, (96, 64), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0).reshape(-1)


def kmeans_2(features: np.ndarray, seed: int = 7, max_iter: int = 40) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    centers = np.empty((2, features.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = features[first]
    dists = np.sum((features - centers[0]) ** 2, axis=1)
    second = int(np.argmax(dists))
    centers[1] = features[second]
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dist = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for index in range(2):
            members = features[labels == index]
            if len(members):
                centers[index] = members.mean(axis=0)
    return labels, centers


def cluster_members(feature_rows: list[np.ndarray], min_size: int) -> np.ndarray:
    n = len(feature_rows)
    if n < min_size:
        return np.zeros(n, dtype=np.int32)
    features = np.stack(feature_rows, axis=0)
    labels, centers = kmeans_2(features)
    counts = np.bincount(labels, minlength=2)
    center_gap = float(np.linalg.norm(centers[0] - centers[1]))
    if counts.min() < max(4, int(round(n * 0.18))) or center_gap < 1.2:
        return np.zeros(n, dtype=np.int32)
    return labels


def median_point_count(records: list[dict[str, Any]]) -> int:
    counts = sorted(int(record["detected_point_count"]) for record in records)
    return counts[len(counts) // 2]


def build_refined_templates(
    classification_csv: Path,
    aligned_dir: Path,
    output_dir: Path,
    stable_per_class: int,
    cluster_min_size: int,
) -> tuple[Path, dict[str, Any]]:
    rows = list(csv.DictReader(classification_csv.open("r", encoding="utf-8-sig")))
    # Load width/height from the classification's template library via summary sibling.
    summary = json.loads((classification_csv.parent / "direct_point_detection_summary.json").read_text(encoding="utf-8"))
    summary_library_path = Path(summary["template_library"])
    if not summary_library_path.is_absolute():
        summary_library_path = Path.cwd() / summary_library_path
    source_library = json.loads(summary_library_path.read_text(encoding="utf-8"))
    width, height = [int(v) for v in source_library["image_size"]]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["valid_detection"] != "1":
            continue
        grouped.setdefault(row["matched_template"], []).append(row)

    refined_templates: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    mask_dir = output_dir / "pure_point_masks"
    comparison_items: list[Path] = []
    stats: dict[str, Any] = {}

    for template_name, records in sorted(grouped.items()):
        records = sorted(
            records,
            key=lambda item: (
                float(item["composite_score"]),
                int(item["missing_count"]) + int(item["extra_count"]),
                abs(int(item["align_dx"])) + abs(int(item["align_dy"])),
            ),
        )
        expected_count = median_point_count(records)
        stable_records: list[dict[str, Any]] = []
        for record in records:
            total_err = int(record["missing_count"]) + int(record["extra_count"])
            detected_count = int(record["detected_point_count"])
            if abs(detected_count - expected_count) > max(12, int(round(expected_count * 0.08))):
                continue
            if total_err > max(140, int(round(expected_count * 0.32))):
                continue
            stable_records.append(record)
            if len(stable_records) >= stable_per_class:
                break
        if not stable_records:
            stable_records = records[: min(len(records), stable_per_class)]

        feature_rows: list[np.ndarray] = []
        point_sets: list[np.ndarray] = []
        for record in stable_records:
            image = cv2.imread(str(Path(record["image"])))
            if image is None:
                continue
            points, _ = direct_tpl.detect_points(image)
            arr = np.asarray(points, dtype=np.float32)
            if len(arr) == 0:
                continue
            arr[:, 0] += float(record["align_dx"])
            arr[:, 1] += float(record["align_dy"])
            point_sets.append(arr)
            feature_rows.append(feature_from_points(arr, width, height))
        if not feature_rows:
            continue

        labels = cluster_members(feature_rows, cluster_min_size)
        dominant_cluster = Counter(labels.tolist()).most_common(1)[0][0]
        dominant_indices = [index for index, label in enumerate(labels.tolist()) if label == dominant_cluster]
        dominant_sets = [point_sets[index] for index in dominant_indices]

        accum = np.zeros((height, width), dtype=np.float32)
        for points in dominant_sets:
            accum += render_mask(points, width, height, radius=2, blur_sigma=1.0).astype(np.float32) / 255.0
        accum /= max(1, len(dominant_sets))
        consensus_gray = np.clip(accum * 255.0, 0, 255).astype(np.uint8)
        consensus_bgr = cv2.cvtColor(consensus_gray, cv2.COLOR_GRAY2BGR)
        consensus_points, detect_stats = direct_tpl.detect_points(consensus_bgr)

        out_path = mask_dir / f"{template_name}_refined_points.jpg"
        direct_tpl.draw_points(consensus_points, width, height, f"{template_name} refined={len(consensus_points)}", out_path)
        comparison_items.append(out_path)
        refined_templates.append(
            {
                "name": template_name,
                "source_image": stable_records[0]["image"],
                "point_mask_image": str(out_path),
                "point_count": len(consensus_points),
                "points": [{"x": round(float(x), 3), "y": round(float(y), 3)} for x, y in consensus_points],
                "detection_stats": detect_stats,
                "member_count": len(dominant_sets),
            }
        )
        stats[template_name] = {
            "candidate_count": len(records),
            "stable_count": len(stable_records),
            "cluster_count": int(len(set(labels.tolist()))),
            "dominant_cluster_id": int(dominant_cluster),
            "dominant_member_count": len(dominant_sets),
            "refined_point_count": len(consensus_points),
        }

        for local_index, record in enumerate(stable_records):
            cluster_rows.append(
                {
                    "template_name": template_name,
                    "image": record["image"],
                    "cluster_id": int(labels[local_index]),
                    "dominant_cluster": int(dominant_cluster),
                    "is_dominant": int(labels[local_index] == dominant_cluster),
                    "composite_score": record["composite_score"],
                    "missing_count": record["missing_count"],
                    "extra_count": record["extra_count"],
                    "align_dx": record["align_dx"],
                    "align_dy": record["align_dy"],
                }
            )

    imgs = [cv2.imread(str(path)) for path in comparison_items]
    if imgs:
        sep = 12
        max_height = max(image.shape[0] for image in imgs if image is not None)
        parts: list[np.ndarray] = []
        for index, image in enumerate(imgs):
            assert image is not None
            if image.shape[0] < max_height:
                pad = np.zeros((max_height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
                image = np.vstack([image, pad])
            parts.append(image)
            if index != len(imgs) - 1:
                parts.append(np.full((max_height, sep, 3), 24, dtype=np.uint8))
        comparison = np.hstack(parts)
        cv2.imwrite(str(output_dir / "refined_template_comparison.jpg"), comparison)

    payload = {
        "type": "direct_point_template_library",
        "version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_size": [width, height],
        "comparison_image": str(output_dir / "refined_template_comparison.jpg"),
        "templates": refined_templates,
        "refine_stats": stats,
    }
    library_path = output_dir / "refined_direct_point_templates.json"
    write_json(library_path, payload)
    write_csv(
        output_dir / "intra_class_clusters.csv",
        cluster_rows,
        [
            "template_name",
            "image",
            "cluster_id",
            "dominant_cluster",
            "is_dominant",
            "composite_score",
            "missing_count",
            "extra_count",
            "align_dx",
            "align_dy",
        ],
    )
    return library_path, stats


def compare_runs(base_csv: Path, refined_csv: Path) -> dict[str, Any]:
    base_rows = {Path(row["image"]).name: row for row in csv.DictReader(base_csv.open("r", encoding="utf-8-sig"))}
    refined_rows = {Path(row["image"]).name: row for row in csv.DictReader(refined_csv.open("r", encoding="utf-8-sig"))}
    common = sorted(set(base_rows) & set(refined_rows))
    if not common:
        return {}
    totals_before: list[int] = []
    totals_after: list[int] = []
    improved = 0
    worsened = 0
    unchanged = 0
    for key in common:
        before = int(base_rows[key]["missing_count"]) + int(base_rows[key]["extra_count"])
        after = int(refined_rows[key]["missing_count"]) + int(refined_rows[key]["extra_count"])
        totals_before.append(before)
        totals_after.append(after)
        if after < before:
            improved += 1
        elif after > before:
            worsened += 1
        else:
            unchanged += 1
    return {
        "frame_count": len(common),
        "avg_total_error_before": round(float(np.mean(totals_before)), 3),
        "avg_total_error_after": round(float(np.mean(totals_after)), 3),
        "median_total_error_before": round(float(np.median(totals_before)), 3),
        "median_total_error_after": round(float(np.median(totals_after)), 3),
        "improved_frames": improved,
        "worsened_frames": worsened,
        "unchanged_frames": unchanged,
    }


def main() -> None:
    args = build_parser().parse_args()
    dense_dir = args.output_dir / f"{args.video.stem}_{args.sample_fps:g}fps_dense"
    aligned_dir, _, dense_meta = save_dense_panels(args.video, dense_dir, args.sample_fps, args.panel_long_side)

    base_run_dir = args.output_dir / "base_classification"
    base_summary = direct_run.run_detection(
        template_library=args.template_library,
        input_dir=aligned_dir,
        output_dir=base_run_dir,
        match_radius=args.match_radius,
        align_radius=args.align_radius,
    )

    refine_dir = args.output_dir / "refined_templates"
    refined_library, refine_stats = build_refined_templates(
        classification_csv=base_run_dir / "direct_point_detection_results.csv",
        aligned_dir=aligned_dir,
        output_dir=refine_dir,
        stable_per_class=args.stable_per_class,
        cluster_min_size=args.cluster_min_size,
    )

    refined_run_dir = args.output_dir / "refined_classification"
    refined_summary = direct_run.run_detection(
        template_library=refined_library,
        input_dir=aligned_dir,
        output_dir=refined_run_dir,
        match_radius=args.match_radius,
        align_radius=args.align_radius,
    )

    comparison = compare_runs(
        base_csv=base_run_dir / "direct_point_detection_results.csv",
        refined_csv=refined_run_dir / "direct_point_detection_results.csv",
    )
    final_summary = {
        "video": str(args.video),
        "sample_fps": args.sample_fps,
        "dense_cut_dir": str(dense_dir),
        "base_template_library": str(args.template_library),
        "base_run": base_summary,
        "refined_template_library": str(refined_library),
        "refine_stats": refine_stats,
        "refined_run": refined_summary,
        "comparison": comparison,
        "dense_meta": dense_meta,
    }
    write_json(args.output_dir / "dense_refine_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
