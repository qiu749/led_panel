from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import build_direct_point_templates as direct_tpl
import split_and_cluster_template_groups as splitter


JSON_KW = {"ensure_ascii": False, "indent": 2}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(direct_tpl.jsonable(payload), **JSON_KW), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))


def panel_key(panel_id: int) -> str:
    return f"panel_{panel_id:02d}"


def valid_panel_groups(rows: list[dict[str, str]], min_group_size: int) -> dict[tuple[int, str], list[dict[str, str]]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("valid_detection") != "1":
            continue
        panel_id = int(row["panel_id"])
        source_template = row["matched_template"]
        grouped.setdefault((panel_id, source_template), []).append(row)
    return {key: items for key, items in grouped.items() if len(items) >= min_group_size}


def linked_group_manifest(rows: list[dict[str, str]], output_dir: Path) -> None:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("valid_detection") != "1":
            continue
        grouped.setdefault(int(row["panel_id"]), []).append(row)
    for panel_id, items in sorted(grouped.items()):
        group_dir = output_dir / "by_panel" / panel_key(panel_id) / "aligned_panels"
        manifest_rows: list[dict[str, Any]] = []
        for row in items:
            src = Path(row["image"])
            dst = group_dir / src.name
            safe_link_or_copy(src, dst)
            manifest_rows.append(
                {
                    "panel_id": panel_id,
                    "sample_index": row["sample_index"],
                    "frame_index": row["frame_index"],
                    "matched_template": row["matched_template"],
                    "image": str(src),
                    "linked_image": str(dst),
                    "composite_score": row["composite_score"],
                    "missing_count": row["missing_count"],
                    "extra_count": row["extra_count"],
                }
            )
        write_csv(
            output_dir / "by_panel" / panel_key(panel_id) / "panel_group_results.csv",
            manifest_rows,
            [
                "panel_id",
                "sample_index",
                "frame_index",
                "matched_template",
                "image",
                "linked_image",
                "composite_score",
                "missing_count",
                "extra_count",
            ],
        )


def infer_image_size(rows: list[dict[str, str]]) -> tuple[int, int]:
    for row in rows:
        image = cv2.imread(str(Path(row["image"])))
        if image is not None:
            return int(image.shape[1]), int(image.shape[0])
    raise RuntimeError("Cannot infer image size from rows")


def build_panel_library(
    panel_id: int,
    grouped_rows: dict[str, list[dict[str, str]]],
    output_dir: Path,
    max_k: int,
    min_cluster_size: int,
    stable_limit: int,
) -> dict[str, Any]:
    width, height = infer_image_size([row for items in grouped_rows.values() for row in items])
    cluster_root = output_dir / panel_key(panel_id)

    member_rows: list[dict[str, Any]] = []
    refined_templates: list[dict[str, Any]] = []
    comparison_paths: list[Path] = []
    panel_cluster_total = 0
    stable_input_total = 0
    cluster_sizes: dict[str, int] = {}

    for source_template, source_rows in sorted(grouped_rows.items()):
        ordered_rows = sorted(
            source_rows,
            key=lambda row: (
                float(row["composite_score"]),
                int(row["missing_count"]) + int(row["extra_count"]),
                abs(int(row["align_dx"])) + abs(int(row["align_dy"])),
            ),
        )
        stable_rows = ordered_rows[: min(len(ordered_rows), stable_limit)]
        features: list[np.ndarray] = []
        point_sets: list[np.ndarray] = []
        kept_rows: list[dict[str, str]] = []
        for row in stable_rows:
            image = cv2.imread(str(Path(row["image"])))
            if image is None:
                continue
            points, _ = direct_tpl.detect_points(image)
            arr = np.asarray(points, dtype=np.float32)
            if len(arr) == 0:
                continue
            features.append(splitter.feature_from_points(arr, width, height))
            point_sets.append(arr)
            kept_rows.append(row)
        if not features:
            continue

        stable_input_total += len(kept_rows)
        feature_array = splitter.standardize(np.stack(features, axis=0))
        if len(feature_array) < max(min_cluster_size * 3, 24):
            k = 1
            labels = np.zeros(len(feature_array), dtype=np.int32)
        else:
            k, labels = splitter.choose_k(feature_array, max_k=max_k, min_cluster_size=min_cluster_size)
        counts = Counter(labels.tolist())

        for cluster_id in sorted(counts):
            panel_cluster_total += 1
            indices = [index for index, label in enumerate(labels.tolist()) if label == cluster_id]
            cluster_name = f"{source_template}_cluster_{cluster_id + 1:02d}"
            cluster_sizes[cluster_name] = len(indices)
            cluster_dir = cluster_root / cluster_name
            image_dir = cluster_dir / "aligned_panels"
            accum = np.zeros((height, width), dtype=np.float32)
            for index in indices:
                row = kept_rows[index]
                src = Path(row["image"])
                dst = image_dir / src.name
                safe_link_or_copy(src, dst)
                member_rows.append(
                    {
                        "panel_id": panel_id,
                        "cluster_id": cluster_name,
                        "image": str(src),
                        "linked_image": str(dst),
                        "source_template": source_template,
                        "composite_score": row["composite_score"],
                        "missing_count": row["missing_count"],
                        "extra_count": row["extra_count"],
                    }
                )
                accum += splitter.render_mask(point_sets[index], width, height, radius=2, blur_sigma=1.0).astype(np.float32) / 255.0
            accum /= max(1, len(indices))
            consensus_gray = np.clip(accum * 255.0, 0, 255).astype(np.uint8)
            consensus_bgr = cv2.cvtColor(consensus_gray, cv2.COLOR_GRAY2BGR)
            consensus_points, detect_stats = direct_tpl.detect_points(consensus_bgr)
            template_name = f"{panel_key(panel_id)}__{cluster_name}"
            out_path = cluster_dir / f"{template_name}_template.jpg"
            direct_tpl.draw_points(consensus_points, width, height, f"{template_name} n={len(indices)}", out_path)
            comparison_paths.append(out_path)
            refined_templates.append(
                {
                    "name": template_name,
                    "panel_scope": panel_id,
                    "source_image": kept_rows[indices[0]]["image"],
                    "point_mask_image": str(out_path),
                    "point_count": len(consensus_points),
                    "points": [{"x": round(float(x), 3), "y": round(float(y), 3)} for x, y in consensus_points],
                    "detection_stats": detect_stats,
                    "member_count": len(indices),
                    "source_template_mode": source_template,
                }
            )

    if comparison_paths:
        images = [cv2.imread(str(path)) for path in comparison_paths]
        max_height = max(image.shape[0] for image in images if image is not None)
        parts: list[np.ndarray] = []
        for index, image in enumerate(images):
            assert image is not None
            if image.shape[0] < max_height:
                pad = np.zeros((max_height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
                image = np.vstack([image, pad])
            parts.append(image)
            if index != len(images) - 1:
                parts.append(np.full((max_height, 12, 3), 24, dtype=np.uint8))
        comparison = np.hstack(parts)
        cv2.imwrite(str(cluster_root / "cluster_template_comparison.jpg"), comparison)

    write_csv(
        cluster_root / "cluster_members.csv",
        member_rows,
        [
            "panel_id",
            "cluster_id",
            "image",
            "linked_image",
            "source_template",
            "composite_score",
            "missing_count",
            "extra_count",
        ],
    )
    panel_library = {
        "type": "direct_point_template_library",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "panel_id": panel_id,
        "image_size": [width, height],
        "comparison_image": str(cluster_root / "cluster_template_comparison.jpg"),
        "templates": refined_templates,
    }
    write_json(cluster_root / "clustered_templates.json", panel_library)
    return {
        "panel_id": panel_id,
        "stable_input_count": stable_input_total,
        "cluster_count": panel_cluster_total,
        "cluster_sizes": cluster_sizes,
        "cluster_root": str(cluster_root),
        "template_library": str(cluster_root / "clustered_templates.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild panel-specific template libraries from multi-panel detection results.")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=4)
    parser.add_argument("--min-cluster-size", type=int, default=12)
    parser.add_argument("--stable-limit", type=int, default=120)
    parser.add_argument("--min-group-size", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.results_csv)
    linked_group_manifest(rows, args.output_dir)
    groups = valid_panel_groups(rows, args.min_group_size)
    if not groups:
        raise RuntimeError("No panel groups large enough to rebuild templates")

    panel_summaries: list[dict[str, Any]] = []
    by_panel: dict[int, dict[str, list[dict[str, str]]]] = {}
    for (panel_id, source_template), items in groups.items():
        by_panel.setdefault(panel_id, {})[source_template] = items

    for panel_id, grouped_rows in sorted(by_panel.items()):
        panel_summaries.append(
            build_panel_library(
                panel_id=panel_id,
                grouped_rows=grouped_rows,
                output_dir=args.output_dir,
                max_k=args.max_k,
                min_cluster_size=args.min_cluster_size,
                stable_limit=args.stable_limit,
            )
        )

    summary = {
        "type": "multi_panel_template_library_rebuild",
        "results_csv": str(args.results_csv),
        "output_dir": str(args.output_dir),
        "panel_count": len(panel_summaries),
        "panel_summaries": panel_summaries,
    }
    write_json(args.output_dir / "rebuild_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
