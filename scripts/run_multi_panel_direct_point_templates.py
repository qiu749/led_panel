from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2

import run_direct_point_templates as direct_run
from build_direct_point_templates import detect_points


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(direct_run.jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_index_rows(index_csv: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(index_csv.open("r", encoding="utf-8-sig", newline="")))


def panel_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    cols = [key for key in rows[0].keys() if key.startswith("panel_") and key.endswith("_aligned")]
    return sorted(cols)


def load_panel_template_libraries(template_library: Path, panel_count: int) -> dict[int, tuple[dict[str, Any], list[dict[str, Any]]]]:
    if template_library.is_file():
        payload, templates = direct_run.load_template_library(template_library)
        return {panel_id: (payload, templates) for panel_id in range(1, panel_count + 1)}

    if not template_library.is_dir():
        raise FileNotFoundError(f"Template library path not found: {template_library}")

    panel_libraries: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for panel_id in range(1, panel_count + 1):
        panel_name = f"panel_{panel_id:02d}"
        candidates = [
            template_library / panel_name / "clustered_templates.json",
            template_library / panel_name / "template_library.json",
            template_library / panel_name / "direct_point_templates.json",
        ]
        selected = next((path for path in candidates if path.exists()), None)
        if selected is None:
            continue
        panel_libraries[panel_id] = direct_run.load_template_library(selected)
    if not panel_libraries:
        raise RuntimeError(f"No panel-specific template libraries found under {template_library}")
    return panel_libraries


def run_panel_detection(
    image_path: Path,
    templates: list[dict[str, Any]],
    width: int,
    height: int,
    match_radius: float,
    align_radius: int,
    scale_search_mode: str,
    scale_values: list[float],
    hybrid_suspect_error_threshold: int,
    hybrid_suspect_edge_error_threshold: int,
    hybrid_suspect_close_error_threshold: int,
    hybrid_suspect_runner_gap_threshold: float,
) -> tuple[dict[str, Any], cv2.typing.MatLike]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    points, stats = detect_points(image)
    detected_points = direct_run.points_to_array(points)
    detected_zero_mask = direct_run.points_to_zero_mask(detected_points, width, height, radius=2)
    detected_distance_map = cv2.distanceTransform(detected_zero_mask, cv2.DIST_L2, 3)

    if scale_search_mode == "all":
        results = direct_run.compare_all_templates(
            detected_points,
            detected_distance_map,
            templates,
            match_radius,
            align_radius,
            scale_values,
        )
        retried_with_scale = 1 if len(scale_values) > 1 else 0
    else:
        results = direct_run.compare_all_templates(
            detected_points,
            detected_distance_map,
            templates,
            match_radius,
            align_radius,
            [1.0],
        )
        retried_with_scale = 0
        runner_up = results[1] if len(results) > 1 else None
        if len(scale_values) > 1 and direct_run.should_retry_hybrid(
            results[0],
            runner_up,
            align_radius,
            hybrid_suspect_error_threshold,
            hybrid_suspect_edge_error_threshold,
            hybrid_suspect_close_error_threshold,
            hybrid_suspect_runner_gap_threshold,
        ):
            results = direct_run.compare_all_templates(
                detected_points,
                detected_distance_map,
                templates,
                match_radius,
                align_radius,
                scale_values,
            )
            retried_with_scale = 1

    best = results[0]
    best_template = next(template for template in templates if template["name"] == best["name"])
    overlay = image.copy()
    overlay_path = Path("_unused.jpg")
    direct_run.draw_overlay(
        image=overlay,
        template_points=best_template["points"],
        detected_points=detected_points,
        result=best,
        output_path=overlay_path,
        title=f"{image_path.stem} | {best['name']}",
    )
    drawn = cv2.imread(str(overlay_path))
    if drawn is None:
        drawn = overlay
    try:
        overlay_path.unlink(missing_ok=True)
    except Exception:
        pass

    row = {
        "image": str(image_path),
        "matched_template": best["name"],
        "detected_point_count": int(len(detected_points)),
        "expected_point_count": int(best_template["point_count"]),
        "missing_count": int(best["missing_count"]),
        "extra_count": int(best["extra_count"]),
        "raw_missing_count": int(best["raw_missing_count"]),
        "raw_extra_count": int(best["raw_extra_count"]),
        "count_gap": int(best["count_gap"]),
        "composite_score": round(float(best["composite"]), 5),
        "det_mean": round(float(best["det_mean"]), 5),
        "tpl_mean": round(float(best["tpl_mean"]), 5),
        "det_p90": round(float(best["det_p90"]), 5),
        "tpl_p90": round(float(best["tpl_p90"]), 5),
        "threshold": round(float(stats["threshold"]), 3),
        "align_dx": int(best["align_dx"]),
        "align_dy": int(best["align_dy"]),
        "align_scale_x": round(float(best["align_scale_x"]), 5),
        "retried_with_scale": int(retried_with_scale),
        "valid_detection": int(len(detected_points) > 0),
        "runner_up_template": results[1]["name"] if len(results) > 1 else "",
        "runner_up_score": round(float(results[1]["composite"]), 5) if len(results) > 1 else "",
        "template_scores": ";".join(f"{item['name']}:{item['composite']:.5f}" for item in results),
    }
    return row, drawn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run direct-point templates on multi-panel aligned frame manifests.")
    parser.add_argument("--template-library", type=Path, required=True)
    parser.add_argument("--index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-radius", type=float, default=5.5)
    parser.add_argument("--align-radius", type=int, default=8)
    parser.add_argument("--x-scale-step", type=float, default=0.01)
    parser.add_argument("--x-scale-span", type=float, default=0.04)
    parser.add_argument("--scale-search-mode", choices=["all", "hybrid"], default="hybrid")
    parser.add_argument("--hybrid-suspect-error-threshold", type=int, default=60)
    parser.add_argument("--hybrid-suspect-edge-error-threshold", type=int, default=24)
    parser.add_argument("--hybrid-suspect-close-error-threshold", type=int, default=10)
    parser.add_argument("--hybrid-suspect-runner-gap-threshold", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scale_values = direct_run.build_scale_values(args.x_scale_step, args.x_scale_span)
    rows = load_index_rows(args.index_csv)
    panel_cols = panel_columns(rows)
    if not panel_cols:
        raise RuntimeError("No panel_*_aligned columns found in index csv")
    panel_libraries = load_panel_template_libraries(args.template_library, len(panel_cols))

    result_rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    panel_counts: dict[str, int] = {col: 0 for col in panel_cols}
    retry_count = 0

    for item in rows:
        sample_index = item["sample_index"]
        frame_index = int(item["frame_index"])
        detected_panel_count = int(item.get("detected_panel_count", "0") or 0)
        raw_frame = item["raw_frame"]
        for panel_idx, panel_col in enumerate(panel_cols, start=1):
            image_ref = item.get(panel_col, "")
            if not image_ref:
                continue
            if panel_idx not in panel_libraries:
                continue
            image_path = Path(image_ref)
            payload, templates = panel_libraries[panel_idx]
            width, height = [int(v) for v in payload["image_size"]]
            panel_row, overlay = run_panel_detection(
                image_path=image_path,
                templates=templates,
                width=width,
                height=height,
                match_radius=args.match_radius,
                align_radius=args.align_radius,
                scale_search_mode=args.scale_search_mode,
                scale_values=scale_values,
                hybrid_suspect_error_threshold=args.hybrid_suspect_error_threshold,
                hybrid_suspect_edge_error_threshold=args.hybrid_suspect_edge_error_threshold,
                hybrid_suspect_close_error_threshold=args.hybrid_suspect_close_error_threshold,
                hybrid_suspect_runner_gap_threshold=args.hybrid_suspect_runner_gap_threshold,
            )
            overlay_path = args.output_dir / "overlays" / f"panel_{panel_idx:02d}" / image_path.name
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay)
            overlay_paths.append(overlay_path)
            retry_count += int(panel_row["retried_with_scale"])
            panel_counts[panel_col] += 1
            panel_row.update(
                {
                    "sample_index": sample_index,
                    "frame_index": frame_index,
                    "detected_panel_count": detected_panel_count,
                    "panel_id": panel_idx,
                    "raw_frame": raw_frame,
                    "overlay": str(overlay_path),
                }
            )
            result_rows.append(panel_row)

    if not result_rows:
        raise RuntimeError("No panel detections were produced")

    fieldnames = [
        "sample_index",
        "frame_index",
        "detected_panel_count",
        "panel_id",
        "raw_frame",
        "image",
        "overlay",
        "matched_template",
        "detected_point_count",
        "expected_point_count",
        "missing_count",
        "extra_count",
        "raw_missing_count",
        "raw_extra_count",
        "count_gap",
        "composite_score",
        "det_mean",
        "tpl_mean",
        "det_p90",
        "tpl_p90",
        "threshold",
        "align_dx",
        "align_dy",
        "align_scale_x",
        "retried_with_scale",
        "valid_detection",
        "runner_up_template",
        "runner_up_score",
        "template_scores",
    ]
    write_csv(args.output_dir / "panel_detection_results.csv", result_rows, fieldnames)
    direct_run.make_contact_sheets(overlay_paths, args.output_dir / "contact_sheets")
    summary = {
        "type": "multi_panel_direct_point_detection_run",
        "template_library": str(args.template_library),
        "index_csv": str(args.index_csv),
        "output_dir": str(args.output_dir),
        "panel_columns": panel_cols,
        "panel_template_sources": {
            f"panel_{panel_id:02d}": panel_libraries[panel_id][0].get("comparison_image", "")
            for panel_id in sorted(panel_libraries)
        },
        "panel_counts": panel_counts,
        "panel_detection_count": len(result_rows),
        "scale_search_mode": args.scale_search_mode,
        "slow_retry_count": retry_count,
        "results_csv": str(args.output_dir / "panel_detection_results.csv"),
        "overlay_dir": str(args.output_dir / "overlays"),
    }
    write_json(args.output_dir / "multi_panel_detection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
