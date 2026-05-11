from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_direct_point_templates import detect_points


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


def load_images(input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(input_dir.glob(pattern))
    return sorted(paths)


def points_to_array(points: list[dict[str, float]] | list[tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    if isinstance(points[0], dict):
        return np.asarray([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def points_to_zero_mask(points: np.ndarray, width: int, height: int, radius: int) -> np.ndarray:
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x, y in points:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        if 0 <= cx < width and 0 <= cy < height:
            cv2.circle(mask, (cx, cy), radius, 0, -1)
    return mask


def estimate_horizontal_pitch(points: np.ndarray) -> float:
    if len(points) < 4:
        return 18.0
    xs = np.unique(np.rint(points[:, 0]).astype(np.int32))
    diffs = np.diff(np.sort(xs))
    diffs = diffs[(diffs >= 6) & (diffs <= 60)]
    if len(diffs) == 0:
        return 18.0
    return float(np.median(diffs))


def compute_edge_weights(points: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    if len(points) == 0:
        return np.empty((0,), dtype=np.float32), 0.0
    pitch = estimate_horizontal_pitch(points)
    edge_band = max(pitch * 1.6, width * 0.035)
    left_edge = float(points[:, 0].min())
    right_edge = float(points[:, 0].max())
    edge_distance = np.minimum(points[:, 0] - left_edge, right_edge - points[:, 0])
    weights = np.ones((len(points),), dtype=np.float32)
    weights[edge_distance <= edge_band] = 0.2
    return weights, float(edge_band)


def sample_distances(distance_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=np.float32)
    coords = np.rint(points).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, distance_map.shape[1] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, distance_map.shape[0] - 1)
    return distance_map[coords[:, 1], coords[:, 0]]


def shift_points(points: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    return points + np.asarray([dx, dy], dtype=np.float32)


def scale_points(points: np.ndarray, scale_x: float, scale_y: float, center: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.copy()
    scale = np.asarray([scale_x, scale_y], dtype=np.float32)
    return ((points - center) * scale) + center


def make_contact_sheets(overlay_paths: list[Path], output_dir: Path, cols: int = 3, rows: int = 4) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[np.ndarray] = []
    for path in overlay_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        thumbs.append(cv2.resize(image, (240, 160), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    per_page = cols * rows
    h, w = thumbs[0].shape[:2]
    for start in range(0, len(thumbs), per_page):
        page = thumbs[start : start + per_page]
        canvas = np.full((rows * h, cols * w, 3), 24, dtype=np.uint8)
        for index, thumb in enumerate(page):
            row = index // cols
            col = index % cols
            canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = thumb
        page_no = start // per_page + 1
        cv2.imwrite(str(output_dir / f"direct_point_sheet_{page_no:02d}.jpg"), canvas)


def load_template_library(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    width, height = [int(v) for v in payload["image_size"]]
    templates: list[dict[str, Any]] = []
    for item in payload["templates"]:
        points = points_to_array(item["points"])
        zero_mask = points_to_zero_mask(points, width, height, radius=2)
        distance_map = cv2.distanceTransform(zero_mask, cv2.DIST_L2, 3)
        templates.append(
            {
                "name": item["name"],
                "source_image": item["source_image"],
                "points": points,
                "point_count": int(item["point_count"]),
                "center": points.mean(axis=0, keepdims=True) if len(points) else np.zeros((1, 2), dtype=np.float32),
                "point_weights": compute_edge_weights(points, width)[0],
                "distance_map": distance_map,
            }
        )
    return payload, templates


def build_scale_values(x_scale_step: float, x_scale_span: float) -> list[float]:
    if x_scale_step <= 0:
        raise ValueError("x_scale_step must be > 0")
    scale_values = [1.0]
    if x_scale_span > 0:
        steps = int(round(x_scale_span / x_scale_step))
        scale_values = sorted(
            {
                round(1.0 + (step * x_scale_step), 5)
                for step in range(-steps, steps + 1)
            }
        )
    return scale_values


def compare_template(
    detected_points: np.ndarray,
    detected_distance_map: np.ndarray,
    template: dict[str, Any],
    match_radius: float,
    align_radius: int,
    x_scale_values: list[float],
) -> dict[str, Any]:
    template_points = template["points"]
    if len(detected_points) == 0:
        missing_mask = np.ones((len(template_points),), dtype=bool)
        extra_mask = np.zeros((0,), dtype=bool)
        fallback = 1000.0 + (len(template_points) * 0.02)
        return {
            "name": template["name"],
            "composite": float(fallback),
            "det_mean": 999.0,
            "tpl_mean": 999.0,
            "det_p90": 999.0,
            "tpl_p90": 999.0,
            "count_gap": int(len(template_points)),
            "missing_count": int(len(template_points)),
            "extra_count": 0,
            "raw_missing_count": int(len(template_points)),
            "raw_extra_count": 0,
            "missing_mask": missing_mask,
            "extra_mask": extra_mask,
            "align_dx": 0,
            "align_dy": 0,
            "align_scale_x": 1.0,
            "template_point_weights": template["point_weights"],
            "detected_point_weights": np.empty((0,), dtype=np.float32),
        }

    count_gap = abs(int(len(detected_points)) - int(len(template_points)))
    detected_point_weights, _ = compute_edge_weights(detected_points, detected_distance_map.shape[1])
    best: dict[str, Any] | None = None

    for scale_x in x_scale_values:
        scaled_template = scale_points(template_points, float(scale_x), 1.0, template["center"])
        scaled_zero_mask = points_to_zero_mask(scaled_template, detected_distance_map.shape[1], detected_distance_map.shape[0], radius=2)
        scaled_distance_map = cv2.distanceTransform(scaled_zero_mask, cv2.DIST_L2, 3)
        scale_penalty = abs(float(scale_x) - 1.0) * 0.5

        for dx in range(-align_radius, align_radius + 1):
            for dy in range(-align_radius, align_radius + 1):
                shifted_detected = shift_points(detected_points, float(-dx), float(-dy))
                shifted_template = shift_points(scaled_template, float(dx), float(dy))

                det_to_tpl = sample_distances(scaled_distance_map, shifted_detected)
                tpl_to_det = sample_distances(detected_distance_map, shifted_template)

                det_mean = float(det_to_tpl.mean()) if len(det_to_tpl) else 999.0
                tpl_mean = float(tpl_to_det.mean()) if len(tpl_to_det) else 999.0
                det_p90 = float(np.percentile(det_to_tpl, 90)) if len(det_to_tpl) else 999.0
                tpl_p90 = float(np.percentile(tpl_to_det, 90)) if len(tpl_to_det) else 999.0

                missing_mask = tpl_to_det > match_radius
                extra_mask = det_to_tpl > match_radius
                raw_missing_count = int(missing_mask.sum())
                raw_extra_count = int(extra_mask.sum())
                missing_count = int(round(float(template["point_weights"][missing_mask].sum())))
                extra_count = int(round(float(detected_point_weights[extra_mask].sum())))

                # Geometry dominates; count gap is only a weak prior.
                composite = det_mean + tpl_mean + (det_p90 * 0.35) + (tpl_p90 * 0.35) + (count_gap * 0.02) + scale_penalty
                candidate = {
                    "name": template["name"],
                    "composite": float(composite),
                    "det_mean": det_mean,
                    "tpl_mean": tpl_mean,
                    "det_p90": det_p90,
                    "tpl_p90": tpl_p90,
                    "count_gap": count_gap,
                    "missing_count": missing_count,
                    "extra_count": extra_count,
                    "raw_missing_count": raw_missing_count,
                    "raw_extra_count": raw_extra_count,
                    "missing_mask": missing_mask,
                    "extra_mask": extra_mask,
                    "align_dx": int(dx),
                    "align_dy": int(dy),
                    "align_scale_x": float(scale_x),
                    "template_point_weights": template["point_weights"],
                    "detected_point_weights": detected_point_weights,
                }
                if best is None or candidate["composite"] < best["composite"]:
                    best = candidate

    assert best is not None
    return best


def compare_all_templates(
    detected_points: np.ndarray,
    detected_distance_map: np.ndarray,
    templates: list[dict[str, Any]],
    match_radius: float,
    align_radius: int,
    x_scale_values: list[float],
) -> list[dict[str, Any]]:
    results = [
        compare_template(detected_points, detected_distance_map, template, match_radius, align_radius, x_scale_values)
        for template in templates
    ]
    results.sort(key=lambda item: item["composite"])
    return results


def should_retry_hybrid(
    best: dict[str, Any],
    runner_up: dict[str, Any] | None,
    align_radius: int,
    suspect_error_threshold: int,
    suspect_edge_error_threshold: int,
    suspect_close_error_threshold: int,
    suspect_runner_gap_threshold: float,
) -> bool:
    weighted_error = int(best["missing_count"]) + int(best["extra_count"])
    raw_error = int(best["raw_missing_count"]) + int(best["raw_extra_count"])
    if raw_error >= suspect_error_threshold:
        return True
    at_edge = abs(int(best["align_dx"])) >= max(0, align_radius - 1) or abs(int(best["align_dy"])) >= max(0, align_radius - 1)
    if at_edge and raw_error >= suspect_edge_error_threshold:
        return True
    if runner_up is None:
        return False
    runner_gap = float(runner_up["composite"]) - float(best["composite"])
    return runner_gap <= suspect_runner_gap_threshold and weighted_error >= suspect_close_error_threshold


def draw_overlay(
    image: np.ndarray,
    template_points: np.ndarray,
    detected_points: np.ndarray,
    result: dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    vis = image.copy()
    missing_mask = result["missing_mask"]
    extra_mask = result["extra_mask"]
    template_point_weights = result["template_point_weights"]
    detected_point_weights = result["detected_point_weights"]
    scaled_template_points = scale_points(template_points, float(result["align_scale_x"]), 1.0, template_points.mean(axis=0, keepdims=True))
    shifted_template_points = shift_points(scaled_template_points, float(result["align_dx"]), float(result["align_dy"]))

    for index, (x, y) in enumerate(shifted_template_points):
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        if bool(missing_mask[index]):
            if float(template_point_weights[index]) < 0.99:
                cv2.circle(vis, (cx, cy), 6, (0, 0, 0), -1)
                cv2.circle(vis, (cx, cy), 5, (0, 165, 255), 2)
            else:
                cv2.circle(vis, (cx, cy), 7, (0, 0, 0), -1)
                cv2.circle(vis, (cx, cy), 6, (0, 0, 255), 2)
        else:
            cv2.circle(vis, (cx, cy), 3, (0, 210, 0), -1)

    for index, (x, y) in enumerate(detected_points):
        if not bool(extra_mask[index]):
            continue
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        if float(detected_point_weights[index]) < 0.99:
            cv2.circle(vis, (cx, cy), 6, (0, 0, 0), -1)
            cv2.circle(vis, (cx, cy), 5, (255, 200, 0), 2)
        else:
            cv2.circle(vis, (cx, cy), 7, (0, 0, 0), -1)
            cv2.circle(vis, (cx, cy), 6, (255, 0, 0), 2)

    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, 30), (0, 0, 0), -1)
    status = (
        f"miss={result['missing_count']} extra={result['extra_count']} "
        f"raw={result['raw_missing_count']}/{result['raw_extra_count']} "
        f"score={result['composite']:.3f} dx={result['align_dx']} dy={result['align_dy']} sx={result['align_scale_x']:.3f}"
    )
    cv2.putText(vis, f"{title} | {status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def run_detection(
    template_library: Path,
    input_dir: Path,
    output_dir: Path,
    match_radius: float,
    align_radius: int,
    x_scale_step: float,
    x_scale_span: float,
    scale_search_mode: str,
    hybrid_suspect_error_threshold: int,
    hybrid_suspect_edge_error_threshold: int,
    hybrid_suspect_close_error_threshold: int,
    hybrid_suspect_runner_gap_threshold: float,
) -> dict[str, Any]:
    payload, templates = load_template_library(template_library)
    width, height = [int(v) for v in payload["image_size"]]
    image_paths = load_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    scale_values = build_scale_values(x_scale_step, x_scale_span)
    fast_scale_values = [1.0]
    slow_retry_count = 0
    if scale_search_mode not in {"all", "hybrid"}:
        raise ValueError("scale_search_mode must be 'all' or 'hybrid'")

    overlay_dir = output_dir / "overlays"
    sheet_dir = output_dir / "contact_sheets"
    rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []
    template_counts: dict[str, int] = {template["name"]: 0 for template in templates}

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        points, stats = detect_points(image)
        detected_points = points_to_array(points)
        detected_zero_mask = points_to_zero_mask(detected_points, width, height, radius=2)
        detected_distance_map = cv2.distanceTransform(detected_zero_mask, cv2.DIST_L2, 3)

        if scale_search_mode == "all":
            results = compare_all_templates(
                detected_points,
                detected_distance_map,
                templates,
                match_radius,
                align_radius,
                scale_values,
            )
            retried_with_scale = int(len(scale_values) > 1)
        else:
            results = compare_all_templates(
                detected_points,
                detected_distance_map,
                templates,
                match_radius,
                align_radius,
                fast_scale_values,
            )
            runner_up = results[1] if len(results) > 1 else None
            retried_with_scale = 0
            if len(scale_values) > 1 and should_retry_hybrid(
                results[0],
                runner_up,
                align_radius,
                hybrid_suspect_error_threshold,
                hybrid_suspect_edge_error_threshold,
                hybrid_suspect_close_error_threshold,
                hybrid_suspect_runner_gap_threshold,
            ):
                results = compare_all_templates(
                    detected_points,
                    detected_distance_map,
                    templates,
                    match_radius,
                    align_radius,
                    scale_values,
                )
                retried_with_scale = 1
                slow_retry_count += 1
        best = results[0]
        best_template = next(template for template in templates if template["name"] == best["name"])
        template_counts[best["name"]] += 1

        overlay_path = overlay_dir / image_path.name
        draw_overlay(
            image=image,
            template_points=best_template["points"],
            detected_points=detected_points,
            result=best,
            output_path=overlay_path,
            title=f"{image_path.stem} | {best['name']}",
        )
        overlay_paths.append(overlay_path)

        rows.append(
            {
                "image": str(image_path),
                "overlay": str(overlay_path),
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
                "retried_with_scale": retried_with_scale,
                "valid_detection": int(len(detected_points) > 0),
                "runner_up_template": results[1]["name"] if len(results) > 1 else "",
                "runner_up_score": round(float(results[1]["composite"]), 5) if len(results) > 1 else "",
                "template_scores": ";".join(f"{item['name']}:{item['composite']:.5f}" for item in results),
            }
        )

    make_contact_sheets(overlay_paths, sheet_dir)
    write_csv(
        output_dir / "direct_point_detection_results.csv",
        rows,
        [
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
        ],
    )

    summary = {
        "type": "direct_point_detection_run",
        "template_library": str(template_library),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "match_radius": float(match_radius),
        "align_radius": int(align_radius),
        "x_scale_step": float(x_scale_step),
        "x_scale_span": float(x_scale_span),
        "scale_search_mode": scale_search_mode,
        "slow_retry_count": slow_retry_count,
        "image_count": len(rows),
        "template_counts": template_counts,
        "results_csv": str(output_dir / "direct_point_detection_results.csv"),
        "contact_sheets_dir": str(sheet_dir),
        "overlay_dir": str(overlay_dir),
    }
    write_json(output_dir / "direct_point_detection_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Match aligned panel images against direct point templates.")
    parser.add_argument("--template-library", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--match-radius", type=float, default=5.5)
    parser.add_argument("--align-radius", type=int, default=8)
    parser.add_argument("--x-scale-step", type=float, default=0.01)
    parser.add_argument("--x-scale-span", type=float, default=0.04)
    parser.add_argument("--scale-search-mode", choices=["all", "hybrid"], default="all")
    parser.add_argument("--hybrid-suspect-error-threshold", type=int, default=60)
    parser.add_argument("--hybrid-suspect-edge-error-threshold", type=int, default=24)
    parser.add_argument("--hybrid-suspect-close-error-threshold", type=int, default=10)
    parser.add_argument("--hybrid-suspect-runner-gap-threshold", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_detection(
        template_library=args.template_library,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        match_radius=args.match_radius,
        align_radius=args.align_radius,
        x_scale_step=args.x_scale_step,
        x_scale_span=args.x_scale_span,
        scale_search_mode=args.scale_search_mode,
        hybrid_suspect_error_threshold=args.hybrid_suspect_error_threshold,
        hybrid_suspect_edge_error_threshold=args.hybrid_suspect_edge_error_threshold,
        hybrid_suspect_close_error_threshold=args.hybrid_suspect_close_error_threshold,
        hybrid_suspect_runner_gap_threshold=args.hybrid_suspect_runner_gap_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
