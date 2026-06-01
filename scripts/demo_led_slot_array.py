from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_direct_point_templates import detect_points


DEFAULT_OUTPUT_DIR = Path("output") / "led_slot_array_demo"


@dataclass
class AxisFit:
    centers: list[float]
    pitch: float
    cv: float
    max_gap_ratio: float
    diffs: list[float]


@dataclass
class GridFit:
    x_axis: AxisFit
    y_axis: AxisFit
    occupied_count: int
    expected_count: int
    fill_ratio: float
    median_residual: float
    regularity_score: float


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


def points_to_array(points: list[tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def cluster_1d(values: np.ndarray, merge_gap: float) -> list[float]:
    if len(values) == 0:
        return []
    sorted_values = np.sort(values.astype(np.float32))
    clusters: list[list[float]] = [[float(sorted_values[0])]]
    for value in sorted_values[1:]:
        if float(value) - clusters[-1][-1] <= merge_gap:
            clusters[-1].append(float(value))
        else:
            clusters.append([float(value)])
    return [float(np.mean(cluster)) for cluster in clusters]


def robust_pitch(diffs: np.ndarray) -> float:
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        return 1.0
    if len(diffs) <= 2:
        return float(np.median(diffs))
    # Use the lower half so grouped layouts do not let large inter-group gaps
    # dominate the local LED pitch estimate.
    cutoff = float(np.percentile(diffs, 65))
    local = diffs[diffs <= cutoff]
    if len(local) == 0:
        local = diffs
    return float(np.median(local))


def fit_axis(values: np.ndarray, image_span: int) -> AxisFit:
    if len(values) == 0:
        return AxisFit([], 1.0, 1.0, 999.0, [])
    raw_span = max(float(np.max(values) - np.min(values)), 1.0)
    # Dense photographed matrices often have only 5-7 px pitch. A 3 px merge
    # gap can chain adjacent rows into one band, so try small gaps first and
    # choose the most grid-like stable fit.
    candidate_gaps = [
        0.8,
        1.0,
        1.2,
        1.5,
        2.0,
        min(2.5, max(1.0, raw_span * 0.008)),
    ]
    best_centers: list[float] | None = None
    best_score = -1.0
    for merge_gap in sorted(set(round(float(g), 3) for g in candidate_gaps)):
        centers_candidate = cluster_1d(values, merge_gap)
        if len(centers_candidate) < 2:
            continue
        diffs_candidate = np.diff(np.asarray(centers_candidate, dtype=np.float32))
        pitch_candidate = robust_pitch(diffs_candidate)
        local_candidate = diffs_candidate[
            diffs_candidate <= max(pitch_candidate * 1.65, pitch_candidate + 3.0)
        ]
        if len(local_candidate) == 0:
            local_candidate = diffs_candidate
        cv_candidate = float(np.std(local_candidate) / max(float(np.mean(local_candidate)), 1e-6))
        pitch_score = min(1.0, pitch_candidate / 4.0)
        count_score = min(1.0, len(centers_candidate) / 8.0)
        score = count_score + pitch_score - cv_candidate
        if score > best_score:
            best_score = score
            best_centers = centers_candidate
    centers = best_centers if best_centers is not None else cluster_1d(values, 1.2)
    if len(centers) < 2:
        return AxisFit(centers, raw_span, 1.0, 999.0, [])
    diffs = np.diff(np.asarray(centers, dtype=np.float32))
    pitch = robust_pitch(diffs)
    local = diffs[diffs <= max(pitch * 1.65, pitch + 3.0)]
    if len(local) == 0:
        local = diffs
    cv = float(np.std(local) / max(float(np.mean(local)), 1e-6))
    max_gap_ratio = float(np.max(diffs) / max(pitch, 1e-6))
    return AxisFit(
        centers=[float(v) for v in centers],
        pitch=float(pitch),
        cv=cv,
        max_gap_ratio=max_gap_ratio,
        diffs=[float(v) for v in diffs],
    )


def nearest_indices(values: np.ndarray, centers: list[float]) -> np.ndarray:
    if len(centers) == 0:
        return np.zeros((len(values),), dtype=np.int32)
    center_arr = np.asarray(centers, dtype=np.float32)
    distances = np.abs(values[:, None].astype(np.float32) - center_arr[None, :])
    return np.argmin(distances, axis=1).astype(np.int32)


def compute_grid_fit(points: np.ndarray, width: int, height: int) -> GridFit:
    x_axis = fit_axis(points[:, 0], width)
    y_axis = fit_axis(points[:, 1], height)
    if not x_axis.centers or not y_axis.centers:
        return GridFit(x_axis, y_axis, 0, 0, 0.0, 1.0, 0.0)

    x_idx = nearest_indices(points[:, 0], x_axis.centers)
    y_idx = nearest_indices(points[:, 1], y_axis.centers)
    occupied = set(zip(y_idx.tolist(), x_idx.tolist()))
    expected = len(x_axis.centers) * len(y_axis.centers)
    fill = len(occupied) / max(expected, 1)

    x_centers = np.asarray(x_axis.centers, dtype=np.float32)
    y_centers = np.asarray(y_axis.centers, dtype=np.float32)
    residual_x = np.abs(points[:, 0] - x_centers[x_idx]) / max(x_axis.pitch, 1e-6)
    residual_y = np.abs(points[:, 1] - y_centers[y_idx]) / max(y_axis.pitch, 1e-6)
    residual = float(np.median(np.hypot(residual_x, residual_y))) if len(points) else 1.0

    gap_penalty_x = min(1.0, 1.65 / max(x_axis.max_gap_ratio, 1e-6))
    gap_penalty_y = min(1.0, 1.65 / max(y_axis.max_gap_ratio, 1e-6))
    cv_penalty_x = max(0.0, 1.0 - min(x_axis.cv, 1.0))
    cv_penalty_y = max(0.0, 1.0 - min(y_axis.cv, 1.0))
    residual_penalty = max(0.0, 1.0 - min(residual, 1.0))
    score = fill * gap_penalty_x * gap_penalty_y * cv_penalty_x * cv_penalty_y
    score = score * 0.85 + residual_penalty * 0.15

    return GridFit(
        x_axis=x_axis,
        y_axis=y_axis,
        occupied_count=len(occupied),
        expected_count=expected,
        fill_ratio=float(fill),
        median_residual=residual,
        regularity_score=float(score),
    )


def split_centers_by_large_gaps(centers: list[float], pitch: float, factor: float = 2.2) -> list[list[int]]:
    if not centers:
        return []
    groups: list[list[int]] = [[0]]
    for idx, gap in enumerate(np.diff(np.asarray(centers, dtype=np.float32)), start=1):
        if float(gap) > max(pitch * factor, pitch + 8.0):
            groups.append([idx])
        else:
            groups[-1].append(idx)
    return groups


def make_regular_slots(points: np.ndarray, fit: GridFit) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    cell_w = max(3.0, fit.x_axis.pitch * 0.55)
    cell_h = max(3.0, fit.y_axis.pitch * 0.55)
    x_idx = nearest_indices(points[:, 0], fit.x_axis.centers)
    y_idx = nearest_indices(points[:, 1], fit.y_axis.centers)

    rows: list[tuple[int, int, float, float]] = []
    for point_index, (x, y) in enumerate(points):
        rows.append((int(y_idx[point_index]), int(x_idx[point_index]), float(y), float(x)))
    rows.sort()

    for slot_id, (row_idx, col_idx, y, x) in enumerate(rows, start=1):
        slots.append(
            {
                "id": slot_id,
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "w": round(float(cell_w), 3),
                "h": round(float(cell_h), 3),
                "row": int(row_idx + 1),
                "col": int(col_idx + 1),
                "group": "grid",
            }
        )
    return slots


def make_grouped_slots(points: np.ndarray, fit: GridFit) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x_centers = fit.x_axis.centers
    y_centers = fit.y_axis.centers
    x_groups = split_centers_by_large_gaps(x_centers, fit.x_axis.pitch)
    y_bands = split_centers_by_large_gaps(y_centers, fit.y_axis.pitch)

    x_idx = nearest_indices(points[:, 0], x_centers)
    y_idx = nearest_indices(points[:, 1], y_centers)
    x_to_group = {center_idx: group_no for group_no, group in enumerate(x_groups, start=1) for center_idx in group}
    y_to_band = {center_idx: band_no for band_no, band in enumerate(y_bands, start=1) for center_idx in band}

    rows: list[tuple[int, int, int, float, float, int, int]] = []
    for point_index, (x, y) in enumerate(points):
        xi = int(x_idx[point_index])
        yi = int(y_idx[point_index])
        rows.append((y_to_band.get(yi, 1), x_to_group.get(xi, 1), yi, float(y), float(x), xi, point_index))
    rows.sort()

    slots: list[dict[str, Any]] = []
    for slot_id, row in enumerate(rows, start=1):
        band, x_group, y_center_idx, y, x, x_center_idx, _ = row
        slots.append(
            {
                "id": slot_id,
                "x": round(x, 3),
                "y": round(y, 3),
                "w": round(max(3.0, fit.x_axis.pitch * 0.55), 3),
                "h": round(max(3.0, fit.y_axis.pitch * 0.55), 3),
                "row": int(y_center_idx + 1),
                "col": int(x_center_idx + 1),
                "group": f"band_{band:02d}_group_{x_group:02d}",
            }
        )

    groups: list[dict[str, Any]] = []
    for band_index, y_group in enumerate(y_bands, start=1):
        for group_index, x_group in enumerate(x_groups, start=1):
            group_slots = [
                slot for slot in slots
                if slot["group"] == f"band_{band_index:02d}_group_{group_index:02d}"
            ]
            if not group_slots:
                continue
            xs = [float(slot["x"]) for slot in group_slots]
            ys = [float(slot["y"]) for slot in group_slots]
            groups.append(
                {
                    "id": f"band_{band_index:02d}_group_{group_index:02d}",
                    "slot_count": len(group_slots),
                    "bbox": [
                        round(min(xs), 3),
                        round(min(ys), 3),
                        round(max(xs) - min(xs), 3),
                        round(max(ys) - min(ys), 3),
                    ],
                }
            )
    return slots, groups


def decide_layout(fit: GridFit, forced_mode: str) -> str:
    if forced_mode != "auto":
        return forced_mode
    enough_points = fit.occupied_count >= 16
    many_ordered_rows = len(fit.y_axis.centers) >= 8 and len(fit.x_axis.centers) >= 8
    locally_regular = fit.x_axis.cv <= 0.28 and fit.y_axis.cv <= 0.28
    if enough_points and many_ordered_rows and locally_regular:
        if fit.x_axis.max_gap_ratio > 2.2 or fit.y_axis.max_gap_ratio > 2.2:
            return "grouped_slots"
        if fit.fill_ratio >= 0.72:
            return "regular_grid"
        return "staggered_grid"
    uniform_gaps = fit.x_axis.max_gap_ratio <= 1.9 and fit.y_axis.max_gap_ratio <= 1.9
    dense_fill = fit.fill_ratio >= 0.72
    stable_fit = fit.regularity_score >= 0.56 and fit.median_residual <= 0.42
    if enough_points and uniform_gaps and dense_fill and stable_fit:
        return "regular_grid"
    return "grouped_slots"


def color_for_group(group_name: str) -> tuple[int, int, int]:
    seed = sum(ord(ch) * (idx + 1) for idx, ch in enumerate(group_name))
    hue = seed % 180
    color = np.uint8([[[hue, 190, 255]]])
    bgr = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_overlay(
    image: np.ndarray,
    points: np.ndarray,
    slots: list[dict[str, Any]],
    layout_mode: str,
    fit: GridFit,
    output_path: Path,
) -> None:
    vis = image.copy()
    if len(points):
        x1 = int(max(0, math.floor(float(np.min(points[:, 0])) - 12)))
        y1 = int(max(0, math.floor(float(np.min(points[:, 1])) - 12)))
        x2 = int(min(image.shape[1] - 1, math.ceil(float(np.max(points[:, 0])) + 12)))
        y2 = int(min(image.shape[0] - 1, math.ceil(float(np.max(points[:, 1])) + 12)))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (70, 70, 70), 1)

    if layout_mode in {"regular_grid", "staggered_grid"}:
        x_min = int(round(min(fit.x_axis.centers) - fit.x_axis.pitch * 0.55))
        x_max = int(round(max(fit.x_axis.centers) + fit.x_axis.pitch * 0.55))
        y_min = int(round(min(fit.y_axis.centers) - fit.y_axis.pitch * 0.55))
        y_max = int(round(max(fit.y_axis.centers) + fit.y_axis.pitch * 0.55))
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1] - 1, x_max)
        y_max = min(image.shape[0] - 1, y_max)
        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 140, 180), 1)
        if layout_mode == "regular_grid":
            for x in fit.x_axis.centers:
                cv2.line(vis, (int(round(x)), y_min), (int(round(x)), y_max), (32, 80, 80), 1)
            for y in fit.y_axis.centers:
                cv2.line(vis, (x_min, int(round(y))), (x_max, int(round(y))), (32, 80, 80), 1)
        for slot in slots:
            cx = int(round(float(slot["x"])))
            cy = int(round(float(slot["y"])))
            cv2.circle(vis, (cx, cy), 4, (0, 220, 255), 1)
    else:
        for slot in slots:
            cx = int(round(float(slot["x"])))
            cy = int(round(float(slot["y"])))
            color = color_for_group(str(slot["group"]))
            cv2.circle(vis, (cx, cy), 5, color, 1)
        by_group: dict[str, list[dict[str, Any]]] = {}
        for slot in slots:
            by_group.setdefault(str(slot["group"]), []).append(slot)
        for group_name, group_slots in by_group.items():
            xs = [float(slot["x"]) for slot in group_slots]
            ys = [float(slot["y"]) for slot in group_slots]
            x1 = int(min(xs) - 8)
            y1 = int(min(ys) - 8)
            x2 = int(max(xs) + 8)
            y2 = int(max(ys) + 8)
            color = color_for_group(group_name)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
            cv2.putText(vis, group_name.replace("band_", "b").replace("_group_", "g"), (x1, max(16, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    for x, y in points:
        cv2.circle(vis, (int(round(float(x))), int(round(float(y)))), 2, (0, 255, 0), -1)

    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, 34), (0, 0, 0), -1)
    title = (
        f"{layout_mode} pts={len(points)} sl={len(slots)} "
        f"rxc={len(fit.y_axis.centers)}x{len(fit.x_axis.centers)} "
        f"score={fit.regularity_score:.2f}"
    )
    cv2.putText(vis, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def analyze_image(image_path: Path, output_dir: Path, mode: str) -> dict[str, Any]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detected_points, stats = detect_points(image)
    points = points_to_array(detected_points)
    if len(points) == 0:
        raise RuntimeError(f"No LED points detected in {image_path}")

    fit = compute_grid_fit(points, image.shape[1], image.shape[0])
    layout_mode = decide_layout(fit, mode)
    if layout_mode in {"regular_grid", "staggered_grid"}:
        slots = make_regular_slots(points, fit)
        groups: list[dict[str, Any]] = [
            {
                "id": layout_mode,
                "slot_count": len(slots),
                "rows": len(fit.y_axis.centers),
                "cols": len(fit.x_axis.centers),
            }
        ]
    else:
        slots, groups = make_grouped_slots(points, fit)

    image_out_dir = output_dir / image_path.stem
    overlay_path = image_out_dir / "slot_overlay.jpg"
    draw_overlay(image, points, slots, layout_mode, fit, overlay_path)

    recipe = {
        "type": "led_slot_array",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_image": str(image_path),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "layout_mode": layout_mode,
        "point_detector": {
            "source": "build_direct_point_templates.detect_points",
            "stats": stats,
        },
        "fit": {
            "regularity_score": round(fit.regularity_score, 5),
            "fill_ratio": round(fit.fill_ratio, 5),
            "occupied_count": int(fit.occupied_count),
            "expected_count": int(fit.expected_count),
            "median_residual": round(fit.median_residual, 5),
            "x": asdict(fit.x_axis),
            "y": asdict(fit.y_axis),
        },
        "groups": groups,
        "slots": slots,
        "state": {
            "encoding": "visible_points_on",
            "on_slot_ids": [slot["id"] for slot in slots],
            "note": "Single-image demo. For production, learn slots from all-on image or multi-frame good video.",
        },
        "debug": {
            "overlay": str(overlay_path),
        },
    }
    recipe_path = image_out_dir / "led_slot_array_recipe.json"
    write_json(recipe_path, recipe)

    summary = {
        "image": str(image_path),
        "layout_mode": layout_mode,
        "detected_points": int(len(points)),
        "slot_count": int(len(slots)),
        "group_count": int(len(groups)),
        "regularity_score": round(fit.regularity_score, 5),
        "fill_ratio": round(fit.fill_ratio, 5),
        "x_gap_ratio": round(fit.x_axis.max_gap_ratio, 5),
        "y_gap_ratio": round(fit.y_axis.max_gap_ratio, 5),
        "rows": len(fit.y_axis.centers),
        "cols": len(fit.x_axis.centers),
        "recipe": str(recipe_path),
        "overlay": str(overlay_path),
    }
    write_json(image_out_dir / "summary.json", summary)
    return summary


def image_paths_from_dir(input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(input_dir.glob(pattern))
    return sorted(paths)


def make_synthetic_grid(path: Path) -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    glow = np.zeros_like(image)
    rows, cols = 16, 27
    x0, y0 = 37, 29
    dx, dy = 9.2, 9.4
    for row in range(rows):
        for col in range(cols):
            x = int(round(x0 + col * dx))
            y = int(round(y0 + row * dy))
            cv2.circle(glow, (x, y), 4, (0, 42, 90), -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 2, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 1, (80, 230, 255), -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (5, 5), 0)
    image = cv2.add(image, glow)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def make_synthetic_grouped(path: Path) -> None:
    image = np.zeros((480, 760, 3), dtype=np.uint8)
    band_ys = [54, 276]
    group_x0 = 82
    group_gap = 62
    local_x = [0, 16]
    dy = 13
    rows_per_band = 14
    for band_y in band_ys:
        for group in range(10):
            gx = group_x0 + group * group_gap
            for lx in local_x:
                for row in range(rows_per_band):
                    x = int(round(gx + lx))
                    y = int(round(band_y + row * dy))
                    cv2.circle(image, (x, y), 2, (0, 130, 255), -1, cv2.LINE_AA)
                    cv2.circle(image, (x, y), 4, (0, 55, 95), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Demo: learn a unified LED slot-array recipe from regular-grid or grouped-slot images."
    )
    parser.add_argument("--image", action="append", type=Path, default=[], help="Input image. Can be repeated.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory of input images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=["auto", "regular_grid", "staggered_grid", "grouped_slots"], default="auto")
    parser.add_argument("--make-synthetic-demo", action="store_true", help="Generate two synthetic demo images and analyze them.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir: Path = args.output_dir
    paths: list[Path] = list(args.image)

    if args.input_dir:
        paths.extend(image_paths_from_dir(args.input_dir))

    if args.make_synthetic_demo:
        synthetic_dir = output_dir / "_synthetic_inputs"
        grid_path = synthetic_dir / "regular_grid_demo.jpg"
        grouped_path = synthetic_dir / "grouped_slots_demo.jpg"
        make_synthetic_grid(grid_path)
        make_synthetic_grouped(grouped_path)
        paths.extend([grid_path, grouped_path])

    unique_paths = []
    seen = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if key not in seen:
            unique_paths.append(path)
            seen.add(key)

    if not unique_paths:
        raise SystemExit("No input images. Use --image, --input-dir, or --make-synthetic-demo.")

    summaries = [analyze_image(path, output_dir, args.mode) for path in unique_paths]
    write_json(
        output_dir / "demo_summary.json",
        {
            "type": "led_slot_array_demo_summary",
            "output_dir": str(output_dir),
            "image_count": len(summaries),
            "results": summaries,
        },
    )
    write_csv(
        output_dir / "demo_summary.csv",
        summaries,
        [
            "image",
            "layout_mode",
            "detected_points",
            "slot_count",
            "group_count",
            "regularity_score",
            "fill_ratio",
            "x_gap_ratio",
            "y_gap_ratio",
            "rows",
            "cols",
            "recipe",
            "overlay",
        ],
    )
    print(json.dumps({"output_dir": str(output_dir), "results": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
