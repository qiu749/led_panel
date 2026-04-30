from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from try_overexposed_point_detection import (
    IMAGE_EXTS,
    Params,
    add_label,
    detect_points,
    gray_to_bgr,
    imread_unicode,
    imwrite_unicode,
    normalize_to_u8,
    overexposure_metrics,
)


@dataclass(frozen=True)
class FilterParams:
    margin_x_ratio: float = 0.04
    margin_top_ratio: float = 0.08
    margin_bottom_ratio: float = 0.06
    right_label_x_ratio: float = 0.94
    right_label_y_ratio: float = 0.38
    row_cluster_tolerance: float = 16.0
    row_keep_tolerance: float = 14.0
    row_min_points: int = 4
    max_main_rows: int = 8
    neighbor_y_tolerance: float = 14.0
    neighbor_min_dx: float = 5.0
    neighbor_max_dx: float = 65.0
    min_horizontal_neighbors: int = 1
    min_row_retention_ratio: float = 0.92
    min_neighbor_retention_ratio: float = 0.95


def point_xy(point: dict[str, float | int]) -> tuple[float, float]:
    return float(point["x"]), float(point["y"])


def in_valid_area(point: dict[str, float | int], width: int, height: int, params: FilterParams) -> bool:
    x, y = point_xy(point)
    left = width * params.margin_x_ratio
    right = width * (1.0 - params.margin_x_ratio)
    top = height * params.margin_top_ratio
    bottom = height * (1.0 - params.margin_bottom_ratio)
    if x < left or x > right or y < top or y > bottom:
        return False

    # The printed "N" mark near the upper-right corner creates compact false peaks.
    if x > width * params.right_label_x_ratio and y < height * params.right_label_y_ratio:
        return False
    return True


def cluster_rows(points: list[dict[str, float | int]], tolerance: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in sorted(points, key=lambda item: float(item["y"])):
        y = float(point["y"])
        if not rows:
            rows.append({"ys": [y], "points": [point]})
            continue
        current = rows[-1]
        median_y = float(np.median(current["ys"]))
        if abs(y - median_y) <= tolerance:
            current["ys"].append(y)
            current["points"].append(point)
        else:
            rows.append({"ys": [y], "points": [point]})

    for row in rows:
        row["center_y"] = float(np.median(row["ys"]))
        row["count"] = len(row["points"])
    return rows


def choose_main_rows(rows: list[dict[str, object]], params: FilterParams) -> list[dict[str, object]]:
    candidates = [row for row in rows if int(row["count"]) >= params.row_min_points]
    candidates.sort(key=lambda row: int(row["count"]), reverse=True)
    main_rows = candidates[: params.max_main_rows]
    main_rows.sort(key=lambda row: float(row["center_y"]))
    return main_rows


def nearest_row_distance(point: dict[str, float | int], rows: list[dict[str, object]]) -> float:
    if not rows:
        return float("inf")
    y = float(point["y"])
    return min(abs(y - float(row["center_y"])) for row in rows)


def has_horizontal_neighbors(
    point: dict[str, float | int],
    points: list[dict[str, float | int]],
    params: FilterParams,
) -> bool:
    x, y = point_xy(point)
    neighbors = 0
    for other in points:
        if other is point:
            continue
        ox, oy = point_xy(other)
        dx = abs(ox - x)
        dy = abs(oy - y)
        if (
            dy <= params.neighbor_y_tolerance
            and params.neighbor_min_dx <= dx <= params.neighbor_max_dx
        ):
            neighbors += 1
            if neighbors >= params.min_horizontal_neighbors:
                return True
    return False


def filter_points(
    points: list[dict[str, float | int]],
    width: int,
    height: int,
    params: FilterParams,
) -> dict[str, object]:
    area_points = [point for point in points if in_valid_area(point, width, height, params)]
    rows = cluster_rows(area_points, params.row_cluster_tolerance)
    main_rows = choose_main_rows(rows, params)
    row_points_raw = [
        point
        for point in area_points
        if nearest_row_distance(point, main_rows) <= params.row_keep_tolerance
    ]
    row_filter_guarded = bool(
        area_points and len(row_points_raw) < len(area_points) * params.min_row_retention_ratio
    )
    row_points = area_points if row_filter_guarded else row_points_raw

    neighbor_points_raw = [
        point
        for point in row_points
        if has_horizontal_neighbors(point, row_points, params)
    ]
    neighbor_filter_guarded = bool(
        row_points
        and len(neighbor_points_raw) < len(row_points) * params.min_neighbor_retention_ratio
    )
    neighbor_points = row_points if neighbor_filter_guarded else neighbor_points_raw

    return {
        "area_points": area_points,
        "rows": rows,
        "main_rows": main_rows,
        "row_points_raw": row_points_raw,
        "row_points": row_points,
        "neighbor_points_raw": neighbor_points_raw,
        "neighbor_points": neighbor_points,
        "row_filter_guarded": row_filter_guarded,
        "neighbor_filter_guarded": neighbor_filter_guarded,
    }


def draw_points(
    image: np.ndarray,
    points: list[dict[str, float | int]],
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    output = image.copy()
    for point in points:
        x = int(point["x"])
        y = int(point["y"])
        cv2.circle(output, (x, y), 5, color, 1, cv2.LINE_AA)
        cv2.circle(output, (x, y), 1, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(output, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output


def draw_valid_area(image: np.ndarray, params: FilterParams) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    left = int(round(width * params.margin_x_ratio))
    right = int(round(width * (1.0 - params.margin_x_ratio)))
    top = int(round(height * params.margin_top_ratio))
    bottom = int(round(height * (1.0 - params.margin_bottom_ratio)))
    cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 80), 2, cv2.LINE_AA)
    rx = int(round(width * params.right_label_x_ratio))
    ry = int(round(height * params.right_label_y_ratio))
    cv2.rectangle(output, (rx, 0), (width - 1, ry), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "green=valid, red=forbidden", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return output


def draw_row_filter(
    image: np.ndarray,
    raw_points: list[dict[str, float | int]],
    kept_points: list[dict[str, float | int]],
    main_rows: list[dict[str, object]],
    label: str,
) -> np.ndarray:
    output = image.copy()
    kept_ids = {id(point) for point in kept_points}
    width = output.shape[1]
    for row in main_rows:
        y = int(round(float(row["center_y"])))
        cv2.line(output, (0, y), (width - 1, y), (0, 220, 255), 1, cv2.LINE_AA)
    for point in raw_points:
        x = int(point["x"])
        y = int(point["y"])
        color = (0, 255, 80) if id(point) in kept_ids else (0, 0, 255)
        cv2.circle(output, (x, y), 4, color, 1, cv2.LINE_AA)
    cv2.putText(output, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return output


def make_contact_sheet(
    result: dict[str, object],
    filtered: dict[str, object],
    params: FilterParams,
) -> np.ndarray:
    resized = result["resized"]
    raw_points = result["points"]
    area_points = filtered["area_points"]
    row_points = filtered["row_points"]
    final_points = filtered["neighbor_points"]
    main_rows = filtered["main_rows"]

    panels = [
        resized,
        draw_points(resized, raw_points, (0, 220, 255), f"01 raw v3 candidates: {len(raw_points)}"),
        draw_valid_area(resized, params),
        draw_row_filter(resized, area_points, row_points, main_rows, f"04 row filter kept: {len(row_points)}"),
        draw_row_filter(resized, row_points, final_points, main_rows, f"05 neighbor kept: {len(final_points)}"),
        draw_points(resized, final_points, (0, 255, 80), f"06 final filtered: {len(final_points)}"),
    ]
    labels = [
        "00 original / resized",
        "01 raw overexposed peaks",
        "02 valid area + label forbidden zone",
        "03 row-cluster filter",
        "04 horizontal-neighbor filter",
        "05 final overlay",
    ]
    labelled = [add_label(panel, label) for panel, label in zip(panels, labels)]
    return np.vstack([np.hstack(labelled[:3]), np.hstack(labelled[3:])])


def params_to_rows(params: object) -> list[str]:
    return [f"{name}: {getattr(params, name)}" for name in params.__dataclass_fields__]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try v4 edge/structure filters on overexposed free point detection.")
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\Desktop2\Images\roi_crops"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\Desktop2\Images\roi_crops_free_point_steps_v4_filtered"))
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images when > 0.")
    parser.add_argument("--pattern", default="", help="Only process files containing this substring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detection_params = Params()
    filter_params = FilterParams()

    output_dir: Path = args.output_dir
    steps_dir = output_dir / "steps_contact"
    overlay_dir = output_dir / "points_overlay"
    points_dir = output_dir / "points_csv"
    steps_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    points_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [
            path
            for path in args.input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTS
            and (not args.pattern or args.pattern in path.name)
        ],
        key=lambda path: path.name,
    )
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    rows: list[dict[str, object]] = []
    for index, path in enumerate(files, 1):
        image = imread_unicode(path)
        if image is None:
            rows.append({"filename": path.name, "error": "read_failed"})
            continue

        metrics = overexposure_metrics(image)
        result = detect_points(image, detection_params)
        resized = result["resized"]
        height, width = resized.shape[:2]
        filtered = filter_points(result["points"], width, height, filter_params)
        final_points = filtered["neighbor_points"]

        stem = path.stem
        contact_path = steps_dir / f"{stem}_steps.jpg"
        overlay_path = overlay_dir / f"{stem}_points.jpg"
        points_path = points_dir / f"{stem}_points.csv"

        contact = make_contact_sheet(result, filtered, filter_params)
        overlay = draw_points(resized, final_points, (0, 255, 80), f"final={len(final_points)}")
        imwrite_unicode(contact_path, contact, quality=86)
        imwrite_unicode(overlay_path, overlay, quality=90)

        with points_path.open("w", newline="", encoding="utf-8-sig") as point_file:
            writer = csv.DictWriter(
                point_file,
                fieldnames=[
                    "x",
                    "y",
                    "response",
                    "brightness",
                    "center_mean",
                    "center_max",
                    "ring_mean",
                    "center_contrast",
                    "peak_contrast",
                ],
            )
            writer.writeheader()
            writer.writerows(final_points)

        rows.append(
            {
                "filename": path.name,
                "original_width": image.shape[1],
                "original_height": image.shape[0],
                "resized_width": width,
                "resized_height": height,
                "mean_v": f"{metrics['mean_v']:.3f}",
                "p50_v": f"{metrics['p50_v']:.3f}",
                "p99_v": f"{metrics['p99_v']:.3f}",
                "bright_ratio_v220": f"{metrics['bright_ratio_v220']:.6f}",
                "raw_count": len(result["points"]),
                "after_valid_area": len(filtered["area_points"]),
                "row_count": len(filtered["main_rows"]),
                "after_row_filter_raw": len(filtered["row_points_raw"]),
                "after_row_filter": len(filtered["row_points"]),
                "after_neighbor_filter_raw": len(filtered["neighbor_points_raw"]),
                "after_neighbor_filter": len(final_points),
                "row_filter_guarded": filtered["row_filter_guarded"],
                "neighbor_filter_guarded": filtered["neighbor_filter_guarded"],
                "point_count": len(final_points),
                "steps_image": str(contact_path),
                "overlay_image": str(overlay_path),
                "points_csv": str(points_path),
                "error": "",
            }
        )
        if index % 100 == 0:
            print(f"processed {index}/{len(files)}")

    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "filename",
        "original_width",
        "original_height",
        "resized_width",
        "resized_height",
        "mean_v",
        "p50_v",
        "p99_v",
        "bright_ratio_v220",
        "raw_count",
        "after_valid_area",
        "row_count",
        "after_row_filter_raw",
        "after_row_filter",
        "after_neighbor_filter_raw",
        "after_neighbor_filter",
        "row_filter_guarded",
        "neighbor_filter_guarded",
        "point_count",
        "steps_image",
        "overlay_image",
        "points_csv",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8-sig") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    readme_path = output_dir / "README.txt"
    readme_lines = [
        "Free point detection v4 filtered experiment",
        f"Input: {args.input_dir}",
        f"Output: {output_dir}",
        f"Images processed: {len(files)}",
        "",
        "Method:",
        "1. Run v3 overexposed detection first.",
        "2. Remove points outside the valid ROI interior and the upper-right label forbidden zone.",
        "3. Cluster y coordinates into rows, keep only dense main LED rows.",
        "4. Keep only points with at least one same-row horizontal neighbor.",
        "",
        "Contact sheet panels:",
        "00 original/resized, 01 raw v3 candidates, 02 valid area, 03 row filter, 04 neighbor filter, 05 final overlay.",
        "",
        "Detection parameters:",
        *params_to_rows(detection_params),
        "",
        "Filter parameters:",
        *params_to_rows(filter_params),
    ]
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    counts = [int(row["point_count"]) for row in rows if row.get("error", "") == ""]
    print("done")
    print(f"out_dir={output_dir}")
    print(f"processed={len(rows)}")
    if counts:
        print(f"point_count_min={min(counts)} median={float(np.median(counts)):.1f} max={max(counts)}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
