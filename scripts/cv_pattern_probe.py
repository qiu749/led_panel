from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CLASS_IDS = {
    "ledgroup": 0,
    "ledgroup2": 1,
    "ledgroup3": 2,
    "unknown": -1,
}

EXPECTED_POINT_COUNTS = {
    "ledgroup": 50,
    "ledgroup2": 26,
    "ledgroup3": 24,
}

CLASS_COLORS = {
    "ledgroup": (0, 0, 255),
    "ledgroup2": (0, 255, 0),
    "ledgroup3": (255, 0, 0),
    "unknown": (0, 255, 255),
}


@dataclass
class RoiBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    buffer.tofile(str(path))


def sort_boxes_by_grid(boxes: list[RoiBox]) -> list[RoiBox]:
    if not boxes:
        return []
    heights = [box.h for box in boxes]
    row_tolerance = max(30.0, float(np.median(heights)) * 0.45)
    rows: list[list[RoiBox]] = []
    for box in sorted(boxes, key=lambda item: item.y + item.h * 0.5):
        cy = box.y + box.h * 0.5
        if not rows:
            rows.append([box])
            continue
        row_cy = float(np.median([item.y + item.h * 0.5 for item in rows[-1]]))
        if abs(cy - row_cy) <= row_tolerance:
            rows[-1].append(box)
        else:
            rows.append([box])
    sorted_boxes: list[RoiBox] = []
    for row in rows:
        sorted_boxes.extend(sorted(row, key=lambda item: item.x + item.w * 0.5))
    return sorted_boxes


def detect_roi_boxes(
    image: np.ndarray,
    threshold: int = 60,
    kernel_size: int = 15,
    dilate_iterations: int = 2,
) -> tuple[list[RoiBox], np.ndarray]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    merged = cv2.dilate(binary, kernel, iterations=dilate_iterations)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[RoiBox] = []
    max_area = width * height / 4
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > 1000 and area < max_area and h > w * 1.5:
            boxes.append(RoiBox(x=x, y=y, w=w, h=h))
    return sort_boxes_by_grid(boxes), merged


def merge_close_points(points: list[dict[str, float]], radius: float) -> list[dict[str, float]]:
    if len(points) < 2 or radius <= 0:
        return points

    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    radius_sq = radius * radius
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            dx = points[left]["x"] - points[right]["x"]
            dy = points[left]["y"] - points[right]["y"]
            if dx * dx + dy * dy <= radius_sq:
                union(left, right)

    groups: dict[int, list[dict[str, float]]] = {}
    for index, point in enumerate(points):
        groups.setdefault(find(index), []).append(point)

    merged: list[dict[str, float]] = []
    for group in groups.values():
        total_area = sum(point["area"] for point in group)
        if total_area <= 0:
            total_area = float(len(group))
        merged.append(
            {
                "x": sum(point["x"] * point["area"] for point in group) / total_area,
                "y": sum(point["y"] * point["area"] for point in group) / total_area,
                "area": total_area,
                "w": max(point["w"] for point in group),
                "h": max(point["h"] for point in group),
            }
        )
    return sorted(merged, key=lambda point: (point["y"], point["x"]))


def remove_edge_singleton_points(points: list[dict[str, float]], width: int) -> list[dict[str, float]]:
    if len(points) < 2 or width <= 0:
        return points

    tolerance = max(8.0, width * 0.06)
    clusters: list[list[int]] = []
    for index, point in sorted(enumerate(points), key=lambda item: item[1]["x"]):
        if not clusters:
            clusters.append([index])
            continue
        center = float(np.median([points[item]["x"] for item in clusters[-1]]))
        if abs(point["x"] - center) <= tolerance:
            clusters[-1].append(index)
        else:
            clusters.append([index])

    remove: set[int] = set()
    for cluster in clusters:
        center = float(np.median([points[item]["x"] for item in cluster]))
        is_edge = center < width * 0.2 or center > width * 0.8
        if len(cluster) == 1 and is_edge:
            remove.add(cluster[0])

    if not remove:
        return points
    return [point for index, point in enumerate(points) if index not in remove]


def detect_led_points(
    roi: np.ndarray,
    merge_radius: float = 14.0,
    value_threshold: int = 120,
    saturation_threshold: int = 70,
    red_threshold: int = 120,
    green_threshold: int = 60,
) -> tuple[list[dict[str, float]], np.ndarray]:
    if roi.size == 0:
        return [], np.zeros((1, 1), dtype=np.uint8)
    b, g, r = cv2.split(roi)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = (
        (v >= value_threshold)
        & (s >= saturation_threshold)
        & (r >= red_threshold)
        & (g >= green_threshold)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    points: list[dict[str, float]] = []
    roi_area = roi.shape[0] * roi.shape[1]
    max_area = max(450, roi_area * 0.02)
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if 6 <= area <= max_area and 2 <= w <= 30 and 2 <= h <= 30:
            component = labels == idx
            mean_r = float(r[component].mean())
            mean_g = float(g[component].mean())
            mean_h = float(hsv[:, :, 0][component].mean())
            if area < 50 and mean_g / (mean_r + 1e-6) < 0.5 and mean_h < 10:
                continue
            points.append(
                {
                    "x": float(centers[idx][0]),
                    "y": float(centers[idx][1]),
                    "area": float(area),
                    "w": float(w),
                    "h": float(h),
                }
            )
    points = merge_close_points(points, merge_radius)
    points = remove_edge_singleton_points(points, roi.shape[1])
    return points, mask


def cluster_axis(values: list[float], tolerance: float) -> list[list[float]]:
    clusters: list[list[float]] = []
    for value in sorted(values):
        if not clusters:
            clusters.append([value])
            continue
        center = float(np.median(clusters[-1]))
        if abs(value - center) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def coefficient_variation(values: list[int]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.std() / (arr.mean() + 1e-6))


def nearest_count_class(point_count: int) -> tuple[str, int, int]:
    cls, expected = min(
        EXPECTED_POINT_COUNTS.items(),
        key=lambda item: (abs(point_count - item[1]), 0 if item[0] == "ledgroup2" else 1),
    )
    return cls, expected, point_count - expected


def projection_class(roi: np.ndarray) -> tuple[str, dict[str, float]]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    x_projection = np.mean(binary, axis=0)
    y_projection = np.mean(binary, axis=1)
    x_std = float(np.std(x_projection))
    y_std = float(np.std(y_projection))
    threshold = 20.0
    if x_std > threshold and y_std < threshold:
        cls = "ledgroup2"
    elif y_std > threshold and x_std < threshold:
        cls = "ledgroup3"
    elif x_std > threshold and y_std > threshold:
        if x_std > y_std * 1.5:
            cls = "ledgroup2"
        elif y_std > x_std * 1.5:
            cls = "ledgroup3"
        else:
            cls = "ledgroup"
    else:
        cls = "ledgroup"
    return cls, {"projection_x_std": x_std, "projection_y_std": y_std}


def point_structure_class(roi: np.ndarray, points: list[dict[str, float]]) -> tuple[str, dict[str, Any]]:
    if not points:
        return "unknown", {
            "row_counts": [],
            "col_counts": [],
            "row_cv": 0.0,
            "col_cv": 0.0,
            "reason": "no_points",
        }

    height, width = roi.shape[:2]
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    row_tol = max(8.0, height * 0.025)
    col_tol = max(8.0, width * 0.06)
    row_clusters = cluster_axis(ys, row_tol)
    col_clusters = cluster_axis(xs, col_tol)
    row_centers = [float(np.median(cluster)) for cluster in row_clusters]
    col_centers = [float(np.median(cluster)) for cluster in col_clusters]

    row_counts = [0 for _ in row_centers]
    col_counts = [0 for _ in col_centers]
    for point in points:
        row_idx = int(np.argmin([abs(point["y"] - center) for center in row_centers]))
        col_idx = int(np.argmin([abs(point["x"] - center) for center in col_centers]))
        row_counts[row_idx] += 1
        col_counts[col_idx] += 1

    row_cv = coefficient_variation(row_counts)
    col_cv = coefficient_variation(col_counts)
    point_count = len(points)
    count_cls, expected_count, count_delta = nearest_count_class(point_count)

    if abs(count_delta) <= 2:
        cls = count_cls
        reason = "count_nearest"
    elif point_count >= 38:
        cls = "ledgroup"
        reason = "count_high_full"
    elif row_cv > col_cv * 1.25 and row_cv > 0.45:
        cls = "ledgroup2"
        reason = "row_variation_dominates"
    elif col_cv > row_cv * 1.25 and col_cv > 0.45:
        cls = "ledgroup3"
        reason = "col_variation_dominates"
    elif count_cls != "ledgroup":
        cls = count_cls
        reason = "count_nearest_loose"
    elif col_cv >= row_cv:
        cls = "ledgroup3"
        reason = "fallback_col_variation"
    else:
        cls = "ledgroup2"
        reason = "fallback_row_variation"

    return cls, {
        "row_counts": row_counts,
        "col_counts": col_counts,
        "row_cv": row_cv,
        "col_cv": col_cv,
        "row_count": len(row_counts),
        "col_count": len(col_counts),
        "expected_point_count": expected_count,
        "point_count_delta": count_delta,
        "count_status": "ok" if count_delta == 0 else "abnormal",
        "reason": reason,
    }


def analyze_roi(roi: np.ndarray, point_merge_radius: float) -> dict[str, Any]:
    points, point_mask = detect_led_points(roi, point_merge_radius)
    point_cls, point_features = point_structure_class(roi, points)
    proj_cls, proj_features = projection_class(roi)
    detection_mode = "base"

    if point_cls == "ledgroup" and 45 <= len(points) < EXPECTED_POINT_COUNTS["ledgroup"]:
        soft_points, soft_mask = detect_led_points(
            roi,
            point_merge_radius,
            value_threshold=105,
            saturation_threshold=60,
            red_threshold=100,
            green_threshold=50,
        )
        base_delta = abs(len(points) - EXPECTED_POINT_COUNTS["ledgroup"])
        soft_delta = abs(len(soft_points) - EXPECTED_POINT_COUNTS["ledgroup"])
        if soft_delta < base_delta:
            points = soft_points
            point_mask = soft_mask
            point_cls, point_features = point_structure_class(roi, points)
            detection_mode = "soft_recovery"

    if point_cls == "unknown":
        final_cls = proj_cls
        final_reason = "projection_fallback"
    else:
        final_cls = point_cls
        final_reason = point_features["reason"]

    return {
        "class_name": final_cls,
        "class_id": CLASS_IDS[final_cls],
        "point_count": len(points),
        "points": points,
        "point_class": point_cls,
        "projection_class": proj_cls,
        "reason": final_reason,
        "features": {**point_features, **proj_features, "detection_mode": detection_mode},
        "point_mask": point_mask,
    }


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 4, y)
    cv2.rectangle(image, (x, y - th - 5), (x + tw + 4, y + baseline + 4), (15, 15, 15), -1)
    cv2.putText(image, text, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)


def count_classes(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        cls = str(row.get(key, "unknown"))
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def dominant_class(counts: dict[str, int]) -> tuple[str | None, int]:
    valid_counts = {key: value for key, value in counts.items() if key != "unknown"}
    if not valid_counts:
        return None, 0
    cls, count = max(valid_counts.items(), key=lambda item: item[1])
    return cls, count


def apply_frame_majority_smoothing(
    rows: list[dict[str, Any]],
    threshold: float,
    min_roi_count: int = 10,
) -> None:
    """Use the frame-level dominant state to suppress isolated ROI class jumps."""
    if len(rows) < min_roi_count:
        return

    counts = count_classes(rows, "raw_class_name")
    majority_cls, majority_count = dominant_class(counts)
    if majority_cls is None:
        return

    confidence = majority_count / max(1, len(rows))
    if confidence < threshold:
        return

    for row in rows:
        row["frame_majority_class"] = majority_cls
        row["frame_majority_count"] = majority_count
        row["frame_majority_ratio"] = confidence
        if row["raw_class_name"] != majority_cls:
            point_count = int(row["point_count"])
            raw_expected = EXPECTED_POINT_COUNTS.get(row["raw_class_name"], 0)
            majority_expected = EXPECTED_POINT_COUNTS.get(majority_cls, 0)
            raw_delta = abs(point_count - raw_expected) if raw_expected else 999
            majority_delta = abs(point_count - majority_expected) if majority_expected else 999
            if raw_delta + 2 < majority_delta:
                row["smoothing_applied"] = 0
                row["smoothing_reason"] = "kept_count_consistent_raw_class"
                continue
            row["class_name"] = majority_cls
            row["class_id"] = CLASS_IDS[majority_cls]
            row["smoothing_applied"] = 1
            row["smoothing_reason"] = "frame_majority"
        else:
            row["smoothing_applied"] = 0
            row["smoothing_reason"] = "none"


def annotate_count_status(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        expected = EXPECTED_POINT_COUNTS.get(row["class_name"], 0)
        point_count = int(row["point_count"])
        row["expected_point_count"] = expected
        row["point_count_delta"] = point_count - expected if expected else 0
        row["count_status"] = "ok" if expected and point_count == expected else "abnormal"


def process_image(
    image_path: Path,
    vis_dir: Path,
    majority_threshold: float,
    point_merge_radius: float,
) -> list[dict[str, Any]]:
    image = imread_unicode(image_path)
    if image is None:
        return []
    boxes, _ = detect_roi_boxes(image)
    vis = image.copy()
    items: list[dict[str, Any]] = []

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy
        roi = image[y1:y2, x1:x2]
        result = analyze_roi(roi, point_merge_radius)
        feature = result["features"]
        items.append(
            {
                "box": box,
                "points": result["points"],
                "image": image_path.name,
                "roi_index": index,
                "bbox": [x1, y1, x2, y2],
                "raw_class_name": result["class_name"],
                "raw_class_id": result["class_id"],
                "class_name": result["class_name"],
                "class_id": result["class_id"],
                "frame_majority_class": "",
                "frame_majority_count": 0,
                "frame_majority_ratio": 0.0,
                "smoothing_applied": 0,
                "smoothing_reason": "none",
                "point_count": result["point_count"],
                "point_class": result["point_class"],
                "projection_class": result["projection_class"],
                "reason": result["reason"],
                "expected_point_count": feature.get("expected_point_count", 0),
                "point_count_delta": feature.get("point_count_delta", 0),
                "count_status": feature.get("count_status", "abnormal"),
                "detection_mode": feature.get("detection_mode", "base"),
                "row_cv": feature.get("row_cv", 0.0),
                "col_cv": feature.get("col_cv", 0.0),
                "projection_x_std": feature.get("projection_x_std", 0.0),
                "projection_y_std": feature.get("projection_y_std", 0.0),
                "row_counts": feature.get("row_counts", []),
                "col_counts": feature.get("col_counts", []),
            }
        )

    apply_frame_majority_smoothing(items, majority_threshold)
    annotate_count_status(items)

    rows: list[dict[str, Any]] = []
    for item in items:
        box = item["box"]
        x1, y1, x2, y2 = box.xyxy
        cls = item["class_name"]
        raw_cls = item["raw_class_name"]
        color = CLASS_COLORS.get(cls, CLASS_COLORS["unknown"])

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        for point in item["points"]:
            px = int(round(x1 + point["x"]))
            py = int(round(y1 + point["y"]))
            cv2.circle(vis, (px, py), 4, (0, 255, 255), 1, cv2.LINE_AA)
        label = f"{item['roi_index'] + 1}:{cls} n={item['point_count']}"
        if item["smoothing_applied"]:
            label += f" raw={raw_cls}"
        draw_label(vis, label, x1, max(18, y1 - 6), color)

        item.pop("box")
        item.pop("points")
        rows.append(item)

    imwrite_unicode(vis_dir / f"vis_{image_path.stem}.jpg", vis)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe pure-CV ROI and LED pattern classification.")
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\Desktop2\Images\extracted_frames"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\Desktop2\Images\cv_pattern_probe"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--majority-threshold", type=float, default=0.6)
    parser.add_argument("--point-merge-radius", type=float, default=14.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vis_dir = args.output_dir / "visualized"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(args.input_dir.glob("*.jpg"))
    if args.limit > 0:
        image_files = image_files[: args.limit]

    all_rows: list[dict[str, Any]] = []
    image_summary: list[dict[str, Any]] = []
    for image_path in image_files:
        rows = process_image(image_path, vis_dir, args.majority_threshold, args.point_merge_radius)
        all_rows.extend(rows)
        class_counts = count_classes(rows, "class_name")
        raw_class_counts = count_classes(rows, "raw_class_name")
        image_summary.append(
            {
                "image": image_path.name,
                "roi_count": len(rows),
                "raw_class_counts": raw_class_counts,
                "class_counts": class_counts,
                "smoothing_applied_count": sum(int(row["smoothing_applied"]) for row in rows),
                "abnormal_count": sum(1 for row in rows if row["count_status"] != "ok"),
            }
        )
        print(
            f"{image_path.name}: roi={len(rows)} raw={raw_class_counts} "
            f"final={class_counts} smooth={sum(int(row['smoothing_applied']) for row in rows)}"
        )

    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "roi_index",
                "bbox",
                "raw_class_name",
                "raw_class_id",
                "class_name",
                "class_id",
                "frame_majority_class",
                "frame_majority_count",
                "frame_majority_ratio",
                "smoothing_applied",
                "smoothing_reason",
                "point_count",
                "expected_point_count",
                "point_count_delta",
                "count_status",
                "detection_mode",
                "point_class",
                "projection_class",
                "reason",
                "row_cv",
                "col_cv",
                "projection_x_std",
                "projection_y_std",
                "row_counts",
                "col_counts",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    json_path = args.output_dir / "image_summary.json"
    json_path.write_text(json.dumps(image_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    anomalies_path = args.output_dir / "anomalies.csv"
    with anomalies_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=writer.fieldnames)
        writer.writeheader()
        writer.writerows([row for row in all_rows if row["count_status"] != "ok"])
    print(f"summary: {csv_path}")
    print(f"image summary: {json_path}")
    print(f"anomalies: {anomalies_path}")
    print(f"visualized: {vis_dir}")


if __name__ == "__main__":
    main()
