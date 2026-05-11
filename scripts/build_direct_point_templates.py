from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def parse_template_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("template must use name=path")
    name, path = value.split("=", 1)
    return name.strip(), Path(path.strip())


def detect_points(image: np.ndarray) -> tuple[list[tuple[float, float]], dict[str, float]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    threshold = max(20, int(round(float(np.percentile(blur, 90)) * 0.78)))
    dilated = cv2.dilate(blur, np.ones((7, 7), np.uint8))
    peak_mask = np.logical_and(blur >= threshold, blur >= (dilated - 1)).astype(np.uint8) * 255
    count, _, stats, centers = cv2.connectedComponentsWithStats(peak_mask, connectivity=8)

    component_stats = stats[1:] if count > 1 else np.empty((0, 5), dtype=np.int32)
    if len(component_stats):
        areas = component_stats[:, cv2.CC_STAT_AREA].astype(np.float32)
        widths = component_stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
        heights = component_stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
        max_dims = np.maximum(widths, heights)
        # Saturated frames produce valid blobs much larger than the crisp template dots.
        # Keep the upper bound adaptive so those blobs are not discarded as noise.
        max_area = min(160.0, max(30.0, float(np.percentile(areas, 95)) * 1.6))
        max_dim = min(16.0, max(4.0, float(np.percentile(max_dims, 95)) * 1.35))
    else:
        max_area = 30.0
        max_dim = 16.0

    points: list[tuple[float, float]] = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 1 or area > max_area or max(w, h) > max_dim:
            continue
        points.append((float(centers[index][0]), float(centers[index][1])))
    return points, {
        "threshold": float(threshold),
        "raw_count": float(len(points)),
        "max_area": float(max_area),
        "max_dim": float(max_dim),
    }


def draw_points(points: list[tuple[float, float]], width: int, height: int, title: str, output_path: Path) -> None:
    canvas = np.zeros((height + 36, width, 3), dtype=np.uint8)
    for x, y in points:
        cx = int(round(x))
        cy = int(round(y)) + 18
        cv2.circle(canvas, (cx, cy), 7, (0, 80, 120), 1)
        cv2.circle(canvas, (cx, cy), 4, (0, 180, 255), -1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 32), (0, 0, 0), -1)
    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def build_library(template_items: list[tuple[str, Path]], output_dir: Path) -> dict[str, Any]:
    templates: list[dict[str, Any]] = []
    mask_dir = output_dir / "pure_point_masks"
    comparison_items: list[Path] = []
    image_size: tuple[int, int] | None = None

    for name, path in template_items:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        if image_size is None:
            image_size = (image.shape[1], image.shape[0])
        points, stats = detect_points(image)
        out_path = mask_dir / f"{name}_direct_points.jpg"
        draw_points(points, image.shape[1], image.shape[0], f"{name} points={len(points)}", out_path)
        comparison_items.append(out_path)
        templates.append(
            {
                "name": name,
                "source_image": str(path),
                "point_mask_image": str(out_path),
                "point_count": len(points),
                "points": [{"x": round(x, 3), "y": round(y, 3)} for x, y in points],
                "detection_stats": stats,
            }
        )

    imgs = [cv2.imread(str(path)) for path in comparison_items]
    sep = 12
    height = max(img.shape[0] for img in imgs if img is not None)
    parts: list[np.ndarray] = []
    for index, img in enumerate(imgs):
        assert img is not None
        if img.shape[0] < height:
            pad = np.zeros((height - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        parts.append(img)
        if index != len(imgs) - 1:
            parts.append(np.full((height, sep, 3), 24, dtype=np.uint8))
    comparison = np.hstack(parts)
    comparison_path = output_dir / "direct_point_template_comparison.jpg"
    cv2.imwrite(str(comparison_path), comparison)

    return {
        "type": "direct_point_template_library",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_size": list(image_size) if image_size is not None else None,
        "comparison_image": str(comparison_path),
        "templates": templates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build direct point-coordinate templates from user-picked images.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", action="append", type=parse_template_arg, required=True, help="name=path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_library(args.template, args.output_dir)
    output_path = args.output_dir / "direct_point_templates.json"
    write_json(output_path, payload)
    print(json.dumps({"template": str(output_path), "templates": len(payload["templates"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
