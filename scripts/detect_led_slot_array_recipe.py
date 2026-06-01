from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def image_paths_from_dir(input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(input_dir.glob(pattern))
    return sorted(paths)


def load_expected(manifest_path: Path | None) -> dict[str, dict[str, Any]]:
    if manifest_path is None:
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[str, dict[str, Any]] = {}
    for item in manifest.get("tests", []):
        image = item.get("image", "")
        expected[Path(image).name.lower()] = item.get("expected", {})
    return expected


def slot_scores(image: np.ndarray, slots: list[dict[str, Any]], radius: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scores: list[float] = []
    for slot in slots:
        x = int(round(float(slot["x"])))
        y = int(round(float(slot["y"])))
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(gray.shape[1], x + radius + 1)
        y2 = min(gray.shape[0], y + radius + 1)
        roi = gray[y1:y2, x1:x2]
        scores.append(float(np.percentile(roi, 90)) if roi.size else 0.0)
    return np.asarray(scores, dtype=np.float32)


def draw_overlay(
    image: np.ndarray,
    slots: list[dict[str, Any]],
    missing_ids: set[int],
    dim_ids: set[int],
    extra_points: list[dict[str, float]],
    output_path: Path,
    title: str,
) -> None:
    vis = image.copy()
    for slot in slots:
        sid = int(slot["id"])
        x = int(round(float(slot["x"])))
        y = int(round(float(slot["y"])))
        if sid in missing_ids:
            cv2.circle(vis, (x, y), 6, (0, 0, 255), 2)
        elif sid in dim_ids:
            cv2.circle(vis, (x, y), 6, (0, 220, 255), 2)
        else:
            cv2.circle(vis, (x, y), 2, (0, 210, 0), -1)

    for item in extra_points:
        x = int(round(float(item["x"])))
        y = int(round(float(item["y"])))
        cv2.circle(vis, (x, y), 8, (255, 0, 0), 2)

    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, 28), (0, 0, 0), -1)
    cv2.putText(vis, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def detect_one(
    image_path: Path,
    recipe: dict[str, Any],
    ref_scores: np.ndarray,
    output_dir: Path,
    missing_ratio: float,
    dim_ratio: float,
    extra_radius: float,
    score_radius: int,
    expected_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    slots = recipe["slots"]
    slot_ids = np.asarray([int(slot["id"]) for slot in slots], dtype=np.int32)
    slot_xy = np.asarray([[float(slot["x"]), float(slot["y"])] for slot in slots], dtype=np.float32)
    image = load_image(image_path)
    scores = slot_scores(image, slots, score_radius)
    ratios = scores / np.maximum(ref_scores, 1.0)

    missing_ids = slot_ids[ratios < missing_ratio].astype(int).tolist()
    dim_ids = slot_ids[(ratios >= missing_ratio) & (ratios < dim_ratio)].astype(int).tolist()

    points, point_stats = detect_points(image)
    extra_points: list[dict[str, float]] = []
    if points:
        detected = np.asarray(points, dtype=np.float32)
        distances = np.sqrt(((detected[:, None, :] - slot_xy[None, :, :]) ** 2).sum(axis=2))
        nearest = distances.min(axis=1)
        for point, distance in zip(detected, nearest):
            if float(distance) > extra_radius:
                extra_points.append(
                    {
                        "x": round(float(point[0]), 2),
                        "y": round(float(point[1]), 2),
                        "nearest_distance": round(float(distance), 3),
                    }
                )

    overall_ok = len(missing_ids) == 0 and len(dim_ids) == 0 and len(extra_points) == 0
    result_dir = output_dir / image_path.stem
    overlay_path = result_dir / "detection_overlay.jpg"
    title = f"{image_path.name} miss={len(missing_ids)} dim={len(dim_ids)} extra={len(extra_points)}"
    draw_overlay(
        image=image,
        slots=slots,
        missing_ids=set(missing_ids),
        dim_ids=set(dim_ids),
        extra_points=extra_points,
        output_path=overlay_path,
        title=title,
    )

    expected = expected_by_name.get(image_path.name.lower(), {})
    row = {
        "image": str(image_path),
        "overall_ok": overall_ok,
        "missing_count": len(missing_ids),
        "dim_count": len(dim_ids),
        "extra_count": len(extra_points),
        "detected_point_count": int(point_stats.get("raw_count", len(points))),
        "missing_ids": missing_ids,
        "dim_ids": dim_ids,
        "extra_points": extra_points,
        "expected": expected,
        "overlay": str(overlay_path),
        "score_percentiles": {
            "p0": round(float(np.percentile(scores, 0)), 3),
            "p5": round(float(np.percentile(scores, 5)), 3),
            "p50": round(float(np.percentile(scores, 50)), 3),
            "p95": round(float(np.percentile(scores, 95)), 3),
            "p100": round(float(np.percentile(scores, 100)), 3),
        },
    }
    write_json(result_dir / "detection_result.json", row)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect missing, dim, and extra LEDs using a led_slot_array recipe.")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="Optional generated test manifest with expected defects.")
    parser.add_argument("--missing-ratio", type=float, default=0.35)
    parser.add_argument("--dim-ratio", type=float, default=0.72)
    parser.add_argument("--extra-radius", type=float, default=4.5)
    parser.add_argument("--score-radius", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    reference_image = load_image(Path(recipe["source_image"]))
    ref_scores = slot_scores(reference_image, recipe["slots"], int(args.score_radius))
    expected_by_name = load_expected(args.manifest)

    paths = list(args.image)
    if args.input_dir:
        paths.extend(image_paths_from_dir(args.input_dir))
    if not paths:
        raise SystemExit("No input images. Use --image or --input-dir.")

    rows = [
        detect_one(
            image_path=path,
            recipe=recipe,
            ref_scores=ref_scores,
            output_dir=args.output_dir,
            missing_ratio=float(args.missing_ratio),
            dim_ratio=float(args.dim_ratio),
            extra_radius=float(args.extra_radius),
            score_radius=int(args.score_radius),
            expected_by_name=expected_by_name,
        )
        for path in paths
    ]

    summary = {
        "type": "led_slot_array_detection_summary",
        "recipe": str(args.recipe),
        "reference_image": recipe.get("source_image", ""),
        "image_count": len(rows),
        "thresholds": {
            "missing_ratio": float(args.missing_ratio),
            "dim_ratio": float(args.dim_ratio),
            "extra_radius": float(args.extra_radius),
            "score_radius": int(args.score_radius),
        },
        "results": rows,
    }
    write_json(args.output_dir / "detection_summary.json", summary)
    write_csv(
        args.output_dir / "detection_summary.csv",
        [
            {
                "image": row["image"],
                "overall_ok": int(bool(row["overall_ok"])),
                "missing_count": row["missing_count"],
                "dim_count": row["dim_count"],
                "extra_count": row["extra_count"],
                "detected_point_count": row["detected_point_count"],
                "overlay": row["overlay"],
            }
            for row in rows
        ],
        ["image", "overall_ok", "missing_count", "dim_count", "extra_count", "detected_point_count", "overlay"],
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
