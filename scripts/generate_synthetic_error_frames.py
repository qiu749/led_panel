from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_template_library(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    templates: dict[str, dict[str, Any]] = {}
    for item in payload["templates"]:
        points = np.asarray([[float(p["x"]), float(p["y"])] for p in item["points"]], dtype=np.float32)
        templates[item["name"]] = {
            "name": item["name"],
            "point_count": int(item["point_count"]),
            "points": points,
            "center": points.mean(axis=0) if len(points) else np.zeros((2,), dtype=np.float32),
        }
    return templates


def read_good_rows(results_csv: Path, per_template: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8-sig", newline="")))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["valid_detection"] != "1":
            continue
        if int(row["missing_count"]) + int(row["extra_count"]) != 0:
            continue
        grouped.setdefault(row["matched_template"], []).append(row)
    picked: list[dict[str, str]] = []
    for template_name in sorted(grouped):
        picked.extend(grouped[template_name][:per_template])
    return picked


def estimate_lit_color(image: np.ndarray, points: np.ndarray) -> tuple[int, int, int]:
    samples: list[np.ndarray] = []
    for x, y in points[: min(len(points), 32)]:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        x1 = max(0, cx - 2)
        y1 = max(0, cy - 2)
        x2 = min(image.shape[1], cx + 3)
        y2 = min(image.shape[0], cy + 3)
        patch = image[y1:y2, x1:x2]
        if patch.size:
            samples.append(patch.reshape(-1, 3))
    if not samples:
        return (255, 255, 255)
    merged = np.concatenate(samples, axis=0)
    color = np.percentile(merged, 90, axis=0)
    return tuple(int(np.clip(v, 0, 255)) for v in color.tolist())


def distance_to_points(candidate: np.ndarray, points: np.ndarray) -> float:
    if len(points) == 0:
        return 9999.0
    return float(np.min(np.linalg.norm(points - candidate[None, :], axis=1)))


def choose_center_missing(points: np.ndarray, count: int) -> np.ndarray:
    center = points.mean(axis=0)
    order = np.argsort(np.linalg.norm(points - center[None, :], axis=1))
    return points[order[:count]]


def choose_edge_missing(points: np.ndarray, count: int) -> np.ndarray:
    left_x = float(np.min(points[:, 0]))
    band = points[:, 0] <= left_x + 12.0
    edge_points = points[band]
    if len(edge_points) == 0:
        edge_points = points[np.argsort(points[:, 0])[:count]]
    order = np.argsort(edge_points[:, 1])
    return edge_points[order[:count]]


def choose_extra_points(current_points: np.ndarray, all_templates: dict[str, dict[str, Any]], count: int) -> np.ndarray:
    center = current_points.mean(axis=0)
    candidates: list[np.ndarray] = []
    for template in all_templates.values():
        for point in template["points"]:
            if distance_to_points(point, current_points) >= 9.0:
                candidates.append(point)
    if not candidates:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.asarray(candidates, dtype=np.float32)
    order = np.argsort(np.linalg.norm(arr - center[None, :], axis=1))
    picked: list[np.ndarray] = []
    for idx in order:
        point = arr[idx]
        if picked and np.min(np.linalg.norm(np.asarray(picked) - point[None, :], axis=1)) < 12.0:
            continue
        picked.append(point)
        if len(picked) >= count:
            break
    return np.asarray(picked, dtype=np.float32)


def estimate_pitch(points: np.ndarray) -> float:
    if len(points) < 4:
        return 18.0
    xs = np.unique(np.rint(points[:, 0]).astype(np.int32))
    diffs = np.diff(np.sort(xs))
    diffs = diffs[(diffs >= 6) & (diffs <= 60)]
    if len(diffs) == 0:
        return 18.0
    return float(np.median(diffs))


def choose_gap_points(current_points: np.ndarray, count: int) -> np.ndarray:
    pitch = estimate_pitch(current_points)
    center = current_points.mean(axis=0)
    candidates: list[np.ndarray] = []
    for point in current_points:
        for dx, dy in (
            (pitch * 0.5, 0.0),
            (-pitch * 0.5, 0.0),
            (pitch * 0.25, pitch * 0.5),
            (-pitch * 0.25, pitch * 0.5),
            (pitch * 0.25, -pitch * 0.5),
            (-pitch * 0.25, -pitch * 0.5),
        ):
            candidate = np.asarray([point[0] + dx, point[1] + dy], dtype=np.float32)
            if candidate[0] < 12 or candidate[0] > 708 or candidate[1] < 12 or candidate[1] > 468:
                continue
            if distance_to_points(candidate, current_points) < pitch * 0.38:
                continue
            candidates.append(candidate)
    if not candidates:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.asarray(candidates, dtype=np.float32)
    order = np.argsort(np.linalg.norm(arr - center[None, :], axis=1))
    picked: list[np.ndarray] = []
    for idx in order:
        point = arr[idx]
        if picked and np.min(np.linalg.norm(np.asarray(picked) - point[None, :], axis=1)) < pitch * 0.75:
            continue
        picked.append(point)
        if len(picked) >= count:
            break
    return np.asarray(picked, dtype=np.float32)


def erase_leds(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    out = image.copy()
    for x, y in points:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        cv2.circle(out, (cx, cy), 11, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 7, (4, 4, 4), -1, cv2.LINE_AA)
    return out


def add_leds(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    glow_radius: int = 10,
    core_radius: int = 5,
    blur_sigma: float = 0.6,
    glow_scale: float = 0.45,
) -> np.ndarray:
    out = image.copy()
    glow = tuple(int(round(c * glow_scale)) for c in color)
    core = tuple(int(c) for c in color)
    for x, y in points:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        cv2.circle(out, (cx, cy), glow_radius, glow, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), core_radius, core, -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), max(1, core_radius - 2), (255, 255, 255), -1, cv2.LINE_AA)
    if blur_sigma > 0:
        return cv2.GaussianBlur(out, (0, 0), blur_sigma)
    return out


def build_variants(
    row: dict[str, str],
    templates: dict[str, dict[str, Any]],
    input_root: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    image_path = Path(row["image"])
    if not image_path.is_absolute():
        image_path = input_root / image_path
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    template = templates[row["matched_template"]]
    lit_color = estimate_lit_color(image, template["points"])

    variants: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    miss_points = choose_center_missing(template["points"], 3)
    variants.append(
        (
            "missing3_center",
            erase_leds(image, miss_points),
            {"missing_points": [[round(float(x), 2), round(float(y), 2)] for x, y in miss_points]},
        )
    )

    edge_points = choose_edge_missing(template["points"], 10 if template["point_count"] < 800 else 18)
    variants.append(
        (
            "missing_edge_band",
            erase_leds(image, edge_points),
            {"missing_points": [[round(float(x), 2), round(float(y), 2)] for x, y in edge_points]},
        )
    )

    extra_points = choose_extra_points(template["points"], templates, 3)
    if len(extra_points):
        variants.append(
            (
                "extra3_center",
                add_leds(image, extra_points, lit_color, glow_radius=8, core_radius=4, blur_sigma=0.2, glow_scale=0.35),
                {"extra_points": [[round(float(x), 2), round(float(y), 2)] for x, y in extra_points]},
            )
        )

    strong_extra_points = choose_extra_points(template["points"], templates, 5)
    if len(strong_extra_points):
        variants.append(
            (
                "extra5_strong",
                add_leds(image, strong_extra_points, lit_color, glow_radius=7, core_radius=3, blur_sigma=0.0, glow_scale=0.28),
                {"extra_points": [[round(float(x), 2), round(float(y), 2)] for x, y in strong_extra_points]},
            )
        )

    gap_points = choose_gap_points(template["points"], 4)
    if len(gap_points):
        variants.append(
            (
                "extra4_gap",
                add_leds(image, gap_points, lit_color, glow_radius=6, core_radius=3, blur_sigma=0.0, glow_scale=0.22),
                {"extra_points": [[round(float(x), 2), round(float(y), 2)] for x, y in gap_points]},
            )
        )

    output_rows: list[dict[str, Any]] = []
    for variant_name, variant_image, meta in variants:
        out_name = f"{image_path.stem}__{variant_name}.jpg"
        out_path = output_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), variant_image)
        output_rows.append(
            {
                "source_image": str(image_path),
                "matched_template": row["matched_template"],
                "variant": variant_name,
                "output_image": str(out_path),
                "expected_ng": 1,
                "meta": json.dumps(meta, ensure_ascii=False),
            }
        )
    return output_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic NG frames from good aligned panel images.")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--template-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-template", type=int, default=3)
    parser.add_argument("--input-root", type=Path, default=Path(r"D:\led-visual-inspection"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    templates = load_template_library(args.template_library)
    good_rows = read_good_rows(args.results_csv, args.per_template)
    generated_rows: list[dict[str, Any]] = []
    image_dir = args.output_dir / "images"
    for row in good_rows:
        generated_rows.extend(build_variants(row, templates, args.input_root, image_dir))
    write_csv(
        args.output_dir / "synthetic_error_manifest.csv",
        generated_rows,
        ["source_image", "matched_template", "variant", "output_image", "expected_ng", "meta"],
    )
    write_json(
        args.output_dir / "synthetic_error_manifest.json",
        {
            "template_library": str(args.template_library),
            "results_csv": str(args.results_csv),
            "per_template": args.per_template,
            "generated_count": len(generated_rows),
            "image_dir": str(image_dir),
        },
    )
    print(json.dumps({"generated_count": len(generated_rows), "image_dir": str(image_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
