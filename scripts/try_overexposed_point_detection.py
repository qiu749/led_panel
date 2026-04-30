from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import maximum_filter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Params:
    target_width: int = 512
    min_height: int = 80
    max_height: int = 220
    clahe_clip_limit: float = 2.6
    small_sigma: float = 0.9
    large_sigma: float = 6.5
    threshold_percentile: float = 96.8
    threshold_std_factor: float = 1.05
    min_threshold: float = 12.0
    min_peak_distance_px: int = 8
    center_radius: int = 2
    ring_inner_radius: int = 5
    ring_outer_radius: int = 10
    min_center_contrast: float = 3.5
    min_peak_contrast: float = 9.0
    min_white_core_percentile: float = 55.0


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    buffer.tofile(str(path))


def normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_value = float(arr.min())
    max_value = float(arr.max())
    if max_value <= min_value + 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    normalized = (arr - min_value) * 255.0 / (max_value - min_value)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def add_label(panel: np.ndarray, label: str) -> np.ndarray:
    height, width = panel.shape[:2]
    bar_height = 28
    output = np.full((height + bar_height, width, 3), 246, dtype=np.uint8)
    output[bar_height:, :] = panel
    cv2.putText(
        output,
        label,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 24, 24),
        1,
        cv2.LINE_AA,
    )
    return output


def stack_panels(panels: list[np.ndarray], labels: list[str]) -> np.ndarray:
    labelled = [add_label(panel, label) for panel, label in zip(panels, labels)]
    return np.vstack([np.hstack(labelled[:3]), np.hstack(labelled[3:])])


def disk_mask(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    yy, xx = np.ogrid[:size, :size]
    center = radius
    return ((xx - center) ** 2 + (yy - center) ** 2 <= radius**2)


def ring_mask(inner_radius: int, outer_radius: int) -> np.ndarray:
    size = 2 * outer_radius + 1
    yy, xx = np.ogrid[:size, :size]
    center = outer_radius
    dist2 = (xx - center) ** 2 + (yy - center) ** 2
    return (dist2 >= inner_radius**2) & (dist2 <= outer_radius**2)


def window_values(image: np.ndarray, x: int, y: int, radius: int, mask: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(width, x + radius + 1)
    y1 = min(height, y + radius + 1)
    patch = image[y0:y1, x0:x1]

    mask_x0 = x0 - (x - radius)
    mask_y0 = y0 - (y - radius)
    mask_x1 = mask_x0 + patch.shape[1]
    mask_y1 = mask_y0 + patch.shape[0]
    local_mask = mask[mask_y0:mask_y1, mask_x0:mask_x1]
    return patch[local_mask]


def resize_keep_ratio(image: np.ndarray, params: Params) -> np.ndarray:
    height, width = image.shape[:2]
    target_height = int(round(height * params.target_width / max(width, 1)))
    target_height = max(params.min_height, min(params.max_height, target_height))
    return cv2.resize(image, (params.target_width, target_height), interpolation=cv2.INTER_AREA)


def overexposure_metrics(image: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    b, g, r = cv2.split(image.astype(np.float32))
    return {
        "mean_v": float(v.mean()),
        "p50_v": float(np.percentile(v, 50)),
        "p99_v": float(np.percentile(v, 99)),
        "bright_ratio_v220": float((v >= 220).mean()),
        "mean_r": float(r.mean()),
        "mean_g": float(g.mean()),
        "mean_b": float(b.mean()),
    }


def detect_points(image: np.ndarray, params: Params) -> dict[str, object]:
    resized = resize_keep_ratio(image, params)
    b, g, r = cv2.split(resized.astype(np.float32))
    white_core = np.minimum(np.minimum(r, g), b).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=params.clahe_clip_limit, tileGridSize=(8, 4))
    enhanced = clahe.apply(white_core)

    small = cv2.GaussianBlur(enhanced, (0, 0), params.small_sigma)
    large = cv2.GaussianBlur(enhanced, (0, 0), params.large_sigma)
    dog = cv2.subtract(small, large)
    dog_blur = cv2.GaussianBlur(dog, (3, 3), 0)

    percentile_threshold = float(np.percentile(dog_blur, params.threshold_percentile))
    std_threshold = float(dog_blur.mean() + params.threshold_std_factor * dog_blur.std())
    threshold = max(percentile_threshold, std_threshold, params.min_threshold)
    white_gate = float(np.percentile(white_core, params.min_white_core_percentile))

    binary = ((dog_blur >= threshold) & (white_core >= white_gate)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    max_filtered = maximum_filter(
        dog_blur,
        size=2 * params.min_peak_distance_px + 1,
        mode="nearest",
    )
    peak_mask = (dog_blur == max_filtered) & (dog_blur >= threshold) & (binary > 0)
    ys, xs = np.where(peak_mask)
    candidates = sorted(
        [(float(dog_blur[y, x]), int(x), int(y), int(white_core[y, x])) for y, x in zip(ys, xs)],
        reverse=True,
    )

    center_mask = disk_mask(params.center_radius)
    ring = ring_mask(params.ring_inner_radius, params.ring_outer_radius)
    kept: list[dict[str, float | int]] = []
    rejected_by_contrast = 0

    height, width = white_core.shape[:2]
    for response, x, y, brightness in candidates:
        if (
            x < params.ring_outer_radius
            or y < params.ring_outer_radius
            or x >= width - params.ring_outer_radius
            or y >= height - params.ring_outer_radius
        ):
            continue
        if any(
            (x - int(point["x"])) ** 2 + (y - int(point["y"])) ** 2
            < params.min_peak_distance_px**2
            for point in kept
        ):
            continue

        center_values = window_values(white_core, x, y, params.center_radius, center_mask)
        ring_values = window_values(white_core, x, y, params.ring_outer_radius, ring)
        if center_values.size == 0 or ring_values.size == 0:
            continue

        center_mean = float(center_values.mean())
        center_max = float(center_values.max())
        ring_mean = float(ring_values.mean())
        center_contrast = center_mean - ring_mean
        peak_contrast = center_max - ring_mean

        if (
            center_contrast < params.min_center_contrast
            or peak_contrast < params.min_peak_contrast
        ):
            rejected_by_contrast += 1
            continue

        kept.append(
            {
                "x": x,
                "y": y,
                "response": response,
                "brightness": brightness,
                "center_mean": center_mean,
                "center_max": center_max,
                "ring_mean": ring_mean,
                "center_contrast": center_contrast,
                "peak_contrast": peak_contrast,
            }
        )

    overlay = resized.copy()
    for point in kept:
        x = int(point["x"])
        y = int(point["y"])
        cv2.circle(overlay, (x, y), 5, (0, 255, 80), 1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 1, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        overlay,
        f"points={len(kept)} thr={threshold:.1f}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    contact = stack_panels(
        [
            resized,
            gray_to_bgr(white_core),
            gray_to_bgr(enhanced),
            cv2.applyColorMap(normalize_to_u8(dog_blur), cv2.COLORMAP_TURBO),
            gray_to_bgr(binary),
            overlay,
        ],
        [
            "01 original / resized",
            "02 white_core = min(R,G,B)",
            "03 CLAHE white_core",
            "04 DoG small-spot response",
            "05 threshold + white gate mask",
            f"06 center/ring kept: {len(kept)}",
        ],
    )

    return {
        "resized": resized,
        "white_core": white_core,
        "enhanced": enhanced,
        "dog": dog_blur,
        "binary": binary,
        "overlay": overlay,
        "contact": contact,
        "points": kept,
        "threshold": threshold,
        "percentile_threshold": percentile_threshold,
        "std_threshold": std_threshold,
        "white_gate": white_gate,
        "candidate_count": len(candidates),
        "rejected_by_contrast": rejected_by_contrast,
    }


def params_to_rows(params: Params) -> list[str]:
    return [f"{name}: {getattr(params, name)}" for name in params.__dataclass_fields__]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try free bright point detection for overexposed red ROI crops.")
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\Desktop2\Images\roi_crops"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\Desktop2\Images\roi_crops_free_point_steps_v3_overexposed"))
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images when > 0.")
    parser.add_argument("--pattern", default="", help="Only process files containing this substring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = Params()

    input_dir: Path = args.input_dir
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
            for path in input_dir.iterdir()
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
        result = detect_points(image, params)
        points = result["points"]

        stem = path.stem
        contact_path = steps_dir / f"{stem}_steps.jpg"
        overlay_path = overlay_dir / f"{stem}_points.jpg"
        points_path = points_dir / f"{stem}_points.csv"
        imwrite_unicode(contact_path, result["contact"], quality=86)
        imwrite_unicode(overlay_path, result["overlay"], quality=90)

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
            writer.writerows(points)

        row: dict[str, object] = {
            "filename": path.name,
            "original_width": image.shape[1],
            "original_height": image.shape[0],
            "resized_width": result["resized"].shape[1],
            "resized_height": result["resized"].shape[0],
            "mean_v": f"{metrics['mean_v']:.3f}",
            "p50_v": f"{metrics['p50_v']:.3f}",
            "p99_v": f"{metrics['p99_v']:.3f}",
            "bright_ratio_v220": f"{metrics['bright_ratio_v220']:.6f}",
            "threshold": f"{float(result['threshold']):.3f}",
            "percentile_threshold": f"{float(result['percentile_threshold']):.3f}",
            "std_threshold": f"{float(result['std_threshold']):.3f}",
            "white_gate": f"{float(result['white_gate']):.3f}",
            "candidate_count": result["candidate_count"],
            "rejected_by_contrast": result["rejected_by_contrast"],
            "point_count": len(points),
            "steps_image": str(contact_path),
            "overlay_image": str(overlay_path),
            "points_csv": str(points_path),
            "error": "",
        }
        rows.append(row)
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
        "threshold",
        "percentile_threshold",
        "std_threshold",
        "white_gate",
        "candidate_count",
        "rejected_by_contrast",
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
        "Free point detection v3 for red-overexposed ROI crops",
        f"Input: {input_dir}",
        f"Output: {output_dir}",
        f"Images processed: {len(files)}",
        "",
        "Method:",
        "1. Resize ROI while keeping aspect ratio.",
        "2. Compute white_core = min(R, G, B) to prefer bright white LED cores over broad red glow.",
        "3. Apply CLAHE to recover local contrast.",
        "4. Apply Difference of Gaussian to emphasize compact bright spots.",
        "5. Find local maxima, then keep only peaks whose center is brighter than the surrounding ring.",
        "",
        "Contact sheet panels:",
        "01 original/resized, 02 white_core, 03 CLAHE white_core, 04 DoG response, 05 threshold mask, 06 kept peaks overlay.",
        "",
        "Parameters:",
        *params_to_rows(params),
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
