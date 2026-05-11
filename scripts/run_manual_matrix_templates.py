from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import marquee_inspection_v2 as grid_probe


def load_images(input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(input_dir.glob(pattern))
    return sorted(paths)


def load_library(path: Path) -> tuple[list[grid_probe.Slot], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    slots = [
        grid_probe.Slot(
            id=int(slot["id"]),
            x=float(slot["x"]),
            y=float(slot["y"]),
            w=float(slot["w"]),
            h=float(slot["h"]),
            threshold=float(slot.get("threshold", 0.25)),
        )
        for slot in payload["slots"]
    ]
    templates = payload["templates"]
    for template in templates:
        template["mask_array"] = np.asarray(template["expected_on_mask"], dtype=bool)
        score_vector = template.get("score_vector")
        if not score_vector:
            score_vector = template.get("mean_score_vector")
        template["score_array"] = np.asarray(score_vector or [], dtype=np.float32)
    return slots, templates


def infer_frame_mask(scores: np.ndarray, all_on_floor: float) -> tuple[np.ndarray, float, str]:
    if float(np.min(scores)) >= all_on_floor:
        return np.ones(scores.shape, dtype=bool), all_on_floor, "all_on_floor"
    values = np.clip(scores * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = float(threshold / 255.0)
    return scores >= threshold_value, threshold_value, "otsu"


def compare_templates(scores: np.ndarray, templates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], float]:
    best_template = templates[0]
    best_analysis: dict[str, Any] = {}
    best_composite_score = math.inf

    # Normalize input scores for comparison
    input_norm = scores / max(float(np.percentile(scores, 95)), 1e-6)
    all_tpl_scores = {}

    for template in templates:
        expected_mask = template["mask_array"]
        template_scores = template["score_array"]
        
        if template_scores.shape != scores.shape:
            continue
            
        # 1. Base MSE (Normalized)
        tpl_norm = template_scores / max(float(np.percentile(template_scores, 95)), 1e-6)
        mse = float(np.mean((input_norm - tpl_norm) ** 2))

        # 2. Quality Analysis for this specific template candidate
        with np.errstate(divide='ignore', invalid='ignore'):
            health = scores / np.maximum(template_scores, 1e-6)
        
        # Bad Point: Expected ON but Health < 28% or absolute score low
        bad_mask = np.logical_and(expected_mask, np.logical_or(health < 0.28, scores < 0.12))
        # Dim Point: Expected ON but Health < 68%
        dim_mask = np.logical_and(expected_mask, np.logical_and(health < 0.68, ~bad_mask))
        # Extra Point (Short/Crosstalk): Expected OFF but score > 0.30
        extra_mask = np.logical_and(~expected_mask, scores > 0.30)

        # 3. COMPOSITE SCORE: Penalize mismatched morphology heavily
        bad_count = int(bad_mask.sum())
        dim_count = int(dim_mask.sum())
        extra_count = int(extra_mask.sum())
        
        # Penalties:
        # - Bad points: 0.5 (Extremely high, indicates wrong template)
        # - Dim points: 0.05 (Significant)
        # - Extra points: 0.02 (Increased from 0.002 to reduce 'All ON' bias)
        composite = mse + (bad_count * 0.5) + (dim_count * 0.05) + (extra_count * 0.02)

        if composite < best_composite_score:
            best_composite_score = composite
            best_template = template
            best_analysis = {
                "bad_mask": bad_mask,
                "dim_mask": dim_mask,
                "extra_mask": extra_mask,
                "health": health,
                "mse": mse,
                "composite": composite,
                "all_scores": all_tpl_scores # Track all candidates
            }

        all_tpl_scores[template["name"]] = composite

    return best_template, best_analysis, best_composite_score

def draw_result_overlay(
    image: np.ndarray,
    slots: list[grid_probe.Slot],
    analysis: dict[str, Any],
    expected_mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    vis = image.copy()
    bad_mask = analysis["bad_mask"].tolist()
    dim_mask = analysis["dim_mask"].tolist()
    extra_mask = analysis["extra_mask"].tolist()
    expected_list = expected_mask.tolist()

    for i, slot in enumerate(slots):
        cx, cy = int(round(slot.x)), int(round(slot.y))
        
        if bad_mask[i]:
            # BAD (DEAD): Bright RED
            cv2.circle(vis, (cx, cy), 7, (0, 0, 0), -1)
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
        elif dim_mask[i]:
            # DIM: YELLOW
            cv2.circle(vis, (cx, cy), 6, (0, 0, 0), -1)
            cv2.circle(vis, (cx, cy), 5, (0, 255, 255), -1)
        elif extra_mask[i]:
            # EXTRA (Short/Crosstalk): BLUE hollow
            cv2.circle(vis, (cx, cy), 3, (255, 0, 0), 1)
        elif expected_list[i]:
            # NORMAL ON: Small GREEN
            cv2.circle(vis, (cx, cy), 3, (0, 210, 0), -1)
        else:
            # NORMAL OFF: Tiny Gray
            cv2.circle(vis, (cx, cy), 1, (60, 60, 60), -1)

    # UI Overlay
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, 30), (0, 0, 0), -1)
    status_text = f"BAD:{int(analysis['bad_mask'].sum())} DIM:{int(analysis['dim_mask'].sum())} EXTRA:{int(analysis['extra_mask'].sum())}"
    cv2.putText(vis, f"{title} | {status_text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


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
        canvas = np.full((rows * h, cols * w, 3), 32, dtype=np.uint8)
        for index, thumb in enumerate(page):
            row = index // cols
            col = index % cols
            canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = thumb
        page_no = start // per_page + 1
        cv2.imwrite(str(output_dir / f"manual_detect_sheet_{page_no:02d}.jpg"), canvas)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_detection(
    template_library: Path,
    input_dir: Path,
    output_dir: Path,
    all_on_floor: float,
) -> dict[str, Any]:
    slots, templates = load_library(template_library)
    image_paths = load_images(input_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    overlay_dir = output_dir / "overlays"
    sheet_dir = output_dir / "contact_sheets"
    rows: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        scores = grid_probe.compute_slot_scores([image], slots)[0]
        best_template, analysis, composite_score = compare_templates(scores, templates)
        
        bad_count = int(analysis["bad_mask"].sum())
        dim_count = int(analysis["dim_mask"].sum())
        extra_count = int(analysis["extra_mask"].sum())
        
        overlay_path = overlay_dir / image_path.name
        title = f"{image_path.stem} | {best_template['name']}"
        
        draw_result_overlay(
            image,
            slots,
            analysis,
            best_template["mask_array"],
            overlay_path,
            title,
        )
        overlay_paths.append(overlay_path)
        all_scores = dict(sorted(analysis.get("all_scores", {}).items(), key=lambda item: item[1]))
        runner_up = next(((name, score) for name, score in all_scores.items() if name != best_template["name"]), ("", 0.0))
        rows.append(
            {
                "image": str(image_path),
                "overlay": str(overlay_path),
                "matched_template": best_template["name"],
                "missing": bad_count, 
                "extra": extra_count,
                "mismatch": bad_count + extra_count,
                "bad_count": bad_count,
                "dim_count": dim_count,
                "composite_score": round(float(composite_score), 5),
                "runner_up_template": runner_up[0],
                "runner_up_score": round(float(runner_up[1]), 5),
                "template_scores": ";".join(f"{name}:{score:.5f}" for name, score in all_scores.items()),
                "detected_on_count": int((scores > 0.3).sum()),
                "expected_on_count": int(best_template["mask_array"].sum()),
                "score_min": round(float(np.min(scores)), 4),
                "score_median": round(float(np.median(scores)), 4),
                "score_max": round(float(np.max(scores)), 4),
            }
        )

    make_contact_sheets(overlay_paths, sheet_dir)
    fieldnames = [
        "image",
        "overlay",
        "matched_template",
        "missing",
        "extra",
        "mismatch",
        "bad_count",
        "dim_count",
        "composite_score",
        "runner_up_template",
        "runner_up_score",
        "template_scores",
        "detected_on_count",
        "expected_on_count",
        "score_min",
        "score_median",
        "score_max",
    ]
    write_csv(output_dir / "manual_detection_results.csv", rows, fieldnames)

    by_template: dict[str, int] = {}
    for row in rows:
        key = str(row["matched_template"])
        by_template[key] = by_template.get(key, 0) + 1
    summary = {
        "template_library": str(template_library),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "images": len(rows),
        "matched_counts": by_template,
        "results_csv": str(output_dir / "manual_detection_results.csv"),
        "contact_sheets": str(sheet_dir),
        "overlays": str(overlay_dir),
    }
    (output_dir / "manual_detection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed manual matrix templates on aligned panel images.")
    parser.add_argument("--template-library", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--all-on-floor", type=float, default=0.55)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_detection(
        template_library=args.template_library,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        all_on_floor=float(args.all_on_floor),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
