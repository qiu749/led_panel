from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import marquee_inspection_v2 as grid_probe


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
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("template name is empty")
    return name, Path(path.strip())


def load_slots(geometry_template: Path) -> tuple[list[grid_probe.Slot], dict[str, Any]]:
    payload = json.loads(geometry_template.read_text(encoding="utf-8"))
    slots = [
        grid_probe.Slot(
            id=int(slot["id"]),
            x=float(slot["x"]),
            y=float(slot["y"]),
            w=float(slot["w"]),
            h=float(slot["h"]),
            threshold=float(slot.get("threshold", 0.25)),
            peak=float(slot.get("peak", 0.0)),
            duty=float(slot.get("duty", 0.0)),
        )
        for slot in payload["slots"]
    ]
    return slots, payload


def infer_mask(scores: np.ndarray, all_on_floor: float) -> tuple[np.ndarray, float, str]:
    if float(np.min(scores)) >= all_on_floor:
        return np.ones(scores.shape, dtype=bool), all_on_floor, "all_on_floor"
    values = np.clip(scores * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = float(threshold / 255.0)
    return scores >= threshold_value, threshold_value, "otsu"


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scale = max(float(np.percentile(scores, 95)), 1e-6)
    return np.clip(scores / scale, 0.0, 1.2).astype(np.float32)


def draw_template_overlay(
    image: np.ndarray,
    slots: list[grid_probe.Slot],
    mask: np.ndarray,
    output_path: Path,
) -> None:
    vis = image.copy()
    for slot, is_on in zip(slots, mask.tolist()):
        cx = int(round(slot.x))
        cy = int(round(slot.y))
        color = (0, 255, 0) if is_on else (80, 80, 80)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 0), 2)
        cv2.circle(vis, (cx, cy), 4, color, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def draw_template_intensity_map(
    slots: list[grid_probe.Slot],
    normalized_scores: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    width = int(max(slot.x + slot.w for slot in slots) + 24)
    height = int(max(slot.y + slot.h for slot in slots) + 40)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for slot, value in zip(slots, normalized_scores.tolist()):
        cx = int(round(slot.x))
        cy = int(round(slot.y)) + 18
        intensity = max(0, min(255, int(round(value * 255))))
        core = (0, int(intensity * 0.75), intensity)
        ring = (0, int(intensity * 0.35), max(20, int(intensity * 0.55)))
        cv2.circle(canvas, (cx, cy), 7, ring, 1)
        cv2.circle(canvas, (cx, cy), 4, core, -1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, 32), (0, 0, 0), -1)
    cv2.putText(canvas, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def build_templates(
    geometry_template: Path,
    template_items: list[tuple[str, Path]],
    output_dir: Path,
    all_on_floor: float,
) -> dict[str, Any]:
    slots, geometry_payload = load_slots(geometry_template)
    templates: list[dict[str, Any]] = []
    overlay_dir = output_dir / "overlays"
    intensity_dir = output_dir / "intensity_maps"

    for name, image_path in template_items:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        scores = grid_probe.compute_slot_scores([image], slots)[0]
        normalized_scores = normalize_scores(scores)
        mask, threshold, threshold_method = infer_mask(scores, all_on_floor)
        on_slot_ids = [slot.id for slot, is_on in zip(slots, mask.tolist()) if is_on]
        overlay_path = overlay_dir / f"{name}_overlay.jpg"
        intensity_path = intensity_dir / f"{name}_intensity.jpg"
        draw_template_overlay(image, slots, mask, overlay_path)
        draw_template_intensity_map(
            slots,
            normalized_scores,
            intensity_path,
            f"{name} p95norm",
        )
        templates.append(
            {
                "name": name,
                "source_image": str(image_path),
                "overlay_image": str(overlay_path),
                "intensity_map": str(intensity_path),
                "threshold_method": threshold_method,
                "threshold": round(float(threshold), 4),
                "expected_on_count": len(on_slot_ids),
                "expected_on_slot_ids": on_slot_ids,
                "expected_on_mask": mask.astype(np.uint8).tolist(),
                "score_vector": [round(float(value), 5) for value in scores.tolist()],
                "normalized_score_vector": [round(float(value), 5) for value in normalized_scores.tolist()],
                "score_stats": {
                    "min": round(float(np.min(scores)), 4),
                    "median": round(float(np.median(scores)), 4),
                    "p90": round(float(np.percentile(scores, 90)), 4),
                    "max": round(float(np.max(scores)), 4),
                },
            }
        )

    return {
        "type": "manual_matrix_template_library",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "geometry_template": str(geometry_template),
        "source_frame_size": geometry_payload.get("source_frame_size"),
        "aligned_panel_size": geometry_payload.get("aligned_panel_size"),
        "grid_info": geometry_payload.get("grid_info"),
        "slots": geometry_payload.get("slots", []),
        "templates": templates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed templates from manually selected aligned panel images.")
    parser.add_argument("--geometry-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template", action="append", type=parse_template_arg, required=True, help="name=path")
    parser.add_argument("--all-on-floor", type=float, default=0.55)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_templates(
        geometry_template=args.geometry_template,
        template_items=args.template,
        output_dir=args.output_dir,
        all_on_floor=float(args.all_on_floor),
    )
    output_path = args.output_dir / "manual_matrix_templates.json"
    write_json(output_path, payload)
    print(json.dumps({"template": str(output_path), "templates": len(payload["templates"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
