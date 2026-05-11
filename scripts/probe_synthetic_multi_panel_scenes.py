from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import matrix_12state_probe as matrix_probe
import run_direct_point_templates as direct_run
import run_multi_panel_direct_point_templates as multi_run
from build_direct_point_templates import detect_points


ROOT = Path(r"D:\led-visual-inspection")
TEMPLATE_LIBRARY = ROOT / "output" / "clustered_templates" / "Video_20260428140010756_750frames_split_v1" / "four_cluster_templates.json"
SOURCE_IMAGES = {
    "shape_a": ROOT / "output" / "template_picker" / "Video_20260428140010756_750frames" / "aligned_panels_750" / "0004_t0003.21s_f000024.jpg",
    "shape_b": ROOT / "output" / "template_picker" / "Video_20260428140010756_750frames" / "aligned_panels_750" / "0010_t0008.16s_f000061.jpg",
    "shape_c": ROOT / "output" / "template_picker" / "Video_20260428140010756_750frames" / "aligned_panels_750" / "0049_t0039.98s_f000299.jpg",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(direct_run.jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def warp_panel_to_canvas(canvas: np.ndarray, panel: np.ndarray, quad: np.ndarray) -> None:
    h, w = panel.shape[:2]
    src = np.array(
        [[0.0, 0.0], [float(w - 1), 0.0], [float(w - 1), float(h - 1)], [0.0, float(h - 1)]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(panel, matrix, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR)
    mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 2
    canvas[mask] = np.maximum(canvas[mask], warped[mask])


def apply_camera_like_render(image: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.astype(np.float32)

    height, width = out.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    rx = (xx - cx) / max(width * 0.5, 1.0)
    ry = (yy - cy) / max(height * 0.5, 1.0)
    radius = np.sqrt(rx * rx + ry * ry)

    vignette = 1.0 - np.clip(radius, 0.0, 1.25) ** 1.8 * 0.22
    out *= vignette[..., None]

    gamma = float(rng.uniform(0.92, 1.08))
    out = np.clip((out / 255.0) ** gamma * 255.0, 0.0, 255.0)

    sigma = float(rng.uniform(0.55, 1.15))
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)

    noise = rng.normal(0.0, 4.0, size=out.shape).astype(np.float32)
    out += noise

    sharpened = cv2.GaussianBlur(out, (0, 0), 1.0)
    out = cv2.addWeighted(out, 1.12, sharpened, -0.12, 0.0)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def make_scene(scene_items: list[tuple[str, np.ndarray]], output_path: Path) -> np.ndarray:
    canvas = np.zeros((1500, 2400, 3), dtype=np.uint8)
    canvas[:] = (6, 10, 8)
    for label, quad in scene_items:
        panel = cv2.imread(str(SOURCE_IMAGES[label]))
        if panel is None:
            raise FileNotFoundError(SOURCE_IMAGES[label])
        warp_panel_to_canvas(canvas, panel, quad)
    cv2.imwrite(str(output_path), canvas)
    return canvas


def make_realistic_scene(scene_items: list[tuple[str, np.ndarray]], output_path: Path, seed: int) -> np.ndarray:
    base = make_scene(scene_items, output_path)
    realistic = apply_camera_like_render(base, seed)
    cv2.imwrite(str(output_path), realistic)
    return realistic


def detect_scene(
    scene_path: Path,
    template_payload: dict[str, Any],
    templates: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    image = cv2.imread(str(scene_path))
    if image is None:
        raise FileNotFoundError(scene_path)
    quads = matrix_probe.detect_matrix_quads(image, max_quads=4)
    scene_overlay = image.copy()
    results: list[dict[str, Any]] = []
    template_size = tuple(int(v) for v in template_payload["image_size"])
    scale_values = direct_run.build_scale_values(0.01, 0.04)
    for idx, quad in enumerate(quads, start=1):
        size = template_size
        panel = matrix_probe.warp_panel_quad(image, quad, size)
        panel_path = output_dir / f"{scene_path.stem}_panel_{idx:02d}.jpg"
        cv2.imwrite(str(panel_path), panel)

        overlay_path = output_dir / f"{scene_path.stem}_panel_{idx:02d}_overlay.jpg"
        panel_row, overlay = multi_run.run_panel_detection(
            image_path=panel_path,
            templates=templates,
            width=int(template_payload["image_size"][0]),
            height=int(template_payload["image_size"][1]),
            match_radius=5.5,
            align_radius=8,
            scale_search_mode="hybrid",
            scale_values=scale_values,
            hybrid_suspect_error_threshold=60,
            hybrid_suspect_edge_error_threshold=24,
            hybrid_suspect_close_error_threshold=10,
            hybrid_suspect_runner_gap_threshold=0.25,
        )
        cv2.imwrite(str(overlay_path), overlay)

        ordered = matrix_probe.order_corners(quad)
        poly = np.round(ordered).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(scene_overlay, [poly], True, (0, 220, 255), 3, cv2.LINE_AA)
        cx, cy = matrix_probe.quad_center(quad)
        cv2.putText(scene_overlay, f"P{idx}", (int(round(cx)) - 18, int(round(cy))), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)

        results.append(
            {
                "panel_id": idx,
                "quad": ordered.tolist(),
                "matched_template": panel_row["matched_template"],
                "missing_count": int(panel_row["missing_count"]),
                "extra_count": int(panel_row["extra_count"]),
                "raw_missing_count": int(panel_row["raw_missing_count"]),
                "raw_extra_count": int(panel_row["raw_extra_count"]),
                "detected_point_count": int(panel_row["detected_point_count"]),
                "threshold": round(float(panel_row["threshold"]), 3),
                "align_dx": int(panel_row["align_dx"]),
                "align_dy": int(panel_row["align_dy"]),
                "align_scale_x": float(panel_row["align_scale_x"]),
                "retried_with_scale": int(panel_row["retried_with_scale"]),
                "panel_image": str(panel_path),
                "overlay_image": str(overlay_path),
            }
        )

    scene_overlay_path = output_dir / f"{scene_path.stem}_scene_overlay.jpg"
    cv2.imwrite(str(scene_overlay_path), scene_overlay)
    return {
        "scene": str(scene_path),
        "detected_panel_count": len(quads),
        "scene_overlay": str(scene_overlay_path),
        "panels": results,
    }


def main() -> None:
    output_dir = ROOT / "output" / "synthetic_multi_panel_probe"
    scenes_dir = output_dir / "scenes"
    results_dir = output_dir / "results"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    scene_specs = {
        "scene_2panels": [
            ("shape_a", np.array([[120, 140], [860, 120], [850, 610], [110, 630]], dtype=np.float32)),
            ("shape_c", np.array([[1280, 240], [2140, 250], [2120, 790], [1260, 770]], dtype=np.float32)),
        ],
        "scene_3panels": [
            ("shape_b", np.array([[120, 120], [880, 150], [850, 660], [100, 640]], dtype=np.float32)),
            ("shape_a", np.array([[1240, 90], [2240, 120], [2200, 700], [1210, 670]], dtype=np.float32)),
            ("shape_c", np.array([[660, 790], [1700, 760], [1720, 1360], [640, 1380]], dtype=np.float32)),
        ],
        "scene_4panels": [
            ("shape_a", np.array([[90, 100], [920, 120], [890, 620], [80, 640]], dtype=np.float32)),
            ("shape_b", np.array([[1320, 120], [2290, 140], [2260, 660], [1300, 640]], dtype=np.float32)),
            ("shape_c", np.array([[120, 820], [1050, 800], [1040, 1400], [100, 1420]], dtype=np.float32)),
            ("shape_a", np.array([[1290, 820], [2300, 850], [2280, 1410], [1270, 1380]], dtype=np.float32)),
        ],
        "scene_4panels_realistic": [
            ("shape_a", np.array([[90, 100], [920, 120], [890, 620], [80, 640]], dtype=np.float32)),
            ("shape_b", np.array([[1320, 120], [2290, 140], [2260, 660], [1300, 640]], dtype=np.float32)),
            ("shape_c", np.array([[120, 820], [1050, 800], [1040, 1400], [100, 1420]], dtype=np.float32)),
            ("shape_a", np.array([[1290, 820], [2300, 850], [2280, 1410], [1270, 1380]], dtype=np.float32)),
        ],
    }

    payload, templates = direct_run.load_template_library(TEMPLATE_LIBRARY)
    template_size = tuple(int(v) for v in payload["image_size"])
    scene_results: list[dict[str, Any]] = []
    for scene_name, spec in scene_specs.items():
        scene_path = scenes_dir / f"{scene_name}.jpg"
        if scene_name.endswith("_realistic"):
            make_realistic_scene(spec, scene_path, seed=37)
        else:
            make_scene(spec, scene_path)
        result = detect_scene(scene_path, payload, templates, results_dir)
        result["template_size"] = list(template_size)
        scene_results.append(result)

    write_json(output_dir / "synthetic_multi_panel_probe.json", {"scenes": scene_results})
    print(json.dumps({"output_dir": str(output_dir), "scene_count": len(scene_results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
