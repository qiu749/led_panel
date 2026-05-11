from __future__ import annotations

import argparse
import csv
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        )
        for slot in payload["slots"]
    ]
    return slots, payload


def image_paths(input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(input_dir.glob(pattern))
    return sorted(paths)


def filter_singleton_source_clusters(paths: list[Path], assignments_csv: Path, min_source_cluster_size: int) -> list[Path]:
    if not assignments_csv.exists():
        raise FileNotFoundError(f"Assignments CSV not found: {assignments_csv}")
    with assignments_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    counts: dict[str, int] = {}
    image_to_cluster: dict[str, str] = {}
    for row in rows:
        image_key = str(Path(row["image"]).resolve()).lower()
        cluster = str(row["cluster"])
        image_to_cluster[image_key] = cluster
        counts[cluster] = counts.get(cluster, 0) + 1

    kept: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        cluster = image_to_cluster.get(key)
        if cluster is None:
            # Assignment CSV may contain relative paths from the same working directory.
            cluster = image_to_cluster.get(str(path).lower())
        if cluster is None:
            continue
        if counts.get(cluster, 0) >= min_source_cluster_size:
            kept.append(path)
    return kept


def top_pixel_mean(roi: np.ndarray) -> tuple[float, float, float]:
    if roi.size == 0:
        return 0.0, 0.0, 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    threshold = max(5.0, float(np.percentile(gray, 85)))
    mask = gray >= threshold
    if int(mask.sum()) < 3:
        mean = roi.reshape(-1, 3).mean(axis=0)
    else:
        mean = roi[mask].mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])


def compute_slot_colors(image: np.ndarray, slots: list[grid_probe.Slot]) -> np.ndarray:
    colors = np.zeros((len(slots), 3), dtype=np.float32)
    for index, slot in enumerate(slots):
        half_w = max(3, int(round(slot.w * 0.42)))
        half_h = max(3, int(round(slot.h * 0.42)))
        cx = int(round(slot.x))
        cy = int(round(slot.y))
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(image.shape[1], cx + half_w + 1)
        y2 = min(image.shape[0], cy + half_h + 1)
        colors[index] = top_pixel_mean(image[y1:y2, x1:x2])
    return colors


def dominant_color_name(colors: np.ndarray, active_mask: np.ndarray) -> str:
    if int(active_mask.sum()) == 0:
        return "dark"
    b, g, r = colors[active_mask].mean(axis=0)
    max_channel = max(float(b), float(g), float(r), 1e-6)
    if max_channel < 25:
        return "dark"
    if r >= g * 1.12 and r >= b * 1.12:
        return "red_orange"
    if g >= r * 1.08 and g >= b * 1.08:
        return "green"
    if b >= r * 1.08 and b >= g * 1.08:
        return "blue"
    return "mixed"


def infer_mask(scores: np.ndarray, all_on_floor: float) -> tuple[np.ndarray, float, str]:
    if float(np.min(scores)) >= all_on_floor:
        return np.ones(scores.shape, dtype=bool), all_on_floor, "all_on_floor"
    values = np.clip(scores * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1)
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = float(threshold / 255.0)
    return scores >= threshold_value, threshold_value, "otsu"


def standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


def pca(features: np.ndarray, dims: int) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(dims, vt.shape[0])]
    return centered @ components.T


def kmeans(features: np.ndarray, k: int, seed: int = 7, max_iter: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    centers = np.empty((k, features.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = features[first]
    distances = np.sum((features - centers[0]) ** 2, axis=1)
    for center_index in range(1, k):
        probs = distances / max(float(distances.sum()), 1e-9)
        chosen = int(rng.choice(n, p=probs))
        centers[center_index] = features[chosen]
        distances = np.minimum(distances, np.sum((features - centers[center_index]) ** 2, axis=1))

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        dist = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist, axis=1).astype(np.int32)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for center_index in range(k):
            members = features[labels == center_index]
            if len(members):
                centers[center_index] = members.mean(axis=0)
    return labels, centers


def silhouette_score(features: np.ndarray, labels: np.ndarray) -> float:
    unique = sorted(set(labels.tolist()))
    if len(unique) <= 1:
        return -1.0
    dist = np.sqrt(np.maximum(np.sum((features[:, None, :] - features[None, :, :]) ** 2, axis=2), 0.0))
    scores: list[float] = []
    for index, label in enumerate(labels):
        same = labels == label
        other = labels != label
        if int(same.sum()) <= 1 or int(other.sum()) == 0:
            continue
        a = float(dist[index, same].sum() / max(int(same.sum()) - 1, 1))
        b = min(float(dist[index, labels == other_label].mean()) for other_label in unique if other_label != int(label))
        scores.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(scores)) if scores else -1.0


def choose_k(features: np.ndarray, min_k: int, max_k: int) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    best_k = min_k
    best_score = -1.0
    for k in range(min_k, max_k + 1):
        labels, _ = kmeans(features, k)
        counts = np.bincount(labels, minlength=k)
        if int(counts.min()) < 2:
            continue
        score = silhouette_score(features, labels)
        scores[k] = score
        if score > best_score:
            best_k = k
            best_score = score
    return best_k, scores


def medoid_index(features: np.ndarray, indices: np.ndarray) -> int:
    cluster = features[indices]
    center = cluster.mean(axis=0)
    distances = np.sum((cluster - center[None, :]) ** 2, axis=1)
    return int(indices[int(np.argmin(distances))])


def draw_overlay(
    image: np.ndarray,
    slots: list[grid_probe.Slot],
    mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    vis = image.copy()
    for slot, is_on in zip(slots, mask.tolist()):
        cx = int(round(slot.x))
        cy = int(round(slot.y))
        color = (0, 220, 0) if is_on else (80, 80, 80)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 0), 2)
        cv2.circle(vis, (cx, cy), 4, color, 1)
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, 25), (0, 0, 0), -1)
    cv2.putText(vis, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def make_contact_sheet(items: list[tuple[Path, str]], output_path: Path, cols: int = 4, thumb_size: tuple[int, int] = (240, 160)) -> None:
    if not items:
        return
    rows = int(np.ceil(len(items) / cols))
    tw, th = thumb_size
    canvas = np.full((rows * th, cols * tw, 3), 30, dtype=np.uint8)
    for index, (path, label) in enumerate(items):
        image = cv2.imread(str(path))
        if image is None:
            continue
        thumb = cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (tw - 1, 22), (0, 0, 0), -1)
        cv2.putText(thumb, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 255), 1, cv2.LINE_AA)
        row = index // cols
        col = index % cols
        canvas[row * th : (row + 1) * th, col * tw : (col + 1) * tw] = thumb
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def cluster_templates(
    geometry_template: Path,
    input_dir: Path,
    output_dir: Path,
    min_k: int,
    max_k: int,
    fixed_k: int | None,
    all_on_floor: float,
    exclude_singletons_from: Path | None,
    min_source_cluster_size: int,
) -> dict[str, Any]:
    slots, geometry_payload = load_slots(geometry_template)
    paths = image_paths(input_dir)
    if exclude_singletons_from is not None:
        paths = filter_singleton_source_clusters(paths, exclude_singletons_from, min_source_cluster_size)
    if not paths:
        raise RuntimeError(f"No images found in {input_dir}")

    images: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    color_rows: list[np.ndarray] = []
    threshold_rows: list[tuple[float, str]] = []
    color_names: list[str] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        scores = grid_probe.compute_slot_scores([image], slots)[0]
        colors = compute_slot_colors(image, slots)
        mask, threshold, threshold_method = infer_mask(scores, all_on_floor)
        images.append(image)
        score_rows.append(scores.astype(np.float32))
        mask_rows.append(mask)
        color_rows.append(colors.astype(np.float32))
        threshold_rows.append((threshold, threshold_method))
        color_names.append(dominant_color_name(colors, mask))

    scores_arr = np.stack(score_rows, axis=0)
    masks_arr = np.stack(mask_rows, axis=0)
    active_counts = masks_arr.sum(axis=1).astype(np.float32)
    mean_colors = np.stack([colors[mask].mean(axis=0) if int(mask.sum()) else np.zeros(3) for colors, mask in zip(color_rows, mask_rows)], axis=0)
    color_sum = np.maximum(mean_colors.sum(axis=1, keepdims=True), 1.0)
    chroma = mean_colors / color_sum

    brightness_norm = scores_arr / np.maximum(np.percentile(scores_arr, 95, axis=1, keepdims=True), 1e-6)
    feature_source = np.concatenate(
        [
            brightness_norm * 2.5,
            masks_arr.astype(np.float32) * 0.6,
            (active_counts[:, None] / max(len(slots), 1)) * 1.2,
            chroma * 3.0,
        ],
        axis=1,
    )
    reduced = pca(standardize(feature_source), dims=min(18, len(paths) - 1))

    if fixed_k is None:
        k, k_scores = choose_k(reduced, min_k=min_k, max_k=max_k)
    else:
        k = fixed_k
        k_scores = {}
    labels, centers = kmeans(reduced, k)

    # Stable cluster ordering by representative frame time/name.
    medoids = [medoid_index(reduced, np.where(labels == cluster_id)[0]) for cluster_id in range(k)]
    order = sorted(range(k), key=lambda cluster_id: paths[medoids[cluster_id]].name)
    remap = {old: new + 1 for new, old in enumerate(order)}
    stable_labels = np.asarray([remap[int(label)] for label in labels], dtype=np.int32)

    overlay_dir = output_dir / "cluster_overlays"
    clusters_dir = output_dir / "clusters"
    rows: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []

    for cluster_no in sorted(set(stable_labels.tolist())):
        indices = np.where(stable_labels == cluster_no)[0]
        medoid = medoid_index(reduced, indices)
        mean_mask = masks_arr[indices].mean(axis=0) >= 0.5
        representative_mask = masks_arr[medoid]
        template_mask = mean_mask if int(mean_mask.sum()) else representative_mask
        color_counter: dict[str, int] = {}
        for index in indices:
            color_counter[color_names[index]] = color_counter.get(color_names[index], 0) + 1
        dominant_color = max(color_counter.items(), key=lambda item: item[1])[0]
        template_name = f"cluster_{cluster_no:02d}_{dominant_color}"
        representative_path = paths[medoid]
        overlay_path = overlay_dir / f"{template_name}_template_overlay.jpg"
        draw_overlay(images[medoid], slots, template_mask, overlay_path, f"{template_name} n={len(indices)} rep={representative_path.stem}")
        member_items: list[tuple[Path, str]] = []
        for index in indices:
            member_items.append((paths[index], f"{paths[index].stem[:16]} {color_names[index]} on={int(active_counts[index])}"))
            rows.append(
                {
                    "image": str(paths[index]),
                    "cluster": template_name,
                    "cluster_no": cluster_no,
                    "dominant_color": color_names[index],
                    "detected_on_count": int(active_counts[index]),
                    "threshold": round(float(threshold_rows[index][0]), 4),
                    "threshold_method": threshold_rows[index][1],
                    "is_representative": int(index == medoid),
                }
            )
        make_contact_sheet(member_items, clusters_dir / f"{template_name}_members.jpg")
        templates.append(
            {
                "name": template_name,
                "cluster_no": cluster_no,
                "member_count": len(indices),
                "dominant_color": dominant_color,
                "representative_image": str(representative_path),
                "overlay_image": str(overlay_path),
                "expected_on_count": int(template_mask.sum()),
                "expected_on_slot_ids": [slot.id for slot, is_on in zip(slots, template_mask.tolist()) if is_on],
                "expected_on_mask": template_mask.astype(np.uint8).tolist(),
                "mean_score_vector": [round(float(value), 5) for value in scores_arr[indices].mean(axis=0).tolist()],
            }
        )

    overview_items = [(Path(template["overlay_image"]), f"{template['name']} n={template['member_count']} on={template['expected_on_count']}") for template in templates]
    make_contact_sheet(overview_items, output_dir / "cluster_template_overview.jpg", cols=3)
    write_csv(
        output_dir / "cluster_assignments.csv",
        sorted(rows, key=lambda row: str(row["image"])),
        ["image", "cluster", "cluster_no", "dominant_color", "detected_on_count", "threshold", "threshold_method", "is_representative"],
    )

    payload = {
        "type": "clustered_matrix_template_library",
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "geometry_template": str(geometry_template),
        "input_dir": str(input_dir),
        "exclude_singletons_from": str(exclude_singletons_from) if exclude_singletons_from is not None else None,
        "min_source_cluster_size": min_source_cluster_size,
        "image_count": len(paths),
        "selected_k": k,
        "k_scores": k_scores,
        "grid_info": geometry_payload.get("grid_info"),
        "slots": geometry_payload.get("slots", []),
        "templates": templates,
    }
    write_json(output_dir / "clustered_matrix_templates.json", payload)
    return {
        "output_dir": str(output_dir),
        "images": len(paths),
        "clusters": len(templates),
        "template_library": str(output_dir / "clustered_matrix_templates.json"),
        "assignments": str(output_dir / "cluster_assignments.csv"),
        "overview": str(output_dir / "cluster_template_overview.jpg"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unsupervised clustering for dense LED matrix templates.")
    parser.add_argument("--geometry-template", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=14)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--all-on-floor", type=float, default=0.55)
    parser.add_argument("--exclude-singletons-from", type=Path, default=None)
    parser.add_argument("--min-source-cluster-size", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = cluster_templates(
        geometry_template=args.geometry_template,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_k=int(args.min_k),
        max_k=int(args.max_k),
        fixed_k=args.k,
        all_on_floor=float(args.all_on_floor),
        exclude_singletons_from=args.exclude_singletons_from,
        min_source_cluster_size=int(args.min_source_cluster_size),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
