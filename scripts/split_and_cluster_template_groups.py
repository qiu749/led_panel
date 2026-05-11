from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import build_direct_point_templates as direct_tpl


PYTHON_JSON_KW = {"ensure_ascii": False, "indent": 2}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(direct_tpl.jsonable(payload), **PYTHON_JSON_KW), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig")))


def render_mask(points: np.ndarray, width: int, height: int, radius: int = 2, blur_sigma: float = 1.0) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y in points:
        cx = int(round(float(x)))
        cy = int(round(float(y)))
        if 0 <= cx < width and 0 <= cy < height:
            cv2.circle(mask, (cx, cy), radius, 255, -1)
    if blur_sigma > 0:
        return cv2.GaussianBlur(mask, (0, 0), blur_sigma)
    return mask


def feature_from_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    mask = render_mask(points, width, height, radius=2, blur_sigma=1.0)
    small = cv2.resize(mask, (96, 64), interpolation=cv2.INTER_AREA)
    return (small.astype(np.float32) / 255.0).reshape(-1)


def standardize(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-6)


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
        if int(same.sum()) <= 1:
            continue
        a = float(dist[index, same].sum() / max(int(same.sum()) - 1, 1))
        b = min(float(dist[index, labels == other].mean()) for other in unique if other != int(label))
        scores.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(scores)) if scores else -1.0


def choose_k(features: np.ndarray, max_k: int, min_cluster_size: int) -> tuple[int, np.ndarray]:
    if len(features) < min_cluster_size * 2:
        return 1, np.zeros(len(features), dtype=np.int32)
    best_k = 1
    best_labels = np.zeros(len(features), dtype=np.int32)
    best_score = -1.0
    for k in range(2, min(max_k, len(features)) + 1):
        labels, _ = kmeans(features, k)
        counts = np.bincount(labels, minlength=k)
        if int(counts.min()) < min_cluster_size:
            continue
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    if best_score < 0.06:
        return 1, np.zeros(len(features), dtype=np.int32)
    return best_k, best_labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split classified panels into folders and cluster one template group.")
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--template-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cluster-template", type=str, required=True)
    parser.add_argument("--max-k", type=int, default=4)
    parser.add_argument("--min-cluster-size", type=int, default=12)
    parser.add_argument("--stable-limit", type=int, default=120)
    return parser


def split_by_template(rows: list[dict[str, str]], output_dir: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        name = row["matched_template"]
        grouped.setdefault(name, []).append(row)
    for template_name, items in grouped.items():
        group_dir = output_dir / "by_template" / template_name
        link_dir = group_dir / "aligned_panels"
        csv_rows: list[dict[str, str]] = []
        for row in items:
            src = Path(row["image"])
            dst = link_dir / src.name
            safe_link_or_copy(src, dst)
            new_row = dict(row)
            new_row["linked_image"] = str(dst)
            csv_rows.append(new_row)
        write_csv(group_dir / "group_results.csv", csv_rows, list(csv_rows[0].keys()) if csv_rows else list(items[0].keys()) + ["linked_image"])
    return grouped


def cluster_one_template(
    template_name: str,
    rows: list[dict[str, str]],
    template_library: Path,
    output_dir: Path,
    max_k: int,
    min_cluster_size: int,
    stable_limit: int,
) -> dict[str, Any]:
    payload = json.loads(template_library.read_text(encoding="utf-8"))
    width, height = [int(v) for v in payload["image_size"]]
    rows = [row for row in rows if row["valid_detection"] == "1"]
    rows = sorted(
        rows,
        key=lambda row: (
            float(row["composite_score"]),
            int(row["missing_count"]) + int(row["extra_count"]),
            abs(int(row["align_dx"])) + abs(int(row["align_dy"])),
        ),
    )
    if not rows:
        raise RuntimeError(f"No valid rows for {template_name}")

    stable_rows = rows[: min(len(rows), stable_limit)]
    features: list[np.ndarray] = []
    point_sets: list[np.ndarray] = []
    kept_rows: list[dict[str, str]] = []
    for row in stable_rows:
        image = cv2.imread(str(Path(row["image"])))
        if image is None:
            continue
        points, _ = direct_tpl.detect_points(image)
        arr = np.asarray(points, dtype=np.float32)
        if len(arr) == 0:
            continue
        arr[:, 0] += float(row["align_dx"])
        arr[:, 1] += float(row["align_dy"])
        features.append(feature_from_points(arr, width, height))
        point_sets.append(arr)
        kept_rows.append(row)
    if not features:
        raise RuntimeError(f"No point sets kept for {template_name}")

    feature_array = standardize(np.stack(features, axis=0))
    k, labels = choose_k(feature_array, max_k=max_k, min_cluster_size=min_cluster_size)
    counts = Counter(labels.tolist())

    cluster_root = output_dir / "clusters" / template_name
    member_rows: list[dict[str, Any]] = []
    comparison_paths: list[Path] = []
    refined_templates: list[dict[str, Any]] = []
    for cluster_id in sorted(counts):
        indices = [index for index, label in enumerate(labels.tolist()) if label == cluster_id]
        cluster_dir = cluster_root / f"cluster_{cluster_id + 1:02d}"
        img_dir = cluster_dir / "aligned_panels"
        accum = np.zeros((height, width), dtype=np.float32)
        for index in indices:
            row = kept_rows[index]
            src = Path(row["image"])
            dst = img_dir / src.name
            safe_link_or_copy(src, dst)
            member_rows.append(
                {
                    "template_name": template_name,
                    "cluster_id": cluster_id + 1,
                    "image": str(src),
                    "linked_image": str(dst),
                    "composite_score": row["composite_score"],
                    "missing_count": row["missing_count"],
                    "extra_count": row["extra_count"],
                    "align_dx": row["align_dx"],
                    "align_dy": row["align_dy"],
                }
            )
            accum += render_mask(point_sets[index], width, height, radius=2, blur_sigma=1.0).astype(np.float32) / 255.0
        accum /= max(1, len(indices))
        consensus_gray = np.clip(accum * 255.0, 0, 255).astype(np.uint8)
        consensus_bgr = cv2.cvtColor(consensus_gray, cv2.COLOR_GRAY2BGR)
        consensus_points, detect_stats = direct_tpl.detect_points(consensus_bgr)
        out_path = cluster_dir / f"{template_name}_cluster_{cluster_id + 1:02d}_template.jpg"
        direct_tpl.draw_points(consensus_points, width, height, f"{template_name} c{cluster_id + 1} n={len(indices)}", out_path)
        comparison_paths.append(out_path)
        refined_templates.append(
            {
                "name": f"{template_name}_cluster_{cluster_id + 1:02d}",
                "source_image": kept_rows[indices[0]]["image"],
                "point_mask_image": str(out_path),
                "point_count": len(consensus_points),
                "points": [{"x": round(float(x), 3), "y": round(float(y), 3)} for x, y in consensus_points],
                "detection_stats": detect_stats,
                "member_count": len(indices),
            }
        )

    if comparison_paths:
        images = [cv2.imread(str(path)) for path in comparison_paths]
        max_height = max(image.shape[0] for image in images if image is not None)
        parts: list[np.ndarray] = []
        for index, image in enumerate(images):
            assert image is not None
            if image.shape[0] < max_height:
                pad = np.zeros((max_height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
                image = np.vstack([image, pad])
            parts.append(image)
            if index != len(images) - 1:
                parts.append(np.full((max_height, 12, 3), 24, dtype=np.uint8))
        cv2.imwrite(str(cluster_root / "cluster_template_comparison.jpg"), np.hstack(parts))

    write_csv(
        cluster_root / "cluster_members.csv",
        member_rows,
        [
            "template_name",
            "cluster_id",
            "image",
            "linked_image",
            "composite_score",
            "missing_count",
            "extra_count",
            "align_dx",
            "align_dy",
        ],
    )
    refined_library = {
        "type": "direct_point_template_library",
        "version": 3,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_size": [width, height],
        "comparison_image": str(cluster_root / "cluster_template_comparison.jpg"),
        "templates": refined_templates,
    }
    write_json(cluster_root / "clustered_templates.json", refined_library)
    return {
        "template_name": template_name,
        "stable_input_count": len(kept_rows),
        "cluster_count": int(k),
        "cluster_sizes": {f"cluster_{cluster_id + 1:02d}": int(count) for cluster_id, count in sorted(counts.items())},
        "cluster_root": str(cluster_root),
        "clustered_templates": str(cluster_root / "clustered_templates.json"),
    }


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.results_csv)
    grouped = split_by_template(rows, args.output_dir)
    if args.cluster_template not in grouped:
        raise RuntimeError(f"Template not found in results: {args.cluster_template}")
    summary = {
        "results_csv": str(args.results_csv),
        "template_library": str(args.template_library),
        "output_dir": str(args.output_dir),
        "template_counts": {name: len(items) for name, items in grouped.items()},
        "cluster_summary": cluster_one_template(
            template_name=args.cluster_template,
            rows=grouped[args.cluster_template],
            template_library=args.template_library,
            output_dir=args.output_dir,
            max_k=args.max_k,
            min_cluster_size=args.min_cluster_size,
            stable_limit=args.stable_limit,
        ),
    }
    write_json(args.output_dir / "split_and_cluster_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
