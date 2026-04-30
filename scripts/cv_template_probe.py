from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import cv_pattern_probe as cvp


TEMPLATE_SIZE = (120, 480)  # width, height
CLASS_NAMES = ("ledgroup", "ledgroup2", "ledgroup3")
EXPECTED_COUNTS = cvp.EXPECTED_POINT_COUNTS
CLASS_COLORS = cvp.CLASS_COLORS


def load_probe_image(image_path: Path, rotate_180: bool = False) -> np.ndarray | None:
    image = cvp.imread_unicode(image_path)
    if image is None:
        return None
    if rotate_180:
        image = cv2.rotate(image, cv2.ROTATE_180)
    return image


@dataclass
class SlotTemplate:
    class_name: str
    roi_index: int
    size: tuple[int, int]
    points: list[tuple[float, float]]
    sample_count: int


def normalize_points(
    points: list[dict[str, float]],
    roi_shape: tuple[int, int],
    size: tuple[int, int] = TEMPLATE_SIZE,
) -> list[tuple[float, float]]:
    src_h, src_w = roi_shape
    dst_w, dst_h = size
    if src_w <= 0 or src_h <= 0:
        return []
    return [
        (
            float(point["x"]) * dst_w / src_w,
            float(point["y"]) * dst_h / src_h,
        )
        for point in points
    ]


def sort_points(points: np.ndarray) -> np.ndarray:
    order = np.lexsort((points[:, 0], points[:, 1]))
    return points[order]


def fit_slots(point_sets: list[list[tuple[float, float]]], expected_count: int) -> list[tuple[float, float]]:
    if not point_sets:
        return []

    for point_set in point_sets:
        if len(point_set) == expected_count:
            centers = sort_points(np.asarray(point_set, dtype=np.float32))
            return [(float(x), float(y)) for x, y in centers]
    return []


def build_templates(
    image_files: list[Path],
    sample_stride: int,
    max_samples_per_template: int,
    point_merge_radius: float,
    rotate_180: bool = False,
) -> dict[str, dict[int, SlotTemplate]]:
    collected: dict[tuple[str, int], list[list[tuple[float, float]]]] = {}

    def enough_samples() -> bool:
        for class_name in CLASS_NAMES:
            for roi_index in range(20):
                if len(collected.get((class_name, roi_index), [])) < max_samples_per_template:
                    return False
        return True

    for image_path in image_files[:: max(1, sample_stride)]:
        image = load_probe_image(image_path, rotate_180=rotate_180)
        if image is None:
            continue
        boxes, _ = cvp.detect_roi_boxes(image)
        if len(boxes) != 20:
            continue

        for roi_index, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy
            roi = image[y1:y2, x1:x2]
            try:
                result = cvp.analyze_roi(roi, point_merge_radius)
            except cv2.error:
                continue
            class_name = result["class_name"]
            expected = EXPECTED_COUNTS.get(class_name)
            if expected is None or result["point_count"] != expected:
                continue
            key = (class_name, roi_index)
            samples = collected.setdefault(key, [])
            if len(samples) >= max_samples_per_template:
                continue
            normalized = normalize_points(result["points"], roi.shape[:2])
            if len(normalized) == expected:
                samples.append(normalized)

        if enough_samples():
            break

    templates: dict[str, dict[int, SlotTemplate]] = {class_name: {} for class_name in CLASS_NAMES}
    missing: list[str] = []
    for class_name in CLASS_NAMES:
        expected = EXPECTED_COUNTS[class_name]
        for roi_index in range(20):
            samples = collected.get((class_name, roi_index), [])
            points = fit_slots(samples, expected)
            if not points:
                missing.append(f"{class_name}:{roi_index}")
                continue
            templates[class_name][roi_index] = SlotTemplate(
                class_name=class_name,
                roi_index=roi_index,
                size=TEMPLATE_SIZE,
                points=points,
                sample_count=len(samples),
            )

    if missing:
        raise RuntimeError("missing templates: " + ", ".join(missing))
    return templates


def save_templates(
    templates: dict[str, dict[int, SlotTemplate]],
    output_path: Path,
    source_orientation: str = "as_captured",
) -> None:
    payload: dict[str, Any] = {
        "size": list(TEMPLATE_SIZE),
        "expected_counts": EXPECTED_COUNTS,
        "source_orientation": source_orientation,
        "templates": {},
    }
    for class_name, by_index in templates.items():
        payload["templates"][class_name] = {}
        for roi_index, template in by_index.items():
            payload["templates"][class_name][str(roi_index)] = {
                "sample_count": template.sample_count,
                "points": [[round(x, 3), round(y, 3)] for x, y in template.points],
            }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_templates(input_path: Path) -> dict[str, dict[int, SlotTemplate]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    size_payload = payload.get("size", TEMPLATE_SIZE)
    size = (int(size_payload[0]), int(size_payload[1]))
    raw_templates = payload.get("templates", {})

    templates: dict[str, dict[int, SlotTemplate]] = {class_name: {} for class_name in CLASS_NAMES}
    missing: list[str] = []
    for class_name in CLASS_NAMES:
        by_index = raw_templates.get(class_name, {})
        for roi_index in range(20):
            raw = by_index.get(str(roi_index))
            if raw is None:
                missing.append(f"{class_name}:{roi_index}")
                continue
            templates[class_name][roi_index] = SlotTemplate(
                class_name=class_name,
                roi_index=roi_index,
                size=size,
                points=[(float(point[0]), float(point[1])) for point in raw.get("points", [])],
                sample_count=int(raw.get("sample_count", 0)),
            )

    if missing:
        raise RuntimeError("missing templates: " + ", ".join(missing))
    return templates


def make_evidence_map(roi: np.ndarray, size: tuple[int, int] = TEMPLATE_SIZE) -> np.ndarray:
    resized = cv2.resize(roi, size, interpolation=cv2.INTER_LINEAR)
    b, g, r = cv2.split(resized)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    r_f = r.astype(np.float32)
    g_f = g.astype(np.float32)
    v_f = v.astype(np.float32)
    s_f = s.astype(np.float32)
    h_f = h.astype(np.float32)

    brightness = np.clip(v_f / 220.0, 0.0, 1.0)
    red = np.clip(r_f / 190.0, 0.0, 1.0)
    green = np.clip(g_f / 120.0, 0.0, 1.0)
    saturation = np.clip(s_f / 70.0, 0.0, 1.0)
    evidence = brightness * red * green * saturation

    evidence[v_f < 80] = 0.0

    return cv2.GaussianBlur(evidence, (3, 3), 0)


def slot_scores(
    evidence: np.ndarray,
    points: list[tuple[float, float]],
    radius: int,
    offset: tuple[int, int] = (0, 0),
) -> list[float]:
    height, width = evidence.shape[:2]
    dx, dy = offset
    scores: list[float] = []
    for x, y in points:
        cx = int(round(x + dx))
        cy = int(round(y + dy))
        x1 = max(0, cx - radius)
        x2 = min(width, cx + radius + 1)
        y1 = max(0, cy - radius)
        y2 = min(height, cy + radius + 1)
        if x1 >= x2 or y1 >= y2:
            scores.append(0.0)
            continue
        scores.append(float(evidence[y1:y2, x1:x2].max()))
    return scores


def candidate_offsets(roi_index: int) -> list[tuple[int, int]]:
    if roi_index not in (0, 19):
        return [(0, 0)]
    return [
        (dx, dy)
        for dx in range(-30, 31, 6)
        for dy in range(-12, 13, 6)
    ]


def full_template_indices(
    full_points: list[tuple[float, float]],
    on_points: list[tuple[float, float]],
) -> set[int]:
    if len(full_points) == len(on_points):
        return set(range(len(full_points)))

    full = np.asarray(full_points, dtype=np.float32)
    on = np.asarray(on_points, dtype=np.float32)
    distances = np.linalg.norm(on[:, None, :] - full[None, :, :], axis=2)
    return {int(index) for index in np.argmin(distances, axis=1)}


def match_roi(
    roi: np.ndarray,
    roi_index: int,
    templates: dict[str, dict[int, SlotTemplate]],
    slot_radius: int,
    slot_threshold: float,
    extra_penalty: float,
) -> dict[str, Any]:
    evidence = make_evidence_map(roi)
    full_template = templates["ledgroup"][roi_index]

    class_results: dict[str, dict[str, Any]] = {}
    for class_name in CLASS_NAMES:
        template = templates[class_name][roi_index]
        best_result: dict[str, Any] | None = None
        for offset in candidate_offsets(roi_index):
            full_scores = slot_scores(evidence, full_template.points, slot_radius, offset)
            full_hits = [score >= slot_threshold for score in full_scores]
            scores = full_scores if class_name == "ledgroup" else slot_scores(
                evidence,
                template.points,
                slot_radius,
                offset,
            )
            hits = [score >= slot_threshold for score in scores]
            on_count = sum(hits)
            on_ratio = on_count / max(1, len(hits))

            if class_name == "ledgroup":
                extra_count = 0
                extra_ratio = 0.0
                score = on_ratio
                extra_indices: list[int] = []
            else:
                on_full_indices = full_template_indices(full_template.points, template.points)
                off_indices = [idx for idx in range(len(full_template.points)) if idx not in on_full_indices]
                extra_indices = [idx for idx in off_indices if full_hits[idx]]
                extra_count = len(extra_indices)
                extra_ratio = extra_count / max(1, len(off_indices))
                score = on_ratio - extra_penalty * extra_ratio

            offset_result = {
                "score": score,
                "on_ratio": on_ratio,
                "on_count": on_count,
                "extra_count": extra_count,
                "extra_ratio": extra_ratio,
                "slot_scores": scores,
                "hits": hits,
                "extra_indices": extra_indices,
                "offset": offset,
            }
            if best_result is None or offset_result["score"] > best_result["score"]:
                best_result = offset_result

        assert best_result is not None
        class_results[class_name] = {
            "score": score,
            **best_result,
        }

    ranked = sorted(class_results.items(), key=lambda item: item[1]["score"], reverse=True)
    best_class, best = ranked[0]
    second_score = ranked[1][1]["score"] if len(ranked) > 1 else 0.0
    expected = EXPECTED_COUNTS[best_class]
    point_count = int(best["on_count"])

    return {
        "original_class_name": best_class,
        "class_name": best_class,
        "class_id": cvp.CLASS_IDS[best_class],
        "point_count": point_count,
        "expected_point_count": expected,
        "point_count_delta": point_count - expected,
        "count_status": "ok" if point_count == expected else "abnormal",
        "class_score": float(best["score"]),
        "score_margin": float(best["score"] - second_score),
        "on_ratio": float(best["on_ratio"]),
        "extra_count": int(best["extra_count"]),
        "extra_ratio": float(best["extra_ratio"]),
        "slot_offset": best["offset"],
        "class_scores": {name: round(float(result["score"]), 6) for name, result in class_results.items()},
        "hits": best["hits"],
        "slot_scores": best["slot_scores"],
        "extra_indices": best["extra_indices"],
        "evidence": evidence,
        "class_details": {
            name: {
                "class_name": name,
                "class_id": cvp.CLASS_IDS[name],
                "point_count": int(result["on_count"]),
                "expected_point_count": EXPECTED_COUNTS[name],
                "point_count_delta": int(result["on_count"]) - EXPECTED_COUNTS[name],
                "count_status": "ok" if int(result["on_count"]) == EXPECTED_COUNTS[name] else "abnormal",
                "class_score": float(result["score"]),
                "on_ratio": float(result["on_ratio"]),
                "extra_count": int(result["extra_count"]),
                "extra_ratio": float(result["extra_ratio"]),
                "slot_offset": result["offset"],
                "hits": result["hits"],
                "slot_scores": result["slot_scores"],
                "extra_indices": result["extra_indices"],
            }
            for name, result in class_results.items()
        },
    }


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 4, y)
    cv2.rectangle(image, (x, y - th - 5), (x + tw + 4, y + baseline + 4), (15, 15, 15), -1)
    cv2.putText(image, text, (x + 2, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_roi_slots(
    image: np.ndarray,
    box: cvp.RoiBox,
    template: SlotTemplate,
    result: dict[str, Any],
) -> None:
    x1, y1, x2, y2 = box.xyxy
    roi_w = max(1, x2 - x1)
    roi_h = max(1, y2 - y1)
    template_w, template_h = template.size
    color = CLASS_COLORS.get(result["class_name"], CLASS_COLORS["unknown"])
    offset_x, offset_y = result.get("slot_offset", (0, 0))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    for index, ((px, py), hit) in enumerate(zip(template.points, result["hits"])):
        cx = int(round(x1 + (px + offset_x) * roi_w / template_w))
        cy = int(round(y1 + (py + offset_y) * roi_h / template_h))
        if hit:
            cv2.circle(image, (cx, cy), 4, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(image, (cx, cy), 6, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, str(index + 1), (cx + 4, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    label = (
        f"{result['class_name']} n={result['point_count']}/{result['expected_point_count']} "
        f"s={result['class_score']:.2f}"
    )
    draw_label(image, label, x1, max(18, y1 - 6), color)


def apply_frame_majority(items: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for item in items:
        class_name = item["result"]["class_name"]
        counts[class_name] = counts.get(class_name, 0) + 1
        item["result"]["frame_majority_class"] = ""
        item["result"]["frame_majority_applied"] = 0

    if not counts:
        return
    majority_class, majority_count = max(counts.items(), key=lambda item: item[1])
    if majority_count < 16:
        return

    for item in items:
        result = item["result"]
        result["frame_majority_class"] = majority_class
        if result["class_name"] == majority_class:
            continue
        if result["class_score"] >= 0.65 and result["score_margin"] >= 0.18:
            continue

        detail = result["class_details"][majority_class]
        result["original_class_name"] = result["class_name"]
        result["class_name"] = majority_class
        result["class_id"] = detail["class_id"]
        result["point_count"] = detail["point_count"]
        result["expected_point_count"] = detail["expected_point_count"]
        result["point_count_delta"] = detail["point_count_delta"]
        result["count_status"] = detail["count_status"]
        result["class_score"] = detail["class_score"]
        result["on_ratio"] = detail["on_ratio"]
        result["extra_count"] = detail["extra_count"]
        result["extra_ratio"] = detail["extra_ratio"]
        result["slot_offset"] = detail["slot_offset"]
        result["hits"] = detail["hits"]
        result["slot_scores"] = detail["slot_scores"]
        result["extra_indices"] = detail["extra_indices"]
        result["frame_majority_applied"] = 1


def process_image(
    image_path: Path,
    templates: dict[str, dict[int, SlotTemplate]],
    output_dir: Path,
    slot_radius: int,
    slot_threshold: float,
    extra_penalty: float,
    save_visual: bool,
    rotate_180: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image = load_probe_image(image_path, rotate_180=rotate_180)
    if image is None:
        return [], {"image": image_path.name, "roi_count": 0, "read_error": True}

    boxes, _ = cvp.detect_roi_boxes(image)
    items: list[dict[str, Any]] = []
    visual = image.copy()

    for roi_index, box in enumerate(boxes):
        if roi_index >= 20:
            continue
        x1, y1, x2, y2 = box.xyxy
        roi = image[y1:y2, x1:x2]
        result = match_roi(roi, roi_index, templates, slot_radius, slot_threshold, extra_penalty)
        items.append({"roi_index": roi_index, "box": box, "result": result})

    apply_frame_majority(items)

    rows: list[dict[str, Any]] = []
    for item in items:
        roi_index = item["roi_index"]
        box = item["box"]
        result = item["result"]
        x1, y1, x2, y2 = box.xyxy
        template = templates[result["class_name"]][roi_index]
        if save_visual or result["count_status"] != "ok":
            draw_roi_slots(visual, box, template, result)
        rows.append(
            {
                "image": image_path.name,
                "roi_index": roi_index,
                "bbox": [x1, y1, x2, y2],
                "original_class_name": result["original_class_name"],
                "class_name": result["class_name"],
                "class_id": result["class_id"],
                "frame_majority_class": result.get("frame_majority_class", ""),
                "frame_majority_applied": result.get("frame_majority_applied", 0),
                "point_count": result["point_count"],
                "expected_point_count": result["expected_point_count"],
                "point_count_delta": result["point_count_delta"],
                "count_status": result["count_status"],
                "class_score": result["class_score"],
                "score_margin": result["score_margin"],
                "on_ratio": result["on_ratio"],
                "extra_count": result["extra_count"],
                "extra_ratio": result["extra_ratio"],
                "slot_offset": result["slot_offset"],
                "class_scores": result["class_scores"],
                "missing_slots": [
                    index + 1 for index, hit in enumerate(result["hits"]) if not hit
                ],
                "extra_slots": [index + 1 for index in result["extra_indices"]],
            }
        )

    class_counts = cvp.count_classes(rows, "class_name")
    abnormal_count = sum(1 for row in rows if row["count_status"] != "ok")
    summary = {
        "image": image_path.name,
        "roi_count": len(rows),
        "class_counts": class_counts,
        "abnormal_count": abnormal_count,
    }

    if save_visual or abnormal_count or len(rows) != 20:
        vis_dir = output_dir / "visualized"
        cvp.imwrite_unicode(vis_dir / f"vis_{image_path.stem}.jpg", visual)

    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe LED template-slot matching.")
    parser.add_argument("--input-dir", type=Path, default=Path(r"D:\Desktop2\Images\extracted_frames"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\Desktop2\Images\cv_template_probe"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--template-sample-stride", type=int, default=5)
    parser.add_argument("--max-template-samples", type=int, default=40)
    parser.add_argument("--point-merge-radius", type=float, default=14.0)
    parser.add_argument("--slot-radius", type=int, default=9)
    parser.add_argument("--slot-threshold", type=float, default=0.42)
    parser.add_argument("--extra-penalty", type=float, default=0.65)
    parser.add_argument("--save-all-visuals", action="store_true")
    parser.add_argument("--rotate-180", action="store_true", help="Rotate source images 180 degrees before building and probing templates.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "roi_index",
        "bbox",
        "original_class_name",
        "class_name",
        "class_id",
        "frame_majority_class",
        "frame_majority_applied",
        "point_count",
        "expected_point_count",
        "point_count_delta",
        "count_status",
        "class_score",
        "score_margin",
        "on_ratio",
        "extra_count",
        "extra_ratio",
        "slot_offset",
        "class_scores",
        "missing_slots",
        "extra_slots",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    vis_dir = args.output_dir / "visualized"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    for old_visual in vis_dir.glob("*.jpg"):
        old_visual.unlink()

    image_files = sorted(args.input_dir.glob("*.jpg"))
    if args.limit > 0:
        run_files = image_files[: args.limit]
    else:
        run_files = image_files

    print("building templates...")
    templates = build_templates(
        image_files=image_files,
        sample_stride=args.template_sample_stride,
        max_samples_per_template=args.max_template_samples,
        point_merge_radius=args.point_merge_radius,
        rotate_180=args.rotate_180,
    )
    templates_path = args.output_dir / "templates.json"
    source_orientation = "rotated_180_from_source_images" if args.rotate_180 else "as_captured"
    save_templates(templates, templates_path, source_orientation=source_orientation)
    print(f"templates: {templates_path}")

    all_rows: list[dict[str, Any]] = []
    image_summary: list[dict[str, Any]] = []
    for image_path in run_files:
        rows, summary = process_image(
            image_path=image_path,
            templates=templates,
            output_dir=args.output_dir,
            slot_radius=args.slot_radius,
            slot_threshold=args.slot_threshold,
            extra_penalty=args.extra_penalty,
            save_visual=args.save_all_visuals,
            rotate_180=args.rotate_180,
        )
        all_rows.extend(rows)
        image_summary.append(summary)
        print(
            f"{image_path.name}: roi={summary['roi_count']} "
            f"classes={summary['class_counts']} abnormal={summary['abnormal_count']}"
        )

    summary_path = args.output_dir / "summary.csv"
    anomalies_path = args.output_dir / "anomalies.csv"
    image_summary_path = args.output_dir / "image_summary.json"
    write_csv(summary_path, all_rows)
    write_csv(anomalies_path, [row for row in all_rows if row["count_status"] != "ok"])
    image_summary_path.write_text(json.dumps(image_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"summary: {summary_path}")
    print(f"anomalies: {anomalies_path}")
    print(f"image summary: {image_summary_path}")
    print(f"visualized: {args.output_dir / 'visualized'}")


if __name__ == "__main__":
    main()
