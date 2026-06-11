"""
detect_dot_panel.py

点阵面板检测：输入一帧/一个视频，自动ROI+矫正，匹配模板，报告缺点/多点。

逻辑（跟红色面板一样）：
  1. 检测ROI，旋转矫正
  2. 检测当前帧亮点
  3. 跟两个模板比对，选最匹配的
  4. 报告：匹配了哪个模板，缺了哪些点，多了哪些点

Example:
  python scripts/detect_dot_panel.py ^
    --video D:/jiemi/新建文件夹/Video_20260529135514029.avi ^
    --model models/dot_panel/Video_20260529135514029/templates.json ^
    --out output/dot_panel_detect_result
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_pattern_probe import merge_close_points, imwrite_unicode
from scan_refresh_seven_segment_probe import detect_display_roi, crop_roi, Roi


@dataclass
class PanelPose:
    roi: Roi
    angle: float
    source: str = "model"


def detect_all_leds(panel: np.ndarray, params: dict) -> list[dict[str, float]]:
    b, g, r = cv2.split(panel)
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    vt = params.get("value_threshold", 100)
    st_ = params.get("saturation_threshold", 40)
    rt = params.get("red_threshold", 100)
    gt = params.get("green_threshold", 50)
    mr = params.get("merge_radius", 25)

    mask_color = ((v >= vt) & (s >= st_) & (r >= rt) & (g >= gt)).astype(np.uint8) * 255
    mask_white = ((v >= 180) & (s < 80)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_color, mask_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    points = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area < 6 or area > 15000 or w < 2 or h < 2:
            continue
        circularity = area / (3.14159 / 4.0 * max(w, h) ** 2)
        aspect = min(w, h) / max(w, h, 1)
        if circularity < 0.25 or aspect < 0.3:
            continue
        if area <= 2000 and w <= 60 and h <= 60:
            points.append({"x": float(centers[idx][0]), "y": float(centers[idx][1]),
                          "area": float(area), "w": float(w), "h": float(h)})
        elif area > 2000:
            region = (labels == idx).astype(np.uint8) * 255
            dist = cv2.distanceTransform(region[y:y+h, x:x+w], cv2.DIST_L2, 5)
            _, peaks = cv2.threshold(dist, dist.max() * 0.4, 255, cv2.THRESH_BINARY)
            n2, _, st2, ctr2 = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), 8)
            if n2 > 2:
                for i in range(1, n2):
                    points.append({"x": float(ctr2[i][0]) + x, "y": float(ctr2[i][1]) + y,
                                  "area": float(st2[i][4]), "w": float(st2[i][2]), "h": float(st2[i][3])})
            else:
                points.append({"x": float(centers[idx][0]), "y": float(centers[idx][1]),
                              "area": float(area), "w": float(w), "h": float(h)})
    return merge_close_points(points, mr)


def compute_angle(points: list[dict[str, float]]) -> float:
    if len(points) < 5:
        return 0.0
    sorted_pts = sorted(points, key=lambda p: p["y"])
    rows: list[list[dict[str, float]]] = []
    current_row = [sorted_pts[0]]
    for point in sorted_pts[1:]:
        if point["y"] - current_row[-1]["y"] < 30:
            current_row.append(point)
        else:
            rows.append(current_row)
            current_row = [point]
    rows.append(current_row)

    angles: list[float] = []
    for row in rows:
        if len(row) < 3:
            continue
        xs = np.array([point["x"] for point in row], dtype=np.float32)
        ys = np.array([point["y"] for point in row], dtype=np.float32)
        if float(xs.max() - xs.min()) <= 50.0:
            continue
        slope = float(np.polyfit(xs, ys, 1)[0])
        angles.append(float(np.degrees(np.arctan(slope))))
    return float(np.median(angles)) if angles else 0.0


def reference_panel_size(model: dict) -> tuple[int, int]:
    roi_data = model["roi"]
    return int(roi_data["w"]), int(roi_data["h"])


def _is_reasonable_auto_roi(auto_roi: Roi, model_roi: Roi) -> bool:
    if auto_roi.w <= 0 or auto_roi.h <= 0:
        return False
    width_ratio = auto_roi.w / max(model_roi.w, 1)
    height_ratio = auto_roi.h / max(model_roi.h, 1)
    return 0.55 <= width_ratio <= 1.35 and 0.55 <= height_ratio <= 1.35


def _recenter_roi_like_model(auto_roi: Roi, model_roi: Roi, frame_shape: tuple[int, ...]) -> Roi:
    frame_h, frame_w = frame_shape[:2]
    target_w = min(model_roi.w, frame_w)
    target_h = min(model_roi.h, frame_h)
    center_x = auto_roi.x + auto_roi.w / 2.0
    center_y = auto_roi.y + auto_roi.h / 2.0
    x = int(round(center_x - target_w / 2.0))
    y = int(round(center_y - target_h / 2.0))
    x = max(0, min(x, frame_w - target_w))
    y = max(0, min(y, frame_h - target_h))
    return Roi(x=x, y=y, w=target_w, h=target_h)


def build_max_projection(frames: list[np.ndarray]) -> np.ndarray | None:
    projection: np.ndarray | None = None
    for frame in frames:
        if frame is None or frame.size == 0:
            continue
        if projection is None:
            projection = frame.copy()
        else:
            np.maximum(projection, frame, out=projection)
    return projection


def _shift_roi(base_roi: Roi, dx: int, dy: int, frame_shape: tuple[int, ...]) -> Roi:
    frame_h, frame_w = frame_shape[:2]
    w = min(base_roi.w, frame_w)
    h = min(base_roi.h, frame_h)
    x = max(0, min(base_roi.x + dx, frame_w - w))
    y = max(0, min(base_roi.y + dy, frame_h - h))
    return Roi(x=x, y=y, w=w, h=h)


def _score_roi(image: np.ndarray, roi: Roi, angle: float, model: dict) -> float:
    panel = align_and_crop(image, roi, angle)
    if panel.size == 0:
        return -1e9
    params = model.get("detection_params", {})
    detected = detect_all_leds(panel, params)
    best_score = -1e9
    for tmpl_data in model["templates"].values():
        result = match_template(detected, tmpl_data["points"], match_radius=30.0)
        score = result["matched_count"] - result["missing_count"] * 2 - result["extra_count"]
        if score > best_score:
            best_score = score
    return float(best_score)


def refine_roi_by_template(image: np.ndarray, anchor_roi: Roi, model: dict, angle: float) -> Roi:
    best_roi = anchor_roi
    best_score = _score_roi(image, anchor_roi, angle, model)
    for dy in range(-48, 49, 8):
        for dx in range(-120, 121, 12):
            if dx == 0 and dy == 0:
                continue
            candidate = _shift_roi(anchor_roi, dx, dy, image.shape)
            score = _score_roi(image, candidate, angle, model)
            if score > best_score:
                best_score = score
                best_roi = candidate
    return best_roi


def estimate_panel_pose(frames: list[np.ndarray], model: dict, refine_template: bool = True) -> PanelPose:
    roi_data = model["roi"]
    model_roi = Roi(int(roi_data["x"]), int(roi_data["y"]), int(roi_data["w"]), int(roi_data["h"]))
    model_angle = float(model.get("angle", 0.0))
    projection = build_max_projection(frames)
    if projection is None:
        return PanelPose(roi=model_roi, angle=model_angle, source="model")

    pose_roi = model_roi
    source = "model"
    try:
        auto_roi = detect_display_roi(projection, 40, 0.06)
        if _is_reasonable_auto_roi(auto_roi, model_roi):
            pose_roi = _recenter_roi_like_model(auto_roi, model_roi, projection.shape)
            source = "auto"
    except Exception:
        pass

    if refine_template:
        refined_roi = refine_roi_by_template(projection, pose_roi, model, model_angle)
        if refined_roi != pose_roi:
            source = f"{source}+search"
        pose_roi = refined_roi
    return PanelPose(roi=pose_roi, angle=model_angle, source=source)


def align_and_crop(frame: np.ndarray, roi: Roi, angle: float) -> np.ndarray:
    panel = crop_roi(frame, roi)
    if panel.size == 0:
        return panel
    if abs(angle) < 0.01:
        return panel
    h, w = panel.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(panel, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def prepare_panel(frame: np.ndarray, pose: PanelPose, model: dict) -> np.ndarray:
    panel = align_and_crop(frame, pose.roi, pose.angle)
    if panel.size == 0:
        return panel
    ref_w, ref_h = reference_panel_size(model)
    if panel.shape[1] != ref_w or panel.shape[0] != ref_h:
        panel = cv2.resize(panel, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
    return panel


def match_template(detected_points: list[dict[str, float]],
                   template_points: list[dict[str, float]],
                   match_radius: float = 30.0) -> dict:
    """比对检测点和模板点，返回匹配/缺失/多余。"""
    matched_template: set[int] = set()
    matched_detected: set[int] = set()
    suppressed_duplicates: set[int] = set()
    candidate_radius = max(match_radius + 14.0, match_radius * 1.45)
    assignments: dict[int, list[tuple[float, int]]] = {}

    for di, dp in enumerate(detected_points):
        best_dist = candidate_radius
        best_ti = -1
        for ti, tp in enumerate(template_points):
            dist = ((dp["x"] - tp["x"]) ** 2 + (dp["y"] - tp["y"]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_ti = ti
        if best_ti >= 0:
            assignments.setdefault(best_ti, []).append((best_dist, di))

    for ti, candidates in assignments.items():
        candidates.sort(key=lambda item: item[0])
        best_dist, best_di = candidates[0]
        if best_dist <= match_radius:
            matched_template.add(ti)
            matched_detected.add(best_di)
            for duplicate_dist, duplicate_di in candidates[1:]:
                if duplicate_dist <= candidate_radius:
                    suppressed_duplicates.add(duplicate_di)

    missing = [template_points[i] for i in range(len(template_points)) if i not in matched_template]
    extra = [
        detected_points[i]
        for i in range(len(detected_points))
        if i not in matched_detected and i not in suppressed_duplicates
    ]

    return {
        "matched_count": len(matched_template),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_points": missing,
        "extra_points": extra,
    }


def detect_frame(frame: np.ndarray, model: dict, out_dir: Path | None = None,
                 frame_label: str = "", pose: PanelPose | None = None) -> dict:
    """对单帧做检测，返回结果。"""
    params = model.get("detection_params", {})
    templates = model["templates"]
    if pose is None:
        pose = estimate_panel_pose([frame], model)

    # ROI + 矫正 + 归一化到模板坐标系
    panel = prepare_panel(frame, pose, model)
    
    # 检测亮点
    detected = detect_all_leds(panel, params)

    # 跟每个模板比对，选最佳匹配
    best_template = ""
    best_result = None
    best_score = -1

    for tmpl_key, tmpl_data in templates.items():
        tmpl_points = [{"x": p["x"], "y": p["y"]} for p in tmpl_data["points"]]
        result = match_template(detected, tmpl_points, match_radius=30.0)
        # 得分：匹配数 - 缺失数 - 多余数
        score = result["matched_count"] - result["missing_count"] * 2 - result["extra_count"]
        if score > best_score:
            best_score = score
            best_template = tmpl_key
            best_result = result

    # 暗帧（几乎没亮点）
    if not best_template or len(detected) < 3:
        output = {
            "ok": True,
            "status": "DARK",
            "detected_count": len(detected),
            "matched_template": "",
            "template_name": "暗帧",
            "expected_count": 0,
            "matched_count": 0,
            "missing_count": 0,
            "extra_count": 0,
            "message": f"DARK: 暗帧({len(detected)}点), 跳过",
            "pose_source": pose.source,
            "pose_roi": {"x": pose.roi.x, "y": pose.roi.y, "w": pose.roi.w, "h": pose.roi.h},
            "pose_angle": float(pose.angle),
        }
        return output

    tmpl_data = templates[best_template]
    expected = tmpl_data["expected_count"]
    ok = best_result["missing_count"] == 0 and best_result["extra_count"] == 0

    output = {
        "ok": ok,
        "status": "PASS" if ok else "NG",
        "detected_count": len(detected),
        "matched_template": best_template,
        "template_name": tmpl_data["name"],
        "expected_count": expected,
        "matched_count": best_result["matched_count"],
        "missing_count": best_result["missing_count"],
        "extra_count": best_result["extra_count"],
        "message": f"{'PASS' if ok else 'NG'}: 检测{len(detected)}点, 匹配[{tmpl_data['name']}] {best_result['matched_count']}/{expected}, 缺{best_result['missing_count']} 多{best_result['extra_count']}",
        "pose_source": pose.source,
        "pose_roi": {"x": pose.roi.x, "y": pose.roi.y, "w": pose.roi.w, "h": pose.roi.h},
        "pose_angle": float(pose.angle),
    }

    # 画标注图
    if out_dir:
        ann = panel.copy()
        tmpl_points = tmpl_data["points"]
        # 画模板期望位置（小蓝圈）
        for pt in tmpl_points:
            cv2.circle(ann, (int(pt["x"]), int(pt["y"])), 14, (180, 130, 0), 1)
        # 画检测到的（绿=匹配，红=多余）
        for pt in detected:
            cv2.circle(ann, (int(pt["x"]), int(pt["y"])), 8, (0, 255, 0), 2)
        # 缺失点（黄叉）
        for pt in best_result["missing_points"]:
            cx, cy = int(pt["x"]), int(pt["y"])
            cv2.drawMarker(ann, (cx, cy), (0, 200, 255), cv2.MARKER_CROSS, 20, 3)
        # 多余点（红圈）
        for pt in best_result["extra_points"]:
            cv2.circle(ann, (int(pt["x"]), int(pt["y"])), 12, (0, 0, 255), 3)
        
        label = f"{output['status']} [{tmpl_data['name']}] {best_result['matched_count']}/{expected} miss={best_result['missing_count']} extra={best_result['extra_count']}"
        color = (0, 200, 0) if ok else (0, 0, 220)
        cv2.putText(ann, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if frame_label:
            cv2.putText(ann, frame_label, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        fname = f"result_{frame_label}.jpg" if frame_label else "result.jpg"
        imwrite_unicode(out_dir / fname, ann)

    return output


def main():
    parser = argparse.ArgumentParser(description="点阵面板检测")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/dot_panel_detect_result"))
    parser.add_argument("--frames", type=str, default="0,150,300,500",
                       help="要检测的帧号，逗号分隔，或'all'每10帧检测一次")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = json.loads(args.model.read_text(encoding="utf-8"))
    print(f"模型: {model['panel_type']}")
    for k, v in model["templates"].items():
        print(f"  {k}: {v['name']} ({v['expected_count']}点)")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.frames == "all":
        frame_list = list(range(0, total, 10))
    else:
        frame_list = [int(f.strip()) for f in args.frames.split(",")]

    print(f"\n检测 {len(frame_list)} 帧:")
    results = []
    for fi in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        result = detect_frame(frame, model, out_dir, frame_label=f"f{fi:04d}")
        results.append({"frame": fi, **result})
        status_icon = "✓" if result["ok"] else "✗"
        print(f"  {status_icon} f{fi:4d}: {result['message']}")

    cap.release()

    # 汇总
    pass_count = sum(1 for r in results if r["ok"])
    print(f"\n汇总: {pass_count}/{len(results)} PASS")
    print(f"输出: {out_dir}")


if __name__ == "__main__":
    main()
