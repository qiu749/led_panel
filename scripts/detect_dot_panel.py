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
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_pattern_probe import merge_close_points, imwrite_unicode
from scan_refresh_seven_segment_probe import detect_display_roi, crop_roi, Roi


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


def align_and_crop(frame: np.ndarray, roi: Roi, angle: float) -> np.ndarray:
    panel = crop_roi(frame, roi)
    if abs(angle) < 0.01:
        return panel
    h, w = panel.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(panel, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def match_template(detected_points: list[dict[str, float]],
                   template_points: list[dict[str, float]],
                   match_radius: float = 30.0) -> dict:
    """比对检测点和模板点，返回匹配/缺失/多余。"""
    matched_template = set()
    matched_detected = set()

    for di, dp in enumerate(detected_points):
        best_dist = match_radius
        best_ti = -1
        for ti, tp in enumerate(template_points):
            if ti in matched_template:
                continue
            dist = ((dp["x"] - tp["x"]) ** 2 + (dp["y"] - tp["y"]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_ti = ti
        if best_ti >= 0:
            matched_template.add(best_ti)
            matched_detected.add(di)

    missing = [template_points[i] for i in range(len(template_points)) if i not in matched_template]
    extra = [detected_points[i] for i in range(len(detected_points)) if i not in matched_detected]

    return {
        "matched_count": len(matched_template),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_points": missing,
        "extra_points": extra,
    }


def detect_frame(frame: np.ndarray, model: dict, out_dir: Path | None = None,
                 frame_label: str = "") -> dict:
    """对单帧做检测，返回结果。"""
    roi_data = model["roi"]
    roi = Roi(roi_data["x"], roi_data["y"], roi_data["w"], roi_data["h"])
    angle = model.get("angle", 0.0)
    params = model.get("detection_params", {})
    templates = model["templates"]

    # ROI + 矫正
    panel = align_and_crop(frame, roi, angle)
    
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
