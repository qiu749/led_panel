"""
从视频提取两个点位模板：48点(T001) 和 81点(全亮)。
输出 templates.json 供检测使用。
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import cv2
import json
import numpy as np
from pathlib import Path
from cv_pattern_probe import merge_close_points, imwrite_unicode
from scan_refresh_seven_segment_probe import detect_display_roi, crop_roi, Roi


def detect_all_leds(panel: np.ndarray, merge_radius: float = 25.0) -> list[dict[str, float]]:
    b, g, r = cv2.split(panel)
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask_color = ((v >= 100) & (s >= 40) & (r >= 100) & (g >= 50)).astype(np.uint8) * 255
    mask_white = ((v >= 180) & (s < 80)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_color, mask_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    roi_area = panel.shape[0] * panel.shape[1]
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
    return merge_close_points(points, merge_radius)


def compute_angle(points: list[dict[str, float]]) -> float:
    if len(points) < 5:
        return 0.0
    sorted_pts = sorted(points, key=lambda p: p["y"])
    rows, current_row = [], [sorted_pts[0]]
    for p in sorted_pts[1:]:
        if p["y"] - current_row[-1]["y"] < 30:
            current_row.append(p)
        else:
            rows.append(current_row)
            current_row = [p]
    rows.append(current_row)
    angles = []
    for row in rows:
        if len(row) >= 3:
            xs = np.array([p["x"] for p in row])
            ys = np.array([p["y"] for p in row])
            if xs.max() - xs.min() > 50:
                angles.append(np.degrees(np.arctan(np.polyfit(xs, ys, 1)[0])))
    return float(np.median(angles)) if angles else 0.0


def align_and_crop(frame: np.ndarray, roi: Roi, angle: float) -> np.ndarray:
    panel = crop_roi(frame, roi)
    if abs(angle) < 0.01:
        return panel
    h, w = panel.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(panel, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


# ─── 主流程 ───
VIDEO = Path(r"D:\jiemi\新建文件夹\Video_20260529135514029.avi")
OUT_DIR = Path("models/dot_panel/Video_20260529135514029")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[1] 构建最大投影...")
cap = cv2.VideoCapture(str(VIDEO))
max_proj = None
while True:
    ok, f = cap.read()
    if not ok: break
    if max_proj is None: max_proj = f.copy()
    else: np.maximum(max_proj, f, out=max_proj)
cap.release()

roi = detect_display_roi(max_proj, 40, 0.06)
panel_raw = crop_roi(max_proj, roi)
points_raw = detect_all_leds(panel_raw, 25)
angle = compute_angle(points_raw)
print(f"    ROI: ({roi.x},{roi.y},{roi.w},{roi.h}), angle={angle:.2f}°")

# 全亮模板 (81点) - 从最大投影
panel_full = align_and_crop(max_proj, roi, angle)
template_81 = detect_all_leds(panel_full, 25)
print(f"[2] 全亮模板: {len(template_81)} 点")

# 48点模板 - 从frame 0
cap = cv2.VideoCapture(str(VIDEO))
ok, frame0 = cap.read()
cap.release()
panel_48 = align_and_crop(frame0, roi, angle)
template_48 = detect_all_leds(panel_48, 25)
print(f"    48点模板: {len(template_48)} 点")

# 保存标注图
for name, panel, pts in [("template_81pts", panel_full, template_81), ("template_48pts", panel_48, template_48)]:
    ann = panel.copy()
    for pt in pts:
        cv2.circle(ann, (int(pt["x"]), int(pt["y"])), 10, (0, 255, 0), 2)
        cv2.circle(ann, (int(pt["x"]), int(pt["y"])), 2, (0, 0, 255), -1)
    cv2.putText(ann, f"{name}: {len(pts)} points", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    imwrite_unicode(OUT_DIR / f"{name}.jpg", ann)

# 保存模板JSON
model = {
    "panel_type": "dot_panel_529",
    "video_source": str(VIDEO),
    "roi": {"x": roi.x, "y": roi.y, "w": roi.w, "h": roi.h},
    "angle": angle,
    "detection_params": {
        "value_threshold": 100,
        "saturation_threshold": 40,
        "red_threshold": 100,
        "green_threshold": 50,
        "merge_radius": 25,
        "point_radius": 15,
        "on_threshold": 0.4,
    },
    "templates": {
        "full_81": {
            "name": "全亮81点",
            "expected_count": len(template_81),
            "points": [{"x": round(p["x"], 1), "y": round(p["y"], 1)} for p in template_81],
        },
        "partial_48": {
            "name": "部分亮48点",
            "expected_count": len(template_48),
            "points": [{"x": round(p["x"], 1), "y": round(p["y"], 1)} for p in template_48],
        },
    },
}

(OUT_DIR / "templates.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[3] 已保存:")
print(f"    {OUT_DIR / 'templates.json'}")
print(f"    {OUT_DIR / 'template_81pts.jpg'}")
print(f"    {OUT_DIR / 'template_48pts.jpg'}")
