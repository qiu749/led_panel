"""修正：加大max_area处理白色过曝块，减小merge_radius保留紧挨双点。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from pathlib import Path
from cv_pattern_probe import merge_close_points, imwrite_unicode
from scripts.scan_refresh_seven_segment_probe import detect_display_roi, crop_roi, Roi

VIDEO = Path(r"D:\jiemi\新建文件夹\Video_20260529135514029.avi")
OUT_DIR = Path("output/dot_panel_cvpattern_test")


def detect_all_leds_v2(panel: np.ndarray, merge_radius: float = 18.0) -> list[dict[str, float]]:
    """改进版: 处理白色过曝区域 + 保留紧挨双点。"""
    b, g, r = cv2.split(panel)
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    
    # 彩色LED
    mask_color = ((v >= 100) & (s >= 40) & (r >= 100) & (g >= 50)).astype(np.uint8) * 255
    # 白色LED（高亮低饱和）
    mask_white = ((v >= 180) & (s < 80)).astype(np.uint8) * 255
    
    mask = cv2.bitwise_or(mask_color, mask_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # 对过大的连通域做分割（白色过曝块可能是多个LED粘连）
    num, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    roi_area = panel.shape[0] * panel.shape[1]
    
    points = []
    # 正常LED点的面积范围估算：根据图片中单个LED约20-40px直径
    # 单点面积约 300-1200, 超过2000的可能是多点粘连
    normal_max_area = 2000
    split_max_area = 15000  # 超过这个就太大了不处理
    
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        
        if area < 6 or area > split_max_area:
            continue
        if w < 2 or h < 2:
            continue
        
        # 圆度检查：真LED点接近圆形，面积/(π/4*max(w,h)²) 应该 > 0.3
        # 排除细长条/不规则碎片
        circularity = area / (3.14159 / 4.0 * max(w, h) ** 2)
        aspect = min(w, h) / max(w, h, 1)
        if circularity < 0.25 or aspect < 0.3:
            continue
            
        if area <= normal_max_area and w <= 60 and h <= 60:
            # 正常大小，直接取中心
            points.append({"x": float(centers[idx][0]), "y": float(centers[idx][1]),
                          "area": float(area), "w": float(w), "h": float(h)})
        elif area > normal_max_area:
            # 过大块：尝试用距离变换+局部峰值分割成多个点
            component_mask = (labels == idx).astype(np.uint8) * 255
            sub_points = split_large_blob(component_mask, x, y, w, h)
            points.extend(sub_points)
    
    points = merge_close_points(points, merge_radius)
    return points


def filter_margin_points(points: list[dict[str, float]], width: int, height: int, margin_ratio: float = 0.02) -> list[dict[str, float]]:
    """去掉太靠边缘的误检点。"""
    left = width * margin_ratio
    right = width * (1.0 - margin_ratio)
    top = height * margin_ratio
    bottom = height * (1.0 - margin_ratio)
    return [p for p in points if left <= p["x"] <= right and top <= p["y"] <= bottom]


def split_large_blob(blob_mask: np.ndarray, ox: int, oy: int, bw: int, bh: int) -> list[dict[str, float]]:
    """对过大的连通域用距离变换分割成多个LED点。"""
    # 裁出局部区域
    region = blob_mask[oy:oy+bh, ox:ox+bw]
    if region.size == 0:
        return []
    
    # 距离变换找峰值
    dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    # 阈值化找种子
    _, peaks = cv2.threshold(dist, dist.max() * 0.4, 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    
    # 连通域找种子中心
    n, lbl, st, ctr = cv2.connectedComponentsWithStats(peaks, 8)
    sub_points = []
    for i in range(1, n):
        cx = float(ctr[i][0]) + ox
        cy = float(ctr[i][1]) + oy
        area = float(st[i][4])
        sub_points.append({"x": cx, "y": cy, "area": area, "w": float(st[i][2]), "h": float(st[i][3])})
    
    # 如果分割失败（只有1个），还是返回整体中心
    if len(sub_points) <= 1:
        cx = ox + bw / 2.0
        cy = oy + bh / 2.0
        return [{"x": cx, "y": cy, "area": float(bw * bh), "w": float(bw), "h": float(bh)}]
    
    return sub_points


# 构建最大投影 + 矫正
print("构建最大投影...")
cap = cv2.VideoCapture(str(VIDEO))
max_proj = None
while True:
    ok, frame = cap.read()
    if not ok: break
    if max_proj is None: max_proj = frame.copy()
    else: np.maximum(max_proj, frame, out=max_proj)
cap.release()

roi = detect_display_roi(max_proj, 40, 0.06)
panel_raw = crop_roi(max_proj, roi)

# 先检测点位得到角度
from cv_pattern_probe import merge_close_points as mcp
points_raw = detect_all_leds_v2(panel_raw, merge_radius=18)

# 行拟合角度
sorted_pts = sorted(points_raw, key=lambda p: p["y"])
rows = []
current_row = [sorted_pts[0]]
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
            k = np.polyfit(xs, ys, 1)[0]
            angles.append(np.degrees(np.arctan(k)))

median_angle = np.median(angles) if angles else 0.0
print(f"倾斜角: {median_angle:.2f}°")

# 矫正
center_rot = (roi.w / 2, roi.h / 2)
M = cv2.getRotationMatrix2D(center_rot, median_angle, 1.0)
cos_a = abs(M[0, 0]); sin_a = abs(M[0, 1])
new_w = int(roi.w * cos_a + roi.h * sin_a) + 20
new_h = int(roi.w * sin_a + roi.h * cos_a) + 20
M[0, 2] += (new_w - roi.w) / 2
M[1, 2] += (new_h - roi.h) / 2

panel_aligned = cv2.warpAffine(panel_raw, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
gray_a = cv2.cvtColor(panel_aligned, cv2.COLOR_BGR2GRAY)
rows_mask = gray_a.max(axis=1) > 10
cols_mask = gray_a.max(axis=0) > 10
if rows_mask.any() and cols_mask.any():
    y1, y2 = np.where(rows_mask)[0][[0, -1]]
    x1, x2 = np.where(cols_mask)[0][[0, -1]]
    panel_aligned = panel_aligned[y1:y2+1, x1:x2+1]

# 最终检测
points_final = detect_all_leds_v2(panel_aligned, merge_radius=18)
print(f"最终检测: {len(points_final)} 点")

annotated = panel_aligned.copy()
for i, pt in enumerate(sorted(points_final, key=lambda p: (p["x"], p["y"]))):
    cx, cy = int(pt["x"]), int(pt["y"])
    cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), 2)
    cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
cv2.putText(annotated, f"v2: {len(points_final)}pts angle={median_angle:.1f}",
           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
imwrite_unicode(OUT_DIR / f"aligned_v2_{len(points_final)}pts.jpg", annotated)

print(f"输出: {OUT_DIR / f'aligned_v2_{len(points_final)}pts.jpg'}")
