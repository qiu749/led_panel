"""用亮点坐标拟合直线得到倾斜角度，再矫正。"""
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
OUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_all_leds(panel: np.ndarray, merge_radius: float = 25.0) -> list[dict[str, float]]:
    b, g, r = cv2.split(panel)
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask_color = ((v >= 80) & (s >= 40) & (r >= 80) & (g >= 40)).astype(np.uint8) * 255
    mask_white = ((v >= 180) & (s < 60)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_color, mask_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    roi_area = panel.shape[0] * panel.shape[1]
    max_area = max(800, roi_area * 0.02)
    points = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if 6 <= area <= max_area and 2 <= w <= 50 and 2 <= h <= 50:
            points.append({"x": float(centers[idx][0]), "y": float(centers[idx][1]),
                          "area": float(area), "w": float(w), "h": float(h)})
    return merge_close_points(points, merge_radius)


# 构建最大投影
print("构建最大投影...")
cap = cv2.VideoCapture(str(VIDEO))
max_proj = None
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if max_proj is None:
        max_proj = frame.copy()
    else:
        np.maximum(max_proj, frame, out=max_proj)
cap.release()
h, w = max_proj.shape[:2]
print(f"画面: {w}x{h}")

# 用detect_display_roi得到大致ROI
roi = detect_display_roi(max_proj, 40, 0.06)
print(f"ROI: x={roi.x} y={roi.y} w={roi.w} h={roi.h}")

# 在ROI内检测点位
panel_raw = crop_roi(max_proj, roi)
points_raw = detect_all_leds(panel_raw, merge_radius=25)
print(f"矫正前检测: {len(points_raw)} 点")

if len(points_raw) >= 5:
    # 用点坐标拟合直线（行方向），得到倾斜角度
    # 对点按y聚类为行，取每行的左右端点拟合
    pts_array = np.array([(p["x"], p["y"]) for p in points_raw])
    
    # 拟合一条直线 y = kx + b 用所有点
    # 或更好：用 PCA 主方向
    mean = pts_array.mean(axis=0)
    centered = pts_array - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 主方向（最大特征值对应的特征向量）
    main_dir = eigenvectors[:, 1]  # 第二个（较大的特征值）
    angle_rad = np.arctan2(main_dir[1], main_dir[0])
    angle_deg = np.degrees(angle_rad)
    print(f"点阵主方向角度: {angle_deg:.2f}°")
    
    # 也尝试按行聚类后用行中心拟合
    sorted_by_y = sorted(points_raw, key=lambda p: p["y"])
    rows = []
    current_row = [sorted_by_y[0]]
    for p in sorted_by_y[1:]:
        if p["y"] - current_row[-1]["y"] < 30:
            current_row.append(p)
        else:
            rows.append(current_row)
            current_row = [p]
    rows.append(current_row)
    
    print(f"聚类得到 {len(rows)} 行")
    for i, row in enumerate(rows):
        xs = [p["x"] for p in row]
        ys = [p["y"] for p in row]
        print(f"  行{i}: {len(row)}点, y_mean={np.mean(ys):.0f}, x_range=[{min(xs):.0f}-{max(xs):.0f}]")
    
    # 用各行中心点拟合倾斜角
    if len(rows) >= 2:
        row_centers = [(np.mean([p["x"] for p in r]), np.mean([p["y"] for p in r])) for r in rows if len(r) >= 3]
        if len(row_centers) >= 2:
            rc = np.array(row_centers)
            # 对每行内的点做线性回归得到行倾斜角
            angles = []
            for row in rows:
                if len(row) >= 3:
                    xs = np.array([p["x"] for p in row])
                    ys = np.array([p["y"] for p in row])
                    if xs.max() - xs.min() > 50:  # 行足够宽
                        k = np.polyfit(xs, ys, 1)[0]
                        angles.append(np.degrees(np.arctan(k)))
            if angles:
                median_angle = np.median(angles)
                print(f"行拟合倾斜角: {median_angle:.2f}° (来自{len(angles)}行)")
                
                # 矫正
                center_rot = (roi.w / 2, roi.h / 2)
                M = cv2.getRotationMatrix2D(center_rot, median_angle, 1.0)
                
                # 计算旋转后的画布大小
                cos_a = abs(M[0, 0])
                sin_a = abs(M[0, 1])
                new_w = int(roi.w * cos_a + roi.h * sin_a) + 20
                new_h = int(roi.w * sin_a + roi.h * cos_a) + 20
                M[0, 2] += (new_w - roi.w) / 2
                M[1, 2] += (new_h - roi.h) / 2
                
                panel_aligned = cv2.warpAffine(panel_raw, M, (new_w, new_h), 
                                               flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
                
                # 裁掉黑边
                gray_a = cv2.cvtColor(panel_aligned, cv2.COLOR_BGR2GRAY)
                rows_mask = gray_a.max(axis=1) > 10
                cols_mask = gray_a.max(axis=0) > 10
                if rows_mask.any() and cols_mask.any():
                    y1, y2 = np.where(rows_mask)[0][[0, -1]]
                    x1, x2 = np.where(cols_mask)[0][[0, -1]]
                    panel_aligned = panel_aligned[y1:y2+1, x1:x2+1]
                
                # 再检测点位
                points_aligned = detect_all_leds(panel_aligned, merge_radius=25)
                print(f"矫正后检测: {len(points_aligned)} 点")
                
                annotated = panel_aligned.copy()
                for pt in points_aligned:
                    cx, cy = int(pt["x"]), int(pt["y"])
                    cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), 2)
                    cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
                cv2.putText(annotated, f"aligned: {len(points_aligned)}pts angle={median_angle:.1f}deg",
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                imwrite_unicode(OUT_DIR / f"aligned_final_{len(points_aligned)}pts.jpg", annotated)
                
                # 保存矫正参数
                print(f"\n矫正参数: angle={median_angle:.2f}°, ROI=({roi.x},{roi.y},{roi.w},{roi.h})")

print(f"\n输出: {OUT_DIR.resolve()}")
