"""先对面板做旋转矫正再检测点位。
用最大投影找面板轮廓，拟合最小旋转矩形得到角度，旋转矫正后再裁ROI。
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from pathlib import Path
from cv_pattern_probe import merge_close_points, imread_unicode, imwrite_unicode

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


# Step 1: 从视频构建最大投影
print("构建最大投影...")
cap = cv2.VideoCapture(str(VIDEO))
max_proj = None
for i in range(200):
    ok, frame = cap.read()
    if not ok:
        break
    if max_proj is None:
        max_proj = frame.copy()
    else:
        np.maximum(max_proj, frame, out=max_proj)
cap.release()

h, w = max_proj.shape[:2]
print(f"视频帧: {w}x{h}")

# Step 2: 找面板轮廓，拟合旋转矩形得到倾斜角度
gray = cv2.cvtColor(max_proj, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 取最大轮廓
contours = sorted(contours, key=cv2.contourArea, reverse=True)
if contours:
    largest = contours[0]
    rect = cv2.minAreaRect(largest)
    center, (rw, rh), angle = rect
    # minAreaRect 的角度：如果宽<高，角度是相对水平的旋转
    # 我们需要让长边水平
    if rw < rh:
        angle += 90
    print(f"旋转矩形: center=({center[0]:.0f},{center[1]:.0f}) size={rw:.0f}x{rh:.0f} angle={angle:.2f}°")
    
    # Step 3: 旋转矫正
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(max_proj, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    
    # Step 4: 在矫正后的图上重新找 ROI
    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, bin_rot = cv2.threshold(gray_rot, 30, 255, cv2.THRESH_BINARY)
    bin_rot = cv2.morphologyEx(bin_rot, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    
    contours_rot, _ = cv2.findContours(bin_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_rot = sorted(contours_rot, key=cv2.contourArea, reverse=True)
    
    if contours_rot:
        x, y, bw, bh = cv2.boundingRect(contours_rot[0])
        # 加点padding
        pad = 10
        x = max(0, x - pad)
        y = max(0, y - pad)
        bw = min(w - x, bw + 2*pad)
        bh = min(h - y, bh + 2*pad)
        
        panel = rotated[y:y+bh, x:x+bw]
        print(f"矫正后ROI: x={x} y={y} w={bw} h={bh}")
        
        # Step 5: 检测点位
        points = detect_all_leds(panel, merge_radius=25)
        print(f"检测到: {len(points)} 点")
        
        # 标注
        annotated = panel.copy()
        for pt in sorted(points, key=lambda p: (p["x"], p["y"])):
            cx, cy = int(pt["x"]), int(pt["y"])
            cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
        cv2.putText(annotated, f"aligned: {len(points)} pts, angle={angle:.1f}deg",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        imwrite_unicode(OUT_DIR / f"T001_aligned_{len(points)}pts.jpg", annotated)
        imwrite_unicode(OUT_DIR / "max_proj_rotated.jpg", rotated)
        
        # 也对视频第一帧做同样矫正
        cap = cv2.VideoCapture(str(VIDEO))
        ok, frame0 = cap.read()
        cap.release()
        if ok:
            frame0_rot = cv2.warpAffine(frame0, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            panel0 = frame0_rot[y:y+bh, x:x+bw]
            points0 = detect_all_leds(panel0, merge_radius=25)
            annotated0 = panel0.copy()
            for pt in points0:
                cx, cy = int(pt["x"]), int(pt["y"])
                cv2.circle(annotated0, (cx, cy), 10, (0, 255, 0), 2)
            cv2.putText(annotated0, f"frame0 aligned: {len(points0)} pts",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            imwrite_unicode(OUT_DIR / f"frame0_aligned_{len(points0)}pts.jpg", annotated0)
            print(f"Frame 0 检测: {len(points0)} 点")
else:
    print("未找到面板轮廓！")

print(f"\n输出: {OUT_DIR.resolve()}")
