"""用全视频最大投影检测面板倾斜角度并矫正。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from pathlib import Path
from cv_pattern_probe import merge_close_points, imwrite_unicode

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


# 用全视频最大投影
print("构建全视频最大投影...")
cap = cv2.VideoCapture(str(VIDEO))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
max_proj = None
idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if max_proj is None:
        max_proj = frame.copy()
    else:
        np.maximum(max_proj, frame, out=max_proj)
    idx += 1
cap.release()
print(f"读取 {idx} 帧, 画面 {max_proj.shape[1]}x{max_proj.shape[0]}")

h, w = max_proj.shape[:2]
imwrite_unicode(OUT_DIR / "full_max_proj.jpg", cv2.resize(max_proj, (w//3, h//3)))

# 找面板亮区
gray = cv2.cvtColor(max_proj, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

if not contours:
    print("未找到面板！")
    sys.exit(1)

# 用最大轮廓的最小旋转矩形
largest = contours[0]
area = cv2.contourArea(largest)
print(f"最大轮廓面积: {area:.0f} (画面面积: {w*h})")

rect = cv2.minAreaRect(largest)
center, (rw, rh), angle = rect

# 确保长边水平
if rw < rh:
    angle += 90
    rw, rh = rh, rw

print(f"旋转矩形: center=({center[0]:.0f},{center[1]:.0f}) size={rw:.0f}x{rh:.0f} angle={angle:.2f}°")

# 矫正旋转
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(max_proj, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

# 在矫正后图上找ROI
gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
_, bin_rot = cv2.threshold(gray_rot, 40, 255, cv2.THRESH_BINARY)
bin_rot = cv2.morphologyEx(bin_rot, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
contours_rot, _ = cv2.findContours(bin_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_rot = sorted(contours_rot, key=cv2.contourArea, reverse=True)

x, y, bw, bh = cv2.boundingRect(contours_rot[0])
pad = 15
x = max(0, x - pad)
y = max(0, y - pad)
bw = min(w - x, bw + 2*pad)
bh = min(h - y, bh + 2*pad)

panel = rotated[y:y+bh, x:x+bw]
print(f"矫正后ROI: x={x} y={y} w={bw} h={bh}")

# 检测点位
points = detect_all_leds(panel, merge_radius=25)
print(f"全亮投影检测: {len(points)} 点")

# 标注
annotated = panel.copy()
for pt in sorted(points, key=lambda p: (p["x"], p["y"])):
    cx, cy = int(pt["x"]), int(pt["y"])
    cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), 2)
    cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
cv2.putText(annotated, f"aligned {len(points)}pts angle={angle:.1f}",
           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
imwrite_unicode(OUT_DIR / f"aligned_maxproj_{len(points)}pts.jpg", annotated)

# 对frame 0和frame 300也做矫正
cap = cv2.VideoCapture(str(VIDEO))
for target_frame, label in [(0, "f000"), (300, "f300")]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    if not ok:
        continue
    frame_rot = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    frame_panel = frame_rot[y:y+bh, x:x+bw]
    pts = detect_all_leds(frame_panel, merge_radius=25)
    ann = frame_panel.copy()
    for pt in pts:
        cx, cy = int(pt["x"]), int(pt["y"])
        cv2.circle(ann, (cx, cy), 10, (0, 255, 0), 2)
    cv2.putText(ann, f"{label} aligned: {len(pts)} pts", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    imwrite_unicode(OUT_DIR / f"aligned_{label}_{len(pts)}pts.jpg", ann)
    print(f"  {label}: {len(pts)} 点")
cap.release()

print(f"\n输出: {OUT_DIR.resolve()}")
