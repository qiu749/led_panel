"""分析T001左边区域的检测问题。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from pathlib import Path
from cv_pattern_probe import detect_led_points, imread_unicode, imwrite_unicode

TEMPLATES_DIR = Path("output/sevenseg_frame_templates/Video_20260529135514029/templates")
OUT_DIR = Path("output/dot_panel_cvpattern_test")

image = imread_unicode(TEMPLATES_DIR / "T001_f0000.jpg")
h, w = image.shape[:2]
print(f"全图: {w}x{h}")

# 检测全图
points, mask = detect_led_points(image, value_threshold=80, saturation_threshold=40,
                                  red_threshold=80, green_threshold=40, merge_radius=30)

# 按x坐标分区看
left_pts = [p for p in points if p["x"] < w * 0.25]
mid_pts = [p for p in points if w * 0.25 <= p["x"] < w * 0.75]
right_pts = [p for p in points if p["x"] >= w * 0.75]
print(f"左区(x<{w*0.25:.0f}): {len(left_pts)} 点")
print(f"中区: {len(mid_pts)} 点")
print(f"右区(x>{w*0.75:.0f}): {len(right_pts)} 点")

# 列出左区点的详细信息
print(f"\n左区点详情:")
for i, pt in enumerate(sorted(left_pts, key=lambda p: (p["y"], p["x"]))):
    print(f"  {i+1}: x={pt['x']:.0f} y={pt['y']:.0f} area={pt['area']:.0f} w={pt['w']:.0f} h={pt['h']:.0f}")

# 保存左区裁切+标注
left_crop = image[:, :int(w*0.3)].copy()
for pt in left_pts:
    cx, cy = int(pt["x"]), int(pt["y"])
    cv2.circle(left_crop, (cx, cy), 12, (0, 255, 0), 2)
    cv2.putText(left_crop, f"{pt['area']:.0f}", (cx+14, cy+5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# 保存mask的左区
mask_left = mask[:, :int(w*0.3)]
imwrite_unicode(OUT_DIR / "T001_left_annotated.jpg", left_crop)
imwrite_unicode(OUT_DIR / "T001_left_mask.jpg", cv2.cvtColor(mask_left, cv2.COLOR_GRAY2BGR))
print(f"\n保存: T001_left_annotated.jpg, T001_left_mask.jpg")
