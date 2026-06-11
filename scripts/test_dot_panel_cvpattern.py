"""
用 cv_pattern_probe.detect_led_points 对新点阵面板测试效果。
输入: T001_f0000.jpg (全亮) 和 T018_f0300.jpg (部分亮)
"""
from __future__ import annotations
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from pathlib import Path
from cv_pattern_probe import detect_led_points, imread_unicode, imwrite_unicode

TEMPLATES_DIR = Path("output/sevenseg_frame_templates/Video_20260529135514029/templates")
OUT_DIR = Path("output/dot_panel_cvpattern_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

images = {
    "T001_full_lit": TEMPLATES_DIR / "T001_f0000.jpg",
    "T018_partial": TEMPLATES_DIR / "T018_f0300.jpg",
}

# 尝试不同参数
param_sets = {
    "default": {"value_threshold": 120, "saturation_threshold": 70, "red_threshold": 120, "green_threshold": 60, "merge_radius": 14.0},
    "low_thresh": {"value_threshold": 80, "saturation_threshold": 40, "red_threshold": 80, "green_threshold": 40, "merge_radius": 14.0},
    "very_low": {"value_threshold": 60, "saturation_threshold": 30, "red_threshold": 60, "green_threshold": 30, "merge_radius": 20.0},
}

for img_name, img_path in images.items():
    print(f"\n{'='*60}")
    print(f"{img_name}: {img_path.name}")
    
    image = imread_unicode(img_path)
    if image is None:
        print("  无法读取")
        continue
    h, w = image.shape[:2]
    print(f"  尺寸: {w}x{h}")
    
    for pname, params in param_sets.items():
        points, mask = detect_led_points(image, **params)
        print(f"  [{pname}]: {len(points)} 点")
        
        # 画标注
        annotated = image.copy()
        for pt in points:
            cx, cy = int(pt["x"]), int(pt["y"])
            cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
        
        # 加点数标注
        cv2.putText(annotated, f"{pname}: {len(points)} points", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        out_name = f"{img_name}_{pname}_{len(points)}pts.jpg"
        imwrite_unicode(OUT_DIR / out_name, annotated)
        
        # 保存mask
        if pname == "low_thresh":
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            imwrite_unicode(OUT_DIR / f"{img_name}_mask.jpg", mask_bgr)

print(f"\n输出: {OUT_DIR.resolve()}")
