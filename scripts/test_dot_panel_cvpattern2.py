"""加大 merge_radius 解决一个点检成两个的问题。"""
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

# 加大 merge_radius: 试 25, 30, 35, 40
for img_name, img_path in images.items():
    image = imread_unicode(img_path)
    if image is None:
        continue
    h, w = image.shape[:2]
    print(f"\n{img_name} ({w}x{h}):")
    
    for radius in [20, 25, 30, 35, 40]:
        points, mask = detect_led_points(
            image,
            value_threshold=80,
            saturation_threshold=40,
            red_threshold=80,
            green_threshold=40,
            merge_radius=radius,
        )
        print(f"  merge_radius={radius}: {len(points)} 点")
        
        if radius == 30:
            annotated = image.copy()
            for i, pt in enumerate(points):
                cx, cy = int(pt["x"]), int(pt["y"])
                cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
                cv2.putText(annotated, str(i+1), (cx+12, cy+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(annotated, f"merge_radius={radius}: {len(points)} pts", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            imwrite_unicode(OUT_DIR / f"{img_name}_r{radius}_{len(points)}pts.jpg", annotated)

print(f"\n输出: {OUT_DIR.resolve()}")
