"""修正：降低饱和度阈值以检出白色LED。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from pathlib import Path
from cv_pattern_probe import detect_led_points, imread_unicode, imwrite_unicode

TEMPLATES_DIR = Path("output/sevenseg_frame_templates/Video_20260529135514029/templates")
OUT_DIR = Path("output/dot_panel_cvpattern_test")

images = {
    "T001_full_lit": TEMPLATES_DIR / "T001_f0000.jpg",
    "T018_partial": TEMPLATES_DIR / "T018_f0300.jpg",
}

for img_name, img_path in images.items():
    image = imread_unicode(img_path)
    panel = image[45:, :]  # 裁掉标注
    
    # 饱和度降到10，让白色LED也能过
    points, mask = detect_led_points(
        panel,
        value_threshold=80,
        saturation_threshold=10,
        red_threshold=60,
        green_threshold=40,
        merge_radius=25,
    )
    print(f"{img_name}: {len(points)} 点")
    
    annotated = panel.copy()
    for pt in sorted(points, key=lambda p: (p["x"], p["y"])):
        cx, cy = int(pt["x"]), int(pt["y"])
        cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
    cv2.putText(annotated, f"{len(points)} points", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    imwrite_unicode(OUT_DIR / f"{img_name}_sat10_{len(points)}pts.jpg", annotated)

print(f"\n输出: {OUT_DIR.resolve()}")
