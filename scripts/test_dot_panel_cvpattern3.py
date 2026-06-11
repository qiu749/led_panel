"""修正：裁掉标注区域，只处理面板有效区域。"""
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
    h, w = image.shape[:2]
    
    # 裁掉顶部标注(40px)，只保留面板区域
    panel = image[45:, :]
    ph, pw = panel.shape[:2]
    
    points, mask = detect_led_points(
        panel,
        value_threshold=80,
        saturation_threshold=40,
        red_threshold=80,
        green_threshold=40,
        merge_radius=25,
    )
    print(f"{img_name}: {len(points)} 点 (裁掉标注后)")
    
    # 标注
    annotated = panel.copy()
    for i, pt in enumerate(sorted(points, key=lambda p: (p["x"], p["y"]))):
        cx, cy = int(pt["x"]), int(pt["y"])
        cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
    cv2.putText(annotated, f"{len(points)} points detected", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    imwrite_unicode(OUT_DIR / f"{img_name}_fixed_{len(points)}pts.jpg", annotated)

print(f"\n输出: {OUT_DIR.resolve()}")
