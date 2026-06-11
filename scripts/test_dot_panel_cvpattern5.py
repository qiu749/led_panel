"""两次检测合并：彩色LED + 白色LED。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from pathlib import Path
from cv_pattern_probe import merge_close_points, imread_unicode, imwrite_unicode

TEMPLATES_DIR = Path("output/sevenseg_frame_templates/Video_20260529135514029/templates")
OUT_DIR = Path("output/dot_panel_cvpattern_test")


def detect_all_leds(panel: np.ndarray, merge_radius: float = 25.0) -> list[dict[str, float]]:
    """检测所有LED点：彩色 + 白色，合并去重。"""
    b, g, r = cv2.split(panel)
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    
    # Mask 1: 彩色LED（有饱和度有亮度）
    mask_color = (
        (v >= 80) & (s >= 40) & (r >= 80) & (g >= 40)
    ).astype(np.uint8) * 255
    
    # Mask 2: 白色LED（高亮度，低饱和度）
    mask_white = (
        (v >= 180) & (s < 60)
    ).astype(np.uint8) * 255
    
    # 合并
    mask = cv2.bitwise_or(mask_color, mask_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # 连通域
    num, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    roi_area = panel.shape[0] * panel.shape[1]
    max_area = max(800, roi_area * 0.02)
    
    points: list[dict[str, float]] = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if 6 <= area <= max_area and 2 <= w <= 50 and 2 <= h <= 50:
            points.append({
                "x": float(centers[idx][0]),
                "y": float(centers[idx][1]),
                "area": float(area),
                "w": float(w),
                "h": float(h),
            })
    
    return merge_close_points(points, merge_radius)


images = {
    "T001_full_lit": TEMPLATES_DIR / "T001_f0000.jpg",
    "T018_partial": TEMPLATES_DIR / "T018_f0300.jpg",
}

for img_name, img_path in images.items():
    image = imread_unicode(img_path)
    panel = image[45:, :]
    
    points = detect_all_leds(panel, merge_radius=25)
    print(f"{img_name}: {len(points)} 点")
    
    annotated = panel.copy()
    for pt in sorted(points, key=lambda p: (p["x"], p["y"])):
        cx, cy = int(pt["x"]), int(pt["y"])
        cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 2, (0, 0, 255), -1)
    cv2.putText(annotated, f"{len(points)} points", (20, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    imwrite_unicode(OUT_DIR / f"{img_name}_dual_{len(points)}pts.jpg", annotated)

print(f"\n输出: {OUT_DIR.resolve()}")
