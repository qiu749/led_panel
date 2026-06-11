"""
测试新型点阵面板的 LED 点检测效果。

用 T001_f0000.jpg (全亮) 和 T018_f0300.jpg (部分亮) 做测试，
看现有的点检测算法能否识别出 LED 点位。

输出:
  - 标注图（红圈标记检测到的点）
  - 点坐标列表
  - 检测数量统计
"""
from __future__ import annotations
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import cv2
import numpy as np
from pathlib import Path
from try_overexposed_point_detection import Params, detect_points, imread_unicode, imwrite_unicode
from try_overexposed_point_detection_v4_filtered import FilterParams, filter_points

TEMPLATES_DIR = Path("output/sevenseg_frame_templates/Video_20260529135514029/templates")
OUT_DIR = Path("output/dot_panel_detection_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 两张模板图
images = {
    "T001_full_lit": TEMPLATES_DIR / "T001_f0000.jpg",
    "T018_partial": TEMPLATES_DIR / "T018_f0300.jpg",
}

# 尝试不同参数组合
param_sets = {
    "default": Params(),
    "sensitive": Params(
        threshold_percentile=94.0,
        threshold_std_factor=0.9,
        min_threshold=8.0,
        min_peak_distance_px=6,
        min_center_contrast=2.5,
        min_peak_contrast=6.0,
        min_white_core_percentile=40.0,
    ),
    "very_sensitive": Params(
        threshold_percentile=90.0,
        threshold_std_factor=0.8,
        min_threshold=5.0,
        min_peak_distance_px=5,
        min_center_contrast=1.5,
        min_peak_contrast=4.0,
        min_white_core_percentile=30.0,
        target_width=1024,
    ),
}

for img_name, img_path in images.items():
    print(f"\n{'='*60}")
    print(f"图片: {img_name} ({img_path.name})")
    print(f"{'='*60}")
    
    image = imread_unicode(img_path)
    if image is None:
        print(f"  无法读取: {img_path}")
        continue
    
    h, w = image.shape[:2]
    print(f"  尺寸: {w}x{h}")
    
    for param_name, params in param_sets.items():
        result = detect_points(image, params)
        points = result["points"]
        print(f"\n  参数[{param_name}]: 检测到 {len(points)} 个点")
        
        if points:
            # 过滤
            filter_params = FilterParams(
                row_cluster_tolerance=h * 0.05,
                row_keep_tolerance=h * 0.04,
                neighbor_y_tolerance=h * 0.04,
                neighbor_min_dx=3.0,
                neighbor_max_dx=w * 0.15,
            )
            filtered_result = filter_points(points, w, h, filter_params)
            filtered_points = filtered_result.get("filtered", points)
            print(f"    过滤后: {len(filtered_points)} 个点")
            
            # 画标注图
            annotated = image.copy()
            scale_x = w / params.target_width
            scale_y = h / max(params.min_height, min(params.max_height, int(round(h * params.target_width / max(w, 1)))))
            
            for pt in points:
                x = int(pt["x"] * scale_x)
                y = int(pt["y"] * scale_y)
                cv2.circle(annotated, (x, y), 8, (0, 0, 255), 2)
            
            for pt in filtered_points:
                x = int(pt["x"] * scale_x)
                y = int(pt["y"] * scale_y)
                cv2.circle(annotated, (x, y), 12, (0, 255, 0), 2)
            
            out_name = f"{img_name}_{param_name}_pts{len(points)}_flt{len(filtered_points)}.jpg"
            imwrite_unicode(OUT_DIR / out_name, annotated)
            print(f"    保存: {out_name}")
    
    # 也试试简单的 threshold + contour 方法（对点阵面板可能更直接）
    print(f"\n  === 简单阈值法 ===")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for thresh_val in [80, 100, 120, 150]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        # 开运算去噪
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 过滤太大和太小的
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 2000:  # LED点的面积范围
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy, area))
        
        print(f"    threshold={thresh_val}: {len(dots)} 个点 (面积20-2000)")
        
        if dots and thresh_val == 100:
            annotated = image.copy()
            for cx, cy, area in dots:
                cv2.circle(annotated, (cx, cy), 8, (255, 0, 255), 2)
            out_name = f"{img_name}_threshold{thresh_val}_dots{len(dots)}.jpg"
            imwrite_unicode(OUT_DIR / out_name, annotated)
            print(f"    保存: {out_name}")

print(f"\n\n输出目录: {OUT_DIR.resolve()}")
