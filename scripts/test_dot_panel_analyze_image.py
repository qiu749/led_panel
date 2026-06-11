"""分析T001原图的像素分布来理解面板结构。"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from pathlib import Path
from cv_pattern_probe import imread_unicode

image = imread_unicode(Path("output/sevenseg_frame_templates/Video_20260529135514029/templates/T001_f0000.jpg"))
h, w = image.shape[:2]
print(f"图片尺寸: {w}x{h}")

# 注意: 图片顶部有模板标注文字 "T001 f0" 黄色，大概30px高
# 实际面板内容从y=30开始

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
v = hsv[:,:,2]
s = hsv[:,:,1]

# 按列看亮度分布（看左边是不是有LED）
print("\n按列平均亮度(V通道), 每100px一个采样:")
for x in range(0, w, 100):
    region = v[40:, x:x+100]  # 跳过顶部标注
    if region.size > 0:
        mean_v = region.mean()
        max_v = region.max()
        bright = (region > 80).sum() / region.size
        print(f"  x={x:4d}-{x+100:4d}: mean_v={mean_v:.1f} max_v={max_v} bright_ratio={bright:.3f}")

# 看左边400px区域的详细信息
left = image[40:, :400]
left_gray = gray[40:, :400]
left_v = v[40:, :400]
print(f"\n左区(x<400, y>40):")
print(f"  mean_gray={left_gray.mean():.1f} max_gray={left_gray.max()}")
print(f"  mean_v={left_v.mean():.1f} max_v={left_v.max()}")
print(f"  bright_pixels(v>80): {(left_v > 80).sum()} / {left_v.size} = {(left_v > 80).mean():.4f}")

# 对比右边
right = image[40:, 1600:]
right_v = v[40:, 1600:]
print(f"\n右区(x>1600, y>40):")
print(f"  mean_v={right_v.mean():.1f} max_v={right_v.max()}")
print(f"  bright_pixels(v>80): {(right_v > 80).sum()} / {right_v.size} = {(right_v > 80).mean():.4f}")

# 分析: 左边是否有点阵LED但颜色不同/亮度不够?
# 看B/G/R通道
b, g, r = cv2.split(left)
print(f"\n左区 RGB:")
print(f"  R: mean={r.mean():.1f} max={r.max()}")
print(f"  G: mean={g.mean():.1f} max={g.max()}")
print(f"  B: mean={b.mean():.1f} max={b.max()}")
