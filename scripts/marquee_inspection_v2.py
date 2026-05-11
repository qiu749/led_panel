"""
跑马灯质检 v2 — 网格感知版
核心改进：用亮度累积热力图 + 投影峰值检测替代连通域聚类，自动发现LED网格结构。

用法：
  python scripts/marquee_inspection_v2.py learn --frames <frames_dir> --output-dir <dir>
  python scripts/marquee_inspection_v2.py detect --frames <frames_dir> --template <template.json> --output-dir <dir>
  python scripts/marquee_inspection_v2.py run    --frames <frames_dir> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_PANEL_SIZE = (420, 720)
DEFAULT_OUTPUT_DIR = Path("output") / "marquee_inspection_v2"


@dataclass
class Slot:
    id: int
    x: float
    y: float
    w: float
    h: float
    threshold: float
    peak: float = 0.0
    duty: float = 0.0


# ──────────────────── 工具函数 ────────────────────


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_frames(frames_dir: Path, stride: int = 1) -> tuple[list[np.ndarray], float]:
    """从图片目录加载帧，按文件名排序。返回帧列表和估算fps（默认15）。"""
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files: list[Path] = []
    for p in patterns:
        files.extend(sorted(glob.glob(str(frames_dir / p))))
    if not files:
        raise RuntimeError(f"未在 {frames_dir} 中找到任何图片帧")
    frames: list[np.ndarray] = []
    for i, fp in enumerate(files):
        if i % stride != 0:
            continue
        img = cv2.imread(fp)
        if img is None:
            continue
        frames.append(img)
    if not frames:
        raise RuntimeError(f"无法读取任何有效帧")
    return frames, 15.0  # 默认fps=15


def detect_panel_bbox(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """面板bbox检测：自动选择暗区检测（黑面板）或亮区检测（白面板）。"""
    dark_result = _detect_dark_panel_bbox(frame)
    bright_result = _detect_bright_panel_bbox(frame)

    # 如果亮区检测结果更合理（面积更大且更居中），优先使用
    if bright_result:
        return bright_result
    return dark_result


def _detect_dark_panel_bbox(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """暗区检测：适用于黑色/深色面板。原始v1/v2逻辑。"""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    dark = (blurred < 100).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((23, 23), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1.0
    best_bbox: tuple[int, int, int, int] | None = None
    frame_area = float(width * height)
    cx_target = width * 0.5
    cy_target = height * 0.5

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < frame_area * 0.03 or area > frame_area * 0.55:
            continue
        aspect_ratio = h / max(w, 1)
        if not 0.3 <= aspect_ratio <= 4.0:
            continue
        extent = cv2.contourArea(contour) / max(area, 1.0)
        ccx = x + w * 0.5
        ccy = y + h * 0.5
        dist = math.hypot((ccx - cx_target) / width, (ccy - cy_target) / height)
        centrality = max(0.0, 1.0 - dist * 2.0)
        edge_penalty = 0.9 if x <= 4 or y <= 4 or x + w >= width - 4 or y + h >= height - 4 else 0.0
        score = area / frame_area + 0.45 * extent + 0.25 * centrality - edge_penalty
        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h)
    return best_bbox


def _detect_bright_panel_bbox(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    亮区检测：适用于白色/浅色面板。
    策略：高亮度区域 + 边缘轮廓检测，找最大的明亮矩形区域。
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 高斯模糊减少噪点
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 亮区：灰度值高于全局均值的区域
    global_mean = float(np.mean(blurred))
    bright_threshold = max(global_mean * 1.05, 140)  # 比均值稍高
    bright = (blurred >= bright_threshold).astype(np.uint8) * 255

    # 形态学清理：闭操作连接缺口，开操作去小噪点
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -1.0
    best_bbox: tuple[int, int, int, int] | None = None
    frame_area = float(width * height)
    cx_target = width * 0.55  # 面板偏右（根据实际画面）
    cy_target = height * 0.60

    for contour in contours:
        x, y, w, h_rect = cv2.boundingRect(contour)
        area = float(w * h_rect)
        # 面板面积应在合理范围
        if area < frame_area * 0.08 or area > frame_area * 0.65:
            continue
        # 宽高比检查：面板通常是竖长方形或正方形
        aspect = h_rect / max(w, 1)
        if not 0.6 <= aspect <= 4.0:
            continue

        # 填充率（实心程度）
        extent = cv2.contourArea(contour) / max(area, 1.0)
        if extent < 0.45:
            continue  # 太空洞，不是面板

        ccx = x + w * 0.5
        ccy = y + h_rect * 0.5
        dist = math.hypot((ccx - cx_target) / width, (ccy - cy_target) / height)
        centrality = max(0.0, 1.0 - dist * 2.5)

        # 偏好更大的、更居中的、更高填充率的
        score = (area / frame_area) * 2.0 + extent * 1.5 + centrality * 1.0

        # 轻微边缘惩罚
        margin = 10
        if x <= margin or y <= margin or x + w >= width - margin or y + h_rect >= height - margin:
            score -= 0.15

        if score > best_score:
            best_score = score
            best_bbox = (x, y, w, h_rect)

    return best_bbox


def median_panel_bbox(frames: list[np.ndarray], sample_count: int = 20) -> tuple[int, int, int, int]:
    """多帧取中值bbox。"""
    boxes = []
    indices = np.linspace(0, len(frames) - 1, min(sample_count, len(frames)), dtype=int)
    for idx in indices:
        bbox = detect_panel_bbox(frames[int(idx)])
        if bbox:
            boxes.append(bbox)
    if not boxes:
        h, w = frames[0].shape[:2]
        mx = int(w * 0.15)
        my = int(h * 0.12)
        return mx, my, w - 2 * mx, h - 2 * my
    arr = np.asarray(boxes, dtype=np.float32)
    x, y, bw, bh = np.median(arr, axis=0)
    return int(round(x)), int(round(y)), int(round(bw)), int(round(bh))


def align_panel(frame: np.ndarray, bbox: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    """裁剪+resize面板到标准尺寸。"""
    pw, ph = size
    ih, iw = frame.shape[:2]
    x, y, bw, bh = bbox
    x = max(0, min(iw - 1, int(x)))
    y = max(0, min(ih - 1, int(y)))
    bw = max(1, min(iw - x, int(bw)))
    bh = max(1, min(ih - y, int(bh)))
    crop = frame[y : y + bh, x : x + bw]
    return cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_LINEAR)


def align_all_frames(frames: list[np.ndarray], bbox: tuple[int, int, int, int],
                     panel_size: tuple[int, int] = DEFAULT_PANEL_SIZE) -> list[np.ndarray]:
    """所有帧对齐到标准面板尺寸。"""
    return [align_panel(f, bbox, panel_size) for f in frames]


# ─────────────── v2 核心：网格感知灯位学习 ────────────────


def auto_crop_to_active_region(heatmap: np.ndarray,
                               brightness_pct: float = 85.0,
                               min_area_ratio: float = 0.03,
                               max_area_ratio: float = 0.40) -> tuple[int, int, int, int]:
    """
    从方差热力图中自动裁剪到LED活跃区域。
    方差热力图特性：静态背景≈0，闪烁LED>0，手部移动产生大面积低方差区。
    策略：高百分位阈值 + 形状紧密度过滤 + 面积上限。
    返回 (x, y, w, h) 相对热力图的裁剪bbox。
    """
    h, w = heatmap.shape
    # 方差图大部分像素接近0（静态背景），用高百分位只保留真正有变化的区域
    thresh_val = float(np.percentile(heatmap, brightness_pct))

    # 如果整体方差太低（面板完全没变化），fallback到全图
    global_std = float(np.std(heatmap))
    if global_std < 1.5:
        return 0, 0, w, h

    binary = (heatmap >= thresh_val).astype(np.uint8) * 255

    # 形态学连接小缺口
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # 找连通域
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if num <= 1:
        return 0, 0, w, h

    best_score = -1.0
    best_idx = 0
    total_area = float(h * w)

    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # 面积过滤：不要太大（可能是手），不要太小（噪声）
        area_ratio = area / total_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        # 形状紧密度：LED区域应该是偏方正的
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 5.0:
            continue

        # 计算该区域的平均方差强度（越高越好=越像真实信号）
        mask = (labels == i).astype(np.float32)
        region_intensity = float(np.sum(heatmap * mask)) / max(area, 1.0)

        # 综合得分：面积适中+形状紧凑+强度高
        compactness = (4 * math.pi * area) / max((bw * 2 + bh * 2) ** 2, 1.0)
        score = region_intensity * 10.0 + compactness * 3.0

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx == 0:
        # fallback: 取最大满足条件的区域
        for i in range(1, num):
            area = float(stats[i, cv2.CC_STAT_AREA])
            ratio = area / total_area
            if min_area_ratio <= ratio <= max_area_ratio:
                if area > best_score:
                    best_score = area
                    best_idx = i

    x = int(stats[best_idx, cv2.CC_STAT_LEFT])
    y = int(stats[best_idx, cv2.CC_STAT_TOP])
    bw = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[best_idx, cv2.CC_STAT_HEIGHT])

    # 扩展边界（给slot留余量）
    pad_x = max(6, int(bw * 0.08))
    pad_y = max(6, int(bh * 0.08))
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    bw = min(w - x, bw + pad_x * 2)
    bh = min(h - y, bh + pad_y * 2)

    return x, y, bw, bh


def build_range_heatmap(panels: list[np.ndarray]) -> np.ndarray:
    """
    亮度范围热力图（v4方法）。

    原理：计算每个像素在所有帧中的 (最大值 - 最小值)。
    LED灯位会亮灭切换，产生大的亮度范围。
    静态区域（白面板、桌面）亮度不变，范围为≈0。

    比偏离累积法的优势：
    - 不受相机整体曝光变化影响（全局变化会同时影响min和max）
    - 手部移动产生的信号通常是"暗→亮"或"亮→暗"的单向变化，
      而LED闪烁是周期性ON/OFF，范围值更大且更集中
    """
    h, w = panels[0].shape[:2]
    n_frames = len(panels)
    if n_frames < 2:
        return np.zeros((h, w), dtype=np.float32)

    # 初始化min/max
    gray0 = cv2.cvtColor(panels[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    min_img = gray0.copy()
    max_img = gray0.copy()

    for fi in range(1, n_frames):
        gray = cv2.cvtColor(panels[fi], cv2.COLOR_BGR2GRAY).astype(np.float32)
        np.minimum(min_img, gray, out=min_img)
        np.maximum(max_img, gray, out=max_img)

    range_img = max_img - min_img
    return range_img.astype(np.float32)


def build_panel_range_heatmap(panels: list[np.ndarray],
                               min_brightness: float = 200.0,
                               blur_sigma: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """
    v6方法：面板限制范围热力图。

    核心改进：只计算**白色面板区域**(均值亮度>min_brightness)的范围值。
    这直接排除了：
    - 手部区域（均值<150，深色手套/皮肤）
    - 深色背景（均值<100）
    - 只保留白面板上的微小变化（LED闪烁 range=13-36）

    注意：min_brightness需要足够高（建议200+），否则手部也会被包含。

    Returns:
      (heatmap, panel_mask): 范围热力图和面板区域掩码
    """
    h, w = panels[0].shape[:2]
    n_frames = len(panels)
    if n_frames < 2:
        return np.zeros((h, w), dtype=np.float32), np.ones((h, w), dtype=np.float32)

    # Step 1: 计算时序均值图 → 识别白面板区域
    gray_stack = []
    for fi in range(n_frames):
        gray = cv2.cvtColor(panels[fi], cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_stack.append(gray)

    # 均值图（平滑以减少局部波动）
    mean_img = np.mean(gray_stack, axis=0)
    mean_blurred = cv2.GaussianBlur(mean_img, (blur_sigma, blur_sigma), 0)

    # 面板掩码：只有足够亮的像素才被认为是面板
    panel_mask = (mean_blurred >= min_brightness).astype(np.float32)

    # 形态学清理：连接面板内部的小缺口
    panel_mask_bin = (panel_mask * 255).astype(np.uint8)
    panel_mask_bin = cv2.morphologyEx(panel_mask_bin, cv2.MORPH_CLOSE,
                                       np.ones((31, 31), np.uint8))
    panel_mask_bin = cv2.morphologyEx(panel_mask_bin, cv2.MORPH_OPEN,
                                       np.ones((15, 15), np.uint8))
    panel_mask = panel_mask_bin.astype(np.float32) / 255.0

    # Step 2: 在面板区域内计算范围热力图
    gray0 = gray_stack[0]
    min_img = gray0.copy()
    max_img = gray0.copy()

    for fi in range(1, n_frames):
        gray = gray_stack[fi]
        np.minimum(min_img, gray, out=min_img)
        np.maximum(max_img, gray, out=max_img)

    range_img = max_img - min_img

    # Step 3: 应用面板掩码 → 非面板区域归零
    restricted_range = range_img * panel_mask

    return restricted_range.astype(np.float32), panel_mask


def build_deviation_heatmap(panels: list[np.ndarray]) -> np.ndarray:
    """
    均值正向偏离累积热力图(v3方法)。
    对有手部移动的场景推荐优先使用 build_range_heatmap()。
    """
    h, w = panels[0].shape[:2]
    n_frames = len(panels)

    # 1. 构建灰度帧栈
    gray_stack = np.zeros((n_frames, h, w), dtype=np.float32)
    for fi, panel in enumerate(panels):
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_stack[fi] = gray

    # 2. 时序均值图像
    mean_img = np.mean(gray_stack, axis=0)

    # 3. 逐帧正向偏离并累积
    accumulation = np.zeros((h, w), dtype=np.float64)
    for fi in range(n_frames):
        deviation = gray_stack[fi] - mean_img
        positive_dev = np.maximum(deviation, 0)  # 只保留变亮的像素
        accumulation += positive_dev

    return accumulation.astype(np.float32)


def build_variance_heatmap(panels: list[np.ndarray]) -> np.ndarray:
    """
    帧间方差热力图（v2遗留方法，作为备选）。
    注意：对于有手部移动的场景，方差图会被运动噪声主导。
    """
    h, w = panels[0].shape[:2]
    gray_stack = np.zeros((len(panels), h, w), dtype=np.float32)
    for fi, panel in enumerate(panels):
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_stack[fi] = gray
    mean_img = np.mean(gray_stack, axis=0)
    var_img = np.mean((gray_stack - mean_img[None, :, :]) ** 2, axis=0)
    std_img = np.sqrt(np.maximum(var_img, 0))
    return std_img


# 保留旧名作为别名（默认使用v3偏离累积方法）
def build_accumulation_heatmap(panels: list[np.ndarray]) -> np.ndarray:
    """别名：实际调用均值正向偏离热力图。"""
    return build_deviation_heatmap(panels)


def find_projection_peaks(projection: np.ndarray, min_distance_ratio: float = 0.08,
                          prominence_ratio: float = 0.15,
                          min_peaks: int = 3) -> np.ndarray:
    """
    在一维投影上找峰值。
    使用简单的局部最大值检测 + 显著性过滤。
    返回峰值的索引数组。
    """
    n = len(projection)
    if n == 0:
        return np.array([], dtype=int)

    # 高斯平滑去噪
    sigma = max(n * 0.02, 3)
    ksize = int(sigma * 4) | 1
    smoothed = cv2.GaussianBlur(projection.reshape(-1, 1).astype(np.float32),
                                 (ksize, 1), sigma).flatten()

    # 局部最大值：比左右邻居都大
    min_dist = max(int(n * min_distance_ratio), 3)
    peaks_idx = []
    for i in range(min_dist, n - min_dist):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            if all(smoothed[i] >= smoothed[j] for j in range(max(0, i - min_dist), min(n, i + min_dist + 1)) if j != i):
                pass  # already checked neighbors
            peaks_idx.append((i, smoothed[i]))

    if len(peaks_idx) < min_peaks:
        # fallback: 取 top-k 最大值
        k = min(min_peaks, n)
        idx = np.argsort(smoothed)[-k:]
        return np.sort(idx)

    # 按高度排序
    peaks_idx.sort(key=lambda x: -x[1])
    # 非极大抑制：保留足够显著的峰
    kept = []
    for idx, val in peaks_idx:
        # 检查是否跟已有的太近
        too_close = any(abs(idx - k) < min_dist for k, _ in kept)
        if not too_close:
            kept.append((idx, val))
        if len(kept) >= 25:  # 合理上限
            break

    if not kept:
        return np.array([n // 2], dtype=int)

    kept.sort(key=lambda x: x[0])
    result = np.array([p[0] for p in kept], dtype=int)

    # 过滤低显著性峰值
    vals = np.array([smoothed[p] for p in result])
    val_range = vals.max() - vals.min()
    if val_range > 0:
        threshold = vals.min() + val_range * prominence_ratio
        mask = vals >= threshold
        result = result[mask]

    return result if len(result) >= min_peaks else np.array([n // 2], dtype=int)


def detect_grid_from_heatmap(heatmap: np.ndarray,
                              expected_rows: int | None = None,
                              expected_cols: int | None = None,
                              ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    从亮度累积热力图中检测LED网格的行列位置。

    Returns:
      row_peaks: Y坐标峰值索引 (行中心)
      col_peaks: X坐标峰值索引 (列中心)
      rows: 检测到的行数
      cols: 检测到的列数
    """
    ph, pw = heatmap.shape

    # X方向投影（按列取均值）
    col_proj = heatmap.mean(axis=0)  # shape (pw,)
    # Y方向投影（按行取均值）
    row_proj = heatmap.mean(axis=1)  # shape (ph,)

    # 找峰值
    col_peaks_raw = find_projection_peaks(col_proj)
    row_peaks_raw = find_projection_peaks(row_proj)

    # 后处理：如果提供了期望行列数，尝试匹配
    cols = len(col_peaks_raw)
    rows = len(row_peaks_raw)

    if expected_cols and abs(cols - expected_cols) <= 2:
        # 调整到期望数量
        col_peaks = _adjust_peak_count(col_proj, col_peaks_raw, expected_cols)
        cols = expected_cols
    elif expected_cols and cols > expected_cols + 3:
        # 太多了，取最显著的N个
        col_vals = col_proj[col_peaks_raw]
        top_n = np.argsort(col_vals)[-expected_cols:]
        col_peaks = np.sort(col_peaks_raw[top_n])
        cols = expected_cols
    else:
        col_peaks = col_peaks_raw

    if expected_rows and abs(rows - expected_rows) <= 2:
        row_peaks = _adjust_peak_count(row_proj, row_peaks_raw, expected_rows)
        rows = expected_rows
    elif expected_rows and rows > expected_rows + 3:
        row_vals = row_proj[row_peaks_raw]
        top_n = np.argsort(row_vals)[-expected_rows:]
        row_peaks = np.sort(row_peaks_raw[top_n])
        rows = expected_rows
    else:
        row_peaks = row_peaks_raw

    return row_peaks, col_peaks, rows, cols


def _adjust_peak_count(projection: np.ndarray, current_peaks: np.ndarray,
                        target: int) -> np.ndarray:
    """微调峰值数量到目标值（增减不超过2）。"""
    diff = target - len(current_peaks)
    if diff == 0:
        return current_peaks
    elif diff > 0:
        # 缺少的：在现有峰值之间插入中间点
        gaps = []
        for i in range(len(current_peaks) - 1):
            gap_size = current_peaks[i + 1] - current_peaks[i]
            gaps.append((gap_size, i, current_peaks[i] + gap_size // 2))
        gaps.sort(reverse=True)
        extra = [g[2] for g in gaps[:diff]]
        return np.sort(np.concatenate([current_peaks, np.array(extra, dtype=int)]))
    else:
        # 多余的：去掉最低的
        vals = projection[current_peaks]
        keep_idx = np.argsort(vals)[-target:]
        return np.sort(current_peaks[keep_idx])


def create_grid_slots(row_peaks: np.ndarray, col_peaks: np.ndarray,
                      heatmap: np.ndarray,
                      cell_margin: float = 0.35) -> list[Slot]:
    """
    根据检测到的网格交点创建slot列表。
    每个grid cell一个slot，按行优先编号。
    """
    slots: list[Slot] = []
    slot_id = 1
    ph, pw = heatmap.shape

    # 估计cell大小用于设置ROI尺寸
    if len(row_peaks) > 1:
        row_spacing = float(np.median(np.diff(row_peaks)))
    else:
        row_spacing = ph / max(len(row_peaks), 1)
    if len(col_peaks) > 1:
        col_spacing = float(np.median(np.diff(col_peaks)))
    else:
        col_spacing = pw / max(len(col_peaks), 1)

    cell_w = col_spacing * (1 - cell_margin * 0.5)
    cell_h = row_spacing * (1 - cell_margin * 0.5)

    for r_idx, ry in enumerate(row_peaks):
        for c_idx, cx in enumerate(col_peaks):
            slot = Slot(
                id=slot_id,
                x=float(cx),
                y=float(ry),
                w=max(cell_w, 10.0),
                h=max(cell_h, 10.0),
                threshold=0.5,  # 初始占位，后续自适应
            )
            slots.append(slot)
            slot_id += 1

    return slots


def compute_slot_scores(panels: list[np.ndarray], slots: list[Slot]) -> np.ndarray:
    """逐帧逐slot评分矩阵。P98亮度/255。"""
    scores = np.zeros((len(panels), len(slots)), dtype=np.float32)
    for fi, panel in enumerate(panels):
        gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        for si, slot in enumerate(slots):
            x1 = max(0, int(slot.x - slot.w * 0.5))
            y1 = max(0, int(slot.y - slot.h * 0.5))
            x2 = min(gray.shape[1], int(slot.x + slot.w * 0.5))
            y2 = min(gray.shape[0], int(slot.y + slot.h * 0.5))
            if x2 <= x1 or y2 <= y1:
                scores[fi, si] = 0.0
                continue
            roi = gray[y1:y2, x1:x2]
            scores[fi, si] = float(np.percentile(roi, 98))
    return scores


def adapt_thresholds(slots: list[Slot], scores: np.ndarray) -> list[Slot]:
    """为每个slot计算自适应阈值。基于背景+peak的自适应策略。"""
    updated: list[Slot] = []
    for si, slot in enumerate(slots):
        series = scores[:, si]
        peak = float(np.max(series))
        background = float(np.percentile(series, 35))
        if peak < 0.38:
            continue  # 太暗，不是有效灯位
        threshold = max(0.32, min(0.75, background + (peak - background) * 0.50))
        duty = float(np.mean(series >= threshold))
        if duty > 0.80:
            continue  # 常亮，不是跑马灯
        updated.append(Slot(
            id=len(updated) + 1,
            x=slot.x, y=slot.y,
            w=slot.w, h=slot.h,
            threshold=threshold,
            peak=peak,
            duty=duty,
        ))
    return updated


# ─────────────── 时序激活 & 序列匹配（复用v1逻辑） ────────────────


def active_matrix(scores: np.ndarray, thresholds: np.ndarray,
                  dominance_ratio: float = 0.55) -> np.ndarray:
    """生成激活布尔矩阵。"""
    if scores.size == 0:
        return np.zeros_like(scores, dtype=bool)
    active = scores >= thresholds[None, :]
    frame_max = scores.max(axis=1)
    active &= scores >= (frame_max[:, None] * dominance_ratio)
    active[frame_max < 0.28, :] = False
    return active


def dominant_sequence(scores: np.ndarray, slots: list[Slot],
                      min_event_frames: int = 2) -> list[dict[str, Any]]:
    """提取主导时序事件序列。"""
    thresholds = np.array([s.threshold for s in slots], dtype=np.float32)
    act = active_matrix(scores, thresholds)

    raw: list[int] = []
    for fi in range(scores.shape[0]):
        idxs = np.where(act[fi])[0]
        if len(idxs) == 0:
            raw.append(-1)
        else:
            best = int(idxs[np.argmax(scores[fi, idxs])])
            raw.append(int(slots[best].id))

    events: list[dict[str, Any]] = []
    start = 0
    current = raw[0] if raw else -1
    for idx in range(1, len(raw) + 1):
        val = raw[idx] if idx < len(raw) else None
        if val == current:
            continue
        length = idx - start
        if current >= 0 and length >= min_event_frames:
            events.append({"slot_id": current, "start_frame": start,
                           "end_frame": idx - 1, "duration_frames": length})
        start = idx
        current = val if val is not None else -1

    # 二次压缩相邻同ID事件
    compressed: list[dict[str, Any]] = []
    for ev in events:
        if compressed and compressed[-1]["slot_id"] == ev["slot_id"]:
            compressed[-1]["end_frame"] = ev["end_frame"]
            compressed[-1]["duration_frames"] += ev["duration_frames"]
        else:
            compressed.append(dict(ev))
    return compressed


def lcs_length(left: list[int], right: list[int]) -> int:
    """最长公共子序列长度。"""
    if not left or not right:
        return 0
    prev = [0] * (len(right) + 1)
    for lv in left:
        curr = [0] * (len(right) + 1)
        for j, rv in enumerate(right, start=1):
            if lv == rv:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def best_order_score(expected: list[int], observed: list[int]) -> tuple[float, int, int]:
    """循环偏移LCS最佳匹配得分。"""
    if not expected or not observed:
        return 0.0, 0, 0
    doubled = expected + expected
    max_w = min(len(doubled), max(len(expected), len(observed)) + 3)
    best = (0.0, 0, 0)
    for offset in range(len(expected)):
        cand = doubled[offset : offset + max_w]
        matched = lcs_length(cand, observed)
        denom = max(len(expected), len(observed), 1)
        sc = matched / denom
        if sc > best[0]:
            best = (sc, offset, matched)
    return best


def compare_and_judge(slots: list[Slot], scores: np.ndarray, fps: float,
                       template_path: Path | None = None) -> dict[str, Any]:
    """完整比对流程：激活→序列→LCS→NG判定。"""
    events = dominant_sequence(scores, slots, min_event_frames=max(1, int(round(fps * 0.04))))
    observed = [int(e["slot_id"]) for e in events]

    # 如果有模板则比对
    if template_path and template_path.exists():
        tpl = json.loads(template_path.read_text(encoding="utf-8"))
        expected = [int(x) for x in tpl.get("expected_sequence", [])]
        score, offset, matched = best_order_score(expected, observed)
        exp_count = len(expected)
    else:
        expected = observed
        score = 1.0
        offset = 0
        matched = len(observed)
        exp_count = len(observed)

    obs_set = set(observed)
    exp_set = set(expected)
    issues: list[dict[str, Any]] = []

    for sid in sorted(exp_set - obs_set):
        issues.append({"type": "missing", "slot_id": sid,
                       "message": f"灯位 {sid} 未按周期亮起"})
    for sid in sorted(obs_set - exp_set):
        issues.append({"type": "extra", "slot_id": sid,
                       "message": f"灯位 {sid} 不在标准顺序内"})
    
    slot_stats = []
    for si, slot in enumerate(slots):
        series = scores[:, si]
        ar = float(np.mean(series >= slot.threshold))
        pk = float(np.max(series))
        slot_stats.append({"slot_id": slot.id, "peak": round(pk, 4),
                           "threshold": round(slot.threshold, 4),
                           "active_ratio": round(ar, 4)})
        if ar > 0.85:
            issues.append({"type": "constant", "slot_id": slot.id,
                           "message": f"灯位 {slot.id} 疑似常亮"})
        if slot.id in exp_set and pk < slot.peak * 0.6 if slot.peak > 0 else False:
            issues.append({"type": "dim", "slot_id": slot.id,
                           "message": f"灯位 {slot.id} 亮度偏低"})

    if score < 0.82:
        issues.append({"type": "order", "slot_id": None,
                       "message": f"顺序不匹配, score={score:.2f}"})

    return {
        "overall_ok": len(issues) == 0,
        "order_score": round(score, 4),
        "matched_events": matched,
        "expected_event_count": exp_count,
        "observed_event_count": len(observed),
        "best_offset": offset,
        "expected_sequence": expected,
        "observed_sequence": observed,
        "events": events,
        "slot_stats": slot_stats,
        "issues": issues,
        "ng_count": len(issues),
    }


# ─────────────── 可视化输出 ────────────────


def save_heatmap_image(heatmap: np.ndarray, row_peaks: np.ndarray,
                        col_peaks: np.ndarray, output_path: Path) -> None:
    """保存热力图（带网格线标注）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 归一化并应用伪彩色
    norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)

    # 放大到可查看大小
    scale = max(1, int(800 / max(color.shape[0], color.shape[1])))
    color = cv2.resize(color, (color.shape[1] * scale, color.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)

    rp = (row_peaks * scale).astype(int)
    cp = (col_peaks * scale).astype(int)

    # 画网格线
    for cy in rp:
        cv2.line(color, (0, cy), (color.shape[1], cy), (0, 200, 255), 1)
    for cx in cp:
        cv2.line(color, (cx, 0), (cx, color.shape[0]), (0, 200, 255), 1)

    # 画交叉点
    for ry in rp:
        for cx in cp:
            cv2.circle(color, (cx, ry), 4, (0, 0, 255), -1)
            cv2.circle(color, (cx, ry), 5, (255, 255, 255), 1)

    cv2.imwrite(str(output_path), color)


def save_slot_overlay(panel_img: np.ndarray, slots: list[Slot],
                       scores_row: np.ndarray | None = None,
                       output_path: Path | None = None) -> np.ndarray | None:
    """在面板图像上标注slot位置。"""
    vis = panel_img.copy()
    for si, slot in enumerate(slots):
        cx, cy = int(slot.x), int(slot.y)
        rw, rh = int(slot.w * 0.5), int(slot.h * 0.5)
        
        if scores_row is not None and si < len(scores_row):
            score = float(scores_row[si])
            if score >= slot.threshold:
                color = (0, 255, 0)   # 绿色=激活
                radius = 11
            else:
                color = (180, 180, 180)  # 灰色=未激活
                radius = 7
        else:
            color = (0, 200, 255)  # 橙色=模板标注
            radius = 8
        
        cv2.circle(vis, (cx, cy), radius, color, 2)
        cv2.rectangle(vis, (cx - rw, cy - rh), (cx + rw, cy + rh), color, 1)
        cv2.putText(vis, str(slot.id), (cx + 6, cy - 6),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis)
    return vis


def draw_timeline_chart(scores: np.ndarray, slots: list[Slot],
                        events: list[dict[str, Any]], output_path: Path) -> None:
    """绘制时序激活时间线条形图（PNG）。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if scores.size == 0:
        return

    normalized = np.clip(scores * 255, 0, 255).astype(np.uint8).T
    heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    row_h = 18
    scale_x = max(1, int(math.ceil(1100 / max(1, heat.shape[1]))))
    heat = cv2.resize(heat, (heat.shape[1] * scale_x, heat.shape[0] * row_h),
                       interpolation=cv2.INTER_NEAREST)

    canvas = np.full((heat.shape[0] + 34, heat.shape[1] + 70, 3), 28, dtype=np.uint8)
    canvas[24 : 24 + heat.shape[0], 62 : 62 + heat.shape[1]] = heat

    for si, slot in enumerate(slots):
        y = 24 + si * row_h + 13
        cv2.putText(canvas, f"{slot.id:02d}", (8, y),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)

    for ev in events:
        x = 62 + int(ev["start_frame"]) * scale_x
        cv2.line(canvas, (x, 18), (x, canvas.shape[0] - 4), (255, 255, 255), 1)

    cv2.imwrite(str(output_path), canvas)


# ─────────────── v3 辅助：热点聚类 & 原始帧评分 ────────────────


def _merge_candidates_to_slots(candidates: list[tuple],
                                img_shape: tuple[int, int],
                                expected_rows: int | None = None,
                                expected_cols: int | None = None,
                                merge_distance: float = 40.0) -> list[Slot]:
    """
    将候选热点中心合并为最终slot列表。
    策略：贪心聚类——从最强热点开始，合并距离内的弱热点。
    """
    if not candidates:
        return []

    # 按强度排序（强优先）
    sorted_cands = sorted(candidates, key=lambda c: -c[3])
    clusters = []  # [(cx, cy, total_intensity, member_count)]

    for cx, cy, area, intensity in sorted_cands:
        merged = False
        for i, (ccx, ccy, ci, cnt) in enumerate(clusters):
            dist = math.hypot(cx - ccx, cy - ccy)
            if dist < merge_distance:
                # 合入已有cluster
                new_cnt = cnt + 1
                new_cx = (ccx * cnt + cx) / new_cnt
                new_cy = (ccy * cnt + cy) / new_cnt
                new_ci = ci + intensity
                clusters[i] = (new_cx, new_cy, new_ci, new_cnt)
                merged = True
                break
        if not merged:
            clusters.append((cx, cy, intensity, 1))

    # 转为Slot，按Y坐标排序（从上到下）
    clusters.sort(key=lambda c: c[1])  # sort by Y

    slots = []
    for idx, (cx, cy, total_int, cnt) in enumerate(clusters):
        # 根据面积估计ROI大小
        avg_area = sum(c[2] for c in candidates[:min(5, len(candidates))]) / max(min(5, len(candidates)), 1)
        roi_size = max(20.0, min(80.0, math.sqrt(avg_area) * 2))
        slots.append(Slot(
            id=idx + 1,
            x=cx, y=cy,
            w=roi_size, h=roi_size,
            threshold=0.5,
        ))

    return slots


def compute_slot_scores_raw(frames: list[np.ndarray], slots: list[Slot]) -> np.ndarray:
    """在原始分辨率帧上计算每个slot的评分（P98亮度/255）。"""
    scores = np.zeros((len(frames), len(slots)), dtype=np.float32)
    for fi, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        for si, slot in enumerate(slots):
            x1 = max(0, int(slot.x - slot.w * 0.5))
            y1 = max(0, int(slot.y - slot.h * 0.5))
            x2 = min(gray.shape[1], int(slot.x + slot.w * 0.5))
            y2 = min(gray.shape[0], int(slot.y + slot.h * 0.5))
            if x2 <= x1 or y2 <= y1:
                scores[fi, si] = 0.0
                continue
            roi = gray[y1:y2, x1:x2]
            scores[fi, si] = float(np.percentile(roi, 98))
    return scores


def save_slot_overlay_raw(frame: np.ndarray, slots: list[Slot],
                          scores_row: np.ndarray | None = None,
                          output_path: Path | None = None) -> np.ndarray | None:
    """在原始帧上标注slot位置（缩放显示）。"""
    vis = frame.copy()
    h, w = frame.shape[:2]
    scale = max(1, min(800 // w, 600 // h)) if w > 800 or h > 600 else 1
    if scale > 1:
        vis = cv2.resize(vis, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)

    for si, slot in enumerate(slots):
        cx, cy = int(slot.x * scale), int(slot.y * scale)
        rw, rh = int(slot.w * 0.5 * scale), int(slot.h * 0.5 * scale)

        if scores_row is not None and si < len(scores_row):
            score = float(scores_row[si])
            if score >= slot.threshold:
                color = (0, 255, 0)
                radius = max(8, int(11 * scale))
            else:
                color = (180, 180, 180)
                radius = max(5, int(7 * scale))
        else:
            color = (0, 200, 255)
            radius = max(6, int(8 * scale))

        cv2.circle(vis, (cx, cy), radius, color, max(2, int(2 * scale)))
        cv2.rectangle(vis, (cx - rw, cy - rh), (cx + rw, cy + rh), color, 1)
        cv2.putText(vis, str(slot.id), (cx + int(6*scale), cy - int(6*scale)),
                     cv2.FONT_HERSHEY_SIMPLEX, max(0.45, 0.45 * scale), color, 1, cv2.LINE_AA)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), vis)
    return vis


def adapt_thresholds_relaxed(slots: list[Slot], scores: np.ndarray) -> list[Slot]:
    """放宽条件的自适应阈值（用于有效slot过少时的重试）。"""
    updated = []
    for si, slot in enumerate(slots):
        series = scores[:, si]
        peak = float(np.max(series))
        background = float(np.percentile(series, 25))  # 更低的背景百分位
        if peak < 0.28:
            continue
        threshold = max(0.22, min(0.70, background + (peak - background) * 0.35))
        duty = float(np.mean(series >= threshold))
        updated.append(Slot(
            id=len(updated) + 1,
            x=slot.x, y=slot.y,
            w=slot.w, h=slot.h,
            threshold=threshold,
            peak=peak,
            duty=duty,
        ))
    return updated


def _fallback_learn_with_bbox(frames: list[np.ndarray], output_dir: Path,
                               panel_size: tuple[int, int],
                               expected_rows: int | None = None,
                               expected_cols: int | None = None,
                               focus_bbox: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """Fallback：使用传统面板bbox方案。"""
    print("[v2 learn] 使用bbox fallback方案")
    bbox = median_panel_bbox(frames)
    panels = align_all_frames(frames, bbox, panel_size)
    heatmap = build_deviation_heatmap(panels)
    crop_x, crop_y, crop_w, crop_h = auto_crop_to_active_region(heatmap)
    heatmap_crop = heatmap[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()

    row_peaks_r, col_peaks_r, rows, cols = detect_grid_from_heatmap(
        heatmap_crop, expected_rows, expected_cols)
    row_peaks = row_peaks_r + crop_y
    col_peaks = col_peaks_r + crop_x
    slots = create_grid_slots(row_peaks, col_peaks, heatmap)
    scores = compute_slot_scores(panels, slots)
    slots = adapt_thresholds(slots, scores)
    scores = compute_slot_scores(panels, slots)
    events = dominant_sequence(scores, slots)

    mid_frame = panels[len(panels)//2]
    save_slot_overlay(mid_frame, slots, scores[len(panels)//2] if scores.shape[0]>0 else None,
                       output_dir / "v2_slot_overlay.jpg")
    draw_timeline_chart(scores, slots, events, output_dir / "v2_timeline.png")

    payload = {
        "type": "marquee_sequence_template", "version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_frames": str(output_dir), "panel_size": list(panel_size),
        "reference_panel_bbox": list(bbox),
        "grid_info": {"rows": rows, "cols": cols, "total_slots": len(slots)},
        "expected_sequence": [int(e["slot_id"]) for e in events],
        "slots": [{"id": s.id, "x": s.x, "y": s.y, "w": s.w, "h": s.h,
                   "threshold": s.threshold, "peak": s.peak, "duty": s.duty}
                  for s in slots],
        "notes": "v3 fallback(bbox)",
    }
    write_json(output_dir / "v2_marquee_template.json", payload)
    return payload


# ─────────────── 主流程：learn / detect / run ────────────────


def run_learn(frames_dir: Path, output_dir: Path,
               panel_size: tuple[int, int] = DEFAULT_PANEL_SIZE,
               expected_rows: int | None = None,
               expected_cols: int | None = None,
               focus_bbox: tuple[int, int, int, int] | None = None,
               ) -> dict[str, Any]:
    """
    learn模式：从帧目录学习LED灯位模板。

    v3策略（白色面板专用）：
    1. 在原始分辨率帧上计算偏离累积热力图
    2. 用高阈值找LED热点候选区域
    3. 对热点做空间聚类得到灯位中心
    4. 如果有期望的网格结构，对齐到网格
    5. 时序评分 + 自适应阈值
    """
    print(f"[v2 learn] 加载帧目录: {frames_dir}")
    frames, fps = load_frames(frames_dir)
    print(f"[v2 learn] 加载了 {len(frames)} 帧")

    # ── Step 1: 构建热力图（v6：面板限制范围法） ──
    print("[v2 learn] Step1: 构建热力图(原始分辨率)...")

    # v6核心：先尝试面板限制范围热力图（排除暗区噪声）
    panel_hm, panel_mask = build_panel_range_heatmap(frames, min_brightness=175.0)
    range_hm = build_range_heatmap(frames)  # 全图范围作为对比

    # 评估两种方法的信号质量
    def _signal_concentration(hm):
        if hm.max() < 1:
            return -999
        thresh = hm.max() * 0.5
        ratio = np.mean(hm > thresh)  # 超过50%峰值的像素占比
        return -ratio  # 占比越小越集中（负数以便排序）

    conc_panel = _signal_concentration(panel_hm)
    conc_range = _signal_concentration(range_hm)

    # 面板限制法的额外评估：面板区域内是否有足够的有效像素
    panel_pixel_count = int(np.sum(panel_mask > 0))
    panel_active_count = int(np.sum(panel_hm > 5))  # 有明显变化的像素
    panel_signal_ratio = panel_active_count / max(panel_pixel_count, 1)

    print(f"[v2 learn] 热力图方法比较:")
    print(f"  全图范围: conc={conc_range:.4f}, max={range_hm.max():.1f}")
    print(f"  面板限制: conc={conc_panel:.4f}, max={panel_hm.max():.1f}, "
          f"面板像素={panel_pixel_count}, 有效像素={panel_active_count} ({panel_signal_ratio*100:.1f}%)")

    # 选择更优的方法（优先使用面板限制法，除非它完全没有信号）
    if panel_hm.max() > 3.0 and conc_panel <= conc_range + 0.05:
        dev_heatmap_raw = panel_hm
        print(f"[v2 learn] 选择: 面板限制范围热力图 (v6)")
        # 同时保存面板掩码可视化
        mask_vis = (panel_mask * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "v2_panel_mask.png"), mask_vis)
    else:
        dev_heatmap_raw = range_hm
        print(f"[v2 learn] 选择: 全图范围热力图 (fallback)")

    # 归一化保存原始热力图
    hm_raw_path = output_dir / "v2_heatmap_raw.png"
    save_heatmap_image(dev_heatmap_raw, np.array([]), np.array([]), hm_raw_path)

    global_max = float(np.max(dev_heatmap_raw))
    global_mean = float(np.mean(dev_heatmap_raw))
    global_std = float(np.std(dev_heatmap_raw))
    print(f"[v2 learn] 热力图统计: max={global_max:.1f}, mean={global_mean:.1f}, std={global_std:.1f}")

    if global_max < 5.0:
        print("[v2 learn] 警告: 热力图信号极弱，可能没有闪烁LED或帧数不足")
        # fallback到方差热力图
        var_heatmap = build_variance_heatmap(frames)
        if float(np.max(var_heatmap)) > global_max:
            dev_heatmap_raw = var_heatmap
            global_max = float(np.max(dev_heatmap_raw))
            print(f"[v2 learn] 切换到方差热力图, max={global_max:.1f}")

    # ── Step 2: 找LED热点区域（v5优化：高阈值 + 时序过滤） ──
    # 边缘屏蔽：排除图像边缘区域（手部移动和相机抖动产生大量伪信号）
    h_raw, w_raw = dev_heatmap_raw.shape
    edge_margin_x = int(w_raw * 0.12)  # 左右各排除12%
    edge_margin_y = int(h_raw * 0.10)  # 上下各排除10%
    edge_mask = np.ones((h_raw, w_raw), dtype=np.float32)
    if edge_margin_x > 0:
        edge_mask[:, :edge_margin_x] = 0
        edge_mask[:, -edge_margin_x:] = 0
    if edge_margin_y > 0:
        edge_mask[:edge_margin_y, :] = 0
        edge_mask[-edge_margin_y:, :] = 0

    masked_heatmap = dev_heatmap_raw * edge_mask

    # 只在非边缘区域内统计阈值
    inner_values = masked_heatmap[edge_mask > 0]
    if len(inner_values) == 0:
        inner_values = dev_heatmap_raw.flatten()

    inner_max = float(np.max(inner_values))
    inner_mean = float(np.mean(inner_values))
    inner_std = float(np.std(inner_values))
    inner_p97 = float(np.percentile(inner_values, 97))
    inner_p95 = float(np.percentile(inner_values, 95))
    inner_p99 = float(np.percentile(inner_values, 99))
    print(f"[v2 learn] Step2: 内部区域统计 max={inner_max:.1f}, mean={inner_mean:.1f}, std={inner_std:.1f}")
    print(f"[v2 learn] Step2: 百分位 P95={inner_p95:.1f}, P97={inner_p97:.1f}, P99={inner_p99:.1f}")

    # v5策略：用更高的百分位阈值 + 候选数量上限
    # 真LED信号(1368) vs 噪声(~764)，差距近2倍，P97应该能有效分离
    spot_threshold = float(np.percentile(inner_values, 97))
    spot_threshold = max(spot_threshold, inner_max * 0.15)  # 至少峰值的15%
    print(f"[v2 learn] Step2: LED热点阈值={spot_threshold:.1f} (P97)")

    binary_spots = (masked_heatmap >= spot_threshold).astype(np.uint8) * 255

    # 形态学连接相邻像素为连通区域（更大的核以合并邻近像素）
    binary_spots = cv2.morphologyEx(binary_spots, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    binary_spots = cv2.morphologyEx(binary_spots, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # 找每个连通区域的质心 = 候选灯位
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_spots, 8)
    print(f"[v2 learn] 检测到 {num_labels - 1} 个候选热点区域")

    # 过滤：去掉太小的（噪声）和太大的（非点状）
    min_spot_area = 50     # 提高最小面积门槛
    max_spot_area = 10000  # 收紧上限

    candidate_centers = []  # (cx, cy, area, intensity)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_spot_area or area > max_spot_area:
            continue
        cx, cy = centroids[i]
        # 计算该区域的热力图强度（使用原始热力图，非masked）
        mask = (labels == i)
        intensity = float(np.mean(dev_heatmap_raw[mask]))
        candidate_centers.append((float(cx), float(cy), area, intensity))

    # 按强度排序
    candidate_centers.sort(key=lambda c: -c[3])
    print(f"[v2 learn] 面积过滤后保留 {len(candidate_centers)} 个有效热点")

    # ── v5新增: 时序自相关过滤 ──
    # 真LED会周期性亮灭 → 时序曲线有规律波动 → 自相关系数高
    # 手部运动/噪声 → 随机或单向变化 → 自相关系数低
    if len(candidate_centers) > 6 and len(frames) >= 10:
        filtered_by_temporal = []
        roi_radius = 25  # 每个候选点周围取25px半径的ROI做时序分析

        for cx, cy, area, intensity in candidate_centers:
            x1 = max(0, int(cx - roi_radius))
            y1 = max(0, int(cy - roi_radius))
            x2 = min(w_raw, int(cx + roi_radius))
            y2 = min(h_raw, int(cy + roi_radius))

            # 构建该位置的时序亮度曲线
            time_series = []
            for fi, frame in enumerate(frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                roi = gray[y1:y2, x1:x2]
                if roi.size > 0:
                    time_series.append(float(np.percentile(roi, 90)))
                else:
                    time_series.append(0.0)

            ts = np.array(time_series, dtype=np.float32)

            # 计算归一化自相关（lag=1到lag=max_lag）
            ts_centered = ts - np.mean(ts)
            ts_var = np.var(ts)
            if ts_var < 1.0:
                # 几乎没有变化 → 不是闪烁LED
                autocorr_score = 0.0
            else:
                # 计算多个lag的自相关，取最大值作为周期性指标
                max_lag = min(len(ts) // 2, 12)
                best_autocorr = -1.0
                for lag in range(1, max_lag + 1):
                    if lag >= len(ts_centered):
                        break
                    corr = np.corrcoef(ts_centered[:-lag], ts_centered[lag:])[0, 1]
                    if not np.isnan(corr):
                        best_autocorr = max(best_autocorr, abs(corr))
                autocorr_score = best_autocorr

            # 同时检查：是否有明显的ON/OFF切换（方差足够大）
            variance_score = ts_var / max(ts_var, 1.0)  # 已经是归一化的
            range_score = (np.max(ts) - np.min(ts)) / max(np.max(ts), 1.0)

            # 综合评分：自相关(40%) + 范围变化(35%) + 热力图强度(25%)
            norm_intensity = intensity / max(inner_max, 1.0)
            combined_score = autocorr_score * 0.40 + range_score * 0.35 + norm_intensity * 0.25

            filtered_by_temporal.append((cx, cy, area, intensity, combined_score,
                                         autocorr_score, range_score))

        # 按综合评分排序
        filtered_by_temporal.sort(key=lambda c: -c[4])

        # 只保留综合评分最高的前N个（期望灯位数的2倍+2）
        max_keep = max(8, len(candidate_centers))  # 先不过度限制
        candidate_centers_filtered = [(c[0], c[1], c[2], c[3]) for c in filtered_by_temporal[:max_keep]]

        # 输出时序过滤的详细信息
        print(f"[v2 learn] 时序过滤: {len(candidate_centers)} -> {len(candidate_centers_filtered)} 个候选")
        if len(filtered_by_temporal) >= 3:
            top3 = filtered_by_temporal[:3]
            for i, (cx, cy, area, inten, score, ac, rs) in enumerate(top3):
                print(f"   Top{i+1}: ({cx:.0f},{cy:.0f}) inten={inten:.1f} "
                      f"autocorr={ac:.3f} range_ratio={rs:.3f} combined={score:.3f}")
    else:
        candidate_centers_filtered = candidate_centers

    # 最终使用过滤后的候选列表
    final_candidates = candidate_centers_filtered

    if len(final_candidates) < 2:
        # v5: 分级降阈值重试
        for retry_pct, close_k, open_k, min_a, max_a in [
            (95, 15, 5, 40, 8000),
            (93, 17, 5, 30, 10000),
            (90, 19, 7, 20, 12000),
        ]:
            spot_threshold = float(np.percentile(inner_values, retry_pct))
            spot_threshold = max(spot_threshold, inner_max * 0.08)
            print(f"[v2 learn] 降低阈值重试(P{retry_pct}): {spot_threshold:.1f}")
            binary_spots = (masked_heatmap >= spot_threshold).astype(np.uint8) * 255
            binary_spots = cv2.morphologyEx(binary_spots, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
            binary_spots = cv2.morphologyEx(binary_spots, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_spots, 8)
            final_candidates = []
            for i in range(1, num_labels):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < min_a or area > max_a:
                    continue
                cx, cy = centroids[i]
                mask = (labels == i)
                intensity = float(np.mean(dev_heatmap_raw[mask]))
                final_candidates.append((float(cx), float(cy), area, intensity))
            final_candidates.sort(key=lambda c: -c[3])
            if len(final_candidates) >= 2:
                print(f"[v2 learn] 重试成功: {len(final_candidates)} 个热点")
                break

    # ── Step 3: 聚类/合并相邻热点为最终灯位（v5优化） ──
    # v5: 使用更激进的合并距离，并且限制最终slot数量
    merge_dist = 60.0  # v4是40px，v5加大到60px以合并邻近伪检测
    final_slots = _merge_candidates_to_slots(final_candidates, dev_heatmap_raw.shape,
                                               expected_rows, expected_cols,
                                               merge_distance=merge_dist)
    print(f"[v2 learn] Step3: 合并为 {len(final_slots)} 个灯位")

    # v5后处理：如果slot仍然太多，按热力图强度保留top-N
    max_expected_slots = 10  # 最多期望的灯位数（按钮+LED指示灯）
    if len(final_slots) > max_expected_slots:
        # 计算每个slot在原始热力图上的强度
        slot_intensities = []
        for si, slot in enumerate(final_slots):
            x1 = max(0, int(slot.x - slot.w * 0.5))
            y1 = max(0, int(slot.y - slot.h * 0.5))
            x2 = min(dev_heatmap_raw.shape[1], int(slot.x + slot.w * 0.5))
            y2 = min(dev_heatmap_raw.shape[0], int(slot.y + slot.h * 0.5))
            inten = float(np.mean(dev_heatmap_raw[y1:y2, x1:x2]))
            slot_intensities.append((si, inten))

        # 按强度排序，保留top-N
        slot_intensities.sort(key=lambda x: -x[1])
        keep_indices = set(si for si, _ in slot_intensities[:max_expected_slots])
        final_slots = [s for i, s in enumerate(final_slots) if i in keep_indices]
        # 重新编号
        for i, s in enumerate(final_slots):
            s.id = i + 1
        print(f"[v2 learn] Step3b: 强度过滤后保留 {len(final_slots)} 个灯位(top-{max_expected_slots})")

    if len(final_slots) < 2:
        print("[v2 learn] 警告: 检测到的灯位过少，尝试使用面板bbox方案作为fallback")
        return _fallback_learn_with_bbox(frames, output_dir, panel_size,
                                          expected_rows, expected_cols, focus_bbox)

    # ── Step 4: 计算时序评分 ──
    # 使用原始帧（不resize），slot坐标在原始帧坐标系中
    scores = compute_slot_scores_raw(frames, final_slots)
    print(f"[v2 learn] Step4: 评分矩阵 {scores.shape}")

    # ── Step 5: 自适应阈值过滤 ──
    slots_valid = adapt_thresholds(final_slots, scores)
    print(f"[v2 learn] Step5: 阈值过滤后 {len(slots_valid)} 个有效slot")

    if len(slots_valid) < 2:
        print("[v2 learn] 有效slot太少，放宽条件重试...")
        slots_valid = adapt_thresholds_relaxed(final_slots, scores)

    # 重新编号
    for i, s in enumerate(slots_valid):
        s.id = i + 1
    final_slots = slots_valid

    scores = compute_slot_scores_raw(frames, final_slots)

    # ── Step 6: 提取时序事件序列 ──
    events = dominant_sequence(scores, final_slots, min_event_frames=max(1, int(round(fps * 0.04))))
    print(f"[v2 learn] Step6: {len(events)} 个时序事件")

    # ── Step 7: 可视化 ──
    mid_idx = len(frames) // 2
    mid_frame = frames[mid_idx].copy()
    mid_scores = scores[mid_idx] if len(scores) > mid_idx else None

    overlay_path = output_dir / "v2_slot_overlay.jpg"
    save_slot_overlay_raw(mid_frame, final_slots, mid_scores, overlay_path)

    tl_path = output_dir / "v2_timeline.png"
    draw_timeline_chart(scores, final_slots, events, tl_path)

    # 裁剪热力图到有效区域再存一份
    if final_slots:
        all_x = [s.x for s in final_slots]
        all_y = [s.y for s in final_slots]
        margin = 60
        hx1 = max(0, int(min(all_x)) - margin)
        hy1 = max(0, int(min(all_y)) - margin)
        hx2 = min(dev_heatmap_raw.shape[1], int(max(all_x)) + margin)
        hy2 = min(dev_heatmap_raw.shape[0], int(max(all_y)) + margin)
        hm_crop = dev_heatmap_raw[hy1:hy2, hx1:hx2]
        row_offsets = np.array([int(s.y) - hy1 for s in final_slots])
        col_offsets = np.array([int(s.x) - hx1 for s in final_slots])
        save_heatmap_image(hm_crop, row_offsets, col_offsets, output_dir / "v2_heatmap.png")
        crop_info = (hx1, hy1, hx2 - hx1, hy2 - hy1)
    else:
        crop_info = (0, 0, dev_heatmap_raw.shape[1], dev_heatmap_raw.shape[0])

    # ── Step 8: 组装模板 ──
    template_payload = {
        "type": "marquee_sequence_template",
        "version": 3,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_frames": str(frames_dir),
        "panel_size": list(dev_heatmap_raw.shape[1::-1]),  # (w,h)
        "reference_panel_bbox": [0, 0, dev_heatmap_raw.shape[1], dev_heatmap_raw.shape[0]],
        "focus_region": list(crop_info),
        "fps": fps,
        "grid_info": {
            "rows": expected_rows if expected_rows else len(set(int(round(s.y / 30)) for s in final_slots)),
            "cols": expected_cols if expected_cols else len(final_slots),
            "total_slots": len(final_slots),
            "positions": [{"id": s.id, "x": round(s.x, 1), "y": round(s.y, 1)}
                          for s in final_slots],
        },
        "expected_sequence": [int(e["slot_id"]) for e in events],
        "step_seconds": round(float(np.median([e["duration_frames"] for e in events])) / max(fps, 1e-6), 4) if events else 0.0,
        "min_order_score": 0.82,
        "slots": [
            {"id": s.id, "x": round(s.x, 1), "y": round(s.y, 1),
             "w": round(s.w, 1), "h": round(s.h, 1),
             "threshold": round(s.threshold, 4), "peak": round(s.peak, 4),
             "duty": round(s.duty, 4)}
            for s in final_slots
        ],
        "notes": "v5范围热力图+P97阈值+时序自相关过滤 - 白色面板LED检测",
    }

    tpl_path = output_dir / "v2_marquee_template.json"
    write_json(tpl_path, template_payload)
    print(f"[v2 learn] 模板已保存 -> {tpl_path}")

    write_csv(output_dir / "v2_learn_events.csv", events,
              ["slot_id", "start_frame", "end_frame", "duration_frames"])

    return template_payload


def run_detect(frames_dir: Path, template_path: Path, output_dir: Path,
               panel_size: tuple[int, int] = DEFAULT_PANEL_SIZE) -> dict[str, Any]:
    """detect模式：用已有模板检测新视频/帧。"""
    print(f"[v2 detect] 加载模板: {template_path}")
    tpl = json.loads(template_path.read_text(encoding="utf-8"))
    if tpl.get("version", 1) < 2:
        print("[v2 detect] 警告: 模板是v1格式，建议先用v2重新learn")

    slots_data = tpl.get("slots", [])
    slots = [
        Slot(id=int(s["id"]), x=s["x"], y=s["y"], w=s["w"], h=s["h"],
             threshold=s["threshold"], peak=s.get("peak", 1.0), duty=s.get("duty", 0.0))
        for s in slots_data
    ]
    panel_size_tpl = tuple(tpl.get("panel_size", panel_size))

    print(f"[v2 detect] 加载帧目录: {frames_dir}")
    frames, fps = load_frames(frames_dir)

    ref_bbox = tuple(int(v) for v in tpl.get("reference_panel_bbox",
                                              median_panel_bbox(frames)))

    panels = align_all_frames(frames, ref_bbox, panel_size_tpl)
    scores = compute_slot_scores(panels, slots)

    result = compare_and_judge(slots, scores, fps, template_path)
    result.update({
        "source_frames": str(frames_dir),
        "template": str(template_path),
        "fps": fps,
        "frame_count": len(frames),
        "panel_size": list(panel_size_tpl),
    })

    # 输出
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "v2_result.json", result)

    mid_i = len(panels) // 2
    save_slot_overlay(panels[mid_i], slots, scores[mid_i],
                       out_dir / "v2_detect_overlay.jpg")
    draw_timeline_chart(scores, slots, result["events"],
                        out_dir / "v2_detect_timeline.png")
    write_csv(out_dir / "v2_detect_events.csv", result["events"],
              ["slot_id", "start_frame", "end_frame", "duration_frames"])

    ok_str = "OK" if result["overall_ok"] else "NG"
    print(f"[v2 detect] 结果: {ok_str}, order_score={result['order_score']}, "
          f"ng={result['ng_count']}, events={len(result['events'])}")
    return result


# ─────────────── CLI 入口 ────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="跑马灯质检 v2 — 网格感知版")
    sub = parser.add_subparsers(dest="command", required=True)

    learn = sub.add_parser("learn", help="从良品帧目录学习网格模板")
    learn.add_argument("--frames", type=Path, required=True, help="帧图片目录")
    learn.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "learn")
    learn.add_argument("--panel-size", type=str, default="420x720")
    learn.add_argument("--rows", type=int, default=None, help="期望行数（可选）")
    learn.add_argument("--cols", type=int, default=None, help="期望列数（可选）")

    detect = sub.add_parser("detect", help="用网格模板检测帧目录")
    detect.add_argument("--frames", type=Path, required=True, help="帧图片目录")
    detect.add_argument("--template", type=Path, required=True, help="v2模板JSON")
    detect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "detect")

    run = sub.add_parser("run", help="先learn再detect验证")
    run.add_argument("--frames", type=Path, required=True, help="帧图片目录")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "run")
    run.add_argument("--panel-size", type=str, default="420x720")
    run.add_argument("--rows", type=int, default=None)
    run.add_argument("--cols", type=int, default=None)

    return parser


def parse_size(s: str) -> tuple[int, int]:
    parts = s.lower().split("x")
    return int(parts[0]), int(parts[1])


def main():
    args = build_parser().parse_args()
    ps = parse_size(args.panel_size) if hasattr(args, 'panel_size') else DEFAULT_PANEL_SIZE

    if args.command == "learn":
        payload = run_learn(
            args.frames, args.output_dir, panel_size=ps,
            expected_rows=getattr(args, 'rows', None),
            expected_cols=getattr(args, 'cols', None),
        )
        gi = payload.get("grid_info", {})
        print(json.dumps({
            "template": str(args.output_dir / "v2_marquee_template.json"),
            "slots": len(payload.get("slots", [])),
            "events": len(payload.get("expected_sequence", [])),
            "grid": f"{gi.get('rows', '?')}x{gi.get('cols', '?')}",
        }, ensure_ascii=False))

    elif args.command == "detect":
        result = run_detect(args.frames, args.template, args.output_dir, ps)
        print(json.dumps({
            "overall_ok": result["overall_ok"],
            "ng_count": result["ng_count"],
            "order_score": result["order_score"],
            "output_dir": str(args.output_dir),
        }, ensure_ascii=False))

    elif args.command == "run":
        tpl_path = args.output_dir / "v2_marquee_template.json"
        payload = run_learn(args.frames, args.output_dir, panel_size=ps,
                            expected_rows=getattr(args, 'rows', None),
                            expected_cols=getattr(args, 'cols', None))
        result = run_detect(args.frames, tpl_path, args.output_dir / "detect", ps)
        gi = payload.get("grid_info", {})
        print(json.dumps({
            "template": str(tpl_path),
            "grid": f"{gi.get('rows', '?')}x{gi.get('cols', '?')}",
            "slots": len(payload.get("slots", [])),
            "events": len(payload.get("expected_sequence", [])),
            "overall_ok": result["overall_ok"],
            "ng_count": result["ng_count"],
            "order_score": result["order_score"],
            "output_dir": str(args.output_dir),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
