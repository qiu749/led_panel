"""
display_panel_core — 数码管/显示条状态机检测核心算法

功能：
  — 显示条 ROI 检测（在暗背景中找亮区域）
  — 透视矫正到标准尺寸
  — 区域采样（数码管段位、图标、状态点）
  — 多帧融合消除频闪
  — 结构化状态相似度
  — 状态机推进（复用 running_led_core 概念但适用于结构化状态）
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import cv2, numpy as np


# ─── 类型定义 ─────────────────────────────────────

@dataclass
class PanelRegion:
    """检测区域：显示条上的一个组件"""
    name: str
    type: str  # "seven_segment_digit", "icon", "dot", "text"
    x: int; y: int; w: int; h: int  # 在标准矫正图中的坐标

@dataclass
class PanelLayout:
    """显示条布局配置"""
    panel_width: int = 640
    panel_height: int = 160
    regions: list[PanelRegion] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path) -> PanelLayout:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            panel_width=d.get("panel_width", 640),
            panel_height=d.get("panel_height", 160),
            regions=[PanelRegion(**r) for r in d.get("regions", [])],
        )

    def to_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"panel_width": self.panel_width, "panel_height": self.panel_height,
                        "regions": [asdict(r) for r in self.regions]},
                       f, ensure_ascii=False, indent=2)

@dataclass
class DisplayState:
    """一帧解码出的结构化状态"""
    frame_index: int = 0
    time_sec: float = 0.0
    # key: region_name, value: on/off (bool) 或 digit 值
    values: dict[str, Any] = field(default_factory=dict)
    # 置信度评分
    confidence: float = 0.0

@dataclass
class FusedDisplayState:
    """多帧融合后的稳定状态"""
    values: dict[str, Any] = field(default_factory=dict)
    fused_frames: int = 0
    confidence: float = 0.0

@dataclass
class KeyDisplayState:
    """学习到的关键状态"""
    id: str
    index: int
    values: dict[str, Any]
    median_hold_frames: int = 5

@dataclass
class KeyDisplayModel:
    """关键状态模型"""
    version: int = 1
    source_video: str = ""
    panel_layout: PanelLayout | None = None
    states: list[KeyDisplayState] = field(default_factory=list)
    sequence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version, "source_video": self.source_video,
            "panel_layout": asdict(self.panel_layout) if self.panel_layout else None,
            "states": [asdict(s) for s in self.states],
            "sequence": self.sequence,
        }

@dataclass
class DisplayEvent:
    frame_index: int; time_sec: float; type: str
    expected_state: str = ""; matched_state: str = ""
    message: str = ""

@dataclass
class DisplayRoi:
    """显示条 ROI：同时保存轴对齐外接框和四角点。"""
    x: int
    y: int
    w: int
    h: int
    points: list[list[float]]
    component_count: int = 0
    bright_span: float = 0.0
    confidence: float = 0.0


# ─── 显示条 ROI 检测 ─────────────────────────────

def find_display_roi(frame: np.ndarray) -> DisplayRoi | None:
    """在暗背景中找亮色显示条 ROI

    策略：高亮组件分组 → 合并同一条显示屏上的组件 → 外扩包含黑色边框。

    不能取最大连通域。数码管刷新/过曝时，最大连通域通常只是某一段
    亮线，会把整条显示屏裁成一小块。
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 多阈值寻找高亮区域
    for th in [210, 180, 150]:
        bright = (gray > th).astype(np.uint8) * 255
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        boxes: list[tuple[int, int, int, int, float]] = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 3:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 2 or bh < 2:
                continue
            # 过滤明显不是显示内容的大面积反光/背景块。
            if bw > w * 0.70 or bh > h * 0.45:
                continue
            boxes.append((x, y, bw, bh, area))

        if len(boxes) < 2:
            continue

        best: tuple[float, int, int, int, int, list[tuple[int, int, int, int, float]]] | None = None
        band_h = max(90, int(h * 0.18))
        centers_y = [y + bh / 2 for _, y, _, bh, _ in boxes]

        for seed_y in centers_y:
            group = [
                b for b in boxes
                if abs((b[1] + b[3] / 2) - seed_y) <= band_h
            ]
            if len(group) < 2:
                continue

            x1 = min(b[0] for b in group)
            y1 = min(b[1] for b in group)
            x2 = max(b[0] + b[2] for b in group)
            y2 = max(b[1] + b[3] for b in group)
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            aspect = bw / max(bh, 1)
            if aspect < 2.0 or bw < w * 0.12:
                continue

            bright_area = sum(b[4] for b in group)
            score = bright_area * min(aspect, 8.0) * (bw / w)
            if best is None or score > best[0]:
                best = (score, x1, y1, x2, y2, group)

        if best is None:
            continue

        _, x1, y1, x2, y2, group = best
        bw = x2 - x1
        bh = y2 - y1

        # 高亮内容只是显示条内部元素，必须向外扩到黑色面板边界。
        # 纵向不能过度外扩，否则透视输出会把灯压扁。
        pad_x = int(bw * 0.14)
        pad_y = max(int(bh * 0.45), int(bw * 0.04))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        roi_w = x2 - x1
        roi_h = y2 - y1
        if roi_w / max(roi_h, 1) > 1.7:
            points = _estimate_display_quad(frame, group, (x1, y1, x2, y2))
            confidence = min(1.0, (len(group) / 12.0) * ((bw / w) / 0.25))
            return DisplayRoi(
                x=x1, y=y1, w=roi_w, h=roi_h, points=points,
                component_count=len(group), bright_span=float(bw),
                confidence=float(confidence),
            )

    return None


def _estimate_display_quad(
    frame: np.ndarray,
    bright_group: list[tuple[int, int, int, int, float]],
    bounds: tuple[int, int, int, int],
) -> list[list[float]]:
    """估计显示屏四角点。

    亮字本身不足以给出黑色显示条边缘，因此先用亮点估计屏幕角度，
    再在扩展后的外接框上构造一个同角度的旋转矩形。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bounds

    centers = []
    for x, y, bw, bh, area in bright_group:
        reps = max(1, min(20, int(round(area))))
        centers.extend([(x + bw / 2.0, y + bh / 2.0)] * reps)

    angle_deg = 0.0
    if len(centers) >= 2:
        pts = np.asarray(centers, dtype=np.float32)
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = centered.T @ centered / max(len(pts), 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        vec = eigvecs[:, int(np.argmax(eigvals))]
        angle_deg = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
        if angle_deg < -90:
            angle_deg += 180
        elif angle_deg > 90:
            angle_deg -= 180

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rw = float(x2 - x1)
    raw_rh = float(y2 - y1)
    # 约束显示条四边形比例。亮点外扩框容易吃到外壳/线缆，导致高度过大；
    # 透视到固定尺寸时会把灯珠/段码压扁。
    target_aspect = 4.0
    min_rh = rw / 6.0
    max_rh = rw / 3.2
    rh = float(np.clip(raw_rh, min_rh, max_rh))

    rect = ((cx, cy), (rw, rh), angle_deg)
    box = cv2.boxPoints(rect).astype(np.float32)
    box[:, 0] = np.clip(box[:, 0], 0, w - 1)
    box[:, 1] = np.clip(box[:, 1], 0, h - 1)
    return _order_quad_points(box).tolist()


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    """按左上、右上、右下、左下排序四角点。"""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def warp_display_panel(frame: np.ndarray, roi: DisplayRoi | tuple[int,int,int,int],
                       target_size: tuple[int,int] = (640, 160)) -> np.ndarray:
    """将显示条 ROI 透视矫正到标准尺寸。"""
    if isinstance(roi, DisplayRoi):
        tw, th = target_size
        src = _order_quad_points(np.asarray(roi.points, dtype=np.float32))
        dst = np.asarray(
            [[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, matrix, target_size, flags=cv2.INTER_LINEAR)

    x, y, w, h = roi
    panel = frame[y:y+h, x:x+w]
    if panel.size == 0:
        raise ValueError("无效 ROI")
    return cv2.resize(panel, target_size, interpolation=cv2.INTER_LINEAR)


# ─── 区域采样 ────────────────────────────────────

def sample_region_threshold(warped: np.ndarray, region: PanelRegion,
                             method: str = "percentile") -> bool:
    """对单个区域采样亮灭

    Args:
        warped: 标准尺寸的矫正图
        region: 区域定义
        method: "percentile" (默认) 或 "mean"

    Returns:
        True=亮, False=灭
    """
    h, w = warped.shape[:2]
    x1 = max(0, region.x); y1 = max(0, region.y)
    x2 = min(w, region.x + region.w); y2 = min(h, region.y + region.h)
    if x2 <= x1 or y2 <= y1:
        return False

    roi = warped[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    if method == "percentile":
        # 使用 p85 作为区域亮度指标
        score = float(np.percentile(gray, 85))
    else:
        score = float(np.mean(gray))

    # 找局部背景（取区域边缘一圈）
    bg = cv2.cvtColor(warped[max(0,y1-5):min(h,y2+5), max(0,x1-5):min(w,x2+5)], cv2.COLOR_BGR2GRAY)
    bg_val = float(np.median(bg)) if bg.size > 0 else 0

    # 亮灭判断：信号强度超过背景一定比例
    threshold = bg_val + 40 if bg_val > 30 else 60
    return score >= threshold


def decode_seven_segment_digit(warped: np.ndarray, region: PanelRegion) -> str:
    """解码一个数码管数字区域

    标准7段：a b c d e f g (从上顺时针)
       a
      f b
       g
      e c
       d
    """
    h, w = warped.shape[:2]
    x1, y1 = max(0, region.x), max(0, region.y)
    x2 = min(w, region.x + region.w)
    y2 = min(h, region.y + region.h)
    if x2 <= x1 or y2 <= y1:
        return "?"

    gray = cv2.cvtColor(warped[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    cell_w = region.w // 3
    cell_h = region.h // 3

    def _segment_on(seg_x, seg_y, seg_w, seg_h) -> bool:
        """判断数字管的一个段是否亮"""
        sx = max(0, min(region.w - 1, seg_x))
        sy = max(0, min(region.h - 1, seg_y))
        ex = max(sx + 1, min(region.w, seg_x + seg_w))
        ey = max(sy + 1, min(region.h, seg_y + seg_h))
        roi = gray[sy:ey, sx:ex]
        if roi.size == 0:
            return False
        score = float(np.percentile(roi, 80))
        # 使用整图亮度作为背景参考
        bg = float(np.median(gray))
        return score > bg + 40

    segments = {
        "a": _segment_on(cell_w, 0, cell_w, cell_h // 2),
        "b": _segment_on(cell_w * 2, cell_h // 3, cell_w // 2, cell_h),
        "c": _segment_on(cell_w * 2, cell_h + cell_h // 3, cell_w // 2, cell_h),
        "d": _segment_on(cell_w, cell_h * 2 + cell_h // 4, cell_w, cell_h // 2),
        "e": _segment_on(0, cell_h + cell_h // 3, cell_w // 2, cell_h),
        "f": _segment_on(0, cell_h // 3, cell_w // 2, cell_h),
        "g": _segment_on(cell_w, cell_h, cell_w, cell_h // 3),
    }

    # 段位 → 数字映射
    seg_pattern = "".join("1" if segments[s] else "0" for s in "abcdefg")

    digit_map = {
        "1111110": "0", "0110000": "1", "1101101": "2",
        "1111001": "3", "0110011": "4", "1011011": "5",
        "1011111": "6", "1110000": "7", "1111111": "8",
        "1111011": "9", "0000000": " ", "0000001": "_",
        # 容错映射（常见误判）
        "1111111": "8", "1111110": "0", "0111111": "6",
        "1011110": "6", "1110011": "9", "1111010": "9",
        "1101111": "6", "1111100": "0", "0111100": "1",
        "0011000": "1", "0110000": "1", "1101100": "2",
        "1111000": "7", "1011011": "5", "1110011": "9",
        "1100111": "6",
    }
    return digit_map.get(seg_pattern, seg_pattern)


def decode_display_state(warped: np.ndarray, layout: PanelLayout,
                          decode_digits: bool = True) -> DisplayState:
    """解码显示条的完整状态"""
    state = DisplayState()
    for region in layout.regions:
        if region.type == "seven_segment_digit" and decode_digits:
            state.values[region.name] = decode_seven_segment_digit(warped, region)
        else:
            state.values[region.name] = sample_region_threshold(warped, region)

    # 综合置信度：亮区域的比例
    on_count = sum(1 for v in state.values.values() if isinstance(v, bool) and v)
    total = sum(1 for v in state.values.values() if isinstance(v, bool))
    state.confidence = on_count / max(total, 1) if total > 0 else 0.5
    return state


def decode_visual_grid_state(
    warped: np.ndarray,
    grid_cols: int = 48,
    grid_rows: int = 12,
) -> DisplayState:
    """把矫正显示条直接编码成自动网格状态。

    这个模式不依赖手工或自动组件布局，适合先验证状态序列：
    只要矫正后的显示条位置稳定，亮灭形态就可以用 Hamming 相似度匹配。
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    high = float(np.percentile(gray, 99))
    threshold = max(70.0, high * 0.35)
    mask = (gray >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    h, w = mask.shape[:2]
    bits: list[str] = []
    on_cells = 0
    for r in range(grid_rows):
        y1 = int(round(r * h / grid_rows))
        y2 = int(round((r + 1) * h / grid_rows))
        for c in range(grid_cols):
            x1 = int(round(c * w / grid_cols))
            x2 = int(round((c + 1) * w / grid_cols))
            cell = mask[y1:y2, x1:x2]
            if cell.size == 0:
                bits.append("0")
                continue
            on = float(np.mean(cell)) >= 0.018
            bits.append("1" if on else "0")
            on_cells += int(on)

    return DisplayState(
        values={
            "visual_bits": "".join(bits),
            "grid_cols": grid_cols,
            "grid_rows": grid_rows,
            "on_cells": on_cells,
            "roi_mean": float(np.mean(gray)),
            "roi_std": float(np.std(gray)),
            "roi_mid_ratio": float(np.mean((gray > 25) & (gray < 150))),
            "roi_bright_ratio": float(np.mean(gray > 180)),
        },
        confidence=on_cells / max(grid_cols * grid_rows, 1),
    )


def display_state_to_vector(state: DisplayState) -> str:
    """将结构化状态转为字符串向量（用于相似度计算和去重）"""
    parts = []
    for k, v in state.values.items():
        if isinstance(v, bool):
            parts.append(f"{k}={int(v)}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)


def display_state_similarity(a: DisplayState, b: DisplayState) -> float:
    """两个结构化状态的相似度 [0, 1]"""
    if _is_bitstring(a.values.get("visual_bits")) and _is_bitstring(b.values.get("visual_bits")):
        return _bitstring_similarity(str(a.values["visual_bits"]), str(b.values["visual_bits"]))
    all_keys = set(a.values.keys()) | set(b.values.keys())
    if not all_keys:
        return 1.0
    matches = 0
    for k in all_keys:
        va = a.values.get(k)
        vb = b.values.get(k)
        if va is None or vb is None:
            continue
        if isinstance(va, bool) and isinstance(vb, bool):
            matches += 1 if va == vb else 0
        else:
            matches += 1 if str(va) == str(vb) else 0
    return matches / len(all_keys)


def _is_bitstring(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 0 and set(value) <= {"0", "1"}


def _bitstring_similarity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    same = sum(1 for i in range(n) if a[i] == b[i])
    return same / max(len(a), len(b))


def display_state_diff(a: DisplayState, b: DisplayState) -> dict[str, Any]:
    """返回两个状态的差异"""
    diff = {}
    all_keys = set(a.values.keys()) | set(b.values.keys())
    for k in all_keys:
        va = a.values.get(k); vb = b.values.get(k)
        if va != vb:
            diff[k] = {"from": va, "to": vb}
    return diff


# ─── 多帧融合 ────────────────────────────────────

def fuse_display_states(
    states: list[DisplayState],
    window_size: int = 5,
    min_on_ratio: float = 0.4,
) -> list[FusedDisplayState]:
    """滑动窗口多帧融合，消除刷新频闪

    Args:
        states: 逐帧状态列表
        window_size: 融合窗口大小
        min_on_ratio: 窗口内亮帧比例≥此值才判为亮

    Returns:
        融合后的稳定状态列表（每个窗口一个）
    """
    if not states:
        return []

    # 收集所有 key
    all_keys = set()
    for s in states:
        all_keys.update(s.values.keys())
    all_keys = sorted(all_keys)

    fused: list[FusedDisplayState] = []
    for i in range(0, len(states), max(1, window_size // 2)):
        window = states[i:i + window_size]
        if not window:
            break

        fused_vals: dict[str, Any] = {}
        for key in all_keys:
            vals = [s.values.get(key) for s in window if key in s.values]
            if not vals:
                continue

            if key == "visual_bits" and all(_is_bitstring(v) for v in vals):
                width = min(len(str(v)) for v in vals)
                fused_bits = []
                for bit_i in range(width):
                    on_ratio = sum(1 for v in vals if str(v)[bit_i] == "1") / max(len(vals), 1)
                    fused_bits.append("1" if on_ratio >= min_on_ratio else "0")
                fused_vals[key] = "".join(fused_bits)
            elif isinstance(vals[0], bool):
                on_ratio = sum(1 for v in vals if v) / max(len(vals), 1)
                fused_vals[key] = on_ratio >= min_on_ratio
            elif key in {"grid_cols", "grid_rows"}:
                fused_vals[key] = vals[0]
            elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
                fused_vals[key] = float(np.mean(vals))
            else:
                # 字符串值（数字）：取出现最多的
                from collections import Counter
                counter = Counter(str(v) for v in vals)
                fused_vals[key] = counter.most_common(1)[0][0]

        if fused_vals:
            bits = fused_vals.get("visual_bits")
            if _is_bitstring(bits):
                fused_vals["on_cells"] = str(bits).count("1")
            fused.append(FusedDisplayState(
                values=fused_vals,
                fused_frames=len(window),
                confidence=sum(1 for k, v in fused_vals.items()
                              if isinstance(v, bool)) / max(len(fused_vals), 1),
            ))

    return fused


# ─── 状态机（简化版） ────────────────────────────

# 复用了 running_led_core 的思路但重新实现，
# 因为状态表达方式不同（结构化 dict 而非二值矩阵）

class DisplayStateMachine:
    """数码管显示状态机"""

    def __init__(self, model: KeyDisplayModel):
        self.model = model
        self.states = model.states
        self.sequence = model.sequence
        self.state_lookup = {s.id: s for s in self.states}
        self.current_index = 0
        self.hold_frames = 0
        self.events: list[DisplayEvent] = []
        self.trace: list[dict] = []

    def reset(self) -> None:
        self.current_index = 0; self.hold_frames = 0
        self.events = []; self.trace = []

    def process_fused_state(self, fused: FusedDisplayState,
                             frame_index: int, time_sec: float) -> dict:
        """处理一个融合后的稳定状态"""
        if not self.states:
            return {"ok": True, "transition": "unknown"}

        cs = self.states[self.current_index]
        ni = (self.current_index + 1) % len(self.states)
        ns = self.states[ni]

        # 计算与当前状态/下一状态的相似度
        sim_current = _values_similarity(fused.values, cs.values)
        sim_next = _values_similarity(fused.values, ns.values)

        # 找最佳匹配
        best_idx, best_sim = self._find_best(fused.values)

        transition = "unknown"
        ok = True
        event_type = ""
        matched_id = cs.id

        th_sim = 0.85  # 状态相似度阈值

        if sim_current >= th_sim:
            self.hold_frames += 1
            transition = "stay"
            matched_id = cs.id
        elif sim_next >= th_sim:
            self.current_index = ni
            self.hold_frames = 0
            transition = "advance"
            matched_id = ns.id
        elif best_sim >= th_sim and best_idx > self.current_index + 1:
            skipped = best_idx - self.current_index
            prev = self.states[max(0, best_idx - skipped)].id
            self.current_index = best_idx
            self.hold_frames = 1
            transition = "advance"
            matched_id = self.states[best_idx].id
            event_type = "skipped_state"
            self._add_event(frame_index, time_sec, "skipped_state",
                            prev, matched_id, f"跳过 {skipped} 个: {prev} -> {matched_id}")
        elif best_sim >= th_sim and best_idx < self.current_index:
            matched_id = self.states[best_idx].id
            event_type = "wrong_order"
            self._add_event(frame_index, time_sec, "wrong_order",
                            cs.id, matched_id, f"倒退: 期望 {cs.id}，匹配到 {matched_id}")
        else:
            event_type = "unknown_state"
            ok = False
            self._add_event(frame_index, time_sec, "unknown_state",
                            cs.id, f"best={best_sim:.2f}", "无法匹配任何关键状态")

        result = {
            "frame_index": frame_index, "time_sec": time_sec,
            "expected_state": cs.id, "matched_state": matched_id,
            "next_state": ns.id, "transition": transition,
            "hold_frames": self.hold_frames, "ok": ok, "event_type": event_type,
        }
        self.trace.append(result)
        return result

    def _find_best(self, values: dict) -> tuple[int, float]:
        bi, bs = -1, -1.0
        for i, s in enumerate(self.states):
            sim = _values_similarity(values, s.values)
            if sim > bs:
                bs, bi = sim, i
        return bi, bs

    def _add_event(self, fi, ts, typ, exp, mat, msg):
        self.events.append(DisplayEvent(fi, ts, typ, exp, mat, msg))

    def finalize(self) -> list[DisplayEvent]:
        fe = []
        visited = set(r["matched_state"] for r in self.trace if r["matched_state"] in self.state_lookup)
        for sid in self.sequence:
            if sid not in visited:
                lf = self.trace[-1]["frame_index"] if self.trace else 0
                lt = self.trace[-1]["time_sec"] if self.trace else 0.0
                e = DisplayEvent(lf, lt, "missing_state", sid, "",
                                 f"检测结束，{sid} 从未到达")
                fe.append(e); self.events.append(e)
        return fe


def _values_similarity(a: dict, b: dict) -> float:
    """两个 dict 状态值的相似度"""
    if _is_bitstring(a.get("visual_bits")) and _is_bitstring(b.get("visual_bits")):
        return _bitstring_similarity(str(a["visual_bits"]), str(b["visual_bits"]))

    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 1.0
    matches = 0
    for k in keys:
        va = a.get(k); vb = b.get(k)
        if va is None or vb is None:
            continue
        if str(va) == str(vb):
            matches += 1
        elif isinstance(va, bool) and isinstance(vb, bool):
            pass  # 不匹配
    return matches / len(keys)


# ─── 工具函数 ────────────────────────────────────

def jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): jsonable(v2) for k, v2 in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
