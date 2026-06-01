"""
dot_panel_sequence_inspect.py

点阵面板序列检测：从标准视频学习点位模板+关键帧序列，然后对同一/不同视频校验。

流程:
  1. 全视频最大投影 → 自动ROI + 旋转矫正
  2. 用全亮投影检测所有LED点位坐标（模板）
  3. 逐帧检测每个点位亮灭 → 二值状态向量
  4. 去重得到关键帧序列
  5. 用状态机校验（同 running_led_core 思路）

Example:
  python scripts/dot_panel_sequence_inspect.py ^
    --video D:/jiemi/新建文件夹/Video_20260529135514029.avi ^
    --out output/dot_panel_inspect/Video_20260529135514029
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_pattern_probe import merge_close_points, imwrite_unicode
from scan_refresh_seven_segment_probe import detect_display_roi, crop_roi, Roi, write_json


# ─── 点检测 ───────────────────────────────────────────────────────────────────

def detect_all_leds(panel: np.ndarray, merge_radius: float = 25.0) -> list[dict[str, float]]:
    """检测面板上所有LED点位（彩色+白色）。"""
    b, g, r = cv2.split(panel)
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask_color = ((v >= 100) & (s >= 40) & (r >= 100) & (g >= 50)).astype(np.uint8) * 255
    mask_white = ((v >= 180) & (s < 80)).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask_color, mask_white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
    roi_area = panel.shape[0] * panel.shape[1]
    normal_max_area = 2000
    split_max_area = 15000

    points = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area < 6 or area > split_max_area:
            continue
        if w < 2 or h < 2:
            continue
        circularity = area / (3.14159 / 4.0 * max(w, h) ** 2)
        aspect = min(w, h) / max(w, h, 1)
        if circularity < 0.25 or aspect < 0.3:
            continue
        if area <= normal_max_area and w <= 60 and h <= 60:
            points.append({"x": float(centers[idx][0]), "y": float(centers[idx][1]),
                          "area": float(area), "w": float(w), "h": float(h)})
        elif area > normal_max_area:
            sub = split_large_blob((labels == idx).astype(np.uint8) * 255, x, y, w, h)
            points.extend(sub)

    return merge_close_points(points, merge_radius)


def split_large_blob(blob_mask: np.ndarray, ox: int, oy: int, bw: int, bh: int) -> list[dict[str, float]]:
    region = blob_mask[oy:oy+bh, ox:ox+bw]
    if region.size == 0:
        return []
    dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(dist, dist.max() * 0.4, 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    n, lbl, st, ctr = cv2.connectedComponentsWithStats(peaks, 8)
    sub_points = []
    for i in range(1, n):
        sub_points.append({"x": float(ctr[i][0]) + ox, "y": float(ctr[i][1]) + oy,
                          "area": float(st[i][4]), "w": float(st[i][2]), "h": float(st[i][3])})
    if len(sub_points) <= 1:
        return [{"x": ox + bw / 2.0, "y": oy + bh / 2.0, "area": float(bw * bh), "w": float(bw), "h": float(bh)}]
    return sub_points


# ─── ROI 对齐 ─────────────────────────────────────────────────────────────────

def compute_alignment(panel: np.ndarray, points: list[dict[str, float]]) -> float:
    """用点位行拟合计算面板倾斜角度。"""
    if len(points) < 5:
        return 0.0
    sorted_pts = sorted(points, key=lambda p: p["y"])
    rows = []
    current_row = [sorted_pts[0]]
    for p in sorted_pts[1:]:
        if p["y"] - current_row[-1]["y"] < 30:
            current_row.append(p)
        else:
            rows.append(current_row)
            current_row = [p]
    rows.append(current_row)

    angles = []
    for row in rows:
        if len(row) >= 3:
            xs = np.array([p["x"] for p in row])
            ys = np.array([p["y"] for p in row])
            if xs.max() - xs.min() > 50:
                k = np.polyfit(xs, ys, 1)[0]
                angles.append(np.degrees(np.arctan(k)))
    return float(np.median(angles)) if angles else 0.0


def align_panel(image: np.ndarray, roi: Roi, angle: float) -> np.ndarray:
    """裁ROI后旋转矫正。"""
    panel = crop_roi(image, roi)
    if abs(angle) < 0.01:
        return panel
    h, w = panel.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(panel, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


# ─── 逐帧亮灭检测 ─────────────────────────────────────────────────────────────

def check_point_states(panel: np.ndarray, template_points: list[dict[str, float]],
                       radius: int = 15, threshold: float = 0.4) -> np.ndarray:
    """对每个模板点位检测是否亮着，返回二值向量。"""
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    h, w = v.shape
    states = np.zeros(len(template_points), dtype=np.int8)

    for i, pt in enumerate(template_points):
        cx, cy = int(round(pt["x"])), int(round(pt["y"]))
        x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
        y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
        if x1 >= x2 or y1 >= y2:
            continue
        region = v[y1:y2, x1:x2]
        # 用90th percentile判断是否亮
        brightness = float(np.percentile(region, 90)) / 255.0
        states[i] = 1 if brightness >= threshold else 0

    return states


# ─── 关键帧去重 ────────────────────────────────────────────────────────────────

def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    return 1.0 - float(np.sum(a != b)) / len(a)


@dataclass
class KeyFrame:
    id: str
    index: int
    on_count: int
    state: np.ndarray
    first_frame: int
    hold_frames: int


def deduplicate_states(all_states: list[tuple[int, np.ndarray]],
                       dedupe_sim: float = 0.97,
                       min_stable: int = 2) -> list[KeyFrame]:
    """从时间序列的帧状态去重得到关键帧。"""
    if not all_states:
        return []
    keys: list[KeyFrame] = []
    candidate = all_states[0][1].copy()
    candidate_frame = all_states[0][0]
    count = 1

    for frame_idx, state in all_states[1:]:
        if hamming_similarity(candidate, state) >= dedupe_sim:
            count += 1
        else:
            if count >= min_stable:
                kid = f"K{len(keys):03d}"
                keys.append(KeyFrame(id=kid, index=len(keys), on_count=int(candidate.sum()),
                                     state=candidate.copy(), first_frame=candidate_frame, hold_frames=count))
            candidate = state.copy()
            candidate_frame = frame_idx
            count = 1

    if count >= min_stable:
        kid = f"K{len(keys):03d}"
        keys.append(KeyFrame(id=kid, index=len(keys), on_count=int(candidate.sum()),
                             state=candidate.copy(), first_frame=candidate_frame, hold_frames=count))
    return keys


# ─── 状态机校验 ────────────────────────────────────────────────────────────────

def verify_sequence(all_states: list[tuple[int, np.ndarray]], key_frames: list[KeyFrame],
                    sim_threshold: float = 0.92) -> dict[str, Any]:
    """用状态机校验帧序列是否按关键帧顺序推进。"""
    if not key_frames:
        return {"status": "ERROR", "message": "无关键帧"}

    cursor = 0
    hold = 0
    hits = 0
    events = []

    for frame_idx, state in all_states:
        current = key_frames[cursor]
        next_idx = cursor + 1
        sim_current = hamming_similarity(state, current.state)

        if sim_current >= sim_threshold:
            hold += 1
            continue

        if next_idx < len(key_frames):
            sim_next = hamming_similarity(state, key_frames[next_idx].state)
            if sim_next >= sim_threshold:
                hits += 1
                cursor = next_idx
                hold = 1
                continue

        # 尝试往前看几步
        found = False
        for skip in range(2, min(5, len(key_frames) - cursor)):
            ahead = cursor + skip
            if ahead < len(key_frames):
                sim_ahead = hamming_similarity(state, key_frames[ahead].state)
                if sim_ahead >= sim_threshold:
                    hits += skip
                    cursor = ahead
                    hold = 1
                    found = True
                    break
        if not found:
            hold += 1

    coverage = (cursor + 1) / len(key_frames)
    passed = coverage >= 0.90
    return {
        "status": "PASS" if passed else "NG",
        "key_frame_count": len(key_frames),
        "cursor_final": cursor,
        "coverage": coverage,
        "hits": hits,
        "events": events,
    }


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="点阵面板序列检测")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output/dot_panel_inspect"))
    parser.add_argument("--stride", type=int, default=2, help="抽帧间隔")
    parser.add_argument("--point-radius", type=int, default=15, help="点位检测半径")
    parser.add_argument("--on-threshold", type=float, default=0.4, help="亮灭判断阈值")
    parser.add_argument("--dedupe-sim", type=float, default=0.97, help="去重相似度")
    parser.add_argument("--verify-sim", type=float, default=0.92, help="校验相似度")
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] 构建最大投影: {args.video.name}")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_proj = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_proj is None:
            max_proj = frame.copy()
        else:
            np.maximum(max_proj, frame, out=max_proj)
        idx += 1
    cap.release()
    print(f"    帧数: {idx}, FPS: {fps:.1f}")

    # ROI
    roi = detect_display_roi(max_proj, 40, 0.06)
    print(f"    ROI: x={roi.x} y={roi.y} w={roi.w} h={roi.h}")

    # 对齐
    panel_proj = crop_roi(max_proj, roi)
    points_raw = detect_all_leds(panel_proj, merge_radius=25)
    angle = compute_alignment(panel_proj, points_raw)
    print(f"    倾斜角: {angle:.2f}°")

    panel_aligned = align_panel(max_proj, roi, angle)

    # 检测模板点位
    template_points = detect_all_leds(panel_aligned, merge_radius=25)
    print(f"[2] 模板点位: {len(template_points)} 个")

    # 保存模板标注图
    annotated = panel_aligned.copy()
    for pt in template_points:
        cv2.circle(annotated, (int(pt["x"]), int(pt["y"])), 10, (0, 255, 0), 2)
    imwrite_unicode(out_dir / "template_points.jpg", annotated)

    # [3] 逐帧检测
    print(f"[3] 逐帧检测 (stride={args.stride})...")
    cap = cv2.VideoCapture(str(args.video))
    all_states: list[tuple[int, np.ndarray]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride != 0:
            frame_idx += 1
            continue
        panel = align_panel(frame, roi, angle)
        state = check_point_states(panel, template_points, args.point_radius, args.on_threshold)
        all_states.append((frame_idx, state))
        frame_idx += 1
    cap.release()
    print(f"    处理帧数: {len(all_states)}")

    # [4] 去重关键帧
    key_frames = deduplicate_states(all_states, args.dedupe_sim)
    print(f"[4] 关键帧: {len(key_frames)} 个")
    for kf in key_frames[:10]:
        print(f"    {kf.id}: on={kf.on_count}/{len(template_points)} frame={kf.first_frame} hold={kf.hold_frames}")
    if len(key_frames) > 10:
        print(f"    ... 共 {len(key_frames)} 个")

    # [5] 自检校验
    result = verify_sequence(all_states, key_frames, args.verify_sim)
    print(f"[5] 自检结果: {result['status']} coverage={result['coverage']:.1%}")

    # 保存模型
    model = {
        "video": str(args.video),
        "roi": asdict(roi),
        "angle": angle,
        "template_points": template_points,
        "point_count": len(template_points),
        "key_frames": [{"id": kf.id, "index": kf.index, "on_count": kf.on_count,
                        "first_frame": kf.first_frame, "hold_frames": kf.hold_frames,
                        "state_rle": rle_encode(kf.state)} for kf in key_frames],
        "key_frame_count": len(key_frames),
        "parameters": {"stride": args.stride, "point_radius": args.point_radius,
                       "on_threshold": args.on_threshold, "dedupe_sim": args.dedupe_sim},
        "self_check": result,
    }
    write_json(out_dir / "model.json", model)

    # 保存几个关键帧的可视化
    kf_dir = out_dir / "keyframes"
    kf_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    for kf in key_frames[:20]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf.first_frame)
        ok, frame = cap.read()
        if not ok:
            continue
        panel = align_panel(frame, roi, angle)
        ann = panel.copy()
        for i, pt in enumerate(template_points):
            cx, cy = int(pt["x"]), int(pt["y"])
            color = (0, 255, 0) if kf.state[i] else (0, 0, 100)
            cv2.circle(ann, (cx, cy), 8, color, 2)
        cv2.putText(ann, f"{kf.id} on={kf.on_count} f={kf.first_frame}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        imwrite_unicode(kf_dir / f"{kf.id}_f{kf.first_frame:04d}.jpg", ann)
    cap.release()

    print(f"\n输出: {out_dir}")
    print(f"  模型: model.json")
    print(f"  模板: template_points.jpg")
    print(f"  关键帧: keyframes/")


def rle_encode(arr: np.ndarray) -> str:
    if len(arr) == 0:
        return ""
    flat = arr.ravel().astype(np.int8)
    changes = np.diff(flat, prepend=-1, append=-1)
    runs = np.where(changes != 0)[0]
    lengths = np.diff(runs)
    values = flat[runs[:-1]]
    return ",".join(f"{int(v)}:{int(l)}" for v, l in zip(values, lengths))


if __name__ == "__main__":
    main()
