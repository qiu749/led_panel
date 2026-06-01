"""
run_display_pipeline — 显示条状态检测一键管道

自动流程：
  1. 读取视频，逐帧找显示条 ROI
  2. 透视矫正到标准尺寸
  3. 自动分割区域（投影法找数码管和图标位置）
  4. 逐帧解码状态
  5. 多帧融合消除频闪
  6. 学习关键状态序列
  7. 状态机自检
  8. 生成 review.html

用法：
  python scripts/run_display_pipeline.py --video D:/jiemi/Video_20260430154222007.avi --out output/display_run
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.display_panel_core import (
    PanelLayout, PanelRegion, DisplayState, FusedDisplayState,
    KeyDisplayState, KeyDisplayModel, DisplayEvent, DisplayStateMachine,
    find_display_roi, warp_display_panel, decode_display_state,
    decode_visual_grid_state, fuse_display_states, display_state_similarity,
    write_json, jsonable,
)


# ─── 自动布局生成 ─────────────────────────────────

def auto_layout(warped: np.ndarray, panel_w: int = 640, panel_h: int = 160) -> PanelLayout:
    """通过多级投影分析自动生成显示条布局

    策略：先用行投影找内容行，再用列投影分割每行的内容组件
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # 自适应二值化：更鲁棒地分割亮内容
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, -10)
    # 去掉小噪点
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    layout = PanelLayout(panel_width=panel_w, panel_height=panel_h)

    # 水平投影：找内容行
    h_proj_all = np.sum(binary > 0, axis=1) / 255
    h_th = np.max(h_proj_all) * 0.05
    h_proj_bin = h_proj_all > h_th

    row_segments = []
    in_row = False; row_start = 0
    for r in range(panel_h):
        if h_proj_bin[r] and not in_row:
            row_start = r; in_row = True
        elif not h_proj_bin[r] and in_row:
            if r - row_start >= 5:
                row_segments.append((row_start, r))
            in_row = False
    if in_row and panel_h - row_start >= 5:
        row_segments.append((row_start, panel_h))

    if not row_segments:
        row_segments = [(5, panel_h - 5)]

    # 对每行，做列投影分割
    region_id = 0
    for rs, re in row_segments:
        row_binary = binary[rs:re, :]
        v_proj = np.sum(row_binary > 0, axis=0) / 255
        v_th = np.max(v_proj) * 0.08
        v_bin = v_proj > v_th

        col_segs = []
        in_col = False; col_start = 0
        for c in range(panel_w):
            if v_bin[c] and not in_col:
                col_start = c; in_col = True
            elif not v_bin[c] and in_col:
                if c - col_start >= 6:
                    col_segs.append((col_start, c))
                in_col = False
        if in_col and panel_w - col_start >= 6:
            col_segs.append((col_start, panel_w))

        if not col_segs:
            # 整行作为一个区域
            col_segs = [(0, panel_w)]

        for cs, ce in col_segs:
            cw = ce - cs
            rh = re - rs
            # 粗分类：用宽高比和尺寸
            if cw >= 40 and rh >= 25:
                rtype = "seven_segment_digit"
            elif cw >= 18 and cw < 40 and rh >= 12:
                rtype = "icon"
            elif cw >= 6 and rh >= 6:
                rtype = "dot"
            else:
                rtype = "icon"
            layout.regions.append(PanelRegion(
                name=f"{rtype}_{region_id + 1}",
                type=rtype,
                x=cs, y=rs + (rh - min(rh, panel_h // 3)) // 2,
                w=cw, h=min(rh, panel_h // 3),
            ))
            region_id += 1

    # 拆分过大的数码管区域为独立数字
    final_regions = []
    for r in layout.regions:
        if r.type == "seven_segment_digit" and r.w > 80:
            # 按 40px 宽度拆分
            n_digits = max(1, r.w // 40)
            d_w = r.w // n_digits
            for d in range(n_digits):
                final_regions.append(PanelRegion(
                    name="digit", type="seven_segment_digit",
                    x=r.x + d * d_w, y=r.y, w=d_w, h=r.h,
                ))
        else:
            final_regions.append(r)
    layout.regions = final_regions

    # 保证区域名全局唯一，避免 decode_display_state 写入 dict 时互相覆盖。
    counters: Counter[str] = Counter()
    for r in layout.regions:
        prefix = "digit" if r.type == "seven_segment_digit" else r.type
        counters[prefix] += 1
        r.name = f"{prefix}_{counters[prefix]}"

    # 如果没有足够区域，补充默认
    if len(layout.regions) < 3:
        layout.regions.clear()
        # 水平三分
        third = panel_w // 5
        for i in range(3):
            dx = 30 + i * third * 1.5
            if dx + 100 <= panel_w:
                layout.regions.append(PanelRegion(
                    name=f"digit_{i+1}", type="seven_segment_digit",
                    x=int(dx), y=10, w=80, h=panel_h // 3,
                ))
        # 左右图标
        if 60 < panel_w:
            layout.regions.append(PanelRegion(
                name="icon_left", type="icon", x=5, y=panel_h//4, w=40, h=panel_h//3))
        if panel_w - 60 > 0:
            layout.regions.append(PanelRegion(
                name="icon_right", type="icon", x=panel_w-50, y=panel_h//4, w=40, h=panel_h//3))

    return layout


# ─── 主流程 ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="数码管显示条状态机检测 — 一键管道")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output") / "display_run")
    parser.add_argument("--sample-fps", type=float, default=7.5)
    parser.add_argument("--layout", type=Path, default=None, help="手动布局 JSON（不指定则自动生成）")
    parser.add_argument("--state-mode", choices=["visual_grid", "layout"], default="visual_grid",
                        help="visual_grid=直接用矫正图学习状态；layout=按数码管/图标区域解码")
    parser.add_argument("--grid-cols", type=int, default=48)
    parser.add_argument("--grid-rows", type=int, default=12)
    parser.add_argument("--min-active-cells", type=int, default=3,
                        help="低于该亮格数的状态视为废帧，不参与关键状态学习/推进")
    parser.add_argument("--blank-event-frames", type=int, default=8,
                        help="连续废帧达到该数量时，在 summary 中记录空白段数量")
    parser.add_argument("--occlusion-event-frames", type=int, default=4,
                        help="连续遮挡帧达到该数量时，在 summary 中记录遮挡段数量")
    parser.add_argument("--window-size", type=int, default=5, help="多帧融合窗口")
    parser.add_argument("--min-on-ratio", type=float, default=0.4, help="融合亮判阈值")
    parser.add_argument("--panel-w", type=int, default=640)
    parser.add_argument("--panel-h", type=int, default=160)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--debug-image-limit", type=int, default=500,
                        help="最多保存多少组 review 调试图；0 表示全部保存")
    parser.add_argument("--no-roi-reuse", action="store_true",
                        help="关闭 ROI 复用；默认会在弱亮/单灯帧复用上一帧 ROI")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    debug_dir = out / "debug"; warp_dir = debug_dir / "warped"
    roi_dir = debug_dir / "roi_marks"
    warp_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. 读取视频 ───────────────────────────────
    print(f"[1] 打开视频: {args.video}")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened(): print("错误"); return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 7.5
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"    总帧: {total}, FPS: {fps:.2f}, 分辨率: {src_w}x{src_h}")

    stride = max(1, round(fps / args.sample_fps))

    # ─── 2. 逐帧找 ROI 并矫正 ─────────────────────
    print(f"[2] 检测显示条 ROI (stride={stride})...")
    warped_frames: list[np.ndarray] = []
    raw_states: list[DisplayState] = []
    rois: list[tuple] = []
    roi_reused_flags: list[bool] = []
    frame_indices: list[int] = []
    debug_frames_saved: set[int] = set()

    idx = 0; found_any = False
    last_roi = None
    reused_roi_count = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % stride != 0:
            idx += 1; continue

        fi = idx; ts = idx / fps
        detected_roi = find_display_roi(frame)
        roi = detected_roi
        roi_reused = False
        if (
            detected_roi is not None
            and last_roi is not None
            and not _is_reliable_display_roi(detected_roi, src_w)
            and not args.no_roi_reuse
        ):
            roi = last_roi
            roi_reused = True
        elif (
            detected_roi is not None
            and last_roi is None
            and not _is_reliable_display_roi(detected_roi, src_w)
        ):
            roi = None
        elif roi is None and last_roi is not None and not args.no_roi_reuse:
            roi = last_roi
            roi_reused = True

        if roi is not None:
            found_any = True
            try:
                warped = warp_display_panel(frame, roi, (args.panel_w, args.panel_h))
                warped_frames.append(warped)
                rois.append(roi)
                roi_reused_flags.append(roi_reused)
                frame_indices.append(fi)
                if not roi_reused and _is_reliable_display_roi(roi, src_w):
                    last_roi = roi
                else:
                    reused_roi_count += 1

                # 保存调试图
                if args.debug_image_limit == 0 or len(debug_frames_saved) < args.debug_image_limit:
                    cv2.imwrite(str(warp_dir / f"frame_{fi:06d}.jpg"), warped)
                    # 在原图上标记 ROI
                    marked = frame.copy()
                    _draw_display_roi(marked, roi)
                    cv2.imwrite(str(roi_dir / f"roi_{fi:06d}.jpg"), marked)
                    debug_frames_saved.add(fi)

                if args.max_frames and len(warped_frames) >= args.max_frames:
                    break
            except ValueError:
                pass

        idx += 1
        if args.max_frames and len(warped_frames) >= args.max_frames:
            break

    cap.release()
    print(f"    找到 {len(warped_frames)} 帧含显示条 ROI")
    if reused_roi_count:
        print(f"    其中 {reused_roi_count} 帧复用上一帧 ROI（用于弱亮/单灯状态）")

    if not found_any:
        print("错误: 未在任何帧中找到显示条 ROI"); return

    # ─── 3. 自动布局 ───────────────────────────────
    if args.state_mode == "visual_grid":
        print(f"[3] 使用 visual_grid 自动状态特征 ({args.grid_cols}x{args.grid_rows})...")
        layout = PanelLayout(panel_width=args.panel_w, panel_height=args.panel_h, regions=[])
    elif args.layout:
        layout = PanelLayout.from_json(args.layout)
    else:
        print(f"[3] 自动生成布局（基于最清晰亮屏矫正图）...")
        layout_i = _select_layout_frame(warped_frames)
        layout = auto_layout(warped_frames[layout_i], args.panel_w, args.panel_h)
        print(f"    布局参考帧: {frame_indices[layout_i]}")

    print(f"    布局: {len(layout.regions)} 个区域")
    for r in layout.regions:
        print(f"      [{r.type}] {r.name}: ({r.x},{r.y}) {r.w}x{r.h}")

    # 保存布局
    layout_path = out / "display_layout.json"
    layout.to_json(layout_path)
    print(f"    布局已保存: {layout_path}")

    # ─── 4. 逐帧解码状态 ──────────────────────────
    print(f"[4] 解码 {len(warped_frames)} 帧状态...")
    raw_states = []
    for i, (fi, warped) in enumerate(zip(frame_indices, warped_frames)):
        if args.state_mode == "visual_grid":
            state = decode_visual_grid_state(warped, args.grid_cols, args.grid_rows)
        else:
            state = decode_display_state(warped, layout)
        state.frame_index = fi
        state.time_sec = fi / fps
        raw_states.append(state)

    # ─── 5. 多帧融合 ──────────────────────────────
    print(f"[5] 多帧融合 (window={args.window_size}, min_on={args.min_on_ratio})...")
    fused = fuse_display_states(raw_states, args.window_size, args.min_on_ratio)
    print(f"    {len(raw_states)} 帧 → {len(fused)} 个融合状态")

    # ─── 6. 去重 → 关键状态序列 ──────────────────
    print(f"[6] 状态去重生成关键序列...")
    key_states: list[KeyDisplayState] = []
    sequence: list[str] = []
    prev_fused = None
    hold_counts: list[int] = []
    cur_hold = 0
    skipped_blank_fused = 0
    skipped_occluded_fused = 0
    blank_intervals = 0
    occlusion_intervals = 0
    blank_run = 0
    occlusion_run = 0

    for i, fs in enumerate(fused):
        if _is_occluded_fused_state(fs, args.min_active_cells):
            skipped_occluded_fused += 1
            occlusion_run += 1
            blank_run = 0
            if occlusion_run == args.occlusion_event_frames:
                occlusion_intervals += 1
            continue
        if _is_blank_fused_state(fs, args.min_active_cells):
            skipped_blank_fused += 1
            blank_run += 1
            occlusion_run = 0
            if blank_run == args.blank_event_frames:
                blank_intervals += 1
            continue
        blank_run = 0
        occlusion_run = 0

        if prev_fused is None:
            prev_fused = fs; cur_hold = 1; continue

        sim = _fused_similarity(prev_fused, fs)
        if sim >= 0.85:
            cur_hold += 1
        else:
            if cur_hold >= 2:  # 稳定确认
                sid = f"K{len(key_states):03d}"
                key_states.append(KeyDisplayState(
                    id=sid, index=len(key_states),
                    values=dict(prev_fused.values),
                    median_hold_frames=cur_hold,
                ))
                sequence.append(sid)
                hold_counts.append(cur_hold)
            prev_fused = fs; cur_hold = 1

    # 最后一个
    if prev_fused is not None and cur_hold >= 2:
        sid = f"K{len(key_states):03d}"
        key_states.append(KeyDisplayState(
            id=sid, index=len(key_states),
            values=dict(prev_fused.values),
            median_hold_frames=cur_hold,
        ))
        sequence.append(sid)
        hold_counts.append(cur_hold)

    print(f"    {len(fused)} 融合状态 → {len(key_states)} 个关键状态")
    if skipped_blank_fused:
        print(f"    跳过 {skipped_blank_fused} 个废帧/空白融合状态")
    if skipped_occluded_fused:
        print(f"    跳过 {skipped_occluded_fused} 个遮挡/人工操作融合状态")

    # 填充平均 hold 值
    avg_hold = int(np.mean(hold_counts)) if hold_counts else 5
    for ks in key_states:
        ks.median_hold_frames = max(avg_hold, ks.median_hold_frames)

    # ─── 7. 保存模型 ──────────────────────────────
    model = KeyDisplayModel(
        version=1, source_video=str(args.video),
        panel_layout=layout,
        states=key_states, sequence=sequence,
    )
    model_path = out / "key_states.json"
    write_json(model_path, model.to_dict())

    # 输出状态预览
    preview_dir = out / "key_state_previews"
    preview_dir.mkdir(exist_ok=True)
    for ks in key_states:
        _save_state_preview(ks, preview_dir)

    # ─── 8. 状态机自检 ────────────────────────────
    print(f"[7] 状态机自检...")
    sm = DisplayStateMachine(model)
    trace_rows = []
    fusion_step = max(1, args.window_size // 2)
    for i, fs in enumerate(fused):
        src_i = min(i * fusion_step + args.window_size // 2, len(frame_indices) - 1)
        fi = frame_indices[src_i]
        ts = raw_states[src_i].time_sec
        skip_type = ""
        if _is_occluded_fused_state(fs, args.min_active_cells):
            skip_type = "occluded_skip"
        elif _is_blank_fused_state(fs, args.min_active_cells):
            skip_type = "blank_skip"
        if skip_type:
            r = {
                "frame_index": fi, "time_sec": ts,
                "expected_state": sm.states[sm.current_index].id if sm.states else "",
                "matched_state": "BLANK", "next_state": "",
                "transition": skip_type, "hold_frames": sm.hold_frames,
                "ok": True, "event_type": "",
            }
            if skip_type == "occluded_skip":
                r["matched_state"] = "OCCLUDED"
            if fi in debug_frames_saved:
                r["roi_image"] = f"debug/roi_marks/roi_{fi:06d}.jpg"
                r["warped_image"] = f"debug/warped/frame_{fi:06d}.jpg"
            if src_i >= 0:
                r["roi_reused"] = roi_reused_flags[src_i]
            trace_rows.append(r)
            continue

        r = sm.process_fused_state(fs, fi, ts)
        if fi in debug_frames_saved:
            r["roi_image"] = f"debug/roi_marks/roi_{fi:06d}.jpg"
            r["warped_image"] = f"debug/warped/frame_{fi:06d}.jpg"
        if src_i >= 0:
            r["roi_reused"] = roi_reused_flags[src_i]
        trace_rows.append(r)

    sm.finalize()

    # ─── 9. 输出文件 ──────────────────────────────
    trace_path = out / "state_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index","time_sec","expected_state",
                                          "matched_state","next_state","transition",
                                          "hold_frames","ok","event_type"])
        w.writeheader()
        for r in trace_rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    events_path = out / "events.json"
    write_json(events_path, {"events": [{
        "frame_index": e.frame_index, "time_sec": e.time_sec,
        "type": e.type, "expected_state": e.expected_state,
        "matched_state": e.matched_state, "message": e.message,
    } for e in sm.events]})

    summary = {
        "overall_ok": all(r["ok"] for r in trace_rows) and len(sm.events) == 0,
        "frame_count": total, "sampled_count": len(warped_frames),
        "roi_reused_count": reused_roi_count,
        "blank_fused_skipped": skipped_blank_fused,
        "occluded_fused_skipped": skipped_occluded_fused,
        "blank_interval_count": blank_intervals,
        "occlusion_interval_count": occlusion_intervals,
        "min_active_cells": args.min_active_cells,
        "fused_count": len(fused), "key_state_count": len(key_states),
        "event_count": len(sm.events),
        "first_error": next(({"frame": r["frame_index"], "type": r["event_type"]}
                            for r in trace_rows if not r["ok"]), None),
    }
    write_json(out / "summary.json", summary)

    ok_s = "OK" if summary["overall_ok"] else "NG"
    print(f"\n===== 结果: {ok_s} =====")
    print(f"  关键状态: {len(key_states)}, 事件: {len(sm.events)}")
    for e in sm.events:
        print(f"    [{e.type}] 帧{e.frame_index}: {e.message}")

    # ─── 10. review.html ─────────────────────────
    print(f"[8] 生成 review.html...")
    _render_review_html(model, trace_rows, sm.events, out / "review.html")
    print(f"    复查页面: {out / 'review.html'}")
    print(f"\n输出目录: {out}")


def _fused_similarity(a: FusedDisplayState, b: FusedDisplayState) -> float:
    av = a.values.get("visual_bits")
    bv = b.values.get("visual_bits")
    if _is_bitstring(av) and _is_bitstring(bv):
        return _bitstring_similarity(str(av), str(bv))

    keys = set(a.values.keys()) | set(b.values.keys())
    if not keys: return 1.0
    matches = 0
    for k in keys:
        va = a.values.get(k); vb = b.values.get(k)
        if va is None or vb is None: continue
        if str(va) == str(vb): matches += 1
    return matches / len(keys)


def _is_blank_fused_state(state: FusedDisplayState, min_active_cells: int) -> bool:
    try:
        return int(state.values.get("on_cells", 0)) < min_active_cells
    except (TypeError, ValueError):
        return False


def _is_occluded_fused_state(state: FusedDisplayState, min_active_cells: int) -> bool:
    """低亮但 ROI 中出现大面积灰色纹理，视为人手/操作遮挡。"""
    try:
        on_cells = int(state.values.get("on_cells", 0))
        mean = float(state.values.get("roi_mean", 0.0))
        std = float(state.values.get("roi_std", 0.0))
        mid_ratio = float(state.values.get("roi_mid_ratio", 0.0))
        bright_ratio = float(state.values.get("roi_bright_ratio", 0.0))
    except (TypeError, ValueError):
        return False

    if on_cells >= min_active_cells:
        return False
    # 纯黑/关灯帧通常 mean/std 都很低；手套、线缆、外壳遮挡会带来灰度和纹理。
    return bright_ratio < 0.003 and (
        (mean >= 8.0 and std >= 8.0) or mid_ratio >= 0.03
    )


def _is_bitstring(value) -> bool:
    return isinstance(value, str) and len(value) > 0 and set(value) <= {"0", "1"}


def _bitstring_similarity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    return sum(1 for i in range(n) if a[i] == b[i]) / max(len(a), len(b))


def _select_layout_frame(warped_frames: list[np.ndarray]) -> int:
    """选择最适合自动布局的矫正帧。

    第一张 ROI 帧经常只是暗帧或半刷新帧，用它做布局会把数码管拆坏。
    这里选择高亮像素最多的帧作为布局参考。
    """
    best_i = 0
    best_score = -1.0
    for i, warped in enumerate(warped_frames):
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        score = float(np.sum(gray > 180)) + float(np.percentile(gray, 99)) * 10.0
        if score > best_score:
            best_i = i
            best_score = score
    return best_i


def _draw_display_roi(frame: np.ndarray, roi) -> None:
    if hasattr(roi, "points"):
        pts = np.asarray(roi.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 3, cv2.LINE_AA)
        for i, p in enumerate(pts.reshape(-1, 2), 1):
            cv2.circle(frame, tuple(p), 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, str(i), tuple(p + np.array([6, -6])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)


def _is_reliable_display_roi(roi, frame_width: int) -> bool:
    """判断当前检测是否足够可靠，可以更新跟踪 ROI。

    单灯/少量灯状态也可能被 find_display_roi 检出，但角度和边界会不稳；
    这些帧应该复用上一帧稳定 ROI。
    """
    if not hasattr(roi, "component_count"):
        return True
    return (
        roi.component_count >= 8
        and roi.bright_span >= frame_width * 0.18
        and roi.confidence >= 0.35
    )


def _save_state_preview(state: KeyDisplayState, path: Path):
    """将关键状态保存为文本预览"""
    lines = [f"=== {state.id} ==="]
    bits = state.values.get("visual_bits")
    cols = int(state.values.get("grid_cols", 0) or 0)
    rows = int(state.values.get("grid_rows", 0) or 0)
    if _is_bitstring(bits) and cols > 0 and rows > 0:
        lines.append(f"  visual_grid: {cols}x{rows}")
        lines.append(f"  on_cells: {state.values.get('on_cells', '')}")
        lines.append("")
        for r in range(rows):
            row = str(bits)[r * cols:(r + 1) * cols]
            lines.append("  " + row.replace("0", ".").replace("1", "#"))
        (path / f"{state.id}.txt").write_text("\n".join(lines), encoding="utf-8")
        return

    for k, v in state.values.items():
        lines.append(f"  {k}: {v}")
    (path / f"{state.id}.txt").write_text("\n".join(lines), encoding="utf-8")


def _render_review_html(model, trace, events, output_path):
    states_js = []
    for s in model.states:
        states_js.append({"id": s.id, "values": s.values})

    config = {
        "states": states_js, "sequence": model.sequence,
        "trace": trace,
        "events": [{"frame_index": e.frame_index, "type": e.type, "message": e.message}
                   for e in events],
    }
    config_json = json.dumps(jsonable(config), ensure_ascii=False)

    html = f"""<!DOCTYPE html><html lang=zh-CN><meta charset=UTF-8>
<title>显示条状态检测复查</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:'Segoe UI',sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;overflow:hidden}}
#app{{display:flex;height:100vh;font-size:13px}}
#sidebar{{width:280px;min-width:280px;background:#16213e;border-right:1px solid #0f3460;overflow-y:auto;padding:8px}}
#sidebar h3{{font-size:13px;color:#8899cc;margin:6px 0 3px}}
.ks-item{{cursor:pointer;padding:4px;border-radius:4px;margin-bottom:2px}}
.ks-item:hover{{background:#1a2744}}.ks-item.active{{background:#1b3a5c;outline:2px solid #4a9eff}}
.ks-item.ng{{outline:2px solid #f44336}}
.ks-id{{font-size:11px;color:#aabbdd;margin-bottom:2px}}
.ks-val{{font-size:10px;color:#8899aa;line-height:1.3}}
#main{{flex:1;display:flex;flex-direction:column}}
#info{{height:70px;background:#16213e;border-top:1px solid #0f3460;padding:6px 12px;display:flex;gap:12px;flex-wrap:wrap;font-size:12px}}
.info .l{{color:#667799}} .info .v{{font-weight:600}}.v.ok{{color:#4caf50}}.v.ng{{color:#f44336}}
#detail{{flex:1;overflow-y:auto;padding:12px;background:#0d1117}}
#visuals{{display:grid;grid-template-columns:1.35fr 1fr;gap:12px;margin-bottom:12px}}
.panel{{background:#111827;border:1px solid #263247;border-radius:6px;padding:8px;min-height:120px}}
.panel h3{{font-size:12px;color:#8fb3ff;margin-bottom:6px;font-weight:600}}
.panel img{{display:block;max-width:100%;height:auto;border-radius:4px;background:#000}}
.empty{{display:flex;align-items:center;justify-content:center;min-height:120px;color:#667799;background:#080b10;border-radius:4px}}
#detail pre{{background:#1a1a2e;padding:8px;border-radius:4px;font-size:12px;color:#c0d0e0}}
#timeline{{height:35px;background:#0d1117;border-top:1px solid #0f3460;overflow:auto}}
#controls{{height:36px;background:#16213e;border-top:1px solid #0f3460;display:flex;align-items:center;gap:4px;padding:0 8px;font-size:12px}}
#controls button{{background:#1a2744;color:#c0d0e0;border:1px solid #0f3460;padding:3px 8px;border-radius:3px;cursor:pointer}}
#controls button.primary{{background:#1565c0;color:#fff}} #controls button.danger{{background:#c62828;color:#fff}}
#controls input{{flex:1;max-width:250px}}
.everr{{color:#ff8a80}}
</style>
<div id=app>
<div id=sidebar><h3>关键状态</h3><div id=klist></div></div>
<div id=main>
<div id=info>
<span class=info><span class=l>步:</span><span class=v id=fi>--</span></span>
<span class=info><span class=l>期望:</span><span class=v id=fe>--</span></span>
<span class=info><span class=l>匹配:</span><span class=v id=fm>--</span></span>
<span class=info><span class=l>过渡:</span><span class=v id=ftr>--</span></span>
<span class=info><span class=l>停留:</span><span class=v id=fh>--</span></span>
<span class=info><span class=l>结果:</span><span class=v id=fres>--</span></span>
</div>
<div id=detail>
<div id=visuals>
<div class=panel><h3>原图 ROI</h3><div id=roi-empty class=empty>无 ROI 图</div><img id=roi-img style=display:none></div>
<div class=panel><h3>矫正显示条</h3><div id=warp-empty class=empty>无矫正图</div><img id=warp-img style=display:none></div>
</div>
<pre id=detail-pre>选择帧查看详情</pre></div>
<div id=timeline><svg id=tsvg></svg></div>
<div id=controls>
<button id=bp class=primary>PLAY</button><button id=bpr>&lt;</button>
<span id=flbl>0/0</span><button id=bnx>&gt;</button><button id=bne class=danger>NG</button>
<input type=range id=sl min=0 value=0>
<div id=evlog style=min-width:120px></div></div></div></div>
<script>
(function(){{var C={config_json},T=C.trace,E=C.events||[],ci=0,pl=null
function gv(t,s,c){{return t[s]!==undefined?t[s]:t[c]}}
function sid(t,k){{return gv(t,k,k.replace(/_([a-z])/g,function(_,x){{return x.toUpperCase()}}))}}
function setImg(imgId,emptyId,src){{var img=document.getElementById(imgId),emp=document.getElementById(emptyId)
if(src){{img.src=src;img.style.display='block';emp.style.display='none'}}else{{img.removeAttribute('src');img.style.display='none';emp.style.display='flex'}}}}
function uf(i){{if(i<0||i>=T.length)return;ci=i;var t=T[i]
document.getElementById('fi').textContent=sid(t,'frame_index')
document.getElementById('fe').textContent=sid(t,'expected_state')
document.getElementById('fm').textContent=sid(t,'matched_state')
document.getElementById('ftr').textContent=sid(t,'transition')
document.getElementById('fh').textContent=sid(t,'hold_frames')
var re=document.getElementById('fres')
if(t.ok){{re.textContent='PASS';re.className='v ok'}}else{{re.textContent=sid(t,'event_type')||'NG';re.className='v ng'}}
sl.value=i;document.getElementById('flbl').textContent=i+'/'+T.length
document.querySelectorAll('.ks-item').forEach(function(e){{e.classList.remove('active')
if(e.dataset.sid===sid(t,'expected_state'))e.classList.add('active')}})
setImg('roi-img','roi-empty',sid(t,'roi_image'))
setImg('warp-img','warp-empty',sid(t,'warped_image'))
ut(i);ue(i);ud(t)}}
function ud(t){{var el=document.getElementById('detail-pre')
var lines=['=== 当前状态 ===','frame: '+sid(t,'frame_index'),'time: '+sid(t,'time_sec')+'s','','expected: '+sid(t,'expected_state'),'matched: '+sid(t,'matched_state'),'next: '+sid(t,'next_state'),'transition: '+sid(t,'transition'),'hold: '+sid(t,'hold_frames'),'ok: '+t.ok]
if(sid(t,'event_type'))lines.push('event: '+sid(t,'event_type'))
if(C.states){{
var s=C.states.find(function(s){{return s.id===sid(t,'matched_state')||s.id===sid(t,'expected_state')}})
if(s&&s.values){{lines.push('','=== 状态值 ===')
for(var k in s.values)lines.push(k+': '+s.values[k])}}}}
el.textContent=lines.join('\\n')}}
function ut(ai){{var svg=document.getElementById('tsvg'),w=Math.max(T.length*3,400)
svg.setAttribute('viewBox','0 0 '+w+' 35');svg.innerHTML=''
var bg=document.createElementNS('http://www.w3.org/2000/svg','rect');bg.setAttribute('width',w);bg.setAttribute('height','35');bg.setAttribute('fill','#0d1117');svg.appendChild(bg)
for(var i=0;i<T.length;i++){{var t=T[i],x=i*3,co=t.ok?'#2e7d32':'#c62828'
var r=document.createElementNS('http://www.w3.org/2000/svg','rect')
r.setAttribute('x',x);r.setAttribute('y','13');r.setAttribute('width','2');r.setAttribute('height','9')
r.setAttribute('fill',co);r.setAttribute('rx','1');svg.appendChild(r)}}
var m=document.createElementNS('http://www.w3.org/2000/svg','line')
m.setAttribute('x1',ai*3);m.setAttribute('y1','0');m.setAttribute('x2',ai*3);m.setAttribute('y2','35')
m.setAttribute('stroke','#fff');m.setAttribute('stroke-width','1.5');svg.appendChild(m)}}
function ue(i){{var t=T[i],el=document.getElementById('evlog')
if(sid(t,'event_type'))el.innerHTML='<span class=everr>'+sid(t,'event_type')+'</span>'
else if(E.find(function(e){{return e.frame_index===sid(t,'frame_index')}})){{
var ev=E.find(function(e){{return e.frame_index===sid(t,'frame_index')}})
el.innerHTML='<span class=everr>'+ev.type+': '+ev.message+'</span>'}}else el.innerHTML='<span class=everr>OK</span>'}}
var kl=document.getElementById('klist')
C.states.forEach(function(s,i){{var d=document.createElement('div');d.className='ks-item';d.dataset.sid=s.id
var id=document.createElement('div');id.className='ks-id';id.textContent=s.id
var vl=document.createElement('div');vl.className='ks-val'
var vals=[];if(s.values)for(var k in s.values){{if(k==='visual_bits')vals.push(k+':'+String(s.values[k]).length+' bits');else vals.push(k+':'+s.values[k])}}
vl.textContent=vals.join(' | ')
d.appendChild(id);d.appendChild(vl)
d.addEventListener('click',function(){{var fi=T.findIndex(function(t){{return sid(t,'matched_state')===s.id||sid(t,'expected_state')===s.id}})
if(fi>=0)uf(fi)}});kl.appendChild(d)}})
document.getElementById('bp').addEventListener('click',function(){{pl=pl?null:setInterval(function(){{if(ci>=T.length-1){{clearInterval(pl);pl=null;return}}uf(ci+1)}},200)
this.textContent=pl?'PAUSE':'PLAY'}})
document.getElementById('bpr').addEventListener('click',function(){{uf(Math.max(0,ci-1))}})
document.getElementById('bnx').addEventListener('click',function(){{uf(Math.min(T.length-1,ci+1))}})
document.getElementById('bne').addEventListener('click',function(){{for(var i=ci+1;i<T.length;i++)if(!T[i].ok){{uf(i);return}}}})
document.getElementById('sl').addEventListener('input',function(){{uf(parseInt(this.value))}})
sl.max=T.length-1;var nf=T.findIndex(function(t){{return!t.ok}});uf(nf>=0?nf:0)
window.addEventListener('resize',function(){{ut(ci)}})}})()
</script></body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
