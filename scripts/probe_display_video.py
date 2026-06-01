"""
probe_display_video — 探查显示条视频：亮屏帧检测、时间段、contact sheet

用法：
  python scripts/probe_display_video.py --video D:/jiemi/Video_20260430154222007.avi --out output/display_probe

输出：
  bright_frames.json      — 亮屏帧索引和时间
  timeline.csv            — 逐帧亮/暗标记
  contact_sheet.jpg       — 亮屏帧联系表
  dark_sample.jpg         — 暗帧样例（用于对比）
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="探查显示条视频")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("output") / "display_probe")
    parser.add_argument("--bright-threshold", type=float, default=0.00005,
                        help="亮屏阈值：高亮像素(>200)占总像素比例大于此值判为亮屏")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数(0=全部)")
    parser.add_argument("--sheet-cols", type=int, default=10, help="contact sheet列数")
    args = parser.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 打开视频
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"错误: 无法打开 {args.video}"); sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 7.5
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频: {total} 帧, {fps:.2f} FPS, {w}x{h}")

    # 逐帧分析
    bright_frames = []   # (idx, time_sec, bright_ratio)
    timeline = []
    frame_bgr_list = []
    dark_sample = None

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高亮像素比例：pixel > 200
        bright_ratio = float(np.mean(gray > 200))
        max_val = int(gray.max())
        is_bright = bright_ratio > args.bright_threshold or max_val > 150

        timeline.append({
            "frame": idx, "time_sec": f"{idx/fps:.3f}",
            "bright_ratio": f"{bright_ratio:.4f}", "max_val": max_val,
            "is_bright": int(is_bright),
        })

        if is_bright:
            bright_frames.append((idx, idx/fps, bright_ratio))
            if len(frame_bgr_list) < 200:  # 最多存200帧用于contact sheet
                frame_bgr_list.append(frame)
        elif dark_sample is None:
            dark_sample = frame.copy()

        idx += 1
        if args.max_frames > 0 and idx >= args.max_frames:
            break

    cap.release()
    n_bright = len(bright_frames)
    print(f"亮屏帧: {n_bright}/{idx} ({n_bright/max(idx,1)*100:.1f}%)")

    # 输出亮屏时间段
    segments = []
    if bright_frames:
        seg_start = bright_frames[0][0]
        seg_end = bright_frames[0][0]
        for fi, ts, br in bright_frames[1:]:
            if fi - seg_end <= 2:  # 间隔≤2帧视为同一段
                seg_end = fi
            else:
                segments.append({
                    "start_frame": seg_start, "end_frame": seg_end,
                    "start_time": f"{seg_start/fps:.3f}", "end_time": f"{seg_end/fps:.3f}",
                    "frames": seg_end - seg_start + 1, "duration_sec": round((seg_end-seg_start+1)/fps, 3),
                })
                seg_start = fi; seg_end = fi
        segments.append({
            "start_frame": seg_start, "end_frame": seg_end,
            "start_time": f"{seg_start/fps:.3f}", "end_time": f"{seg_end/fps:.3f}",
            "frames": seg_end - seg_start + 1, "duration_sec": round((seg_end-seg_start+1)/fps, 3),
        })

    print(f"亮屏时间段: {len(segments)}")
    for s in segments:
        print(f"  {s['start_time']}s-{s['end_time']}s ({s['frames']}帧)")

    # 保存 bright_frames.json
    with open(out / "bright_frames.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_frames": idx, "fps": fps,
            "bright_frame_count": n_bright, "bright_ratio_total": f"{n_bright/max(idx,1)*100:.1f}%",
            "segments": segments,
            "frames": [{"frame": fi, "time_sec": round(ts, 3), "bright_ratio": round(br, 4)}
                       for fi, ts, br in bright_frames[:500]],  # 最多存500条
        }, f, ensure_ascii=False, indent=2)

    # 保存 timeline.csv
    with open(out / "timeline.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "time_sec", "bright_ratio", "max_val", "is_bright"])
        w.writeheader(); w.writerows(timeline)

    # 生成 contact sheet
    if frame_bgr_list:
        _make_contact_sheet(frame_bgr_list, out / "contact_sheet.jpg", args.sheet_cols)

    # 保存暗帧样例
    if dark_sample is not None:
        cv2.imwrite(str(out / "dark_sample.jpg"), dark_sample)

    # 保存首张亮帧（用于后续手动标注）
    if frame_bgr_list:
        cv2.imwrite(str(out / "first_bright_frame.jpg"), frame_bgr_list[0])

    print(f"\n输出目录: {out}")
    print(f"  bright_frames.json, timeline.csv, contact_sheet.jpg")


def _make_contact_sheet(frames: list[np.ndarray], out_path: Path, cols: int):
    """生成亮帧联系表"""
    n = len(frames)
    rows = (n + cols - 1) // cols
    thumb_w, thumb_h = 160, 120
    sheet = np.ones((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8) * 40

    for i, f in enumerate(frames):
        r, c = i // cols, i % cols
        thumb = cv2.resize(f, (thumb_w, thumb_h))
        y0, x0 = r * thumb_h, c * thumb_w
        sheet[y0:y0+thumb_h, x0:x0+thumb_w] = thumb
        # 帧号
        cv2.putText(sheet, str(i), (x0+2, y0+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    cv2.imwrite(str(out_path), sheet)


if __name__ == "__main__":
    main()
