"""
merge_multi_video_templates

从多个视频分别提取模板，然后合并去重生成统一模型。
解决单视频模型因拍摄timing差异导致的模板缺失问题。

流程:
  1. 对每个视频独立运行 extract_scan_frame_templates 提取模板
  2. 以第一个视频的模板为基础（主模板集）
  3. 对后续视频的模板，用 Jaccard + color_similarity 去重合并
  4. 合并后的序列取所有视频序列的最长公共子序列(LCS)作为标准序列
  5. 输出统一的 templates.json 和 summary.json

Example:
  python scripts/merge_multi_video_templates.py ^
    --videos D:/jiemi/七段管/Video_20260520165824830.avi ^
             D:/jiemi/七段管/Video_20260520165703980.avi ^
             D:/jiemi/七段管/Video_20260515112415516.avi ^
    --out models/seven_segment/merged_multi_video ^
    --primary 0
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.extract_scan_frame_templates import (
    FrameTemplate,
    SequenceSegment,
    bit_jaccard,
    color_similarity,
    make_mask,
    mask_to_bits,
    mask_to_color_bits,
)
from scripts.scan_refresh_seven_segment_probe import (
    Roi,
    crop_roi,
    detect_display_roi,
    first_pass_projection,
    make_contact_sheet,
    write_csv,
    write_json,
)


def load_templates_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_single_video(
    video: Path,
    out_dir: Path,
    args: argparse.Namespace,
    force_roi: Roi | None = None,
) -> Path:
    """对单个视频提取模板。如果指定了force_roi，强制使用该ROI（确保跨视频一致性）。"""
    import subprocess
    script = Path(__file__).resolve().parent / "extract_scan_frame_templates.py"
    cmd = [
        sys.executable, str(script),
        "--video", str(video),
        "--out", str(out_dir),
        "--stride-frames", str(args.stride_frames),
        "--template-threshold", str(args.template_threshold),
        "--color-threshold", str(args.color_threshold),
        "--active-threshold", str(args.active_threshold),
        "--grid-cols", str(args.grid_cols),
        "--grid-rows", str(args.grid_rows),
        "--cell-threshold", str(args.cell_threshold),
        "--max-templates", str(args.max_templates),
    ]
    print(f"  提取模板: {video.name} -> {out_dir.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[:500]}")
        raise RuntimeError(f"模板提取失败: {video.name}")
    
    # 如果有force_roi，重新提取（用统一ROI尺寸）
    if force_roi is not None:
        reextract_with_unified_roi(video, out_dir, args, force_roi)
    
    return out_dir / "templates.json"


def reextract_with_unified_roi(
    video: Path,
    out_dir: Path,
    args: argparse.Namespace,
    target_roi: Roi,
) -> None:
    """用统一ROI重新提取模板，确保跨视频的bit pattern一致性。
    
    策略：先自动检测该视频的实际ROI，裁切后resize到target_roi的尺寸(w,h)，
    再做网格化。这样不同视频的相同面板内容会映射到相同的bit pattern。
    """
    # 检测该视频的实际ROI
    cap = cv2.VideoCapture(str(video))
    max_proj = None
    for i in range(360):
        ok, f = cap.read()
        if not ok:
            break
        if max_proj is None:
            max_proj = f.copy()
        else:
            np.maximum(max_proj, f, out=max_proj)
    cap.release()
    
    if max_proj is None:
        return
    
    try:
        actual_roi = detect_display_roi(max_proj, args.active_threshold, 0.06)
    except Exception:
        return
    
    # 如果实际ROI跟目标ROI尺寸差不多（<5%），不需要重新提取
    w_diff = abs(actual_roi.w - target_roi.w) / max(target_roi.w, 1)
    h_diff = abs(actual_roi.h - target_roi.h) / max(target_roi.h, 1)
    if w_diff < 0.03 and h_diff < 0.03:
        return  # 尺寸接近，用原提取结果即可
    
    print(f"    ROI尺寸差异: {actual_roi.w}x{actual_roi.h} vs 目标{target_roi.w}x{target_roi.h}, 重新提取...")
    
    # 重新逐帧提取，crop后resize到目标尺寸
    templates_dir = out_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    
    from scripts.extract_scan_frame_templates import (
        FrameTemplate, SequenceSegment, find_best_template as find_best_tmpl_str,
        mask_to_bits, save_template_assets, append_sequence_segment, jsonable,
    )
    
    templates: list[FrameTemplate] = []
    active_pixel_samples: list[list[int]] = []
    segments: list[SequenceSegment] = []
    frame_index = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % max(args.stride_frames, 1) != 0:
            frame_index += 1
            continue
        
        crop = crop_roi(frame, actual_roi)
        # 关键：resize到统一尺寸
        crop = cv2.resize(crop, (target_roi.w, target_roi.h), interpolation=cv2.INTER_AREA)
        
        mask = make_mask(crop, args.active_threshold)
        active_pixels = int(np.count_nonzero(mask))
        if active_pixels < 120:
            frame_index += 1
            continue
        
        bits = mask_to_bits(mask, args.grid_cols, args.grid_rows, args.cell_threshold)
        color_bits_str, color_counts = mask_to_color_bits(crop, mask, args.grid_cols, args.grid_rows, args.cell_threshold)
        
        best_i, best_sim, best_color = find_best_tmpl_str(bits, color_bits_str, templates, args.color_threshold)
        
        if best_i >= 0 and best_sim >= args.template_threshold and best_color >= args.color_threshold:
            tmpl = templates[best_i]
            samples = active_pixel_samples[best_i]
            samples.append(active_pixels)
            tmpl.count += 1
            tmpl.last_frame = frame_index
            tmpl.last_time = frame_index / fps
            tmpl.active_pixels_median = float(np.median(samples))
            tmpl.active_pixels_max = max(tmpl.active_pixels_max, active_pixels)
            template_id = tmpl.id
            similarity = best_sim
        elif len(templates) < args.max_templates:
            template_id = f"T{len(templates) + 1:03d}"
            image, mask_image = save_template_assets(template_id, frame_index, crop, mask, templates_dir)
            templates.append(FrameTemplate(
                id=template_id,
                first_frame=frame_index,
                first_time=frame_index / fps,
                last_frame=frame_index,
                last_time=frame_index / fps,
                count=1,
                active_pixels_median=float(active_pixels),
                active_pixels_max=active_pixels,
                bits=bits,
                color_bits=color_bits_str,
                color_counts=color_counts,
                image=image,
                mask_image=mask_image,
            ))
            active_pixel_samples.append([active_pixels])
            similarity = 1.0
        else:
            template_id = "UNMATCHED"
            similarity = best_sim
        
        append_sequence_segment(segments, template_id, frame_index, fps, similarity, 6)
        frame_index += 1
    
    cap.release()
    
    # 写入重新提取的结果
    from dataclasses import asdict
    output_payload = {
        "templates": [jsonable(asdict(t)) for t in sorted(templates, key=lambda t: t.first_frame)],
        "sequence": [jsonable(asdict(s)) for s in segments],
    }
    (out_dir / "templates.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"    重新提取完成: {len(templates)} 模板, {len(segments)} 段")


def merge_templates(
    primary_templates: list[dict[str, Any]],
    secondary_templates: list[dict[str, Any]],
    jaccard_threshold: float,
    color_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """将 secondary 中的新模板合并到 primary 中。
    
    Returns:
        merged: 合并后的完整模板列表
        mapping: secondary_id -> primary_id 的映射（用于序列对齐）
    """
    mapping: dict[str, str] = {}
    new_templates: list[dict[str, Any]] = []
    
    for sec_t in secondary_templates:
        sec_bits = sec_t["bits"]
        sec_color = sec_t.get("color_bits", "")
        best_match_id = ""
        best_sim = 0.0
        
        for pri_t in primary_templates:
            pri_bits = pri_t["bits"]
            pri_color = pri_t.get("color_bits", "")
            
            j_sim = bit_jaccard(sec_bits, pri_bits)
            if j_sim < jaccard_threshold:
                continue
            
            c_sim = color_similarity(sec_color, pri_color) if sec_color and pri_color else 1.0
            combined = min(j_sim, c_sim)
            
            if combined > best_sim:
                best_sim = combined
                best_match_id = pri_t["id"]
        
        if best_match_id and best_sim >= min(jaccard_threshold, color_threshold):
            mapping[sec_t["id"]] = best_match_id
        else:
            # 新模板，追加到primary
            new_id = f"T{len(primary_templates) + len(new_templates) + 1:03d}"
            new_t = dict(sec_t)
            new_t["id"] = new_id
            mapping[sec_t["id"]] = new_id
            new_templates.append(new_t)
    
    merged = list(primary_templates) + new_templates
    return merged, mapping


def align_sequences(
    primary_seq: list[str],
    secondary_seq: list[str],
    mapping: dict[str, str],
) -> list[str]:
    """将 secondary 序列映射到 primary ID 空间，然后取 LCS 作为统一序列。"""
    # 映射 secondary 序列到 primary ID
    mapped_sec = []
    prev = ""
    for sid in secondary_seq:
        pid = mapping.get(sid, sid)
        if pid != prev:
            mapped_sec.append(pid)
            prev = pid
    
    # LCS
    n, m = len(primary_seq), len(mapped_sec)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if primary_seq[i] == mapped_sec[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    
    # 回溯 LCS
    lcs: list[str] = []
    i, j = 0, 0
    while i < n and j < m:
        if primary_seq[i] == mapped_sec[j]:
            lcs.append(primary_seq[i])
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    
    return lcs


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge templates from multiple videos into a unified model")
    parser.add_argument("--videos", type=Path, nargs="+", required=True, help="视频文件列表")
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    parser.add_argument("--primary", type=int, default=0, help="主视频索引（作为基础模板集）")
    parser.add_argument("--stride-frames", type=int, default=4)
    parser.add_argument("--template-threshold", type=float, default=0.72)
    parser.add_argument("--color-threshold", type=float, default=0.70)
    parser.add_argument("--active-threshold", type=int, default=55)
    parser.add_argument("--grid-cols", type=int, default=96)
    parser.add_argument("--grid-rows", type=int, default=64)
    parser.add_argument("--cell-threshold", type=int, default=28)
    parser.add_argument("--max-templates", type=int, default=160)
    parser.add_argument("--merge-jaccard", type=float, default=0.75, help="合并阈值: Jaccard >= 此值视为同一模板")
    parser.add_argument("--merge-color", type=float, default=0.65, help="合并阈值: color_similarity >= 此值")
    args = parser.parse_args()
    
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    videos = args.videos
    primary_idx = args.primary
    if not all(v.exists() for v in videos):
        missing = [str(v) for v in videos if not v.exists()]
        raise FileNotFoundError(f"视频不存在: {missing}")
    
    print(f"=== 多视频模板合并 ===")
    print(f"视频数: {len(videos)}")
    print(f"主视频: [{args.primary}] {videos[args.primary].name}")
    print(f"输出: {out_dir}")
    print()
    
    # Step 1: 对每个视频提取模板
    # 先提取主视频获取其ROI作为统一标准
    primary_video = videos[primary_idx]
    primary_out = out_dir / "per_video" / primary_video.stem
    extract_single_video(primary_video, primary_out, args, force_roi=None)
    primary_model = load_templates_json(primary_out / "templates.json")
    
    # 读取主视频的ROI
    primary_summary_path = primary_out / "summary.json"
    if primary_summary_path.exists():
        ps = json.loads(primary_summary_path.read_text(encoding="utf-8"))
        primary_roi_data = ps.get("roi", {})
        primary_roi = Roi(
            int(primary_roi_data.get("x", 780)),
            int(primary_roi_data.get("y", 400)),
            int(primary_roi_data.get("w", 1122)),
            int(primary_roi_data.get("h", 820)),
        )
    else:
        primary_roi = Roi(780, 400, 1122, 820)
    
    print(f"    主视频ROI: {primary_roi.w}x{primary_roi.h}")
    
    video_models: list[dict[str, Any]] = [primary_model]
    templates_p = primary_model.get("templates", [])
    sequence_p = primary_model.get("sequence", [])
    unique_seq_p = []
    prev = ""
    for s in sequence_p:
        tid = s.get("template_id", "") if isinstance(s, dict) else str(s)
        if tid != prev:
            unique_seq_p.append(tid)
            prev = tid
    print(f"    [0] {primary_video.name}: {len(templates_p)} 模板, {len(unique_seq_p)} 序列步")
    
    # 对其他视频提取模板（强制统一ROI尺寸）
    for i, video in enumerate(videos):
        if i == primary_idx:
            continue
        video_out = out_dir / "per_video" / video.stem
        extract_single_video(video, video_out, args, force_roi=primary_roi)
        model_data = load_templates_json(video_out / "templates.json")
        video_models.append(model_data)
        templates = model_data.get("templates", [])
        sequence = model_data.get("sequence", [])
        unique_seq = []
        prev = ""
        for s in sequence:
            tid = s.get("template_id", "") if isinstance(s, dict) else str(s)
            if tid != prev:
                unique_seq.append(tid)
                prev = tid
        print(f"    [{i}] {video.name}: {len(templates)} 模板, {len(unique_seq)} 序列步")
    
    print()
    
    # Step 2: 以主视频为基础合并
    primary_data = video_models[0]  # 主视频始终在索引0
    merged_templates = list(primary_data.get("templates", []))
    
    # 提取主视频序列
    primary_seq_raw = primary_data.get("sequence", [])
    primary_seq: list[str] = []
    prev = ""
    for s in primary_seq_raw:
        tid = s.get("template_id", "") if isinstance(s, dict) else str(s)
        if tid != prev:
            primary_seq.append(tid)
            prev = tid
    
    unified_seq = list(primary_seq)
    
    for i, model_data in enumerate(video_models[1:], start=1):
        sec_templates = model_data.get("templates", [])
        sec_seq_raw = model_data.get("sequence", [])
        sec_seq: list[str] = []
        prev = ""
        for s in sec_seq_raw:
            tid = s.get("template_id", "") if isinstance(s, dict) else str(s)
            if tid != prev:
                sec_seq.append(tid)
                prev = tid
        
        # 合并模板
        merged_templates, mapping = merge_templates(
            merged_templates, sec_templates,
            args.merge_jaccard, args.merge_color
        )
        
        # 对齐序列
        unified_seq = align_sequences(unified_seq, sec_seq, mapping)
        
        new_count = sum(1 for v in mapping.values() if v.startswith("T") and int(v[1:]) > len(primary_data.get("templates", [])))
        actual_video_idx = [j for j in range(len(videos)) if j != primary_idx][i - 1]
        print(f"  合并 [{actual_video_idx}] {videos[actual_video_idx].name}: 新增 {new_count} 模板, 映射 {len(mapping)} 项")
    
    # Step 3: 过滤序列中不存在的模板ID
    merged_ids = {t["id"] for t in merged_templates}
    unified_seq = [tid for tid in unified_seq if tid in merged_ids]
    
    print(f"\n=== 合并结果 ===")
    print(f"  总模板数: {len(merged_templates)}")
    print(f"  统一序列长度: {len(unified_seq)}")
    
    # Step 4: 复制主视频的模板图片
    primary_video_dir = out_dir / "per_video" / videos[primary_idx].stem
    templates_dir = out_dir / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    for t in merged_templates:
        image_path = t.get("image", "")
        if image_path:
            src = primary_video_dir / image_path
            if src.exists():
                dst = out_dir / image_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        mask_path = t.get("mask_image", "")
        if mask_path:
            src = primary_video_dir / mask_path
            if src.exists():
                dst = out_dir / mask_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
    
    # Step 5: 写入统一模型
    unified_sequence = [{"template_id": tid} for tid in unified_seq]
    
    output_payload = {
        "templates": merged_templates,
        "sequence": unified_sequence,
    }
    
    templates_json_path = out_dir / "templates.json"
    templates_json_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    # 写 summary.json (基于主视频的summary)
    primary_summary_path = primary_video_dir / "summary.json"
    if primary_summary_path.exists():
        summary = json.loads(primary_summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    
    summary["merge_info"] = {
        "source_videos": [str(v) for v in videos],
        "primary_index": primary_idx,
        "per_video_template_counts": [len(m.get("templates", [])) for m in video_models],
        "merged_template_count": len(merged_templates),
        "unified_sequence_length": len(unified_seq),
        "merge_jaccard_threshold": args.merge_jaccard,
        "merge_color_threshold": args.merge_color,
    }
    summary["template_count"] = len(merged_templates)
    summary["sequence_segment_count"] = len(unified_seq)
    
    write_json(out_dir / "summary.json", summary)
    
    # 复制 full_reconstruction.jpg
    recon_src = primary_video_dir / "full_reconstruction.jpg"
    if recon_src.exists():
        shutil.copy2(recon_src, out_dir / "full_reconstruction.jpg")
    
    print(f"\n  输出: {templates_json_path}")
    print(f"  summary: {out_dir / 'summary.json'}")
    print(f"\n使用方式:")
    print(f"  python seven_segment_live_camera_server.py --model {templates_json_path}")


if __name__ == "__main__":
    main()
