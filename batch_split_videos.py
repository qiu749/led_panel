"""
批量处理视频
切分所有视频文件
"""

import os
import cv2
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_video(video_path: str, output_dir: str, segment_duration: int = 10):
    """
    切分视频为多个片段
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        segment_duration: 每个片段时长（秒）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = Path(video_path).stem
    
    logger.info(f"处理视频: {video_name}")
    logger.info(f"  分辨率: {width}x{height}, {fps}fps, {total_frames}帧")
    
    frames_per_segment = int(fps * segment_duration)
    num_segments = (total_frames + frames_per_segment - 1) // frames_per_segment
    
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    segments = []
    
    for seg_idx in range(num_segments):
        output_path = os.path.join(video_output_dir, f"segment_{seg_idx:03d}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_frame = seg_idx * frames_per_segment
        end_frame = min((seg_idx + 1) * frames_per_segment, total_frames)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(int(end_frame - start_frame)):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        segments.append(output_path)
    
    cap.release()
    
    logger.info(f"  切分完成: {num_segments} 个片段 -> {video_output_dir}")
    
    return segments


def process_all_videos(input_dir: str, output_dir: str, segment_duration: int = 10):
    """
    处理目录下所有视频
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        segment_duration: 每个片段时长（秒）
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
    
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f'*{ext}'))
    
    if not video_files:
        logger.warning(f"在 {input_dir} 中未找到视频文件")
        return
    
    logger.info(f"找到 {len(video_files)} 个视频文件")
    logger.info(f"每个片段时长: {segment_duration} 秒")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, video_path in enumerate(video_files, 1):
        logger.info(f"\n[{i}/{len(video_files)}]")
        try:
            split_video(str(video_path), output_dir, segment_duration)
        except Exception as e:
            logger.error(f"处理失败: {e}")
    
    logger.info(f"\n全部处理完成! 结果保存在: {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='批量切分视频')
    parser.add_argument('--input', type=str, required=True, help='输入目录')
    parser.add_argument('--output', type=str, default='split_videos', help='输出目录')
    parser.add_argument('--duration', type=int, default=10, help='每个片段时长(秒)')
    
    args = parser.parse_args()
    
    process_all_videos(
        input_dir=args.input,
        output_dir=args.output,
        segment_duration=args.duration
    )
