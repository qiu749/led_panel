"""
视频分析工具
用于提取帧、切分视频、分析视频信息
"""

import os
import cv2
import json
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoAnalyzer:
    """视频分析器"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        self.info = self._get_info()
    
    def _get_info(self) -> Dict:
        """获取视频信息"""
        return {
            'path': self.video_path,
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS)
        }
    
    def print_info(self):
        """打印视频信息"""
        info = self.info
        print("=" * 60)
        print("视频信息")
        print("=" * 60)
        print(f"路径: {info['path']}")
        print(f"分辨率: {info['width']}x{info['height']}")
        print(f"帧率: {info['fps']} fps")
        print(f"总帧数: {info['total_frames']}")
        print(f"时长: {info['duration']:.2f} 秒")
        print("=" * 60)
    
    def extract_frames(
        self,
        output_dir: str,
        interval: int = 30,
        max_frames: Optional[int] = None,
        prefix: str = 'frame'
    ) -> List[str]:
        """
        提取视频帧
        
        Args:
            output_dir: 输出目录
            interval: 提取间隔（每N帧提取一帧）
            max_frames: 最大提取帧数
            prefix: 文件名前缀
            
        Returns:
            提取的帧路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        extracted = []
        frame_count = 0
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                output_path = os.path.join(output_dir, f"{prefix}_{frame_count:06d}.jpg")
                cv2.imwrite(output_path, frame)
                extracted.append(output_path)
                
                if max_frames and len(extracted) >= max_frames:
                    break
            
            frame_count += 1
        
        logger.info(f"提取了 {len(extracted)} 帧到 {output_dir}")
        return extracted
    
    def split_video(
        self,
        output_dir: str,
        segment_duration: int = 10,
        prefix: str = 'segment'
    ) -> List[str]:
        """
        切分视频为多个片段
        
        Args:
            output_dir: 输出目录
            segment_duration: 每个片段时长（秒）
            prefix: 文件名前缀
            
        Returns:
            切分的视频路径列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fps = self.info['fps']
        width = self.info['width']
        height = self.info['height']
        total_frames = self.info['total_frames']
        
        frames_per_segment = int(fps * segment_duration)
        num_segments = (total_frames + frames_per_segment - 1) // frames_per_segment
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        segments = []
        
        for seg_idx in range(num_segments):
            output_path = os.path.join(output_dir, f"{prefix}_{seg_idx:03d}.mp4")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            start_frame = seg_idx * frames_per_segment
            end_frame = min((seg_idx + 1) * frames_per_segment, total_frames)
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(int(end_frame - start_frame)):
                ret, frame = self.cap.read()
                if not ret:
                    break
                out.write(frame)
            
            out.release()
            segments.append(output_path)
            logger.info(f"创建片段 {seg_idx + 1}/{num_segments}: {output_path}")
        
        return segments
    
    def close(self):
        """关闭视频文件"""
        if self.cap:
            self.cap.release()


def analyze_video(video_path: str, output_dir: str = 'analysis_output'):
    """
    分析视频并保存结果
    
    Args:
        video_path: 视频路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = VideoAnalyzer(video_path)
    analyzer.print_info()
    
    with open(os.path.join(output_dir, 'video_info.json'), 'w', encoding='utf-8') as f:
        json.dump(analyzer.info, f, indent=2, ensure_ascii=False)
    
    frames_dir = os.path.join(output_dir, 'frames')
    analyzer.extract_frames(frames_dir, interval=analyzer.info['fps'])
    
    analyzer.close()
    
    logger.info(f"分析完成! 结果保存在 {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='视频分析工具')
    parser.add_argument('--video', type=str, required=True, help='视频路径')
    parser.add_argument('--output', type=str, default='analysis_output', help='输出目录')
    parser.add_argument('--extract-frames', action='store_true', help='提取帧')
    parser.add_argument('--split', action='store_true', help='切分视频')
    parser.add_argument('--segment-duration', type=int, default=10, help='片段时长(秒)')
    parser.add_argument('--interval', type=int, default=30, help='帧提取间隔')
    
    args = parser.parse_args()
    
    analyzer = VideoAnalyzer(args.video)
    analyzer.print_info()
    
    if args.extract_frames:
        analyzer.extract_frames(os.path.join(args.output, 'frames'), interval=args.interval)
    
    if args.split:
        analyzer.split_video(args.output, segment_duration=args.segment_duration)
    
    if not args.extract_frames and not args.split:
        analyze_video(args.video, args.output)
    
    analyzer.close()
