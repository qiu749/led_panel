import os
import argparse
os.environ['YOLO_VERBOSE'] = 'False'

from pathlib import Path
from matplotlib import font_manager

first_font = font_manager.fontManager.ttflist[0].fname if font_manager.fontManager.ttflist else None
if first_font:
    config_dir = Path(os.path.expanduser("~")) / ".config" / "Ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    target_font = config_dir / "Arial.ttf"
    if not target_font.exists():
        import shutil
        shutil.copy(first_font, target_font)

import cv2
import numpy as np
from ultralytics import YOLO
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_yolo(model_path: str, 
                 source: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = '0',
                 show: bool = True,
                 output_path: str = None):
    """
    使用 YOLOv11 进行推理
    
    Args:
        model_path: 训练好的模型路径
        source: 输入源 (图片路径/视频路径/0 表示摄像头)
        conf_threshold: 置信度阈值
        iou_threshold: IOU 阈值
        device: 推理设备 ('0' 或 'cpu')
        show: 是否显示结果
        output_path: 输出视频路径
    """
    try:
        logger.info(f"加载模型: {model_path}")
        model = YOLO(model_path)

        logger.info(f"开始推理: {source}")
        
        if output_path:
            save_dir = str(Path(output_path).parent)
            save_name = Path(output_path).stem
        else:
            save_dir = 'runs/predict'
            save_name = 'results'
        
        results = model.predict(
            source=source,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            show=show,
            save=True,
            project=save_dir,
            name=save_name,
            exist_ok=True
        )
        
        total_detections = 0
        for r in results:
            if r.boxes is not None:
                total_detections += len(r.boxes)
                logger.info(f"帧检测数: {len(r.boxes)}")
        
        logger.info(f"总检测数: {total_detections}")
        logger.info("推理完成！")
        
        if output_path and results:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                logger.info(f"检测到的类别: {result.boxes.cls.cpu().numpy()}")
                logger.info(f"检测置信度: {result.boxes.conf.cpu().numpy()}")
        
        return results

    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 推理')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--source', type=str, default='0', help='输入源')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU 阈值')
    parser.add_argument('--device', type=str, default='cpu', help='设备')
    parser.add_argument('--show', action='store_true', default=True, help='显示结果')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频路径')
    
    args = parser.parse_args()
    predict_yolo(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        show=args.show,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
