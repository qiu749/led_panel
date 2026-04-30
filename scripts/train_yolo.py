import os
import argparse
os.environ['YOLO_VERBOSE'] = 'True'

from pathlib import Path
from matplotlib import font_manager

def _get_first_font():
    fonts = font_manager.fontManager.ttflist
    if fonts:
        return fonts[0].fname
    return None

first_font = _get_first_font()
if first_font:
    config_dir = Path(os.path.expanduser("~")) / ".config" / "Ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    target_font = config_dir / "Arial.ttf"
    if not target_font.exists():
        import shutil
        shutil.copy(first_font, target_font)

from ultralytics import YOLO
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_yolo(model_path: str = 'yolo11n.pt', 
                data_path: str = 'D:\\led-visual-inspection\\datasets\\data.yaml',
                epochs: int = 100,
                imgsz: int = 640,
                batch: int = 4,
                device: str = 'cpu'):
    """
    训练 YOLOv11 模型
    
    Args:
        model_path: 预训练模型路径
        data_path: 数据集配置文件路径
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch: 批次大小
        device: 训练设备 ('0' 为 GPU, 'cpu' 为 CPU)
    """
    try:
        logger.info(f"开始加载 YOLOv11 训练...")
        logger.info(f"模型: {model_path}")
        logger.info(f"数据集: {data_path}")
        logger.info(f"参数: epochs={epochs}, imgsz={imgsz}, batch={batch}, device={device}")

        model = YOLO(model_path)

        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project='models',
            name='led_panel',
            exist_ok=True,
            patience=50,
            save=True,
            plots=False
        )

        logger.info("训练完成！")
        logger.info(f"最佳模型保存位置: {results.save_dir}")
        
        return results

    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='训练 YOLOv11 模型')
    parser.add_argument('--model', type=str, default='D:\\led-visual-inspection\\yolo11n.pt', help='预训练模型')
    parser.add_argument('--data', type=str, default='D:\\led-visual-inspection\\datasets\\data.yaml', help='数据集配置')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=4, help='批次大小')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (0 或 cpu)')
    
    args = parser.parse_args()
    train_yolo(
        model_path=args.model,
        data_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )


if __name__ == '__main__':
    main()
