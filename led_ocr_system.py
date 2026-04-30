"""
LED显示屏检测与OCR识别系统
结合YOLO目标检测、图像增强和RapidOCR文字识别
"""

import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """检测结果数据类"""
    bbox: List[int]
    confidence: float
    class_name: str
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    ocr_details: List[Dict] = None
    
    def __post_init__(self):
        if self.ocr_details is None:
            self.ocr_details = []


class ImageEnhancer:
    """图像增强器"""
    
    @staticmethod
    def enhance(image) -> Dict[str, np.ndarray]:
        """
        对图像进行多种增强处理
        
        Args:
            image: 输入图像
            
        Returns:
            dict: 各种增强方法处理后的图像
        """
        methods = {}
        
        methods['original'] = image.copy()
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        methods['clahe'] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods['binary'] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        methods['adaptive'] = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        
        h, w = image.shape[:2]
        scaled = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        methods['scaled_2x'] = scaled
        
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        methods['sharpened'] = sharpened
        
        return methods


class LEDInspectionSystem:
    """LED显示屏检测与OCR识别系统"""
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        conf_threshold: float = 0.25,
        ocr_padding: int = 80,
        ocr_conf_threshold: float = 0.3,
        use_enhancement: bool = True
    ):
        """
        初始化检测系统
        
        Args:
            model_path: YOLO模型路径
            device: 运行设备 ('cpu' 或 'cuda')
            conf_threshold: YOLO检测置信度阈值
            ocr_padding: OCR检测框扩展像素
            ocr_conf_threshold: OCR识别置信度阈值
            use_enhancement: 是否使用图像增强
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.ocr_padding = ocr_padding
        self.ocr_conf_threshold = ocr_conf_threshold
        self.use_enhancement = use_enhancement
        
        self.detector = None
        self.ocr_engine = None
        self.enhancer = ImageEnhancer()
        
        self._load_models()
    
    def _load_models(self):
        """加载模型"""
        logger.info("正在加载YOLO模型...")
        from ultralytics import YOLO
        self.detector = YOLO(self.model_path)
        logger.info("YOLO模型加载成功!")
        
        logger.info("正在加载RapidOCR模型...")
        from rapidocr import RapidOCR
        self.ocr_engine = RapidOCR()
        logger.info("RapidOCR模型加载成功!")
    
    def detect(self, frame) -> List[DetectionResult]:
        """
        检测LED显示屏
        
        Args:
            frame: 输入图像
            
        Returns:
            检测结果列表
        """
        results = self.detector.predict(
            frame, 
            conf=self.conf_threshold, 
            device=self.device, 
            verbose=False
        )
        
        detections = []
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                conf = float(box.conf.cpu().numpy()[0])
                cls_name = results[0].names[int(box.cls.cpu().numpy()[0])]
                
                detections.append(DetectionResult(
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    confidence=conf,
                    class_name=cls_name
                ))
        
        return detections
    
    def _ocr_with_enhancement(self, region) -> tuple:
        """
        使用图像增强进行OCR识别
        
        Args:
            region: 图像区域
            
        Returns:
            (texts, details) 识别结果
        """
        enhanced_images = self.enhancer.enhance(region)
        
        all_texts = []
        all_details = []
        seen_texts = set()
        
        for method_name, enhanced in enhanced_images.items():
            ocr_result = self.ocr_engine(enhanced)
            
            if ocr_result.boxes is not None and ocr_result.txts is not None:
                for box, txt, score in zip(ocr_result.boxes, ocr_result.txts, ocr_result.scores):
                    if score > self.ocr_conf_threshold and txt not in seen_texts:
                        seen_texts.add(txt)
                        all_texts.append(txt)
                        all_details.append({
                            'text': txt,
                            'confidence': float(score),
                            'method': method_name
                        })
        
        return all_texts, all_details
    
    def recognize_ocr(self, frame, detection: DetectionResult) -> DetectionResult:
        """
        对检测区域进行OCR识别
        
        Args:
            frame: 原始图像
            detection: 检测结果
            
        Returns:
            更新后的检测结果
        """
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        
        x1_p = max(0, x1 - self.ocr_padding)
        y1_p = max(0, y1 - self.ocr_padding)
        x2_p = min(width, x2 + self.ocr_padding)
        y2_p = min(height, y2 + self.ocr_padding)
        
        region = frame[y1_p:y2_p, x1_p:x2_p]
        
        if region.shape[0] < 20 or region.shape[1] < 20:
            return detection
        
        if self.use_enhancement:
            texts, details = self._ocr_with_enhancement(region)
        else:
            ocr_result = self.ocr_engine(region)
            texts = []
            details = []
            
            if ocr_result.boxes is not None and ocr_result.txts is not None:
                for box, txt, score in zip(ocr_result.boxes, ocr_result.txts, ocr_result.scores):
                    if score > self.ocr_conf_threshold:
                        texts.append(txt)
                        details.append({
                            'text': txt,
                            'confidence': float(score)
                        })
        
        detection.ocr_text = ' '.join(texts)
        detection.ocr_confidence = sum(d['confidence'] for d in details) / len(details) if details else 0.0
        detection.ocr_details = details
        
        return detection
    
    def process_frame(self, frame) -> List[DetectionResult]:
        """
        处理单帧图像
        
        Args:
            frame: 输入图像
            
        Returns:
            检测结果列表（包含OCR结果）
        """
        detections = self.detect(frame)
        
        for detection in detections:
            self.recognize_ocr(frame, detection)
        
        return detections
    
    def draw_results(self, frame, detections: List[DetectionResult]) -> np.ndarray:
        """
        绘制检测结果
        
        Args:
            frame: 原始图像
            detections: 检测结果列表
            
        Returns:
            绘制后的图像
        """
        viz_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(viz_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if det.ocr_text:
                cv2.putText(viz_frame, f"OCR: {det.ocr_text}", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return viz_frame


def process_video(
    video_path: str,
    model_path: str,
    output_path: str = 'output.mp4',
    process_every_n_frames: int = 3,
    device: str = 'cpu',
    use_enhancement: bool = True,
    verbose: bool = True
):
    """
    处理视频文件
    
    Args:
        video_path: 输入视频路径
        model_path: YOLO模型路径
        output_path: 输出视频路径
        process_every_n_frames: 每N帧处理一次
        device: 运行设备
        use_enhancement: 是否使用图像增强
        verbose: 是否打印详细信息
    """
    system = LEDInspectionSystem(
        model_path=model_path, 
        device=device,
        use_enhancement=use_enhancement
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
    logger.info(f"图像增强: {'开启' if use_enhancement else '关闭'}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    all_results = []
    
    logger.info("开始处理视频...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % process_every_n_frames == 0:
            detections = system.process_frame(frame)
            viz_frame = system.draw_results(frame, detections)
            
            for det in detections:
                if det.ocr_text:
                    all_results.append({
                        'frame': frame_count,
                        'class': det.class_name,
                        'ocr_text': det.ocr_text,
                        'ocr_confidence': det.ocr_confidence
                    })
                    if verbose:
                        logger.info(f"帧{frame_count}: {det.class_name} -> {det.ocr_text}")
            
            cv2.putText(viz_frame, f"Frame: {frame_count}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            viz_frame = frame.copy()
        
        out.write(viz_frame)
        
        if frame_count % 50 == 0:
            logger.info(f"进度: {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    logger.info(f"处理完成! 输出视频: {output_path}")
    logger.info(f"共识别 {len(all_results)} 个OCR结果")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LED显示屏检测与OCR识别')
    parser.add_argument('--video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--model', type=str, required=True, help='YOLO模型路径')
    parser.add_argument('--output', type=str, default='output.mp4', help='输出视频路径')
    parser.add_argument('--skip-frames', type=int, default=3, help='每N帧处理一次')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备')
    parser.add_argument('--no-enhance', action='store_true', help='关闭图像增强')
    
    args = parser.parse_args()
    
    process_video(
        video_path=args.video,
        model_path=args.model,
        output_path=args.output,
        process_every_n_frames=args.skip_frames,
        device=args.device,
        use_enhancement=not args.no_enhance
    )
