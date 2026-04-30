import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
import logging

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


class EasyOCREngine:
    """
    EasyOCR 引擎类：用于文字识别
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, lang: str = 'ch', use_gpu: bool = False):
        if EasyOCREngine._initialized:
            return
        EasyOCREngine._initialized = True
        self.lang = lang
        self.use_gpu = use_gpu
        self.reader = None
        self._load_model()
        
    def _load_model(self):
        try:
            import easyocr
            logger.info(f"Loading EasyOCR model (lang={self.lang})")
            self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=self.use_gpu)
            logger.info("EasyOCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR model: {e}")
            raise
    
    def recognize(self, image: np.ndarray) -> List[Dict]:
        """
        识别图像中的文字
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            识别结果列表，每个结果包含 {'text': str, 'confidence': float, 'bbox': List[int]}
        """
        try:
            if self.reader is None:
                logger.error("OCR model not loaded")
                return []
            
            results = self.reader.readtext(image)
            
            recognized = []
            for (bbox, text, prob) in results:
                recognized.append({
                    'text': text,
                    'confidence': float(prob),
                    'bbox': [int(p) for p in bbox[0]] + [int(p) for p in bbox[2]]
                })
            
            return recognized
            
        except Exception as e:
            logger.error(f"EasyOCREngine recognize error: {e}")
            return []
    
    def recognize_region(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        识别指定区域的文字
        
        Args:
            image: 原始图像
            bbox: 区域边界框 [x1, y1, x2, y2]
            
        Returns:
            识别结果 {'text': str, 'confidence': float, 'all_texts': List[Dict]}
        """
        try:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return {'text': '', 'confidence': 0.0, 'all_texts': []}
            
            region = image[y1:y2, x1:x2]
            results = self.recognize(region)
            
            if not results:
                return {'text': '', 'confidence': 0.0, 'all_texts': []}
            
            combined_text = ' '.join([r['text'] for r in results])
            avg_confidence = sum([r['confidence'] for r in results]) / len(results)
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'all_texts': results
            }
            
        except Exception as e:
            logger.error(f"EasyOCREngine recognize_region error: {e}")
            return {'text': '', 'confidence': 0.0, 'all_texts': []}


class PaddleOCREngine:
    """
    PaddleOCR 引擎类：用于文字识别
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, lang: str = 'ch', device: str = 'cpu', use_angle_cls: bool = True):
        if PaddleOCREngine._initialized:
            return
        PaddleOCREngine._initialized = True
        self.lang = lang
        self.device = device
        self.use_angle_cls = use_angle_cls
        self.ocr = None
        if not PADDLEOCR_AVAILABLE:
            logger.warning("PaddleOCR not available, skipping initialization")
            return
        self._load_model()
        
    def _load_model(self):
        try:
            logger.info(f"Loading PaddleOCR model (lang={self.lang}, device={self.device})")
            self.ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=(self.device != 'cpu'),
                show_log=False
            )
            logger.info("PaddleOCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR model: {e}")
            raise
    
    def recognize(self, image: np.ndarray) -> List[Dict]:
        """
        识别图像中的文字
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            识别结果列表，每个结果包含 {'text': str, 'confidence': float, 'bbox': List[int]}
        """
        try:
            if self.ocr is None:
                logger.error("OCR model not loaded")
                return []
            
            results = self.ocr.ocr(image, cls=self.use_angle_cls)
            
            if not results or len(results) == 0 or results[0] is None:
                return []
            
            recognized = []
            for line in results[0]:
                if line is None:
                    continue
                
                bbox_points = line[0]
                text_info = line[1]
                text = text_info[0]
                confidence = float(text_info[1])
                
                bbox = []
                if bbox_points:
                    bbox = [int(p) for point in bbox_points for p in point]
                
                recognized.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
            
            return recognized
            
        except Exception as e:
            logger.error(f"PaddleOCREngine recognize error: {e}")
            return []
    
    def recognize_region(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        识别指定区域的文字
        
        Args:
            image: 原始图像
            bbox: 区域边界框 [x1, y1, x2, y2]
            
        Returns:
            识别结果 {'text': str, 'confidence': float, 'all_texts': List[Dict]}
        """
        try:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return {'text': '', 'confidence': 0.0, 'all_texts': []}
            
            region = image[y1:y2, x1:x2]
            results = self.recognize(region)
            
            if not results:
                return {'text': '', 'confidence': 0.0, 'all_texts': []}
            
            combined_text = ' '.join([r['text'] for r in results])
            avg_confidence = sum([r['confidence'] for r in results]) / len(results)
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'all_texts': results
            }
            
        except Exception as e:
            logger.error(f"PaddleOCREngine recognize_region error: {e}")
            return {'text': '', 'confidence': 0.0, 'all_texts': []}


class RapidOCREngine:
    """
    RapidOCR 引擎类：用于文字识别 (基于ONNX，效果更好)
    """
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if RapidOCREngine._initialized:
            return
        RapidOCREngine._initialized = True
        self.ocr = None
        if not RAPIDOCR_AVAILABLE:
            logger.warning("RapidOCR not available, skipping initialization")
            return
        self._load_model()
        
    def _load_model(self):
        try:
            logger.info("Loading RapidOCR model")
            self.ocr = RapidOCR()
            logger.info("RapidOCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RapidOCR model: {e}")
            raise
    
    def recognize(self, image: np.ndarray) -> List[Dict]:
        """
        识别图像中的文字
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            识别结果列表，每个结果包含 {'text': str, 'confidence': float, 'bbox': List[int]}
        """
        try:
            if self.ocr is None:
                logger.error("OCR model not loaded")
                return []
            
            result, _ = self.ocr(image)
            
            if not result:
                return []
            
            recognized = []
            for line in result:
                box = line[0]
                text = line[1]
                confidence = float(line[2])
                
                bbox = []
                if box is not None:
                    bbox = [int(p) for point in box for p in point]
                
                recognized.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
            
            return recognized
            
        except Exception as e:
            logger.error(f"RapidOCREngine recognize error: {e}")
            return []
    
    def recognize_region(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        识别指定区域的文字
        
        Args:
            image: 原始图像
            bbox: 区域边界框 [x1, y1, x2, y2]
            
        Returns:
            识别结果 {'text': str, 'confidence': float, 'all_texts': List[Dict]}
        """
        try:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return {'text': '', 'confidence': 0.0, 'all_texts': []}
            
            region = image[y1:y2, x1:x2]
            results = self.recognize(region)
            
            if not results:
                return {'text': '', 'confidence': 0.0, 'all_texts': []}
            
            combined_text = ' '.join([r['text'] for r in results])
            avg_confidence = sum([r['confidence'] for r in results]) / len(results)
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'all_texts': results
            }
            
        except Exception as e:
            logger.error(f"RapidOCREngine recognize_region error: {e}")
            return {'text': '', 'confidence': 0.0, 'all_texts': []}
    
    def recognize_full_image_with_detection(self, image: np.ndarray, 
                                             detections: List[Dict],
                                             expand: int = 100) -> List[Dict]:
        """
        对全图进行OCR，然后匹配检测框附近的文字
        
        Args:
            image: 原始图像
            detections: YOLO检测结果列表
            expand: 检测框扩展范围（像素）
            
        Returns:
            更新后的检测结果列表，包含OCR信息
        """
        try:
            full_result, _ = self.ocr(image)
            
            all_texts = []
            if full_result:
                for line in full_result:
                    box = line[0]
                    text = line[1]
                    conf = line[2]
                    
                    if box is not None:
                        text_x1 = min(p[0] for p in box)
                        text_y1 = min(p[1] for p in box)
                        text_x2 = max(p[0] for p in box)
                        text_y2 = max(p[1] for p in box)
                        
                        all_texts.append({
                            'box': box,
                            'text': text,
                            'conf': float(conf),
                            'cx': (text_x1 + text_x2) / 2,
                            'cy': (text_y1 + text_y2) / 2
                        })
            
            for det in detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                
                nearby_texts = []
                for text_info in all_texts:
                    if (x1 - expand <= text_info['cx'] <= x2 + expand and 
                        y1 - expand <= text_info['cy'] <= y2 + expand):
                        nearby_texts.append({
                            'text': text_info['text'],
                            'confidence': text_info['conf']
                        })
                
                if nearby_texts:
                    det['ocr_text'] = ' '.join([t['text'] for t in nearby_texts])
                    det['ocr_confidence'] = sum(t['confidence'] for t in nearby_texts) / len(nearby_texts)
                    det['ocr_details'] = nearby_texts
                else:
                    det['ocr_text'] = ''
                    det['ocr_confidence'] = 0.0
                    det['ocr_details'] = []
            
            return detections
            
        except Exception as e:
            logger.error(f"RapidOCREngine recognize_full_image_with_detection error: {e}")
            for det in detections:
                det['ocr_text'] = ''
                det['ocr_confidence'] = 0.0
                det['ocr_details'] = []
            return detections


class YoloPose:
    """
    YOLOv11 Pose 模型类：用于检测面板四个角点
    """
    def __init__(self, model_path: str = 'yolo11n-pose.pt', device: str = '0', conf: float = 0.5):
        """
        初始化 YOLOv11 Pose 模型
        
        Args:
            model_path: 模型路径
            device: 推理设备 ('0' 或 'cpu')
            conf: 置信度阈值
        """
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """
        加载 YOLOv11 Pose 模型
        """
        try:
            logger.info(f"Loading YOLOv11 Pose model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("YOLOv11 Pose model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv11 Pose model: {e}")
            raise
        
    def predict(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        预测面板四个关键点
        
        Args:
            frame: 输入图像
            
        Returns:
            四个关键点的列表 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]，失败返回 None
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            results = self.model.predict(
                frame,
                conf=self.conf,
                device=self.device,
                verbose=False
            )
            
            if not results or len(results) == 0:
                return None
            
            result = results[0]
            if result.keypoints is None or len(result.keypoints) == 0:
                return None
            
            keypoints = result.keypoints.xy.cpu().numpy()[0]
            
            if len(keypoints) < 4:
                logger.warning(f"Expected 4 keypoints, got {len(keypoints)}")
                return None
            
            selected_keypoints = [(int(kp[0]), int(kp[1])) for kp in keypoints[:4]]
            return selected_keypoints
            
        except Exception as e:
            logger.error(f"YoloPose predict error: {e}")
            return None


class YoloDetect:
    """
    YOLOv11 检测模型类：用于检测发光体，支持对检测框内容进行OCR识别
    """
    def __init__(self, model_path: str = 'yolo11n.pt', device: str = '0', conf: float = 0.25,
                 iou: float = 0.45, imgsz: Optional[int] = None, max_det: int = 300,
                 enable_ocr: bool = True, ocr_lang: str = 'ch'):
        """
        初始化 YOLOv11 检测模型
        
        Args:
            model_path: 模型路径
            device: 推理设备 ('0' 或 'cpu')
            conf: 置信度阈值
            iou: NMS IOU 阈值
            imgsz: 推理尺寸，None 表示使用 Ultralytics 默认值
            max_det: 单张图最大检测数
            enable_ocr: 是否启用OCR识别
            ocr_lang: OCR语言 ('ch', 'en', etc.)
        """
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.enable_ocr = enable_ocr
        self.ocr_lang = ocr_lang
        self.model = None
        self.ocr_engine = None
        self._load_model()
        if enable_ocr:
            self._init_ocr()
        
    def _load_model(self):
        """
        加载 YOLOv11 检测模型
        """
        try:
            logger.info(f"Loading YOLOv11 detection model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("YOLOv11 detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv11 detection model: {e}")
            raise

    def get_runtime_metadata(self) -> Dict:
        """
        返回当前推理配置，方便排查网页链路与原生 YOLO 链路差异
        """
        model_names = {}
        resolved_model_path = self.model_path

        if self.model is not None:
            raw_names = getattr(self.model, 'names', {}) or {}
            if isinstance(raw_names, dict):
                model_names = {str(k): str(v) for k, v in raw_names.items()}
            elif isinstance(raw_names, list):
                model_names = {str(i): str(v) for i, v in enumerate(raw_names)}
            resolved_model_path = getattr(self.model, 'ckpt_path', self.model_path)

        return {
            'model_path': self.model_path,
            'resolved_model_path': resolved_model_path,
            'device': self.device,
            'conf': self.conf,
            'iou': self.iou,
            'imgsz': self.imgsz,
            'max_det': self.max_det,
            'names': model_names
        }
    
    def _init_ocr(self):
        """
        初始化 OCR 引擎 - 优先使用 RapidOCR
        """
        try:
            if RAPIDOCR_AVAILABLE:
                self.ocr_engine = RapidOCREngine()
                logger.info("RapidOCR engine initialized for YoloDetect")
            else:
                use_gpu = self.device not in ['cpu', '-1']
                self.ocr_engine = EasyOCREngine(lang=self.ocr_lang, use_gpu=use_gpu)
                logger.info("EasyOCR engine initialized for YoloDetect")
        except Exception as e:
            logger.warning(f"Failed to initialize OCR engine: {e}")
            self.ocr_engine = None
        
    def predict(self, frame: np.ndarray) -> List[Dict]:
        """
        预测发光体
        
        Args:
            frame: 输入图像
            
        Returns:
            检测结果列表，每个结果包含 {'bbox': [x1,y1,x2,y2], 'confidence': float, 'class': str}
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return []
            
            predict_kwargs = {
                'conf': self.conf,
                'iou': self.iou,
                'device': self.device,
                'max_det': self.max_det,
                'verbose': False
            }
            if self.imgsz is not None:
                predict_kwargs['imgsz'] = self.imgsz

            results = self.model.predict(frame, **predict_kwargs)
            
            if not results or len(results) == 0:
                return []
            
            result = results[0]
            detections = []
            
            if result.boxes is None:
                return []
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                confidence = float(box.conf.cpu().numpy()[0])
                class_id = int(box.cls.cpu().numpy()[0])
                class_name = result.names[class_id] if result.names else f"class_{class_id}"
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence,
                    'class': class_name
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"YoloDetect predict error: {e}")
            return []
    
    def predict_with_ocr(self, frame: np.ndarray, padding: int = 5) -> List[Dict]:
        """
        预测发光体并对检测框内容进行OCR识别
        
        Args:
            frame: 输入图像
            padding: 检测框扩展像素（用于更好地识别边缘文字）
            
        Returns:
            检测结果列表，每个结果包含:
            {
                'bbox': [x1,y1,x2,y2],
                'confidence': float,
                'class': str,
                'ocr_text': str,           # OCR识别的文字
                'ocr_confidence': float,   # OCR置信度
                'ocr_details': List[Dict]  # OCR详细结果
            }
        """
        try:
            detections = self.predict(frame)
            
            if not detections or not self.enable_ocr or self.ocr_engine is None:
                for det in detections:
                    det['ocr_text'] = ''
                    det['ocr_confidence'] = 0.0
                    det['ocr_details'] = []
                return detections
            
            for det in detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                
                padded_bbox = [
                    max(0, x1 - padding),
                    max(0, y1 - padding),
                    min(frame.shape[1], x2 + padding),
                    min(frame.shape[0], y2 + padding)
                ]
                
                ocr_result = self.ocr_engine.recognize_region(frame, padded_bbox)
                
                det['ocr_text'] = ocr_result['text']
                det['ocr_confidence'] = ocr_result['confidence']
                det['ocr_details'] = ocr_result['all_texts']
            
            return detections
            
        except Exception as e:
            logger.error(f"YoloDetect predict_with_ocr error: {e}")
            return []
    
    def predict_with_full_ocr(self, frame: np.ndarray, expand: int = 100) -> List[Dict]:
        """
        预测发光体并对全图进行OCR，匹配检测框附近的文字
        适用于检测框区域太小的情况
        
        Args:
            frame: 输入图像
            expand: 检测框扩展范围（像素）
            
        Returns:
            检测结果列表，每个结果包含:
            {
                'bbox': [x1,y1,x2,y2],
                'confidence': float,
                'class': str,
                'ocr_text': str,
                'ocr_confidence': float,
                'ocr_details': List[Dict]
            }
        """
        try:
            detections = self.predict(frame)
            
            if not detections:
                return detections
            
            if not self.enable_ocr or self.ocr_engine is None:
                for det in detections:
                    det['ocr_text'] = ''
                    det['ocr_confidence'] = 0.0
                    det['ocr_details'] = []
                return detections
            
            if isinstance(self.ocr_engine, RapidOCREngine):
                return self.ocr_engine.recognize_full_image_with_detection(frame, detections, expand)
            else:
                return self.predict_with_ocr(frame, padding=expand)
            
        except Exception as e:
            logger.error(f"YoloDetect predict_with_full_ocr error: {e}")
            return []
    
    def draw_detections_with_ocr(self, frame: np.ndarray, detections: List[Dict],
                                  draw_ocr: bool = True) -> np.ndarray:
        """
        在图像上绘制检测结果和OCR文字
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            draw_ocr: 是否绘制OCR文字
            
        Returns:
            绘制后的图像
        """
        try:
            viz_frame = frame.copy()
            
            for det in detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                confidence = det['confidence']
                class_name = det['class']
                
                cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(viz_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if draw_ocr and 'ocr_text' in det and det['ocr_text']:
                    ocr_text = det['ocr_text']
                    ocr_conf = det.get('ocr_confidence', 0)
                    
                    text_label = f"OCR: {ocr_text} ({ocr_conf:.2f})"
                    cv2.putText(viz_frame, text_label, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            return viz_frame
            
        except Exception as e:
            logger.error(f"YoloDetect draw_detections_with_ocr error: {e}")
            return frame
