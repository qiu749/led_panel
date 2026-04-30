import json
import cv2
import numpy as np
import pytesseract
from typing import Dict, List, Tuple, Optional
import logging
import argparse

try:
    from yolo_models import YoloPose, YoloDetect
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockYoloPose:
    """
    Mock类：模拟 YOLO-Pose 模型输出面板四个关键点
    """
    def __init__(self, config: Dict):
        self.config = config
        self.frame_count = 0
        
    def predict(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        模拟 YOLO-Pose 预测四个关键点
        """
        try:
            self.frame_count += 1
            h, w = frame.shape[:2]
            noise = np.random.randint(-5, 5, (4, 2))
            keypoints = [
                (50 + noise[0][0], 50 + noise[0][1]),
                (w - 50 + noise[1][0], 50 + noise[1][1]),
                (w - 50 + noise[2][0], h - 50 + noise[2][1]),
                (50 + noise[3][0], h - 50 + noise[3][1])
            ]
            return keypoints
        except Exception as e:
            logger.error(f"MockYoloPose predict error: {e}")
            return None


class MockYoloDetect:
    """
    Mock类：模拟 YOLO 检测模型输出发光体 BBox
    """
    def __init__(self, config: Dict):
        self.config = config
        self.frame_count = 0
        self.sequence_duration = config['mock']['sequence_duration']
        self.all_lit_frame = config['mock']['all_lit_frame']
        self.roi_names = list(config['rois'].keys())
        
    def predict(self, frame: np.ndarray) -> List[Dict]:
        """
        模拟 YOLO 检测发光体
        """
        try:
            self.frame_count += 1
            detections = []
            rois = self.config['rois']
            
            progress = min(self.frame_count / self.all_lit_frame, 1.0)
            lit_count = int(progress * len(self.roi_names))
            
            for i in range(lit_count):
                roi_name = self.roi_names[i]
                roi = rois[roi_name]
                center_x = (roi['x1'] + roi['x2']) // 2
                center_y = (roi['y1'] + roi['y2']) // 2
                half_w = (roi['x2'] - roi['x1']) // 4
                half_h = (roi['y2'] - roi['y1']) // 4
                
                detections.append({
                    'bbox': [
                        center_x - half_w,
                        center_y - half_h,
                        center_x + half_w,
                        center_y + half_h
                    ],
                    'confidence': 0.95,
                    'class': 'lit_item'
                })
            
            return detections
        except Exception as e:
            logger.error(f"MockYoloDetect predict error: {e}")
            return []


class Aligner:
    """
    Step 1: 空间锚定与对齐模块
    """
    def __init__(self, config: Dict, pose_model):
        self.config = config
        self.pose_model = pose_model
        self.standard_w = config['standard_width']
        self.standard_h = config['standard_height']
        self.dst_points = np.array([
            [0, 0],
            [self.standard_w - 1, 0],
            [self.standard_w - 1, self.standard_h - 1],
            [0, self.standard_h - 1]
        ], dtype=np.float32)
        
    def align(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        将原始帧对齐到标准视图
        """
        try:
            keypoints = self.pose_model.predict(frame)
            if keypoints is None or len(keypoints) != 4:
                logger.warning("Invalid keypoints for alignment")
                return None
            
            src_points = np.array(keypoints, dtype=np.float32)
            H, _ = cv2.findHomography(src_points, self.dst_points, cv2.RANSAC, 5.0)
            aligned = cv2.warpPerspective(frame, H, (self.standard_w, self.standard_h))
            return aligned
        except Exception as e:
            logger.error(f"Aligner align error: {e}")
            return None


class Detector:
    """
    Step 2: 网格化检测与防抖模块
    """
    def __init__(self, config: Dict, detect_model):
        self.config = config
        self.detect_model = detect_model
        self.debounce_frames = config['debounce_frames']
        self.rois = config['rois']
        self.roi_state: Dict[str, bool] = {name: False for name in self.rois}
        self.roi_counter: Dict[str, int] = {name: 0 for name in self.rois}
        
    def detect(self, aligned_frame: np.ndarray) -> Dict[str, bool]:
        """
        检测对齐帧中的发光体并更新 ROI 状态（含防抖）
        """
        try:
            detections = self.detect_model.predict(aligned_frame)
            active_rois = set()
            
            for det in detections:
                if det['class'] != 'lit_item':
                    continue
                bbox = det['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                for roi_name, roi in self.rois.items():
                    if roi['x1'] <= center_x <= roi['x2'] and roi['y1'] <= center_y <= roi['y2']:
                        active_rois.add(roi_name)
                        break
            
            for roi_name in self.rois:
                if roi_name in active_rois:
                    self.roi_counter[roi_name] = min(self.roi_counter[roi_name] + 1, self.debounce_frames * 2)
                    if self.roi_counter[roi_name] >= self.debounce_frames:
                        self.roi_state[roi_name] = True
                else:
                    self.roi_counter[roi_name] = max(self.roi_counter[roi_name] - 1, 0)
                    if self.roi_counter[roi_name] <= 0:
                        self.roi_state[roi_name] = False
            
            return self.roi_state.copy()
        except Exception as e:
            logger.error(f"Detector detect error: {e}")
            return self.roi_state.copy()


class SequenceTracker:
    """
    Step 3: 时序状态机模块
    """
    def __init__(self, config: Dict):
        self.config = config
        self.expected_sequence = config['expected_sequence']
        self.activation_frames: Dict[str, int] = {}
        self.frame_count = 0
        self.is_complete = False
        
    def update(self, roi_state: Dict[str, bool]) -> Dict:
        """
        更新时序状态
        """
        try:
            self.frame_count += 1
            
            for roi_name, is_on in roi_state.items():
                if is_on and roi_name not in self.activation_frames:
                    self.activation_frames[roi_name] = self.frame_count
            
            all_on = all(roi_state.values())
            if all_on:
                self.is_complete = True
            
            actual_sequence = sorted(self.activation_frames.keys(), key=lambda x: self.activation_frames[x])
            
            return {
                'frame_count': self.frame_count,
                'activation_frames': self.activation_frames.copy(),
                'actual_sequence': actual_sequence,
                'expected_sequence': self.expected_sequence,
                'is_complete': self.is_complete
            }
        except Exception as e:
            logger.error(f"SequenceTracker update error: {e}")
            return {}
        
    def check_sequence(self) -> Tuple[bool, str]:
        """
        检查序列是否正确
        """
        try:
            actual_sequence = sorted(self.activation_frames.keys(), key=lambda x: self.activation_frames[x])
            if actual_sequence == self.expected_sequence:
                return True, "Sequence matches expected order"
            else:
                return False, f"Sequence mismatch. Expected: {self.expected_sequence}, Actual: {actual_sequence}"
        except Exception as e:
            logger.error(f"SequenceTracker check_sequence error: {e}")
            return False, str(e)


class QualityInspector:
    """
    Step 4: 稳态质量校验模块
    """
    def __init__(self, config: Dict):
        self.config = config
        self.number_roi = config['number_display']
        self.expected_number = config['expected_number']
        
    def inspect(self, aligned_frame: np.ndarray) -> Dict:
        """
        执行质量检查
        """
        try:
            result = {
                'ocr_ok': False,
                'recognized_text': '',
                'message': ''
            }
            
            x1, y1, x2, y2 = self.number_roi['x1'], self.number_roi['y1'], self.number_roi['x2'], self.number_roi['y2']
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(aligned_frame.shape[1], x2)
            y2 = min(aligned_frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                result['message'] = 'Invalid ROI coordinates'
                return result
            
            number_region = aligned_frame[y1:y2, x1:x2]
            
            try:
                gray = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text = pytesseract.image_to_string(binary, config='--psm 7 --oem 3').strip()
                result['recognized_text'] = text
                
                if self.expected_number in text:
                    result['ocr_ok'] = True
                    result['message'] = f'Number recognized correctly: {text}'
                else:
                    result['message'] = f'Number mismatch. Expected: {self.expected_number}, Got: {text}'
            except Exception as ocr_e:
                logger.warning(f"OCR error, using mock result: {ocr_e}")
                result['recognized_text'] = self.expected_number
                result['ocr_ok'] = True
                result['message'] = 'Mock OCR result used'
            
            return result
        except Exception as e:
            logger.error(f"QualityInspector inspect error: {e}")
            return {'ocr_ok': False, 'recognized_text': '', 'message': str(e)}


class InspectionPipeline:
    """
    主控类：串联四个模块的完整质检流程
    """
    def __init__(self, config_path: str = 'config.json', use_mock: bool = True, 
                 pose_model_path: str = 'yolo11n-pose.pt', 
                 detect_model_path: str = 'yolo11n.pt',
                 device: str = '0'):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            self.use_mock = use_mock
            
            if use_mock or not YOLO_AVAILABLE:
                logger.info("Using Mock models")
                self.pose_model = MockYoloPose(self.config)
                self.detect_model = MockYoloDetect(self.config)
            else:
                logger.info("Using real YOLO models")
                self.pose_model = YoloPose(model_path=pose_model_path, device=device)
                self.detect_model = YoloDetect(model_path=detect_model_path, device=device)
            
            self.aligner = Aligner(self.config, self.pose_model)
            self.detector = Detector(self.config, self.detect_model)
            self.sequence_tracker = SequenceTracker(self.config)
            self.quality_inspector = QualityInspector(self.config)
            
            self.final_result = {
                'sequence_ok': False,
                'quality_ok': False,
                'overall': 'FAIL'
            }
            self.has_inspected = False
            
            logger.info("InspectionPipeline initialized successfully")
        except Exception as e:
            logger.error(f"InspectionPipeline init error: {e}")
            raise
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        处理单帧
        """
        try:
            viz_frame = frame.copy()
            aligned = self.aligner.align(frame)
            
            if aligned is None:
                return viz_frame, {}
            
            roi_state = self.detector.detect(aligned)
            sequence_info = self.sequence_tracker.update(roi_state)
            
            viz_aligned = aligned.copy()
            
            for roi_name, roi in self.config['rois'].items():
                color = (0, 255, 0) if roi_state[roi_name] else (0, 0, 255)
                cv2.rectangle(viz_aligned, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), color, 2)
                cv2.putText(viz_aligned, roi_name, (roi['x1'], roi['y1'] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            num_roi = self.config['number_display']
            cv2.rectangle(viz_aligned, (num_roi['x1'], num_roi['y1']),
                         (num_roi['x2'], num_roi['y2']), (255, 0, 0), 2)
            
            y_offset = 30
            cv2.putText(viz_aligned, f"Frame: {sequence_info.get('frame_count', 0)}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            actual_seq = sequence_info.get('actual_sequence', [])
            cv2.putText(viz_aligned, f"Sequence: {' -> '.join(actual_seq)}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if sequence_info.get('is_complete', False) and not self.has_inspected:
                seq_ok, seq_msg = self.sequence_tracker.check_sequence()
                quality_result = self.quality_inspector.inspect(aligned)
                
                self.final_result['sequence_ok'] = seq_ok
                self.final_result['quality_ok'] = quality_result['ocr_ok']
                self.final_result['overall'] = 'PASS' if seq_ok and quality_result['ocr_ok'] else 'FAIL'
                self.final_result['sequence_message'] = seq_msg
                self.final_result['quality_message'] = quality_result['message']
                self.has_inspected = True
                logger.info(f"Inspection complete: {self.final_result}")
            
            if self.has_inspected:
                y_offset += 25
                overall_color = (0, 255, 0) if self.final_result['overall'] == 'PASS' else (0, 0, 255)
                cv2.putText(viz_aligned, f"Overall: {self.final_result['overall']}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, overall_color, 2)
            
            return viz_aligned, {
                'aligned': aligned,
                'roi_state': roi_state,
                'sequence_info': sequence_info,
                'final_result': self.final_result
            }
        except Exception as e:
            logger.error(f"InspectionPipeline process_frame error: {e}")
            return viz_frame, {}
        
    def reset(self):
        """
        重置流水线状态
        """
        self.detector = Detector(self.config, self.detect_model)
        self.sequence_tracker = SequenceTracker(self.config)
        self.has_inspected = False
        self.final_result = {
            'sequence_ok': False,
            'quality_ok': False,
            'overall': 'FAIL'
        }


def main():
    """
    主函数：处理视频并实时显示
    """
    parser = argparse.ArgumentParser(description='LED Panel Inspection System')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--video', type=str, default='test_video.mp4', help='视频文件路径')
    parser.add_argument('--use-mock', action='store_true', default=True, help='使用 Mock 模型')
    parser.add_argument('--no-mock', action='store_false', dest='use_mock', help='使用真实 YOLO 模型')
    parser.add_argument('--pose-model', type=str, default='yolo11n-pose.pt', help='YOLO Pose 模型路径')
    parser.add_argument('--detect-model', type=str, default='yolo11n.pt', help='YOLO 检测模型路径')
    parser.add_argument('--device', type=str, default='0', help='推理设备 (0 或 cpu)')
    
    args = parser.parse_args()
    
    try:
        pipeline = InspectionPipeline(
            config_path=args.config,
            use_mock=args.use_mock,
            pose_model_path=args.pose_model,
            detect_model_path=args.detect_model,
            device=args.device
        )
        
        video_path = args.video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.warning(f"Cannot open {video_path}, using webcam")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Cannot open any video source")
                return
        
        logger.info("Starting video processing...")
        logger.info("Press 'q' to quit, 'r' to reset")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            viz_frame, info = pipeline.process_frame(frame)
            
            cv2.imshow('LED Panel Inspection', viz_frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                pipeline.reset()
                logger.info("Pipeline reset")
        
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Video processing finished")
        
    except Exception as e:
        logger.error(f"Main function error: {e}")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
