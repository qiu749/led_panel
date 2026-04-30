"""
分析YOLO检测框内容
显示检测框区域并进行OCR识别
"""

import os
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_frames_with_yolo(
    frames_dir: str,
    model_path: str,
    output_dir: str = 'yolo_analysis',
    padding: int = 80
):
    """
    分析帧图像的YOLO检测框内容
    
    Args:
        frames_dir: 帧图像目录
        model_path: YOLO模型路径
        output_dir: 输出目录
        padding: 检测框扩展像素
    """
    from ultralytics import YOLO
    from rapidocr import RapidOCR
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("加载模型...")
    detector = YOLO(model_path)
    ocr = RapidOCR()
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    logger.info(f"找到 {len(frame_files)} 帧图像")
    
    all_results = []
    
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            continue
        
        height, width = frame.shape[:2]
        
        results = detector.predict(frame, conf=0.25, device='cpu', verbose=False)
        
        frame_results = {
            'frame': frame_file,
            'detections': []
        }
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf.cpu().numpy()[0])
                cls_name = results[0].names[int(box.cls.cpu().numpy()[0])]
                
                x1_p = max(0, x1 - padding)
                y1_p = max(0, y1 - padding)
                x2_p = min(width, x2 + padding)
                y2_p = min(height, y2 + padding)
                
                region = frame[y1_p:y2_p, x1_p:x2_p]
                
                region_filename = f"{frame_file.replace('.jpg', '')}_box{i}.jpg"
                region_path = os.path.join(output_dir, region_filename)
                cv2.imwrite(region_path, region)
                
                ocr_texts = []
                if region.shape[0] > 20 and region.shape[1] > 20:
                    ocr_result = ocr(region)
                    
                    if ocr_result.boxes is not None and ocr_result.txts is not None:
                        for obox, txt, score in zip(ocr_result.boxes, ocr_result.txts, ocr_result.scores):
                            if score > 0.3:
                                ocr_texts.append(f"{txt}({score:.2f})")
                
                detection_info = {
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'bbox_size': f"{x2-x1}x{y2-y1}",
                    'region_size': f"{region.shape[1]}x{region.shape[0]}",
                    'region_file': region_filename,
                    'ocr': ocr_texts
                }
                frame_results['detections'].append(detection_info)
                
                print(f"\n{frame_file} - 检测框 {i+1}:")
                print(f"  类别: {cls_name} (置信度: {conf:.2f})")
                print(f"  检测框: [{x1}, {y1}, {x2}, {y2}] 尺寸: {x2-x1}x{y2-y1}")
                print(f"  OCR区域: {region.shape[1]}x{region.shape[0]} (padding={padding})")
                print(f"  保存到: {region_path}")
                if ocr_texts:
                    print(f"  OCR结果: {', '.join(ocr_texts)}")
                else:
                    print(f"  OCR结果: 无文字")
        
        all_results.append(frame_results)
        
        vis_frame = frame.copy()
        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        vis_path = os.path.join(output_dir, f"vis_{frame_file}")
        cv2.imwrite(vis_path, vis_frame)
    
    print(f"\n分析完成!")
    print(f"检测框区域保存到: {output_dir}")
    print(f"共处理 {len(frame_files)} 帧")
    
    total_detections = sum(len(r['detections']) for r in all_results)
    print(f"共检测到 {total_detections} 个目标")
    
    return all_results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='分析YOLO检测框内容')
    parser.add_argument('--frames', type=str, required=True, help='帧图像目录')
    parser.add_argument('--model', type=str, required=True, help='YOLO模型路径')
    parser.add_argument('--output', type=str, default='yolo_analysis', help='输出目录')
    parser.add_argument('--padding', type=int, default=80, help='检测框扩展像素')
    
    args = parser.parse_args()
    
    analyze_frames_with_yolo(
        frames_dir=args.frames,
        model_path=args.model,
        output_dir=args.output,
        padding=args.padding
    )
