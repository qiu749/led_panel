import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_models import YoloDetect


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_names(names) -> Dict[str, str]:
    if isinstance(names, dict):
        return {str(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {str(i): str(name) for i, name in enumerate(names)}
    return {}


def run_raw_yolo(
    image,
    model_path: str,
    conf: float,
    iou: float,
    imgsz: Optional[int],
    max_det: int,
    device: str
) -> Dict:
    model = YOLO(model_path)
    predict_kwargs = {
        'conf': conf,
        'iou': iou,
        'device': device,
        'max_det': max_det,
        'verbose': False
    }
    if imgsz is not None:
        predict_kwargs['imgsz'] = imgsz

    results = model.predict(image, **predict_kwargs)
    result = results[0]

    detections: List[Dict] = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
            class_id = int(box.cls.cpu().numpy()[0])
            confidence = float(box.conf.cpu().numpy()[0])
            names = normalize_names(result.names)
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class_id': class_id,
                'class_name': names.get(str(class_id), f'class_{class_id}')
            })

    return {
        'metadata': {
            'model_path': model_path,
            'resolved_model_path': str(getattr(model, 'ckpt_path', model_path)),
            'device': device,
            'conf': conf,
            'iou': iou,
            'imgsz': imgsz,
            'max_det': max_det,
            'names': normalize_names(getattr(model, 'names', {}))
        },
        'detections': detections
    }


def run_service_wrapper(
    image,
    model_path: str,
    conf: float,
    iou: float,
    imgsz: Optional[int],
    max_det: int,
    device: str
) -> Dict:
    detector = YoloDetect(
        model_path=model_path,
        device=device,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        enable_ocr=False
    )
    detections = detector.predict(image)
    converted: List[Dict] = []
    for det in detections:
        converted.append({
            'bbox': det['bbox'],
            'confidence': det['confidence'],
            'class_name': det['class']
        })

    return {
        'metadata': detector.get_runtime_metadata(),
        'detections': converted
    }


def map_detections_to_rois(detections: List[Dict], rois: Dict) -> Dict:
    roi_hits = {name: [] for name in rois}
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        for roi_name, roi in rois.items():
            if roi['x1'] <= cx <= roi['x2'] and roi['y1'] <= cy <= roi['y2']:
                roi_hits[roi_name].append(det)
                break

    return {
        'roi_count': sum(1 for hits in roi_hits.values() if hits),
        'roi_hits': {name: len(hits) for name, hits in roi_hits.items()},
        'roi_state': {name: bool(hits) for name, hits in roi_hits.items()}
    }


def draw_detections(image, detections: List[Dict], output_path: Path, color) -> None:
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det.get('class_name', 'unknown')
        if 'confidence' in det:
            label = f"{label} {det['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    cv2.imwrite(str(output_path), canvas)


def draw_roi_overlay(image, rois: Dict, roi_state: Dict[str, bool], output_path: Path) -> None:
    canvas = image.copy()
    for roi_name, roi in rois.items():
        color = (0, 255, 0) if roi_state.get(roi_name, False) else (0, 0, 255)
        cv2.rectangle(canvas, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), color, 2)
        cv2.putText(
            canvas,
            roi_name,
            (roi['x1'], max(20, roi['y1'] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    cv2.imwrite(str(output_path), canvas)


def load_rois(config_path: Optional[str]) -> Optional[Dict]:
    if not config_path:
        return None
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get('rois')


def main():
    parser = argparse.ArgumentParser(description='对比原生 YOLO 输出与业务封装输出')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument('--model', required=True, help='模型权重路径')
    parser.add_argument('--device', default='cpu', help='推理设备')
    parser.add_argument('--raw-conf', type=float, default=0.25, help='原生 YOLO conf')
    parser.add_argument('--raw-iou', type=float, default=0.45, help='原生 YOLO iou')
    parser.add_argument('--raw-imgsz', type=int, default=0, help='原生 YOLO imgsz，0 表示默认')
    parser.add_argument('--service-conf', type=float, default=0.25, help='业务封装 conf')
    parser.add_argument('--service-iou', type=float, default=0.45, help='业务封装 iou')
    parser.add_argument('--service-imgsz', type=int, default=0, help='业务封装 imgsz，0 表示默认')
    parser.add_argument('--max-det', type=int, default=300, help='单图最大检测数')
    parser.add_argument('--roi-config', type=str, default='', help='可选，ROI 配置 JSON 路径')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(ROOT / 'output' / 'compare_yolo_pipelines'),
        help='输出目录'
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f'无法读取图片: {image_path}')

    raw_imgsz = args.raw_imgsz or None
    service_imgsz = args.service_imgsz or None

    raw_result = run_raw_yolo(
        image=image,
        model_path=args.model,
        conf=args.raw_conf,
        iou=args.raw_iou,
        imgsz=raw_imgsz,
        max_det=args.max_det,
        device=args.device
    )
    service_result = run_service_wrapper(
        image=image,
        model_path=args.model,
        conf=args.service_conf,
        iou=args.service_iou,
        imgsz=service_imgsz,
        max_det=args.max_det,
        device=args.device
    )

    report = {
        'image_path': str(image_path.resolve()),
        'image_shape': list(image.shape),
        'raw_yolo': raw_result,
        'service_wrapper': service_result
    }

    rois = load_rois(args.roi_config)
    if rois:
        grouped = map_detections_to_rois(service_result['detections'], rois)
        report['service_wrapper']['roi_grouping'] = grouped
        draw_roi_overlay(
            image=image,
            rois=rois,
            roi_state=grouped['roi_state'],
            output_path=output_dir / 'service_roi_overlay.jpg'
        )

    draw_detections(image, raw_result['detections'], output_dir / 'raw_yolo.jpg', (255, 0, 0))
    draw_detections(image, service_result['detections'], output_dir / 'service_wrapper.jpg', (0, 255, 0))

    report_path = output_dir / 'report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info('原生 YOLO 权重: %s', raw_result['metadata']['resolved_model_path'])
    logger.info('业务封装权重: %s', service_result['metadata']['resolved_model_path'])
    logger.info('原生 YOLO names: %s', raw_result['metadata']['names'])
    logger.info('业务封装 names: %s', service_result['metadata']['names'])
    logger.info('原生 YOLO 检测框数: %d', len(raw_result['detections']))
    logger.info('业务封装检测框数: %d', len(service_result['detections']))
    if rois:
        logger.info('业务封装 ROI 命中数: %d', report['service_wrapper']['roi_grouping']['roi_count'])
    logger.info('对比报告已保存: %s', report_path)


if __name__ == '__main__':
    main()
