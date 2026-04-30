import os
import cv2
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = r"D:\Desktop2\Images\led_model_v2\weights\best.pt"
IMAGE_DIRS = [
    r"D:\Desktop2\Images\c92375603e023cfd70c37b072edc9da3_frames",
    r"D:\Desktop2\Images",
]
OUTPUT_DIR = r"D:\Desktop2\Images\roi_crops"
PADDING = 10

def extract_roi_crops():
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_paths = []
    for img_dir in IMAGE_DIRS:
        if os.path.isdir(img_dir):
            for f in os.listdir(img_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(img_dir, f))
        elif os.path.isfile(img_dir) and img_dir.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(img_dir)

    image_paths = [p for p in image_paths if '_detect' not in p and 'roi_crops' not in p]
    print(f"找到 {len(image_paths)} 张图片")

    total_rois = 0
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"读取失败: {img_path}")
            continue

        basename = Path(img_path).stem
        results = model(img, verbose=False)

        roi_count = 0
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]

                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(img.shape[1], x2 + PADDING)
                y2 = min(img.shape[0], y2 + PADDING)

                roi = img[y1:y2, x1:x2]

                roi_filename = f"{basename}_roi_{cls_name}_{roi_count}.jpg"
                roi_path = os.path.join(OUTPUT_DIR, roi_filename)
                cv2.imwrite(roi_path, roi)
                roi_count += 1
                total_rois += 1

        print(f"处理: {basename} - 检测到 {roi_count} 个ROI")

    print(f"\n完成! 共提取 {total_rois} 个ROI，保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_roi_crops()