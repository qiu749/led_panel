import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROI_DIR = r"D:\Desktop2\Images\roi_crops"
ORIGINAL_DIR = r"D:\Desktop2\Images"
LABELS_DIR = r"d:\led-visual-inspection\roi_labeled\labels"
CROP_OUTPUT_DIR = r"d:\led-visual-inspection\roi_labeled\visualized"
FULL_OUTPUT_DIR = r"d:\led-visual-inspection\roi_labeled\visualized_full"
MODEL_PATH = r"D:\Desktop2\Images\led_model_v2\weights\best.pt"
PADDING = 10
ROI_PATTERN = re.compile(r"(.+)_roi_(.+)_(\d+)$")


def yolo_to_xyxy(parts, image_width, image_height):
    x_center = float(parts[1])
    y_center = float(parts[2])
    box_w = float(parts[3])
    box_h = float(parts[4])
    x1 = int((x_center - box_w / 2) * image_width)
    y1 = int((y_center - box_h / 2) * image_height)
    x2 = int((x_center + box_w / 2) * image_width)
    y2 = int((y_center + box_h / 2) * image_height)
    return x1, y1, x2, y2


def collect_roi_boxes(model, image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"读取失败: {image_path}")

    roi_boxes = []
    results = model(image, verbose=False)
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]

            x1 = max(0, x1 - PADDING)
            y1 = max(0, y1 - PADDING)
            x2 = min(image.shape[1], x2 + PADDING)
            y2 = min(image.shape[0], y2 + PADDING)

            roi_boxes.append(
                {
                    "class_name": cls_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

    return image, roi_boxes


os.makedirs(CROP_OUTPUT_DIR, exist_ok=True)
os.makedirs(FULL_OUTPUT_DIR, exist_ok=True)

label_files = sorted(Path(LABELS_DIR).glob("*.txt"))
print(f"找到 {len(label_files)} 个标注文件")

groups = defaultdict(list)
for label_file in label_files:
    match = ROI_PATTERN.match(label_file.stem)
    if not match:
        print(f"跳过无法解析的标签文件: {label_file.name}")
        continue

    original_stem, class_name, roi_index = match.groups()
    groups[original_stem].append(
        {
            "label_file": label_file,
            "class_name": class_name,
            "roi_index": int(roi_index),
        }
    )

model = YOLO(MODEL_PATH)

for original_stem, items in groups.items():
    original_path = os.path.join(ORIGINAL_DIR, f"{original_stem}.jpg")
    if not os.path.exists(original_path):
        print(f"原图不存在: {original_path}")
        continue

    full_image, roi_boxes = collect_roi_boxes(model, original_path)
    items.sort(key=lambda item: item["roi_index"])
    print(f"\n处理原图: {original_stem}.jpg")

    for item in items:
        roi_index = item["roi_index"]
        if roi_index >= len(roi_boxes):
            print(f"  ROI 索引越界: {item['label_file'].name}")
            continue

        roi_box = roi_boxes[roi_index]
        if roi_box["class_name"] != item["class_name"]:
            print(
                f"  ROI 类别不匹配: {item['label_file'].name} -> "
                f"{roi_box['class_name']}"
            )

        crop_image_name = item["label_file"].stem + ".jpg"
        crop_image_path = os.path.join(ROI_DIR, crop_image_name)
        crop_image = cv2.imread(crop_image_path)
        if crop_image is None:
            print(f"  ROI 图片不存在: {crop_image_path}")
            continue

        crop_h, crop_w = crop_image.shape[:2]
        with open(item["label_file"], "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        print(f"  ROI {roi_index}: {crop_image_name}，标注数量 {len(lines)}")

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue

            x1, y1, x2, y2 = yolo_to_xyxy(parts, crop_w, crop_h)
            cv2.rectangle(crop_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                crop_image,
                "on",
                (x1, max(10, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 0),
                1,
            )

            full_x1 = roi_box["x1"] + x1
            full_y1 = roi_box["y1"] + y1
            full_x2 = roi_box["x1"] + x2
            full_y2 = roi_box["y1"] + y2
            cv2.rectangle(full_image, (full_x1, full_y1), (full_x2, full_y2), (0, 255, 0), 1)

        cv2.rectangle(
            full_image,
            (roi_box["x1"], roi_box["y1"]),
            (roi_box["x2"], roi_box["y2"]),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            full_image,
            f"roi_{item['class_name']}_{roi_index}",
            (roi_box["x1"], max(20, roi_box["y1"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
        )

        crop_output_path = os.path.join(CROP_OUTPUT_DIR, crop_image_name)
        cv2.imwrite(crop_output_path, crop_image)
        print(f"    保存 ROI 可视化: {crop_output_path}")

    full_output_path = os.path.join(FULL_OUTPUT_DIR, f"{original_stem}.jpg")
    cv2.imwrite(full_output_path, full_image)
    print(f"  保存原图可视化: {full_output_path}")

print(f"\n完成！ROI 可视化保存在: {CROP_OUTPUT_DIR}")
print(f"完成！原图可视化保存在: {FULL_OUTPUT_DIR}")
