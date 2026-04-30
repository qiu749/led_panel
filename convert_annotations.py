import json
import os
from pathlib import Path

def convert_json_to_yolo(json_dir, output_dir, image_width=720, image_height=1280):
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])
        if not shapes:
            continue

        img_w = data.get('imageWidth', image_width)
        img_h = data.get('imageHeight', image_height)

        txt_file = json_file.replace('.json', '.txt')
        txt_path = os.path.join(output_dir, txt_file)

        with open(txt_path, 'w') as f:
            for shape in shapes:
                label = shape.get('label', 'light')
                points = shape.get('points', [])

                if len(points) != 4:
                    continue

                x1, y1 = points[0]
                x2, y2 = points[2]

                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                converted += 1

    return converted

if __name__ == "__main__":
    base_dir = "dataset_qwen_grounding/images"

    train_json_dir = os.path.join(base_dir, "train")
    train_label_dir = os.path.join(base_dir, "..", "labels", "train")
    train_count = convert_json_to_yolo(train_json_dir, train_label_dir)
    print(f"Converted {train_count} training labels")

    val_json_dir = os.path.join(base_dir, "val")
    val_label_dir = os.path.join(base_dir, "..", "labels", "val")
    val_count = convert_json_to_yolo(val_json_dir, val_label_dir)
    print(f"Converted {val_count} validation labels")
