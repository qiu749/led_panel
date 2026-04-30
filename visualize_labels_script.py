import cv2
import os
import glob
import numpy as np

# 配置路径，需要和 auto_label.py 里一致
IMAGE_DIR = r"D:\Desktop2\Images\extracted_frames"
LABEL_DIR = r"D:\Desktop2\Images\labels"
OUTPUT_DIR = r"D:\Desktop2\Images\visualized_labels"

# YOLO ID 到类别名的映射 (反向映射)
ID_TO_CLASS = {
    0: "ledgroup1",
    1: "ledgroup2",
    2: "ledgroup3"
}

# 类别对应的框颜色 (BGR格式)
CLASS_COLORS = {
    0: (0, 0, 255),    # 红
    1: (0, 255, 0),    # 绿
    2: (255, 0, 0)     # 蓝
}

def yolo_to_bbox(yolo_box, img_width, img_height):
    """将 YOLO 的相对坐标 [x_center, y_center, w, h] 转换为绝对坐标 [xmin, ymin, xmax, ymax]"""
    x_center, y_center, w, h = yolo_box
    
    # 恢复绝对坐标
    w_abs = w * img_width
    h_abs = h * img_height
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    
    xmin = int(x_center_abs - w_abs / 2)
    ymin = int(y_center_abs - h_abs / 2)
    xmax = int(x_center_abs + w_abs / 2)
    ymax = int(y_center_abs + h_abs / 2)
    
    return xmin, ymin, xmax, ymax

def visualize_labels():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 寻找所有的 txt 标签文件
    label_files = glob.glob(os.path.join(LABEL_DIR, "*.txt"))
    if not label_files:
        print(f"在 {LABEL_DIR} 中没有找到任何标签文件。")
        return

    print(f"找到 {len(label_files)} 个标签文件，准备开始可视化...")

    for label_path in label_files:
        # 获取不带扩展名的文件名，用来推导图片路径
        base_name = os.path.splitext(os.path.basename(label_path))[0]
        
        # 尝试寻找对应的图片 (支持常见的格式)
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            temp_path = os.path.join(IMAGE_DIR, base_name + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        if img_path is None:
            print(f"警告: 找不到标签 {base_name}.txt 对应的图片文件。")
            continue

        # 解决 cv2 读取中文路径问题
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img is None:
            print(f"警告: 无法读取图片 {img_path}")
            continue

        img_height, img_width = img.shape[:2]

        # 读取标签文件
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            cls_id = int(parts[0])
            yolo_box = [float(p) for p in parts[1:]]
            
            # 坐标转换
            xmin, ymin, xmax, ymax = yolo_to_bbox(yolo_box, img_width, img_height)
            
            # 获取颜色和类别名
            color = CLASS_COLORS.get(cls_id, (255, 255, 255)) # 默认白色
            cls_name = ID_TO_CLASS.get(cls_id, "unknown")
            
            # 画矩形框
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            
            # 写上标签文字 (稍微往上一点避免遮挡框)
            text_size = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(img, (xmin, ymin - text_size[1] - 5), (xmin + text_size[0], ymin), color, -1)
            cv2.putText(img, cls_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 保存可视化结果
        out_path = os.path.join(OUTPUT_DIR, f"vis_{base_name}.jpg")
        cv2.imencode('.jpg', img)[1].tofile(out_path)
        print(f"已保存可视化结果: {out_path}")

    print(f"可视化完成！所有结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_labels()
