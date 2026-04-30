import cv2
import numpy as np
import os
import glob

# 配置路径
IMAGE_DIR = r"D:\Desktop2\Images\extracted_frames"
LABEL_DIR = r"D:\Desktop2\Images\labels_cv"  # 存放到新的目录，避免覆盖大模型的
VIS_DIR = r"D:\Desktop2\Images\visualized_cv"

# 类别映射
CLASS_MAP = {
    "ledgroup1": 0,  # 满亮
    "ledgroup2": 1,  # 竖条纹
    "ledgroup3": 2   # 横条纹
}

# 类别颜色 (BGR)
CLASS_COLORS = {
    0: (0, 0, 255),    # 红
    1: (0, 255, 0),    # 绿
    2: (255, 0, 0)     # 蓝
}

def analyze_pattern(roi):
    """分析 ROI 区域内的亮灭模式"""
    if roi.size == 0:
        return "ledgroup1" # 默认满亮
        
    # 将 ROI 转换为灰度图并二值化，突出亮灯区域
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # 统计 X 轴和 Y 轴方向上的亮度投影
    # 如果是竖条纹，X 轴投影应该有明显的波峰波谷
    # 如果是横条纹，Y 轴投影应该有明显的波峰波谷
    
    x_projection = np.mean(thresh, axis=0) # 沿 Y 轴求均值，得到 X 轴投影
    y_projection = np.mean(thresh, axis=1) # 沿 X 轴求均值，得到 Y 轴投影
    
    # 计算波动的剧烈程度 (标准差或方差)
    x_std = np.std(x_projection)
    y_std = np.std(y_projection)
    
    # 设定一个阈值来判断波动是否足够大
    # 这个阈值可能需要根据实际图像调整
    THRESHOLD = 20
    
    if x_std > THRESHOLD and y_std < THRESHOLD:
        return "ledgroup2" # 竖条纹：X轴波动大，Y轴平稳
    elif y_std > THRESHOLD and x_std < THRESHOLD:
        return "ledgroup3" # 横条纹：Y轴波动大，X轴平稳
    elif x_std > THRESHOLD and y_std > THRESHOLD:
        # 如果都有波动，可能是一种网格状或者更复杂的模式，先看看哪个波动更大
        if x_std > y_std * 1.5:
             return "ledgroup2"
        elif y_std > x_std * 1.5:
             return "ledgroup3"
        else:
             return "ledgroup1" # 无法明确区分方向，归为满亮
    else:
        return "ledgroup1" # 都没有明显波动，说明是均匀的满亮或全灭

def convert_to_yolo(box, img_width, img_height):
    x, y, w, h = box
    x_center = (x + w / 2.0) / img_width
    y_center = (y + h / 2.0) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return f"{x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

def process_images():
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    if not image_files:
        print(f"在 {IMAGE_DIR} 中未找到图片。")
        return

    print(f"找到 {len(image_files)} 张图片，开始 OpenCV 处理...")

    for img_path in image_files[:20]: # 先测试 20 张
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 1. 读取图像
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img is None:
            continue
            
        vis_img = img.copy()
        img_height, img_width = img.shape[:2]

        # 2. 图像预处理 (提取高亮区域)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 阈值分割，提取亮起的 LED 灯珠 (因为背景很黑，灯很亮)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

        # 3. 形态学操作：膨胀
        # 目的：将同一个矩阵内部密集的细小灯珠“融合”成一个完整的大块
        # kernel 的大小决定了融合的程度。需要横向和纵向都膨胀。
        # 考虑到矩阵之间的间隙，kernel 不能太大，否则相邻的矩阵会粘连。
        kernel = np.ones((15, 15), np.uint8) 
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        
        # 可选：闭运算填补内部空洞
        dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # 4. 轮廓检测
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. 过滤并获取矩形框
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            # 过滤掉太小或太大的噪点框 (需要根据实际图片尺寸微调)
            # 假设图片是 640x630，一个矩阵大概占宽 1/12，高 1/3
            if area > 1000 and area < (img_width * img_height / 4): 
                # 检查长宽比，长条形矩阵高应该大于宽
                if h > w * 1.5:
                    boxes.append((x, y, w, h))

        print(f"[{base_name}] 找到 {len(boxes)} 个有效矩阵框")
        
        # 6. 如果不是正好 20 个，说明膨胀参数可能需要调整，这里先记录
        if len(boxes) != 20:
             print(f"  --> 警告：未找到正好 20 个框，找到了 {len(boxes)} 个。")
        
        # 7. 分析花纹并保存标签
        label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")
        with open(label_path, 'w') as f:
            for box in boxes:
                x, y, w, h = box
                
                # 提取原始图像中的 ROI 区域
                roi = img[y:y+h, x:x+w]
                
                # 分析模式
                pattern_cls = analyze_pattern(roi)
                cls_id = CLASS_MAP[pattern_cls]
                
                # 转换为 YOLO 格式并写入
                yolo_str = convert_to_yolo(box, img_width, img_height)
                f.write(f"{cls_id} {yolo_str}\n")
                
                # 在可视化图上画框
                color = CLASS_COLORS[cls_id]
                cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(vis_img, pattern_cls, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 保存可视化结果
        vis_path = os.path.join(VIS_DIR, f"vis_{base_name}.jpg")
        cv2.imencode('.jpg', vis_img)[1].tofile(vis_path)

    print(f"处理完成！标签保存在 {LABEL_DIR}，可视化结果在 {VIS_DIR}")

if __name__ == "__main__":
    process_images()