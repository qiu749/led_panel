import os
import json
import base64
import cv2
import numpy as np
import re
from openai import OpenAI

# 1. 配置参数
API_KEY = "sk-fb8bb08e273f4719821cf6f3d3d93cd8"  # 替换为你的阿里云百炼 API Key
IMAGE_DIR = r"D:\Desktop2\Images\extracted_frames"
LABEL_DIR = r"D:\Desktop2\Images\labels"
MAX_TEST_IMAGES = 10  # 限制测试处理的图片数量

# 初始化阿里云百炼的 OpenAI 兼容客户端
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 类别映射字典 (YOLO txt 需要数字 ID)
CLASS_MAP = {
    "ledgroup1": 0,
    "ledgroup2": 1,
    "ledgroup3": 2
}

# 2. 图片转 Base64 编码函数
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 3. 坐标转换函数：[xmin, ymin, xmax, ymax] -> YOLO [x_center, y_center, w, h]
def convert_to_yolo(box, img_width, img_height):
    xmin, ymin, xmax, ymax = box
    
    # 计算绝对中心点和宽高
    x_center_abs = (xmin + xmax) / 2.0
    y_center_abs = (ymin + ymax) / 2.0
    w_abs = xmax - xmin
    h_abs = ymax - ymin
    
    # 【修复边界】防止大模型输出坐标越界（比如小于 0 或超出图片宽高）
    x_center_abs = max(0, min(x_center_abs, img_width))
    y_center_abs = max(0, min(y_center_abs, img_height))
    
    # 归一化 (除以图片的真实宽高)
    x_center = x_center_abs / 1000.0  # 大模型通常在内部将图像缩放到 1000x1000 坐标系进行输出
    y_center = y_center_abs / 1000.0
    w = w_abs / 1000.0
    h = h_abs / 1000.0
    
    return f"{x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

# 4. 主流程
def process_images():
    if not os.path.exists(LABEL_DIR):
        os.makedirs(LABEL_DIR)

    processed_count = 0  # 记录已处理图片数

    for filename in os.listdir(IMAGE_DIR):
        # 达到限制数量时退出循环
        if processed_count >= MAX_TEST_IMAGES:
            print(f"已达到测试限制数量 ({MAX_TEST_IMAGES} 张)，停止处理。")
            break

        if not filename.endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(IMAGE_DIR, filename)
        label_path = os.path.join(LABEL_DIR, filename.rsplit('.', 1)[0] + '.txt')
        
        # 如果已经存在标注文件并且文件大小不为 0，才跳过（方便中断后继续，并避免空文件导致的跳过）
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            continue

        processed_count += 1
        print(f"[{processed_count}/{MAX_TEST_IMAGES}] 正在标注: {filename} ...")
        
        # 解决 cv2.imread 读取中文路径失败的问题
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        if img is None:
            print(f"无法读取图片: {image_path}")
            continue
        img_height, img_width = img.shape[:2]
        
        base64_image = encode_image(image_path)
        
        # 组装动态 Prompt
        system_prompt = f"""
你是一个专业的工业机器视觉数据标注员。 
你的任务是分析这张由极其密集的 LED 小灯泡组成的点阵屏图片。 
这块点阵屏由上下两排构成，每一排有 10 个独立的发光矩阵组，因此图片中必定有且仅有 20 个发光矩阵组。
图像的尺寸是宽 {img_width} 像素，高 {img_height} 像素。 

你的核心任务是：
1. 找出这 20 个独立发光的 LED 矩阵组，为这 20 个矩阵组【每一个都单独画一个大框】。
   - 警告：你的输出结果中必须恰好包含 20 个框！不能多也不能少。
   - 警告：不要框单颗微小的灯泡！你要框的是这 20 个大的“长条形发光矩阵”（每组是一个整体）。框的边界必须紧贴每个矩阵组的外围亮灯边缘。
2. 仔细观察每个矩阵组内部 LED 灯泡的亮灭排布图案（花纹），将这 20 个框分类为以下三种之一：
   - "ledgroup1"：该矩阵组内部的所有 LED 灯泡【全部密集亮起】，形成一个饱满的实心发光矩形（满亮模式）。
   - "ledgroup2"：该矩阵组内部的 LED 灯泡呈【竖向交替】亮起（一列亮、一列灭），内部形成由多根竖线组成的竖条纹图案。
   - "ledgroup3"：该矩阵组内部的 LED 灯泡呈【横向交替】亮起（一行亮、一行灭），内部形成由多根横线组成的横条纹图案。

请严格以 JSON 数组格式输出结果，包含恰好 20 个对象，不要包含任何其他说明文字。每个对象包含 class 和 box（box 必须使用模型原生输出的 [0, 1000] 缩放坐标系 [xmin, ymin, xmax, ymax]）： 
[ 
  {{"class": "ledgroup1", "box": [100, 150, 150, 400]}}, 
  {{"class": "ledgroup2", "box": [200, 150, 250, 400]}}
]
        """

        try:
            # 请求阿里云百炼大模型
            # 严重警告：千万不能用 "qwen3.5-plus" 等纯文本模型，它们看不见图片，只会瞎猜坐标！
            # 必须使用视觉大模型，强烈推荐 "qwen-vl-max-latest" (它基于 Qwen2.5-VL，目标检测能力极强)
            response = client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "请标注这张图片。只返回JSON，不要输出任何其他内容。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ]
            )
            
            # 解析结果
            result_text = response.choices[0].message.content.strip()
            
            # 兼容大模型有时会在最外层包一个 dict 的情况，或包含 markdown 标记
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
                
            result_text = result_text.strip()
            # 修复大模型偶尔把列表提前闭合的语法错误，例如 `}], {"class":...` 替换为 `}, {"class":...`
            result_text = result_text.replace("}], {", "}, {")
            result_text = result_text.replace("}]\n,", "},\n")
            # 修复偶尔漏掉坐标数组右括号的错误，例如 `"box": [1,2,3,4}` 替换为 `"box": [1,2,3,4]}`
            result_text = re.sub(r'(\d)\s*\}', r'\1]}', result_text)
            
            try:
                parsed_json = json.loads(result_text)
            except json.JSONDecodeError as e:
                print(f"JSON 解析失败，原始输出: {result_text}")
                raise e
            # 检查并处理字典格式
            if isinstance(parsed_json, dict):
                annotations = list(parsed_json.values())[0]
            else:
                annotations = parsed_json

            # 写入 YOLO txt 文件
            with open(label_path, "w") as f:
                for ann in annotations:
                    cls_name = ann["class"]
                    box = ann["box"]
                    
                    if cls_name in CLASS_MAP:
                        cls_id = CLASS_MAP[cls_name]
                        yolo_coords = convert_to_yolo(box, img_width, img_height)
                        f.write(f"{cls_id} {yolo_coords}\n")
                        
        except Exception as e:
            print(f"标注 {filename} 失败: {e}")

if __name__ == "__main__":
    process_images()
    print("自动化标注完成！")