import os
import json
import base64
import random
import shutil
import dashscope
from dashscope import MultiModalConversation
import cv2  # 仅用于读取图片尺寸和复制文件

# ================= 1. 配置区域 =================
dashscope.api_key = "sk-fb8bb08e273f4719821cf6f3d3d93cd8"  # <--- 务必替换为你的 Key

SOURCE_IMAGE_FOLDER = "./frames"
OUTPUT_DATASET_ROOT = "./dataset_qwen_grounding"
VAL_SPLIT_RATIO = 0.2

# ⚠️ 关键：必须使用支持高精度定位的模型，推荐 qwen-vl-max 或 qwen2.5-vl-72b-instruct
# 如果配额不足，可以尝试 qwen-vl-plus，但 max 效果最好
MODEL_NAME = 'qwen-vl-max'

print(f"🚀 启动：Qwen-VL 直接定位方案 (模型：{MODEL_NAME})")
print("💡 策略：完全依赖大模型识别发光区域，放弃传统算法。")


# ===========================================

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_lit_boxes_from_llm(image_path):
    """
    调用大模型直接返回发光区域的坐标。
    返回格式：[[x1, y1, x2, y2], ...] (像素坐标)
    """
    img_base64 = encode_image(image_path)

    # 获取图片原始尺寸，用于后续校验（可选，大模型通常返回归一化或相对坐标，需确认）
    # Qwen-VL 通常返回 [y_min, x_min, y_max, x_max] 且是基于 1000x1000 的归一化值，或者是绝对像素？
    # 文档说明：qwen-vl-max 返回的 box_2d 通常是 [y_min, x_min, y_max, x_max]，数值范围是 0-1000 (归一化)

    prompt_text = (
        "任务：请仔细观察这张空调面板图片。\n"
        "目标：找出所有**正在发光的 LED 指示灯/数字**。\n"
        "难点提示：\n"
        "1. 图片表面有塑料膜，会有强烈的白色反光，请**务必忽略**这些反光区域。\n"
        "2. 未发光的灰色印刷文字（如'模式'、'风速'等圆圈内的字）请**忽略**。\n"
        "3. 只关注那些发出微弱彩色光（如淡紫、淡蓝、红色）的小区域。\n"
        "4. 即使光线很暗，只要它是亮着的 LED，就要框出来。\n"
        "\n"
        "输出要求：\n"
        "请直接返回一个 JSON 列表，每个元素包含 'box_2d' 和 'label'。\n"
        "box_2d 格式为 [ymin, xmin, ymax, xmax]，数值范围 0-1000 (归一化坐标)。\n"
        "示例：[{\"box_2d\": [100, 200, 150, 250], \"label\": \"lit\"}, ...]\n"
        "如果没有发现任何亮灯，返回空列表 []。"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{img_base64}"},
                {"text": prompt_text}
            ]
        }
    ]

    try:
        response = MultiModalConversation.call(model=MODEL_NAME, messages=messages)

        if response.status_code == 200:
            content = response.output.choices[0].message.content
            text_response = content[0]['text'].strip()

            # 清理 Markdown
            if '```json' in text_response:
                text_response = text_response.split('```json')[1].split('```')[0]
            elif '```' in text_response:
                text_response = text_response.split('```')[1].split('```')[0]

            data = json.loads(text_response)

            # 解析坐标
            boxes_pixel = []
            # 我们需要知道原图尺寸来转换归一化坐标
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                return []
            h, w, _ = img_cv.shape

            for item in data:
                if 'box_2d' in item:
                    ymin, xmin, ymax, xmax = item['box_2d']
                    # Qwen 返回的是 0-1000 的归一化值
                    x1 = int((xmin / 1000.0) * w)
                    y1 = int((ymin / 1000.0) * h)
                    x2 = int((xmax / 1000.0) * w)
                    y2 = int((ymax / 1000.0) * h)

                    # 简单的合法性检查
                    if x2 > x1 and y2 > y1:
                        boxes_pixel.append([x1, y1, x2, y2])

            return boxes_pixel
        else:
            print(f"   ❌ API 错误：{response.message}")
            return []

    except Exception as e:
        print(f"   ❌ 异常：{e}")
        # 打印部分响应以便调试
        # print(response)
        return []


def boxes_to_yolo(boxes, img_w, img_h):
    lines = []
    for x1, y1, x2, y2 in boxes:
        xc = (x1 + x2) / 2.0 / img_w
        yc = (y1 + y2) / 2.0 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        if 0 <= xc <= 1 and 0 <= yc <= 1:
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def setup_dataset(root):
    dirs = [
        os.path.join(root, "images", "train"),
        os.path.join(root, "images", "val"),
        os.path.join(root, "labels", "train"),
        os.path.join(root, "labels", "val")
    ]
    for d in dirs: os.makedirs(d, exist_ok=True)

    with open(os.path.join(root, "data.yaml"), 'w') as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: 'lit'\n")


# ================= 主流程 =================

if __name__ == "__main__":
    setup_dataset(OUTPUT_DATASET_ROOT)

    files = sorted([f for f in os.listdir(SOURCE_IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not files:
        print("❌ 没找到图片")
        exit()

    random.shuffle(files)
    split = int(len(files) * (1 - VAL_SPLIT_RATIO))
    train_files, val_files = files[:split], files[split:]

    print(f"📊 训练集：{len(train_files)}, 验证集：{len(val_files)}")


    def process(files, subset):
        for i, fname in enumerate(files):
            src = os.path.join(SOURCE_IMAGE_FOLDER, fname)
            print(f"[{subset}] ({i + 1}/{len(files)}) 处理：{fname}")

            # 1. 直接问大模型要坐标
            boxes = get_lit_boxes_from_llm(src)

            # 2. 保存
            dst_img = os.path.join(OUTPUT_DATASET_ROOT, "images", subset, fname)
            dst_lbl = os.path.join(OUTPUT_DATASET_ROOT, "labels", subset, os.path.splitext(fname)[0] + ".txt")

            shutil.copy(src, dst_img)

            img_cv = cv2.imread(src)
            h, w, _ = img_cv.shape

            yolo_lines = boxes_to_yolo(boxes, w, h)

            with open(dst_lbl, 'w') as f:
                if yolo_lines:
                    f.write('\n'.join(yolo_lines))

            count = len(yolo_lines)
            if count > 0:
                print(f"   ✅ 检测到 {count} 个亮灯区域")
            else:
                print(f"   ⚪ 未检测到亮灯 (可能是全灭或大模型漏检)")


    process(train_files, "train")
    process(val_files, "val")

    print("\n🎉 完成！请使用 YOLOv8 训练单类别模型。")