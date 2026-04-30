import os
import base64
from pathlib import Path
import yaml
from volcenginesdkarkruntime import Ark

ROI_DIR = r"D:\Desktop2\Images\roi_crops"
OUTPUT_DIR = r"d:\led-visual-inspection\roi_labeled"
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")

API_KEY = "f6c47434-48f9-49b4-bd0f-2a69353cf795"
MODEL_NAME = "doubao-seed-2-0-pro-260215"

CLASSES = ["on"]

# 初始化 Ark 客户端，设置超时和重试机制
client = Ark(
    api_key=API_KEY,
    timeout=60,
    max_retries=3
)

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def normalize_coord_token(token):
    token = token.strip().strip(",")

    try:
        value = float(token)
    except ValueError:
        return None

    if 0.0 <= value <= 1.0:
        return value

    # Model sometimes omits the leading decimal point, e.g. 716 -> 0.716.
    if token.isdigit():
        value = float(f"0.{token}")
        if 0.0 <= value <= 1.0:
            return value

    # Model sometimes prefixes an index before the decimal point, e.g.
    # 1.174 -> 0.174, 7.785 -> 0.785, 6.640 -> 0.640.
    if "." in token:
        integer_part, fractional_part = token.split(".", 1)
        if integer_part.isdigit() and fractional_part.isdigit():
            value = float(f"0.{fractional_part}")
            if 0.0 <= value <= 1.0:
                return value

    return None

def parse_label_line(line):
    parts = line.split()
    if len(parts) != 5 or parts[0] != "on":
        return None

    coords = []
    for value in parts[1:]:
        normalized = normalize_coord_token(value)
        if normalized is None:
            return None
        coords.append(normalized)

    if not all(0.0 <= value <= 1.0 for value in coords):
        return None

    return coords

def annotate_with_qwen_vl(image_path):
    prompt = (
        "任务：检测图片中**所有（约有20-50个）**亮着的LED灯点。这些灯点表现为红色背景下的高亮发白/发黄中心斑点。\n"
        "要求：你必须找出每一个灯点，绝对不能遗漏，图片中有几十个这样的点！请从上到下、从左到右仔细扫描全图。\n"
        "输出规则：\n"
        "每个亮点单独占一行，找到多少个亮点就输出多少行。\n"
        "每行必须严格输出：on x_center y_center width height\n"
        "x_center y_center width height 必须是 0 到 1 之间的小数。\n"
        "不要输出整数，不要省略小数点，不要输出逗号、中文、解释、序号或多余文字。\n"
        "如果没有亮点，只输出 off。\n"
        "示例：\n"
        "on 0.512 0.438 0.024 0.081\n"
        "on 0.534 0.438 0.024 0.081\n"
        "(以此类推，直到列出所有点)"
    )
    image_base64 = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            thinking={"type": "disabled"},
            max_tokens=2048,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"请求失败: {str(e)}")
    return ""

# ==================== 主逻辑（不变） ====================
def main():
    os.makedirs(LABELS_DIR, exist_ok=True)
    image_files = sorted(
        f for f in os.listdir(ROI_DIR) if f.lower().endswith((".jpg", ".png"))
    )
    print(f"找到 {len(image_files)} 张图")
    skipped_count = 0
    processed_count = 0

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(ROI_DIR, img_file)
        txt_path = os.path.join(LABELS_DIR, Path(img_file).stem + ".txt")
        if os.path.exists(txt_path):
            skipped_count += 1
            print(f"[{i+1}] 跳过: {img_file} -> 已存在 {Path(txt_path).name}")
            continue

        processed_count += 1
        print(f"[{i+1}] 处理: {img_file}")
        label = annotate_with_qwen_vl(img_path)
        print(f"  返回: {label}")

        count = 0
        invalid_count = 0
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in label.split("\n"):
                line = line.strip()
                if not line or line == "off":
                    continue

                coords = parse_label_line(line)
                if coords is None:
                    invalid_count += 1
                    continue

                f.write("0 " + " ".join(f"{value:.6f}" for value in coords) + "\n")
                count += 1

        if count > 0:
            print(f"  保存 {count} 个标签，丢弃 {invalid_count} 条异常结果\n")
        else:
            print(f"  无亮点，丢弃 {invalid_count} 条异常结果\n")

    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f:
        yaml.dump({"path": OUTPUT_DIR, "train": "labels", "val": "labels", "names": {0: "on"}, "nc": 1}, f)

    print(f"完成: 新处理 {processed_count} 张，跳过 {skipped_count} 张")

if __name__ == "__main__":
    main()
