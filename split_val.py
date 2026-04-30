import os
import shutil
import random

# ================= 配置区域 =================
# 数据集根目录 (根据你的日志修改这里)
base_dir = r"D:\led-visual-inspection\datasets"

images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

train_img_path = os.path.join(images_dir, "train")
train_lbl_path = os.path.join(labels_dir, "train")

val_img_path = os.path.join(images_dir, "val")
val_lbl_path = os.path.join(labels_dir, "val")

# 验证集比例 (0.1 代表 10%)
val_ratio = 0.1


# ===========================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")


def split_dataset():
    # 1. 获取所有训练集图片文件名 (假设后缀是 jpg, 根据实际情况修改)
    # 支持常见图片格式
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [f for f in os.listdir(train_img_path) if f.lower().endswith(img_extensions)]

    if not all_images:
        print("错误：训练集图片目录为空或未找到图片！")
        return

    print(f"发现训练集图片数量: {len(all_images)}")

    # 2. 随机打乱
    random.shuffle(all_images)

    # 3. 计算划分数量
    num_val = int(len(all_images) * val_ratio)
    if num_val == 0 and len(all_images) > 0:
        num_val = 1  # 至少保证有一张验证集，防止报错
        print("警告：图片太少，强制设置 1 张为验证集。")

    val_images = all_images[:num_val]
    # 剩下的留作训练集 (其实不用动，本来就在那里，但逻辑上是剩下的)

    print(f"即将移动 {num_val} 张图片及其标签到验证集...")

    # 4. 创建验证集目录
    ensure_dir(val_img_path)
    ensure_dir(val_lbl_path)

    # 5. 移动文件
    moved_count = 0
    for img_name in val_images:
        # 构造源路径
        src_img = os.path.join(train_img_path, img_name)

        # 构造对应的标签文件名 (替换后缀为 .txt)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        src_lbl = os.path.join(train_lbl_path, lbl_name)

        # 构造目标路径
        dst_img = os.path.join(val_img_path, img_name)
        dst_lbl = os.path.join(val_lbl_path, lbl_name)

        # 移动图片
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        else:
            print(f"警告：找不到图片 {src_img}")
            continue

        # 移动标签 (如果标签存在)
        if os.path.exists(src_lbl):
            shutil.move(src_lbl, dst_lbl)
        else:
            # 如果没有对应的标签文件，YOLO通常会忽略或报错，视配置而定
            # 这里我们只移动图片，或者你也可以选择删除这张图片以保持对应
            print(f"提示：图片 {img_name} 没有对应的标签文件 {lbl_name}，仅移动了图片。")

        moved_count += 1

    print(f"完成！成功移动 {moved_count} 组数据到验证集。")
    print(f"现在训练集剩余: {len(all_images) - moved_count} 张")
    print(f"验证集现有: {moved_count} 张")


if __name__ == "__main__":
    split_dataset()