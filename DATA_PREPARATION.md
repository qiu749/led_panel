# 数据集准备指南

## 📁 目录结构

```
datasets/
├── data.yaml          # 数据集配置文件
├── images/
│   ├── train/         # 训练图片
│   └── val/           # 验证图片
└── labels/
    ├── train/         # 训练标注
    └── val/           # 验证标注
```

## 🔖 标注格式

YOLOv8 使用 **归一化坐标** 格式的标注文件（.txt），每个文件对应一张图片。

### 标注格式说明

每一行格式：
```
class_id x_center y_center width height
```

### 类别定义

在 `datasets/data.yaml` 中定义了两个类别：
- `0`: lit_item (发光的 LED)
- `1`: panel_corner (面板角点 - 用于 Pose 检测，可选)

### 标注示例

对于 `images/train/frame_001.jpg`，对应的标注文件是 `labels/train/frame_001.txt`：

```
0 0.35 0.25 0.15 0.15
0 0.55 0.25 0.15 0.15
0 0.75 0.25 0.15 0.15
```

这个例子标注了 3 个发光的 LED。

## 🛠️ 推荐的标注工具

### 1. LabelImg (简单易用)
```bash
pip install labelImg
labelImg
```

### 2. Roboflow (在线标注平台)
- 访问: https://roboflow.com
- 支持导出为 YOLOv8 格式

### 3. CVAT (专业标注平台)
- 访问: https://www.cvat.ai
- 功能强大，适合团队协作

## 📝 标注建议

### lit_item (发光 LED)
- 框住发光的区域，不要太大也不要太小
- 确保包含整个发光部分
- 反光部分不要标注，只标真实发光的 LED

### panel_corner (面板角点 - Pose 检测)
- 如果你想使用自定义的 Pose 模型：
  - 标注面板的四个角
  - 使用小的 Bounding Box 标注每个角
  - 按顺时针或逆时针顺序标注

## 📊 数据集规模建议

- **训练集**: 至少 100-200 张图片
- **验证集**: 20-50 张图片 (约 10-20% 的数据)
- 更多数据 = 更好的模型性能

## 🚀 训练步骤

1. 准备好数据集放入 `datasets/` 目录
2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
3. 运行训练:
   ```bash
   python scripts/train_yolo.py --epochs 100 --batch 16
   ```

## 📌 注意事项

- 图片和标注文件名必须一致（除了扩展名）
- 标注文件必须是 .txt 格式
- 坐标必须是归一化的（0-1 之间）
- 确保没有空的标注文件
