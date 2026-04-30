# 组件点亮质检桌面端 MVP 计划

## 目标

做一个可在端侧电脑直接运行的图形化桌面软件，用于连接本机相机、实时显示画面、执行组件点亮质检，并支持在每个 ROI 内切换不同灯点检测方法。

这个版本的目标不是最终安装包，而是最快速可用的验证版。先把相机取流、检测交互、结果显示和算法切换跑通，再进入安装包、配置管理和产线化稳定性阶段。

## 核心判断

不再优先做网页端。相机接在端侧电脑上，网页方案会遇到浏览器权限、HTTPS、摄像头枚举、局域网访问和工业相机 SDK 调用限制。桌面端更适合当前任务。

底层采集方式不做“每次点击重新拍照”，而是软件启动相机后持续取流。用户点击检测时，程序直接复制当前最新帧进入检测流程。界面上可以叫“拍照检测”或“开始检测”，但工程实现按实时流截帧走。

## MVP 功能范围

第一版需要完成这些功能：

1. 启动桌面窗口，标题为“组件点亮质检测试”。
2. 自动枚举本机可用相机。
3. 支持选择相机，包括海康机器人相机和普通 OpenCV 摄像头。
4. 点击“开始预览”后持续显示实时视频。
5. 实时画面区域要足够大，方便观察模型框选效果。
6. 点击“检测当前帧”后，从实时流中取当前最新帧。
7. ROI 检测使用现有 ROI YOLO 模型。
8. 每个 ROI 内的灯点检测支持切换：
   - YOLO 灯点检测
   - 传统图像方法检测
9. 检测后在界面显示标注图。
10. 输出检测摘要，包括 ROI 数量、ROI 类别是否一致、每个 ROI 的灯点数量和判定。
11. 保存检测结果，包括原图、标注图和 JSON 结果。
12. 预留灯控接口，但第一版可以先用空实现或 mock 实现。

## 检测规则

当前业务先按用户确认过的规则实现：

1. 一张图里必须有 20 个 ROI。
2. 20 个 ROI 的类别必须全部一致。
3. ROI 类别决定灯点模板。
4. 每个 ROI 使用对应类别的期望灯点数量做判断。

当前期望数量：

| ROI 类型 | 期望灯点数量 |
| --- | ---: |
| ledgroup | 50 |
| ledgroup2 | 26 |
| ledgroup3 | 24 |

第一版先使用数量判断。后续再升级为模板 slot 判断，也就是不仅判断数量，还判断每个固定位置是否点亮。

## 推荐流程

```text
启动软件
  -> 枚举相机
  -> 选择相机
  -> 开始预览
  -> 相机线程持续取流并缓存 latest_frame
  -> 用户点击检测当前帧
  -> 复制 latest_frame
  -> ROI YOLO 检测
  -> 校验 ROI 数量是否为 20
  -> 校验 ROI 类型是否全部一致
  -> 按 ROI 类型选择灯点模板
  -> 对每个 ROI 执行灯点检测
       -> 方法 A: YOLO
       -> 方法 B: 传统图像方法
  -> 汇总结果
  -> 显示标注图和检测摘要
  -> 保存原图、标注图、JSON
```

## 界面布局

第一版界面不追求华丽，追求测试效率。

建议布局：

```text
┌──────────────────────────────────────────────────────────┐
│ 组件点亮质检测试                                         │
├──────────────────────────────────────────────────────────┤
│ 相机: [下拉选择] [刷新] [开始预览] [停止预览]             │
│ ROI模型: [路径输入]                                      │
│ 灯点模型: [路径输入]                                     │
│ 灯点方法: [YOLO / 传统方法] [检测当前帧] [保存目录]       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                    实时视频 / 检测结果图                 │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ 检测摘要 / 日志                                          │
│ ROI数量、ROI类型、每个ROI灯点数量、PASS/FAIL             │
└──────────────────────────────────────────────────────────┘
```

## 模块设计

第一阶段可以先做单文件 MVP，验证通过后再拆模块。

建议后续拆分为：

```text
desktop_app.py
app_client/
  main.py
  ui/
    main_window.py
  devices/
    camera_base.py
    hik_camera.py
    opencv_camera.py
    light_controller.py
  inspection/
    roi_detector.py
    point_detector_yolo.py
    point_detector_traditional.py
    inspection_runner.py
  config/
    default_config.json
```

## 相机方案

海康机器人相机走 MVS SDK。

基本策略：

1. 启动时枚举 `MV_GIGE_DEVICE | MV_USB_DEVICE`。
2. 打开相机后关闭触发模式，设置为连续采集。
3. 后台线程调用 `MV_CC_GetImageBuffer` 持续取帧。
4. 使用 `MV_CC_ConvertPixelTypeEx` 转为 BGR8。
5. 每次取到新帧后更新 `latest_frame`。
6. UI 显示只读 `latest_frame` 的副本。
7. 检测线程复制 `latest_frame`，避免检测过程中相机线程改动数据。

普通 USB 摄像头作为兜底方案，使用 `cv2.VideoCapture(index)`。

## ROI 检测

ROI 检测固定使用现有三分类模型：

```text
D:\Desktop2\Images\led_model_v2\weights\best.pt
```

模型类别：

```text
ledgroup
ledgroup2
ledgroup3
```

第一版默认参数：

```text
conf = 0.2
iou = 0.45
max_det = 50
device = cpu
```

ROI 检测后要按位置排序，建议规则为从上到下、同一行从左到右。

## ROI 内灯点检测：YOLO 方法

YOLO 灯点检测对每个 ROI crop 单独运行。

可选模型优先级：

```text
D:\led-visual-inspection\models\led_panel_yolo11m\weights\best.pt
D:\led-visual-inspection\models\led_panel\weights\best.pt
```

第一版只统计检测框数量，并把检测框映射回全图显示。

注意点：

1. YOLO 方法适合作为对照基线。
2. 如果 ROI crop 很小，需要确认模型训练尺度和推理尺度。
3. 需要在 JSON 中记录每个 ROI 的 YOLO 检测框、置信度、类别。

## ROI 内灯点检测：传统方法

传统方法先复用当前过曝红光亮点实验方案。

核心思路：

1. ROI crop 等比例缩放。
2. 使用 `white_core = min(R, G, B)` 抑制红色过曝背景，只保留偏白亮芯。
3. 对 `white_core` 做 CLAHE 增强。
4. 使用 DoG 提取小亮点响应。
5. 用阈值和白芯门限筛候选点。
6. 用中心/环形对比过滤假点。
7. 用有效区域过滤边缘假点。
8. 用行聚类和横向邻居约束过滤孤立假点。
9. 输出点坐标和数量。

第一版先使用自由找点。后续更稳的方向是模板 slot 判断。

## 传统方法的已知风险

自由找点在以下情况容易不稳定：

1. 红光严重过曝时，一个灯点可能裂成多个亮斑。
2. ROI 边缘、印刷字符、划痕可能被误识别为点。
3. 拍摄角度导致灯点行不是水平线时，行聚类过滤可能误删。
4. 亮度差异大时，弱亮点可能被阈值漏掉。
5. 不同 ROI 类型的点阵结构不同，自由找点无法利用先验位置。

所以第一版允许切换 YOLO/传统方法，方便对比；生产版本建议升级为模板 slot 方法。

## 结果保存

每次检测保存到新的时间戳目录：

```text
output/desktop_runs/YYYYMMDD_HHMMSS/
  original.jpg
  result.jpg
  result.json
```

JSON 建议字段：

```json
{
  "timestamp": "...",
  "camera": "...",
  "roi_model": "...",
  "point_method": "yolo 或 traditional",
  "roi_count": 20,
  "roi_count_ok": true,
  "roi_class_consistent": true,
  "roi_class": "ledgroup",
  "expected_points_per_roi": 50,
  "overall_ok": true,
  "rois": []
}
```

每个 ROI 记录：

```json
{
  "index": 0,
  "class_name": "ledgroup",
  "confidence": 0.91,
  "bbox": [x1, y1, x2, y2],
  "point_count": 50,
  "expected_count": 50,
  "ok": true,
  "points": [],
  "detections": []
}
```

## 灯控接口预留

第一版先不接真实灯控，但要预留接口，后面接 PLC、串口、网口或 IO 板时不用重构检测流程。

建议接口：

```python
class LightController:
    def connect(self): ...
    def disconnect(self): ...
    def set_scene(self, scene_name: str): ...
    def set_channel(self, channel: int, enabled: bool): ...
    def turn_on(self): ...
    def turn_off(self): ...
    def wait_stable(self, milliseconds: int = 200): ...
    def get_status(self) -> dict: ...
```

检测流程里预留：

```text
检测前 -> light_controller.set_scene(...)
等待稳定 -> light_controller.wait_stable(...)
取当前帧 -> run inspection
检测后 -> 可选切下一路灯
```

## 第一版验收标准

MVP 完成后至少满足：

1. 软件能启动。
2. 能枚举并选择相机。
3. 能看到实时画面。
4. 点击检测不会重新打开相机，而是直接取当前流帧。
5. 能跑 ROI YOLO。
6. 能显示 ROI 框和类别。
7. 能判断 ROI 是否为 20 个。
8. 能判断 ROI 类型是否全部一致。
9. 每个 ROI 内能切换 YOLO/传统方法检测灯点。
10. 能显示每个 ROI 的点数和 PASS/FAIL。
11. 能保存原图、结果图和 JSON。

## 第二阶段计划

第一版验证可用后，再做这些增强：

1. 把单文件 MVP 拆成正式模块。
2. 加配置文件，保存模型路径、阈值、相机参数、输出目录。
3. 加曝光、增益、触发模式等相机参数设置。
4. 加模板 slot 检测方法，替代自由找点作为生产方案。
5. 加灯控真实实现。
6. 加检测历史记录。
7. 加安装包构建脚本。
8. 加开机自启动和异常日志。
9. 加一键导出问题样本。

## 当前推荐执行顺序

1. 先做 `desktop_app.py` 单文件 MVP。
2. 验证海康相机实时预览。
3. 验证从流中截帧检测。
4. 验证 ROI YOLO 结果。
5. 验证 ROI 内 YOLO 灯点检测。
6. 验证 ROI 内传统方法检测。
7. 保存结果样例给人工检查。
8. 再决定是否拆模块和打安装包。
