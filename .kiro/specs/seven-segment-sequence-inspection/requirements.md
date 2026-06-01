# Requirements Document

## Introduction

本文档定义 **七段管测试序列检测系统**（Seven-Segment Test Sequence Inspector，下称
`七段管检测系统`）的第一版需求。

业务场景：工厂对七段管面板进行出厂检测时，会让面板执行一套固定的**段测试序列**——
依次点亮全部段、水平段、垂直段、逐个段扫描、底部扫描条变色等步骤。
质检工程师需要确认：

1. **灯要亮**：每个 LED 元素在该亮的时候确实亮了，颜色正确。
2. **顺序要对**：整个测试序列按照标准视频学到的关键状态顺序逐步推进，
   没有漏步、跳步、乱序、卡死。

第一版只做**离线**处理：输入一条标准视频学习状态模型，再输入待检视频进行校验。

### 视频实际内容（基于分析 D:\jiemi\七段管\Video_20260515112415516.avi）

- 分辨率 2448×2048，60fps，1220 帧（约 20 秒）
- 面板包含**异构 LED 元素**：
  - 3 位七段管（每位 7 段，共 21 段）
  - 左侧阶梯图案（7 条递减宽度的水平条）
  - 散布的点状 LED（约 20-30 个固定位置小点）
  - 底部全宽扫描条（宽约 970px）
- 测试序列：全黑 → 全亮(CYAN) → 仅水平段 → 仅垂直段 → 逐元素扫描 → 扫描条RED → 扫描条GREEN → 扫描条CYAN → 全黑
- 颜色：CYAN(H≈97-110)、GREEN(H≈73-79)、RED(H≈5-10)

### 与跑马灯原型的关系

本系统与 `docs/running_led_state_machine_prototype.md` 思路相同（标准视频学习关键状态 →
状态机校验），但**完全独立实现**，不共享代码、不共享数据结构。
关键区别：本系统保留颜色信息（不仅是亮灭），且面板元素异构（段+条+点）。

## Glossary

- **七段管检测系统 (Seven_Segment_Inspector)**：本规格定义的整体系统。
- **面板 (Panel)**：视频中包含所有 LED 元素的区域。
- **LED 元素 (LED_Element)**：面板上一个独立的发光单元，可以是七段管的一段、一个点状 LED、
  阶梯条的一条、或扫描条。系统从标准视频自动发现所有元素。
- **元素位置 (Element_Position)**：一个 LED 元素在面板上的 ROI（矩形区域）。
- **元素状态 (Element_State)**：一个 LED 元素在某帧的状态，为 `(on/off, color_code)` 二元组。
- **颜色码 (Color_Code)**：离散颜色标识，从标准视频学习得到。至少包含 `OFF`、`CYAN`、`GREEN`、`RED`。
- **帧状态 (Frame_State)**：一帧内所有 LED 元素的 `元素状态` 按固定顺序拼接而成的向量。
- **关键状态 (Key_State)**：从标准视频学习并去重得到的、稳定出现过的 `帧状态`，编号 `K000`、`K001`...
- **关键状态序列 (Key_State_Sequence)**：关键状态按出现顺序排列的序列。
- **状态模型 (State_Model)**：元素位置、颜色调色板、关键状态序列与阈值的持久化文件。
- **状态轨迹 (State_Trace)**：待检视频每帧的状态机判断结果。
- **异常事件 (Anomaly_Event)**：状态机判定的偏差。
- **标准视频 (Standard_Video)**：用于学习的参考视频。
- **待检视频 (Test_Video)**：需要被校验的目标视频。
- **采样帧率 (Sample_FPS)**：从视频抽帧的频率。
- **段相似度 (State_Similarity)**：两个 `帧状态` 之间的相似度。
- **复合相似度 (Composite_Similarity)**：同时考虑亮灭和颜色的相似度。

## Requirements

### Requirement 1: LED 元素自动发现与标定

**User Story:** 作为质检工程师，我希望系统能从标准视频自动发现面板上所有 LED 元素的位置，
以便后续每帧都能用同一套元素位置解码状态。

#### Acceptance Criteria

1. WHEN 标准视频被传入学习流程，THE 系统 SHALL 自动检测面板区域 ROI。
2. WHEN 面板 ROI 被确定，THE 系统 SHALL 从标准视频的"全亮"帧中自动发现所有 LED 元素位置，
   包括七段管的各段、阶梯条、点状 LED、扫描条。
3. THE 系统 SHALL 为每个发现的 LED 元素分配一个唯一 ID 并记录其 ROI 坐标。
4. THE 系统 SHALL 输出发现的元素总数，并将元素列表写入状态模型。
5. IF 标准视频中未检测到"全亮"帧（即没有一帧的亮元素数量超过阈值），
   THEN THE 系统 SHALL 终止学习流程并报错。
6. WHERE 用户通过命令行指定了固定面板尺寸，THE 系统 SHALL 使用指定值跳过自动面板检测。
7. THE 系统 SHALL 在 debug 输出中保存元素发现的可视化图（在全亮帧上标注所有元素 ROI）。

### Requirement 2: 颜色调色板学习

**User Story:** 作为质检工程师，我希望系统从标准视频自动学习颜色调色板，
以便把每个元素的颜色映射为离散颜色码。

#### Acceptance Criteria

1. WHEN 元素发现完成，THE 系统 SHALL 在所有被判定为亮的元素 ROI 中采样色彩数据。
2. THE 系统 SHALL 在 HSV 色彩空间中对样本聚类，输出一个有限的颜色调色板（不超过 8 色）。
3. THE 调色板 SHALL 始终包含 `OFF` 码代表灭态。
4. THE 系统 SHALL 为每个彩色码记录 HSV 中心值、容差半径与样本计数。
5. WHERE 用户通过命令行提供了显式调色板配置，THE 系统 SHALL 使用该配置跳过自动聚类。
6. IF 自动聚类得到的彩色码数量为 0，THEN THE 系统 SHALL 终止并报错。

### Requirement 3: 单帧状态提取

**User Story:** 作为质检工程师，我希望系统对任意一帧提取所有元素的亮灭和颜色状态，
以便同时校验"灯要亮"和"颜色对"。

#### Acceptance Criteria

1. WHEN 一帧画面与已加载的状态模型一同被传入，THE 系统 SHALL 为每个 LED 元素输出
   `(on/off, color_code)` 二元组。
2. WHEN 元素被判定为 off，THE 系统 SHALL 将其 color_code 设为 `OFF`。
3. WHEN 元素被判定为 on，THE 系统 SHALL 将其 color_code 设为调色板中距离最近的彩色码。
4. THE 系统 SHALL 将所有元素的状态按固定顺序拼接为一个 `帧状态` 向量。
5. THE 系统 SHALL 对每帧返回整体置信度（成功解码的元素比例）。
6. IF 某元素 ROI 的有效像素不足，THEN THE 系统 SHALL 将该元素标记为 `unknown`。

### Requirement 4: 关键状态序列学习

**User Story:** 作为质检工程师，我希望系统从标准视频提取稳定的关键状态序列，
作为后续顺序校验的参考。

#### Acceptance Criteria

1. WHEN 标准视频所有采样帧的帧状态都已生成，THE 系统 SHALL 按时间顺序遍历。
2. WHEN 当前帧状态与最近关键状态的复合相似度 ≥ `dedupe_similarity`，
   THE 系统 SHALL 视为同一关键状态。
3. WHEN 复合相似度 < `dedupe_similarity`，THE 系统 SHALL 将其作为新候选关键状态。
4. WHILE 候选关键状态未连续稳定出现 `min_stable_frames` 帧，
   THE 系统 SHALL 不将其加入关键状态序列。
5. WHEN 候选稳定帧数达标，THE 系统 SHALL 分配 ID（格式 `K{index:03d}`）并追加到序列。
6. THE 系统 SHALL 为每个关键状态记录中位停留帧数和最大停留帧数。
7. IF 学习结束时关键状态数量 < 2，THEN THE 系统 SHALL 终止并报错。
8. THE 系统 SHALL 将完整状态模型写入 `seven_segment_model.json`。

### Requirement 5: 状态机顺序校验

**User Story:** 作为质检工程师，我希望系统用状态机对待检视频逐帧推进，
在异常时给出明确事件。

#### Acceptance Criteria

1. WHEN 待检视频与状态模型一同被传入，THE 系统 SHALL 按 Sample_FPS 抽样。
2. WHEN 校验开始，THE 系统 SHALL 将状态机指针置为 K000，hold_frames 和 unknown_frames 置 0。
3. WHEN 当前帧与当前关键状态的复合相似度 ≥ `state_similarity`，
   THE 系统 SHALL 标记为 `stay`，hold_frames +1。
4. WHEN 当前帧与下一关键状态的复合相似度 ≥ `state_similarity` 且大于与当前状态的相似度，
   THE 系统 SHALL 推进指针，标记为 `advance`，重置计数器。
5. WHILE 当前帧既不匹配当前也不匹配下一状态，且 unknown_frames < transition_tolerance_frames，
   THE 系统 SHALL 标记为 `unknown_transition`，unknown_frames +1。
6. THE 系统 SHALL 为每个采样帧在 state_trace.csv 中追加一行。
7. WHEN 待检视频处理完毕，THE 系统 SHALL 输出 events.json 和 summary.json。
8. summary.json 中 overall_ok 在且仅在无异常事件时为 true。

### Requirement 6: 异常事件类型

**User Story:** 作为质检工程师，我希望系统输出明确分类的异常事件，以便定位问题。

#### Acceptance Criteria

1. WHEN 某元素在应亮时未亮（与关键状态对比），THE 系统 SHALL 产生 `missing_element` 事件，
   包含元素 ID 和帧号。
2. WHEN 某元素亮了但颜色与关键状态不匹配，THE 系统 SHALL 产生 `wrong_color` 事件，
   包含元素 ID、观测颜色、预期颜色。
3. WHEN 当前帧最像的关键状态在序列中位于当前指针 +2 或更后，
   THE 系统 SHALL 产生 `skipped_state` 事件。
4. WHEN 当前帧最像的关键状态在序列中位于当前指针之前且持续超过容忍帧数，
   THE 系统 SHALL 产生 `wrong_order` 事件。
5. WHEN hold_frames 超过该状态的 median_hold_frames × hold_timeout_factor，
   THE 系统 SHALL 产生 `stuck_state` 事件。
6. WHEN 校验结束时有关键状态从未被推进到，THE 系统 SHALL 产生 `missing_state` 事件。
7. WHEN unknown_frames 连续超过 max_unknown_frames，THE 系统 SHALL 产生 `unknown_state` 事件。
8. IF 整帧解码置信度低于 panel_lost_threshold，THEN THE 系统 SHALL 产生 `panel_lost` 事件。
9. 每条异常事件 SHALL 包含：frame_index、time_sec、type、expected_state、matched_state、message。

### Requirement 7: 离线产物与文件结构

**User Story:** 作为开发者，我希望产出稳定、可读、可机器消费的文件。

#### Acceptance Criteria

1. 学习流程 SHALL 输出 `seven_segment_model.json`（UTF-8 JSON）。
2. 学习流程 SHALL 在 `keyframes/` 子目录为每个关键状态输出缩略图 `K{index:03d}.png`。
3. 检测流程 SHALL 输出 `state_trace.csv`、`events.json`、`summary.json`（UTF-8）。
4. 复查渲染 SHALL 输出单文件 `review.html`（图片 base64 内嵌，无外部依赖）。
5. review.html SHALL 包含：左侧关键状态序列、右侧当前帧状态可视化、底部时间轴与异常列表、
   帧号/时间/当前状态/相似度/PASS-NG 标识。
6. review.html SHALL 支持：播放/暂停、前后帧、跳到下一异常、点击关键状态跳转。
7. 学习流程 SHALL 输出 debug/ 目录，包含 sampled_frames/、element_overlays/、palette.json。
8. 检测流程 SHALL 输出 debug/ 目录，包含 sampled_frames/、state_overlays/。

### Requirement 8: 命令行入口

**User Story:** 作为开发者，我希望以脚本化命令运行整个流程。

#### Acceptance Criteria

1. THE 系统 SHALL 提供 `scripts/learn_seven_segment_sequence.py`，
   接受 `--video`、`--output-dir`、`--sample-fps`，可选 `--fixed-panel-size`、`--palette-config`。
2. THE 系统 SHALL 提供 `scripts/detect_seven_segment_sequence.py`，
   接受 `--video`、`--model`、`--output-dir`，可选 `--sample-fps`、`--threshold-*` 覆盖参数。
3. THE 系统 SHALL 提供 `scripts/render_seven_segment_review.py`，
   接受 `--model`、`--trace`、`--events`、`--output`。
4. THE 系统 SHALL 提供 `scripts/seven_segment_core.py` 存放公共算法。
5. WHEN 必要参数缺失，脚本 SHALL 以非零退出码终止并打印用法帮助。
6. WHEN --video 指向不存在的文件，脚本 SHALL 以非零退出码终止并打印错误信息。

### Requirement 9: 阈值集合与默认值

**User Story:** 作为开发者，我希望所有阈值集中可见、可被命令行覆盖。

#### Acceptance Criteria

1. 状态模型 SHALL 包含 thresholds 对象，至少包含：
   `sample_fps`、`element_on_threshold`、`color_match_tolerance`、
   `dedupe_similarity`、`state_similarity`、`min_stable_frames`、
   `transition_tolerance_frames`、`hold_timeout_factor`、`max_unknown_frames`、
   `panel_lost_threshold`。
2. 默认值：`sample_fps=15`、`element_on_threshold=0.35`、`color_match_tolerance=0.25`、
   `dedupe_similarity=0.985`、`state_similarity=0.95`、`min_stable_frames=2`、
   `transition_tolerance_frames=3`、`hold_timeout_factor=2.5`、
   `max_unknown_frames=3`、`panel_lost_threshold=0.5`。
3. 检测流程 SHALL 默认使用模型中的阈值，命令行覆盖优先级更高。
4. 被覆盖的阈值 SHALL 记录在 summary.json 中。

### Requirement 10: 与跑马灯原型的隔离

**User Story:** 作为维护者，我希望本系统与跑马灯原型完全独立。

#### Acceptance Criteria

1. THE 系统 SHALL 不导入跑马灯原型脚本中的任何函数、类或常量。
2. THE 系统 SHALL 不读写跑马灯原型的输出目录。
3. 状态模型根字段 SHALL 包含 `schema: "seven_segment_sequence_inspection.v1"`。
4. WHEN 检测脚本被传入不兼容模型，THE 系统 SHALL 检测 schema 不匹配并报错退出。

### Requirement 11: 自检与回归

**User Story:** 作为质检工程师，我希望系统在基础场景下行为可预测。

#### Acceptance Criteria

1. WHEN 标准视频先学习再自检，summary.json SHALL 输出 overall_ok=true 且 event_count=0。
2. WHEN 待检视频是标准视频的连续片段，系统 SHALL 允许从最近关键状态开始推进。
3. WHEN 待检视频被删除一段跨越关键状态切换的帧，系统 SHALL 报告 skipped_state 或 missing_state。
4. WHEN 待检视频被对调两个关键状态片段，系统 SHALL 报告 wrong_order。
5. WHEN 待检视频某关键状态被大幅延长，系统 SHALL 报告 stuck_state。
6. WHEN 待检视频某元素被持续遮挡，系统 SHALL 报告 missing_element。
7. WHEN 待检视频扫描条颜色被替换，系统 SHALL 报告 wrong_color。

### Requirement 12: 范围边界（非目标）

**User Story:** 作为干系人，我希望第一版范围边界明确。

#### Acceptance Criteria

1. 第一版 SHALL 仅处理离线已存盘视频，不接入实时相机流。
2. 第一版 SHALL 不向 C# 客户端暴露 API 或嵌入逻辑。
3. 第一版 SHALL 不做亮度均匀性、闪烁频率等额外缺陷分类。
4. 第一版 SHALL 不训练机器学习模型，所有判定基于规则与传统图像处理。
5. 第一版 SHALL 不处理一帧中存在多块独立面板的情况。
