组件点亮质检 - C#现场正式版

目录说明：
1. LedInspectionStation.exe
   C# 前端主界面，负责大图预览、自动检测、状态显示、报警和操作交互。
2. backend\
   Python 检测后端 exe，负责海康/OpenCV 相机取流、模板匹配CV、多帧融合和结果保存。

使用方式：
1. 解压整个目录。
2. 先确认现场电脑已安装海康 MVS 驱动，且官方 MVS 软件能看到相机。
3. 双击 LedInspectionStation.exe。
4. 程序会自动拉起 backend\LedInspectionBackend.exe。
5. 点“设置”后可刷新相机、选择相机。
6. 点“开始预览”。
7. 点“自动检测”进入连续检测；按空格可单次检测；按 F12 可展开/收起设置。

当前版本特性：
1. 默认使用已验证通过的模板匹配CV重新建模版本。
2. 自动检测默认每 2 秒触发一次。
3. NG 后继续检测，不会自动停住。
4. PASS 时画面尽量干净；NG 时只标异常组件和缺失点。
5. 设置区域默认收起，避免影响检测员操作。

结果保存：
默认保存到 backend 运行目录同级下：
output\desktop_runs\YYYYMMDD_HHMMSS_csharp_station\

当前注意事项：
1. 这一包的前端是 .NET 8 Framework-dependent 版本。
2. 现场电脑如果没有 .NET 8 Desktop Runtime，需要先安装对应运行时。
3. backend 已经是独立 exe，不需要额外安装 Python。
4. 如果相机枚举不到，优先检查 MVS 驱动、USB/GigE 连接和海康官方客户端。
