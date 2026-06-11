using System.Diagnostics;
using System.Media;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;

namespace LedInspectionStation;

public sealed class SevenSegmentStationForm : Form
{
    private const int BackendPort = 8765;
    private readonly HttpClient http = new() { BaseAddress = new Uri($"http://127.0.0.1:{BackendPort}") };
    private readonly System.Windows.Forms.Timer refreshTimer = new() { Interval = 80 };
    private readonly System.Windows.Forms.Timer statusTimer = new() { Interval = 200 };

    private ComboBox cameraCombo = null!;
    private Button startButton = null!;
    private Button quickStartButton = null!;
    private Button stopButton = null!;
    private Button resetButton = null!;
    private Button ackButton = null!;
    private Button chooseVideoButton = null!;
    private Button videoButton = null!;
    private Button openImagesButton = null!;
    private CheckBox alarmCheck = null!;
    private TextBox videoPathText = null!;
    private Label resultLabel = null!;
    private Label detailLabel = null!;
    private Label metricsLabel = null!;
    private Label backendLabel = null!;
    private TextBox logText = null!;
    private PictureBox frameBox = null!;
    private PictureBox templateBox = null!;
    private PictureBox roiBox = null!;
    private PictureBox maskBox = null!;
    private PictureBox diffBox = null!;

    private Process? backendProcess;
    private bool backendReady;
    private bool running;
    private bool videoRunning;
    private bool loadingImages;
    private int imageRefreshCounter = 0;
    private string lastResult = "WAIT";
    private string lastReportDir = "";

    public SevenSegmentStationForm()
    {
        BuildUi();
        KeyPreview = true;
        KeyDown += OnKeyDown;
        FormClosing += OnFormClosing;
        refreshTimer.Tick += async (_, _) => await RefreshImagesAsync();
        statusTimer.Tick += async (_, _) => await RefreshStatusAsync();
        _ = InitializeBackendAsync();
    }

    private void BuildUi()
    {
        Text = "七段管实时检测工位 - 试产线版";
        WindowState = FormWindowState.Maximized;
        MinimumSize = new Size(1280, 760);
        BackColor = Color.FromArgb(15, 23, 42);
        Font = new Font("Microsoft YaHei UI", 10F);

        var header = new Panel
        {
            Dock = DockStyle.Top,
            Height = 124,
            BackColor = Color.FromArgb(17, 24, 39),
            Padding = new Padding(16, 12, 16, 10)
        };

        var headerRows = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 2
        };
        headerRows.RowStyles.Add(new RowStyle(SizeType.Absolute, 52));
        headerRows.RowStyles.Add(new RowStyle(SizeType.Absolute, 52));

        var cameraControls = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false
        };
        cameraCombo = new ComboBox { Width = 410, DropDownStyle = ComboBoxStyle.DropDownList, Margin = new Padding(0, 8, 8, 0) };
        startButton = MakeButton("开始检测", 112, async (_, _) => await StartDetectionAsync(), Color.FromArgb(37, 99, 235));
        quickStartButton = MakeButton("快速开始(空格)", 142, async (_, _) => await QuickStartAsync(), Color.FromArgb(22, 163, 74));
        stopButton = MakeButton("停止", 82, async (_, _) => await StopDetectionAsync(), Color.FromArgb(71, 85, 105));
        resetButton = MakeButton("复位", 82, async (_, _) => await ResetDetectionAsync(), Color.FromArgb(71, 85, 105));
        ackButton = MakeButton("确认继续", 112, async (_, _) => await AckAsync(), Color.FromArgb(180, 83, 9));
        alarmCheck = new CheckBox
        {
            Text = "NG报警",
            Checked = true,
            Width = 90,
            Height = 42,
            Margin = new Padding(4, 8, 8, 0),
            ForeColor = Color.White,
            Appearance = Appearance.Button,
            FlatStyle = FlatStyle.Flat,
            TextAlign = ContentAlignment.MiddleCenter
        };
        alarmCheck.FlatAppearance.BorderColor = Color.FromArgb(100, 116, 139);
        videoPathText = new TextBox
        {
            Width = 720,
            Text = @"D:\jiemi\七段管\Video_20260515112415516.avi",
            Margin = new Padding(8, 10, 8, 0)
        };
        chooseVideoButton = MakeButton("选择视频", 112, (_, _) => BrowseVideoFile(), Color.FromArgb(71, 85, 105));
        videoButton = MakeButton("开始检测视频", 140, async (_, _) => await RunVideoDetectionAsync(), Color.FromArgb(37, 99, 235));
        openImagesButton = MakeButton("打开匹配图片", 128, (_, _) => OpenReportImages(), Color.FromArgb(71, 85, 105));
        openImagesButton.Enabled = false;

        cameraControls.Controls.AddRange(new Control[]
        {
            MakeCaption("相机"), cameraCombo, startButton, quickStartButton, stopButton, resetButton, ackButton,
            alarmCheck
        });

        var videoControls = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false
        };
        videoControls.Controls.AddRange(new Control[]
        {
            MakeCaption("视频路径"), videoPathText, chooseVideoButton, videoButton, openImagesButton
        });

        headerRows.Controls.Add(cameraControls, 0, 0);
        headerRows.Controls.Add(videoControls, 0, 1);
        header.Controls.Add(headerRows);

        var main = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 2,
            RowCount = 1,
            Padding = new Padding(16),
            BackColor = Color.FromArgb(15, 23, 42)
        };
        main.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 54));
        main.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 46));

        diffBox = MakePicture();
        main.Controls.Add(MakePanel("虚拟质量预览", diffBox), 0, 0);

        var four = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 2 };
        four.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        four.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
        four.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        four.RowStyles.Add(new RowStyle(SizeType.Percent, 50));
        templateBox = MakePicture();
        roiBox = MakePicture();
        maskBox = MakePicture();
        frameBox = MakePicture();
        four.Controls.Add(MakePanel("标准模板真图", templateBox), 0, 0);
        four.Controls.Add(MakePanel("当前真实 ROI", roiBox), 1, 0);
        four.Controls.Add(MakePanel("抽象亮区", maskBox), 0, 1);
        four.Controls.Add(MakePanel("真实相机画面", frameBox), 1, 1);
        main.Controls.Add(four, 1, 0);

        var footer = new Panel
        {
            Dock = DockStyle.Bottom,
            Height = 160,
            BackColor = Color.FromArgb(17, 24, 39),
            Padding = new Padding(16, 12, 16, 12)
        };
        resultLabel = new Label
        {
            Dock = DockStyle.Left,
            Width = 230,
            Text = "WAIT",
            TextAlign = ContentAlignment.MiddleCenter,
            Font = new Font("Microsoft YaHei UI", 38F, FontStyle.Bold),
            ForeColor = Color.White,
            BackColor = Color.FromArgb(71, 85, 105)
        };
        var footerRight = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 1, RowCount = 3, Padding = new Padding(18, 0, 0, 0) };
        footerRight.RowStyles.Add(new RowStyle(SizeType.Absolute, 42));
        footerRight.RowStyles.Add(new RowStyle(SizeType.Absolute, 36));
        footerRight.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        detailLabel = MakeFooterLabel("正在启动后端...");
        metricsLabel = MakeFooterLabel("覆盖率 0%  当前模板 -  相似度 0.00");
        backendLabel = MakeFooterLabel("后端: 未连接");
        logText = new TextBox
        {
            Dock = DockStyle.Fill,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            BackColor = Color.FromArgb(2, 6, 23),
            ForeColor = Color.FromArgb(203, 213, 225),
            BorderStyle = BorderStyle.FixedSingle
        };
        footerRight.Controls.Add(detailLabel, 0, 0);
        footerRight.Controls.Add(metricsLabel, 0, 1);
        footerRight.Controls.Add(logText, 0, 2);
        footer.Controls.Add(footerRight);
        footer.Controls.Add(resultLabel);

        Controls.Add(main);
        Controls.Add(footer);
        Controls.Add(header);
    }

    private static Label MakeCaption(string text) => new()
    {
        Text = text,
        Width = Math.Max(42, text.Length * 18),
        Height = 42,
        TextAlign = ContentAlignment.MiddleRight,
        ForeColor = Color.White,
        Margin = new Padding(0, 8, 5, 0)
    };

    private static Button MakeButton(string text, int width, EventHandler handler, Color color)
    {
        var button = new Button
        {
            Text = text,
            Width = width,
            Height = 42,
            Margin = new Padding(4, 8, 4, 0),
            FlatStyle = FlatStyle.Flat,
            BackColor = color,
            ForeColor = Color.White,
            Font = new Font("Microsoft YaHei UI", 10.5F, FontStyle.Bold)
        };
        button.FlatAppearance.BorderColor = Color.FromArgb(100, 116, 139);
        button.Click += handler;
        return button;
    }

    private static PictureBox MakePicture() => new()
    {
        Dock = DockStyle.Fill,
        BackColor = Color.Black,
        SizeMode = PictureBoxSizeMode.Zoom
    };

    private static Control MakePanel(string title, PictureBox picture)
    {
        var panel = new Panel { Dock = DockStyle.Fill, Padding = new Padding(8), BackColor = Color.FromArgb(30, 41, 59), Margin = new Padding(6) };
        var label = new Label
        {
            Dock = DockStyle.Top,
            Height = 32,
            Text = title,
            ForeColor = Color.White,
            Font = new Font("Microsoft YaHei UI", 12F, FontStyle.Bold)
        };
        panel.Controls.Add(picture);
        panel.Controls.Add(label);
        return panel;
    }

    private static Label MakeFooterLabel(string text) => new()
    {
        Text = text,
        Dock = DockStyle.Fill,
        TextAlign = ContentAlignment.MiddleLeft,
        ForeColor = Color.White,
        Font = new Font("Microsoft YaHei UI", 15F, FontStyle.Bold)
    };

    private async Task InitializeBackendAsync()
    {
        SetResult("WAIT", "正在启动七段管检测服务...", Color.FromArgb(71, 85, 105));
        await StartBackendProcessAsync();
        for (var i = 0; i < 60; i++)
        {
            if (await PingBackendAsync())
            {
                backendReady = true;
                await LoadCamerasAsync();
                statusTimer.Start();
                refreshTimer.Start();
                SetResult("READY", "服务已连接，选择相机后开始检测", Color.FromArgb(71, 85, 105));
                return;
            }
            await Task.Delay(300);
        }
        SetResult("ERROR", "检测服务启动失败，请查看日志或确认端口 8765 未被占用", Color.FromArgb(185, 28, 28));
    }

    private async Task StartBackendProcessAsync()
    {
        if (await PingBackendAsync())
        {
            return;
        }

        var packagedBackend = FindPackagedBackendExe();
        ProcessStartInfo startInfo;
        if (!string.IsNullOrWhiteSpace(packagedBackend))
        {
            var backendDir = Path.GetDirectoryName(packagedBackend) ?? AppContext.BaseDirectory;
            startInfo = new ProcessStartInfo
            {
                FileName = packagedBackend,
                Arguments = $"--host 127.0.0.1 --port {BackendPort}",
                WorkingDirectory = backendDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };
        }
        else
        {
            var root = FindRepoRoot();
            var backend = Path.Combine(root, "seven_segment_live_camera_server.py");
            if (!File.Exists(backend))
            {
                throw new FileNotFoundException("找不到 seven_segment_live_camera_server.py", backend);
            }
            startInfo = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{backend}\" --host 127.0.0.1 --port {BackendPort}",
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };
        }

        backendProcess = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        backendProcess.OutputDataReceived += (_, e) => AppendLog(e.Data);
        backendProcess.ErrorDataReceived += (_, e) => AppendLog(e.Data);
        backendProcess.Start();
        backendProcess.BeginOutputReadLine();
        backendProcess.BeginErrorReadLine();
    }

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "seven_segment_live_camera_server.py")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }
        return Directory.GetCurrentDirectory();
    }

    private static string? FindPackagedBackendExe()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "backend", "LedInspectionBackend.exe"),
            Path.Combine(AppContext.BaseDirectory, "LedInspectionBackend.exe")
        };
        return candidates.FirstOrDefault(File.Exists);
    }

    private async Task<bool> PingBackendAsync()
    {
        try
        {
            using var doc = await GetJsonAsync("/status");
            return doc.RootElement.TryGetProperty("sequence_count", out _);
        }
        catch
        {
            return false;
        }
    }

    private async Task LoadCamerasAsync()
    {
        using var doc = await GetJsonAsync("/cameras");
        cameraCombo.Items.Clear();
        if (doc.RootElement.TryGetProperty("cameras", out var cameras))
        {
            foreach (var camera in cameras.EnumerateArray())
            {
                var label = camera.GetProperty("label").GetString();
                if (!string.IsNullOrWhiteSpace(label))
                {
                    cameraCombo.Items.Add(label);
                }
            }
        }
        if (cameraCombo.Items.Count > 0 && cameraCombo.SelectedIndex < 0)
        {
            cameraCombo.SelectedIndex = 0;
        }
        backendLabel.Text = $"后端: 已连接  相机 {cameraCombo.Items.Count} 台";
    }

    private async Task StartDetectionAsync()
    {
        if (!backendReady)
        {
            return;
        }
        var camera = cameraCombo.SelectedItem?.ToString() ?? "";
        using var doc = await PostJsonAsync("/start", new { camera });
        running = true;
        videoRunning = false;
        lastReportDir = "";
        openImagesButton.Enabled = false;
        videoButton.Text = "开始检测视频";
        videoButton.Enabled = true;
        chooseVideoButton.Enabled = true;
        RenderStatus(doc.RootElement);
    }

    private async Task QuickStartAsync()
    {
        if (!backendReady)
        {
            return;
        }

        quickStartButton.Enabled = false;
        try
        {
            await StartDetectionAsync();
        }
        finally
        {
            quickStartButton.Enabled = true;
        }
    }

    private async Task StopDetectionAsync()
    {
        using var doc = await PostJsonAsync("/stop", new { });
        running = false;
        videoRunning = false;
        lastReportDir = "";
        openImagesButton.Enabled = false;
        videoButton.Text = "开始检测视频";
        videoButton.Enabled = true;
        chooseVideoButton.Enabled = true;
        RenderStatus(doc.RootElement);
        SetResult("READY", "检测已停止", Color.FromArgb(71, 85, 105));
    }

    private async Task ResetDetectionAsync()
    {
        using var doc = await PostJsonAsync("/reset", new { });
        RenderStatus(doc.RootElement);
    }

    private async Task AckAsync()
    {
        using var doc = await PostJsonAsync("/ack", new { });
        RenderStatus(doc.RootElement);
    }

    private async Task RunVideoDetectionAsync()
    {
        var rawVideoPath = videoPathText.Text;
        var videoPath = ResolveTypedVideoPath();
        if (string.IsNullOrWhiteSpace(videoPath))
        {
            SetResult("ERROR", $"请先输入或选择视频文件，当前输入为空或无效: [{rawVideoPath}]", Color.FromArgb(185, 28, 28));
            return;
        }
        if (!File.Exists(videoPath))
        {
            SetResult("ERROR", "视频文件不存在: " + videoPath, Color.FromArgb(185, 28, 28));
            return;
        }

        videoButton.Enabled = false;
        chooseVideoButton.Enabled = false;
        SetResult("RUNNING", "正在启动视频逐帧快速检测...", Color.FromArgb(37, 99, 235));
        try
        {
            videoPathText.Text = videoPath;
            lastReportDir = "";
            openImagesButton.Enabled = false;
            AppendLog("视频路径: " + videoPath);
            using var doc = await PostJsonAsync("/video_start", new { path = videoPath, realtime = false }, timeoutSeconds: 20);
            videoRunning = true;
            running = false;
            videoButton.Text = "视频检测中";
            RenderStatus(doc.RootElement);
        }
        catch (Exception ex)
        {
            videoRunning = false;
            SetResult("ERROR", ex.Message, Color.FromArgb(185, 28, 28));
            videoButton.Enabled = true;
            chooseVideoButton.Enabled = true;
            videoButton.Text = "开始检测视频";
        }
    }

    private void BrowseVideoFile()
    {
        using var dialog = new OpenFileDialog
        {
            Title = "选择七段管检测视频",
            Filter = "视频文件|*.avi;*.mp4;*.mov;*.mkv|所有文件|*.*",
            CheckFileExists = true,
            Multiselect = false
        };
        var path = ResolveTypedVideoPath();
        if (!string.IsNullOrWhiteSpace(path))
        {
            var dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(dir) && Directory.Exists(dir))
            {
                dialog.InitialDirectory = dir;
            }
        }
        if (dialog.ShowDialog(this) == DialogResult.OK)
        {
            videoPathText.Text = dialog.FileName;
            SetResult("READY", "已选择视频，点击“开始检测视频”", Color.FromArgb(71, 85, 105));
        }
    }

    private string ResolveTypedVideoPath()
    {
        var path = videoPathText.Text.Trim().Trim('"', '\'', '“', '”');
        if (string.IsNullOrWhiteSpace(path))
        {
            return "";
        }
        if (Uri.TryCreate(path, UriKind.Absolute, out var uri) && uri.IsFile)
        {
            path = uri.LocalPath;
        }
        path = Environment.ExpandEnvironmentVariables(path);
        try
        {
            return Path.GetFullPath(path);
        }
        catch
        {
            return path;
        }
    }

    private void OpenReportImages()
    {
        if (string.IsNullOrWhiteSpace(lastReportDir))
        {
            SetResult("ERROR", "还没有可打开的匹配图片，请先完成一次视频检测", Color.FromArgb(185, 28, 28));
            return;
        }

        var imageDir = Path.Combine(lastReportDir, "template_debug_images");
        var targetDir = Directory.Exists(imageDir) ? imageDir : lastReportDir;
        if (!Directory.Exists(targetDir))
        {
            SetResult("ERROR", "报告目录不存在: " + targetDir, Color.FromArgb(185, 28, 28));
            openImagesButton.Enabled = false;
            return;
        }

        Process.Start(new ProcessStartInfo
        {
            FileName = targetDir,
            UseShellExecute = true
        });
    }

    private async Task RefreshStatusAsync()
    {
        if (!backendReady)
        {
            return;
        }
        try
        {
            using var doc = await GetJsonAsync("/status");
            RenderStatus(doc.RootElement);
        }
        catch (Exception ex)
        {
            backendLabel.Text = "后端: " + ex.Message;
        }
    }

    private void RenderStatus(JsonElement root)
    {
        var result = root.TryGetProperty("result", out var resultElement) ? resultElement.GetString() ?? "WAIT" : "WAIT";
        var message = root.TryGetProperty("message", out var msgElement) ? msgElement.GetString() ?? "" : "";
        var template = root.TryGetProperty("template_id", out var tmplElement) ? tmplElement.GetString() ?? "-" : "-";
        var sim = root.TryGetProperty("similarity", out var simElement) ? simElement.GetDouble() : 0.0;
        var coverage = root.TryGetProperty("coverage", out var covElement) ? covElement.GetDouble() : 0.0;
        var hits = root.TryGetProperty("ordered_hits", out var hitElement) ? hitElement.GetInt32() : 0;
        var sequenceCount = root.TryGetProperty("sequence_count", out var seqElement) ? seqElement.GetInt32() : 0;
        var unknown = root.TryGetProperty("unknown_segments", out var unkElement) ? unkElement.GetInt32() : 0;
        var captureFps = root.TryGetProperty("capture_fps", out var capElement) ? capElement.GetDouble() : 0.0;
        var detectFps = root.TryGetProperty("detect_fps", out var detElement) ? detElement.GetDouble() : 0.0;
        var camera = root.TryGetProperty("camera", out var camElement) ? camElement.GetString() ?? "" : "";
        var frozen = root.TryGetProperty("frozen", out var frozenElement) && frozenElement.GetBoolean();
        var videoActive = root.TryGetProperty("video_active", out var videoActiveElement) && videoActiveElement.GetBoolean();
        var reportDir = root.TryGetProperty("report_dir", out var reportElement) ? reportElement.GetString() ?? "" : "";
        if (!string.IsNullOrWhiteSpace(reportDir))
        {
            lastReportDir = reportDir;
            openImagesButton.Enabled = Directory.Exists(lastReportDir) || Directory.Exists(Path.Combine(lastReportDir, "template_debug_images"));
        }
        var qualityText = "质量未检";
        if (root.TryGetProperty("quality", out var quality) && quality.ValueKind == JsonValueKind.Object)
        {
            var qualityStatus = root.TryGetProperty("quality_status", out var qs) ? qs.GetString() ?? "unchecked" : "unchecked";
            var missingRatio = quality.TryGetProperty("missing_ratio", out var mr) ? mr.GetDouble() : 0.0;
            var extraRatio = quality.TryGetProperty("extra_ratio", out var er) ? er.GetDouble() : 0.0;
            var maxMissing = quality.TryGetProperty("max_missing_blob_area", out var mm) ? mm.GetInt32() : 0;
            qualityText = $"质量 {qualityStatus} 缺失 {missingRatio:P1} 最大缺口 {maxMissing}px 多亮 {extraRatio:P1}";
        }

        metricsLabel.Text = $"覆盖率 {coverage:P1}  命中 {hits}/{sequenceCount}  模板 {template}  相似度 {sim:F2}  未知 {unknown}  {qualityText}  {captureFps:F1}/{detectFps:F1}fps";
        backendLabel.Text = $"后端: {(running || videoActive ? "检测中" : "就绪")}  {camera}";
        if (videoRunning && !videoActive)
        {
            videoRunning = false;
            videoButton.Text = "开始检测视频";
            videoButton.Enabled = true;
            chooseVideoButton.Enabled = true;
        }
        if (!string.IsNullOrWhiteSpace(message) || result != lastResult)
        {
            var color = result switch
            {
                "PASS" => Color.FromArgb(22, 163, 74),
                "REVIEW" => Color.FromArgb(220, 38, 38),
                "NG" => Color.FromArgb(185, 28, 28),
                "ERROR" => Color.FromArgb(185, 28, 28),
                "RUNNING" => Color.FromArgb(37, 99, 235),
                _ => Color.FromArgb(71, 85, 105)
            };
            SetResult(result, string.IsNullOrWhiteSpace(message) ? result : message, color);
        }
        var alarmResult = result is "NG" or "REVIEW" or "ERROR";
        if (frozen && alarmResult && alarmCheck.Checked && lastResult != result)
        {
            _ = Task.Run(PlayAlarm);
        }
        lastResult = result;

        if (root.TryGetProperty("logs", out var logs))
        {
            logText.Lines = logs.EnumerateArray().Select(item => item.GetString() ?? "").Reverse().Take(80).ToArray();
        }
    }

    private async Task RefreshImagesAsync()
    {
        if (!backendReady || loadingImages)
        {
            return;
        }
        loadingImages = true;
        try
        {
            var stamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var tasks = new List<Task>();
            
            // 1. 核心差异图与抽象图（高频更新，每轮都刷新）
            tasks.Add(LoadImageAsync(diffBox, $"/diff.jpg?t={stamp}"));
            tasks.Add(LoadImageAsync(maskBox, $"/mask.jpg?t={stamp}"));
            
            imageRefreshCounter++;
            
            // 2. 真实相机画面（低频，每 12 轮刷新一次，约 1 秒更新一次）
            if (imageRefreshCounter % 12 == 0)
            {
                tasks.Add(LoadImageAsync(frameBox, $"/frame.jpg?t={stamp}"));
            }
            
            // 3. 当前真实 ROI（中频，每 5 轮刷新一次，约 400ms 更新一次）
            if (imageRefreshCounter % 5 == 0)
            {
                tasks.Add(LoadImageAsync(roiBox, $"/roi.jpg?t={stamp}"));
            }
            
            // 4. 标准模板真图（低频，每 10 轮刷新一次）
            if (imageRefreshCounter % 10 == 0)
            {
                tasks.Add(LoadImageAsync(templateBox, $"/template.jpg?t={stamp}"));
            }
            
            await Task.WhenAll(tasks);
        }
        finally
        {
            loadingImages = false;
        }
    }

    private async Task LoadImageAsync(PictureBox target, string path)
    {
        try
        {
            var bytes = await http.GetByteArrayAsync(path);
            using var ms = new MemoryStream(bytes);
            using var source = Image.FromStream(ms);
            var next = new Bitmap(source);
            var old = target.Image;
            target.Image = next;
            old?.Dispose();
        }
        catch
        {
            // Images may be unavailable before the first frame; keep the last visible frame.
        }
    }

    private async Task<JsonDocument> GetJsonAsync(string path)
    {
        using var response = await http.GetAsync(path);
        return await ParseJsonResponseAsync(response);
    }

    private async Task<JsonDocument> PostJsonAsync(string path, object payload, int timeoutSeconds = 20)
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSeconds));
        var json = JsonSerializer.Serialize(payload);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        using var response = await http.PostAsync(path, content, cts.Token);
        return await ParseJsonResponseAsync(response, cts.Token);
    }

    private static async Task<JsonDocument> ParseJsonResponseAsync(HttpResponseMessage response, CancellationToken cancellationToken = default)
    {
        var text = await response.Content.ReadAsStringAsync(cancellationToken);
        if (!response.IsSuccessStatusCode)
        {
            var message = text;
            try
            {
                using var errorDoc = JsonDocument.Parse(text);
                var root = errorDoc.RootElement;
                if (root.TryGetProperty("message", out var messageElement))
                {
                    message = messageElement.GetString() ?? message;
                }
                else if (root.TryGetProperty("error", out var errorElement))
                {
                    message = errorElement.GetString() ?? message;
                }
            }
            catch
            {
                // Use the raw body.
            }
            throw new InvalidOperationException(message);
        }
        return JsonDocument.Parse(text);
    }

    private void SetResult(string title, string detail, Color color)
    {
        resultLabel.Text = title;
        resultLabel.BackColor = color;
        detailLabel.Text = detail;
    }

    private void AppendLog(string? line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return;
        }
        if (InvokeRequired)
        {
            BeginInvoke(() => AppendLog(line));
            return;
        }
        logText.AppendText(line + Environment.NewLine);
    }

    private static void PlayAlarm()
    {
        for (var i = 0; i < 3; i++)
        {
            SystemSounds.Hand.Play();
            Thread.Sleep(240);
        }
    }

    private async void OnKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.KeyCode == Keys.Space)
        {
            e.Handled = true;
            await QuickStartAsync();
        }
        else if (e.KeyCode == Keys.F5)
        {
            e.Handled = true;
            await ResetDetectionAsync();
        }
        else if (e.KeyCode == Keys.Escape && WindowState == FormWindowState.Maximized)
        {
            WindowState = FormWindowState.Normal;
        }
    }

    private async void OnFormClosing(object? sender, FormClosingEventArgs e)
    {
        refreshTimer.Stop();
        statusTimer.Stop();
        try
        {
            await PostJsonAsync("/stop", new { }, timeoutSeconds: 3);
        }
        catch
        {
            // Ignore shutdown errors.
        }
        try
        {
            if (backendProcess is { HasExited: false })
            {
                backendProcess.Kill(entireProcessTree: true);
            }
        }
        catch
        {
            // Ignore shutdown errors.
        }
    }
}
