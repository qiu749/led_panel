using System.Diagnostics;
using System.Media;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;

namespace LedInspectionStation;

public partial class Form1 : Form
{
    private const int BackendPort = 8765;
    private const string ModelMe70 = "ME70";
    private const string ModelDotPanel = "FS-GRMA-JC";
    private const string ModelBrightnessPanel = "BRIGHTNESS_PANEL";
    private readonly HttpClient http = new() { BaseAddress = new Uri($"http://127.0.0.1:{BackendPort}") };
    private readonly System.Windows.Forms.Timer previewTimer = new() { Interval = 180 };
    private readonly System.Windows.Forms.Timer autoDetectTimer = new() { Interval = 2000 };

    private PictureBox picture = null!;
    private Label resultLabel = null!;
    private Label detailLabel = null!;
    private Label backendLabel = null!;
    private Button previewButton = null!;
    private Button stopButton = null!;
    private Button detectButton = null!;
    private Button autoButton = null!;
    private Button videoButton = null!;
    private Button settingsButton = null!;
    private Panel settingsPanel = null!;
    private ComboBox modelCombo = null!;
    private ComboBox cameraCombo = null!;
    private NumericUpDown frameCountInput = null!;
    private NumericUpDown intervalInput = null!;
    private CheckBox alarmCheck = null!;
    private CheckBox rotateCheck = null!;
    private TextBox outputDirText = null!;
    private TextBox logText = null!;

    private Process? backendProcess;
    private bool backendReady;
    private bool previewing;
    private bool autoDetecting;
    private bool detecting;
    private DateTime resultHoldUntil = DateTime.MinValue;
    private string lastStatusText = "就绪";

    public Form1()
    {
        InitializeComponent();
        BuildOperatorUi();
        KeyPreview = true;
        KeyDown += OnFormKeyDown;
        FormClosing += OnFormClosing;
        previewTimer.Tick += async (_, _) => await RefreshPreviewAsync();
        autoDetectTimer.Tick += async (_, _) => await RunDetectionAsync(triggeredByAuto: true);
        _ = InitializeBackendAsync();
    }

    private void BuildOperatorUi()
    {
        Text = "组件点亮质检 - 现场正式版";
        WindowState = FormWindowState.Maximized;
        BackColor = Color.FromArgb(16, 16, 16);
        Font = new Font("Microsoft YaHei UI", 10F, FontStyle.Regular);

        picture = new PictureBox
        {
            Dock = DockStyle.Fill,
            BackColor = Color.Black,
            SizeMode = PictureBoxSizeMode.Zoom
        };

        var bottom = new Panel
        {
            Dock = DockStyle.Bottom,
            Height = 196,
            BackColor = Color.FromArgb(26, 26, 26),
            Padding = new Padding(16, 8, 16, 10)
        };

        var bottomLayout = new TableLayoutPanel
        {
            Dock = DockStyle.Fill,
            ColumnCount = 1,
            RowCount = 2,
            BackColor = Color.FromArgb(26, 26, 26)
        };
        bottomLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 60));
        bottomLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));

        // 工具栏：型号 + 操作按钮，整行铺满并允许换行，避免按钮被挡住。
        var toolbar = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = true,
            AutoScroll = false
        };

        modelCombo = new ComboBox
        {
            Width = 150,
            Height = 44,
            DropDownStyle = ComboBoxStyle.DropDownList,
            Margin = new Padding(4, 6, 8, 0),
            Font = new Font("Microsoft YaHei UI", 11F, FontStyle.Bold),
            BackColor = Color.FromArgb(54, 54, 54),
            ForeColor = Color.White,
            FlatStyle = FlatStyle.Flat
        };
        modelCombo.Items.AddRange(new object[] { ModelMe70, ModelDotPanel, ModelBrightnessPanel });
        modelCombo.SelectedIndex = 0;
        modelCombo.SelectedIndexChanged += OnModelChanged;

        previewButton = MakeButton("开始预览", 108, OnPreviewClicked);
        stopButton = MakeButton("停止", 74, async (_, _) => await StopPreviewAsync());
        detectButton = MakeButton("检测一次", 108, async (_, _) => await RunDetectionAsync(triggeredByAuto: false));
        autoButton = MakeButton("自动检测", 112, OnAutoClicked);
        videoButton = MakeButton("导入视频检测", 132, async (_, _) => await RunVideoDetectionAsync());
        settingsButton = MakeButton("设置", 74, (_, _) => ToggleSettings());
        alarmCheck = new CheckBox
        {
            Text = "NG报警",
            Checked = true,
            Width = 88,
            Height = 44,
            Margin = new Padding(5, 6, 5, 0),
            ForeColor = Color.White,
            TextAlign = ContentAlignment.MiddleCenter,
            Appearance = Appearance.Button,
            FlatStyle = FlatStyle.Flat
        };
        alarmCheck.FlatAppearance.BorderColor = Color.FromArgb(90, 90, 90);

        toolbar.Controls.Add(MakeCaption("型号"));
        toolbar.Controls.Add(modelCombo);
        toolbar.Controls.AddRange(new Control[] { previewButton, stopButton, detectButton, autoButton, videoButton, settingsButton, alarmCheck });

        // 结果行：左侧大字 PASS/NG，右侧明细。
        var resultRow = new Panel { Dock = DockStyle.Fill, BackColor = Color.FromArgb(26, 26, 26) };
        resultLabel = new Label
        {
            AutoSize = false,
            Text = "READY",
            TextAlign = ContentAlignment.MiddleCenter,
            Dock = DockStyle.Left,
            Width = 220,
            Font = new Font("Microsoft YaHei UI", 34F, FontStyle.Bold),
            ForeColor = Color.White,
            BackColor = Color.FromArgb(78, 78, 78)
        };
        detailLabel = new Label
        {
            AutoSize = false,
            Text = "等待开始预览",
            Dock = DockStyle.Fill,
            TextAlign = ContentAlignment.MiddleLeft,
            Font = new Font("Microsoft YaHei UI", 18F, FontStyle.Bold),
            ForeColor = Color.White,
            Padding = new Padding(18, 0, 12, 0)
        };
        resultRow.Controls.Add(detailLabel);
        resultRow.Controls.Add(resultLabel);

        bottomLayout.Controls.Add(toolbar, 0, 0);
        bottomLayout.Controls.Add(resultRow, 0, 1);
        bottom.Controls.Add(bottomLayout);

        settingsPanel = BuildSettingsPanel();

        Controls.Add(picture);
        Controls.Add(settingsPanel);
        Controls.Add(bottom);
    }

    private Panel BuildSettingsPanel()
    {
        var panel = new Panel
        {
            Dock = DockStyle.Bottom,
            Height = 0,
            BackColor = Color.FromArgb(38, 38, 38),
            Padding = new Padding(16, 8, 16, 8)
        };

        var row = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 52,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false
        };

        cameraCombo = new ComboBox { Width = 360, DropDownStyle = ComboBoxStyle.DropDownList };
        frameCountInput = new NumericUpDown { Width = 72, Minimum = 3, Maximum = 31, Increment = 2, Value = 7 };
        intervalInput = new NumericUpDown { Width = 86, Minimum = 20, Maximum = 1000, Increment = 5, Value = 45 };
        rotateCheck = new CheckBox { Text = "画面旋转180", ForeColor = Color.White, Width = 118, Height = 32 };
        rotateCheck.CheckedChanged += async (_, _) => await ConfigureBackendAsync();
        outputDirText = new TextBox { Width = 420, Text = "" };

        row.Controls.Add(MakeCaption("相机"));
        row.Controls.Add(cameraCombo);
        row.Controls.Add(MakeCaption("多帧"));
        row.Controls.Add(frameCountInput);
        row.Controls.Add(MakeCaption("间隔ms"));
        row.Controls.Add(intervalInput);
        row.Controls.Add(rotateCheck);
        row.Controls.Add(MakeCaption("输出目录"));
        row.Controls.Add(outputDirText);
        row.Controls.Add(MakeButton("刷新相机", 92, async (_, _) => await LoadCamerasAsync()));

        backendLabel = new Label
        {
            Dock = DockStyle.Top,
            Height = 26,
            ForeColor = Color.Gainsboro,
            Text = "后端: 未连接"
        };

        logText = new TextBox
        {
            Dock = DockStyle.Fill,
            Multiline = true,
            ReadOnly = true,
            ScrollBars = ScrollBars.Vertical,
            BackColor = Color.FromArgb(20, 20, 20),
            ForeColor = Color.Gainsboro
        };

        panel.Controls.Add(logText);
        panel.Controls.Add(backendLabel);
        panel.Controls.Add(row);
        return panel;
    }

    private static Label MakeCaption(string text)
    {
        return new Label
        {
            Text = text,
            Width = Math.Max(46, text.Length * 18),
            Height = 32,
            TextAlign = ContentAlignment.MiddleRight,
            ForeColor = Color.White,
            Margin = new Padding(8, 8, 4, 0)
        };
    }

    private static Button MakeButton(string text, int width, EventHandler onClick)
    {
        var button = new Button
        {
            Text = text,
            Width = width,
            Height = 46,
            Margin = new Padding(5, 0, 5, 0),
            FlatStyle = FlatStyle.Flat,
            BackColor = Color.FromArgb(54, 54, 54),
            ForeColor = Color.White,
            Font = new Font("Microsoft YaHei UI", 10.5F, FontStyle.Bold)
        };
        button.FlatAppearance.BorderColor = Color.FromArgb(90, 90, 90);
        button.Click += onClick;
        return button;
    }

    private async Task InitializeBackendAsync()
    {
        SetStatus("READY", "正在启动检测后端...", Color.FromArgb(78, 78, 78));
        await StartBackendProcessAsync();
        for (var i = 0; i < 45; i++)
        {
            if (await PingBackendAsync())
            {
                backendReady = true;
                await LoadCamerasAsync();
                previewTimer.Start();
                SetStatus("READY", "后端已连接，请开始预览", Color.FromArgb(78, 78, 78));
                return;
            }
            await Task.Delay(300);
        }
        SetStatus("ERROR", "检测后端启动失败，请检查 Python 环境", Color.FromArgb(170, 30, 30));
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
                Arguments = $"--port {BackendPort}",
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
            var backend = Path.Combine(root, "csharp_backend.py");
            if (!File.Exists(backend))
            {
                throw new FileNotFoundException("找不到 csharp_backend.py", backend);
            }
            startInfo = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{backend}\" --port {BackendPort}",
                WorkingDirectory = root,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };
        }

        backendProcess = new Process
        {
            StartInfo = startInfo,
            EnableRaisingEvents = true
        };
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
            if (File.Exists(Path.Combine(dir.FullName, "csharp_backend.py")))
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
            using var response = await http.GetAsync("/status");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    private async Task LoadCamerasAsync()
    {
        if (!backendReady && !await PingBackendAsync())
        {
            return;
        }

        var doc = await GetJsonAsync("/cameras");
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
        await RefreshStatusAsync();
    }

    private async void OnPreviewClicked(object? sender, EventArgs e)
    {
        await ConfigureBackendAsync();
        var camera = cameraCombo.SelectedItem?.ToString() ?? "";
        await PostJsonAsync("/start_preview", new { camera });
        previewing = true;
        SetStatus("LIVE", "预览中，按自动检测开始连续检测", Color.FromArgb(55, 95, 150));
        await RefreshStatusAsync();
    }

    private async Task StopPreviewAsync()
    {
        autoDetecting = false;
        autoDetectTimer.Stop();
        UpdateAutoButton();
        await PostJsonAsync("/stop_preview", new { });
        previewing = false;
        SetStatus("READY", "预览已停止", Color.FromArgb(78, 78, 78));
        await RefreshStatusAsync();
    }

    private async void OnAutoClicked(object? sender, EventArgs e)
    {
        if (!previewing)
        {
            await ConfigureBackendAsync();
            var camera = cameraCombo.SelectedItem?.ToString() ?? "";
            await PostJsonAsync("/start_preview", new { camera });
            previewing = true;
        }
        autoDetecting = !autoDetecting;
        UpdateAutoButton();
        if (autoDetecting)
        {
            settingsPanel.Height = 0;
            autoDetectTimer.Start();
            SetStatus("AUTO", "自动检测中，每 2 秒检测一次", Color.FromArgb(55, 95, 150));
            await RunDetectionAsync(triggeredByAuto: true);
        }
        else
        {
            autoDetectTimer.Stop();
            SetStatus("LIVE", "自动检测已停止，预览继续", Color.FromArgb(55, 95, 150));
        }
    }

    private void UpdateAutoButton()
    {
        autoButton.Text = autoDetecting ? "停止自动" : "自动检测";
        autoButton.BackColor = autoDetecting ? Color.FromArgb(160, 84, 28) : Color.FromArgb(54, 54, 54);
    }

    private async Task RunDetectionAsync(bool triggeredByAuto)
    {
        if (InvokeRequired)
        {
            var uiTask = (Task)Invoke(new Func<Task>(() => RunDetectionAsync(triggeredByAuto)));
            await uiTask;
            return;
        }

        if (detecting)
        {
            return;
        }
        detecting = true;
        detectButton.Enabled = false;
        try
        {
            await ConfigureBackendAsync();
            var model = SelectedModel();
            SetStatus("WAIT", BuildCaptureMessage(model), Color.FromArgb(150, 110, 35));
            var result = await PostJsonAsync(
                "/detect",
                new
                {
                    model,
                    frame_count = (int)frameCountInput.Value,
                    interval_ms = (int)intervalInput.Value
                });
            var root = result.RootElement;
            var ok = root.TryGetProperty("overall_ok", out var okElement) && okElement.GetBoolean();
            var message = root.TryGetProperty("message", out var msgElement) ? msgElement.GetString() ?? "" : "";
            var fallback = BuildFallbackDetail(root, model);
            SetStatus(ok ? "PASS" : "NG", string.IsNullOrWhiteSpace(message) ? fallback : message, ok ? Color.FromArgb(30, 140, 70) : Color.FromArgb(190, 30, 30));
            resultHoldUntil = DateTime.Now.AddSeconds(ok ? 1.2 : 3.0);
            await LoadImageAsync("/result.jpg");
            if (!ok && alarmCheck.Checked)
            {
                _ = Task.Run(PlayAlarm);
            }
        }
        catch (Exception ex)
        {
            SetStatus("ERROR", ex.Message, Color.FromArgb(170, 30, 30));
        }
        finally
        {
            detectButton.Enabled = true;
            detecting = false;
        }
    }

    private string SelectedModel()
    {
        if (InvokeRequired)
        {
            return (string)Invoke(() => SelectedModel());
        }
        return modelCombo.SelectedItem?.ToString() ?? ModelMe70;
    }

    private static string BuildCaptureMessage(string model)
    {
        return model switch
        {
            ModelDotPanel => "采集中，匹配点阵模板...",
            ModelBrightnessPanel => "采集中，对比白字亮度模板...",
            _ => "采集中，正在融合多帧..."
        };
    }

    private async Task RunVideoDetectionAsync()
    {
        if (detecting)
        {
            return;
        }
        string videoPath;
        using (var dialog = new OpenFileDialog
        {
            Title = "选择测试视频",
            Filter = "视频文件|*.avi;*.mp4;*.mov;*.mkv;*.wmv|所有文件|*.*",
            CheckFileExists = true,
            Multiselect = false
        })
        {
            if (dialog.ShowDialog(this) != DialogResult.OK)
            {
                return;
            }
            videoPath = dialog.FileName;
        }

        if (string.IsNullOrWhiteSpace(videoPath) || !File.Exists(videoPath))
        {
            MessageBox.Show("所选视频文件不存在或无效，请重新选择！", "文件错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        detecting = true;
        videoButton.Enabled = false;
        detectButton.Enabled = false;
        try
        {
            await ConfigureBackendAsync();
            SetStatus("WAIT", $"正在载入视频: {Path.GetFileName(videoPath)}", Color.FromArgb(55, 95, 150));
            
            // 将视频路径作为 "camera" 传给后端启动视频流预览！
            await PostJsonAsync("/start_preview", new { camera = videoPath });
            previewing = true;
            
            // 自动开启“自动检测”，使其像实时相机一样自动、循环播放和检测
            autoDetecting = true;
            UpdateAutoButton();
            autoDetectTimer.Start();
            
            SetStatus("AUTO", $"[视频模式] 正在播放并自动检测: {Path.GetFileName(videoPath)}", Color.FromArgb(55, 95, 150));
            
            // 立即进行一次初始检测
            _ = Task.Run(async () => {
                await Task.Delay(200); // 给视频流一点点启动稳定时间
                await RunDetectionAsync(triggeredByAuto: true);
            });
        }
        catch (Exception ex)
        {
            SetStatus("ERROR", "视频载入失败: " + ex.Message, Color.FromArgb(170, 30, 30));
        }
        finally
        {
            videoButton.Enabled = true;
            detectButton.Enabled = true;
            detecting = false;
        }
    }

    private static string BuildFallbackDetail(JsonElement root, string model)
    {
        if (model == ModelDotPanel)
        {
            var matched = root.TryGetProperty("matched_count", out var m) ? m.GetInt32() : 0;
            var expected = root.TryGetProperty("expected_count", out var e) ? e.GetInt32() : 0;
            var missing = root.TryGetProperty("missing_count", out var ms) ? ms.GetInt32() : 0;
            var extra = root.TryGetProperty("extra_count", out var ex) ? ex.GetInt32() : 0;
            return $"匹配 {matched}/{expected} 缺{missing} 多{extra}";
        }
        if (model == ModelBrightnessPanel)
        {
            var badComponents = root.TryGetProperty("bad_component_count", out var bad) ? bad.GetInt32() : 0;
            var ngGroups = "-";
            if (root.TryGetProperty("ng_groups", out var groups) && groups.ValueKind == JsonValueKind.Array)
            {
                ngGroups = string.Join(",", groups.EnumerateArray().Select(item => item.GetString()).Where(item => !string.IsNullOrWhiteSpace(item)));
                if (string.IsNullOrWhiteSpace(ngGroups))
                {
                    ngGroups = "-";
                }
            }
            var align = 0.0;
            if (root.TryGetProperty("alignment", out var alignment) && alignment.ValueKind == JsonValueKind.Object)
            {
                align = alignment.TryGetProperty("score", out var score) ? score.GetDouble() : 0.0;
            }
            return $"白字亮度 NG列={ngGroups} 暗亮区={badComponents} align={align:F2}";
        }
        var roiCount = root.TryGetProperty("roi_count", out var roiElement) ? roiElement.GetInt32() : 0;
        var ngCount = root.TryGetProperty("ng_count", out var ngElement) ? ngElement.GetInt32() : 0;
        return $"ROI={roiCount}/20 NG={ngCount}";
    }

    private void OnModelChanged(object? sender, EventArgs e)
    {
        var model = SelectedModel();
        if (model == ModelDotPanel)
        {
            // FS-GRMA-JC 单 ROI 模板匹配，少量帧即可。
            frameCountInput.Value = Math.Max(frameCountInput.Minimum, Math.Min(frameCountInput.Maximum, 5));
            intervalInput.Value = Math.Max(intervalInput.Minimum, Math.Min(intervalInput.Maximum, 60));
            SetStatus(lastStatusText.StartsWith("PASS") || lastStatusText.StartsWith("NG") ? "READY" : "READY",
                "已切换型号: FS-GRMA-JC（白/蓝点阵面板）", Color.FromArgb(78, 78, 78));
        }
        else if (model == ModelBrightnessPanel)
        {
            frameCountInput.Value = Math.Max(frameCountInput.Minimum, Math.Min(frameCountInput.Maximum, 7));
            intervalInput.Value = Math.Max(intervalInput.Minimum, Math.Min(intervalInput.Maximum, 60));
            SetStatus("READY", "已切换型号: 白字面板亮度（模板化亮度检测）", Color.FromArgb(78, 78, 78));
        }
        else
        {
            frameCountInput.Value = Math.Max(frameCountInput.Minimum, Math.Min(frameCountInput.Maximum, 7));
            intervalInput.Value = Math.Max(intervalInput.Minimum, Math.Min(intervalInput.Maximum, 45));
            SetStatus("READY", "已切换型号: ME70（红色点阵面板）", Color.FromArgb(78, 78, 78));
        }
    }

    private async Task ConfigureBackendAsync()
    {
        await PostJsonAsync(
            "/configure",
            new
            {
                rotate_180 = rotateCheck.Checked,
                output_dir = outputDirText.Text.Trim()
            });
    }

    private async Task RefreshPreviewAsync()
    {
        if (!backendReady || detecting)
        {
            return;
        }
        if (!previewing && DateTime.Now >= resultHoldUntil)
        {
            return;
        }
        try
        {
            if (DateTime.Now < resultHoldUntil)
            {
                await LoadImageAsync("/result.jpg");
            }
            else
            {
                await LoadImageAsync("/frame.jpg");
            }
        }
        catch
        {
            // No frame yet; keep the current display.
        }
    }

    private async Task LoadImageAsync(string path)
    {
        var bytes = await http.GetByteArrayAsync(path);
        using var ms = new MemoryStream(bytes);
        using var source = Image.FromStream(ms);
        var next = new Bitmap(source);
        var old = picture.Image;
        picture.Image = next;
        old?.Dispose();
    }

    private async Task RefreshStatusAsync()
    {
        try
        {
            var doc = await GetJsonAsync("/status");
            var root = doc.RootElement;
            var previewText = root.TryGetProperty("previewing", out var previewElement) && previewElement.GetBoolean() ? "预览已连接" : "预览未启动";
            var camera = root.TryGetProperty("camera", out var cameraElement) ? cameraElement.GetString() : "";
            backendLabel.Text = $"后端: 已连接  {previewText}  {camera}";
            if (root.TryGetProperty("logs", out var logs))
            {
                logText.Lines = logs.EnumerateArray().Select(item => item.GetString() ?? "").ToArray();
            }
        }
        catch
        {
            backendLabel.Text = "后端: 未连接";
        }
    }

    private async Task<JsonDocument> GetJsonAsync(string path)
    {
        using var response = await http.GetAsync(path);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync();
        return await JsonDocument.ParseAsync(stream);
    }

    private async Task<JsonDocument> PostJsonAsync(string path, object payload)
    {
        var json = JsonSerializer.Serialize(payload);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        using var response = await http.PostAsync(path, content);
        response.EnsureSuccessStatusCode();
        await using var stream = await response.Content.ReadAsStreamAsync();
        return await JsonDocument.ParseAsync(stream);
    }

    private void ToggleSettings()
    {
        if (autoDetecting)
        {
            settingsPanel.Height = 0;
            return;
        }
        settingsPanel.Height = settingsPanel.Height == 0 ? 190 : 0;
    }

    private void SetStatus(string title, string detail, Color color)
    {
        lastStatusText = $"{title} {detail}";
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
            Thread.Sleep(260);
        }
    }

    private async void OnFormKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.KeyCode == Keys.Space)
        {
            e.Handled = true;
            await RunDetectionAsync(triggeredByAuto: false);
        }
        else if (e.KeyCode == Keys.F12)
        {
            ToggleSettings();
        }
        else if (e.KeyCode == Keys.Escape && WindowState == FormWindowState.Maximized)
        {
            WindowState = FormWindowState.Normal;
        }
    }

    private async void OnFormClosing(object? sender, FormClosingEventArgs e)
    {
        previewTimer.Stop();
        autoDetectTimer.Stop();
        try
        {
            await PostJsonAsync("/stop_preview", new { });
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
