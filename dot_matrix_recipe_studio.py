from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dot_matrix_video_learner as learner
import render_dot_matrix_video_review as review_renderer
from desktop_app import (
    CameraInfo,
    HikvisionCameraStream,
    OpenCVCameraStream,
    enumerate_hikvision_cameras,
    enumerate_opencv_cameras,
)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def open_local_path(path: Path) -> None:
    target = path.resolve()
    try:
        os.startfile(str(target))
        return
    except Exception:
        pass
    webbrowser.open(target.as_uri())


class JobRunner:
    def __init__(self, post_ui: Callable[[Callable[[], None]], None], log: Callable[[str], None]) -> None:
        self._post_ui = post_ui
        self._log = log
        self._lock = threading.Lock()
        self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def run(
        self,
        description: str,
        work: Callable[[], Any],
        on_success: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception, str], None] | None = None,
    ) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def finish() -> None:
            with self._lock:
                self._busy = False

        def worker() -> None:
            self._log(f"开始: {description}")
            try:
                result = work()
            except Exception as exc:
                detail = traceback.format_exc()
                self._log(f"失败: {description} -> {exc}")
                self._log(detail)
                self._post_ui(lambda: self._handle_error(exc, detail, on_error, finish))
                return
            self._log(f"完成: {description}")
            self._post_ui(lambda: self._handle_success(result, on_success, finish))

        threading.Thread(target=worker, name=f"job-{description}", daemon=True).start()
        return True

    def _handle_success(self, result: Any, on_success: Callable[[Any], None] | None, finish: Callable[[], None]) -> None:
        try:
            if on_success is not None:
                on_success(result)
        finally:
            finish()

    def _handle_error(
        self,
        exc: Exception,
        detail: str,
        on_error: Callable[[Exception, str], None] | None,
        finish: Callable[[], None],
    ) -> None:
        try:
            if on_error is not None:
                on_error(exc, detail)
            else:
                messagebox.showerror("执行失败", f"{exc}\n\n{detail}")
        finally:
            finish()


class CameraPanel(ttk.LabelFrame):
    def __init__(
        self,
        master: tk.Misc,
        log: Callable[[str], None],
        on_record_good: Callable[[], None],
        on_record_test: Callable[[], None],
    ) -> None:
        super().__init__(master, text="相机采集")
        self.log = log
        self.on_record_good = on_record_good
        self.on_record_test = on_record_test
        self.cameras: list[CameraInfo] = []
        self.camera_stream: OpenCVCameraStream | HikvisionCameraStream | None = None
        self.camera_var = tk.StringVar()
        self.record_seconds_var = tk.DoubleVar(value=6.0)
        self.record_fps_var = tk.IntVar(value=12)
        self.preview_image: ImageTk.PhotoImage | None = None
        self._build()
        self.refresh_cameras()

    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", padx=8, pady=8)
        ttk.Label(header, text="相机").grid(row=0, column=0, sticky="w")
        self.camera_combo = ttk.Combobox(header, textvariable=self.camera_var, state="readonly", width=34)
        self.camera_combo.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(header, text="刷新", command=self.refresh_cameras).grid(row=0, column=2, padx=4)
        ttk.Button(header, text="开始预览", command=self.start_preview).grid(row=0, column=3, padx=4)
        ttk.Button(header, text="停止预览", command=self.stop_preview).grid(row=0, column=4, padx=4)
        ttk.Label(header, text="录制秒数").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(header, from_=1.0, to=30.0, increment=1.0, textvariable=self.record_seconds_var, width=8).grid(row=1, column=1, sticky="w", padx=(8, 8), pady=(8, 0))
        ttk.Label(header, text="采样 FPS").grid(row=1, column=2, sticky="e", pady=(8, 0))
        ttk.Spinbox(header, from_=5, to=30, increment=1, textvariable=self.record_fps_var, width=8).grid(row=1, column=3, sticky="w", padx=(8, 8), pady=(8, 0))
        ttk.Button(header, text="录为好品视频", command=self.on_record_good).grid(row=2, column=1, sticky="w", padx=(8, 8), pady=(10, 0))
        ttk.Button(header, text="录为待测视频", command=self.on_record_test).grid(row=2, column=3, sticky="w", padx=(8, 8), pady=(10, 0))
        header.columnconfigure(1, weight=1)

        self.preview_label = ttk.Label(self, text="未开始预览", anchor="center")
        self.preview_label.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def refresh_cameras(self) -> None:
        labels: list[str] = []
        cameras: list[CameraInfo] = []
        try:
            cameras.extend(enumerate_hikvision_cameras())
        except Exception as exc:
            self.log(f"枚举海康相机失败: {exc}")
        try:
            cameras.extend(enumerate_opencv_cameras())
        except Exception as exc:
            self.log(f"枚举 OpenCV 摄像头失败: {exc}")
        self.cameras = cameras
        labels = [camera.label for camera in cameras]
        self.camera_combo["values"] = labels
        if labels:
            current = self.camera_var.get()
            self.camera_var.set(current if current in labels else labels[0])
            self.log(f"摄像头数量: {len(labels)}")
        else:
            self.camera_var.set("")
            self.log("未找到可用相机。")

    def selected_camera(self) -> CameraInfo | None:
        label = self.camera_var.get()
        for camera in self.cameras:
            if camera.label == label:
                return camera
        return None

    def start_preview(self) -> None:
        camera = self.selected_camera()
        if camera is None:
            messagebox.showwarning("未选择相机", "请先选择可用相机。")
            return
        self.stop_preview()
        try:
            if camera.kind == "hikvision":
                self.camera_stream = HikvisionCameraStream(camera, self.log)
            else:
                self.camera_stream = OpenCVCameraStream(camera, self.log)
            self.camera_stream.start()
            self.log(f"开始预览: {camera.label}")
        except Exception as exc:
            self.camera_stream = None
            messagebox.showerror("预览失败", str(exc))

    def stop_preview(self) -> None:
        if self.camera_stream is not None:
            self.camera_stream.stop()
            self.camera_stream = None
            self.log("预览已停止")

    def is_previewing(self) -> bool:
        return self.camera_stream is not None

    def get_latest_frame(self) -> Any:
        if self.camera_stream is None:
            return None
        return self.camera_stream.get_latest_frame()

    def update_preview(self) -> None:
        frame = self.get_latest_frame()
        if frame is None:
            self.preview_label.configure(text="未开始预览", image="")
            return
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image)
        pil.thumbnail((520, 320))
        self.preview_image = ImageTk.PhotoImage(pil)
        self.preview_label.configure(image=self.preview_image, text="")


class DotMatrixRecipeStudio(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("点阵 LED 冻结配方 Demo")
        self.geometry("1480x980")
        self.minsize(1320, 860)

        self.ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.runner = JobRunner(self.post_ui, self.log)
        self.image_refs: dict[str, ImageTk.PhotoImage | None] = {}

        self.session_root = ROOT / "output" / "recipe_studio_sessions" / f"session_{now_stamp()}"
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.captures_dir = self.session_root / "captures"
        self.captures_dir.mkdir(parents=True, exist_ok=True)

        self.output_root_var = tk.StringVar(value=str(self.session_root))
        self.good_video_var = tk.StringVar()
        self.test_video_var = tk.StringVar()
        self.learn_sample_fps_var = tk.DoubleVar(value=25.0)
        self.learn_max_frames_var = tk.IntVar(value=260)
        self.detect_sample_fps_var = tk.DoubleVar(value=12.0)
        self.detect_max_frames_var = tk.IntVar(value=120)
        self.candidate_recipe_var = tk.StringVar()
        self.approval_path_var = tk.StringVar()
        self.frozen_recipe_var = tk.StringVar()
        self.override_var = tk.BooleanVar(value=False)
        self.candidate_summary_var = tk.StringVar(value="未加载 candidate")
        self.detect_summary_var = tk.StringVar(value="未运行 detect")
        self.status_var = tk.StringVar(value="就绪")

        self.current_candidate: dict[str, Any] | None = None
        self.current_candidate_dir: Path | None = None
        self.current_frozen_dir: Path | None = None
        self.current_candidate_review: Path | None = None
        self.current_detect_review: Path | None = None
        self.current_detect_output: Path | None = None
        self.state_keep_vars: dict[str, tk.BooleanVar] = {}
        self.state_rows: list[tk.Widget] = []

        self._build_ui()
        self.after(100, self._drain_log_queue)
        self.after(100, self._drain_ui_queue)
        self.after(80, self._refresh_preview)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.log(f"会话目录: {self.session_root}")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", padx=10, pady=(10, 6))
        ttk.Label(header, text="输出根目录").pack(side="left")
        ttk.Entry(header, textvariable=self.output_root_var).pack(side="left", fill="x", expand=True, padx=(8, 8))
        ttk.Button(header, text="选择目录", command=self._choose_output_root).pack(side="left", padx=4)
        ttk.Button(header, text="打开目录", command=lambda: self._open_path(Path(self.output_root_var.get()))).pack(side="left", padx=4)
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self.input_tab = ttk.Frame(self.notebook)
        self.learn_tab = ttk.Frame(self.notebook)
        self.approval_tab = ttk.Frame(self.notebook)
        self.detect_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.input_tab, text="1. 输入与采集")
        self.notebook.add(self.learn_tab, text="2. 候选学习")
        self.notebook.add(self.approval_tab, text="3. 确认与冻结")
        self.notebook.add(self.detect_tab, text="4. 产线检测")

        self._build_input_tab()
        self._build_learn_tab()
        self._build_approval_tab()
        self._build_detect_tab()

        log_frame = ttk.LabelFrame(outer, text="运行日志")
        log_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.log_text = tk.Text(log_frame, height=8, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_input_tab(self) -> None:
        file_frame = ttk.LabelFrame(self.input_tab, text="本地视频")
        file_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(file_frame, text="好品视频").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(file_frame, textvariable=self.good_video_var).grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(file_frame, text="选择", command=lambda: self._pick_video(self.good_video_var)).grid(row=0, column=2, padx=8, pady=8)
        ttk.Label(file_frame, text="待测视频").grid(row=1, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(file_frame, textvariable=self.test_video_var).grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(file_frame, text="选择", command=lambda: self._pick_video(self.test_video_var)).grid(row=1, column=2, padx=8, pady=8)
        file_frame.columnconfigure(1, weight=1)

        self.camera_panel = CameraPanel(self.input_tab, self.log, self._record_good_from_camera, self._record_test_from_camera)
        self.camera_panel.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def _build_learn_tab(self) -> None:
        controls = ttk.LabelFrame(self.learn_tab, text="候选学习")
        controls.pack(fill="x", padx=10, pady=10)
        ttk.Label(controls, text="Sample FPS").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Spinbox(controls, from_=5, to=60, increment=1, textvariable=self.learn_sample_fps_var, width=10).grid(row=0, column=1, padx=8, pady=8, sticky="w")
        ttk.Label(controls, text="Max Frames").grid(row=0, column=2, padx=8, pady=8, sticky="w")
        ttk.Spinbox(controls, from_=0, to=2000, increment=20, textvariable=self.learn_max_frames_var, width=10).grid(row=0, column=3, padx=8, pady=8, sticky="w")
        ttk.Button(controls, text="生成 candidate", command=self._run_learn_candidate).grid(row=0, column=4, padx=8, pady=8)
        ttk.Label(controls, text="Candidate").grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(controls, textvariable=self.candidate_recipe_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=8, pady=8)
        ttk.Button(controls, text="载入", command=self._load_candidate_from_entry).grid(row=1, column=4, padx=8, pady=8)
        ttk.Button(controls, text="打开 review", command=self._open_candidate_review).grid(row=1, column=5, padx=8, pady=8)
        controls.columnconfigure(2, weight=0)
        controls.columnconfigure(3, weight=0)
        controls.columnconfigure(1, weight=1)

        ttk.Label(self.learn_tab, textvariable=self.candidate_summary_var, justify="left", anchor="nw").pack(fill="x", padx=14, pady=(0, 10))

    def _build_approval_tab(self) -> None:
        top = ttk.Frame(self.approval_tab)
        top.pack(fill="x", padx=10, pady=10)
        ttk.Label(top, text="Candidate").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(top, textvariable=self.candidate_recipe_var).grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(top, text="浏览", command=self._browse_candidate).grid(row=0, column=2, padx=8, pady=8)
        ttk.Button(top, text="重新载入", command=self._load_candidate_from_entry).grid(row=0, column=3, padx=8, pady=8)
        ttk.Checkbutton(top, text="允许 override diagnostics", variable=self.override_var, command=self._update_freeze_button_state).grid(row=1, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(top, text="Approval").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(top, textvariable=self.approval_path_var).grid(row=1, column=1, sticky="ew", padx=(80, 8), pady=4)
        ttk.Button(top, text="保存 approval", command=self._save_approval).grid(row=1, column=2, padx=8, pady=4)
        self.freeze_button = ttk.Button(top, text="冻结 frozen recipe", command=self._run_freeze)
        self.freeze_button.grid(row=1, column=3, padx=8, pady=4)
        ttk.Label(top, text="Frozen").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(top, textvariable=self.frozen_recipe_var).grid(row=2, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(top, text="打开 frozen", command=self._open_frozen_recipe).grid(row=2, column=2, padx=8, pady=8)
        ttk.Button(top, text="打开 review", command=self._open_candidate_review).grid(row=2, column=3, padx=8, pady=8)
        top.columnconfigure(1, weight=1)

        body = ttk.PanedWindow(self.approval_tab, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)

        summary_frame = ttk.LabelFrame(left, text="候选摘要与风险")
        summary_frame.pack(fill="both", expand=False, pady=(0, 10))
        self.candidate_detail_text = tk.Text(summary_frame, height=8, wrap="word")
        self.candidate_detail_text.pack(fill="both", expand=True, padx=8, pady=8)

        issues_frame = ttk.LabelFrame(left, text="风险项")
        issues_frame.pack(fill="both", expand=True)
        self.issues_listbox = tk.Listbox(issues_frame, height=10)
        self.issues_listbox.pack(fill="both", expand=True, padx=8, pady=8)

        previews_frame = ttk.LabelFrame(right, text="预览")
        previews_frame.pack(fill="x", expand=False, pady=(0, 10))
        grid = ttk.Frame(previews_frame)
        grid.pack(fill="x", padx=8, pady=8)
        self.roi_preview = ttk.Label(grid, text="ROI 预览")
        self.roi_preview.grid(row=0, column=0, padx=6, pady=6)
        self.projection_preview = ttk.Label(grid, text="投影图")
        self.projection_preview.grid(row=0, column=1, padx=6, pady=6)
        self.state_preview = ttk.Label(grid, text="显示状态")
        self.state_preview.grid(row=1, column=0, padx=6, pady=6)
        self.phase_preview = ttk.Label(grid, text="507 相位")
        self.phase_preview.grid(row=1, column=1, padx=6, pady=6)

        states_frame = ttk.LabelFrame(right, text="显示状态审批")
        states_frame.pack(fill="both", expand=True)
        self.states_canvas = tk.Canvas(states_frame, highlightthickness=0)
        self.states_scroll = ttk.Scrollbar(states_frame, orient="vertical", command=self.states_canvas.yview)
        self.states_inner = ttk.Frame(self.states_canvas)
        self.states_inner.bind("<Configure>", lambda _event: self.states_canvas.configure(scrollregion=self.states_canvas.bbox("all")))
        self.states_canvas.create_window((0, 0), window=self.states_inner, anchor="nw")
        self.states_canvas.configure(yscrollcommand=self.states_scroll.set)
        self.states_canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        self.states_scroll.pack(side="right", fill="y", padx=(0, 8), pady=8)

    def _build_detect_tab(self) -> None:
        controls = ttk.LabelFrame(self.detect_tab, text="产线检测")
        controls.pack(fill="x", padx=10, pady=10)
        ttk.Label(controls, text="Frozen Recipe").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        ttk.Entry(controls, textvariable=self.frozen_recipe_var).grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        ttk.Button(controls, text="浏览", command=self._browse_frozen).grid(row=0, column=2, padx=8, pady=8)
        ttk.Label(controls, text="Sample FPS").grid(row=1, column=0, padx=8, pady=8, sticky="w")
        ttk.Spinbox(controls, from_=5, to=60, increment=1, textvariable=self.detect_sample_fps_var, width=10).grid(row=1, column=1, padx=8, pady=8, sticky="w")
        ttk.Label(controls, text="Max Frames").grid(row=1, column=2, padx=8, pady=8, sticky="w")
        ttk.Spinbox(controls, from_=0, to=2000, increment=20, textvariable=self.detect_max_frames_var, width=10).grid(row=1, column=3, padx=8, pady=8, sticky="w")
        ttk.Button(controls, text="开始 detect", command=self._run_detect).grid(row=1, column=4, padx=8, pady=8)
        ttk.Button(controls, text="打开 detect review", command=self._open_detect_review).grid(row=1, column=5, padx=8, pady=8)
        ttk.Button(controls, text="打开 review 视频", command=self._open_detect_review_video).grid(row=1, column=6, padx=8, pady=8)
        ttk.Button(controls, text="打开输出目录", command=self._open_detect_output).grid(row=1, column=7, padx=8, pady=8)
        controls.columnconfigure(1, weight=1)

        ttk.Label(self.detect_tab, textvariable=self.detect_summary_var, justify="left", anchor="nw").pack(fill="x", padx=14, pady=(0, 10))

        events_frame = ttk.LabelFrame(self.detect_tab, text="事件列表")
        events_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        columns = ("frame", "category", "type", "expected", "matched", "score")
        self.events_tree = ttk.Treeview(events_frame, columns=columns, show="headings", height=12)
        for key, title, width in (
            ("frame", "帧号", 90),
            ("category", "分类", 130),
            ("type", "类型", 140),
            ("expected", "期望", 100),
            ("matched", "匹配", 100),
            ("score", "分数", 90),
        ):
            self.events_tree.heading(key, text=title)
            self.events_tree.column(key, width=width, anchor="w")
        self.events_tree.pack(fill="both", expand=True, padx=8, pady=8)

    def _choose_output_root(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.output_root_var.get() or str(ROOT))
        if chosen:
            self.output_root_var.set(chosen)

    def _pick_video(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv"), ("All Files", "*.*")],
        )
        if path:
            variable.set(path)

    def _browse_candidate(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 candidate_recipe.json",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        if path:
            self.candidate_recipe_var.set(path)
            self._load_candidate(Path(path))

    def _browse_frozen(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 frozen_recipe.json",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
        )
        if path:
            self.frozen_recipe_var.set(path)

    def _load_candidate_from_entry(self) -> None:
        path_value = self.candidate_recipe_var.get().strip()
        if not path_value:
            messagebox.showwarning("未选择 candidate", "请先选择或生成 candidate_recipe.json。")
            return
        self._load_candidate(Path(path_value))

    def _load_candidate(self, path: Path) -> None:
        if not path.exists():
            messagebox.showerror("文件不存在", f"找不到 candidate recipe:\n{path}")
            return
        recipe = load_json(path)
        self.current_candidate = recipe
        self.current_candidate_dir = path.parent
        self.candidate_recipe_var.set(str(path))
        self.current_candidate_review = self.current_candidate_dir / "review.html"
        self.frozen_recipe_var.set(str(self.current_candidate_dir / "frozen_recipe.json"))
        self.approval_path_var.set(str(self.current_candidate_dir / "approval.json"))

        approval_template_path = self.current_candidate_dir / "approval_template.json"
        approval_path = Path(self.approval_path_var.get())
        approval = None
        if approval_path.exists():
            approval = load_json(approval_path)
        elif approval_template_path.exists():
            approval = load_json(approval_template_path)
        else:
            approval = learner.make_approval_template(recipe)
        self.override_var.set(bool(approval.get("override_diagnostics", False)))

        self._populate_candidate_summary(recipe)
        self._populate_issues(recipe)
        self._populate_state_rows(recipe, approval)
        self._load_candidate_previews(recipe)
        self._update_freeze_button_state()
        self.log(f"已载入 candidate: {path}")
        self.notebook.select(self.approval_tab)

    def _populate_candidate_summary(self, recipe: dict[str, Any]) -> None:
        diagnostics = recipe.get("diagnostics", {})
        summary_lines = [
            f"状态: {recipe.get('status', '')}",
            f"结论: {diagnostics.get('verdict', '')}",
            f"物理槽位: {recipe.get('slot_count', 0)}",
            f"扫描相位: {len(recipe.get('scan_phase_templates', []))}",
            f"显示状态: {len(recipe.get('display_states', []))}",
            f"原始扫描态: {len(recipe.get('states', []))}",
            f"ROI: {recipe.get('panel_bbox', None)}",
        ]
        self.candidate_summary_var.set("\n".join(summary_lines))
        self.candidate_detail_text.delete("1.0", "end")
        self.candidate_detail_text.insert("1.0", "\n".join(summary_lines))

    def _populate_issues(self, recipe: dict[str, Any]) -> None:
        self.issues_listbox.delete(0, "end")
        issues = recipe.get("diagnostics", {}).get("issues", [])
        if not issues:
            self.issues_listbox.insert("end", "没有自动风险项，但仍需人工确认。")
            return
        for item in issues:
            line = f"[{item.get('severity', '')}] {item.get('code', '')} - {item.get('message', '')}"
            self.issues_listbox.insert("end", line)

    def _populate_state_rows(self, recipe: dict[str, Any], approval: dict[str, Any]) -> None:
        for widget in self.state_rows:
            widget.destroy()
        self.state_rows = []
        self.state_keep_vars = {}

        accepted = {str(item) for item in approval.get("accepted_display_states", [])}
        ignored = {str(item) for item in approval.get("ignored_states", [])}
        low_coverage = {
            str(item.get("data", {}).get("state_id", ""))
            for item in recipe.get("diagnostics", {}).get("issues", [])
            if item.get("code") == "low_coverage_display_state"
        }
        for state in recipe.get("display_states", []):
            sid = str(state.get("id", ""))
            default_keep = sid in accepted and sid not in ignored
            keep_var = tk.BooleanVar(value=default_keep)
            self.state_keep_vars[sid] = keep_var
            row = ttk.Frame(self.states_inner)
            row.pack(fill="x", padx=6, pady=4)
            text = f"{sid}  亮点={state.get('on_count', 0)}  RGB={state.get('mean_rgb', [])}"
            if sid in low_coverage:
                text += "  [低覆盖]"
            ttk.Checkbutton(row, text=text, variable=keep_var, command=self._update_freeze_button_state).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="预览", command=lambda state_id=sid: self._show_state_preview(state_id)).pack(side="right", padx=4)
            self.state_rows.append(row)
        display_states = recipe.get("display_states", [])
        if display_states:
            self._show_state_preview(str(display_states[0].get("id", "")))
        scan_states = recipe.get("scan_phase_color_templates", []) or recipe.get("scan_phase_templates", [])
        if scan_states:
            self._show_phase_preview(str(scan_states[0].get("id", "")))

    def _load_candidate_previews(self, recipe: dict[str, Any]) -> None:
        assert self.current_candidate_dir is not None
        roi_path = self.current_candidate_dir / "slot_overlay_crop.jpg"
        if not roi_path.exists():
            roi_path = self.current_candidate_dir / "slot_overlay.jpg"
        projection_path = self.current_candidate_dir / "slot_source_projection.jpg"
        self._set_image(self.roi_preview, roi_path, "roi_preview")
        self._set_image(self.projection_preview, projection_path, "projection_preview")
        scan_states = recipe.get("scan_phase_color_templates", []) or recipe.get("scan_phase_templates", [])
        if scan_states:
            self._show_phase_preview(str(scan_states[0].get("id", "")))

    def _set_image(self, label: ttk.Label, path: Path, key: str, size: tuple[int, int] = (340, 220)) -> None:
        if not path.exists():
            self.image_refs[key] = None
            label.configure(image="", text=path.name)
            return
        try:
            pil = Image.open(path)
            pil.thumbnail(size)
            image = ImageTk.PhotoImage(pil)
            self.image_refs[key] = image
            label.configure(image=image, text="")
        except Exception as exc:
            self.image_refs[key] = None
            label.configure(image="", text=f"无法加载: {path.name}\n{exc}")

    def _show_state_preview(self, state_id: str) -> None:
        if self.current_candidate_dir is None:
            return
        path = self.current_candidate_dir / "display_state_previews" / f"{state_id}.jpg"
        self._set_image(self.state_preview, path, "state_preview")

    def _show_phase_preview(self, phase_id: str) -> None:
        if self.current_candidate_dir is None:
            return
        color_path = self.current_candidate_dir / "scan_phase_color_previews" / f"{phase_id}.jpg"
        path = color_path if color_path.exists() else self.current_candidate_dir / "scan_phase_previews" / f"{phase_id}.jpg"
        self._set_image(self.phase_preview, path, "phase_preview")

    def _record_good_from_camera(self) -> None:
        self._record_camera_clip("good", "好品视频", self.good_video_var)

    def _record_test_from_camera(self) -> None:
        self._record_camera_clip("test", "待测视频", self.test_video_var)

    def _record_camera_clip(self, slug: str, title: str, target_var: tk.StringVar) -> None:
        if not self.camera_panel.is_previewing():
            messagebox.showwarning("未开始预览", "请先启动相机预览。")
            return
        seconds = max(1.0, float(self.camera_panel.record_seconds_var.get()))
        fps = max(5, int(self.camera_panel.record_fps_var.get()))
        output_path = self.captures_dir / f"{slug}_{now_stamp()}.mp4"

        def work() -> Path:
            started = time.monotonic()
            interval = 1.0 / float(fps)
            writer = None
            frame_count = 0
            last_shape: tuple[int, int] | None = None
            while time.monotonic() - started < seconds:
                frame = self.camera_panel.get_latest_frame()
                if frame is not None:
                    h, w = frame.shape[:2]
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
                        last_shape = (w, h)
                    elif last_shape != (w, h):
                        raise RuntimeError("录制期间相机分辨率发生变化，已中止。")
                    writer.write(frame)
                    frame_count += 1
                time.sleep(interval)
            if writer is not None:
                writer.release()
            if frame_count == 0 or not output_path.exists():
                raise RuntimeError("未录到有效帧。")
            return output_path

        def on_success(path: Path) -> None:
            target_var.set(str(path))
            self.status_var.set(f"{title}录制完成: {path.name}")
            self.log(f"{title}录制完成: {path}")
            self.notebook.select(self.learn_tab if target_var is self.good_video_var else self.detect_tab)

        if not self.runner.run(f"录制{title}", work, on_success=on_success):
            messagebox.showinfo("忙碌中", "当前已有任务在执行，请稍后。")

    def _run_learn_candidate(self) -> None:
        video = Path(self.good_video_var.get().strip()) if self.good_video_var.get().strip() else None
        if video is None or not video.exists():
            messagebox.showwarning("未选择好品视频", "请先选择或录制好品视频。")
            return
        output_root = Path(self.output_root_var.get().strip() or self.session_root)
        output_root.mkdir(parents=True, exist_ok=True)
        out_dir = output_root / f"candidate_{now_stamp()}"
        argv = [
            "learn-candidate",
            "--video",
            str(video),
            "--output-dir",
            str(out_dir),
            "--sample-fps",
            str(float(self.learn_sample_fps_var.get())),
            "--max-frames",
            str(int(self.learn_max_frames_var.get())),
        ]

        def work() -> dict[str, Any]:
            args = learner.build_parser().parse_args(argv)
            summary = learner.learn(args)
            review_renderer.render_review(out_dir / "candidate_recipe.json", out_dir, None, out_dir / "review.html")
            return summary

        def on_success(summary: dict[str, Any]) -> None:
            candidate_path = Path(summary["candidate_recipe"])
            self.candidate_recipe_var.set(str(candidate_path))
            self.status_var.set(f"candidate 已生成: {candidate_path.name}")
            self._load_candidate(candidate_path)
            self.notebook.select(self.approval_tab)

        if not self.runner.run("learn-candidate", work, on_success=on_success):
            messagebox.showinfo("忙碌中", "当前已有任务在执行，请稍后。")

    def _collect_approval_payload(self) -> dict[str, Any]:
        if self.current_candidate is None:
            raise RuntimeError("尚未载入 candidate。")
        accepted = [sid for sid, var in self.state_keep_vars.items() if bool(var.get())]
        ignored = [sid for sid, var in self.state_keep_vars.items() if not bool(var.get())]
        return {
            "approval_version": 1,
            "expected_slot_count": int(self.current_candidate.get("slot_count", len(self.current_candidate.get("slots", []))) or 0),
            "expected_scan_phase_count": len(self.current_candidate.get("scan_phase_templates", [])),
            "accepted_display_states": accepted,
            "merged_display_states": {},
            "ignored_states": ignored,
            "expected_sequence": [],
            "override_diagnostics": bool(self.override_var.get()),
            "approval_notes": "由 dot_matrix_recipe_studio UI 生成。",
        }

    def _save_approval(self) -> bool:
        if self.current_candidate_dir is None:
            messagebox.showwarning("未载入 candidate", "请先生成或载入 candidate。")
            return False
        payload = self._collect_approval_payload()
        if not payload["accepted_display_states"]:
            messagebox.showwarning("状态为空", "至少要保留一个显示状态。")
            return False
        approval_path = self.current_candidate_dir / "approval.json"
        write_json(approval_path, payload)
        self.approval_path_var.set(str(approval_path))
        self.status_var.set("approval.json 已保存")
        self.log(f"approval 已保存: {approval_path}")
        self._update_freeze_button_state()
        return True

    def _run_freeze(self) -> None:
        if self.current_candidate_dir is None:
            messagebox.showwarning("未载入 candidate", "请先载入 candidate。")
            return
        if not self._save_approval():
            return
        approval_path = Path(self.approval_path_var.get())
        output_path = self.current_candidate_dir / "frozen_recipe.json"
        argv = [
            "freeze",
            "--candidate",
            str(Path(self.candidate_recipe_var.get())),
            "--approval",
            str(approval_path),
            "--output",
            str(output_path),
        ]

        def work() -> dict[str, Any]:
            args = learner.build_parser().parse_args(argv)
            return learner.freeze(args)

        def on_success(summary: dict[str, Any]) -> None:
            frozen_path = Path(summary["frozen_recipe"])
            self.frozen_recipe_var.set(str(frozen_path))
            self.current_frozen_dir = frozen_path.parent
            self.status_var.set(f"frozen recipe 已生成: {frozen_path.name}")
            self.log(f"frozen recipe 已生成: {frozen_path}")
            self.notebook.select(self.detect_tab)

        if not self.runner.run("freeze", work, on_success=on_success):
            messagebox.showinfo("忙碌中", "当前已有任务在执行，请稍后。")

    def _run_detect(self) -> None:
        frozen_path = Path(self.frozen_recipe_var.get().strip()) if self.frozen_recipe_var.get().strip() else None
        test_video = Path(self.test_video_var.get().strip()) if self.test_video_var.get().strip() else None
        if frozen_path is None or not frozen_path.exists():
            messagebox.showwarning("未选择 frozen recipe", "请先生成或选择 frozen_recipe.json。")
            return
        if test_video is None or not test_video.exists():
            messagebox.showwarning("未选择待测视频", "请先选择或录制待测视频。")
            return
        output_root = Path(self.output_root_var.get().strip() or self.session_root)
        output_root.mkdir(parents=True, exist_ok=True)
        out_dir = output_root / f"detect_{now_stamp()}"
        argv = [
            "detect",
            "--recipe",
            str(frozen_path),
            "--video",
            str(test_video),
            "--output-dir",
            str(out_dir),
            "--sample-fps",
            str(float(self.detect_sample_fps_var.get())),
            "--max-frames",
            str(int(self.detect_max_frames_var.get())),
        ]

        def work() -> dict[str, Any]:
            args = learner.build_parser().parse_args(argv)
            summary = learner.detect(args)
            review_renderer.render_review(frozen_path, frozen_path.parent, out_dir, out_dir / "review.html")
            return summary

        def on_success(summary: dict[str, Any]) -> None:
            self.current_detect_output = out_dir
            self.current_detect_review = out_dir / "review.html"
            self.status_var.set(f"detect 完成: {'PASS' if summary.get('overall_ok') else 'NG'}")
            self._load_detect_summary(out_dir / "detection_summary.json", out_dir / "events.json")

        if not self.runner.run("detect", work, on_success=on_success):
            messagebox.showinfo("忙碌中", "当前已有任务在执行，请稍后。")

    def _load_detect_summary(self, summary_path: Path, events_path: Path) -> None:
        if not summary_path.exists():
            return
        summary = load_json(summary_path)
        lines = [
            f"总判定: {'PASS' if summary.get('overall_ok') else 'NG'}",
            f"配方状态: {summary.get('recipe_status', '')}",
            f"事件总数: {summary.get('event_count', 0)}",
            f"事件分类: {summary.get('event_categories', {})}",
            f"trace: {summary.get('trace_csv', '')}",
            f"review_video: {summary.get('review_video', '')}",
        ]
        self.detect_summary_var.set("\n".join(lines))
        for item in self.events_tree.get_children():
            self.events_tree.delete(item)
        events_payload = load_json(events_path) if events_path.exists() else {"events": []}
        for item in events_payload.get("events", []):
            self.events_tree.insert(
                "",
                "end",
                values=(
                    item.get("frame_index", ""),
                    item.get("category", ""),
                    item.get("type", ""),
                    item.get("expected", ""),
                    item.get("matched", ""),
                    item.get("score", ""),
                ),
            )
        self.log(f"detect 输出目录: {summary_path.parent}")

    def _update_freeze_button_state(self) -> None:
        if self.current_candidate is None:
            self.freeze_button.configure(state="disabled")
            return
        accepted_count = sum(1 for var in self.state_keep_vars.values() if bool(var.get()))
        verdict = str(self.current_candidate.get("diagnostics", {}).get("verdict", "BLOCK"))
        allowed = accepted_count > 0 and (verdict == "PASS" or bool(self.override_var.get()))
        self.freeze_button.configure(state="normal" if allowed else "disabled")

    def _open_candidate_review(self) -> None:
        if self.current_candidate_review is None or not self.current_candidate_review.exists():
            messagebox.showwarning("没有 review", "当前没有可打开的 review.html。")
            return
        self._open_path(self.current_candidate_review)

    def _open_frozen_recipe(self) -> None:
        path = Path(self.frozen_recipe_var.get().strip()) if self.frozen_recipe_var.get().strip() else None
        if path is None or not path.exists():
            messagebox.showwarning("没有 frozen recipe", "当前没有可打开的 frozen_recipe.json。")
            return
        self._open_path(path)

    def _open_detect_review(self) -> None:
        if self.current_detect_review is None or not self.current_detect_review.exists():
            messagebox.showwarning("没有 detect review", "请先运行 detect。")
            return
        self._open_path(self.current_detect_review)

    def _open_detect_output(self) -> None:
        if self.current_detect_output is None or not self.current_detect_output.exists():
            messagebox.showwarning("没有输出目录", "请先运行 detect。")
            return
        self._open_path(self.current_detect_output)

    def _open_detect_review_video(self) -> None:
        if self.current_detect_output is None or not self.current_detect_output.exists():
            messagebox.showwarning("没有 review 视频", "请先运行 detect。")
            return
        path = self.current_detect_output / "review_video.mp4"
        if not path.exists():
            messagebox.showwarning("没有 review 视频", "当前 detect 输出里还没有 review_video.mp4。")
            return
        self._open_path(path)

    def _open_path(self, path: Path) -> None:
        try:
            open_local_path(path)
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def post_ui(self, fn: Callable[[], None]) -> None:
        self.ui_queue.put(fn)

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{stamp}] {message}")

    def _drain_log_queue(self) -> None:
        messages: list[str] = []
        while True:
            try:
                messages.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        if messages:
            self.log_text.insert("end", "\n".join(messages) + "\n")
            self.log_text.see("end")
        self.after(120, self._drain_log_queue)

    def _drain_ui_queue(self) -> None:
        while True:
            try:
                fn = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            fn()
        self.after(80, self._drain_ui_queue)

    def _refresh_preview(self) -> None:
        self.camera_panel.update_preview()
        self.after(80, self._refresh_preview)

    def _on_close(self) -> None:
        try:
            self.camera_panel.stop_preview()
        finally:
            self.destroy()


def main() -> None:
    app = DotMatrixRecipeStudio()
    app.mainloop()


if __name__ == "__main__":
    main()
