from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
import tkinter as tk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from desktop_app import (
    CameraInfo,
    HikvisionCameraStream,
    OpenCVCameraStream,
    enumerate_hikvision_cameras,
    enumerate_opencv_cameras,
)


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models" / "seven_segment"
OUTPUT_DIR = ROOT / "output" / "sevenseg_station"
DEFAULT_SOURCE_VIDEO = Path(r"D:\jiemi\七段管\Video_20260515112415516.avi")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def open_path(path: Path) -> None:
    if path.exists():
        os.startfile(str(path))


class SevenSegmentStation(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("七段管产线验证端")
        self.geometry("1500x920")
        self.minsize(1180, 760)
        self.configure(bg="#101010")

        self.cameras: list[CameraInfo] = []
        self.stream: OpenCVCameraStream | HikvisionCameraStream | None = None
        self.previewing = False
        self.detecting = False
        self.auto_detecting = False
        self.ng_latched = False
        self.current_model_dir: Path | None = self.find_latest_model()
        self.last_report_dir: Path | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self.model_templates: list[dict[str, Any]] = []
        self.model_sequence: list[dict[str, Any]] = []
        self.camera_warning_shown = False

        self.build_ui()
        self.refresh_models()
        self.refresh_cameras()
        self.after(80, self.preview_loop)

    # UI -----------------------------------------------------------------
    def build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=8)
        style.configure("TLabel", background="#101010", foreground="white", font=("Microsoft YaHei UI", 10))
        style.configure("TCombobox", font=("Microsoft YaHei UI", 10))

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = tk.Frame(self, bg="#181818", height=76)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(2, weight=1)

        ttk.Label(top, text="相机").grid(row=0, column=0, padx=(16, 6), pady=14)
        self.camera_combo = ttk.Combobox(top, width=36, state="readonly")
        self.camera_combo.grid(row=0, column=1, padx=4, pady=14)
        ttk.Button(top, text="刷新相机", command=self.refresh_cameras).grid(row=0, column=2, sticky="w", padx=8)

        ttk.Label(top, text="模型").grid(row=0, column=3, padx=(16, 6), pady=14)
        self.model_combo = ttk.Combobox(top, width=42, state="readonly")
        self.model_combo.grid(row=0, column=4, padx=4, pady=14)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self.select_model_from_combo())
        ttk.Button(top, text="打开工程师页", command=self.open_engineer_window).grid(row=0, column=5, padx=8)

        body = tk.Frame(self, bg="#101010")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        body.rowconfigure(0, weight=1)

        self.image_label = tk.Label(body, bg="black")
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

        side = tk.Frame(body, bg="#151515", width=340)
        side.grid(row=0, column=1, sticky="ns", padx=(0, 16), pady=16)
        side.grid_propagate(False)

        self.status_label = tk.Label(
            side,
            text="READY",
            bg="#4e4e4e",
            fg="white",
            font=("Microsoft YaHei UI", 42, "bold"),
            height=2,
        )
        self.status_label.pack(fill="x", padx=16, pady=(16, 10))
        self.detail_label = tk.Label(
            side,
            text="选择相机和模型后开始预览",
            bg="#151515",
            fg="white",
            font=("Microsoft YaHei UI", 14, "bold"),
            wraplength=300,
            justify="left",
        )
        self.detail_label.pack(fill="x", padx=16, pady=(0, 18))

        for text, command in [
            ("开始预览", self.start_preview),
            ("停止预览", self.stop_preview),
            ("相机检测一次", self.detect_once),
            ("导入视频检测", self.detect_imported_video),
            ("自动检测", self.toggle_auto_detect),
            ("确认并继续", self.confirm_and_continue),
            ("打开报告", self.open_last_report),
        ]:
            ttk.Button(side, text=text, command=command).pack(fill="x", padx=16, pady=6)

        settings = tk.LabelFrame(side, text="检测设置", bg="#151515", fg="white", font=("Microsoft YaHei UI", 10, "bold"))
        settings.pack(fill="x", padx=16, pady=18)
        tk.Label(settings, text="采集秒数", bg="#151515", fg="white").grid(row=0, column=0, padx=8, pady=8, sticky="w")
        self.detect_seconds_var = tk.DoubleVar(value=20.0)
        tk.Spinbox(settings, from_=3, to=90, increment=1, textvariable=self.detect_seconds_var, width=8).grid(row=0, column=1, padx=8, pady=8)
        self.alarm_var = tk.BooleanVar(value=True)
        tk.Checkbutton(settings, text="NG报警", variable=self.alarm_var, bg="#151515", fg="white", selectcolor="#151515").grid(row=1, column=0, columnspan=2, padx=8, pady=8, sticky="w")

        self.log_text = tk.Text(side, height=12, bg="#0b0b0b", fg="#d0d0d0", insertbackground="white")
        self.log_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    def set_status(self, title: str, detail: str, color: str) -> None:
        self.status_label.configure(text=title, bg=color)
        self.detail_label.configure(text=detail)

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{stamp}] {message}\n")
        self.log_text.see("end")

    # Camera -------------------------------------------------------------
    def refresh_cameras(self) -> None:
        cameras: list[CameraInfo] = []
        try:
            cameras.extend(enumerate_hikvision_cameras())
        except Exception as exc:
            self.log(f"海康相机枚举失败: {exc}")
        try:
            cameras.extend(enumerate_opencv_cameras())
        except Exception as exc:
            self.log(f"OpenCV相机枚举失败: {exc}")
        self.cameras = cameras
        self.camera_combo["values"] = [c.label for c in cameras]
        if cameras and self.camera_combo.current() < 0:
            self.camera_combo.current(0)
        if cameras:
            self.camera_warning_shown = False
            self.log(f"相机数量: {len(cameras)}")
        else:
            self.log("未发现相机；可使用“导入视频检测”测试")
            self.set_status("READY", "未发现相机，可导入视频测试", "#4e4e4e")

    def selected_camera(self) -> CameraInfo | None:
        label = self.camera_combo.get()
        for camera in self.cameras:
            if camera.label == label:
                return camera
        return self.cameras[0] if self.cameras else None

    def start_preview(self) -> None:
        if self.previewing:
            return
        camera = self.selected_camera()
        if camera is None:
            self.set_status("NO CAM", "没有可用相机；请刷新相机或导入视频检测", "#8a6a1f")
            if not self.camera_warning_shown:
                self.log("没有可用相机，已跳过预览")
                self.camera_warning_shown = True
            return
        try:
            if camera.kind == "hikvision":
                self.stream = HikvisionCameraStream(camera, self.log)
            else:
                self.stream = OpenCVCameraStream(camera, self.log)
            self.stream.start()
            self.previewing = True
            self.set_status("LIVE", "预览中", "#375f96")
        except Exception as exc:
            messagebox.showerror("相机", str(exc))

    def stop_preview(self) -> None:
        self.auto_detecting = False
        self.previewing = False
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
        self.set_status("READY", "预览已停止", "#4e4e4e")

    def latest_frame(self) -> np.ndarray | None:
        return self.stream.get_latest_frame() if self.stream is not None else None

    def preview_loop(self) -> None:
        if self.previewing and not self.ng_latched:
            frame = self.latest_frame()
            if frame is not None:
                self.show_image(frame)
        if self.auto_detecting and not self.detecting and not self.ng_latched:
            if self.stream is None:
                self.auto_detecting = False
                self.set_status("NO CAM", "自动检测已停止：没有相机", "#8a6a1f")
            else:
                self.detect_once(auto=True)
        self.after(180, self.preview_loop)

    def show_image(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(200, self.image_label.winfo_width())
        max_h = max(200, self.image_label.winfo_height())
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.preview_image)

    # Models -------------------------------------------------------------
    def find_latest_model(self) -> Path | None:
        if not MODELS_DIR.exists():
            return None
        candidates = [p for p in MODELS_DIR.iterdir() if (p / "templates.json").exists()]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def refresh_models(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        models = [p for p in sorted(MODELS_DIR.iterdir()) if (p / "templates.json").exists()]
        self.model_combo["values"] = [p.name for p in models]
        if self.current_model_dir and self.current_model_dir.name in self.model_combo["values"]:
            self.model_combo.set(self.current_model_dir.name)
        elif models:
            self.current_model_dir = models[-1]
            self.model_combo.set(models[-1].name)

    def select_model_from_combo(self) -> None:
        name = self.model_combo.get()
        path = MODELS_DIR / name
        if (path / "templates.json").exists():
            self.current_model_dir = path
            self.log(f"已选择模型: {name}")

    def bootstrap_model_metadata(self, model_dir: Path, source_video: Path | None = None) -> None:
        templates_path = model_dir / "templates.json"
        if not templates_path.exists():
            return
        payload = load_json(templates_path)
        for tmpl in payload.get("templates", []):
            tmpl.setdefault("name", tmpl.get("id", ""))
            tmpl.setdefault("enabled", True)
            tmpl.setdefault("merged_to", "")
        write_json(templates_path, payload)

        summary = load_json(model_dir / "summary.json") if (model_dir / "summary.json").exists() else {}
        write_json(
            model_dir / "model.json",
            {
                "name": model_dir.name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "source_video": str(source_video) if source_video else "",
                "roi": summary.get("roi", {}),
                "parameters": summary.get("parameters", {}),
                "templates_path": "templates.json",
            },
        )

    # Modeling -----------------------------------------------------------
    def open_engineer_window(self) -> None:
        EngineerWindow(self)

    def model_from_video(self) -> None:
        video = filedialog.askopenfilename(
            title="选择标准视频",
            filetypes=[("Video", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")],
            initialdir=str(DEFAULT_SOURCE_VIDEO.parent if DEFAULT_SOURCE_VIDEO.exists() else ROOT),
        )
        if not video:
            return
        name = simpledialog.askstring("模型名称", "请输入模型名称", initialvalue=now_name("sevenseg"))
        if not name:
            return
        self.run_modeling(Path(video), MODELS_DIR / name)

    def model_from_camera(self, seconds: float) -> None:
        if not self.previewing:
            self.start_preview()
        if self.stream is None:
            return
        name = simpledialog.askstring("模型名称", "请输入模型名称", initialvalue=now_name("camera_model"))
        if not name:
            return
        video_path = OUTPUT_DIR / "recordings" / f"{name}.avi"

        def worker() -> None:
            try:
                self.after(0, lambda: self.set_status("REC", f"录制标准流程 {seconds:.0f} 秒", "#967023"))
                self.record_video(video_path, seconds)
                self.run_modeling(video_path, MODELS_DIR / name)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("建模失败", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def record_video(self, out_path: Path, seconds: float) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer: cv2.VideoWriter | None = None
        frame_count = 0
        deadline = time.time() + max(1.0, seconds)
        while time.time() < deadline:
            frame = self.latest_frame()
            if frame is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
                writer.write(frame)
                frame_count += 1
            time.sleep(1 / 30)
        if writer is not None:
            writer.release()
        if frame_count <= 0:
            raise RuntimeError("没有录到相机画面")

    def run_modeling(self, video: Path, model_dir: Path) -> None:
        def worker() -> None:
            try:
                self.after(0, lambda: self.set_status("MODEL", "建模中，请稍候", "#967023"))
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "extract_scan_frame_templates.py"),
                    "--video",
                    str(video),
                    "--out",
                    str(model_dir),
                ]
                subprocess.run(cmd, cwd=str(ROOT), check=True)
                self.bootstrap_model_metadata(model_dir, video)
                self.current_model_dir = model_dir
                self.after(0, self.refresh_models)
                self.after(0, lambda: self.set_status("READY", f"建模完成: {model_dir.name}", "#4e4e4e"))
                self.after(0, lambda: TemplateEditor(self, model_dir))
            except Exception as exc:
                self.after(0, lambda: self.set_status("ERROR", f"建模失败: {exc}", "#aa1e1e"))

        threading.Thread(target=worker, daemon=True).start()

    # Detection ----------------------------------------------------------
    def detect_once(self, auto: bool = False) -> None:
        if self.detecting:
            return
        if self.current_model_dir is None:
            messagebox.showwarning("模型", "请先选择或创建模型")
            return
        if not self.previewing:
            self.start_preview()
        if self.stream is None:
            self.set_status("NO CAM", "没有相机，无法相机检测；可导入视频检测", "#8a6a1f")
            return
        self.detecting = True
        seconds = max(3.0, float(self.detect_seconds_var.get()))

        def worker() -> None:
            try:
                run_name = now_name("detect")
                video = OUTPUT_DIR / "captures" / f"{run_name}.avi"
                report = OUTPUT_DIR / "reports" / run_name
                self.after(0, lambda: self.set_status("WAIT", f"采集中 {seconds:.0f} 秒", "#967023"))
                self.record_video(video, seconds)
                self.run_detection_video(video, report)
            except Exception as exc:
                self.after(0, lambda: self.set_status("ERROR", f"检测失败: {exc}", "#aa1e1e"))
            finally:
                self.detecting = False

        threading.Thread(target=worker, daemon=True).start()

    def detect_imported_video(self) -> None:
        if self.detecting:
            return
        if self.current_model_dir is None:
            messagebox.showwarning("模型", "请先选择或创建模型")
            return
        video = filedialog.askopenfilename(
            title="选择待测视频",
            filetypes=[("Video", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")],
            initialdir=str(DEFAULT_SOURCE_VIDEO.parent if DEFAULT_SOURCE_VIDEO.exists() else ROOT),
        )
        if not video:
            return
        self.detecting = True

        def worker() -> None:
            try:
                run_name = now_name("import_video_detect")
                report = OUTPUT_DIR / "reports" / run_name
                self.after(0, lambda: self.set_status("WAIT", "导入视频检测中", "#967023"))
                self.run_detection_video(Path(video), report)
            except Exception as exc:
                self.after(0, lambda: self.set_status("ERROR", f"视频检测失败: {exc}", "#aa1e1e"))
            finally:
                self.detecting = False

        threading.Thread(target=worker, daemon=True).start()

    def run_detection_video(self, video: Path, report: Path) -> None:
        if self.current_model_dir is None:
            raise RuntimeError("未选择模型")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "detect_scan_frame_sequence.py"),
            "--video",
            str(video),
            "--model",
            str(self.current_model_dir / "templates.json"),
            "--out",
            str(report),
            "--min-template-count",
            "3",
            "--tolerant-match-threshold",
            "0.58",
            "--overall-mode",
            "visual-or-sequence",
        ]
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        self.last_report_dir = report
        summary = load_json(report / "summary.json")
        ok = summary.get("overall_status") == "PASS"
        if ok:
            self.compact_pass_report(report)
        self.after(0, lambda: self.apply_detection_result(ok, summary, report))

    def apply_detection_result(self, ok: bool, summary: dict[str, Any], report: Path) -> None:
        matched = int(summary.get("matched_frames", 0))
        unknown = int(summary.get("unknown_frames", 0))
        events = int(summary.get("event_count", 0))
        if ok:
            self.ng_latched = False
            self.set_status("PASS", f"匹配帧={matched} 未知={unknown}", "#1e8c46")
        else:
            self.ng_latched = True
            self.auto_detecting = False
            self.set_status("NG", f"未知={unknown} 异常={events}，已停住待确认", "#be1e1e")
            if self.alarm_var.get():
                self.bell()
        result_image = report / "timeline_overview.jpg"
        if result_image.exists():
            image = cv2.imread(str(result_image))
            if image is not None:
                self.show_image(image)

    def compact_pass_report(self, report: Path) -> None:
        for name in ["playback_data.json", "playback_data.js", "abstract_player.html"]:
            path = report / name
            if path.exists():
                path.unlink()
        frames_dir = report / "playback_frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)

    def toggle_auto_detect(self) -> None:
        if self.ng_latched:
            return
        if self.stream is None:
            if not self.previewing:
                self.start_preview()
            if self.stream is None:
                self.auto_detecting = False
                self.set_status("NO CAM", "没有相机，自动检测未启动；可导入视频检测", "#8a6a1f")
                return
        self.auto_detecting = not self.auto_detecting
        self.set_status("AUTO" if self.auto_detecting else "LIVE", "自动检测中" if self.auto_detecting else "自动检测已停止", "#375f96")

    def confirm_and_continue(self) -> None:
        self.ng_latched = False
        self.set_status("LIVE", "已确认，可继续检测", "#375f96")

    def open_last_report(self) -> None:
        if self.last_report_dir is None:
            self.set_status("READY", "暂无检测报告", "#4e4e4e")
            return
        player = self.last_report_dir / "abstract_player.html"
        if player.exists():
            webbrowser.open(player.as_uri())
        else:
            open_path(self.last_report_dir)


class EngineerWindow(tk.Toplevel):
    def __init__(self, app: SevenSegmentStation) -> None:
        super().__init__(app)
        self.app = app
        self.title("七段管工程师工具")
        self.geometry("520x260")
        self.configure(bg="#202020")

        ttk.Button(self, text="从视频建模", command=app.model_from_video).pack(fill="x", padx=22, pady=(24, 8))
        ttk.Button(self, text="从相机定时建模", command=lambda: app.model_from_camera(app.detect_seconds_var.get())).pack(fill="x", padx=22, pady=8)
        ttk.Button(self, text="编辑当前模板", command=self.edit_current).pack(fill="x", padx=22, pady=8)
        ttk.Button(self, text="打开当前模型目录", command=self.open_model_dir).pack(fill="x", padx=22, pady=8)

    def edit_current(self) -> None:
        if self.app.current_model_dir is None:
            messagebox.showwarning("模型", "没有当前模型")
            return
        TemplateEditor(self.app, self.app.current_model_dir)

    def open_model_dir(self) -> None:
        if self.app.current_model_dir:
            open_path(self.app.current_model_dir)


class TemplateEditor(tk.Toplevel):
    def __init__(self, app: SevenSegmentStation, model_dir: Path) -> None:
        super().__init__(app)
        self.app = app
        self.model_dir = model_dir
        self.templates_path = model_dir / "templates.json"
        self.payload = load_json(self.templates_path)
        self.templates: list[dict[str, Any]] = self.payload.get("templates", [])
        self.sequence: list[dict[str, Any]] = self.payload.get("sequence", [])
        self.photo: ImageTk.PhotoImage | None = None

        self.title(f"模板审核 - {model_dir.name}")
        self.geometry("1180x760")
        self.configure(bg="#121212")
        self.build_ui()
        self.refresh_list()

    def build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = tk.Frame(self, bg="#1c1c1c", width=360)
        left.grid(row=0, column=0, sticky="ns")
        left.grid_propagate(False)
        self.listbox = tk.Listbox(left, bg="#0b0b0b", fg="white", selectbackground="#2b6cb0", font=("Consolas", 11))
        self.listbox.pack(fill="both", expand=True, padx=12, pady=12)
        self.listbox.bind("<<ListboxSelect>>", lambda _e: self.show_selected())

        right = tk.Frame(self, bg="#121212")
        right.grid(row=0, column=1, sticky="nsew", padx=14, pady=14)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        form = tk.Frame(right, bg="#121212")
        form.grid(row=0, column=0, sticky="ew")
        tk.Label(form, text="模板名称", bg="#121212", fg="white").grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar()
        tk.Entry(form, textvariable=self.name_var, width=36).grid(row=0, column=1, padx=8)
        self.enabled_var = tk.BooleanVar(value=True)
        tk.Checkbutton(form, text="启用", variable=self.enabled_var, bg="#121212", fg="white", selectcolor="#121212").grid(row=0, column=2, padx=8)
        ttk.Button(form, text="保存名称/启用", command=self.apply_fields).grid(row=0, column=3, padx=8)

        self.image_label = tk.Label(right, bg="black")
        self.image_label.grid(row=1, column=0, sticky="nsew", pady=12)

        buttons = tk.Frame(right, bg="#121212")
        buttons.grid(row=2, column=0, sticky="ew")
        for text, cmd in [
            ("上移", lambda: self.move_sequence(-1)),
            ("下移", lambda: self.move_sequence(1)),
            ("删除/禁用", self.disable_selected),
            ("合并到...", self.merge_selected),
            ("保存审核模型", self.save),
            ("打开建模报告", self.open_review),
        ]:
            ttk.Button(buttons, text=text, command=cmd).pack(side="left", padx=5, pady=5)

    def template_by_id(self, template_id: str) -> dict[str, Any] | None:
        for tmpl in self.templates:
            if str(tmpl.get("id")) == template_id:
                return tmpl
        return None

    def sequence_ids(self) -> list[str]:
        return [str(s.get("template_id", "")) for s in self.sequence]

    def selected_index(self) -> int | None:
        sel = self.listbox.curselection()
        return int(sel[0]) if sel else None

    def selected_template_id(self) -> str | None:
        idx = self.selected_index()
        ids = self.sequence_ids()
        return ids[idx] if idx is not None and idx < len(ids) else None

    def refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for index, template_id in enumerate(self.sequence_ids()):
            tmpl = self.template_by_id(template_id) or {}
            status = "ON" if tmpl.get("enabled", True) and not tmpl.get("merged_to") else "OFF"
            name = tmpl.get("name") or template_id
            self.listbox.insert("end", f"{index + 1:03d} {status} {template_id} {name}")
        if self.sequence:
            self.listbox.selection_set(0)
            self.show_selected()

    def show_selected(self) -> None:
        template_id = self.selected_template_id()
        if not template_id:
            return
        tmpl = self.template_by_id(template_id)
        if not tmpl:
            return
        self.name_var.set(str(tmpl.get("name") or template_id))
        self.enabled_var.set(bool(tmpl.get("enabled", True)) and not bool(tmpl.get("merged_to")))
        image_path = self.model_dir / str(tmpl.get("image", ""))
        if image_path.exists():
            image = Image.open(image_path)
            image.thumbnail((760, 520), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)

    def apply_fields(self) -> None:
        template_id = self.selected_template_id()
        tmpl = self.template_by_id(template_id) if template_id else None
        if not tmpl:
            return
        tmpl["name"] = self.name_var.get().strip() or template_id
        tmpl["enabled"] = bool(self.enabled_var.get())
        if tmpl["enabled"]:
            tmpl["merged_to"] = ""
        self.refresh_list()

    def disable_selected(self) -> None:
        template_id = self.selected_template_id()
        tmpl = self.template_by_id(template_id) if template_id else None
        if not tmpl:
            return
        tmpl["enabled"] = False
        self.sequence = [s for s in self.sequence if str(s.get("template_id")) != template_id]
        self.refresh_list()

    def merge_selected(self) -> None:
        source_id = self.selected_template_id()
        if not source_id:
            return
        target_id = simpledialog.askstring("合并模板", f"把 {source_id} 合并到哪个模板ID？", parent=self)
        if not target_id or target_id == source_id or self.template_by_id(target_id) is None:
            return
        source = self.template_by_id(source_id)
        if source:
            source["enabled"] = False
            source["merged_to"] = target_id
        for seg in self.sequence:
            if str(seg.get("template_id")) == source_id:
                seg["template_id"] = target_id
        self.collapse_adjacent_sequence()
        self.refresh_list()

    def move_sequence(self, direction: int) -> None:
        idx = self.selected_index()
        if idx is None:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self.sequence):
            return
        self.sequence[idx], self.sequence[new_idx] = self.sequence[new_idx], self.sequence[idx]
        self.refresh_list()
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(new_idx)
        self.show_selected()

    def collapse_adjacent_sequence(self) -> None:
        collapsed: list[dict[str, Any]] = []
        last = ""
        for seg in self.sequence:
            tid = str(seg.get("template_id", ""))
            if tid and tid != last:
                collapsed.append(seg)
                last = tid
        self.sequence = collapsed

    def save(self) -> None:
        self.apply_fields()
        self.payload["templates"] = self.templates
        self.payload["sequence"] = self.sequence
        write_json(self.templates_path, self.payload)
        model_json = load_json(self.model_dir / "model.json") if (self.model_dir / "model.json").exists() else {}
        model_json["reviewed_at"] = datetime.now().isoformat(timespec="seconds")
        model_json["enabled_template_count"] = sum(1 for t in self.templates if t.get("enabled", True) and not t.get("merged_to"))
        model_json["sequence_count"] = len(self.sequence)
        write_json(self.model_dir / "model.json", model_json)
        self.app.current_model_dir = self.model_dir
        self.app.refresh_models()
        messagebox.showinfo("保存", "审核模型已保存")

    def open_review(self) -> None:
        review = self.model_dir / "review.html"
        if review.exists():
            webbrowser.open(review.as_uri())
        else:
            open_path(self.model_dir)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = SevenSegmentStation()
    app.mainloop()


if __name__ == "__main__":
    main()
