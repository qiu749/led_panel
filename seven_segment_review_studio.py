from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def open_local_path(path: Path) -> None:
    target = path.resolve()
    try:
        os.startfile(str(target))
        return
    except Exception:
        pass
    webbrowser.open(target.as_uri())


class SevenSegmentReviewStudio(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("七段管 / 显示条 Demo")
        self.geometry("1380x900")
        self.minsize(1200, 780)

        self.video_var = tk.StringVar(value=r"D:\jiemi\Video_20260430154222007.avi")
        self.output_root = ROOT / "output" / "seven_segment_sessions" / f"session_{now_stamp()}"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.output_var = tk.StringVar(value=str(self.output_root))
        self.sample_fps_var = tk.DoubleVar(value=7.5)
        self.max_frames_var = tk.IntVar(value=0)
        self.state_mode_var = tk.StringVar(value="layout")
        self.grid_cols_var = tk.IntVar(value=48)
        self.grid_rows_var = tk.IntVar(value=12)
        self.window_size_var = tk.IntVar(value=5)
        self.min_on_ratio_var = tk.DoubleVar(value=0.4)
        self.status_var = tk.StringVar(value="就绪")
        self.summary_var = tk.StringVar(value="未运行")

        self.current_run_dir: Path | None = None
        self.current_review_path: Path | None = None
        self.image_refs: dict[str, ImageTk.PhotoImage | None] = {}
        self.process: subprocess.Popen[str] | None = None
        self.running = False

        self._build_ui()

    def _build_ui(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", padx=10, pady=10)
        ttk.Label(header, text="视频").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(header, textvariable=self.video_var).grid(row=0, column=1, columnspan=5, sticky="ew", padx=6, pady=6)
        ttk.Button(header, text="选择视频", command=self._pick_video).grid(row=0, column=6, padx=6, pady=6)

        ttk.Label(header, text="输出目录").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(header, textvariable=self.output_var).grid(row=1, column=1, columnspan=5, sticky="ew", padx=6, pady=6)
        ttk.Button(header, text="选择目录", command=self._pick_output).grid(row=1, column=6, padx=6, pady=6)

        ttk.Label(header, text="模式").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(header, textvariable=self.state_mode_var, values=["layout", "visual_grid"], state="readonly", width=14).grid(row=2, column=1, sticky="w", padx=6, pady=6)
        ttk.Label(header, text="sample_fps").grid(row=2, column=2, sticky="e", padx=6, pady=6)
        ttk.Spinbox(header, from_=1.0, to=30.0, increment=0.5, textvariable=self.sample_fps_var, width=8).grid(row=2, column=3, sticky="w", padx=6, pady=6)
        ttk.Label(header, text="max_frames").grid(row=2, column=4, sticky="e", padx=6, pady=6)
        ttk.Spinbox(header, from_=0, to=2000, increment=20, textvariable=self.max_frames_var, width=8).grid(row=2, column=5, sticky="w", padx=6, pady=6)
        ttk.Button(header, text="开始检测", command=self._run_pipeline).grid(row=2, column=6, padx=6, pady=6)

        ttk.Label(header, text="grid").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        ttk.Spinbox(header, from_=8, to=128, increment=1, textvariable=self.grid_cols_var, width=8).grid(row=3, column=1, sticky="w", padx=(6, 2), pady=6)
        ttk.Spinbox(header, from_=4, to=64, increment=1, textvariable=self.grid_rows_var, width=8).grid(row=3, column=2, sticky="w", padx=(2, 6), pady=6)
        ttk.Label(header, text="window").grid(row=3, column=3, sticky="e", padx=6, pady=6)
        ttk.Spinbox(header, from_=1, to=15, increment=1, textvariable=self.window_size_var, width=8).grid(row=3, column=4, sticky="w", padx=6, pady=6)
        ttk.Label(header, text="min_on_ratio").grid(row=3, column=5, sticky="e", padx=6, pady=6)
        ttk.Spinbox(header, from_=0.1, to=1.0, increment=0.05, textvariable=self.min_on_ratio_var, width=8).grid(row=3, column=6, sticky="w", padx=6, pady=6)

        header.columnconfigure(1, weight=1)
        header.columnconfigure(2, weight=0)
        header.columnconfigure(3, weight=0)
        header.columnconfigure(4, weight=0)
        header.columnconfigure(5, weight=0)

        actions = ttk.Frame(self)
        actions.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(actions, text="打开 review", command=self._open_review).pack(side="left", padx=6)
        ttk.Button(actions, text="打开输出目录", command=self._open_output).pack(side="left", padx=6)
        ttk.Button(actions, text="打开 trace.csv", command=self._open_trace).pack(side="left", padx=6)
        ttk.Label(actions, textvariable=self.status_var).pack(side="right", padx=6)

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=2)
        body.add(right, weight=3)

        summary_frame = ttk.LabelFrame(left, text="摘要")
        summary_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(summary_frame, textvariable=self.summary_var, justify="left", anchor="nw").pack(fill="x", padx=8, pady=8)

        log_frame = ttk.LabelFrame(left, text="运行日志")
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

        preview_frame = ttk.LabelFrame(right, text="预览")
        preview_frame.pack(fill="both", expand=True)
        grid = ttk.Frame(preview_frame)
        grid.pack(fill="both", expand=True, padx=8, pady=8)
        self.roi_label = ttk.Label(grid, text="ROI 标记图")
        self.roi_label.grid(row=0, column=0, padx=8, pady=8)
        self.warp_label = ttk.Label(grid, text="Warp 矫正图")
        self.warp_label.grid(row=0, column=1, padx=8, pady=8)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    def _pick_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("Video Files", "*.avi;*.mp4;*.mov;*.mkv"), ("All Files", "*.*")],
        )
        if path:
            self.video_var.set(path)

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(initialdir=self.output_var.get() or str(ROOT))
        if path:
            self.output_var.set(path)

    def _log(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{stamp}] {text}\n")
        self.log_text.see("end")

    def _run_pipeline(self) -> None:
        if self.running:
            messagebox.showinfo("忙碌中", "当前已有任务在执行。")
            return
        video = Path(self.video_var.get().strip())
        if not video.exists():
            messagebox.showwarning("视频不存在", f"找不到视频:\n{video}")
            return

        out_dir = Path(self.output_var.get().strip()) / f"run_{now_stamp()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.current_run_dir = out_dir
        self.current_review_path = out_dir / "review.html"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_display_pipeline.py"),
            "--video",
            str(video),
            "--out",
            str(out_dir),
            "--sample-fps",
            str(float(self.sample_fps_var.get())),
            "--state-mode",
            str(self.state_mode_var.get()),
            "--grid-cols",
            str(int(self.grid_cols_var.get())),
            "--grid-rows",
            str(int(self.grid_rows_var.get())),
            "--window-size",
            str(int(self.window_size_var.get())),
            "--min-on-ratio",
            str(float(self.min_on_ratio_var.get())),
        ]
        max_frames = int(self.max_frames_var.get())
        if max_frames > 0:
            cmd.extend(["--max-frames", str(max_frames)])

        self.running = True
        self.status_var.set("运行中")
        self.summary_var.set("运行中...")
        self.log_text.delete("1.0", "end")
        self._log("启动七段管管道")
        self._log(" ".join(cmd))

        def worker() -> None:
            try:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self.after(0, lambda text=line.rstrip(): self._log(text))
                code = self.process.wait()
                if code != 0:
                    raise RuntimeError(f"管道退出码 {code}")
                self.after(0, lambda: self._load_results(out_dir))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("执行失败", str(exc)))
                self.after(0, lambda: self.status_var.set("执行失败"))
            finally:
                self.process = None
                self.running = False

        threading.Thread(target=worker, name="seven-segment-pipeline", daemon=True).start()

    def _load_results(self, out_dir: Path) -> None:
        summary_path = out_dir / "summary.json"
        if not summary_path.exists():
            self.status_var.set("完成但缺少 summary.json")
            return
        summary = load_json(summary_path)
        lines = [
            f"总判定: {'PASS' if summary.get('overall_ok') else 'NG'}",
            f"sampled_count: {summary.get('sampled_count', 0)}",
            f"roi_reused_count: {summary.get('roi_reused_count', 0)}",
            f"fused_count: {summary.get('fused_count', 0)}",
            f"key_state_count: {summary.get('key_state_count', 0)}",
            f"event_count: {summary.get('event_count', 0)}",
            f"first_error: {summary.get('first_error', None)}",
        ]
        self.summary_var.set("\n".join(lines))
        self.status_var.set("完成")
        self._log(f"输出目录: {out_dir}")
        self._load_preview_images(out_dir)

    def _load_preview_images(self, out_dir: Path) -> None:
        roi_marks = sorted((out_dir / "debug" / "roi_marks").glob("*.jpg"))
        warped = sorted((out_dir / "debug" / "warped").glob("*.jpg"))
        self._set_image(self.roi_label, roi_marks[0] if roi_marks else None, "roi")
        self._set_image(self.warp_label, warped[0] if warped else None, "warp")

    def _set_image(self, label: ttk.Label, path: Path | None, key: str) -> None:
        if path is None or not path.exists():
            label.configure(image="", text="没有预览图")
            self.image_refs[key] = None
            return
        try:
            pil = Image.open(path)
            pil.thumbnail((500, 320))
            image = ImageTk.PhotoImage(pil)
            self.image_refs[key] = image
            label.configure(image=image, text="")
        except Exception as exc:
            label.configure(image="", text=f"加载失败: {exc}")
            self.image_refs[key] = None

    def _open_review(self) -> None:
        if self.current_review_path is None or not self.current_review_path.exists():
            messagebox.showwarning("没有 review", "请先运行一遍七段管检测。")
            return
        open_local_path(self.current_review_path)

    def _open_output(self) -> None:
        if self.current_run_dir is None or not self.current_run_dir.exists():
            messagebox.showwarning("没有输出", "请先运行一遍七段管检测。")
            return
        open_local_path(self.current_run_dir)

    def _open_trace(self) -> None:
        if self.current_run_dir is None:
            messagebox.showwarning("没有 trace", "请先运行一遍七段管检测。")
            return
        trace = self.current_run_dir / "state_trace.csv"
        if not trace.exists():
            messagebox.showwarning("没有 trace", "当前输出目录里没有 state_trace.csv。")
            return
        open_local_path(trace)


def main() -> None:
    app = SevenSegmentReviewStudio()
    app.mainloop()


if __name__ == "__main__":
    main()
