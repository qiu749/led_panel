"""
running_led_core — 跑马灯 LED 状态机检测核心算法
"""
from __future__ import annotations
import base64, json, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import cv2
import numpy as np

DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "on_threshold": 0.35, "dedupe_similarity": 0.985, "state_similarity": 0.95,
    "min_stable_frames": 2, "transition_tolerance_frames": 3,
    "hold_timeout_factor": 2.5, "max_unknown_frames": 3,
}

@dataclass
class KeyState:
    id: str; index: int; on_count: int; mask_rle: str; median_hold_frames: int; thumbnail: str = ""
    @property
    def mask(self) -> np.ndarray: return rle_decode(self.mask_rle)

@dataclass
class PanelInfo:
    rows: int; cols: int; slot_count: int; fixed_size: tuple[int, int]

@dataclass
class KeyStatesModel:
    version: int = 1; created_at: str = ""; source_video: str = ""; sample_fps: float = 15.0
    panel: PanelInfo | None = None
    thresholds: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    states: list[KeyState] = field(default_factory=list)
    sequence: list[str] = field(default_factory=list)
    def to_dict(self) -> dict:
        return {"version": self.version, "created_at": self.created_at, "source_video": self.source_video,
                "sample_fps": self.sample_fps, "panel": asdict(self.panel) if self.panel else None,
                "thresholds": self.thresholds, "states": [asdict(s) for s in self.states],
                "sequence": self.sequence}
    @classmethod
    def from_dict(cls, d: dict) -> KeyStatesModel:
        return cls(version=d.get("version",1), created_at=d.get("created_at",""),
                   source_video=d.get("source_video",""), sample_fps=d.get("sample_fps",15.0),
                   panel=PanelInfo(**d["panel"]) if d.get("panel") else None,
                   thresholds=d.get("thresholds",dict(DEFAULT_THRESHOLDS)),
                   states=[KeyState(**s) for s in d.get("states",[])],
                   sequence=d.get("sequence",[]))

@dataclass
class FrameResult:
    frame_index: int; time_sec: float; expected_state: str; matched_state: str; next_state: str
    similarity_current: float; similarity_next: float; transition: str; hold_frames: int; ok: bool; event_type: str = ""

@dataclass
class Event:
    frame_index: int; time_sec: float; type: str; expected_state: str; matched_state: str; message: str

def rle_encode(mask: np.ndarray) -> str:
    if len(mask) == 0: return ""
    flat = mask.ravel().astype(np.int8)
    changes = np.diff(flat, prepend=-1, append=-1)
    runs = np.where(changes != 0)[0]
    lengths = np.diff(runs); values = flat[runs[:-1]]
    return ",".join(f"{int(v)}:{int(l)}" for v, l in zip(values, lengths))

def rle_decode(rle_str: str) -> np.ndarray:
    if not rle_str: return np.array([], dtype=np.int8)
    seg = [(int(v), int(l)) for p in rle_str.split(",") for v, l in [p.split(":")]]
    total = sum(l for _, l in seg); result = np.zeros(total, dtype=np.int8); pos = 0
    for v, l in seg: result[pos:pos+l] = v; pos += l
    return result

def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b): raise ValueError
    if len(a) == 0: return 1.0
    return 1.0 - int(np.sum(a != b)) / len(a)

def on_count(mask: np.ndarray) -> int:
    return int(np.sum(mask == 1))

def sample_led_grid(panel_img: np.ndarray, rows: int, cols: int, margin: float = 0.1) -> np.ndarray:
    gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY); h, w = gray.shape
    ch, cw = h / rows, w / cols; scores = np.zeros(rows * cols, dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cx, cy = int((c+0.5)*cw), int((r+0.5)*ch)
            ws = max(3, int(min(cw, ch)*(1-2*margin)))
            hw, hh = ws//2, ws//2
            win = gray[max(0,cy-hh):min(h,cy+hh+1), max(0,cx-hw):min(w,cx+hw+1)]
            scores[r*cols+c] = float(np.percentile(win, 90)) if win.size else 0.0
    return scores

def binarize_scores(scores: np.ndarray, on_threshold: float = 0.35) -> np.ndarray:
    nb = max(1.0, float(np.percentile(scores, 95)))
    norm = scores / nb
    ot = cv2.threshold((norm*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]/255.0
    return (norm >= max(on_threshold, ot*0.6)).astype(np.int8)

def detect_panel_bbox(frame: np.ndarray) -> tuple[int,int,int,int] | None:
    h, w = frame.shape[:2]; gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    for thresh, close_k, open_k in [(92, (23,23), (9,9)), (180, (15,15), (9,9))]:
        if thresh == 180: bin = (blur > thresh).astype(np.uint8)*255
        else: bin = (blur < thresh).astype(np.uint8)*255
        bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, np.ones(close_k, np.uint8))
        bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, np.ones(open_k, np.uint8))
        bbox = _best_bbox(bin, h, w)
        if bbox: return bbox
    return None

def _best_bbox(binary: np.ndarray, h: int, w: int) -> tuple[int,int,int,int] | None:
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score, best = -1.0, None; fa = float(w*h); cx_t, cy_t = w*0.5, h*0.5
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c); area = float(bw*bh)
        if not (fa*0.03 <= area <= fa*0.55): continue
        if not (0.3 <= bh/max(bw,1) <= 4.0): continue
        if cv2.contourArea(c)/max(area,1) < 0.4: continue
        cent = 1.0 - min(1.0, math.hypot((x+bw*0.5-cx_t)/w, (y+bh*0.5-cy_t)/h)*1.8)
        ep = 0.1 if (x<=4 or y<=4 or x+bw>=w-4 or y+bh>=h-4) else 0.0
        score = cent - ep
        if score > best_score: best_score, best = score, (x, y, bw, bh)
    return best

def crop_panel(frame: np.ndarray, bbox: tuple[int,int,int,int], fs: tuple[int,int]=(720,480)) -> np.ndarray:
    x, y, w, h = bbox; panel = frame[y:y+h, x:x+w]
    if panel.size == 0: raise ValueError
    return cv2.resize(panel, fs, interpolation=cv2.INTER_LINEAR)

def open_video(path: Path) -> tuple:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise FileNotFoundError(f"无法打开: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, n, fps, w, h

def sample_frames(cap, sample_fps: float, source_fps: float, max_frames: int = 0) -> list:
    stride = max(1, round(source_fps/max(sample_fps,1)))
    result = []; idx = 0
    while True:
        ok, f = cap.read()
        if not ok or f is None: break
        if idx % stride == 0:
            result.append((idx, f, idx/max(source_fps,1)))
            if max_frames > 0 and len(result) >= max_frames: break
        idx += 1
    return result

def deduplicate_key_states(masks: list[np.ndarray], dedupe_sim: float = 0.985, min_stable: int = 2) -> tuple:
    if not masks: return [], []
    keys, kidx, cand, cc = [], [], None, 0
    for i, m in enumerate(masks):
        if cand is None: cand, cc = m.copy(), 1; continue
        if hamming_similarity(cand, m) >= dedupe_sim: cc += 1
        else:
            if cc >= min_stable: keys.append(cand.copy()); kidx.append(i-cc)
            cand, cc = m.copy(), 1
    if cand is not None and cc >= min_stable: keys.append(cand.copy()); kidx.append(len(masks)-cc)
    return keys, kidx

class StateMachine:
    def __init__(self, model: KeyStatesModel):
        self.model = model; self.th = model.thresholds; self.states = model.states
        self.sequence = model.sequence; self.state_count = len(self.states)
        self.state_lookup = {s.id: s for s in self.states}
        self.current_index = 0; self.hold_frames = 0; self.unknown_frames = 0
        self.events: list[Event] = []; self.trace: list[FrameResult] = []; self._k0_matched = False
    def reset(self):
        self.current_index = 0; self.hold_frames = 0; self.unknown_frames = 0
        self.events = []; self.trace = []; self._k0_matched = False
    def process_frame(self, mask: np.ndarray, fi: int, ts: float) -> FrameResult:
        if self.state_count == 0:
            return FrameResult(fi, ts, "", "", "", 0, 0, "error", 0, False, "no_states")
        cs = self.states[self.current_index]; ni = (self.current_index+1)%self.state_count; ns = self.states[ni]
        sc = hamming_similarity(mask, cs.mask); sn = hamming_similarity(mask, ns.mask)
        bi, bs = self._find_best(mask)
        th_sim = float(self.th.get("state_similarity", 0.95))
        th_tr = int(self.th.get("transition_tolerance_frames", 3))
        th_hf = float(self.th.get("hold_timeout_factor", 2.5))
        if not self._k0_matched:
            if sc >= th_sim:
                self._k0_matched = True; self.hold_frames = 1; mid, tr = cs.id, "stay"; okf = True
            elif bs >= th_sim and bi > 0:
                self.current_index = bi; self.hold_frames = 1; self._k0_matched = True
                mid = self.states[bi].id; tr = "advance"; okf = False
                self._add_event(fi, ts, "skipped_state", "K000", mid, f"起始跳过前{bi}个状态->{mid}")
            else:
                self.unknown_frames += 1; tr = "unknown_transition"; mid = "?"; okf = True
                if self.unknown_frames > 100:
                    self._add_event(fi, ts, "panel_lost", "K000", "", "长时间未匹配K0"); okf = False
            et = self.events[-1].type if self.events and self.events[-1].frame_index == fi else ""
            r = FrameResult(fi, ts, cs.id, mid, ns.id, sc, sn, tr, self.hold_frames, okf, et)
            self.trace.append(r); return r
        if sc >= th_sim:
            self.hold_frames += 1; self.unknown_frames = 0; tr, mid = "stay", cs.id
        elif sn >= th_sim:
            self.current_index = ni; self.hold_frames = 0; self.unknown_frames = 0; tr, mid = "advance", ns.id
        elif self.unknown_frames < th_tr:
            self.unknown_frames += 1; tr, mid = "unknown_transition", cs.id
        else:
            cur_on = int(np.sum(mask)); total = len(mask)
            if total > 0 and cur_on < total * 0.05:
                self._add_event(fi, ts, "unknown_state", cs.id, f"on={cur_on}", f"亮点过少({cur_on}/{total})")
                mid, tr = cs.id, "unknown_transition"
            elif bi < 0:
                self._add_event(fi, ts, "unknown_state", cs.id, "", "无法匹配任何关键状态"); mid, tr = cs.id, "unknown_transition"
            elif bi == self.current_index:
                self.unknown_frames = 0; tr, mid = "stay", self.states[bi].id
            elif bi == ni:
                self.current_index = ni; self.hold_frames = 0; self.unknown_frames = 0; tr, mid = "advance", ns.id
            elif bi > self.current_index:
                prev = self.states[max(0, bi-(bi-self.current_index))].id; skipped = bi-self.current_index
                self.current_index = bi; self.hold_frames = 1; self.unknown_frames = 0; tr = "advance"
                mid = self.states[bi].id
                self._add_event(fi, ts, "skipped_state", prev, mid, f"跳过{skipped}个状态:{prev}->{mid}")
            else:
                mid = self.states[bi].id
                self._add_event(fi, ts, "wrong_order", cs.id, mid, f"倒退:期望{cs.id}，匹配到{mid}")
                tr = "error"
        if tr == "stay" and self._k0_matched:
            mh = cs.median_hold_frames
            if mh > 0 and self.hold_frames > mh * th_hf:
                self._add_event(fi, ts, "stuck_state", cs.id, mid, f"在{cs.id}停留{self.hold_frames}/{mh*th_hf:.0f}帧")
        et = self.events[-1].type if self.events and self.events[-1].frame_index == fi else ""
        okf = not any(e.frame_index == fi for e in self.events)
        r = FrameResult(fi, ts, cs.id, mid, ns.id, sc, sn, tr, self.hold_frames, okf, et)
        self.trace.append(r); return r
    def _find_best(self, mask: np.ndarray) -> tuple[int, float]:
        bi, bs = -1, -1.0
        for i, s in enumerate(self.states):
            sim = hamming_similarity(mask, s.mask)
            if sim > bs: bs, bi = sim, i
        return bi, bs
    def _add_event(self, fi, ts, typ, exp, mat, msg):
        self.events.append(Event(fi, ts, typ, exp, mat, msg))
    def finalize(self) -> list[Event]:
        fe = []; visited = {r.matched_state for r in self.trace if r.matched_state in self.state_lookup}
        for sid in self.sequence:
            if sid not in visited:
                lf = self.trace[-1].frame_index if self.trace else 0
                lt = self.trace[-1].time_sec if self.trace else 0.0
                e = Event(lf, lt, "missing_state", sid, "", f"检测结束，{sid}从未到达")
                fe.append(e); self.events.append(e)
        return fe
    @property
    def overall_ok(self) -> bool: return all(r.ok for r in self.trace) and len(self.events) == 0

def jsonable(v):
    if isinstance(v, dict): return {str(k): jsonable(v2) for k, v2 in v.items()}
    if isinstance(v, (list, tuple)): return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray): return jsonable(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    return v

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")

def debug_write_mask_png(path: Path, mask: np.ndarray, rows: int, cols: int) -> None:
    img = np.zeros((rows*12, cols*12, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            i = r*cols+c
            if i < len(mask):
                val = int(mask[i])*255; color = (0, val, val) if val else (20,20,30)
                cv2.rectangle(img, (c*12, r*12), ((c+1)*12-1, (r+1)*12-1), color, -1)
                cv2.rectangle(img, (c*12, r*12), ((c+1)*12-1, (r+1)*12-1), (60,60,60), 1)
    cv2.imwrite(str(path), img)
