from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "seven_segment" / "Video_20260515112415516"
OUT_DIR = ROOT / "output" / "sevenseg_station" / "live_camera_demo"


def canonical_sequence(payload: dict) -> list[str]:
    raw_templates = payload.get("templates", [])
    merge_map = {
        str(item.get("id")): str(item.get("merged_to"))
        for item in raw_templates
        if item.get("merged_to")
    }
    allowed = {
        str(item.get("id"))
        for item in raw_templates
        if item.get("enabled", True) is not False
        and int(item.get("count", 0)) >= 3
        and not item.get("merged_to")
    }

    def canonical(template_id: str) -> str:
        seen: set[str] = set()
        current = template_id
        while current in merge_map and current not in seen:
            seen.add(current)
            current = merge_map[current]
        return current

    seq: list[str] = []
    previous = ""
    for item in payload.get("sequence", []):
        template_id = canonical(str(item.get("template_id", "")))
        if template_id in allowed and template_id != previous:
            seq.append(template_id)
            previous = template_id
    return seq


def build_model_data() -> dict:
    payload = json.loads((MODEL_DIR / "templates.json").read_text(encoding="utf-8"))
    summary = json.loads((MODEL_DIR / "summary.json").read_text(encoding="utf-8"))
    templates = []
    for item in payload.get("templates", []):
        if item.get("enabled", True) is False:
            continue
        if item.get("merged_to"):
            continue
        if int(item.get("count", 0)) < 3:
            continue
        image = str(item.get("image", "")).replace("\\", "/")
        templates.append(
            {
                "id": str(item["id"]),
                "bits": str(item["bits"]),
                "count": int(item.get("count", 0)),
                "image": f"/models/seven_segment/Video_20260515112415516/{image}",
            }
        )
    return {
        "name": "Video_20260515112415516",
        "sourceWidth": int(summary.get("meta", {}).get("width", 2448)),
        "sourceHeight": int(summary.get("meta", {}).get("height", 2048)),
        "roi": summary.get("roi", {"x": 770, "y": 350, "w": 1126, "h": 817}),
        "gridCols": 96,
        "gridRows": 64,
        "activeThreshold": 55,
        "matchThreshold": 0.68,
        "tolerantThreshold": 0.58,
        "tolerantDilate": 4,
        "minOrderedCoverage": 0.90,
        "maxUnknownSegments": 3,
        "templates": templates,
        "sequence": canonical_sequence(payload),
    }


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>七段管实时相机展示</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; background: #0f172a; color: #e5e7eb; font-family: Arial, "Microsoft YaHei", sans-serif; }
    header { background: #111827; border-bottom: 1px solid #334155; padding: 18px 24px; }
    h1 { margin: 0 0 8px; font-size: 24px; }
    main { padding: 18px 24px 28px; }
    .bar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin: 12px 0; }
    select, input { background: #111827; color: #e5e7eb; border: 1px solid #475569; border-radius: 6px; padding: 8px 10px; }
    button { border: 0; border-radius: 6px; padding: 9px 14px; font-weight: 700; background: #2563eb; color: white; cursor: pointer; }
    button.secondary { background: #475569; }
    button.warn { background: #b45309; }
    .pill { padding: 8px 12px; border-radius: 6px; background: #334155; font-weight: 700; }
    .pill.ok { background: #15803d; }
    .pill.ng { background: #b91c1c; }
    .pill.wait { background: #475569; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 14px 0; }
    .metric { background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 10px 12px; }
    .metric b { display: block; color: #94a3b8; font-size: 12px; margin-bottom: 4px; }
    .metric span { font-size: 22px; font-weight: 800; color: #f8fafc; }
    .stage { display: grid; grid-template-columns: 1.15fr 1fr 1fr 1fr; gap: 14px; }
    .panel { background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 12px; min-width: 0; }
    .panel h2 { margin: 0 0 10px; font-size: 17px; color: #f8fafc; }
    canvas, video, img { width: 100%; background: #020617; border-radius: 8px; display: block; }
    canvas { aspect-ratio: 3 / 2; }
    .videoWrap { position: relative; background: #020617; border-radius: 8px; overflow: hidden; }
    #video { display: block; }
    #overlay { position: absolute; inset: 0; pointer-events: none; background: transparent; }
    .templateBox { width: 100%; aspect-ratio: 3 / 2; display: grid; place-items: center; background: #020617; border-radius: 8px; overflow: hidden; }
    .templateBox img { width: 100%; height: 100%; object-fit: contain; }
    .roiControls { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
    .roiControls label { display: grid; gap: 4px; font-size: 12px; color: #cbd5e1; }
    .timeline { display: flex; gap: 3px; flex-wrap: wrap; margin-top: 8px; max-height: 74px; overflow: auto; }
    .step { width: 12px; height: 12px; border-radius: 2px; background: #334155; }
    .step.hit { background: #22c55e; }
    .step.cursor { outline: 2px solid #facc15; }
    .log { height: 110px; overflow: auto; white-space: pre-wrap; background: #020617; border: 1px solid #334155; border-radius: 8px; padding: 10px; color: #cbd5e1; font-size: 13px; }
    .note { color: #cbd5e1; line-height: 1.55; max-width: 1180px; }
    @media (max-width: 1400px) { .stage { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 760px) { .stage, .roiControls { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>七段管实时相机展示</h1>
    <div class="note">临时演示页：浏览器直接打开相机，实时裁 ROI、匹配模板、按关键模板有序出现判断。不会保存视频。</div>
  </header>
  <main>
    <div class="bar">
      <select id="cameraSelect"></select>
      <button id="refreshBtn" class="secondary">刷新相机</button>
      <button id="startBtn">打开相机</button>
      <button id="stopBtn" class="secondary">停止</button>
      <button id="resetBtn" class="warn">重置检测</button>
      <span id="statusPill" class="pill wait">未开始</span>
      <span id="cameraInfo" class="pill">-</span>
    </div>
    <div class="metrics">
      <div class="metric"><b>当前模板</b><span id="currentTemplate">-</span></div>
      <div class="metric"><b>相似度</b><span id="similarity">0.00</span></div>
      <div class="metric"><b>有序命中</b><span id="orderedHits">0 / 0</span></div>
      <div class="metric"><b>覆盖率</b><span id="coverage">0%</span></div>
      <div class="metric"><b>未知亮段</b><span id="unknownCount">0</span></div>
      <div class="metric"><b>判定</b><span id="resultText">WAIT</span></div>
    </div>
    <div class="stage">
      <section class="panel">
        <h2>相机预览和 ROI</h2>
        <div class="videoWrap">
          <video id="video" autoplay playsinline muted></video>
          <canvas id="overlay"></canvas>
        </div>
        <div class="roiControls">
          <label>X <input id="roiX" type="number" min="0" step="1"></label>
          <label>Y <input id="roiY" type="number" min="0" step="1"></label>
          <label>W <input id="roiW" type="number" min="10" step="1"></label>
          <label>H <input id="roiH" type="number" min="10" step="1"></label>
        </div>
        <div class="bar">
          <button id="useDefaultRoi" class="secondary">使用模型 ROI</button>
          <button id="centerRoi" class="secondary">居中 ROI</button>
        </div>
      </section>
      <section class="panel">
        <h2>当前 ROI 真图</h2>
        <canvas id="roiCanvas"></canvas>
      </section>
      <section class="panel">
        <h2>当前抽象亮区</h2>
        <canvas id="abstractCanvas"></canvas>
      </section>
      <section class="panel">
        <h2>命中模板和差异</h2>
        <div class="templateBox"><img id="templateImage" alt="命中模板"></div>
        <canvas id="diffCanvas"></canvas>
      </section>
    </div>
    <section class="panel" style="margin-top:14px;">
      <h2>关键时序</h2>
      <div id="timeline" class="timeline"></div>
      <div id="log" class="log"></div>
    </section>
  </main>
  <script>
    const MODEL = __MODEL_DATA__;
    const templates = MODEL.templates;
    const templateById = new Map(templates.map(t => [t.id, t]));
    const seq = MODEL.sequence;
    const cols = MODEL.gridCols, rows = MODEL.gridRows;
    const video = document.getElementById("video");
    const overlay = document.getElementById("overlay");
    const overlayCtx = overlay.getContext("2d");
    const roiCanvas = document.getElementById("roiCanvas");
    const roiCtx = roiCanvas.getContext("2d", { willReadFrequently: true });
    const abstractCanvas = document.getElementById("abstractCanvas");
    const abstractCtx = abstractCanvas.getContext("2d");
    const diffCanvas = document.getElementById("diffCanvas");
    const diffCtx = diffCanvas.getContext("2d");
    const tinyCanvas = document.createElement("canvas");
    const tinyCtx = tinyCanvas.getContext("2d", { willReadFrequently: true });
    const cameraSelect = document.getElementById("cameraSelect");
    const roiInputs = {
      x: document.getElementById("roiX"),
      y: document.getElementById("roiY"),
      w: document.getElementById("roiW"),
      h: document.getElementById("roiH"),
    };
    let stream = null;
    let timer = null;
    let roi = { x: 0, y: 0, w: 100, h: 100 };
    let cursor = 0;
    let hit = new Set();
    let unknownSegments = 0;
    let lastTemplate = "";
    let lastStatus = "dark";
    let frameNo = 0;
    let templateMatrices = [];

    function log(line) {
      const el = document.getElementById("log");
      const time = new Date().toLocaleTimeString();
      el.textContent = `[${time}] ${line}\n` + el.textContent;
    }

    function bitStringToArray(bits) {
      return Uint8Array.from(bits, ch => ch === "1" ? 1 : 0);
    }

    function dilateBits(bits, radius) {
      if (radius <= 1) return bits;
      const out = new Uint8Array(bits.length);
      const half = Math.floor(radius / 2);
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          let on = 0;
          for (let yy = y - half; yy <= y + half && !on; yy++) {
            if (yy < 0 || yy >= rows) continue;
            for (let xx = x - half; xx <= x + half; xx++) {
              if (xx < 0 || xx >= cols) continue;
              if (bits[yy * cols + xx]) { on = 1; break; }
            }
          }
          out[y * cols + x] = on;
        }
      }
      return out;
    }

    function jaccard(a, b) {
      let inter = 0, union = 0;
      for (let i = 0; i < a.length; i++) {
        const av = a[i] ? 1 : 0, bv = b[i] ? 1 : 0;
        if (av && bv) inter++;
        if (av || bv) union++;
      }
      return union ? inter / union : 0;
    }

    function prepareTemplates() {
      templateMatrices = templates.map(t => {
        const bits = bitStringToArray(t.bits);
        return { ...t, bitsArray: bits, tolerantBits: dilateBits(bits, MODEL.tolerantDilate) };
      });
    }

    function setStatus(text, cls) {
      const pill = document.getElementById("statusPill");
      pill.textContent = text;
      pill.className = `pill ${cls || "wait"}`;
      document.getElementById("resultText").textContent = text;
    }

    async function refreshCameras() {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter(d => d.kind === "videoinput");
      cameraSelect.innerHTML = "";
      cams.forEach((cam, i) => {
        const opt = document.createElement("option");
        opt.value = cam.deviceId;
        opt.textContent = cam.label || `Camera ${i + 1}`;
        cameraSelect.appendChild(opt);
      });
      document.getElementById("cameraInfo").textContent = `${cams.length} 个相机`;
    }

    async function startCamera() {
      stopCamera();
      const deviceId = cameraSelect.value;
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: MODEL.sourceWidth },
          height: { ideal: MODEL.sourceHeight },
        },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      await refreshCameras();
      applyDefaultRoi();
      resetDetection();
      timer = setInterval(processFrame, 120);
      setStatus("预览检测中", "wait");
      log("相机已打开，开始实时匹配。");
    }

    function stopCamera() {
      if (timer) clearInterval(timer);
      timer = null;
      if (stream) stream.getTracks().forEach(t => t.stop());
      stream = null;
    }

    function resetDetection() {
      cursor = 0;
      hit = new Set();
      unknownSegments = 0;
      lastTemplate = "";
      lastStatus = "dark";
      frameNo = 0;
      renderTimeline();
      setStatus("等待关键帧", "wait");
      log("检测进度已重置。");
    }

    function videoSize() {
      return { w: video.videoWidth || MODEL.sourceWidth, h: video.videoHeight || MODEL.sourceHeight };
    }

    function applyDefaultRoi() {
      const size = videoSize();
      roi = {
        x: Math.round(MODEL.roi.x * size.w / MODEL.sourceWidth),
        y: Math.round(MODEL.roi.y * size.h / MODEL.sourceHeight),
        w: Math.round(MODEL.roi.w * size.w / MODEL.sourceWidth),
        h: Math.round(MODEL.roi.h * size.h / MODEL.sourceHeight),
      };
      syncRoiInputs();
    }

    function centerRoi() {
      const size = videoSize();
      roi.w = Math.round(size.w * MODEL.roi.w / MODEL.sourceWidth);
      roi.h = Math.round(size.h * MODEL.roi.h / MODEL.sourceHeight);
      roi.x = Math.round((size.w - roi.w) / 2);
      roi.y = Math.round((size.h - roi.h) / 2);
      syncRoiInputs();
    }

    function syncRoiInputs() {
      roiInputs.x.value = roi.x; roiInputs.y.value = roi.y; roiInputs.w.value = roi.w; roiInputs.h.value = roi.h;
    }

    function readRoiInputs() {
      const size = videoSize();
      roi = {
        x: Math.max(0, Math.min(size.w - 1, Number(roiInputs.x.value) || 0)),
        y: Math.max(0, Math.min(size.h - 1, Number(roiInputs.y.value) || 0)),
        w: Math.max(10, Math.min(size.w, Number(roiInputs.w.value) || 10)),
        h: Math.max(10, Math.min(size.h, Number(roiInputs.h.value) || 10)),
      };
      if (roi.x + roi.w > size.w) roi.w = size.w - roi.x;
      if (roi.y + roi.h > size.h) roi.h = size.h - roi.y;
    }

    function drawOverlay() {
      const rect = video.getBoundingClientRect();
      overlay.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
      overlay.height = Math.max(1, Math.round(rect.height * devicePixelRatio));
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      const size = videoSize();
      const sx = overlay.width / size.w, sy = overlay.height / size.h;
      overlayCtx.strokeStyle = "#22c55e";
      overlayCtx.lineWidth = 3 * devicePixelRatio;
      overlayCtx.strokeRect(roi.x * sx, roi.y * sy, roi.w * sx, roi.h * sy);
      overlayCtx.fillStyle = "rgba(34,197,94,.16)";
      overlayCtx.fillRect(roi.x * sx, roi.y * sy, roi.w * sx, roi.h * sy);
    }

    function frameToBits() {
      readRoiInputs();
      roiCanvas.width = roi.w;
      roiCanvas.height = roi.h;
      roiCtx.drawImage(video, roi.x, roi.y, roi.w, roi.h, 0, 0, roi.w, roi.h);
      tinyCanvas.width = cols;
      tinyCanvas.height = rows;
      tinyCtx.drawImage(roiCanvas, 0, 0, cols, rows);
      const img = tinyCtx.getImageData(0, 0, cols, rows).data;
      const bits = new Uint8Array(cols * rows);
      let active = 0;
      for (let i = 0; i < cols * rows; i++) {
        const r = img[i * 4], g = img[i * 4 + 1], b = img[i * 4 + 2];
        const gray = Math.max(r, g, b);
        if (gray > MODEL.activeThreshold) {
          bits[i] = 1;
          active++;
        }
      }
      return { bits, active };
    }

    function findBest(bits) {
      let best = { id: "", sim: 0, template: null };
      for (const t of templateMatrices) {
        const sim = jaccard(bits, t.bitsArray);
        if (sim > best.sim) best = { id: t.id, sim, template: t };
      }
      if (best.sim >= MODEL.matchThreshold) return best;
      const tolerant = dilateBits(bits, MODEL.tolerantDilate);
      for (const t of templateMatrices) {
        const sim = jaccard(tolerant, t.tolerantBits);
        if (sim > best.sim) best = { id: t.id, sim, template: t };
      }
      return best.sim >= MODEL.tolerantThreshold ? best : { id: "", sim: best.sim, template: null };
    }

    function acceptTemplate(id) {
      if (!id || id === lastTemplate) return;
      const atOrAfter = seq.indexOf(id, cursor);
      if (atOrAfter >= 0) {
        for (let i = cursor; i <= atOrAfter; i++) hit.add(i);
        cursor = atOrAfter + 1;
        log(`命中 ${id}，时序推进到 ${cursor}/${seq.length}`);
      } else {
        const before = seq.slice(0, cursor).includes(id);
        log(before ? `重复看到 ${id}，已接受` : `看到非标准模板 ${id}`);
      }
      lastTemplate = id;
    }

    function updateResult(status, id, sim) {
      const coverage = hit.size / Math.max(seq.length, 1);
      document.getElementById("currentTemplate").textContent = id || "-";
      document.getElementById("similarity").textContent = sim.toFixed(2);
      document.getElementById("orderedHits").textContent = `${hit.size} / ${seq.length}`;
      document.getElementById("coverage").textContent = `${Math.round(coverage * 100)}%`;
      document.getElementById("unknownCount").textContent = String(unknownSegments);
      if (coverage >= MODEL.minOrderedCoverage && unknownSegments <= MODEL.maxUnknownSegments) {
        setStatus("PASS", "ok");
      } else if (status === "unknown") {
        setStatus("复核中", "ng");
      } else {
        setStatus("检测中", "wait");
      }
      renderTimeline();
    }

    function drawGrid(canvas, ctx, bits, color) {
      canvas.width = 720;
      canvas.height = 480;
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const pad = 18, cw = (canvas.width - pad * 2) / cols, ch = (canvas.height - pad * 2) / rows;
      ctx.fillStyle = "rgba(30,41,59,.55)";
      for (let y = 0; y < rows; y += 4) ctx.fillRect(pad, pad + y * ch, canvas.width - pad * 2, 1);
      ctx.fillStyle = color;
      for (let i = 0; i < bits.length; i++) {
        if (!bits[i]) continue;
        const x = i % cols, y = Math.floor(i / cols);
        ctx.fillRect(pad + x * cw, pad + y * ch, Math.max(1, cw * .92), Math.max(1, ch * .92));
      }
    }

    function drawDiff(bits, target) {
      diffCanvas.width = 720;
      diffCanvas.height = 480;
      diffCtx.fillStyle = "#020617";
      diffCtx.fillRect(0, 0, diffCanvas.width, diffCanvas.height);
      if (!target) return;
      const pad = 18, cw = (diffCanvas.width - pad * 2) / cols, ch = (diffCanvas.height - pad * 2) / rows;
      for (let i = 0; i < bits.length; i++) {
        const a = bits[i], t = target.bitsArray[i];
        if (!a && !t) continue;
        diffCtx.fillStyle = a && t ? "#22c55e" : (a ? "#ef4444" : "#facc15");
        const x = i % cols, y = Math.floor(i / cols);
        diffCtx.fillRect(pad + x * cw, pad + y * ch, Math.max(1, cw * .92), Math.max(1, ch * .92));
      }
    }

    function renderTimeline() {
      const el = document.getElementById("timeline");
      el.innerHTML = "";
      seq.forEach((id, i) => {
        const d = document.createElement("div");
        d.className = `step ${hit.has(i) ? "hit" : ""} ${i === cursor ? "cursor" : ""}`;
        d.title = `${i + 1}. ${id}`;
        el.appendChild(d);
      });
    }

    function processFrame() {
      if (!stream || video.readyState < 2) return;
      frameNo++;
      drawOverlay();
      const { bits, active } = frameToBits();
      let status = "dark", id = "", sim = 0, tmpl = null;
      if (active > 8) {
        const best = findBest(bits);
        sim = best.sim;
        if (best.id) {
          status = "matched";
          id = best.id;
          tmpl = best.template;
          acceptTemplate(id);
        } else {
          status = "unknown";
          if (lastStatus !== "unknown") unknownSegments++;
          lastTemplate = "";
        }
      } else {
        lastTemplate = "";
      }
      lastStatus = status;
      drawGrid(abstractCanvas, abstractCtx, bits, status === "unknown" ? "#ef4444" : (status === "dark" ? "#64748b" : "#22c55e"));
      drawDiff(bits, tmpl);
      document.getElementById("templateImage").src = tmpl ? tmpl.image : "";
      updateResult(status, id, sim);
    }

    document.getElementById("refreshBtn").onclick = () => refreshCameras().catch(e => log(`刷新相机失败: ${e.message}`));
    document.getElementById("startBtn").onclick = () => startCamera().catch(e => {
      setStatus("相机失败", "ng");
      log(`打开相机失败: ${e.message}`);
    });
    document.getElementById("stopBtn").onclick = stopCamera;
    document.getElementById("resetBtn").onclick = resetDetection;
    document.getElementById("useDefaultRoi").onclick = applyDefaultRoi;
    document.getElementById("centerRoi").onclick = centerRoi;
    Object.values(roiInputs).forEach(input => input.onchange = () => { readRoiInputs(); drawOverlay(); });
    window.onresize = drawOverlay;
    prepareTemplates();
    renderTimeline();
    refreshCameras().catch(e => log(`请先允许浏览器访问相机: ${e.message}`));
  </script>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_data = build_model_data()
    html = HTML.replace("__MODEL_DATA__", json.dumps(model_data, ensure_ascii=False))
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    print(OUT_DIR / "index.html")


if __name__ == "__main__":
    main()
