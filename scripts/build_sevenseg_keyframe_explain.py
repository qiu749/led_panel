from __future__ import annotations

import html
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "seven_segment" / "Video_20260515112415516"
RESULT_DIR = ROOT / "output" / "sevenseg_station" / "live_video_518_20260519_141925"
OUT_DIR = ROOT / "output" / "sevenseg_station" / "keyframe_explain"
ASSET_DIR = OUT_DIR / "assets"


def copy_asset(src: Path, name: str) -> str:
    if not src.exists():
        return ""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    dst = ASSET_DIR / name
    shutil.copyfile(src, dst)
    return f"assets/{name}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    model = load_json(MODEL_DIR / "templates.json")
    summary = load_json(MODEL_DIR / "summary.json")
    result_summary = load_json(RESULT_DIR / "summary.json") if (RESULT_DIR / "summary.json").exists() else {}
    sequence = canonical_sequence(model)
    sequence_index = {tid: i + 1 for i, tid in enumerate(sequence)}

    full_img = copy_asset(MODEL_DIR / "full_reconstruction.jpg", "standard_full_reconstruction.jpg")
    contact_img = copy_asset(MODEL_DIR / "template_contact.jpg", "template_contact.jpg")
    result_timeline = copy_asset(RESULT_DIR / "timeline_overview.jpg", "result_timeline_overview.jpg")
    result_full = copy_asset(RESULT_DIR / "full_reconstruction.jpg", "result_full_reconstruction.jpg")
    result_roi = copy_asset(RESULT_DIR / "roi_marked.jpg", "result_roi_marked.jpg")

    template_cards = []
    enabled_templates = [
        item
        for item in model.get("templates", [])
        if item.get("enabled", True) is not False and not item.get("merged_to")
    ]
    enabled_templates.sort(key=lambda item: int(item.get("first_frame", 0)))
    for item in enabled_templates:
        tid = str(item["id"])
        img_rel = str(item.get("image", "")).replace("\\", "/")
        img = copy_asset(MODEL_DIR / img_rel, f"{tid}.jpg")
        mask_rel = str(item.get("mask_image", "")).replace("\\", "/")
        mask = copy_asset(MODEL_DIR / mask_rel, f"{tid}_mask.jpg")
        seq_no = sequence_index.get(tid)
        role = "关键帧" if seq_no else "辅助/短暂帧"
        template_cards.append(
            f"""
            <article class="template-card {'key' if seq_no else 'aux'}">
              <div class="template-head">
                <b>{html.escape(tid)}</b>
                <span>{role}</span>
              </div>
              <img src="{html.escape(img)}" alt="{html.escape(tid)}">
              <div class="template-meta">
                <span>时序: {seq_no if seq_no else '-'}</span>
                <span>首次: f{int(item.get('first_frame', 0))}</span>
                <span>出现: {int(item.get('count', 0))}</span>
              </div>
              <details>
                <summary>查看二值模板</summary>
                <img src="{html.escape(mask)}" alt="{html.escape(tid)} mask">
              </details>
            </article>
            """
        )

    seq_blocks = []
    for i, tid in enumerate(sequence, start=1):
        seq_blocks.append(f'<span class="seq-step" title="{html.escape(tid)}">{i}<b>{html.escape(tid)}</b></span>')

    ordered = result_summary.get("sequence_check", {})
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>七段管关键帧时序检测方案说明</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#5f6b7a; --line:#d8e0ea; --bg:#f5f7fb; --panel:#ffffff; --blue:#1d4ed8; --green:#16803a; --amber:#b7791f; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Arial, "Microsoft YaHei", sans-serif; }}
    header {{ background: #101827; color: white; padding: 34px 42px; }}
    header h1 {{ margin: 0 0 10px; font-size: 34px; letter-spacing: 0; }}
    header p {{ margin: 0; color: #cbd5e1; max-width: 1100px; line-height: 1.7; }}
    main {{ padding: 26px 42px 48px; }}
    section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 22px; margin: 18px 0; }}
    h2 {{ margin: 0 0 14px; font-size: 24px; }}
    h3 {{ margin: 18px 0 10px; }}
    p, li {{ color: #344256; line-height: 1.75; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fbfdff; }}
    .metric b {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    .metric span {{ font-size: 28px; font-weight: 800; color: var(--ink); }}
    .img-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fbfdff; }}
    .img-card img {{ width: 100%; display: block; background: #000; border-radius: 6px; }}
    .img-card b {{ display: block; margin-bottom: 8px; }}
    .flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; }}
    .flow div {{ border: 1px solid #bfd0ee; background: #eef5ff; color: #173b7a; border-radius: 8px; padding: 14px; font-weight: 700; min-height: 68px; display: grid; align-content: center; }}
    .rule {{ border-left: 5px solid var(--blue); background: #eef5ff; padding: 12px 14px; border-radius: 6px; margin: 12px 0; }}
    .ok {{ color: var(--green); font-weight: 800; }}
    .warn {{ color: var(--amber); font-weight: 800; }}
    .template-toolbar {{ position: sticky; top: 0; background: rgba(245,247,251,.96); border: 1px solid var(--line); border-radius: 8px; padding: 12px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; z-index: 5; }}
    input[type=search] {{ flex: 1; min-width: 220px; padding: 9px 11px; border: 1px solid #b7c3d4; border-radius: 6px; }}
    button {{ border: 0; border-radius: 6px; padding: 9px 12px; background: var(--blue); color: white; font-weight: 700; cursor: pointer; }}
    button.secondary {{ background: #526173; }}
    .templates {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 14px; margin-top: 14px; }}
    .template-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fff; }}
    .template-card.key {{ border-color: #98d5aa; box-shadow: inset 0 0 0 2px #dcfce7; }}
    .template-card.aux {{ opacity: .74; }}
    .template-card img {{ width: 100%; display: block; background: #000; border-radius: 6px; }}
    .template-head {{ display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-bottom: 8px; }}
    .template-head b {{ font-size: 18px; }}
    .template-head span {{ font-size: 12px; color: white; background: var(--green); border-radius: 999px; padding: 3px 8px; }}
    .aux .template-head span {{ background: #64748b; }}
    .template-meta {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; color: var(--muted); font-size: 12px; }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: var(--blue); font-weight: 700; }}
    .sequence {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .seq-step {{ border: 1px solid #bdd7c4; background: #effaf2; border-radius: 5px; padding: 5px 7px; font-size: 12px; color: #14532d; }}
    .seq-step b {{ margin-left: 5px; }}
    .small {{ color: var(--muted); font-size: 13px; }}
    @media (max-width: 780px) {{ header, main {{ padding-left: 18px; padding-right: 18px; }} header h1 {{ font-size: 26px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>七段管关键帧时序检测方案</h1>
    <p>目标不是 OCR 读数字，而是判断七段管屏幕是否按标准流程点亮。系统从标准视频中提取关键帧模板，检测时只要求视频或相机画面中出现这些关键模板，并且出现顺序正确。</p>
  </header>
  <main>
    <section>
      <h2>一、方案结论</h2>
      <div class="rule">这套方案适合产线演示和后续工位落地：不要求每一帧实时完全对上，只判断<strong>关键模板是否出现</strong>、<strong>出现顺序是否正确</strong>、<strong>亮区是否和模板一致</strong>。</div>
      <div class="metric-grid">
        <div class="metric"><b>标准关键模板</b><span>{len(sequence)}</span></div>
        <div class="metric"><b>模板总数</b><span>{len(enabled_templates)}</span></div>
        <div class="metric"><b>当前测试结果</b><span class="ok">{html.escape(str(result_summary.get('overall_status', '-')))}</span></div>
        <div class="metric"><b>关键帧覆盖率</b><span>{float(ordered.get('ordered_coverage', 0.0)):.1%}</span></div>
        <div class="metric"><b>未知亮帧</b><span>{int(result_summary.get('unknown_frames', 0) or 0)}</span></div>
        <div class="metric"><b>异常事件</b><span>{int(result_summary.get('event_count', 0) or 0)}</span></div>
      </div>
    </section>

    <section>
      <h2>二、检测原理</h2>
      <div class="flow">
        <div>1. 标准视频建模<br><span class="small">提取亮区关键帧</span></div>
        <div>2. 人工审核模板<br><span class="small">删除、合并、命名、排序</span></div>
        <div>3. 相机/视频检测<br><span class="small">裁 ROI 并抽象成亮区</span></div>
        <div>4. 模板匹配<br><span class="small">不做 OCR，只比亮区形态</span></div>
        <div>5. 有序时序判定<br><span class="small">出现顺序正确即可</span></div>
        <div>6. PASS/NG 和复核<br><span class="small">NG 保留报告</span></div>
      </div>
      <h3>为什么不要求实时逐帧对齐？</h3>
      <p>普通视频和工业相机采样频率不同，扫描刷新里的短暂相位可能被漏采。如果要求第 N 帧必须等于第 N 个模板，正常画面也会误报。现在采用“关键模板有序出现”判定：只要视频里抓到的模板序列是标准序列的有序子序列，并且覆盖率达到阈值，就判定时序通过。</p>
      <h3>为什么不 OCR？</h3>
      <p>七段管、图标、汉字和单位符号是固定工艺流程。直接做亮区模板匹配比 OCR 更稳定，也更容易解释：绿色是命中，红色是多亮，黄色是漏亮。</p>
    </section>

    <section>
      <h2>三、标准图和检测结果</h2>
      <div class="grid">
        <div class="img-card"><b>标准全亮参考图</b><img src="{html.escape(full_img)}" alt="标准全亮参考图"></div>
        <div class="img-card"><b>标准模板接触图</b><img src="{html.escape(contact_img)}" alt="标准模板接触图"></div>
        <div class="img-card"><b>导入视频全亮重建图</b><img src="{html.escape(result_full)}" alt="导入视频全亮重建图"></div>
        <div class="img-card"><b>导入视频 ROI</b><img src="{html.escape(result_roi)}" alt="导入视频 ROI"></div>
      </div>
      <h3>导入视频时间轴</h3>
      <div class="img-card"><img src="{html.escape(result_timeline)}" alt="导入视频时间轴"></div>
    </section>

    <section>
      <h2>四、标准关键时序</h2>
      <p>下面是审核后的关键模板时序。检测时不强制每个模板都被采到，但命中的模板必须沿着这条顺序往前推进。</p>
      <div class="sequence">
        {''.join(seq_blocks)}
      </div>
    </section>

    <section>
      <h2>五、所有关键帧模板</h2>
      <div class="template-toolbar">
        <input id="filter" type="search" placeholder="搜索模板，例如 T003、T071">
        <button onclick="setMode('key')">只看关键帧</button>
        <button class="secondary" onclick="setMode('all')">查看全部模板</button>
        <span class="small">绿色边框表示参与关键时序；灰色为辅助/短暂模板。</span>
      </div>
      <div id="templates" class="templates">
        {''.join(template_cards)}
      </div>
    </section>

    <section>
      <h2>六、给用户看的说明话术</h2>
      <ul>
        <li>我们不是识别数字，也不是做 OCR，而是检查屏幕是否按标准流程点亮。</li>
        <li>先用一段标准视频提取模板，形成标准关键帧和标准时序。</li>
        <li>检测时，相机或视频中只要出现这些关键模板，并且顺序正确，就可以判 PASS。</li>
        <li>视频采样漏掉个别很短的模板不会直接 NG，系统会看关键帧覆盖率。</li>
        <li>NG 时会保存异常帧、差异图和报告，工程师可以一眼看到多亮、漏亮或时序跳变。</li>
      </ul>
    </section>
  </main>
  <script>
    let mode = "all";
    const filter = document.getElementById("filter");
    const cards = Array.from(document.querySelectorAll(".template-card"));
    function applyFilter() {{
      const q = filter.value.trim().toLowerCase();
      for (const card of cards) {{
        const text = card.textContent.toLowerCase();
        const modeOk = mode === "all" || card.classList.contains("key");
        card.style.display = modeOk && text.includes(q) ? "" : "none";
      }}
    }}
    function setMode(next) {{
      mode = next;
      applyFilter();
    }}
    filter.addEventListener("input", applyFilter);
  </script>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html_text, encoding="utf-8")
    print(OUT_DIR / "index.html")


if __name__ == "__main__":
    build()
