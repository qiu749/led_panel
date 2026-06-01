from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def rel(path: str | Path, base: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return Path(path).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rgb_to_hex(rgb: list[int] | tuple[int, int, int]) -> str:
    r, g, b = [max(0, min(255, int(v))) for v in rgb[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def render_state_grid(recipe: dict[str, Any], on_ids: set[int], state: dict[str, Any] | None = None, width: int = 287, height: int = 207) -> str:
    slots = recipe.get("slots", [])
    color_by_id: dict[int, str] = {}
    mean_color = "#ffb02e"
    if state:
        if state.get("mean_rgb"):
            mean_color = rgb_to_hex(state.get("mean_rgb", [255, 176, 46]))
        for item in state.get("slot_colors", []):
            try:
                color_by_id[int(item["id"])] = rgb_to_hex(item.get("rgb", [255, 176, 46]))
            except Exception:
                continue
    panel_bbox = recipe.get("panel_bbox")
    offset_x = 0
    offset_y = 0
    if panel_bbox and len(panel_bbox) >= 4:
        offset_x = int(panel_bbox[0])
        offset_y = int(panel_bbox[1])
        width = int(panel_bbox[2])
        height = int(panel_bbox[3])
    elif recipe.get("image_size"):
        width, height = [int(v) for v in recipe["image_size"]]
    view_box = f"0 0 {width} {height}"
    circles: list[str] = []
    for slot in slots:
        sid = int(slot["id"])
        x = float(slot["x"]) - offset_x
        y = float(slot["y"]) - offset_y
        color = color_by_id.get(sid, mean_color) if sid in on_ids else "#2b2b2b"
        opacity = "1" if sid in on_ids else "0.65"
        circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.5" fill="{color}" opacity="{opacity}" />')
    return f'<svg viewBox="{view_box}" class="state-svg" role="img">{"".join(circles)}</svg>'


def build_learn_sections(recipe: dict[str, Any], review_dir: Path) -> str:
    states = recipe.get("states", [])
    sequence = recipe.get("sequence", [])
    display_states = recipe.get("display_states", [])
    display_sequence = recipe.get("display_sequence", [])
    scan_phase_templates = recipe.get("scan_phase_templates", [])
    scan_phase_color_templates = recipe.get("scan_phase_color_templates", [])
    logical_states = recipe.get("logical_states", [])
    logical_sequence = recipe.get("logical_sequence", [])
    display_cards: list[str] = []
    for state in display_states:
        sid = html.escape(str(state.get("id", "")))
        on_ids = set(int(v) for v in state.get("on_slot_ids", []))
        color_text = ",".join(str(v) for v in state.get("mean_rgb", []))
        display_cards.append(
            f"""
            <section class="state-card display-card">
              <div class="card-head">
                <strong>{sid}</strong>
                <span>亮点 {len(on_ids)}</span>
                <span>颜色 {html.escape(color_text)}</span>
              </div>
              {render_state_grid(recipe, on_ids, state)}
            </section>
            """
        )
    logical_cards: list[str] = []
    for state in logical_states:
        sid = html.escape(str(state.get("id", "")))
        on_ids = set(int(v) for v in state.get("on_slot_ids", []))
        members = ", ".join(str(v) for v in state.get("scan_members", []))
        logical_cards.append(
            f"""
            <section class="state-card logical-card">
              <div class="card-head">
                <strong>{sid}</strong>
                <span>亮点 {len(on_ids)}</span>
                <span>{html.escape(members)}</span>
              </div>
              {render_state_grid(recipe, on_ids, state)}
            </section>
            """
        )
    phase_cards: list[str] = []
    for state in scan_phase_templates:
        sid = html.escape(str(state.get("id", "")))
        on_ids = set(int(v) for v in state.get("on_slot_ids", []))
        evidence = f"{state.get('best_raw_state', '')} r={state.get('best_raw_recall', '')}"
        phase_cards.append(
            f"""
            <section class="state-card phase-card">
              <div class="card-head">
                <strong>{sid}</strong>
                <span>亮点 {len(on_ids)}</span>
                <span>{html.escape(evidence)}</span>
              </div>
              {render_state_grid(recipe, on_ids, state)}
            </section>
            """
        )
    phase_color_cards: list[str] = []
    for state in scan_phase_color_templates:
        sid = html.escape(str(state.get("id", "")))
        on_ids = set(int(v) for v in state.get("on_slot_ids", []))
        label = f"{state.get('display_state_id', '')} / {state.get('phase_template_id', '')}"
        phase_color_cards.append(
            f"""
            <section class="state-card phase-color-card">
              <div class="card-head">
                <strong>{sid}</strong>
                <span>亮点 {len(on_ids)}</span>
                <span>{html.escape(label)}</span>
              </div>
              {render_state_grid(recipe, on_ids, state)}
            </section>
            """
        )
    cards: list[str] = []
    for state in states:
        sid = html.escape(str(state.get("id", "")))
        on_ids = set(int(v) for v in state.get("on_slot_ids", []))
        cards.append(
            f"""
            <section class="state-card">
              <div class="card-head">
                <strong>{sid}</strong>
                <span>亮点 {len(on_ids)}</span>
                <span>中位 {state.get('median_duration_frames', '')} 帧</span>
              </div>
              {render_state_grid(recipe, on_ids, state)}
            </section>
            """
        )
    seq_items = "".join(
        f"<span class='seq-item'>{html.escape(str(item.get('state_id', '')))}<small>{item.get('duration_frames', '')}f</small></span>"
        for item in sequence
    )
    logical_seq_items = "".join(
        f"<span class='seq-item logical'>{html.escape(str(item.get('state_id', '')))}<small>{html.escape(','.join(str(v) for v in item.get('scan_members', [])))}</small></span>"
        for item in logical_sequence
    )
    display_seq_items = "".join(
        f"<span class='seq-item display'>{html.escape(str(item.get('state_id', '')))}<small>{item.get('duration_frames', '')}f</small></span>"
        for item in display_sequence
    )
    display_section = ""
    if display_cards:
        display_section = f"""
        <h2>显示模板</h2>
        <p class="hint">最终用于检测的显示状态。这里按颜色和亮灯分组；如果亮点数明显小于总槽位数，说明它可能是半亮图案、扫描相位或状态分组还没合好。</p>
        <div class="sequence">{display_seq_items}</div>
        <div class="state-grid display-grid">{''.join(display_cards)}</div>
        """
    phase_section = ""
    if phase_cards:
        phase_section = f"""
        <h2>507 扫描相位模板</h2>
        <p class="hint">这是点阵扫描的两套互补子模板。彩色卡片表示“显示颜色 + 507 相位”，用于理解相机实际拍到的半帧形态。</p>
        <div class="state-grid phase-color-grid">{''.join(phase_color_cards)}</div>
        <details class="debug-details">
          <summary>只看 507 几何形状</summary>
          <div class="state-grid phase-grid">{''.join(phase_cards)}</div>
        </details>
        """
    elif phase_color_cards:
        phase_section = f"""
        <h2>507 扫描相位模板</h2>
        <p class="hint">这是点阵扫描的两套互补子模板。彩色卡片表示“显示颜色 + 507 相位”。</p>
        <div class="state-grid phase-color-grid">{''.join(phase_color_cards)}</div>
        """
    elif phase_cards:
        phase_section = f"""
        <h2>507 扫描相位模板</h2>
        <p class="hint">这是点阵扫描的两套互补子模板，只展示几何形状。</p>
        <div class="state-grid phase-grid">{''.join(phase_cards)}</div>
        """
    raw_section = f"""
    <h2>显示模板</h2>
    <div class="sequence">{seq_items}</div>
    <div class="state-grid">{''.join(cards)}</div>
    """
    logical_section = ""
    if display_cards:
        raw_section = f"""
        <details class="debug-details">
          <summary>二值逻辑调试层</summary>
          <div class="sequence">{logical_seq_items}</div>
          <div class="state-grid logical-grid">{''.join(logical_cards)}</div>
        </details>
        <details class="debug-details">
          <summary>原始扫描相位调试层</summary>
          <div class="sequence">{seq_items}</div>
          <div class="state-grid">{''.join(cards)}</div>
        </details>
        """
    elif logical_cards:
        logical_section = f"""
        <h2>显示模板</h2>
        <div class="sequence">{logical_seq_items}</div>
        <div class="state-grid logical-grid">{''.join(logical_cards)}</div>
        """
        raw_section = f"""
        <details class="debug-details">
          <summary>原始扫描相位调试层</summary>
          <div class="sequence">{seq_items}</div>
          <div class="state-grid">{''.join(cards)}</div>
        </details>
        """
    return f"""
    {display_section}
    {phase_section}
    {logical_section}
    {raw_section}
    """


def build_detection_sections(trace_rows: list[dict[str, str]], events: list[dict[str, Any]]) -> str:
    event_by_frame = {str(item.get("frame_index")): item for item in events}
    timeline_parts: list[str] = []
    for row in trace_rows:
        event = event_by_frame.get(str(row.get("frame_index")))
        cls = "tick event" if event else "tick"
        label = html.escape(str(row.get("matched_state", "")))
        title = html.escape(
            f"frame {row.get('frame_index')} expected {row.get('expected_state')} matched {row.get('matched_state')} sim {row.get('similarity')} {row.get('event_type')}"
        )
        timeline_parts.append(f"<span class='{cls}' title='{title}'>{label}</span>")

    event_rows = "".join(
        f"""
        <tr>
          <td>{item.get('frame_index')}</td>
          <td>{html.escape(str(item.get('category', '')))}</td>
          <td>{html.escape(str(item.get('type', '')))}</td>
          <td>{html.escape(str(item.get('expected', '')))}</td>
          <td>{html.escape(str(item.get('matched', '')))}</td>
          <td>{float(item.get('score', 0.0)):.3f}</td>
        </tr>
        """
        for item in events
    )
    return f"""
    <h2>检测时间线</h2>
    <p class="hint">绿色表示匹配正常，红色表示发生了跳状态、未知状态或卡住等事件。鼠标停在小块上可以看帧号和相似度。</p>
    <div class="timeline">{''.join(timeline_parts)}</div>
    <h2>检测事件</h2>
    <table>
      <thead><tr><th>帧号</th><th>分类</th><th>类型</th><th>期望</th><th>匹配</th><th>分数</th></tr></thead>
      <tbody>{event_rows or '<tr><td colspan="6">没有事件</td></tr>'}</tbody>
    </table>
    """


def build_diagnostics_section(recipe: dict[str, Any], learn_dir: Path | None) -> str:
    status = str(recipe.get("status") or recipe.get("recipe_stage") or "unknown")
    diagnostics = recipe.get("diagnostics", {}) or {}
    verdict = str(diagnostics.get("verdict", "UNKNOWN"))
    verdict_label = {"PASS": "可冻结", "REVIEW": "需确认", "BLOCK": "不可信"}.get(verdict, "未知")
    verdict_class = {"PASS": "pass", "REVIEW": "review", "BLOCK": "block"}.get(verdict, "review")
    status_label = {"candidate": "候选配方", "frozen": "正式冻结配方"}.get(status, status)
    issues = diagnostics.get("issues", []) or []
    issue_rows = "".join(
        f"""
        <tr>
          <td>{html.escape(str(item.get('severity', '')))}</td>
          <td><code>{html.escape(str(item.get('code', '')))}</code></td>
          <td>{html.escape(str(item.get('message', '')))}</td>
        </tr>
        """
        for item in issues
    )
    if not issue_rows:
        issue_rows = "<tr><td colspan='3'>没有自动诊断风险，但仍需要人工确认后才能冻结。</td></tr>"
    approval_hint = ""
    if status == "candidate" and learn_dir:
        approval_path = learn_dir / "approval_template.json"
        if approval_path.exists():
            approval_hint = f"<p class='hint'>人工确认文件：<code>{html.escape(str(approval_path))}</code>。确认后另存或改名为 <code>approval.json</code>，再执行 freeze。</p>"
    if status == "frozen":
        approval = recipe.get("approval_summary", {})
        approval_hint = f"<p class='hint'>冻结来源：<code>{html.escape(str(recipe.get('source_candidate', '')))}</code>；审批：<code>{html.escape(str(approval.get('approval', '')))}</code>。</p>"
    return f"""
    <section class="diagnostics-panel">
      <div class="verdict {verdict_class}">
        <strong>{html.escape(verdict_label)}</strong>
        <span>{html.escape(status_label)} / diagnostics={html.escape(verdict)}</span>
      </div>
      <p class="hint">这页现在是“确认页”：自动学习只给候选，产线检测只能使用 frozen recipe。低置信度项必须在 approval.json 中明确处理。</p>
      {approval_hint}
      <table class="issue-table">
        <thead><tr><th>级别</th><th>风险代码</th><th>说明</th></tr></thead>
        <tbody>{issue_rows}</tbody>
      </table>
    </section>
    """


def render_review(recipe_path: Path, learn_dir: Path | None, detection_dir: Path | None, output: Path) -> None:
    recipe = load_json(recipe_path)
    detection_summary: dict[str, Any] | None = None
    trace_rows: list[dict[str, str]] = []
    events: list[dict[str, Any]] = []
    if detection_dir:
        summary_path = detection_dir / "detection_summary.json"
        events_path = detection_dir / "events.json"
        trace_path = detection_dir / "state_trace.csv"
        if summary_path.exists():
            detection_summary = load_json(summary_path)
        if events_path.exists():
            events = load_json(events_path).get("events", [])
        trace_rows = read_csv(trace_path)

    slot_overlay = ""
    roi_debug = ""
    if learn_dir:
        overlay_path = learn_dir / "slot_overlay_crop.jpg"
        if not overlay_path.exists():
            overlay_path = learn_dir / "slot_overlay.jpg"
        if overlay_path.exists():
            slot_overlay = f"<img class='hero-img' src='{html.escape(rel(overlay_path, output.parent))}' alt='slot overlay'>"
        before_path = learn_dir / "roi_crop_before_align.jpg"
        after_path = learn_dir / "roi_crop_after_align.jpg"
        projection_path = learn_dir / "slot_source_projection.jpg"
        debug_imgs = []
        for label, path in (("ROI 原图", before_path), ("ROI 对齐后", after_path), ("槽位学习投影图", projection_path)):
            if path.exists():
                debug_imgs.append(
                    f"<figure><img src='{html.escape(rel(path, output.parent))}' alt='{html.escape(label)}'><figcaption>{html.escape(label)}</figcaption></figure>"
                )
        if debug_imgs:
            roi_debug = f"<h2>ROI 与槽位来源</h2><p class='hint'>这里用于检查裁剪区域是否正确、稳像是否偏移、投影图是否覆盖完整点阵。</p><div class='debug-grid'>{''.join(debug_imgs)}</div>"

    source = recipe.get("source", {})
    detection_badge = ""
    if detection_summary:
        ok = bool(detection_summary.get("overall_ok"))
        detection_badge = f"<span class='badge {'ok' if ok else 'ng'}'>{'PASS' if ok else 'NG'} {detection_summary.get('event_count', 0)} events</span>"

    learn_sections = build_learn_sections(recipe, output.parent)
    diagnostics_section = build_diagnostics_section(recipe, learn_dir)
    display_state_count = len(recipe.get("display_states") or recipe.get("logical_states") or recipe.get("states", []))
    detection_sections = build_detection_sections(trace_rows, events) if detection_dir else ""
    slot_count = int(recipe.get("slot_count", len(recipe.get("slots", []))) or 0)
    display_states = recipe.get("display_states") or recipe.get("logical_states") or recipe.get("states", [])
    partial_states = [
        f"{state.get('id', '')}={state.get('on_count', len(state.get('on_slot_ids', [])))}"
        for state in display_states
        if slot_count and int(state.get("on_count", len(state.get("on_slot_ids", []))) or 0) < int(slot_count * 0.9)
    ]
    phase_count = len(recipe.get("scan_phase_templates", []))
    phase_color_count = len(recipe.get("scan_phase_color_templates", []))
    warning_html = ""
    if partial_states:
        warning_html = f"<div class='warn'><strong>注意：</strong>这些显示态亮点数低于槽位数 90%，需要人工确认是正常图案还是未合并完整：{html.escape(', '.join(partial_states))}</div>"

    payload = {
        "recipe": str(recipe_path),
        "learn_dir": str(learn_dir) if learn_dir else "",
        "detection_dir": str(detection_dir) if detection_dir else "",
    }
    doc = f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<title>点阵视频学习报告</title>
<style>
  :root {{ color-scheme: dark; font-family: Inter, Segoe UI, Arial, sans-serif; background: #080808; color: #e8e8e8; }}
  body {{ margin: 0; padding: 24px; background: #080808; }}
  h1 {{ margin: 0 0 10px; font-size: 24px; }}
  h2 {{ margin: 28px 0 12px; font-size: 18px; }}
  .meta {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 16px; color: #bdbdbd; }}
  .pill, .badge {{ padding: 6px 10px; border: 1px solid #333; border-radius: 6px; background: #151515; }}
  .guide, .warn {{ margin: 14px 0 18px; padding: 12px 14px; border: 1px solid #2b3b45; background: #0d1418; border-radius: 6px; color: #d9e7ef; line-height: 1.6; }}
  .warn {{ border-color: #76552b; background: #1a1209; color: #ffd18a; }}
  .diagnostics-panel {{ margin: 14px 0 18px; padding: 14px; border: 1px solid #2c2c2c; background: #101010; border-radius: 6px; }}
  .verdict {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 10px 12px; border-radius: 6px; margin-bottom: 10px; border: 1px solid #333; }}
  .verdict strong {{ font-size: 22px; }}
  .verdict.pass {{ border-color: #2d6b38; background: #0f1b12; color: #80ef96; }}
  .verdict.review {{ border-color: #84622f; background: #1d160b; color: #ffd18a; }}
  .verdict.block {{ border-color: #8b3333; background: #241010; color: #ff8a8a; }}
  .issue-table code {{ color: #ffcf8a; }}
  .hint {{ margin: -4px 0 12px; color: #aeb7bd; line-height: 1.55; }}
  .badge.ok {{ color: #6ee78b; border-color: #265d31; }}
  .badge.ng {{ color: #ff7777; border-color: #8b3333; }}
  .hero {{ display: grid; grid-template-columns: minmax(260px, 520px) 1fr; gap: 18px; align-items: start; }}
  .hero-img {{ max-width: 100%; border: 1px solid #333; background: #000; }}
  .summary {{ display: grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap: 10px; }}
  .summary div {{ padding: 12px; border: 1px solid #292929; background: #111; border-radius: 6px; }}
  .summary strong {{ display: block; color: #fff; font-size: 20px; }}
  .sequence {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .seq-item {{ padding: 8px 10px; border: 1px solid #3a3121; background: #17120b; color: #ffb02e; border-radius: 6px; }}
  .seq-item.display {{ border-color: #294d69; background: #0d1620; color: #8bd2ff; }}
  .seq-item.logical {{ border-color: #254834; background: #0d1a12; color: #84eba1; }}
  .seq-item small {{ display: block; color: #a99880; margin-top: 2px; }}
  .state-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
  .state-card {{ border: 1px solid #2c2c2c; background: #101010; border-radius: 6px; overflow: hidden; }}
  .display-card {{ border-color: #315879; }}
  .phase-card {{ border-color: #6b5b2f; }}
  .phase-color-card {{ border-color: #7a6636; }}
  .logical-card {{ border-color: #2f6b3e; }}
  .card-head {{ display: flex; justify-content: space-between; gap: 8px; padding: 8px 10px; background: #151515; color: #cfcfcf; }}
  .state-svg {{ display: block; width: 100%; height: 180px; background: #000; }}
  .debug-details {{ margin-top: 28px; border: 1px solid #2c2c2c; border-radius: 6px; background: #0e0e0e; padding: 10px; }}
  .debug-details summary {{ cursor: pointer; color: #bdbdbd; font-size: 16px; font-weight: 700; }}
  .debug-details .sequence {{ margin: 12px 0; }}
  .debug-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
  .debug-grid figure {{ margin: 0; border: 1px solid #2c2c2c; background: #101010; border-radius: 6px; overflow: hidden; }}
  .debug-grid img {{ display: block; max-width: 100%; width: 100%; background: #000; }}
  .debug-grid figcaption {{ padding: 8px 10px; color: #bdbdbd; background: #151515; }}
  .timeline {{ display: flex; flex-wrap: wrap; gap: 3px; align-items: center; }}
  .tick {{ min-width: 34px; padding: 5px 6px; border-radius: 4px; background: #172014; border: 1px solid #294225; font-size: 12px; color: #8ee081; text-align: center; }}
  .tick.event {{ background: #301111; border-color: #8b3030; color: #ff8a8a; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
  th, td {{ border: 1px solid #2c2c2c; padding: 8px; text-align: left; }}
  th {{ background: #141414; }}
  code {{ color: #ffcf8a; }}
  @media (max-width: 800px) {{ .hero {{ grid-template-columns: 1fr; }} }}
</style>
<body>
  <h1>点阵视频学习报告</h1>
  <div class="meta">
    <span class="pill">布局: {html.escape(str(recipe.get('layout_mode', '')))}</span>
    <span class="pill">物理槽位: {slot_count}</span>
    <span class="pill">显示状态: {display_state_count}</span>
    <span class="pill">507 相位: {phase_count}</span>
    <span class="pill">彩色 507: {phase_color_count}</span>
    <span class="pill">原始扫描态: {len(recipe.get('states', []))}</span>
    <span class="pill">ROI: {html.escape(str(recipe.get('panel_bbox', 'none')))}</span>
    {detection_badge}
  </div>
  <div class="guide">
    <strong>阅读顺序：</strong>
    先看左侧槽位叠加图，确认绿色点是否压在真实 LED 上；
    再看“显示模板”，这是最终检测用的完整颜色状态；
    然后看“507 扫描相位模板”，这是相机拍到的两套互补半帧；
    最下面折叠的调试层只用于排查，不要当最终模板。
  </div>
  {diagnostics_section}
  {warning_html}
  <div class="hero">
    <div>{slot_overlay}</div>
    <div class="summary">
      <div><strong>{slot_count}</strong>物理槽位</div>
      <div><strong>{display_state_count}</strong>显示状态</div>
      <div><strong>{phase_count}</strong>507 几何相位</div>
      <div><strong>{html.escape(str(source.get('sampled_count', source.get('frame_count', ''))))}</strong>采样帧</div>
    </div>
  </div>
  {roi_debug}
  {learn_sections}
  {detection_sections}
  <script type="application/json" id="review-data">{html.escape(json.dumps(payload, ensure_ascii=False))}</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render HTML review for dot matrix video learner outputs.")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--learn-dir", type=Path, default=None)
    parser.add_argument("--detection-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    render_review(args.recipe, args.learn_dir, args.detection_dir, args.output)
    print(json.dumps({"review": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
