"""
run_running_led_pipeline — 一键跑完：学习→检测→渲染
"""
from __future__ import annotations
import argparse, csv, json, sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.running_led_core import (
    DEFAULT_THRESHOLDS, KeyState, KeyStatesModel, PanelInfo, StateMachine,
    binarize_scores, crop_panel, debug_write_mask_png, deduplicate_key_states,
    detect_panel_bbox, jsonable, on_count, open_video, rle_encode,
    sample_frames, sample_led_grid, write_json,
)


def main():
    parser = argparse.ArgumentParser(description="跑马灯状态机检测 — 一键管道")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output") / "running_led_run")
    parser.add_argument("--sample-fps", type=float, default=7.5)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--cols", type=int, default=30)
    parser.add_argument("--fw", type=int, default=600, help="fixed width")
    parser.add_argument("--fh", type=int, default=800, help="fixed height")
    parser.add_argument("--on-threshold", type=float, default=0.35)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    debug_dir = out / "debug"; sampled_dir = debug_dir / "sampled_frames"
    masks_dir = debug_dir / "state_masks"; keyframes_dir = out / "keyframes"
    for d in [sampled_dir, masks_dir, keyframes_dir]: d.mkdir(parents=True, exist_ok=True)

    fs = (args.fw, args.fh); slot_count = args.rows * args.cols

    # ─── 1. 打开视频 ─────────────────────────────
    print(f"[1] 打开视频: {args.video}")
    cap, total_frames, src_fps, src_w, src_h = open_video(args.video)
    print(f"    总帧: {total_frames}, FPS: {src_fps:.2f}, 分辨率: {src_w}x{src_h}")

    # ─── 2. 抽帧 ─────────────────────────────────
    print(f"[2] 按 {args.sample_fps} FPS 抽帧...")
    sampled = sample_frames(cap, args.sample_fps, src_fps)
    cap.release()
    print(f"    抽出 {len(sampled)} 帧")

    if len(sampled) == 0:
        print("错误: 视频无帧"); return

    # ─── 3. 面板检测 ─────────────────────────────
    print("[3] 检测面板...")
    bbox = detect_panel_bbox(sampled[0][1])
    if bbox is None:
        print("    警告: 未检测到面板，使用全帧")
        bbox = (0, 0, src_w, src_h)
    else:
        print(f"    面板: {bbox}")

    # ─── 4. 逐帧生成状态矩阵 ────────────────────
    print("[4] 逐帧生成 LED 状态矩阵...")
    masks = []
    for i, (fi, frame, ts) in enumerate(sampled):
        try: panel = crop_panel(frame, bbox, fs)
        except ValueError: panel = cv2.resize(frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]], fs)
        scores = sample_led_grid(panel, args.rows, args.cols)
        binary = binarize_scores(scores, args.on_threshold)
        masks.append(binary)
        if i % 20 == 0:
            cv2.imwrite(str(sampled_dir / f"frame_{fi:06d}.jpg"), panel)
            debug_write_mask_png(masks_dir / f"mask_{fi:06d}.png", binary, args.rows, args.cols)
    print(f"    生成 {len(masks)} 帧矩阵")

    # ─── 5. 关键帧去重 ──────────────────────────
    print("[5] 提取关键状态...")
    key_masks, key_idx = deduplicate_key_states(
        masks, DEFAULT_THRESHOLDS["dedupe_similarity"], int(DEFAULT_THRESHOLDS["min_stable_frames"])
    )
    print(f"    从{len(masks)}帧去重得{len(key_masks)}个关键状态")

    if len(key_masks) < 2:
        print("警告: 关键状态少于2个，可能无法形成有效序列")

    # ─── 6. 构建模型 ────────────────────────────
    hold_counts = []
    for i in range(len(key_idx)):
        end = key_idx[i+1] if i+1 < len(key_idx) else len(masks)
        hold_counts.append(end - key_idx[i])
    med_hold = int(np.median(hold_counts)) if hold_counts else 5
    median_holds = [med_hold] * len(hold_counts) if hold_counts else [5]

    states = []; sequence = []
    for i, (km, ki) in enumerate(zip(key_masks, key_idx)):
        sid = f"K{i:03d}"
        debug_write_mask_png(keyframes_dir / f"{sid}.png", km, args.rows, args.cols)
        states.append(KeyState(sid, i, on_count(km), rle_encode(km),
                               median_holds[i] if i < len(median_holds) else 5,
                               f"keyframes/{sid}.png"))
        sequence.append(sid)

    model = KeyStatesModel(
        version=1, created_at=datetime.now().isoformat(timespec="seconds"),
        source_video=str(args.video), sample_fps=args.sample_fps,
        panel=PanelInfo(args.rows, args.cols, slot_count, fs),
        thresholds=dict(DEFAULT_THRESHOLDS),
        states=states, sequence=sequence,
    )
    model_path = out / "key_states.json"
    write_json(model_path, model.to_dict())
    print(f"    模型: {model_path} ({len(states)} 个状态)")

    # ─── 7. 状态机检测 ─────────────────────────
    print("[6] 状态机检测...")
    sm = StateMachine(model)
    trace_rows = []
    for i, (fi, frame, ts) in enumerate(sampled):
        try: panel = crop_panel(frame, bbox, fs)
        except ValueError: panel = cv2.resize(frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]], fs)
        scores = sample_led_grid(panel, args.rows, args.cols)
        binary = binarize_scores(scores, args.on_threshold)
        result = sm.process_frame(binary, fi, ts)
        trace_rows.append({
            "frame_index": fi, "time_sec": f"{ts:.3f}",
            "expected_state": result.expected_state, "matched_state": result.matched_state,
            "next_state": result.next_state, "similarity_current": f"{result.similarity_current:.4f}",
            "similarity_next": f"{result.similarity_next:.4f}", "transition": result.transition,
            "hold_frames": result.hold_frames, "ok": int(result.ok), "event_type": result.event_type,
        })
    sm.finalize()

    # ─── 8. 输出 trace ──────────────────────────
    trace_path = out / "state_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["frame_index","time_sec","expected_state","matched_state",
                                          "next_state","similarity_current","similarity_next",
                                          "transition","hold_frames","ok","event_type"])
        w.writeheader(); w.writerows(trace_rows)
    print(f"    轨迹: {trace_path}")

    # ─── 9. 输出 events ─────────────────────────
    events_path = out / "events.json"
    write_json(events_path, {"events": [{
        "frame_index": e.frame_index, "time_sec": e.time_sec, "type": e.type,
        "expected_state": e.expected_state, "matched_state": e.matched_state, "message": e.message,
    } for e in sm.events]})
    print(f"    事件: {events_path} ({len(sm.events)} 个)")
    for e in sm.events:
        print(f"      [{e.type}] 帧{e.frame_index}: {e.message}")

    # ─── 10. 输出 summary ────────────────────────
    first_err = None
    for r in trace_rows:
        if not r["ok"]: first_err = {"frame_index": r["frame_index"], "type": r["event_type"]}; break
    summary = {
        "overall_ok": sm.overall_ok, "frame_count": total_frames,
        "sampled_count": len(sampled), "event_count": len(sm.events),
        "key_state_count": len(states), "first_error": first_err,
        "visited_state_count": len(set(r["matched_state"] for r in trace_rows if r["transition"]=="advance")),
    }
    write_json(out / "summary.json", summary)

    ok_str = "OK" if summary["overall_ok"] else "NG"
    print(f"\n===== 结果: {ok_str} =====")
    print(f"  事件数: {summary['event_count']}")
    if first_err: print(f"  第一处异常: 帧{first_err['frame_index']} ({first_err['type']})")

    # ─── 11. 生成 review.html ───────────────────
    print("\n[7] 生成复查页面...")
    _render_html(model, trace_rows, sm.events, out / "review.html")
    print(f"    复查页面: {out / 'review.html'}")


def _render_html(model, trace, events, output_path):
    rows = model.panel.rows if model.panel else 20
    cols = model.panel.cols if model.panel else 30
    states_js = []
    for s in model.states:
        states_js.append({"id": s.id, "index": s.index, "onCount": s.on_count,
                          "mask": s.mask.tolist(), "medianHoldFrames": s.median_hold_frames})

    config = {
        "rows": rows, "cols": cols, "slotCount": model.panel.slot_count if model.panel else rows*cols,
        "sequence": model.sequence, "states": states_js,
        "trace": [{"frameIndex": t["frame_index"], "timeSec": t["time_sec"],
                   "expectedState": t["expected_state"], "matchedState": t["matched_state"],
                   "nextState": t["next_state"], "simCurrent": t["similarity_current"],
                   "simNext": t["similarity_next"], "transition": t["transition"],
                   "holdFrames": t["hold_frames"], "ok": t["ok"], "eventType": t["event_type"]}
                  for t in trace],
        "events": [{"frame_index": e.frame_index if hasattr(e,"frame_index") else e["frame_index"], "type": e.type if hasattr(e,"type") else e["type"], "message": e.message if hasattr(e,"message") else e["message"]} for e in (events if isinstance(events, list) else events.get("events", []))],
    }
    config_json = json.dumps(jsonable(config), ensure_ascii=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_template(config_json, rows, cols), encoding="utf-8")


def _html_template(config_json, rows, cols):
    return f"""<!DOCTYPE html><html lang=zh-CN><meta charset=UTF-8>
<title>跑马灯状态检测复查</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:'Segoe UI',sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;overflow:hidden}}
#app{{display:flex;height:100vh}} #sidebar{{width:220px;min-width:220px;background:#16213e;border-right:1px solid #0f3460;overflow-y:auto;padding:8px}}
#sidebar h3{{font-size:13px;color:#8899cc;margin:8px 0 4px}}
.kf-item{{cursor:pointer;padding:3px;border-radius:4px;display:flex;align-items:center;gap:4px;font-size:11px}}
.kf-item:hover{{background:#1a2744}} .kf-item.active{{background:#1b3a5c;outline:2px solid #4a9eff}}
.kf-item.ng{{outline:2px solid #ff4444}}
.kf-thumb canvas{{border:1px solid #333;width:40px;height:{40*rows//cols}px}}
#main{{flex:1;display:flex;flex-direction:column}} #matrix-panel{{flex:1;display:flex;align-items:center;justify-content:center;background:#0d1117}}
#matrix-panel canvas{{image-rendering:pixelated}}
#info{{height:70px;background:#16213e;border-top:1px solid #0f3460;padding:6px 16px;display:flex;gap:16px;flex-wrap:wrap;font-size:12px}}
.info .l{{color:#667799}} .info .v{{font-weight:600}} .v.ok{{color:#4caf50}} .v.ng{{color:#f44336}}
#timeline{{height:35px;background:#0d1117;border-top:1px solid #0f3460;overflow-x:auto}}
#controls{{height:40px;background:#16213e;border-top:1px solid #0f3460;display:flex;align-items:center;gap:6px;padding:0 10px;font-size:12px}}
#controls button{{background:#1a2744;color:#c0d0e0;border:1px solid #0f3460;padding:4px 10px;border-radius:3px;cursor:pointer}}
#controls button.primary{{background:#1565c0;color:#fff}} #controls button.danger{{background:#c62828;color:#fff}}
#controls input{{flex:1;max-width:300px}}
#evlog{{max-height:36px;overflow-y:auto;font-size:11px;flex:1;min-width:150px}}
.everr{{color:#ff8a80}} .evok{{color:#a5d6a7}}
</style>
<div id=app>
<div id=sidebar><h3>关键帧</h3><div id=kflist></div></div>
<div id=main>
<div id=matrix-panel><canvas id=mc></canvas></div>
<div id=info>
<span class=info><span class=l>帧:</span><span class=v id=fi>--</span></span>
<span class=info><span class=l>时间:</span><span class=v id=ft>--</span></span>
<span class=info><span class=l>期望:</span><span class=v id=fe>--</span></span>
<span class=info><span class=l>匹配:</span><span class=v id=fm>--</span></span>
<span class=info><span class=l>下一:</span><span class=v id=fn>--</span></span>
<span class=info><span class=l>相似度:</span><span class=v id=fsim>--</span></span>
<span class=info><span class=l>过渡:</span><span class=v id=ftr>--</span></span>
<span class=info><span class=l>停留:</span><span class=v id=fh>--</span></span>
<span class=info><span class=l>结果:</span><span class=v id=fres>--</span></span>
</div>
<div id=timeline><svg id=tsvg></svg></div>
<div id=controls>
<button id=bp class=primary>PLAY</button><button id=bpr>&lt;</button>
<span id=flbl>0/0</span><button id=bnx>&gt;</button><button id=bne class=danger>NEXT NG</button>
<input type=range id=sl min=0 max=0 value=0>
<div id=evlog></div></div></div></div>
<script>
(function(){{var C={config_json},R=C.rows,Cl=C.cols,S=C.states,SM={{}};S.forEach(function(s){{SM[s.id]=s}})
var T=C.trace,E=C.events||[],mc=document.getElementById('mc'),ctx=mc.getContext('2d'),cs=12,g=1
mc.width=Cl*(cs+g)+g;mc.height=R*(cs+g)+g
var ci=0,pl=null
function dr(m,h){{ctx.clearRect(0,0,mc.width,mc.height);ctx.fillStyle='#0d1117';ctx.fillRect(0,0,mc.width,mc.height)
for(var r=0;r<R;r++)for(var c=0;c<Cl;c++){{var i=r*Cl+c,x=c*(cs+g)+g,y=r*(cs+g)+g
ctx.fillStyle=i<m.length&&m[i]===1?'#ffb300':'#2a2a3d';ctx.fillRect(x,y,cs,cs)
ctx.strokeStyle='#3a3a50';ctx.lineWidth=.5;ctx.strokeRect(x,y,cs,cs)}}}}
function uf(i){{if(i<0||i>=T.length)return;ci=i;var t=T[i]
var st=SM[t.matchedState||t.expectedState];if(st)dr(st.mask,t.expectedState)
document.getElementById('fi').textContent=t.frameIndex;document.getElementById('ft').textContent=t.timeSec+'s'
document.getElementById('fe').textContent=t.expectedState;document.getElementById('fm').textContent=t.matchedState
document.getElementById('fn').textContent=t.nextState;document.getElementById('fsim').textContent=t.simCurrent
document.getElementById('ftr').textContent=t.transition;document.getElementById('fh').textContent=t.holdFrames
var re=document.getElementById('fres')
if(t.ok){{re.textContent='PASS';re.className='v ok'}}else{{re.textContent=t.eventType||'NG';re.className='v ng'}}
sl.value=i;document.getElementById('flbl').textContent=t.frameIndex+' ('+(i+1)+'/'+T.length+')'
document.querySelectorAll('.kf-item').forEach(function(e){{e.classList.remove('active')
if(e.dataset.sid===t.expectedState)e.classList.add('active')}})
ut(i);ue(i)}}
function ut(ai){{var svg=document.getElementById('tsvg'),w=Math.max(T.length*3,svg.parentElement.clientWidth)
svg.setAttribute('viewBox','0 0 '+w+' 35');svg.innerHTML=''
var bg=document.createElementNS('http://www.w3.org/2000/svg','rect');bg.setAttribute('width',w);bg.setAttribute('height','35');bg.setAttribute('fill','#0d1117');svg.appendChild(bg)
for(var i=0;i<T.length;i++){{var t=T[i],x=i*3,co=t.ok?'#2e7d32':'#c62828'
var r=document.createElementNS('http://www.w3.org/2000/svg','rect')
r.setAttribute('x',x);r.setAttribute('y','13');r.setAttribute('width','2');r.setAttribute('height','9')
r.setAttribute('fill',co);r.setAttribute('rx','1');svg.appendChild(r)}}
var m=document.createElementNS('http://www.w3.org/2000/svg','line')
m.setAttribute('x1',ai*3);m.setAttribute('y1','0');m.setAttribute('x2',ai*3);m.setAttribute('y2','35')
m.setAttribute('stroke','#fff');m.setAttribute('stroke-width','1.5');svg.appendChild(m)}}
function ue(i){{var t=T[i],el=document.getElementById('evlog')
if(t.eventType)el.innerHTML='<span class=everr>'+t.eventType+'</span>'
else if(E.some(function(e){{return e.frame_index===t.frameIndex}})){{
var ev=E.find(function(e){{return e.frame_index===t.frameIndex}})
el.innerHTML='<span class=everr>'+ev.type+': '+ev.message+'</span>'}}else el.innerHTML='<span class=evok>OK</span>'}}
var kf=document.getElementById('kflist')
S.forEach(function(s,i){{var d=document.createElement('div');d.className='kf-item';d.dataset.sid=s.id
var tc=document.createElement('canvas');tc.width=Cl;tc.height=R;tc.style.width='40px';tc.style.height=40*R/Cl+'px';tc.style.imageRendering='pixelated'
var tctx=tc.getContext('2d'),id=tctx.createImageData(Cl,R)
for(var r=0;r<R;r++)for(var c=0;c<Cl;c++){{var idx=r*Cl+c,pi=idx*4
if(idx<s.mask.length&&s.mask[idx]===1){{id.data[pi]=255;id.data[pi+1]=179;id.data[pi+2]=0}}
else{{id.data[pi]=30;id.data[pi+1]=30;id.data[pi+2]=50}}id.data[pi+3]=255}}
tctx.putImageData(id,0,0);d.appendChild(tc);var lb=document.createElement('span');lb.textContent=s.id;d.appendChild(lb)
d.addEventListener('click',function(){{var fi=T.findIndex(function(t){{return t.matchedState===s.id||t.expectedState===s.id}});if(fi>=0)uf(fi)}})
kf.appendChild(d)}})
document.getElementById('bp').addEventListener('click',function(){{pl=pl?null:setInterval(function(){{if(ci>=T.length-1){{clearInterval(pl);pl=null;return}}uf(ci+1)}},100)
this.textContent=pl?'PAUSE':'PLAY'}})
document.getElementById('bpr').addEventListener('click',function(){{uf(Math.max(0,ci-1))}})
document.getElementById('bnx').addEventListener('click',function(){{uf(Math.min(T.length-1,ci+1))}})
document.getElementById('bne').addEventListener('click',function(){{for(var i=ci+1;i<T.length;i++)if(!T[i].ok){{uf(i);return}}}})
document.getElementById('sl').addEventListener('input',function(){{uf(parseInt(this.value))}})
sl.max=T.length-1;var nf=T.findIndex(function(t){{return!t.ok}});uf(nf>=0?nf:0)
window.addEventListener('resize',function(){{ut(ci)}})
}})()
</script></body></html>"""


if __name__ == "__main__":
    main()
