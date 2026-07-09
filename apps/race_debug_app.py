#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import cv2 as cv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transbot_race.config import RaceConfig  # noqa: E402
from transbot_race.state_machine import RaceStateMachine  # noqa: E402
from transbot_race.vision import draw_debug_overlay, preprocess_blackline, scan_line_features  # noqa: E402


HOST = "127.0.0.1"
PORT = 8776
CONFIG = RaceConfig()
SSH_TARGET = "yahboom"
REMOTE_ROOT = "/home/pi/tranbotse-project"
UI_STATE_PATH = ROOT / "configs/race_ui_state.json"
UI_STATE = {
    "cam1": 90,
    "cam2": 12,
    "j1": 55,
    "j2": 205,
    "j3": 45,
    "arm_ms": 1200,
    "live_max_sec": 60,
}
RUN_PROCS: set[subprocess.Popen] = set()
VIDEO_PROCS: set[subprocess.Popen] = set()
RUN_LOCK = threading.Lock()
VIDEO_LOCK = threading.Lock()
RUN_LOGS: deque[str] = deque(maxlen=240)
LOG_LOCK = threading.Lock()


def _load_ui_state() -> None:
    try:
        with UI_STATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            UI_STATE.update({key: data[key] for key in UI_STATE.keys() & data.keys()})
    except FileNotFoundError:
        pass


def _write_ui_state() -> None:
    UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with UI_STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(UI_STATE, f, ensure_ascii=False, indent=2)
        f.write("\n")


_load_ui_state()


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Transbot Race Debug</title>
  <style>
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#111827; background:#f4f6fb; }
    header { display:flex; align-items:center; justify-content:space-between; padding:14px 18px; background:#fff; border-bottom:1px solid #d9dee8; }
    h1 { font-size:20px; margin:0; }
    main { display:grid; grid-template-columns:minmax(560px,1fr) minmax(420px,460px); gap:14px; padding:14px; align-items:start; }
    aside { position:sticky; top:12px; max-height:calc(100vh - 24px); overflow:auto; padding-right:2px; }
    section { background:#fff; border:1px solid #d9dee8; border-radius:8px; padding:12px; margin-bottom:12px; }
    h2 { font-size:14px; margin:0 0 10px; }
    .row { display:grid; grid-template-columns:92px 1fr 70px; gap:8px; align-items:center; margin:8px 0; }
    .two { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .buttons { display:flex; flex-wrap:wrap; gap:8px; }
    input[type=range] { width:100%; }
    input[type=number], input[type=text], select { width:100%; box-sizing:border-box; border:1px solid #c8d0dc; border-radius:6px; padding:5px; }
    button { border:1px solid #c5cedb; background:#fff; border-radius:7px; padding:8px 10px; cursor:pointer; font-weight:600; }
    button.primary { background:#1463ff; color:#fff; border-color:#1463ff; }
    button.danger { background:#d92d20; color:#fff; border-color:#d92d20; }
    button.final { background:#111827; color:#fff; border-color:#111827; }
    .pill { display:inline-block; padding:3px 8px; border-radius:999px; background:#eef4ff; color:#1849a9; font-size:12px; font-weight:700; }
    #image { max-width:100%; border-radius:8px; border:1px solid #d9dee8; background:#111; }
    #log { height:330px; overflow:auto; background:#050816; color:#d6e2ff; border-radius:8px; padding:10px; font:13px ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; border:2px solid #1463ff; }
    .hint { color:#667085; font-size:12px; line-height:1.4; }
    .videoWrap { position:relative; display:inline-block; max-width:100%; margin-bottom:12px; }
    #video { display:block; max-width:100%; border-radius:8px; border:1px solid #d9dee8; background:#111; }
    #cropBox { position:absolute; border:2px solid #ffb020; box-shadow:0 0 0 9999px rgba(0,0,0,.14); pointer-events:none; box-sizing:border-box; }
    @media (max-width: 960px) { main { grid-template-columns:1fr; } aside { position:static; max-height:none; overflow:visible; } }
  </style>
</head>
<body>
  <header>
    <h1>Transbot Race Debug</h1>
    <div>
      <button onclick="loadConfig()">Reload Config</button>
      <button onclick="location.href='http://127.0.0.1:8765'">Legacy App</button>
    </div>
  </header>
  <main>
    <div>
      <section>
        <h2>实机运行入口 <span class="pill">single-state + final</span></h2>
        <div class="row"><label>SSH</label><input id="ssh_target" type="text" value="yahboom"><button onclick="deploy()">Deploy</button></div>
        <div class="row"><label>最长秒</label><input id="live_max_sec" type="range" min="2" max="120" step="1"><input id="live_max_secn" type="number" step="1"></div>
        <div class="buttons">
          <button class="primary" onclick="startProfile('line')">直线单测</button>
          <button class="primary" onclick="startProfile('corner_right')">右直角单测</button>
          <button class="primary" onclick="startProfile('corner_left')">左直角单测</button>
          <button class="primary" onclick="startProfile('gap')">虚线/丢线单测</button>
          <button class="final" onclick="startProfile('final')">最终超级运行</button>
          <button class="danger" onclick="stopRace()">强制停车</button>
          <button onclick="reloadVideo()">Reload Video</button>
        </div>
        <p class="hint">先 Deploy，再跑单项。单项跑稳后用“最终超级运行”；旧版 Safe Tuning 仍保留在 Legacy App。</p>
      </section>
      <div class="videoWrap">
        <img id="video" alt="live video" onload="updateCropBox()" onerror="log('VIDEO FAILED: 请确认 SSH、摄像头、或点击 Reload Video')" />
        <div id="cropBox"></div>
      </div>
      <section>
        <h2>单帧诊断</h2>
        <div class="row"><label>Image path</label><input id="image_path" type="text" value="artifacts/baseline/line_follow_demo_result.jpg"><button class="primary" onclick="analyze()">Analyze</button></div>
        <p class="hint">只用于检查预处理/状态机判断，不作为主流程。</p>
      </section>
      <img id="image" />
    </div>
    <aside>
      <section>
        <h2>巡线 / 直角共用参数</h2>
        <div class="row"><label>速度</label><input id="line.speed" type="range" min="0.01" max="0.06" step="0.005"><input id="line.speedn" type="number" step="0.005"></div>
        <div class="row"><label>转向Kp</label><input id="line.kp" type="range" min="0.05" max="0.6" step="0.01"><input id="line.kpn" type="number" step="0.01"></div>
        <div class="row"><label>最大转向</label><input id="line.max_w" type="range" min="0.05" max="0.6" step="0.01"><input id="line.max_wn" type="number" step="0.01"></div>
        <button onclick="resetLegacyDefaults()">恢复旧版实测默认</button>
      </section>
      <section>
        <h2>相机云台</h2>
        <div class="row"><label>水平</label><input id="ui.cam1" type="range" min="0" max="180" step="1"><input id="ui.cam1n" type="number" step="1"></div>
        <div class="row"><label>俯仰</label><input id="ui.cam2" type="range" min="8" max="100" step="1"><input id="ui.cam2n" type="number" step="1"></div>
        <div class="buttons">
          <button class="primary" onclick="applyCamera()">应用相机</button>
          <button onclick="saveDefaults()">保存默认</button>
        </div>
      </section>
      <section>
        <h2>机械臂停靠位</h2>
        <div class="row"><label>j1</label><input id="ui.j1" type="range" min="0" max="225" step="1"><input id="ui.j1n" type="number" step="1"></div>
        <div class="row"><label>j2</label><input id="ui.j2" type="range" min="30" max="270" step="1"><input id="ui.j2n" type="number" step="1"></div>
        <div class="row"><label>夹爪</label><input id="ui.j3" type="range" min="30" max="180" step="1"><input id="ui.j3n" type="number" step="1"></div>
        <div class="row"><label>时间</label><input id="ui.arm_ms" type="range" min="600" max="2200" step="50"><input id="ui.arm_msn" type="number" step="50"></div>
        <div class="buttons">
          <button class="primary" onclick="applyArm()">应用机械臂</button>
          <button onclick="saveDefaults()">保存默认</button>
        </div>
      </section>
      <section>
        <h2>直角弯参数</h2>
        <div class="row"><label>方向</label><select id="corner.mode"><option value="auto">自动</option><option value="right">右转</option><option value="left">左转</option><option value="off">关闭</option></select><button onclick="saveConfig()">应用</button></div>
        <div class="row"><label>直走秒</label><input id="corner.forward_sec" type="range" min="0" max="5" step="0.05"><input id="corner.forward_secn" type="number" step="0.05"></div>
        <div class="row"><label>转弯秒</label><input id="corner.turn_sec" type="range" min="0.5" max="4" step="0.05"><input id="corner.turn_secn" type="number" step="0.05"></div>
        <div class="row"><label>触发线</label><input id="vision.trigger_y_frac" type="range" min="0.1" max="0.8" step="0.05"><input id="vision.trigger_y_fracn" type="number" step="0.05"></div>
        <div class="row"><label>确认帧</label><input id="corner.confirm_frames" type="range" min="1" max="6" step="1"><input id="corner.confirm_framesn" type="number" step="1"></div>
        <div class="row"><label>转弯角速</label><input id="corner.turn_w" type="range" min="0.05" max="0.6" step="0.01"><input id="corner.turn_wn" type="number" step="0.01"></div>
        <p class="hint">旧版逻辑：分支跨过触发线后，直走指定秒数，再转弯指定秒数。</p>
      </section>
      <section>
        <h2>虚线/细线预处理</h2>
        <div class="row"><label>min_width</label><input id="vision.min_run_width_px" type="range" min="1" max="40" step="1"><input id="vision.min_run_width_pxn" type="number" step="1"></div>
        <div class="row"><label>min_area</label><input id="vision.min_run_area_px" type="range" min="1" max="200" step="1"><input id="vision.min_run_area_pxn" type="number" step="1"></div>
        <div class="row"><label>blind_s</label><input id="gap.blind_sec" type="range" min="0" max="3" step="0.05"><input id="gap.blind_secn" type="number" step="0.05"></div>
      </section>
      <section>
        <h2>Crop / 视野</h2>
        <div class="two">
          <div class="row"><label>x0</label><input id="camera.crop.0" type="range" min="0" max="639" step="1"><input id="camera.crop.0n" type="number" step="1"></div>
          <div class="row"><label>y0</label><input id="camera.crop.1" type="range" min="0" max="479" step="1"><input id="camera.crop.1n" type="number" step="1"></div>
          <div class="row"><label>x1</label><input id="camera.crop.2" type="range" min="1" max="640" step="1"><input id="camera.crop.2n" type="number" step="1"></div>
          <div class="row"><label>y1</label><input id="camera.crop.3" type="range" min="1" max="480" step="1"><input id="camera.crop.3n" type="number" step="1"></div>
        </div>
        <div class="buttons">
          <button onclick="presetCrop(300,265,430,455)">旧版 box</button>
          <button onclick="presetCrop(255,210,455,435)">宽 box</button>
          <button onclick="saveDefaults()">保存默认</button>
        </div>
      </section>
      <section>
        <h2>醒目运行日志</h2>
        <div id="log"></div>
      </section>
    </aside>
  </main>
  <script>
    const ids = ["line.speed","line.kp","line.max_w","vision.trigger_y_frac","corner.confirm_frames","corner.forward_sec","corner.turn_w","corner.turn_sec","vision.min_run_width_px","vision.min_run_area_px","gap.blind_sec","camera.crop.0","camera.crop.1","camera.crop.2","camera.crop.3","ui.cam1","ui.cam2","ui.j1","ui.j2","ui.j3","ui.arm_ms","live_max_sec"];
    function log(msg){ const el=document.getElementById("log"); el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\\n` + el.textContent; }
    function bind(id){ const r=document.getElementById(id), n=document.getElementById(id+"n"); if(!r||!n)return; const sync=(from)=>{ if(from===r)n.value=r.value; else r.value=n.value; if(id.startsWith("camera.crop")) updateCropBox(); }; r.addEventListener("input",()=>sync(r)); n.addEventListener("input",()=>sync(n)); }
    ids.forEach(bind);
    function setVal(id,v){ document.getElementById(id).value=v; const n=document.getElementById(id+"n"); if(n)n.value=v; if(id.startsWith("camera.crop")) updateCropBox(); }
    function getVal(id){ return Number(document.getElementById(id).value); }
    async function api(path, body){ const res=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})}); const data=await res.json(); if(!res.ok||!data.ok) throw new Error(data.error||res.statusText); return data; }
    async function showError(label, fn){ try { return await fn(); } catch(e) { log(label+" FAILED: "+e.message); setTimeout(refreshLogs,500); } }
    function flatten(cfg){ return {line:cfg.line, vision:cfg.vision, corner:cfg.corner, gap:cfg.gap}; }
    function readCfgValue(cfg,id){ const parts=id.split("."); let cur=cfg; for(const p of parts){ cur=Array.isArray(cur)?cur[Number(p)]:cur[p]; } return cur; }
    function writeCfgValue(body,id,value){ const parts=id.split("."); let cur=body; for(let i=0;i<parts.length-1;i++){ const p=parts[i]; if(cur[p]===undefined)cur[p]={}; cur=cur[p]; } cur[parts[parts.length-1]]=value; }
    async function loadConfig(){ const cfg=await (await fetch("/api/config")).json(); for(const id of ids){ setVal(id, id==="live_max_sec" ? cfg.ui.live_max_sec : readCfgValue(cfg,id)); } document.getElementById("corner.mode").value=cfg.corner.mode; log("CONFIG loaded"); reloadVideo(); }
    async function saveConfig(){ const body={line:{},vision:{},corner:{},gap:{},camera:{},ui:{}}; body.camera.crop=[getVal("camera.crop.0"),getVal("camera.crop.1"),getVal("camera.crop.2"),getVal("camera.crop.3")]; for(const id of ids){ if(id.startsWith("camera.crop"))continue; if(id==="live_max_sec"){ body.ui.live_max_sec=getVal(id); continue; } writeCfgValue(body,id,getVal(id)); } body.corner.mode=document.getElementById("corner.mode").value; await api("/api/config",body); log("CONFIG saved"); }
    async function analyze(){ await showError("ANALYZE", async()=>{ await saveConfig(); const d=await api("/api/analyze",{image_path:document.getElementById("image_path").value}); document.getElementById("image").src="data:image/jpeg;base64,"+d.image; log(JSON.stringify(d.summary,null,2)); }); }
    async function deploy(){ await showError("DEPLOY", async()=>{ await saveConfig(); const d=await api("/api/deploy",{ssh_target:document.getElementById("ssh_target").value}); log("DEPLOY OK: "+d.message); }); }
    async function startProfile(profile){ await showError("START "+profile, async()=>{ await saveConfig(); const d=await api("/api/live/start",{profile,ssh_target:document.getElementById("ssh_target").value,max_sec:getVal("live_max_sec")}); log("RUNNING "+profile+": "+d.message); setTimeout(refreshLogs,800); }); }
    async function stopRace(){ await showError("STOP", async()=>{ const d=await api("/api/live/stop",{ssh_target:document.getElementById("ssh_target").value}); log(d.message); }); }
    async function resetLegacyDefaults(){ await showError("DEFAULTS", async()=>{ const d=await api("/api/defaults/legacy",{}); log(d.message); await loadConfig(); }); }
    async function applyCamera(){ await showError("CAMERA", async()=>{ const d=await api("/api/camera",{ssh_target:document.getElementById("ssh_target").value,cam1:getVal("ui.cam1"),cam2:getVal("ui.cam2")}); log(d.message); }); }
    async function applyArm(){ await showError("ARM", async()=>{ const d=await api("/api/arm",{ssh_target:document.getElementById("ssh_target").value,j1:getVal("ui.j1"),j2:getVal("ui.j2"),j3:getVal("ui.j3"),arm_ms:getVal("ui.arm_ms")}); log(d.message); }); }
    async function saveDefaults(){ await showError("SAVE DEFAULTS", async()=>{ await saveConfig(); const d=await api("/api/defaults/save",{ui:{cam1:getVal("ui.cam1"),cam2:getVal("ui.cam2"),j1:getVal("ui.j1"),j2:getVal("ui.j2"),j3:getVal("ui.j3"),arm_ms:getVal("ui.arm_ms"),live_max_sec:getVal("live_max_sec")}}); log(d.message); }); }
    function presetCrop(x0,y0,x1,y1){ setVal("camera.crop.0",x0); setVal("camera.crop.1",y0); setVal("camera.crop.2",x1); setVal("camera.crop.3",y1); log(`box=[${x0},${y0},${x1},${y1}]`); }
    function updateCropBox(){
      const img=document.getElementById("video"), box=document.getElementById("cropBox");
      if(!img||!box)return;
      const nw=img.naturalWidth||640, nh=img.naturalHeight||480;
      const sx=img.clientWidth/nw, sy=img.clientHeight/nh;
      const x0=getVal("camera.crop.0"), y0=getVal("camera.crop.1"), x1=getVal("camera.crop.2"), y1=getVal("camera.crop.3");
      box.style.left=(x0*sx)+"px"; box.style.top=(y0*sy)+"px";
      box.style.width=Math.max(1,(x1-x0)*sx)+"px"; box.style.height=Math.max(1,(y1-y0)*sy)+"px";
    }
    async function refreshLogs(){ try { const d=await (await fetch("/api/live/logs")).json(); if(d.logs&&d.logs.length){ document.getElementById("log").textContent=d.logs.join("\\n"); } } catch(e) { log("LOG REFRESH FAILED: "+e.message); } }
    function reloadVideo(){ document.getElementById("video").src="/video?ssh_target="+encodeURIComponent(document.getElementById("ssh_target").value)+"&ts="+Date.now(); log("video reconnect"); }
    window.addEventListener("resize", updateCropBox);
    setInterval(refreshLogs,1200);
    loadConfig();
  </script>
</body>
</html>
"""


def _deep_update_cfg(cfg: RaceConfig, data: dict) -> None:
    for section_name in ("camera", "vision", "line", "corner", "gap"):
        section = getattr(cfg, section_name)
        values = data.get(section_name)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if hasattr(section, key):
                current = getattr(section, key)
                if isinstance(current, bool):
                    setattr(section, key, bool(value))
                elif isinstance(current, int):
                    setattr(section, key, int(value))
                elif isinstance(current, float):
                    setattr(section, key, float(value))
                elif isinstance(current, tuple) and isinstance(value, list):
                    setattr(section, key, tuple(int(item) for item in value))
                else:
                    setattr(section, key, str(value))


def _load_config_file() -> None:
    path = ROOT / "configs/race_config.json"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        _deep_update_cfg(CONFIG, data)


def _clamp_int(value: object, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _config_payload() -> dict:
    payload = asdict(CONFIG)
    payload["ui"] = dict(UI_STATE)
    return payload


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _write_config_file() -> None:
    path = ROOT / "configs/race_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(CONFIG), f, ensure_ascii=False, indent=2)
        f.write("\n")


def _ssh_target(data: dict) -> str:
    value = str(data.get("ssh_target") or SSH_TARGET).strip()
    value = value or SSH_TARGET
    if not re.fullmatch(r"[A-Za-z0-9_.@-]+", value):
        raise ValueError("ssh_target contains unsupported characters")
    return value


def _transbot_code(body: str) -> str:
    return f"""
import sys, time
sys.path.insert(0, '/home/pi/Transbot/py_install')
from Transbot_Lib import Transbot
bot = Transbot()
try:
    for _ in range(2):
        bot.set_car_motion(0, 0)
        time.sleep(0.03)
{body}
    for _ in range(2):
        bot.set_car_motion(0, 0)
        time.sleep(0.03)
finally:
    pass
"""


def _remote_python(ssh_target: str, code: str, timeout: float = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", ssh_target, "python3", "-"],
        input=code,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def _log_event(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with LOG_LOCK:
        RUN_LOGS.appendleft(f"[{stamp}] {message}")


def _read_process_stream(proc: subprocess.Popen, stream_name: str) -> None:
    stream = proc.stdout if stream_name == "stdout" else proc.stderr
    if stream is None:
        return
    for line in stream:
        if isinstance(line, bytes):
            text = line.decode("utf-8", errors="replace").rstrip()
        else:
            text = line.rstrip()
        if text:
            _log_event(f"{stream_name}: {text}")


def _track_process_logs(proc: subprocess.Popen) -> None:
    for stream_name in ("stdout", "stderr"):
        thread = threading.Thread(target=_read_process_stream, args=(proc, stream_name), daemon=True)
        thread.start()


def _check_ssh_ready(ssh_target: str) -> None:
    res = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            ssh_target,
            "echo transbot_ssh_ready",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=7,
    )
    if res.returncode != 0 or "transbot_ssh_ready" not in res.stdout:
        detail = (res.stderr or res.stdout or "ssh probe failed").strip()
        raise RuntimeError(f"SSH CHECK FAILED for {ssh_target}: {detail}")


def _check_remote_runner(ssh_target: str) -> None:
    res = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            ssh_target,
            f"test -f {REMOTE_ROOT}/apps/race_runner.py && echo transbot_runner_ready",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=7,
    )
    if res.returncode != 0 or "transbot_runner_ready" not in res.stdout:
        detail = (res.stderr or res.stdout or f"{REMOTE_ROOT}/apps/race_runner.py not found").strip()
        raise RuntimeError(f"DEPLOY REQUIRED on {ssh_target}: {detail}")


def _sync_live_files(ssh_target: str) -> None:
    package = "transbot_race apps/race_runner.py configs/race_config.json"
    cmd = f"tar -czf - {package} | ssh {ssh_target} 'mkdir -p {REMOTE_ROOT} && tar -xzf - -C {REMOTE_ROOT}'"
    res = subprocess.run(cmd, cwd=ROOT, shell=True, text=True, capture_output=True, timeout=45)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "live code sync failed")
    _log_event(f"CODE SYNCED: runner/state/config -> {ssh_target}:{REMOTE_ROOT}")


def _apply_profile(profile: str) -> str:
    profile = (profile or "final").strip()
    if profile == "line":
        CONFIG.corner.mode = "off"
        CONFIG.gap.enabled = False
        return "直线单状态机：corner=off, gap=off"
    if profile == "corner_right":
        # Single-corner tests use the first-commit behavior: detect a corner,
        # then force the physical turn direction from the button.
        CONFIG.corner.mode = "auto"
        CONFIG.corner.right_turn_dir = -1.0
        CONFIG.corner.left_turn_dir = -1.0
        CONFIG.gap.enabled = True
        return "右直角单状态机：检测任意直角后强制右转"
    if profile == "corner_left":
        CONFIG.corner.mode = "auto"
        CONFIG.corner.right_turn_dir = 1.0
        CONFIG.corner.left_turn_dir = 1.0
        CONFIG.gap.enabled = True
        return "左直角单状态机：检测任意直角后强制左转"
    if profile == "gap":
        CONFIG.corner.mode = "off"
        CONFIG.gap.enabled = True
        return "虚线/丢线单状态机：corner=off, gap=on"
    if profile == "final":
        CONFIG.corner.mode = "auto"
        CONFIG.corner.right_turn_dir = -1.0
        CONFIG.corner.left_turn_dir = 1.0
        CONFIG.gap.enabled = True
        return "最终全流程：corner=auto, gap=on"
    raise ValueError(f"unknown run profile: {profile}")


def _reset_legacy_single_state_defaults() -> None:
    UI_STATE.update({"cam1": 90, "cam2": 12, "j1": 55, "j2": 205, "j3": 45, "arm_ms": 1200, "live_max_sec": 60})
    CONFIG.camera.crop = (300, 265, 430, 455)
    CONFIG.camera.expand_left_px = 20
    CONFIG.camera.expand_right_px = 140
    CONFIG.vision.percentile = 32
    CONFIG.vision.threshold_min = 25
    CONFIG.vision.threshold_max = 120
    CONFIG.vision.active_col_ratio = 0.18
    CONFIG.vision.min_run_width_px = 10
    CONFIG.vision.min_run_area_px = 30
    CONFIG.vision.branch_width_ratio = 2.2
    CONFIG.vision.branch_min_crop_ratio = 0.22
    CONFIG.vision.trigger_y_frac = 0.30
    CONFIG.line.speed = 0.06
    CONFIG.line.kp = 0.24
    CONFIG.line.max_w = 0.24
    CONFIG.line.slow_on_error = 0.45
    CONFIG.line.max_slowdown = 0.55
    CONFIG.line.invert_turn = False
    CONFIG.corner.mode = "right"
    CONFIG.corner.confirm_frames = 2
    CONFIG.corner.forward_sec = 5.0
    CONFIG.corner.turn_w = 0.38
    CONFIG.corner.turn_sec = 2.3
    CONFIG.corner.right_turn_dir = -1.0
    CONFIG.corner.left_turn_dir = 1.0
    CONFIG.corner.reacquire_confirm_frames = 3
    CONFIG.corner.reacquire_err_norm = 0.50
    CONFIG.gap.enabled = True
    CONFIG.gap.missing_frames = 4
    CONFIG.gap.blind_sec = 1.2
    CONFIG.gap.blind_speed_factor = 0.70
    CONFIG.gap.blind_turn_factor = 0.35
    CONFIG.gap.search_w = 0.16


def stop_live_processes(ssh_target: str) -> None:
    release_video_processes()
    with RUN_LOCK:
        for proc in list(RUN_PROCS):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            RUN_PROCS.discard(proc)
    try:
        subprocess.run(
            [
                "ssh",
                ssh_target,
                "sh",
                "-lc",
                "pkill -f 'apps/race_runner.py' || true; "
                "pkill -f 'python3 -u -' || true; "
                "python3 - <<'PY'\nimport sys, time\nsys.path.insert(0, '/home/pi/Transbot/py_install')\nfrom Transbot_Lib import Transbot\nbot=Transbot()\nfor _ in range(30):\n    bot.set_car_motion(0, 0)\n    time.sleep(0.04)\nPY",
            ],
            text=True,
            capture_output=True,
            timeout=12,
        )
        _log_event("STOP: remote runner/video killed, chassis zeroed 30x")
    except Exception:
        _log_event("STOP: local runner stopped; remote stop command failed")


def release_video_processes(ssh_target: str | None = None) -> None:
    with VIDEO_LOCK:
        procs = list(VIDEO_PROCS)
        VIDEO_PROCS.clear()
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=0.8)
            except subprocess.TimeoutExpired:
                proc.kill()
    if ssh_target:
        subprocess.run(
            [
                "ssh",
                ssh_target,
                "sh",
                "-lc",
                "ps -eo pid=,args= | awk '$0 ~ /python3 -u -/ {print $1}' | xargs -r kill",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=4,
        )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/":
            raw = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path == "/api/config":
            _json_response(self, 200, _config_payload())
            return
        if self.path == "/api/live/logs":
            with LOG_LOCK:
                logs = list(RUN_LOGS)
            _json_response(self, 200, {"ok": True, "logs": logs})
            return
        if self.path.startswith("/video"):
            self.stream_video()
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if self.path == "/api/config":
                _deep_update_cfg(CONFIG, data)
                if isinstance(data.get("ui"), dict):
                    UI_STATE.update({key: data["ui"][key] for key in UI_STATE.keys() & data["ui"].keys()})
                _json_response(self, 200, {"ok": True, "config": _config_payload()})
                return
            if self.path == "/api/camera":
                _json_response(self, 200, self._camera(data))
                return
            if self.path == "/api/arm":
                _json_response(self, 200, self._arm(data))
                return
            if self.path == "/api/analyze":
                _json_response(self, 200, self._analyze(data))
                return
            if self.path == "/api/deploy":
                _json_response(self, 200, self._deploy(data))
                return
            if self.path == "/api/live/start":
                _json_response(self, 200, self._start_live(data))
                return
            if self.path == "/api/live/stop":
                _json_response(self, 200, self._stop_live(data))
                return
            if self.path == "/api/defaults/legacy":
                _reset_legacy_single_state_defaults()
                _write_config_file()
                _write_ui_state()
                _log_event("CONFIG: reset to legacy single-state defaults")
                _json_response(self, 200, {"ok": True, "message": "已恢复旧版单项、box、相机、机械臂默认参数"})
                return
            if self.path == "/api/defaults/save":
                _deep_update_cfg(CONFIG, data)
                if isinstance(data.get("ui"), dict):
                    UI_STATE.update({key: data["ui"][key] for key in UI_STATE.keys() & data["ui"].keys()})
                _write_config_file()
                _write_ui_state()
                _log_event("DEFAULTS SAVED: config + camera/arm/ui")
                _json_response(self, 200, {"ok": True, "message": "已保存当前 config、box、相机、机械臂为默认值"})
                return
            self.send_error(404)
        except Exception as exc:
            _log_event(f"ERROR: {exc}")
            _json_response(self, 500, {"ok": False, "error": str(exc)})

    def _analyze(self, data: dict) -> dict:
        path = Path(str(data.get("image_path", "artifacts/baseline/line_follow_demo_result.jpg")))
        if not path.is_absolute():
            path = ROOT / path
        frame = cv.imread(str(path))
        if frame is None:
            raise RuntimeError(f"cannot read image: {path}")
        mask = preprocess_blackline(frame, CONFIG.vision)
        features = scan_line_features(mask, CONFIG.vision)
        sm = RaceStateMachine(CONFIG)
        command = sm.step(features, now=time.monotonic())
        overlay = draw_debug_overlay(frame, features, CONFIG.vision.trigger_y_frac)
        ok, buf = cv.imencode(".jpg", overlay, [int(cv.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            raise RuntimeError("failed to encode debug image")
        summary = {
            "state": command.state.value,
            "reason": command.reason,
            "v": round(command.v, 4),
            "w": round(command.w, 4),
            "found": features.found,
            "err_norm": round(features.err_norm, 4),
            "line_width_px": round(features.line_width_px, 2),
            "branch_left": None if features.branch_left is None else features.branch_left.y,
            "branch_right": None if features.branch_right is None else features.branch_right.y,
        }
        return {"ok": True, "summary": summary, "image": base64.b64encode(buf).decode("ascii")}

    def _camera(self, data: dict) -> dict:
        ssh_target = _ssh_target(data)
        cam1 = _clamp_int(data.get("cam1", UI_STATE["cam1"]), 0, 180)
        cam2 = _clamp_int(data.get("cam2", UI_STATE["cam2"]), 8, 100)
        _check_ssh_ready(ssh_target)
        code = _transbot_code(f"""
    bot.set_pwm_servo(1, {cam1})
    time.sleep(0.08)
    bot.set_pwm_servo(2, {cam2})
    time.sleep(0.18)
""")
        res = _remote_python(ssh_target, code, timeout=8)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "camera servo failed")
        UI_STATE["cam1"], UI_STATE["cam2"] = cam1, cam2
        _write_ui_state()
        _log_event(f"CAMERA: servo1={cam1}, servo2={cam2}")
        return {"ok": True, "message": f"camera servo1={cam1}, servo2={cam2}"}

    def _arm(self, data: dict) -> dict:
        ssh_target = _ssh_target(data)
        j1 = _clamp_int(data.get("j1", UI_STATE["j1"]), 0, 225)
        j2 = _clamp_int(data.get("j2", UI_STATE["j2"]), 30, 270)
        j3 = _clamp_int(data.get("j3", UI_STATE["j3"]), 30, 180)
        arm_ms = _clamp_int(data.get("arm_ms", UI_STATE["arm_ms"]), 600, 2200)
        _check_ssh_ready(ssh_target)
        code = _transbot_code(f"""
    bot.set_uart_servo_angle_array({j1}, {j2}, {j3}, {arm_ms})
    time.sleep({arm_ms / 1000.0 + 0.25:.2f})
""")
        res = _remote_python(ssh_target, code, timeout=max(6, arm_ms / 1000 + 4))
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "arm servo failed")
        UI_STATE["j1"], UI_STATE["j2"], UI_STATE["j3"], UI_STATE["arm_ms"] = j1, j2, j3, arm_ms
        _write_ui_state()
        _log_event(f"ARM: j1={j1}, j2={j2}, j3={j3}, {arm_ms}ms")
        return {"ok": True, "message": f"arm j1={j1}, j2={j2}, j3={j3}, {arm_ms}ms"}

    def _deploy(self, data: dict) -> dict:
        _write_config_file()
        ssh_target = _ssh_target(data)
        with LOG_LOCK:
            RUN_LOGS.clear()
        _log_event(f"SSH CHECK: target={ssh_target}")
        _check_ssh_ready(ssh_target)
        _sync_live_files(ssh_target)
        _log_event(f"DEPLOYED: {ssh_target}:{REMOTE_ROOT}")
        return {"ok": True, "message": f"deployed integrated race code to {ssh_target}:{REMOTE_ROOT}"}

    def _start_live(self, data: dict) -> dict:
        ssh_target = _ssh_target(data)
        max_sec = max(1.0, min(300.0, float(data.get("max_sec", 60))))
        profile = str(data.get("profile", "final"))
        force_turn_dir = 0.0
        with LOG_LOCK:
            RUN_LOGS.clear()
        _log_event(f"SSH CHECK: target={ssh_target}")
        _check_ssh_ready(ssh_target)
        _check_remote_runner(ssh_target)
        profile_note = _apply_profile(profile)
        _write_config_file()
        _sync_live_files(ssh_target)
        stop_live_processes(ssh_target)
        corner_stop_flag = " --stop-after-corner" if profile in {"corner_right", "corner_left"} else ""
        if profile == "corner_right":
            force_turn_dir = -1.0
        elif profile == "corner_left":
            force_turn_dir = 1.0
        force_turn_flag = f" --force-turn-dir {force_turn_dir:.1f}" if force_turn_dir else ""
        command = (
            f"cd {REMOTE_ROOT} && "
            f"python3 -u apps/race_runner.py --config configs/race_config.json --max-sec {max_sec:.1f}{corner_stop_flag}{force_turn_flag}"
        )
        proc = subprocess.Popen(
            ["ssh", ssh_target, command],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(1.25)
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=1)
            raise RuntimeError((err or out or "race runner exited immediately").strip())
        with RUN_LOCK:
            RUN_PROCS.add(proc)
        _track_process_logs(proc)
        _log_event(f"RUNNING: {profile_note}, max_sec={max_sec:.1f}, force_turn_dir={force_turn_dir:.1f}, target={ssh_target}")
        return {"ok": True, "message": f"{profile_note}; runner on {ssh_target} for {max_sec:.1f}s"}

    def _stop_live(self, data: dict) -> dict:
        ssh_target = _ssh_target(data)
        stop_live_processes(ssh_target)
        return {"ok": True, "message": "stop sent"}

    def stream_video(self) -> None:
        ssh_target = SSH_TARGET
        if "?" in self.path:
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(self.path).query)
            ssh_target = _ssh_target({"ssh_target": query.get("ssh_target", [SSH_TARGET])[0]})
        release_video_processes(ssh_target)
        _log_event(f"VIDEO: connecting {ssh_target}")
        code = r"""
import cv2 as cv, sys, time
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit('camera open failed')
while True:
    ok, frame = cap.read()
    if not ok:
        time.sleep(0.05)
        continue
    ok, buf = cv.imencode('.jpg', frame, [int(cv.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        continue
    data = buf.tobytes()
    sys.stdout.buffer.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
    sys.stdout.buffer.flush()
    time.sleep(0.08)
"""
        proc = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", ssh_target, "python3", "-u", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        with VIDEO_LOCK:
            VIDEO_PROCS.add(proc)
        threading.Thread(target=_read_process_stream, args=(proc, "stderr"), daemon=True).start()
        assert proc.stdin is not None
        proc.stdin.write(code.encode("utf-8"))
        proc.stdin.close()
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            assert proc.stdout is not None
            while True:
                chunk = proc.stdout.read(8192)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        finally:
            proc.terminate()
            with VIDEO_LOCK:
                VIDEO_PROCS.discard(proc)


def main() -> None:
    _load_config_file()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Race debug app: http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
