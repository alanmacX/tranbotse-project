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
RUN_PROCS: set[subprocess.Popen] = set()
RUN_LOCK = threading.Lock()
RUN_LOGS: deque[str] = deque(maxlen=240)
LOG_LOCK = threading.Lock()


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
    main { display:grid; grid-template-columns:minmax(640px,1fr) 390px; gap:14px; padding:14px; }
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
        <div class="row"><label>run sec</label><input id="live_max_sec" type="range" min="2" max="120" step="1"><input id="live_max_secn" type="number" step="1"></div>
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
      <section>
        <h2>单帧诊断</h2>
        <div class="row"><label>Image path</label><input id="image_path" type="text" value="artifacts/baseline/line_follow_demo_result.jpg"><button class="primary" onclick="analyze()">Analyze</button></div>
        <p class="hint">只用于检查预处理/状态机判断，不作为主流程。</p>
      </section>
      <img id="video" alt="live video" style="max-width:100%; border-radius:8px; border:1px solid #d9dee8; background:#111; margin-bottom:12px;" />
      <img id="image" />
    </div>
    <aside>
      <section>
        <h2>直线状态机参数</h2>
        <div class="row"><label>speed</label><input id="line.speed" type="range" min="0.01" max="0.08" step="0.005"><input id="line.speedn" type="number" step="0.005"></div>
        <div class="row"><label>kp</label><input id="line.kp" type="range" min="0.05" max="0.6" step="0.01"><input id="line.kpn" type="number" step="0.01"></div>
        <div class="row"><label>max_w</label><input id="line.max_w" type="range" min="0.05" max="0.6" step="0.01"><input id="line.max_wn" type="number" step="0.01"></div>
        <button onclick="resetLegacyDefaults()">恢复旧版实测默认</button>
      </section>
      <section>
        <h2>直角状态机参数</h2>
        <div class="row"><label>mode</label><select id="corner.mode"><option>auto</option><option>right</option><option>left</option><option>off</option></select><button onclick="saveConfig()">Apply</button></div>
        <div class="row"><label>trigger</label><input id="vision.trigger_y_frac" type="range" min="0.1" max="0.8" step="0.05"><input id="vision.trigger_y_fracn" type="number" step="0.05"></div>
        <div class="row"><label>confirm</label><input id="corner.confirm_frames" type="range" min="1" max="6" step="1"><input id="corner.confirm_framesn" type="number" step="1"></div>
        <div class="row"><label>forward_s</label><input id="corner.forward_sec" type="range" min="0" max="5" step="0.05"><input id="corner.forward_secn" type="number" step="0.05"></div>
        <div class="row"><label>turn_w</label><input id="corner.turn_w" type="range" min="0.05" max="0.6" step="0.01"><input id="corner.turn_wn" type="number" step="0.01"></div>
        <div class="row"><label>turn_s</label><input id="corner.turn_sec" type="range" min="0.5" max="4" step="0.05"><input id="corner.turn_secn" type="number" step="0.05"></div>
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
      </section>
      <section>
        <h2>醒目运行日志</h2>
        <div id="log"></div>
      </section>
    </aside>
  </main>
  <script>
    const ids = ["line.speed","line.kp","line.max_w","vision.trigger_y_frac","corner.confirm_frames","corner.forward_sec","corner.turn_w","corner.turn_sec","vision.min_run_width_px","vision.min_run_area_px","gap.blind_sec","camera.crop.0","camera.crop.1","camera.crop.2","camera.crop.3","live_max_sec"];
    function log(msg){ const el=document.getElementById("log"); el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\\n` + el.textContent; }
    function bind(id){ const r=document.getElementById(id), n=document.getElementById(id+"n"); if(!r||!n)return; const sync=(from)=>{ if(from===r)n.value=r.value; else r.value=n.value; }; r.addEventListener("input",()=>sync(r)); n.addEventListener("input",()=>sync(n)); }
    ids.forEach(bind);
    function setVal(id,v){ document.getElementById(id).value=v; const n=document.getElementById(id+"n"); if(n)n.value=v; }
    function getVal(id){ return Number(document.getElementById(id).value); }
    async function api(path, body){ const res=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})}); const data=await res.json(); if(!res.ok||!data.ok) throw new Error(data.error||res.statusText); return data; }
    function flatten(cfg){ return {line:cfg.line, vision:cfg.vision, corner:cfg.corner, gap:cfg.gap}; }
    function readCfgValue(cfg,id){ const parts=id.split("."); let cur=cfg; for(const p of parts){ cur=Array.isArray(cur)?cur[Number(p)]:cur[p]; } return cur; }
    function writeCfgValue(body,id,value){ const parts=id.split("."); let cur=body; for(let i=0;i<parts.length-1;i++){ const p=parts[i]; if(cur[p]===undefined)cur[p]={}; cur=cur[p]; } cur[parts[parts.length-1]]=value; }
    async function loadConfig(){ const cfg=await (await fetch("/api/config")).json(); for(const id of ids){ if(id==="live_max_sec")continue; setVal(id, readCfgValue(cfg,id)); } setVal("live_max_sec",60); document.getElementById("corner.mode").value=cfg.corner.mode; log("CONFIG loaded"); }
    async function saveConfig(){ const body={line:{},vision:{},corner:{},gap:{},camera:{}}; body.camera.crop=[getVal("camera.crop.0"),getVal("camera.crop.1"),getVal("camera.crop.2"),getVal("camera.crop.3")]; for(const id of ids){ if(id==="live_max_sec"||id.startsWith("camera.crop"))continue; writeCfgValue(body,id,getVal(id)); } body.corner.mode=document.getElementById("corner.mode").value; await api("/api/config",body); log("CONFIG saved"); }
    async function analyze(){ await saveConfig(); const d=await api("/api/analyze",{image_path:document.getElementById("image_path").value}); document.getElementById("image").src="data:image/jpeg;base64,"+d.image; log(JSON.stringify(d.summary,null,2)); }
    async function deploy(){ await saveConfig(); const d=await api("/api/deploy",{ssh_target:document.getElementById("ssh_target").value}); log("DEPLOY OK: "+d.message); }
    async function startProfile(profile){ await saveConfig(); const d=await api("/api/live/start",{profile,ssh_target:document.getElementById("ssh_target").value,max_sec:getVal("live_max_sec")}); log("RUNNING "+profile+": "+d.message); setTimeout(refreshLogs,800); }
    async function stopRace(){ const d=await api("/api/live/stop",{ssh_target:document.getElementById("ssh_target").value}); log(d.message); }
    async function resetLegacyDefaults(){ const d=await api("/api/defaults/legacy",{}); log(d.message); await loadConfig(); }
    async function refreshLogs(){ const d=await (await fetch("/api/live/logs")).json(); if(d.logs&&d.logs.length){ document.getElementById("log").textContent=d.logs.join("\\n"); } }
    function reloadVideo(){ document.getElementById("video").src="/video?ssh_target="+encodeURIComponent(document.getElementById("ssh_target").value)+"&ts="+Date.now(); log("video reconnect"); }
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


def _log_event(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with LOG_LOCK:
        RUN_LOGS.appendleft(f"[{stamp}] {message}")


def _read_process_stream(proc: subprocess.Popen, stream_name: str) -> None:
    stream = proc.stdout if stream_name == "stdout" else proc.stderr
    if stream is None:
        return
    for line in stream:
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


def _apply_profile(profile: str) -> str:
    profile = (profile or "final").strip()
    if profile == "line":
        CONFIG.corner.mode = "off"
        CONFIG.gap.enabled = False
        return "直线单状态机：corner=off, gap=off"
    if profile == "corner_right":
        CONFIG.corner.mode = "right"
        CONFIG.gap.enabled = True
        return "右直角单状态机：corner=right, gap=on"
    if profile == "corner_left":
        CONFIG.corner.mode = "left"
        CONFIG.gap.enabled = True
        return "左直角单状态机：corner=left, gap=on"
    if profile == "gap":
        CONFIG.corner.mode = "off"
        CONFIG.gap.enabled = True
        return "虚线/丢线单状态机：corner=off, gap=on"
    if profile == "final":
        CONFIG.corner.mode = "auto"
        CONFIG.gap.enabled = True
        return "最终全流程：corner=auto, gap=on"
    raise ValueError(f"unknown run profile: {profile}")


def _reset_legacy_single_state_defaults() -> None:
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
    CONFIG.vision.trigger_y_frac = 0.35
    CONFIG.line.speed = 0.035
    CONFIG.line.kp = 0.24
    CONFIG.line.max_w = 0.24
    CONFIG.line.slow_on_error = 0.45
    CONFIG.line.max_slowdown = 0.55
    CONFIG.line.invert_turn = False
    CONFIG.corner.mode = "auto"
    CONFIG.corner.confirm_frames = 2
    CONFIG.corner.forward_sec = 1.2
    CONFIG.corner.turn_w = 0.38
    CONFIG.corner.turn_sec = 2.45
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
            _json_response(self, 200, asdict(CONFIG))
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
                _json_response(self, 200, {"ok": True, "config": asdict(CONFIG)})
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
                _log_event("CONFIG: reset to legacy single-state defaults")
                _json_response(self, 200, {"ok": True, "message": "已恢复旧版单项实测默认参数"})
                return
            self.send_error(404)
        except Exception as exc:
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

    def _deploy(self, data: dict) -> dict:
        _write_config_file()
        ssh_target = _ssh_target(data)
        with LOG_LOCK:
            RUN_LOGS.clear()
        _log_event(f"SSH CHECK: target={ssh_target}")
        _check_ssh_ready(ssh_target)
        package = "transbot_race apps/race_runner.py configs/race_config.json requirements.txt"
        cmd = f"tar -czf - {package} | ssh {ssh_target} 'mkdir -p {REMOTE_ROOT} && tar -xzf - -C {REMOTE_ROOT}'"
        res = subprocess.run(cmd, cwd=ROOT, shell=True, text=True, capture_output=True, timeout=45)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip() or "deploy failed")
        _log_event(f"DEPLOYED: {ssh_target}:{REMOTE_ROOT}")
        return {"ok": True, "message": f"deployed integrated race code to {ssh_target}:{REMOTE_ROOT}"}

    def _start_live(self, data: dict) -> dict:
        ssh_target = _ssh_target(data)
        max_sec = max(1.0, min(300.0, float(data.get("max_sec", 60))))
        profile_note = _apply_profile(str(data.get("profile", "final")))
        _write_config_file()
        with LOG_LOCK:
            RUN_LOGS.clear()
        _log_event(f"SSH CHECK: target={ssh_target}")
        _check_ssh_ready(ssh_target)
        stop_live_processes(ssh_target)
        command = (
            f"cd {REMOTE_ROOT} && "
            f"python3 -u apps/race_runner.py --config configs/race_config.json --max-sec {max_sec:.1f}"
        )
        proc = subprocess.Popen(
            ["ssh", ssh_target, command],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.5)
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=1)
            raise RuntimeError((err or out or "race runner exited immediately").strip())
        with RUN_LOCK:
            RUN_PROCS.add(proc)
        _track_process_logs(proc)
        _log_event(f"RUNNING: {profile_note}, max_sec={max_sec:.1f}, target={ssh_target}")
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
            ["ssh", ssh_target, "python3", "-u", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
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
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            proc.terminate()


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Race debug app: http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
