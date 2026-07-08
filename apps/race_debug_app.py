#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
import sys
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2 as cv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transbot_race.config import RaceConfig  # noqa: E402
from transbot_race.state_machine import RaceStateMachine  # noqa: E402
from transbot_race.vision import draw_debug_overlay, preprocess_blackline, scan_line_features  # noqa: E402


HOST = "127.0.0.1"
PORT = 8776
CONFIG = RaceConfig()


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
    input[type=range] { width:100%; }
    input[type=number], input[type=text], select { width:100%; box-sizing:border-box; border:1px solid #c8d0dc; border-radius:6px; padding:5px; }
    button { border:1px solid #c5cedb; background:#fff; border-radius:7px; padding:8px 10px; cursor:pointer; font-weight:600; }
    button.primary { background:#1463ff; color:#fff; border-color:#1463ff; }
    #image { max-width:100%; border-radius:8px; border:1px solid #d9dee8; background:#111; }
    #log { height:220px; overflow:auto; background:#0b1020; color:#d6e2ff; border-radius:8px; padding:8px; font:12px ui-monospace,SFMono-Regular,Menlo,monospace; white-space:pre-wrap; }
    .hint { color:#667085; font-size:12px; line-height:1.4; }
  </style>
</head>
<body>
  <header>
    <h1>Transbot Race Debug</h1>
    <button onclick="loadConfig()">Reload Config</button>
  </header>
  <main>
    <div>
      <section>
        <h2>Offline Frame Analysis</h2>
        <div class="row"><label>Image path</label><input id="image_path" type="text" value="artifacts/baseline/line_follow_demo_result.jpg"><button class="primary" onclick="analyze()">Analyze</button></div>
        <p class="hint">Use this while SSH is unavailable. It runs the same preprocessing and state-machine step on a saved image.</p>
      </section>
      <img id="image" />
    </div>
    <aside>
      <section>
        <h2>Line Control</h2>
        <div class="row"><label>speed</label><input id="line.speed" type="range" min="0.01" max="0.08" step="0.005"><input id="line.speedn" type="number" step="0.005"></div>
        <div class="row"><label>kp</label><input id="line.kp" type="range" min="0.05" max="0.6" step="0.01"><input id="line.kpn" type="number" step="0.01"></div>
        <div class="row"><label>max_w</label><input id="line.max_w" type="range" min="0.05" max="0.6" step="0.01"><input id="line.max_wn" type="number" step="0.01"></div>
      </section>
      <section>
        <h2>Corner State</h2>
        <div class="row"><label>mode</label><select id="corner.mode"><option>auto</option><option>right</option><option>left</option><option>off</option></select><button onclick="saveConfig()">Apply</button></div>
        <div class="row"><label>trigger</label><input id="vision.trigger_y_frac" type="range" min="0.1" max="0.8" step="0.05"><input id="vision.trigger_y_fracn" type="number" step="0.05"></div>
        <div class="row"><label>confirm</label><input id="corner.confirm_frames" type="range" min="1" max="6" step="1"><input id="corner.confirm_framesn" type="number" step="1"></div>
        <div class="row"><label>forward_s</label><input id="corner.forward_sec" type="range" min="0" max="5" step="0.05"><input id="corner.forward_secn" type="number" step="0.05"></div>
        <div class="row"><label>turn_s</label><input id="corner.turn_sec" type="range" min="0.5" max="4" step="0.05"><input id="corner.turn_secn" type="number" step="0.05"></div>
      </section>
      <section>
        <h2>Noise / Gap</h2>
        <div class="row"><label>min_width</label><input id="vision.min_run_width_px" type="range" min="1" max="40" step="1"><input id="vision.min_run_width_pxn" type="number" step="1"></div>
        <div class="row"><label>min_area</label><input id="vision.min_run_area_px" type="range" min="1" max="200" step="1"><input id="vision.min_run_area_pxn" type="number" step="1"></div>
        <div class="row"><label>blind_s</label><input id="gap.blind_sec" type="range" min="0" max="3" step="0.05"><input id="gap.blind_secn" type="number" step="0.05"></div>
      </section>
      <section>
        <h2>Result</h2>
        <div id="log"></div>
      </section>
    </aside>
  </main>
  <script>
    const ids = ["line.speed","line.kp","line.max_w","vision.trigger_y_frac","corner.confirm_frames","corner.forward_sec","corner.turn_sec","vision.min_run_width_px","vision.min_run_area_px","gap.blind_sec"];
    function log(msg){ const el=document.getElementById("log"); el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\\n` + el.textContent; }
    function bind(id){ const r=document.getElementById(id), n=document.getElementById(id+"n"); const sync=(from)=>{ if(from===r)n.value=r.value; else r.value=n.value; }; r.addEventListener("input",()=>sync(r)); n.addEventListener("input",()=>sync(n)); }
    ids.forEach(bind);
    function setVal(id,v){ document.getElementById(id).value=v; const n=document.getElementById(id+"n"); if(n)n.value=v; }
    function getVal(id){ return Number(document.getElementById(id).value); }
    async function api(path, body){ const res=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})}); const data=await res.json(); if(!res.ok||!data.ok) throw new Error(data.error||res.statusText); return data; }
    function flatten(cfg){ return {line:cfg.line, vision:cfg.vision, corner:cfg.corner, gap:cfg.gap}; }
    async function loadConfig(){ const cfg=await (await fetch("/api/config")).json(); for(const id of ids){ const [a,b]=id.split("."); setVal(id, cfg[a][b]); } document.getElementById("corner.mode").value=cfg.corner.mode; log("config loaded"); }
    async function saveConfig(){ const body={line:{},vision:{},corner:{},gap:{}}; for(const id of ids){ const [a,b]=id.split("."); body[a][b]=getVal(id); } body.corner.mode=document.getElementById("corner.mode").value; await api("/api/config",body); log("config saved"); }
    async function analyze(){ await saveConfig(); const d=await api("/api/analyze",{image_path:document.getElementById("image_path").value}); document.getElementById("image").src="data:image/jpeg;base64,"+d.image; log(JSON.stringify(d.summary,null,2)); }
    loadConfig();
  </script>
</body>
</html>
"""


def _deep_update_cfg(cfg: RaceConfig, data: dict) -> None:
    for section_name in ("vision", "line", "corner", "gap"):
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
                else:
                    setattr(section, key, str(value))


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


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


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Race debug app: http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()

