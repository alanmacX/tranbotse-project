#!/usr/bin/env python3
"""Fail-safe step-by-step Transbot controller with complete run recording."""
from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apps.race_runner import make_bot, stop_chassis  # noqa: E402
from transbot_race.path_memory import read_motion_sample  # noqa: E402


INDEX_HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Transbot 手动保底台</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#142033;background:#eef2f7}body{margin:0}header{padding:14px 18px;background:#142033;color:white;display:flex;justify-content:space-between;align-items:center}h1{font-size:18px;margin:0}.badge{padding:5px 9px;border-radius:99px;background:#506078}.busy{background:#a66b00}.ok{background:#18794e}.grid{display:grid;grid-template-columns:minmax(480px,1fr) 390px;gap:14px;padding:14px}.card{background:white;border:1px solid #d5dce7;border-radius:10px;padding:13px}#video{width:100%;max-width:800px;background:#111;border-radius:8px}.controls{display:grid;grid-template-columns:1fr 1fr;gap:9px}.controls button{min-height:54px;font-size:16px}button{border:1px solid #b9c4d3;border-radius:8px;background:white;font-weight:700;padding:9px;cursor:pointer}button:disabled{opacity:.4}.danger{background:#bf1d36;color:white;border-color:#bf1d36}.primary{background:#1769e0;color:white;border-color:#1769e0}.row{display:grid;grid-template-columns:1fr 110px;align-items:center;gap:8px;margin:9px 0}input{padding:8px;border:1px solid #b9c4d3;border-radius:7px;width:90px}#log{height:190px;overflow:auto;background:#0d1524;color:#d8e5ff;border-radius:8px;padding:9px;font:12px ui-monospace,monospace;white-space:pre-wrap}.hint{font-size:12px;color:#64748b;line-height:1.5}.stop{width:100%;font-size:20px;min-height:62px;margin-bottom:10px}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body><header><h1>Transbot 手动保底台</h1><span id="status" class="badge">连接中</span></header>
<div class="grid"><section class="card"><img id="video" src="/video.mjpg"><p class="hint">视频与每次动作的运动采样持续保存；页面关闭不会让动作无限继续，后端有目标量和最长时间双重限位。</p></section>
<aside><section class="card"><button class="danger stop" onclick="stopNow()">立即停车</button><div class="row"><label>前后距离（m）</label><input id="distance" type="number" min="0.01" max="0.50" step="0.01" value="0.05"></div><div class="row"><label>旋转角度（°）</label><input id="angle" type="number" min="1" max="180" step="1" value="15"></div><div class="row"><label>线速度（m/s）</label><input id="v" type="number" min="0.01" max="0.08" step="0.005" value="0.04"></div><div class="row"><label>角速度（rad/s）</label><input id="w" type="number" min="0.05" max="0.40" step="0.01" value="0.20"></div><div class="controls"><button class="primary" onclick="move('forward')">前进一步</button><button onclick="move('backward')">后退一步</button><button onclick="move('left')">左转指定角度</button><button onclick="move('right')">右转指定角度</button></div></section>
<section class="card"><b>完整动作日志</b><div id="log"></div><p id="run" class="hint"></p></section></aside></div>
<script>
const $=id=>document.getElementById(id);let busy=false;function log(x){$('log').textContent=new Date().toLocaleTimeString()+' '+x+'\n'+$('log').textContent}
async function api(path,body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d}
async function move(action){if(busy)return;try{const d=await api('/api/action',{action,distance:Number($('distance').value),angle_deg:Number($('angle').value),v:Number($('v').value),w:Number($('w').value)});log('已提交 '+d.action_id+' '+action)}catch(e){log('错误 '+e.message)}}
async function stopNow(){try{const d=await api('/api/stop',{});log(d.message)}catch(e){log('停车错误 '+e.message)}}
async function poll(){try{const r=await fetch('/api/status');const d=await r.json();busy=d.busy;$('status').textContent=busy?'动作执行中':'已停车';$('status').className='badge '+(busy?'busy':'ok');$('run').textContent='记录目录：'+d.run_dir;document.querySelectorAll('.controls button').forEach(b=>b.disabled=busy);if(d.last_result&&d.last_result.id!==window.lastId){window.lastId=d.last_result.id;log(JSON.stringify(d.last_result))}}catch(e){$('status').textContent='连接断开';$('status').className='badge'}setTimeout(poll,300)}poll();
</script></body></html>"""


@dataclass(frozen=True)
class StepAction:
    action: str
    target: float
    v: float
    w: float


class Recorder:
    def __init__(self, base: Path, args: argparse.Namespace) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.root = base / f"manual_{stamp}"
        self.frames = self.root / "frames"
        self.actions = self.root / "actions"
        self.frames.mkdir(parents=True, exist_ok=False)
        self.actions.mkdir()
        self.lock = threading.Lock()
        self.action_log = (self.root / "actions.jsonl").open("a", encoding="utf-8", buffering=1)
        self.motion_log = (self.root / "motion.jsonl").open("a", encoding="utf-8", buffering=1)
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=2,
            ).stdout.strip() or None
            git_dirty = bool(subprocess.run(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=2,
            ).stdout.strip())
        except Exception:
            git_commit, git_dirty = None, None
        (self.root / "meta.json").write_text(json.dumps({
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "camera": args.camera, "dry_run": args.dry_run,
            "camera_enabled": (not args.dry_run or args.allow_local_camera),
            "frame_width": args.width, "frame_height": args.height,
            "record_fps": args.record_fps,
            "git_commit": git_commit, "git_dirty": git_dirty,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def elapsed(self) -> float:
        return time.monotonic() - self.started_mono

    def write_event(self, event: dict) -> None:
        row = {"t": round(self.elapsed(), 4), **event}
        with self.lock:
            self.action_log.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_motion(self, row: dict) -> None:
        with self.lock:
            self.motion_log.write(json.dumps({"t": round(self.elapsed(), 4), **row}, ensure_ascii=False) + "\n")

    def close(self) -> None:
        with self.lock:
            self.action_log.close(); self.motion_log.close()
        (self.root / "manifest.json").write_text(json.dumps({
            "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "duration_sec": round(time.time() - self.started_wall, 3),
            "actions": "actions.jsonl", "motion": "motion.jsonl", "frames": "frames/",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class CameraThread(threading.Thread):
    def __init__(self, args: argparse.Namespace, recorder: Recorder) -> None:
        super().__init__(daemon=True)
        self.args, self.recorder = args, recorder
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_frame = None
        self.frame_index = 0

    def _set_placeholder(self) -> None:
        frame = np.full((self.args.height, self.args.width, 3), 28, dtype=np.uint8)
        cv.putText(
            frame, "DRY RUN - LOCAL CAMERA DISABLED", (35, self.args.height // 2),
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv.LINE_AA,
        )
        ok, enc = cv.imencode(".jpg", frame)
        with self.lock:
            self.latest_frame = frame
            self.latest_jpeg = enc.tobytes() if ok else None

    def run(self) -> None:
        if self.args.dry_run and not self.args.allow_local_camera:
            self._set_placeholder()
            self.stop_event.wait()
            return
        cap = cv.VideoCapture(self.args.camera)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.args.width); cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.args.height)
        last_save = -1e9
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05); continue
            ok_j, enc = cv.imencode(".jpg", frame, [int(cv.IMWRITE_JPEG_QUALITY), 82])
            if ok_j:
                with self.lock:
                    self.latest_frame = frame.copy(); self.latest_jpeg = enc.tobytes()
            now = self.recorder.elapsed()
            if now - last_save >= 1.0 / max(0.2, self.args.record_fps):
                stem = f"{self.frame_index:06d}_{int(now*1000):08d}.jpg"
                cv.imwrite(str(self.recorder.frames / stem), frame, [int(cv.IMWRITE_JPEG_QUALITY), 88])
                self.frame_index += 1; last_save = now
        cap.release()

    def snapshot(self, name: str) -> str | None:
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is None:
            return None
        path = self.recorder.actions / f"{name}.jpg"
        cv.imwrite(str(path), frame, [int(cv.IMWRITE_JPEG_QUALITY), 92])
        return str(path.relative_to(self.recorder.root))


class StepController:
    def __init__(self, bot, camera: CameraThread, recorder: Recorder) -> None:
        self.bot, self.camera, self.recorder = bot, camera, recorder
        self.lock = threading.Lock(); self.stop_event = threading.Event()
        self.command_lock = threading.Lock()
        self.busy = False; self.last_result = None; self.counter = 0

    def parse(self, data: dict) -> StepAction:
        action = str(data.get("action", ""))
        if action not in {"forward", "backward", "left", "right"}:
            raise ValueError("action must be forward/backward/left/right")
        v = max(0.01, min(0.08, float(data.get("v", 0.04))))
        w = max(0.05, min(0.40, float(data.get("w", 0.20))))
        if action in {"forward", "backward"}:
            target = max(0.005, min(0.50, float(data.get("distance", 0.05))))
        else:
            target = math.radians(max(1.0, min(180.0, float(data.get("angle_deg", 15.0)))))
        return StepAction(action, target, v, w)

    def submit(self, action: StepAction) -> int:
        with self.lock:
            if self.busy:
                raise RuntimeError("another action is still running")
            self.busy = True; self.stop_event.clear(); self.counter += 1; action_id = self.counter
        threading.Thread(target=self._execute, args=(action_id, action), daemon=True).start()
        return action_id

    def stop(self) -> None:
        with self.command_lock:
            self.stop_event.set(); stop_chassis(self.bot, count=4, delay=0.02)
        self.recorder.write_event({"event": "emergency_stop_requested"})

    def _execute(self, action_id: int, action: StepAction) -> None:
        sign = 1.0 if action.action in {"forward", "left"} else -1.0
        cmd_v = sign * action.v if action.action in {"forward", "backward"} else 0.0
        cmd_w = sign * action.w if action.action in {"left", "right"} else 0.0
        nominal = action.target / max(abs(cmd_v) if cmd_v else abs(cmd_w), 1e-6)
        timeout = min(20.0, nominal * 1.7 + 1.0)
        started = time.monotonic(); progress = 0.0; last = started
        before = self.camera.snapshot(f"{action_id:04d}_before")
        self.recorder.write_event({"event":"action_start","id":action_id,"action":action.action,"target":action.target,"cmd_v":cmd_v,"cmd_w":cmd_w,"before":before})
        result = "target_reached"
        try:
            with self.command_lock:
                if self.stop_event.is_set():
                    result = "stopped_by_user"
                else:
                    self.bot.set_car_motion(cmd_v, cmd_w)
            while progress < action.target:
                if result == "stopped_by_user" or self.stop_event.is_set(): result = "stopped_by_user"; break
                now = time.monotonic(); dt = max(0.0, min(0.2, now-last)); last = now
                sample = read_motion_sample(self.bot, cmd_v, cmd_w)
                rate = abs(sample.linear) if cmd_v else abs(sample.angular)
                progress += rate * dt
                self.recorder.write_motion({"action_id":action_id,"requested_v":cmd_v,"requested_w":cmd_w,"measured_v":sample.linear,"measured_w":sample.angular,"source":sample.source,"progress":round(progress,5),"target":action.target})
                if now-started >= timeout: result = "timeout"; break
                time.sleep(0.05)
        except Exception as exc:
            result = f"error:{type(exc).__name__}:{exc}"
        finally:
            stop_chassis(self.bot, count=5, delay=0.02)
            after = self.camera.snapshot(f"{action_id:04d}_after")
            row={"id":action_id,"action":action.action,"result":result,"progress":round(progress,5),"target":action.target,"duration":round(time.monotonic()-started,3),"after":after}
            self.recorder.write_event({"event":"action_end",**row})
            with self.lock:
                self.last_result=row; self.busy=False

    def status(self) -> dict:
        with self.lock:
            return {"busy":self.busy,"last_result":self.last_result,"run_dir":str(self.recorder.root)}


class App:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args=args; self.recorder=Recorder(Path(args.record_dir), args); self.bot=make_bot(args.dry_run)
        self.camera=CameraThread(args,self.recorder); self.controller=StepController(self.bot,self.camera,self.recorder)
        stop_chassis(self.bot,count=4,delay=.02); self.camera.start()

    def close(self) -> None:
        self.controller.stop(); self.camera.stop_event.set(); self.camera.join(timeout=2); self.recorder.close()


APP: App


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, data: dict) -> None:
        body=json.dumps(data,ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def read_json(self) -> dict:
        size=int(self.headers.get("Content-Length","0")); return json.loads(self.rfile.read(size) or b"{}")
    def do_GET(self) -> None:
        if self.path=="/":
            body=INDEX_HTML.encode(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path=="/api/status": self.send_json(200,APP.controller.status()); return
        if self.path=="/video.mjpg": self.stream_video(); return
        self.send_error(404)
    def do_POST(self) -> None:
        try:
            if self.path=="/api/action":
                action=APP.controller.parse(self.read_json()); action_id=APP.controller.submit(action); self.send_json(202,{"ok":True,"action_id":action_id}); return
            if self.path=="/api/stop":
                APP.controller.stop(); self.send_json(200,{"ok":True,"message":"停车命令已发送并记录"}); return
            self.send_error(404)
        except (ValueError,RuntimeError) as exc: self.send_json(409,{"ok":False,"error":str(exc)})
        except Exception as exc: self.send_json(500,{"ok":False,"error":f"{type(exc).__name__}: {exc}"})
    def stream_video(self) -> None:
        self.send_response(200); self.send_header("Content-Type","multipart/x-mixed-replace; boundary=frame"); self.end_headers()
        try:
            while True:
                with APP.camera.lock: jpg=APP.camera.latest_jpeg
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "+str(len(jpg)).encode()+b"\r\n\r\n"+jpg+b"\r\n")
                time.sleep(.1)
        except (BrokenPipeError,ConnectionResetError): pass
    def log_message(self, fmt, *args): return


def main() -> int:
    global APP
    p=argparse.ArgumentParser(description="Safe recorded manual Transbot step controller")
    p.add_argument("--host",default="0.0.0.0"); p.add_argument("--port",type=int,default=8780); p.add_argument("--camera",type=int,default=0)
    p.add_argument("--width",type=int,default=640); p.add_argument("--height",type=int,default=480); p.add_argument("--record-fps",type=float,default=5.0)
    p.add_argument("--record-dir",default=str(ROOT/"artifacts"/"manual_runs")); p.add_argument("--dry-run",action="store_true")
    p.add_argument(
        "--allow-local-camera", action="store_true",
        help="Allow VideoCapture in dry-run mode (off by default to protect the host camera)",
    )
    args=p.parse_args(); APP=App(args); server=ThreadingHTTPServer((args.host,args.port),Handler)
    def shutdown(*_): threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGINT,shutdown); signal.signal(signal.SIGTERM,shutdown)
    print(f"manual_drive_url=http://{args.host}:{args.port} record_dir={APP.recorder.root}",flush=True)
    try: server.serve_forever()
    finally: APP.close(); server.server_close()
    return 0


if __name__=="__main__": raise SystemExit(main())
