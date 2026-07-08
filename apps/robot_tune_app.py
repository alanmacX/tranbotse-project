#!/usr/bin/env python3
import base64
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


HOST = "127.0.0.1"
PORT = 8765
SSH_TARGET = "yahboom"
STATE_PATH = os.path.join(os.path.dirname(__file__), "robot_tune_state.json")

STATE = {
    "cam1": 90,
    "cam2": 12,
    "j1": 55,
    "j2": 205,
    "j3": 45,
    "arm_ms": 1200,
    "crop": [300, 265, 430, 455],
    "line_speed": 0.035,
    "line_max_sec": 12,
    "line_invert": False,
    "manual_v": 0.04,
    "manual_w": 0.25,
    "manual_sec": 0.8,
    "enable_corner": False,
    "right_top_area_th": 800,
    "right_area_th": 1000,
    "right_wide_ratio_th": 1.8,
    "right_confirm_frames": 3,
    "right_turn_dir": 1,
    "right_turn_pre_forward_sec": 0.25,
    "right_turn_pre_forward_v": 0.035,
    "right_turn_w": 0.28,
    "right_turn_min_sec": 0.50,
    "turn_finish_err_th": 0.55,
    "reacquire_count_th": 2,
    "cam_fx": 520.0,
    "cam_fy": 520.0,
    "cam_cx": 320.0,
    "cam_cy": 240.0,
    "target_real_w_mm": 25.0,
    "target_real_h_mm": 25.0,
    "target_px_w": 50.0,
    "target_px_h": 50.0,
    "corner_forward_sec": 1.2,
    "corner_turn_sec": 2.45,
    "corner_trigger_frac": 0.35,
}
STATE_LOCK = threading.Lock()
VIDEO_PROCS = set()
VIDEO_LOCK = threading.Lock()


def load_state():
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            STATE.update({k: data[k] for k in STATE.keys() & data.keys()})
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"state load warning: {exc}")


def save_state():
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(STATE, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_PATH)


load_state()


def clamp(value, lo, hi):
    return max(lo, min(hi, int(value)))


def remote_python(code, timeout=10):
    return subprocess.run(
        ["ssh", SSH_TARGET, "python3", "-"],
        input=code,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def release_remote_camera(timeout=4):
    # The MJPEG stream is a remote `python3 -u -` process that owns /dev/video0.
    # Stop it before one-shot detection or line-follow demos open the camera.
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
    subprocess.run(
        [
            "ssh",
            SSH_TARGET,
            "sh",
            "-lc",
            "ps -eo pid=,args= | awk '$0 ~ /python3 -u -/ {print $1}' | xargs -r kill",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    time.sleep(0.25)


def transbot_code(body):
    return f"""
import sys, time
sys.path.append('/home/pi/Transbot/py_install')
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


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Transbot Safe Tuning</title>
  <style>
    :root { color-scheme: light; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f5f7fa; color: #111827; }
    header { height: 54px; display:flex; align-items:center; justify-content:space-between; padding:0 18px; background:white; border-bottom:1px solid #d9dee8; }
    h1 { font-size: 18px; margin:0; font-weight: 700; }
    main { display:grid; grid-template-columns: minmax(680px, 1fr) 360px; gap: 14px; padding: 14px; }
    .stage { background:#111; border-radius:8px; position:relative; overflow:hidden; min-height: 520px; display:flex; align-items:center; justify-content:center; }
    #videoWrap { position:relative; width:640px; height:480px; }
    #video { width:640px; height:480px; display:block; background:#222; }
    #cropBox { position:absolute; border:2px solid #ffb020; box-shadow:0 0 0 9999px rgba(0,0,0,.15); pointer-events:none; }
    #centerLine { position:absolute; top:0; bottom:0; width:1px; left:319px; background:#2f80ed; opacity:.9; }
    aside { display:flex; flex-direction:column; gap:12px; }
    section { background:white; border:1px solid #d9dee8; border-radius:8px; padding:12px; }
    h2 { font-size:14px; margin:0 0 10px; }
    .row { display:grid; grid-template-columns: 72px 1fr 54px; align-items:center; gap:8px; margin:8px 0; }
    .checkrow { display:flex; align-items:center; gap:8px; margin:8px 0; }
    input[type=range] { width:100%; }
    input[type=number] { width:52px; padding:4px; border:1px solid #c8d0dc; border-radius:6px; }
    button { border:1px solid #c5cedb; background:#fff; border-radius:7px; padding:8px 10px; cursor:pointer; font-weight:600; }
    button.primary { background:#1463ff; color:white; border-color:#1463ff; }
    button.danger { background:#c91f37; color:white; border-color:#c91f37; }
    button:hover { filter:brightness(.97); }
    .buttons { display:flex; gap:8px; flex-wrap:wrap; }
    #log { height:120px; overflow:auto; background:#0b1020; color:#d6e2ff; border-radius:8px; padding:8px; font:12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space:pre-wrap; }
    #debugImg { width:100%; border-radius:8px; border:1px solid #d9dee8; display:none; }
    .hint { color:#667085; font-size:12px; line-height:1.4; }
    @media (max-width: 1080px) { main { grid-template-columns: 1fr; } aside { max-width: 720px; } }
  </style>
</head>
<body>
  <header>
    <h1>Transbot Safe Tuning</h1>
    <div class="buttons">
      <button onclick="location.href='/'">原有控制</button>
      <button onclick="location.href='/corner'">直角弯测试</button>
      <button onclick="reloadVideo()">重连视频</button>
      <button class="danger" onclick="stopRobot()">停车</button>
    </div>
  </header>
  <main>
    <div class="stage">
      <div id="videoWrap">
        <img id="video" src="/video" />
        <div id="centerLine"></div>
        <div id="cropBox"></div>
      </div>
    </div>
    <aside>
      <section>
        <h2>相机云台</h2>
        <div class="row"><label>水平</label><input id="cam1" type="range" min="0" max="180"><input id="cam1n" type="number"></div>
        <div class="row"><label>俯仰</label><input id="cam2" type="range" min="8" max="100"><input id="cam2n" type="number"></div>
        <div class="buttons">
          <button class="primary" onclick="applyCamera()">应用相机</button>
          <button onclick="presetCam(90,12)">低角度</button>
          <button onclick="presetCam(90,18)">微上</button>
        </div>
        <p class="hint">俯仰限制在 8..100，避开 0 度硬限位抖动。</p>
      </section>

      <section>
        <h2>机械臂停车位</h2>
        <div class="row"><label>j1</label><input id="j1" type="range" min="0" max="225"><input id="j1n" type="number"></div>
        <div class="row"><label>j2</label><input id="j2" type="range" min="30" max="270"><input id="j2n" type="number"></div>
        <div class="row"><label>夹爪</label><input id="j3" type="range" min="30" max="180"><input id="j3n" type="number"></div>
        <div class="row"><label>时间</label><input id="arm_ms" type="range" min="600" max="2200"><input id="arm_msn" type="number"></div>
        <div class="buttons">
          <button class="primary" onclick="applyArm()">应用机械臂</button>
          <button onclick="presetArm(55,205,45)">当前推荐</button>
          <button onclick="presetArm(75,205,45)">中间偏左</button>
          <button onclick="presetArm(110,250,45)">原始收回</button>
        </div>
      </section>

      <section>
        <h2>算法 Crop</h2>
        <div class="row"><label>x0</label><input id="x0" type="range" min="0" max="639"><input id="x0n" type="number"></div>
        <div class="row"><label>y0</label><input id="y0" type="range" min="0" max="479"><input id="y0n" type="number"></div>
        <div class="row"><label>x1</label><input id="x1" type="range" min="1" max="640"><input id="x1n" type="number"></div>
        <div class="row"><label>y1</label><input id="y1" type="range" min="1" max="480"><input id="y1n" type="number"></div>
        <div class="buttons">
          <button class="primary" onclick="saveCrop()">更新框</button>
          <button onclick="presetCrop(300,265,430,455)">紧 crop</button>
          <button onclick="presetCrop(255,210,455,435)">宽 crop</button>
          <button onclick="detectOnce()">检测一次</button>
        </div>
        <p class="hint">橙框就是算法看到的 FOV。先把无关背景裁掉，再调黑线检测/PID。</p>
      </section>

      <section>
        <h2>检测结果</h2>
        <img id="debugImg" />
        <div id="log"></div>
      </section>

      <section>
        <h2>巡线 Demo</h2>
        <div class="row"><label>速度</label><input id="line_speed" type="range" min="0.01" max="0.06" step="0.005"><input id="line_speedn" type="number" step="0.005"></div>
        <div class="row"><label>最长秒</label><input id="line_max_sec" type="range" min="2" max="60"><input id="line_max_secn" type="number"></div>
        <label class="checkrow"><input id="line_invert" type="checkbox"> 巡线转向反向</label>
        <div class="buttons">
          <button class="primary" onclick="runLineDemo()">跑线</button>
          <button onclick="presetLine(0.035,12)">12s</button>
          <button onclick="presetLine(0.035,60)">60s</button>
          <button class="danger" onclick="stopRobot()">停车</button>
        </div>
        <p class="hint">运行时会独占相机；如果视频黑了，结束后点“重连视频”。丢线会自动停车。</p>
      </section>

      <section>
        <h2>内参距离估算</h2>
        <div class="row"><label>fx</label><input id="cam_fx" type="range" min="100" max="1400" step="1"><input id="cam_fxn" type="number" step="1"></div>
        <div class="row"><label>fy</label><input id="cam_fy" type="range" min="100" max="1400" step="1"><input id="cam_fyn" type="number" step="1"></div>
        <div class="row"><label>cx</label><input id="cam_cx" type="range" min="0" max="640" step="1"><input id="cam_cxn" type="number" step="1"></div>
        <div class="row"><label>cy</label><input id="cam_cy" type="range" min="0" max="480" step="1"><input id="cam_cyn" type="number" step="1"></div>
        <div class="row"><label>实宽mm</label><input id="target_real_w_mm" type="range" min="1" max="500" step="1"><input id="target_real_w_mmn" type="number" step="1"></div>
        <div class="row"><label>实高mm</label><input id="target_real_h_mm" type="range" min="1" max="500" step="1"><input id="target_real_h_mmn" type="number" step="1"></div>
        <div class="row"><label>像素宽</label><input id="target_px_w" type="range" min="1" max="640" step="1"><input id="target_px_wn" type="number" step="1"></div>
        <div class="row"><label>像素高</label><input id="target_px_h" type="range" min="1" max="480" step="1"><input id="target_px_hn" type="number" step="1"></div>
        <div class="buttons">
          <button class="primary" onclick="calcDistance()">计算距离</button>
          <button onclick="presetIntrinsics(520,520,320,240)">默认内参</button>
        </div>
        <p class="hint" id="distanceResult">Zx = fx * 实宽 / 像素宽；Zy = fy * 实高 / 像素高。需要真实标定后才准。</p>
      </section>

      <section>
        <h2>底盘手动 / 空转</h2>
        <div class="row"><label>线速度</label><input id="manual_v" type="range" min="0.01" max="0.08" step="0.005"><input id="manual_vn" type="number" step="0.005"></div>
        <div class="row"><label>角速度</label><input id="manual_w" type="range" min="0.05" max="0.5" step="0.025"><input id="manual_wn" type="number" step="0.025"></div>
        <div class="row"><label>秒</label><input id="manual_sec" type="range" min="0.2" max="2" step="0.1"><input id="manual_secn" type="number" step="0.1"></div>
        <div class="buttons">
          <button onclick="manualMotion('forward')">前进</button>
          <button onclick="manualMotion('backward')">后退</button>
          <button onclick="manualMotion('left')">左转</button>
          <button onclick="manualMotion('right')">右转</button>
          <button onclick="manualMotion('spin_left')">空转左</button>
          <button onclick="manualMotion('spin_right')">空转右</button>
          <button class="danger" onclick="stopRobot()">停车</button>
        </div>
        <p class="hint">每个动作会自动限时并停车。用它确认硬件方向：点左转应向左，点右转应向右。</p>
      </section>
    </aside>
  </main>
  <script>
    const ids = ["cam1","cam2","j1","j2","j3","arm_ms","x0","y0","x1","y1","line_speed","line_max_sec","manual_v","manual_w","manual_sec","cam_fx","cam_fy","cam_cx","cam_cy","target_real_w_mm","target_real_h_mm","target_px_w","target_px_h"];
    let state = {};
    const log = (msg) => {
      const el = document.getElementById("log");
      el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}\n` + el.textContent;
    };
    function bindPair(id) {
      const r = document.getElementById(id), n = document.getElementById(id+"n");
      const sync = (from) => { if (from === r) n.value = r.value; else r.value = n.value; if (id[0] === "x" || id[0] === "y") drawCrop(); };
      r.addEventListener("input", () => sync(r));
      n.addEventListener("input", () => sync(n));
    }
    ids.forEach(bindPair);
    function setVal(id, v) { document.getElementById(id).value = v; document.getElementById(id+"n").value = v; }
    function getVal(id) { return Number(document.getElementById(id).value); }
    function drawCrop() {
      const box = document.getElementById("cropBox");
      const x0=getVal("x0"), y0=getVal("y0"), x1=getVal("x1"), y1=getVal("y1");
      box.style.left = x0 + "px"; box.style.top = y0 + "px";
      box.style.width = Math.max(1, x1-x0) + "px"; box.style.height = Math.max(1, y1-y0) + "px";
    }
    async function api(path, body) {
      const res = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body || {})});
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || res.statusText);
      return data;
    }
    async function init() {
      const res = await fetch("/api/state"); state = await res.json();
      setVal("cam1", state.cam1); setVal("cam2", state.cam2);
      setVal("j1", state.j1); setVal("j2", state.j2); setVal("j3", state.j3); setVal("arm_ms", state.arm_ms);
      setVal("x0", state.crop[0]); setVal("y0", state.crop[1]); setVal("x1", state.crop[2]); setVal("y1", state.crop[3]);
      setVal("line_speed", state.line_speed); setVal("line_max_sec", state.line_max_sec);
      setVal("manual_v", state.manual_v); setVal("manual_w", state.manual_w); setVal("manual_sec", state.manual_sec);
      document.getElementById("line_invert").checked = !!state.line_invert;
      setVal("cam_fx", state.cam_fx); setVal("cam_fy", state.cam_fy);
      setVal("cam_cx", state.cam_cx); setVal("cam_cy", state.cam_cy);
      setVal("target_real_w_mm", state.target_real_w_mm); setVal("target_real_h_mm", state.target_real_h_mm);
      setVal("target_px_w", state.target_px_w); setVal("target_px_h", state.target_px_h);
      drawCrop(); log("ready");
    }
    async function applyCamera() {
      try { const d = await api("/api/camera", {cam1:getVal("cam1"), cam2:getVal("cam2")}); log(d.message); }
      catch(e) { log("相机失败: "+e.message); }
    }
    async function applyArm() {
      try { const d = await api("/api/arm", {j1:getVal("j1"), j2:getVal("j2"), j3:getVal("j3"), arm_ms:getVal("arm_ms")}); log(d.message); }
      catch(e) { log("机械臂失败: "+e.message); }
    }
    async function stopRobot() {
      try { const d = await api("/api/stop", {}); log(d.message); }
      catch(e) { log("停车失败: "+e.message); }
    }
    function presetCam(a,b) { setVal("cam1", a); setVal("cam2", b); applyCamera(); }
    function presetArm(a,b,c) { setVal("j1", a); setVal("j2", b); setVal("j3", c); applyArm(); }
    function presetCrop(a,b,c,d) { setVal("x0",a); setVal("y0",b); setVal("x1",c); setVal("y1",d); saveCrop(); }
    async function saveCrop() {
      drawCrop();
      try { const d = await api("/api/crop", {crop:[getVal("x0"),getVal("y0"),getVal("x1"),getVal("y1")]}); log(d.message); }
      catch(e) { log("crop 失败: "+e.message); }
    }
    async function detectOnce() {
      saveCrop();
      try {
        const d = await api("/api/detect", {crop:[getVal("x0"),getVal("y0"),getVal("x1"),getVal("y1")]});
        log(d.info);
        const img = document.getElementById("debugImg");
        img.src = "data:image/jpeg;base64," + d.image;
        img.style.display = "block";
      } catch(e) { log("检测失败: "+e.message); }
    }
    function presetLine(speed, sec) { setVal("line_speed", speed); setVal("line_max_sec", sec); }
    async function runLineDemo() {
      saveCrop();
      log("巡线启动，会暂时占用相机...");
      try {
        const d = await api("/api/line_demo", {
          crop:[getVal("x0"),getVal("y0"),getVal("x1"),getVal("y1")],
          speed:getVal("line_speed"),
          max_sec:getVal("line_max_sec"),
          invert:document.getElementById("line_invert").checked,
          enable_corner:false,
          corner:{}
        });
        log(`${d.summary.stop_reason}; frames=${d.summary.frames}; found=${d.summary.found}`);
        const img = document.getElementById("debugImg");
        img.src = "data:image/jpeg;base64," + d.image;
        img.style.display = "block";
      } catch(e) { log("巡线失败: "+e.message); }
    }
    async function manualMotion(action) {
      try {
        const d = await api("/api/motion", {action, v:getVal("manual_v"), w:getVal("manual_w"), sec:getVal("manual_sec")});
        log(d.message);
      } catch(e) { log("底盘失败: "+e.message); }
    }
    function intrinsicsParams() {
      return {
        cam_fx:getVal("cam_fx"), cam_fy:getVal("cam_fy"),
        cam_cx:getVal("cam_cx"), cam_cy:getVal("cam_cy"),
        target_real_w_mm:getVal("target_real_w_mm"),
        target_real_h_mm:getVal("target_real_h_mm"),
        target_px_w:getVal("target_px_w"),
        target_px_h:getVal("target_px_h")
      };
    }
    function presetIntrinsics(fx, fy, cx, cy) {
      setVal("cam_fx", fx); setVal("cam_fy", fy); setVal("cam_cx", cx); setVal("cam_cy", cy);
      calcDistance();
    }
    async function calcDistance() {
      try {
        const d = await api("/api/distance", intrinsicsParams());
        document.getElementById("distanceResult").textContent = d.message;
        log(d.message);
      } catch(e) { log("距离估算失败: "+e.message); }
    }
    function reloadVideo() { document.getElementById("video").src = "/video?ts=" + Date.now(); log("video reconnect"); }
    init();
  </script>
</body>
</html>
"""


CORNER_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Transbot Corner Test</title>
  <style>
    :root { font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin:0; background:#f5f7fa; color:#111827; }
    header { height:54px; display:flex; align-items:center; justify-content:space-between; padding:0 18px; background:white; border-bottom:1px solid #d9dee8; }
    h1 { font-size:18px; margin:0; }
    main { display:grid; grid-template-columns:minmax(680px, 1fr) 380px; gap:14px; padding:14px; }
    .stage { background:#111; border-radius:8px; min-height:520px; display:flex; align-items:center; justify-content:center; overflow:hidden; }
    #videoWrap { position:relative; width:640px; height:480px; }
    #video { width:640px; height:480px; display:block; background:#222; }
    #cropBox { position:absolute; border:2px solid #ffb020; box-shadow:0 0 0 9999px rgba(0,0,0,.15); pointer-events:none; }
    #centerLine { position:absolute; top:0; bottom:0; left:319px; width:1px; background:#2f80ed; }
    section { background:white; border:1px solid #d9dee8; border-radius:8px; padding:12px; margin-bottom:12px; }
    h2 { font-size:14px; margin:0 0 10px; }
    .row { display:grid; grid-template-columns:72px 1fr 58px; gap:8px; align-items:center; margin:8px 0; }
    input[type=range] { width:100%; }
    input[type=number] { width:56px; padding:4px; border:1px solid #c8d0dc; border-radius:6px; }
    button { border:1px solid #c5cedb; background:white; border-radius:7px; padding:8px 10px; cursor:pointer; font-weight:600; }
    button.primary { background:#1463ff; color:white; border-color:#1463ff; }
    button.danger { background:#c91f37; color:white; border-color:#c91f37; }
    .buttons { display:flex; gap:8px; flex-wrap:wrap; }
    #log { height:180px; overflow:auto; background:#0b1020; color:#d6e2ff; border-radius:8px; padding:8px; font:12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space:pre-wrap; }
    #debugImg { width:100%; border-radius:8px; border:1px solid #d9dee8; display:none; }
    .hint { color:#667085; font-size:12px; line-height:1.4; }
  </style>
</head>
<body>
  <header>
    <h1>直角弯测试</h1>
    <div class="buttons">
      <button onclick="location.href='/'">原有控制</button>
      <button onclick="reloadVideo()">重连视频</button>
      <button class="danger" onclick="stopRobot()">停车</button>
    </div>
  </header>
  <main>
    <div class="stage">
      <div id="videoWrap">
        <img id="video" src="/video" />
        <div id="centerLine"></div>
        <div id="cropBox"></div>
      </div>
    </div>
    <aside>
      <section>
        <h2>Crop</h2>
        <div class="row"><label>x0</label><input id="x0" type="range" min="0" max="639"><input id="x0n" type="number"></div>
        <div class="row"><label>y0</label><input id="y0" type="range" min="0" max="479"><input id="y0n" type="number"></div>
        <div class="row"><label>x1</label><input id="x1" type="range" min="1" max="640"><input id="x1n" type="number"></div>
        <div class="row"><label>y1</label><input id="y1" type="range" min="1" max="480"><input id="y1n" type="number"></div>
        <div class="buttons">
          <button onclick="presetCrop(306,251,420,373)">常用 crop</button>
          <button onclick="detectOnce()">检测一次</button>
        </div>
      </section>
      <section>
        <h2>动作</h2>
        <div class="row"><label>速度</label><input id="speed" type="range" min="0.01" max="0.06" step="0.005"><input id="speedn" type="number" step="0.005"></div>
        <div class="row"><label>最长秒</label><input id="max_sec" type="range" min="2" max="60"><input id="max_secn" type="number"></div>
        <div class="row"><label>直走秒</label><input id="forward_sec" type="range" min="0" max="5" step="0.05"><input id="forward_secn" type="number" step="0.05"></div>
        <div class="row"><label>转弯秒</label><input id="turn_sec" type="range" min="0.5" max="4" step="0.05"><input id="turn_secn" type="number" step="0.05"></div>
        <div class="row"><label>触发线</label><input id="trigger_frac" type="range" min="0.1" max="0.8" step="0.05"><input id="trigger_fracn" type="number" step="0.05"></div>
        <div class="buttons">
          <button class="primary" onclick="runCornerDemo()">跑直角弯</button>
          <button class="danger" onclick="stopRobot()">停车</button>
        </div>
        <p class="hint">右分支跨过触发线后，直走指定秒数，再转弯指定秒数。</p>
      </section>
      <section>
        <h2>结果</h2>
        <img id="debugImg" />
        <div id="log"></div>
      </section>
    </aside>
  </main>
  <script>
    const ids = ["x0","y0","x1","y1","speed","max_sec","forward_sec","turn_sec","trigger_frac"];
    const log = (msg) => { const el=document.getElementById("log"); el.textContent=`[${new Date().toLocaleTimeString()}] ${msg}\n`+el.textContent; };
    function bindPair(id) {
      const r=document.getElementById(id), n=document.getElementById(id+"n");
      const sync=(from)=>{ if(from===r)n.value=r.value; else r.value=n.value; if(["x0","y0","x1","y1"].includes(id))drawCrop(); };
      r.addEventListener("input",()=>sync(r)); n.addEventListener("input",()=>sync(n));
    }
    ids.forEach(bindPair);
    function setVal(id,v){ document.getElementById(id).value=v; document.getElementById(id+"n").value=v; }
    function getVal(id){ return Number(document.getElementById(id).value); }
    function drawCrop(){ const b=document.getElementById("cropBox"), x0=getVal("x0"), y0=getVal("y0"), x1=getVal("x1"), y1=getVal("y1"); b.style.left=x0+"px"; b.style.top=y0+"px"; b.style.width=Math.max(1,x1-x0)+"px"; b.style.height=Math.max(1,y1-y0)+"px"; }
    async function api(path, body){ const res=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})}); const data=await res.json(); if(!res.ok||!data.ok)throw new Error(data.error||res.statusText); return data; }
    async function init(){ const s=await (await fetch("/api/state")).json(); setVal("x0",s.crop[0]); setVal("y0",s.crop[1]); setVal("x1",s.crop[2]); setVal("y1",s.crop[3]); setVal("speed",s.line_speed); setVal("max_sec",s.line_max_sec); setVal("forward_sec",s.corner_forward_sec ?? 1.2); setVal("turn_sec",s.corner_turn_sec ?? 2.45); setVal("trigger_frac",s.corner_trigger_frac ?? 0.35); drawCrop(); log("ready"); }
    function presetCrop(a,b,c,d){ setVal("x0",a); setVal("y0",b); setVal("x1",c); setVal("y1",d); drawCrop(); }
    async function detectOnce(){ try{ const d=await api("/api/detect",{crop:[getVal("x0"),getVal("y0"),getVal("x1"),getVal("y1")]}); log(d.info); const img=document.getElementById("debugImg"); img.src="data:image/jpeg;base64,"+d.image; img.style.display="block"; }catch(e){ log("检测失败: "+e.message); } }
    async function runCornerDemo(){ log("直角巡线启动：先跑直线，识别直角后自动转弯..."); try{ const d=await api("/api/corner_demo",{crop:[getVal("x0"),getVal("y0"),getVal("x1"),getVal("y1")],speed:getVal("speed"),max_sec:getVal("max_sec"),forward_sec:getVal("forward_sec"),turn_sec:getVal("turn_sec"),trigger_frac:getVal("trigger_frac")}); log(`${d.summary.stop_reason}; frames=${d.summary.frames}; found=${d.summary.found}`); const img=document.getElementById("debugImg"); img.src="data:image/jpeg;base64,"+d.image; img.style.display="block"; }catch(e){ log("直角失败: "+e.message); } }
    async function stopRobot(){ try{ const d=await api("/api/stop",{}); log(d.message); }catch(e){ log("停车失败: "+e.message); } }
    function reloadVideo(){ document.getElementById("video").src="/video?ts="+Date.now(); log("video reconnect"); }
    init();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_json(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            payload = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/corner":
            payload = CORNER_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if path == "/api/state":
            with STATE_LOCK:
                self.send_json(STATE)
            return
        if path == "/video":
            self.stream_video()
            return
        self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/camera":
                self.api_camera()
            elif path == "/api/arm":
                self.api_arm()
            elif path == "/api/stop":
                self.api_stop()
            elif path == "/api/motion":
                self.api_motion()
            elif path == "/api/crop":
                self.api_crop()
            elif path == "/api/detect":
                self.api_detect()
            elif path == "/api/line_demo":
                self.api_line_demo()
            elif path == "/api/corner":
                self.api_corner()
            elif path == "/api/corner_demo":
                self.api_corner_demo()
            elif path == "/api/distance":
                self.api_distance()
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=500)

    def api_camera(self):
        data = self.read_json()
        cam1 = clamp(data.get("cam1", STATE["cam1"]), 0, 180)
        cam2 = clamp(data.get("cam2", STATE["cam2"]), 8, 100)
        code = transbot_code(f"""
    bot.set_pwm_servo(1, {cam1})
    time.sleep(0.08)
    bot.set_pwm_servo(2, {cam2})
    time.sleep(0.18)
""")
        res = remote_python(code, timeout=8)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip())
        with STATE_LOCK:
            STATE["cam1"], STATE["cam2"] = cam1, cam2
            save_state()
        self.send_json({"ok": True, "message": f"camera servo1={cam1}, servo2={cam2}"})

    def api_arm(self):
        data = self.read_json()
        j1 = clamp(data.get("j1", STATE["j1"]), 0, 225)
        j2 = clamp(data.get("j2", STATE["j2"]), 30, 270)
        j3 = clamp(data.get("j3", STATE["j3"]), 30, 180)
        arm_ms = clamp(data.get("arm_ms", STATE["arm_ms"]), 600, 2200)
        code = transbot_code(f"""
    bot.set_uart_servo_angle_array({j1}, {j2}, {j3}, {arm_ms})
    time.sleep({arm_ms / 1000.0 + 0.25:.2f})
""")
        res = remote_python(code, timeout=max(6, arm_ms / 1000 + 4))
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip())
        with STATE_LOCK:
            STATE["j1"], STATE["j2"], STATE["j3"], STATE["arm_ms"] = j1, j2, j3, arm_ms
            save_state()
        self.send_json({"ok": True, "message": f"arm j1={j1}, j2={j2}, j3={j3}, {arm_ms}ms"})

    def api_stop(self):
        code = transbot_code("""
    for _ in range(8):
        bot.set_car_motion(0, 0)
        time.sleep(0.04)
""")
        res = remote_python(code, timeout=6)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip())
        self.send_json({"ok": True, "message": "chassis stopped"})

    def api_motion(self):
        data = self.read_json()
        action = str(data.get("action", "stop"))
        v = max(0.0, min(0.08, float(data.get("v", STATE["manual_v"]))))
        w = max(0.0, min(0.5, float(data.get("w", STATE["manual_w"]))))
        sec = max(0.2, min(2.0, float(data.get("sec", STATE["manual_sec"]))))
        commands = {
            "stop": (0.0, 0.0),
            "forward": (v, 0.0),
            "backward": (-v, 0.0),
            "left": (0.0, w),
            "right": (0.0, -w),
            "spin_left": (0.0, w),
            "spin_right": (0.0, -w),
        }
        if action not in commands:
            raise RuntimeError(f"unknown action: {action}")
        cmd_v, cmd_w = commands[action]
        code = transbot_code(f"""
    bot.set_car_motion({cmd_v:.4f}, {cmd_w:.4f})
    time.sleep({sec:.2f})
""")
        res = remote_python(code, timeout=sec + 5)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip())
        with STATE_LOCK:
            STATE["manual_v"] = v
            STATE["manual_w"] = w
            STATE["manual_sec"] = sec
            save_state()
        self.send_json({"ok": True, "message": f"{action}: v={cmd_v:.3f}, w={cmd_w:.3f}, {sec:.1f}s"})

    def normalize_crop(self, crop):
        if not isinstance(crop, list) or len(crop) != 4:
            crop = STATE["crop"]
        x0 = clamp(crop[0], 0, 638)
        y0 = clamp(crop[1], 0, 478)
        x1 = clamp(crop[2], x0 + 1, 640)
        y1 = clamp(crop[3], y0 + 1, 480)
        return [x0, y0, x1, y1]

    def api_crop(self):
        crop = self.normalize_crop(self.read_json().get("crop"))
        with STATE_LOCK:
            STATE["crop"] = crop
            save_state()
        self.send_json({"ok": True, "message": f"crop={crop}", "crop": crop})

    def api_corner(self):
        data = self.read_json()
        corner = data.get("corner") if isinstance(data.get("corner"), dict) else {}
        with STATE_LOCK:
            STATE["enable_corner"] = bool(data.get("enable_corner", STATE["enable_corner"]))
            STATE["right_top_area_th"] = clamp(corner.get("right_top_area_th", STATE["right_top_area_th"]), 100, 4000)
            STATE["right_area_th"] = clamp(corner.get("right_area_th", STATE["right_area_th"]), 100, 5000)
            STATE["right_wide_ratio_th"] = max(1.0, min(5.0, float(corner.get("right_wide_ratio_th", STATE["right_wide_ratio_th"]))))
            STATE["right_confirm_frames"] = clamp(corner.get("right_confirm_frames", STATE["right_confirm_frames"]), 1, 8)
            STATE["right_turn_dir"] = 1 if int(corner.get("right_turn_dir", STATE["right_turn_dir"])) >= 0 else -1
            STATE["right_turn_pre_forward_sec"] = max(0.0, min(1.5, float(corner.get("right_turn_pre_forward_sec", STATE["right_turn_pre_forward_sec"]))))
            STATE["right_turn_pre_forward_v"] = max(0.0, min(0.08, float(corner.get("right_turn_pre_forward_v", STATE["right_turn_pre_forward_v"]))))
            STATE["right_turn_w"] = max(0.05, min(0.6, float(corner.get("right_turn_w", STATE["right_turn_w"]))))
            STATE["right_turn_min_sec"] = max(0.1, min(2.0, float(corner.get("right_turn_min_sec", STATE["right_turn_min_sec"]))))
            STATE["turn_finish_err_th"] = max(0.1, min(1.0, float(corner.get("turn_finish_err_th", STATE["turn_finish_err_th"]))))
            save_state()
            msg = f"corner enable={STATE['enable_corner']} dir={STATE['right_turn_dir']}"
        self.send_json({"ok": True, "message": msg})

    def api_distance(self):
        data = self.read_json()
        fx = max(1.0, float(data.get("cam_fx", STATE["cam_fx"])))
        fy = max(1.0, float(data.get("cam_fy", STATE["cam_fy"])))
        cx = max(0.0, min(640.0, float(data.get("cam_cx", STATE["cam_cx"]))))
        cy = max(0.0, min(480.0, float(data.get("cam_cy", STATE["cam_cy"]))))
        real_w = max(0.1, float(data.get("target_real_w_mm", STATE["target_real_w_mm"])))
        real_h = max(0.1, float(data.get("target_real_h_mm", STATE["target_real_h_mm"])))
        px_w = max(0.1, float(data.get("target_px_w", STATE["target_px_w"])))
        px_h = max(0.1, float(data.get("target_px_h", STATE["target_px_h"])))

        z_w = fx * real_w / px_w
        z_h = fy * real_h / px_h
        z_avg = (z_w + z_h) / 2.0
        with STATE_LOCK:
            STATE["cam_fx"] = fx
            STATE["cam_fy"] = fy
            STATE["cam_cx"] = cx
            STATE["cam_cy"] = cy
            STATE["target_real_w_mm"] = real_w
            STATE["target_real_h_mm"] = real_h
            STATE["target_px_w"] = px_w
            STATE["target_px_h"] = px_h
            save_state()
        msg = f"距离估算: Z宽={z_w:.1f}mm, Z高={z_h:.1f}mm, 平均={z_avg:.1f}mm ({z_avg/10.0:.1f}cm)"
        self.send_json({
            "ok": True,
            "message": msg,
            "z_width_mm": z_w,
            "z_height_mm": z_h,
            "z_avg_mm": z_avg,
        })

    def api_detect(self):
        release_remote_camera()
        crop = self.normalize_crop(self.read_json().get("crop"))
        x0, y0, x1, y1 = crop
        code = f"""
import base64, cv2 as cv, numpy as np, time, json
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
frame = None
for _ in range(15):
    ok, f = cap.read()
    if ok:
        frame = f.copy()
    time.sleep(0.025)
cap.release()
if frame is None:
    raise SystemExit('no frame')
x0,y0,x1,y1 = {x0},{y0},{x1},{y1}
h,w = frame.shape[:2]
crop = frame[y0:y1, x0:x1]
hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv, np.array([0,0,0]), np.array([180,170,105]))
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT,(3,3)), iterations=1)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(5,13)), iterations=1)
contours,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
valid = []
for c in contours:
    area = cv.contourArea(c)
    x,y,ww,hh = cv.boundingRect(c)
    if 40 < area < 6500 and (hh > 30 or ww > 30):
        valid.append(c)
ann = frame.copy()
cv.rectangle(ann, (x0,y0), (x1-1,y1-1), (80,180,255), 2)
info = 'no line'
if valid:
    c = max(valid, key=cv.contourArea)
    m = cv.moments(c)
    if abs(m['m00']) > 1e-6:
        cx_crop = int(m['m10']/m['m00'])
        cy_crop = int(m['m01']/m['m00'])
        cx = cx_crop + x0
        cy = cy_crop + y0
        err_crop = cx_crop - (x1-x0)//2
        err_full = cx - w//2
        cv.drawContours(ann, [c + np.array([[[x0,y0]]])], -1, (0,255,0), 2)
        cv.circle(ann, (cx,cy), 6, (0,0,255), -1)
        cv.line(ann, ((x0+x1)//2,y0), ((x0+x1)//2,y1), (255,0,0), 1)
        cv.line(ann, (cx,cy), ((x0+x1)//2,cy), (0,0,255), 2)
        info = f'crop_cx={{cx_crop}} crop_err={{err_crop}} full_cx={{cx}} full_err={{err_full}} area={{int(cv.contourArea(c))}}'
cv.putText(ann, info, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
ok, buf = cv.imencode('.jpg', ann, [int(cv.IMWRITE_JPEG_QUALITY), 82])
print(json.dumps({{'info': info, 'image': base64.b64encode(buf).decode('ascii')}}))
"""
        res = remote_python(code, timeout=8)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip())
        data = json.loads(res.stdout.strip().splitlines()[-1])
        with STATE_LOCK:
            STATE["crop"] = crop
            save_state()
        self.send_json({"ok": True, "info": data["info"], "image": data["image"], "crop": crop})

    def api_line_demo(self):
        data = self.read_json()
        release_remote_camera()
        crop = self.normalize_crop(data.get("crop"))
        if crop[2] - crop[0] < 60 or crop[3] - crop[1] < 60:
            raise RuntimeError(f"crop too small for line demo: {crop}; use a taller box around the black line")
        speed = float(data.get("speed", STATE["line_speed"]))
        speed = max(0.01, min(0.06, speed))
        max_sec = clamp(data.get("max_sec", STATE["line_max_sec"]), 2, 60)
        invert = bool(data.get("invert", STATE["line_invert"]))
        turn_sign = 1.0 if invert else -1.0
        corner = data.get("corner") if isinstance(data.get("corner"), dict) else {}
        enable_corner = bool(data.get("enable_corner", STATE["enable_corner"]))
        right_top_area_th = clamp(corner.get("right_top_area_th", STATE["right_top_area_th"]), 100, 4000)
        right_area_th = clamp(corner.get("right_area_th", STATE["right_area_th"]), 100, 5000)
        right_wide_ratio_th = max(1.0, min(5.0, float(corner.get("right_wide_ratio_th", STATE["right_wide_ratio_th"]))))
        right_confirm_frames = clamp(corner.get("right_confirm_frames", STATE["right_confirm_frames"]), 1, 8)
        right_turn_dir = 1 if int(corner.get("right_turn_dir", STATE["right_turn_dir"])) >= 0 else -1
        right_turn_pre_forward_sec = max(0.0, min(1.5, float(corner.get("right_turn_pre_forward_sec", STATE["right_turn_pre_forward_sec"]))))
        right_turn_pre_forward_v = max(0.0, min(0.08, float(corner.get("right_turn_pre_forward_v", STATE["right_turn_pre_forward_v"]))))
        right_turn_w = max(0.05, min(0.6, float(corner.get("right_turn_w", STATE["right_turn_w"]))))
        right_turn_min_sec = max(0.1, min(2.0, float(corner.get("right_turn_min_sec", STATE["right_turn_min_sec"]))))
        turn_finish_err_th = max(0.1, min(1.0, float(corner.get("turn_finish_err_th", STATE["turn_finish_err_th"]))))
        req_x0, y0, req_x1, y1 = crop
        # The tight line-follow crop can miss the right branch. Expand the
        # internal corner-detection ROI, while steering against the original
        # crop center so the car does not bias left.
        x0 = max(0, req_x0 - 20)
        x1 = min(640, req_x1 + 140)
        track_center = ((req_x0 + req_x1) / 2.0) - x0
        code = f"""
import base64, cv2 as cv, json, sys, time
import numpy as np
sys.path.append('/home/pi/Transbot/py_install')
from Transbot_Lib import Transbot

x0, y0, x1, y1 = {x0}, {y0}, {x1}, {y1}
crop_center = (x1 - x0) // 2
MAX_SEC = {float(max_sec)}
V_BASE = {float(speed)}
KP_W = 0.20
W_LIMIT = 0.28
TURN_SIGN = {turn_sign}
ENABLE_CORNER = {str(enable_corner)}
RIGHT_TOP_AREA_TH = {right_top_area_th}
RIGHT_AREA_TH = {right_area_th}
RIGHT_WIDE_RATIO_TH = {right_wide_ratio_th}
RIGHT_CONFIRM_FRAMES = {right_confirm_frames}
RIGHT_TURN_DIR = {right_turn_dir}
RIGHT_TURN_PRE_FORWARD_SEC = {right_turn_pre_forward_sec}
RIGHT_TURN_PRE_FORWARD_V = {right_turn_pre_forward_v}
RIGHT_TURN_W = {right_turn_w}
RIGHT_TURN_MIN_SEC = {right_turn_min_sec}
TURN_FINISH_ERR_TH = {turn_finish_err_th}
REACQUIRE_COUNT_TH = {STATE["reacquire_count_th"]}
MISS_LIMIT = 5
SHORT_LIMIT = 5
MIN_AREA = 160
MIN_HEIGHT = 28

bot = Transbot()
logs = []
stop_reason = 'max_time'
frames = 0
found = 0
last_ann = None
last_mask = None
last_aug = None
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

def stop(n=8):
    for _ in range(n):
        bot.set_car_motion(0, 0)
        time.sleep(0.035)

def detect(crop):
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    geq = clahe.apply(gray)
    th = int(np.percentile(geq, 32))
    th = max(25, min(120, th))
    mask = (geq <= th).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT,(3,3)), iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(5,17)), iterations=1)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv.contourArea(c)
        bx, by, bw, bh = cv.boundingRect(c)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if 55 < area < 8000 and bh > 24 and aspect > 1.15:
            valid.append((c, area, bx, by, bw, bh))
    return mask, cv.cvtColor(geq, cv.COLOR_GRAY2BGR), valid, th

try:
    stop(3)
    bot.set_floodlight(80)
    time.sleep(0.1)
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        stop_reason = 'camera_open_failed'
        raise SystemExit(stop_reason)
    for _ in range(8):
        cap.read()
        time.sleep(0.025)
    miss = 0
    short = 0
    corner_count = 0
    reacquire_count = 0
    drive_state = 'NORMAL_LINE'
    turn_started_at = 0.0
    start = time.time()
    while time.time() - start < MAX_SEC:
        ok, frame = cap.read()
        if not ok:
            miss += 1
            stop(1)
            logs.append({{'t': round(time.time()-start,2), 'found': False, 'reason': 'read_fail', 'miss': miss}})
            if miss >= MISS_LIMIT:
                stop_reason = 'camera_read_miss'
                break
            continue
        crop_img = frame[y0:y1, x0:x1]
        mask, aug, valid, th = detect(crop_img)
        mh, mw = mask.shape[:2]
        top_area = int(cv.countNonZero(mask[0:mh//3, :]))
        bottom_area = int(cv.countNonZero(mask[mh*2//3:mh, :]))
        left_area = int(cv.countNonZero(mask[:, 0:mw//3]))
        center_area = int(cv.countNonZero(mask[:, mw//3:mw*2//3]))
        right_area = int(cv.countNonZero(mask[:, mw*2//3:mw]))
        ann = frame.copy()
        cv.rectangle(ann, (x0,y0), (x1-1,y1-1), (80,180,255), 2)
        frames += 1
        if not valid:
            miss += 1
            short = 0
            stop(1)
            logs.append({{'t': round(time.time()-start,2), 'found': False, 'miss': miss, 'th': th}})
            cv.putText(ann, f'no line miss={{miss}} th={{th}}', (10,30), cv.FONT_HERSHEY_SIMPLEX, .62, (0,255,255), 2)
            if miss >= MISS_LIMIT:
                stop_reason = 'line_lost'
                last_ann, last_mask, last_aug = ann, mask, aug
                break
        else:
            c, area, bx, by, bw, bh = max(valid, key=lambda t: t[1])
            wide_ratio = max(bw, bh) / max(min(bw, bh), 1)
            m = cv.moments(c)
            if abs(m['m00']) < 1e-6:
                miss += 1
                stop(1)
                if miss >= MISS_LIMIT:
                    stop_reason = 'zero_moment'
                    last_ann, last_mask, last_aug = ann, mask, aug
                    break
            else:
                miss = 0
                found += 1
                cx_crop = int(m['m10']/m['m00'])
                cy_crop = int(m['m01']/m['m00'])
                cx = cx_crop + x0
                cy = cy_crop + y0
                err = cx_crop - crop_center
                err_norm = max(-1.0, min(1.0, err / max(crop_center, 1)))
                right_angle_candidate = (
                    ENABLE_CORNER
                    and drive_state == 'NORMAL_LINE'
                    and right_area >= RIGHT_AREA_TH
                    and top_area >= RIGHT_TOP_AREA_TH
                    and wide_ratio >= RIGHT_WIDE_RATIO_TH
                )
                if right_angle_candidate:
                    corner_count += 1
                else:
                    corner_count = 0
                if corner_count >= RIGHT_CONFIRM_FRAMES:
                    drive_state = 'PRE_FORWARD'
                    corner_count = 0
                    logs.append({{'t': round(time.time()-start,2), 'event': 'right_angle_confirmed', 'right_area': right_area, 'top_area': top_area, 'wide_ratio': round(wide_ratio,2)}})
                    if RIGHT_TURN_PRE_FORWARD_SEC > 0 and RIGHT_TURN_PRE_FORWARD_V > 0:
                        bot.set_car_motion(RIGHT_TURN_PRE_FORWARD_V, 0)
                        time.sleep(RIGHT_TURN_PRE_FORWARD_SEC)
                    bot.set_car_motion(0, RIGHT_TURN_DIR * RIGHT_TURN_W)
                    time.sleep(RIGHT_TURN_MIN_SEC)
                    turn_started_at = time.time()
                    drive_state = 'REACQUIRE'
                    last_ann, last_mask, last_aug = ann, mask, aug
                    continue
                if drive_state == 'REACQUIRE':
                    if abs(err_norm) <= TURN_FINISH_ERR_TH and bottom_area > MIN_AREA:
                        reacquire_count += 1
                    else:
                        reacquire_count = 0
                    if reacquire_count >= REACQUIRE_COUNT_TH:
                        drive_state = 'NORMAL_LINE'
                        reacquire_count = 0
                        logs.append({{'t': round(time.time()-start,2), 'event': 'reacquired', 'err_norm': round(err_norm,3)}})
                    else:
                        bot.set_car_motion(0, RIGHT_TURN_DIR * RIGHT_TURN_W)
                        cv.putText(ann, f'REACQUIRE err={{err}} bottom={{bottom_area}}', (10,54), cv.FONT_HERSHEY_SIMPLEX, .52, (0,255,255), 2)
                        logs.append({{'t': round(time.time()-start,2), 'state': drive_state, 'found': True, 'err': int(err), 'bottom_area': bottom_area}})
                        last_ann, last_mask, last_aug = ann, mask, aug
                        time.sleep(0.075)
                        continue
                too_short = area < MIN_AREA or bh < MIN_HEIGHT
                short = short + 1 if too_short else 0
                if short >= SHORT_LIMIT:
                    stop_reason = 'line_too_short'
                    stop(3)
                    logs.append({{'t': round(time.time()-start,2), 'found': True, 'err': int(err), 'area': int(area), 'height': int(bh), 'short': short, 'stop': stop_reason, 'th': th}})
                    cv.putText(ann, f'STOP short area={{int(area)}} h={{bh}}', (10,30), cv.FONT_HERSHEY_SIMPLEX, .62, (0,255,255), 2)
                    last_ann, last_mask, last_aug = ann, mask, aug
                    break
                v = V_BASE
                w = max(-W_LIMIT, min(W_LIMIT, TURN_SIGN * KP_W * err_norm))
                if abs(err_norm) > 0.82:
                    v = 0.0
                bot.set_car_motion(v, w)
                cv.drawContours(ann, [c + np.array([[[x0,y0]]])], -1, (0,255,0), 2)
                cv.circle(ann, (cx,cy), 5, (0,0,255), -1)
                cv.line(ann, ((x0+x1)//2,y0), ((x0+x1)//2,y1), (255,0,0), 1)
                cv.line(ann, (cx,cy), ((x0+x1)//2,cy), (0,0,255), 2)
                cv.putText(ann, f'{{drive_state}} err={{err}} v={{v:.3f}} w={{w:.3f}} area={{int(area)}} h={{bh}}', (10,30), cv.FONT_HERSHEY_SIMPLEX, .52, (0,255,255), 2)
                cv.putText(ann, f'R={{right_area}} T={{top_area}} ratio={{wide_ratio:.2f}} cc={{corner_count}}', (10,54), cv.FONT_HERSHEY_SIMPLEX, .48, (0,255,255), 1)
                logs.append({{'t': round(time.time()-start,2), 'state': drive_state, 'found': True, 'err': int(err), 'v': round(v,3), 'w': round(w,3), 'area': int(area), 'height': int(bh), 'short': short, 'th': th, 'right_area': right_area, 'top_area': top_area, 'wide_ratio': round(wide_ratio,2), 'corner_count': corner_count}})
        last_ann, last_mask, last_aug = ann, mask, aug
        time.sleep(0.075)
    cap.release()
finally:
    stop(14)

if last_ann is None:
    last_ann = np.zeros((480,640,3), dtype=np.uint8)
    last_mask = np.zeros((max(1,y1-y0), max(1,x1-x0)), dtype=np.uint8)
    last_aug = cv.cvtColor(last_mask, cv.COLOR_GRAY2BGR)
crop_view = last_ann[y0:y1, x0:x1]
comp = np.hstack([
    last_ann,
    cv.resize(crop_view, (300,480)),
    cv.resize(last_aug, (300,480)),
    cv.resize(cv.cvtColor(last_mask, cv.COLOR_GRAY2BGR), (300,480), interpolation=cv.INTER_NEAREST)
])
ok, buf = cv.imencode('.jpg', comp, [int(cv.IMWRITE_JPEG_QUALITY), 82])
summary = {{'stop_reason': stop_reason, 'frames': frames, 'found': found, 'last_logs': logs[-12:]}}
print(json.dumps({{'summary': summary, 'image': base64.b64encode(buf).decode('ascii')}}, ensure_ascii=False))
"""
        res = remote_python(code, timeout=max_sec + 12)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip())
        result = None
        for line in reversed(res.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and '"summary"' in line:
                result = json.loads(line)
                break
        if result is None:
            raise RuntimeError(res.stdout.strip() or "line demo produced no JSON result")
        with STATE_LOCK:
            STATE["crop"] = crop
            STATE["line_speed"] = speed
            STATE["line_max_sec"] = max_sec
            STATE["line_invert"] = invert
            STATE["enable_corner"] = enable_corner
            STATE["right_top_area_th"] = right_top_area_th
            STATE["right_area_th"] = right_area_th
            STATE["right_wide_ratio_th"] = right_wide_ratio_th
            STATE["right_confirm_frames"] = right_confirm_frames
            STATE["right_turn_dir"] = right_turn_dir
            STATE["right_turn_pre_forward_sec"] = right_turn_pre_forward_sec
            STATE["right_turn_pre_forward_v"] = right_turn_pre_forward_v
            STATE["right_turn_w"] = right_turn_w
            STATE["right_turn_min_sec"] = right_turn_min_sec
            STATE["turn_finish_err_th"] = turn_finish_err_th
            save_state()
        self.send_json({"ok": True, "summary": result["summary"], "image": result["image"]})

    def api_corner_demo(self):
        data = self.read_json()
        release_remote_camera()
        crop = self.normalize_crop(data.get("crop"))
        if crop[2] - crop[0] < 60 or crop[3] - crop[1] < 60:
            raise RuntimeError(f"crop too small for corner demo: {crop}; use a taller box around the black line")
        speed = max(0.01, min(0.06, float(data.get("speed", STATE["line_speed"]))))
        max_sec = clamp(data.get("max_sec", STATE["line_max_sec"]), 2, 60)
        req_x0, y0, req_x1, y1 = crop
        # Expand only the internal vision ROI so the right branch is visible.
        # Steering still uses the user's original crop center.
        x0 = max(0, req_x0 - 20)
        x1 = min(640, req_x1 + 140)
        track_center = ((req_x0 + req_x1) / 2.0) - x0
        # Internal corner tuning. Keep this hidden from UI until features are stable.
        # On this chassis negative angular velocity is the observed right turn.
        turn_dir = -1
        turn_w = 0.38
        turn_min_sec = max(0.5, min(4.0, float(data.get("turn_sec", STATE.get("corner_turn_sec", 2.45)))))
        forward_sec = max(0.0, min(5.0, float(data.get("forward_sec", STATE.get("corner_forward_sec", 1.2)))))
        trigger_frac = max(0.1, min(0.8, float(data.get("trigger_frac", STATE.get("corner_trigger_frac", 0.35)))))
        code = f"""
import base64, cv2 as cv, json, sys, time
import numpy as np
sys.path.append('/home/pi/Transbot/py_install')
from Transbot_Lib import Transbot

x0, y0, x1, y1 = {x0}, {y0}, {x1}, {y1}
crop_w = x1 - x0
crop_h = y1 - y0
crop_center = {track_center}
MAX_SEC = {float(max_sec)}
V_BASE = {float(speed)}
TURN_DIR = {turn_dir}
TURN_W = {turn_w}
TURN_MIN_SEC = {turn_min_sec}
FORWARD_SEC = {forward_sec}
MISS_LIMIT = 7
CORNER_CONFIRM = 2
TRIGGER_Y_FRAC = {trigger_frac}
REACQUIRE_CONFIRM = 3
SCAN_BANDS = 5
KP = 0.24
MAX_W = 0.24

bot = Transbot()
logs = []
stop_reason = 'max_time'
frames = 0
found = 0
last_ann = None
last_mask = None
last_aug = None
state = 'FOLLOW_STRAIGHT'
corner_count = 0
reacquire_count = 0
last_err_norm = 0.0
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))

def stop(n=8):
    for _ in range(n):
        bot.set_car_motion(0, 0)
        time.sleep(0.035)

def detect(crop):
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    geq = clahe.apply(gray)
    th = int(np.percentile(geq, 32))
    th = max(25, min(120, th))
    mask = (geq <= th).astype(np.uint8) * 255
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT,(3,3)), iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT,(5,17)), iterations=1)
    return mask, cv.cvtColor(geq, cv.COLOR_GRAY2BGR), th

def runs_from_band(mask, y0b, y1b):
    band = mask[y0b:y1b, :]
    h = max(1, y1b - y0b)
    col = cv.reduce(band, 0, cv.REDUCE_SUM, dtype=cv.CV_32S).reshape(-1)
    active = col > int(255 * h * 0.18)
    runs = []
    start_run = None
    for i, on in enumerate(active.tolist() + [False]):
        if on and start_run is None:
            start_run = i
        elif (not on) and start_run is not None:
            end_run = i - 1
            if end_run - start_run + 1 >= 3:
                roi = band[:, start_run:end_run+1]
                area = int(cv.countNonZero(roi))
                runs.append({{'x0': start_run, 'x1': end_run, 'cx': (start_run + end_run) / 2.0, 'w': end_run - start_run + 1, 'area': area}})
            start_run = None
    return runs

def scan_features(mask):
    h, w = mask.shape[:2]
    bands = []
    for i in range(SCAN_BANDS):
        # bottom band is index 0, upper bands look farther ahead.
        y1b = h - int(i * h / SCAN_BANDS)
        y0b = h - int((i + 1) * h / SCAN_BANDS)
        runs = runs_from_band(mask, y0b, y1b)
        best = None
        if runs:
            # For line following, prefer the run closest to image center. Wide runs are
            # still retained as branch evidence below.
            best = min(runs, key=lambda r: abs(r['cx'] - crop_center) - min(r['area'], 800) * 0.001)
        bands.append({{'y0': y0b, 'y1': y1b, 'runs': runs, 'best': best}})
    valid = [b for b in bands if b['best']]
    if not valid:
        return {{'found': False, 'bands': bands}}
    bottom = bands[0]['best'] or (bands[1]['best'] if len(bands) > 1 else None)
    mid = bands[1]['best'] or bottom
    far = bands[3]['best'] or bands[4]['best'] or mid
    near_widths = [b['best']['w'] for b in bands[:2] if b['best']]
    all_widths = [b['best']['w'] for b in valid]
    widths = near_widths or all_widths
    line_px_w = float(np.median(widths)) if widths else 8.0
    err_cx = (0.65 * bottom['cx'] + 0.35 * mid['cx']) if bottom and mid else valid[0]['best']['cx']
    err_norm = max(-1.0, min(1.0, (err_cx - crop_center) / max(crop_center, 1)))
    branch = None
    for bi in [2, 3, 4, 1]:
        for r in bands[bi]['runs']:
            reaches_right = r['x1'] > crop_center + max(12, line_px_w * 1.4)
            wide_enough = r['w'] > max(line_px_w * 2.2, crop_w * 0.22)
            center_ok = r['x0'] < crop_center + max(10, line_px_w * 1.3)
            if reaches_right and wide_enough and center_ok:
                if branch is None or r['area'] > branch['area']:
                    branch = dict(r)
                    branch['band'] = bi
                    branch['y'] = (bands[bi]['y0'] + bands[bi]['y1']) / 2.0
    bottom_ok = bottom is not None and bottom['area'] > 30
    mid_ok = mid is not None and mid['area'] > 30
    straight_ok = bottom_ok and mid_ok and abs(err_norm) < 0.70
    corner_candidate = bool(branch and straight_ok)
    return {{'found': True, 'bands': bands, 'bottom': bottom, 'mid': mid, 'far': far, 'line_px_w': line_px_w, 'err_norm': err_norm, 'corner_candidate': corner_candidate, 'branch': branch}}

def fixed_forward_sec():
    return FORWARD_SEC

def corner_trigger_ready(features):
    branch = features.get('branch')
    if not branch:
        return False, None
    branch_y = float(branch.get('y', 0.0))
    return branch_y >= crop_h * TRIGGER_Y_FRAC, branch_y

def draw_scan(ann, features):
    cv.rectangle(ann, (x0,y0), (x1-1,y1-1), (80,180,255), 2)
    cv.line(ann, (x0+int(crop_center),y0), (x0+int(crop_center),y1), (255,0,0), 1)
    trigger_y = y0 + int(crop_h * TRIGGER_Y_FRAC)
    cv.line(ann, (x0, trigger_y), (x1, trigger_y), (255, 0, 255), 2)
    for idx, b in enumerate(features.get('bands', [])):
        yy = y0 + (b['y0'] + b['y1']) // 2
        cv.line(ann, (x0, yy), (x1, yy), (90,90,90), 1)
        for r in b['runs']:
            color = (0, 220, 0) if b['best'] is r else (180, 180, 0)
            cv.rectangle(ann, (x0 + int(r['x0']), y0 + b['y0']), (x0 + int(r['x1']), y0 + b['y1'] - 1), color, 1)
    branch = features.get('branch')
    if branch:
        bi = int(branch['band'])
        b = features['bands'][bi]
        cv.rectangle(ann, (x0+int(branch['x0']), y0+b['y0']), (x0+int(branch['x1']), y0+b['y1']-1), (0,165,255), 2)

try:
    stop(3)
    bot.set_floodlight(80)
    time.sleep(0.08)
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        stop_reason = 'camera_open_failed'
        raise SystemExit(stop_reason)
    for _ in range(8):
        cap.read()
        time.sleep(0.025)
    miss = 0
    start = time.time()
    while time.time() - start < MAX_SEC:
        ok, frame = cap.read()
        if not ok:
            miss += 1
            stop(1)
            if miss >= MISS_LIMIT:
                stop_reason = 'camera_read_miss'
                break
            continue
        crop_img = frame[y0:y1, x0:x1]
        mask, aug, th = detect(crop_img)
        features = scan_features(mask)
        ann = frame.copy()
        draw_scan(ann, features)
        frames += 1
        if not features.get('found'):
            miss += 1
            # Like common line followers: if the line vanishes briefly, rotate in
            # the last correction direction instead of freezing immediately.
            if state == 'REACQUIRE':
                bot.set_car_motion(0, TURN_DIR * TURN_W)
            else:
                bot.set_car_motion(0, max(-0.16, min(0.16, -0.14 * last_err_norm)))
            logs.append({{'t': round(time.time()-start,2), 'state': state, 'found': False, 'miss': miss}})
            if miss >= MISS_LIMIT:
                stop_reason = 'line_lost'
                last_ann, last_mask, last_aug = ann, mask, aug
                break
        else:
            miss = 0
            found += 1
            err_norm = float(features['err_norm'])
            last_err_norm = err_norm
            bottom = features.get('bottom') or features.get('mid') or features.get('far')
            cx_crop = int(bottom['cx'])
            cy_crop = int((features['bands'][0]['y0'] + features['bands'][0]['y1']) / 2)
            cx = cx_crop + x0
            cy = cy_crop + y0
            err = int(err_norm * crop_center)
            corner_candidate = bool(features.get('corner_candidate'))
            if state == 'FOLLOW_STRAIGHT':
                trigger_ready, branch_y = corner_trigger_ready(features)
                trigger_candidate = bool(corner_candidate and trigger_ready)
                if trigger_candidate:
                    corner_count += 1
                else:
                    corner_count = 0
                if corner_count >= CORNER_CONFIRM:
                    drive_sec = fixed_forward_sec()
                    state = 'DEAD_RECKON_TO_TURN'
                    branch = features.get('branch') or {{}}
                    logs.append({{'t': round(time.time()-start,2), 'event': 'corner_confirmed_fixed_time', 'state': state, 'forward_sec': round(drive_sec,2), 'trigger_y': round(crop_h * TRIGGER_Y_FRAC,1), 'branch_y': None if branch_y is None else round(branch_y,1), 'branch': branch, 'turn_dir': TURN_DIR, 'turn_w': TURN_W, 'turn_min_sec': TURN_MIN_SEC}})
                    cv.putText(ann, f'FORWARD {{drive_sec:.2f}}s', (10,54), cv.FONT_HERSHEY_SIMPLEX, .55, (0,255,255), 2)
                    bot.set_car_motion(V_BASE, 0)
                    time.sleep(drive_sec)
                    state = 'TURNING'
                    logs.append({{'t': round(time.time()-start,2), 'event': 'turn_start', 'state': state, 'forward_sec': round(drive_sec,2), 'turn_sec': TURN_MIN_SEC, 'turn_w': TURN_W, 'turn_dir': TURN_DIR}})
                    bot.set_car_motion(0, TURN_DIR * TURN_W)
                    time.sleep(TURN_MIN_SEC)
                    state = 'REACQUIRE'
                    corner_count = 0
                    last_ann, last_mask, last_aug = ann, mask, aug
                    continue
                w = max(-MAX_W, min(MAX_W, -KP * err_norm))
                v = V_BASE * (1.0 - min(0.55, abs(err_norm) * 0.45))
                bot.set_car_motion(v, w)
            elif state == 'REACQUIRE':
                if abs(err_norm) < 0.50 and bottom is not None and bottom.get('area', 0) > 30:
                    reacquire_count += 1
                else:
                    reacquire_count = 0
                if reacquire_count >= REACQUIRE_CONFIRM:
                    stop_reason = 'corner_done'
                    logs.append({{'t': round(time.time()-start,2), 'event': 'reacquired', 'err_norm': round(err_norm,3)}})
                    stop(4)
                    last_ann, last_mask, last_aug = ann, mask, aug
                    break
                bot.set_car_motion(0, TURN_DIR * TURN_W)
            cv.circle(ann, (cx,cy), 5, (0,0,255), -1)
            cv.line(ann, (cx,cy), (x0+int(crop_center),cy), (0,0,255), 2)
            cv.putText(ann, f'{{state}} err={{err}} cand={{int(corner_candidate)}} cc={{corner_count}}', (10,30), cv.FONT_HERSHEY_SIMPLEX, .52, (0,255,255), 2)
            branch = features.get('branch') or {{}}
            logs.append({{'t': round(time.time()-start,2), 'state': state, 'found': True, 'err': int(err), 'err_norm': round(err_norm,3), 'line_px_w': round(features.get('line_px_w',0),1), 'corner_candidate': bool(corner_candidate), 'corner_count': corner_count, 'branch_band': branch.get('band'), 'branch_w': branch.get('w'), 'branch_y': branch.get('y'), 'trigger_y': round(crop_h * TRIGGER_Y_FRAC,1)}})
        last_ann, last_mask, last_aug = ann, mask, aug
        time.sleep(0.075)
    cap.release()
finally:
    stop(14)

if last_ann is None:
    last_ann = np.zeros((480,640,3), dtype=np.uint8)
    last_mask = np.zeros((max(1,y1-y0), max(1,x1-x0)), dtype=np.uint8)
    last_aug = cv.cvtColor(last_mask, cv.COLOR_GRAY2BGR)
crop_view = last_ann[y0:y1, x0:x1]
comp = np.hstack([
    last_ann,
    cv.resize(crop_view, (300,480)),
    cv.resize(last_aug, (300,480)),
    cv.resize(cv.cvtColor(last_mask, cv.COLOR_GRAY2BGR), (300,480), interpolation=cv.INTER_NEAREST)
])
ok, buf = cv.imencode('.jpg', comp, [int(cv.IMWRITE_JPEG_QUALITY), 82])
summary = {{'stop_reason': stop_reason, 'frames': frames, 'found': found, 'last_logs': logs[-16:]}}
print(json.dumps({{'summary': summary, 'image': base64.b64encode(buf).decode('ascii')}}, ensure_ascii=False))
"""
        res = remote_python(code, timeout=max_sec + 18)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or res.stdout.strip())
        result = None
        for line in reversed(res.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and '"summary"' in line:
                result = json.loads(line)
                break
        if result is None:
            raise RuntimeError(res.stdout.strip() or "corner demo produced no JSON result")
        with STATE_LOCK:
            STATE["crop"] = crop
            STATE["line_speed"] = speed
            STATE["line_max_sec"] = max_sec
            STATE["corner_forward_sec"] = forward_sec
            STATE["corner_turn_sec"] = turn_min_sec
            STATE["corner_trigger_frac"] = trigger_frac
            save_state()
        self.send_json({"ok": True, "summary": result["summary"], "image": result["image"]})

    def stream_video(self):
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
            ["ssh", SSH_TARGET, "python3", "-u", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        with VIDEO_LOCK:
            VIDEO_PROCS.add(proc)
        proc.stdin.write(code.encode("utf-8"))
        proc.stdin.close()
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while True:
                chunk = proc.stdout.read(16384)
                if not chunk:
                    break
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
            with VIDEO_LOCK:
                VIDEO_PROCS.discard(proc)


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
