const HISTORY = 240;
const series = { e0: [], theta: [], conf: [] };
let mockOn = false;
const el = (id) => document.getElementById(id);

const LABELS = {
  corner: "直角", ring_entry: "环岛入口", ring_exit: "环岛出口",
  fork: "三岔路", return: "返回", finished: "完成",
  track: "巡线", lost: "丢线", stopped: "停车",
  follow: "跟随", predict: "预测", pivot: "对准", plan: "预览执行",
  cruise: "普通巡线", corner_executor: "直角执行器",
  ring_executor: "环岛执行器", fork_executor: "岔路执行器",
  terminal_executor: "终点执行器", none: "无",
  active: "运行中", zero_command: "归零命令",
  await_stationary: "等待静止", dispose_outgoing: "销毁旧阶段",
  initialize_incoming: "初始化新阶段", switch_owner: "切换控制权",
  ramp: "从零平滑起步", measured: "实测静止",
  measured_unverified: "未验证新鲜度的反馈",
  timed_fallback: "保守计时静止", clear: "安全放行",
  aligning: "视觉对齐",
  route_lost: "路线丢失", stop_latched: "停车锁定",
  route_lost_cleared: "路线丢失锁已清除", clear_rejected: "清除请求被拒绝",
  blocked: "安全阻止", normal: "正常",
};

function bilingual(value) {
  if (value === null || value === undefined || value === "") return "--";
  const raw = String(value);
  return `${LABELS[raw] || "未知"} · ${raw}`;
}

function setConnected(live) {
  el("dot").classList.toggle("live", live);
  el("connText").textContent = live ? "遥测已连接" : "遥测未连接";
}

function setSignedBar(id, value, scale) {
  const frac = Math.max(-1, Math.min(1, Number(value || 0) / scale));
  const fill = el("bar_" + id);
  const half = 50 * Math.abs(frac);
  fill.style.left = frac >= 0 ? "50%" : `${50 - half}%`;
  fill.style.width = `${half}%`;
}

function setBar(id, value, scale) {
  el("bar_" + id).style.width = `${Math.max(0, Math.min(1, Number(value || 0) / scale)) * 100}%`;
}

function setValue(id, value, digits = 3) {
  el("val_" + id).textContent = Number(value || 0).toFixed(digits);
}

function commandText(command) {
  if (!command) return "v -- / w --";
  return `v ${Number(command.v || 0).toFixed(4)} / w ${Number(command.w || 0).toFixed(4)} · ${command.reason || "--"}`;
}

function timing(id, value) {
  el(id).textContent = value === null || value === undefined ? "-- ms" : `${Number(value).toFixed(1)} ms`;
}

function render(s) {
  const state = s.state || "stopped";
  el("stateLight").className = `light ${state}`;
  el("stateName").textContent = bilingual(state);
  el("modeName").textContent = bilingual(s.mode);
  el("missionState").textContent = bilingual(s.mission_state || s.course_session);
  el("sessionDetector").textContent = bilingual(s.session_detector);
  el("sessionExecutor").textContent = bilingual(s.session_executor);
  el("executorPhase").textContent = bilingual(s.executor_phase);
  el("candidateProducer").textContent = bilingual(s.candidate_producer);
  el("controlOwner").textContent = bilingual(s.control_owner);
  el("barrierState").textContent = bilingual(s.transition_barrier_state);
  el("stationaryMethod").textContent = bilingual(s.stationary_method);
  el("safetyState").textContent = bilingual(s.safety_state);
  el("stopCause").textContent = bilingual(s.stop_cause);
  el("transitionEvent").textContent = bilingual(s.mission_transition_event || s.transition_event);
  el("candidateCommand").textContent = commandText(s.candidate_command);
  el("finalCommand").textContent = commandText(s.final_command);

  setValue("e0", s.e0); setValue("theta", s.theta);
  setValue("conf", s.conf); setValue("v", s.v, 4); setValue("w", s.w);
  setSignedBar("e0", s.e0, 1); setSignedBar("theta", s.theta, 1.2);
  setSignedBar("w", s.w, 0.35); setBar("conf", s.conf, 1); setBar("v", s.v, 0.12);

  timing("timingCapture", s.timing_capture_ms);
  timing("timingVision", s.timing_ordinary_vision_ms);
  timing("timingGeometry", s.timing_geometry_ms);
  timing("timingControl", s.timing_control_ms);
  timing("timingWork", s.timing_loop_total_ms);
  el("deadlineMiss").textContent = s.deadline_miss ? `是 · ${Number(s.deadline_lag_ms || 0).toFixed(1)} ms` : "否";
  el("deadlineMiss").classList.toggle("dangerText", !!s.deadline_miss);

  for (const key of Object.keys(series)) {
    series[key].push(Number(s[key] || 0));
    if (series[key].length > HISTORY) series[key].shift();
  }
  scheduleChart();
}

let chartPending = false;
function scheduleChart() {
  if (chartPending) return;
  chartPending = true;
  requestAnimationFrame(() => { chartPending = false; drawChart(); });
}

function drawChart() {
  const canvas = el("chart"), ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d";
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  const plot = (data, color, min, max) => {
    if (data.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
    data.forEach((value, index) => {
      const x = index / (HISTORY - 1) * w;
      const y = h - (value - min) / (max - min) * h;
      index === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plot(series.e0, "#2f81f7", -1, 1);
  plot(series.theta, "#d29922", -1.2, 1.2);
  plot(series.conf, "#3fb950", 0, 1);
}

function connect() {
  const source = new EventSource("/api/telemetry");
  source.onopen = () => setConnected(true);
  source.onmessage = (event) => { try { render(JSON.parse(event.data)); } catch (_) {} };
  source.onerror = () => setConnected(false);
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await response.json();
  if (!response.ok || !data.ok) throw new Error(data.error || response.statusText);
  return data;
}

function setMsg(text, error = false) {
  el("msg").textContent = text;
  el("msg").classList.toggle("err", error);
}

async function guard(label, fn) {
  setMsg(`${label} ...`);
  try {
    const data = await fn();
    setMsg(`${label}: ${data.message || "完成"}${data.debug_path ? `\nDebug: ${data.debug_path}` : ""}`);
    await refreshRunStatus();
  } catch (error) {
    setMsg(`${label} 失败: ${error.message}`, true);
  }
}

const sshTarget = () => el("ssh").value.trim() || "yahboom";

async function toggleMock() {
  mockOn = !mockOn;
  const result = await post("/api/mock", { enabled: mockOn });
  el("mockBtn").textContent = mockOn ? "关闭模拟" : "模拟数据";
  return result;
}

async function loadConfig() {
  const cfg = await (await fetch("/api/config")).json();
  el("strategyMargin").value = cfg.path_memory.camera_to_axle_m;
  el("cornerConfirm").value = cfg.path_memory.corner_confirm_frames;
  el("cornerTurnW").value = cfg.path_memory.corner_replay_max_w;
  el("roundaboutDirection").value = String(cfg.mission.ring_entry_direction);
  el("roundaboutTurnW").value = cfg.path_memory.roundabout_replay_max_w;
  el("roundaboutMarginDistance").value = cfg.path_memory.roundabout_margin_distance_m;
  el("roundaboutEntryLeftTurn").value = cfg.path_memory.roundabout_entry_left_turn_deg;
  el("roundaboutFixedRadius").value = cfg.path_memory.roundabout_fixed_radius_m;
  el("roundaboutChordScale").value = cfg.path_memory.roundabout_chord_distance_scale;
  el("roundaboutMargin").checked = !!cfg.path_memory.roundabout_margin_enabled;
  el("trackerVMax").value = cfg.tracker.v_max;
  el("trackerKE").value = cfg.tracker.k_e;
  el("trackerKTheta").value = cfg.tracker.k_theta;
  el("trackerKff").value = cfg.tracker.k_ff;
  el("trackerMaxW").value = cfg.tracker.max_w;
  el("trackerConfDecay").value = cfg.tracker.conf_decay;
  el("visionMinWidth").value = cfg.vision.min_run_width_px;
  el("visionMinArea").value = cfg.vision.min_run_area_px;
  el("cam1").value = cfg.ui.cam1;
  el("cam2").value = cfg.ui.cam2;
  el("armJ1").value = cfg.ui.j1;
  el("armJ2").value = cfg.ui.j2;
  el("armJ3").value = cfg.ui.j3;
  el("armMs").value = cfg.ui.arm_ms;
  [el("cropX0").value, el("cropY0").value, el("cropX1").value, el("cropY1").value] = cfg.camera.crop;
}

async function refreshRunStatus() {
  try {
    const status = await (await fetch("/api/run/current")).json();
    el("runName").textContent = status.run ? `${status.running ? "运行中" : "最近运行"} · ${status.run}` : "暂无 Debug 运行";
    el("downloadBtn").disabled = !(status.sync_available ?? status.download_available);
    const age = status.frame_age_sec;
    el("frameAge").textContent = age === null ? "无帧" : `${Number(age).toFixed(1)} s`;
    el("frameAge").classList.toggle("stale", age !== null && age > 1.0);
    el("videoEmpty").hidden = age !== null && age < 5.0;
  } catch (_) {}
}

function reconnectVideo() {
  el("video").src = `/video?ts=${Date.now()}`;
  setMsg("正在连接 runner 发布画面");
}

el("mockBtn").addEventListener("click", () => guard("模拟数据", toggleMock));
el("deployBtn").addEventListener("click", () => guard("部署", () => post("/api/deploy", { ssh_target: sshTarget() })));
el("stopBtn").addEventListener("click", () => guard("停车", () => post("/api/live/stop", { ssh_target: sshTarget() })));
el("clearRouteBtn").addEventListener("click", () => guard("路线恢复", () => post("/api/live/clear-route-loss", { ssh_target: sshTarget() })));
el("reloadVideoBtn").addEventListener("click", reconnectVideo);
el("downloadBtn").addEventListener("click", () => guard(
  "同步 Debug",
  () => post("/api/run/current/sync", { ssh_target: sshTarget() }),
));
document.querySelector("button.run").addEventListener("click", () => guard("启动", () => post("/api/live/start", {
  profile: "final", ssh_target: sshTarget(), max_sec: Number(el("maxSecN").value),
  debug: el("debugCapture").checked,
})));
el("strategySaveBtn").addEventListener("click", () => guard("保存高级设置", () => post("/api/defaults/save", {
  path_memory: {
    camera_to_axle_m: Number(el("strategyMargin").value),
    corner_confirm_frames: Number(el("cornerConfirm").value),
    corner_replay_max_w: Number(el("cornerTurnW").value),
    roundabout_replay_max_w: Number(el("roundaboutTurnW").value),
    roundabout_margin_distance_m: Number(el("roundaboutMarginDistance").value),
    roundabout_entry_left_turn_deg: Number(el("roundaboutEntryLeftTurn").value),
    roundabout_fixed_radius_m: Number(el("roundaboutFixedRadius").value),
    roundabout_chord_distance_scale: Number(el("roundaboutChordScale").value),
    roundabout_margin_enabled: el("roundaboutMargin").checked,
  },
  mission: {
    ring_entry_direction: Number(el("roundaboutDirection").value),
  },
  tracker: {
    v_max: Number(el("trackerVMax").value),
    k_e: Number(el("trackerKE").value),
    k_theta: Number(el("trackerKTheta").value),
    k_ff: Number(el("trackerKff").value),
    max_w: Number(el("trackerMaxW").value),
    conf_decay: Number(el("trackerConfDecay").value),
  },
  vision: {
    min_run_width_px: Number(el("visionMinWidth").value),
    min_run_area_px: Number(el("visionMinArea").value),
  },
  camera: {
    crop: ["cropX0", "cropY0", "cropX1", "cropY1"].map((id) => Number(el(id).value)),
  },
  ui: {
    cam1: Number(el("cam1").value), cam2: Number(el("cam2").value),
    j1: Number(el("armJ1").value), j2: Number(el("armJ2").value),
    j3: Number(el("armJ3").value), arm_ms: Number(el("armMs").value),
  },
})));
el("cameraApplyBtn").addEventListener("click", () => guard("应用云台", () => post("/api/camera", {
  ssh_target: sshTarget(), cam1: Number(el("cam1").value), cam2: Number(el("cam2").value),
})));
el("armApplyBtn").addEventListener("click", () => guard("应用机械臂", () => post("/api/arm", {
  ssh_target: sshTarget(), j1: Number(el("armJ1").value), j2: Number(el("armJ2").value),
  j3: Number(el("armJ3").value), arm_ms: Number(el("armMs").value),
})));
el("analyzeBtn").addEventListener("click", () => guard("分析单帧", async () => {
  const result = await post("/api/analyze", { image_path: el("imagePath").value.trim() });
  const image = el("analysisImage");
  image.src = `data:image/jpeg;base64,${result.image}`;
  image.hidden = false;
  return { message: JSON.stringify(result.summary) };
}));

el("video").addEventListener("load", () => { el("videoEmpty").hidden = true; });
loadConfig(); refreshRunStatus(); reconnectVideo(); connect();
setInterval(refreshRunStatus, 1000);
