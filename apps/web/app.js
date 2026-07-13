// Transbot race dashboard: subscribes to the backend SSE telemetry stream,
// drives the state panel, signed value bars, and a rolling timeseries chart.

const HISTORY = 240;
const series = { e0: [], theta: [], conf: [] };
let mockOn = false;

const el = (id) => document.getElementById(id);

function setConnected(live) {
  el("dot").classList.toggle("live", live);
  el("connText").textContent = live ? "connected" : "disconnected";
}

function setSignedBar(id, value, scale) {
  // value in [-scale, scale] -> fill grows left/right from the 50% midline.
  const frac = Math.max(-1, Math.min(1, value / scale));
  const fill = el("bar_" + id);
  const half = 50 * Math.abs(frac);
  if (frac >= 0) {
    fill.style.left = "50%";
    fill.style.width = half + "%";
  } else {
    fill.style.left = 50 - half + "%";
    fill.style.width = half + "%";
  }
}

function setBar(id, value, scale) {
  el("bar_" + id).style.width = Math.max(0, Math.min(1, value / scale)) * 100 + "%";
}

function render(s) {
  const state = s.state || "stopped";
  el("stateLight").className = "light " + state;
  el("stateName").textContent = state;
  el("modeName").textContent = s.mode || "--";
  if (s.path_strategy) {
    el("strategyStatus").textContent = [
      s.path_strategy,
      s.path_strategy_reason,
      s.path_strategy_remaining_m != null ? Number(s.path_strategy_remaining_m).toFixed(3) + " m" : "",
      s.path_strategy_points ? s.path_strategy_points + " pts" : "",
    ].filter(Boolean).join(" · ");
  }

  const set = (id, v, digits = 3) => (el("val_" + id).textContent = Number(v || 0).toFixed(digits));
  set("e0", s.e0); set("theta", s.theta); set("kappa", s.kappa);
  set("conf", s.conf); set("v", s.v, 4); set("w", s.w);

  setSignedBar("e0", s.e0 || 0, 1);
  setSignedBar("theta", s.theta || 0, 1.2);
  setSignedBar("kappa", s.kappa || 0, 1);
  setSignedBar("w", s.w || 0, 0.35);
  setBar("conf", s.conf || 0, 1);
  setBar("v", s.v || 0, 0.12);

  for (const key of ["e0", "theta", "conf"]) {
    series[key].push(Number(s[key] || 0));
    if (series[key].length > HISTORY) series[key].shift();
  }
  scheduleChart();
}

let chartPending = false;
function scheduleChart() {
  if (chartPending) return;
  chartPending = true;
  requestAnimationFrame(() => {
    chartPending = false;
    drawChart();
  });
}

function drawChart() {
  const c = el("chart");
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#21262d";
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();

  const plot = (data, color, min, max) => {
    if (data.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = (i / (HISTORY - 1)) * w;
      const y = h - ((v - min) / (max - min)) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plot(series.e0, "#2f81f7", -1, 1);
  plot(series.theta, "#d29922", -1.2, 1.2);
  plot(series.conf, "#3fb950", 0, 1);
}

function connect() {
  const es = new EventSource("/api/telemetry");
  es.onopen = () => setConnected(true);
  es.onmessage = (ev) => {
    try { render(JSON.parse(ev.data)); } catch (_) {}
  };
  es.onerror = () => { setConnected(false); };
}

async function toggleMock() {
  mockOn = !mockOn;
  await fetch("/api/mock", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: mockOn }),
  });
  el("mockBtn").textContent = mockOn ? "Mock off" : "Mock on";
}

async function loadCropBox() {
  try {
    const cfg = await (await fetch("/api/config")).json();
    const crop = cfg.camera && cfg.camera.crop;
    const img = el("video"), box = el("cropBox");
    if (!crop) return;
    const draw = () => {
      const nw = img.naturalWidth || 640, nh = img.naturalHeight || 480;
      const sx = img.clientWidth / nw, sy = img.clientHeight / nh;
      box.style.display = "block";
      box.style.left = crop[0] * sx + "px";
      box.style.top = crop[1] * sy + "px";
      box.style.width = (crop[2] - crop[0]) * sx + "px";
      box.style.height = (crop[3] - crop[1]) * sy + "px";
    };
    img.addEventListener("load", draw);
    window.addEventListener("resize", draw);
  } catch (_) {}
}

el("mockBtn").addEventListener("click", toggleMock);
el("video").src = "/video?ts=" + Date.now();
loadCropBox();
connect();

// -- Margin strategy -----------------------------------------------------
const strategyMode = el("strategyMode");

function showStrategyFields() {
  document.querySelectorAll(".strategyFields").forEach((node) => {
    node.hidden = !node.dataset.modes.split(",").includes(strategyMode.value);
  });
}

async function loadStrategy() {
  const cfg = await (await fetch("/api/config")).json();
  strategyMode.value = cfg.path_memory.mode;
  el("obstacleEnabled").checked = !!cfg.obstacle.enabled;
  el("strategyMargin").value = cfg.path_memory.camera_to_axle_m;
  el("cornerConfirm").value = cfg.path_memory.corner_confirm_frames;
  el("cornerTheta").value = cfg.path_memory.corner_theta_threshold;
  el("cornerTurnAngle").value = cfg.path_memory.corner_turn_angle_rad;
  el("cornerReacquireAngle").value = cfg.path_memory.corner_reacquire_angle_rad;
  el("cornerTurnW").value = cfg.path_memory.corner_replay_max_w;
  el("cornerTurnSpeed").value = cfg.path_memory.corner_turn_speed_ratio;
  showStrategyFields();
}

strategyMode.addEventListener("change", showStrategyFields);
el("strategySaveBtn").addEventListener("click", () => guard("STRATEGY", async () => {
  await post("/api/defaults/save", {
    path_memory: {
      enabled: true,
      mode: strategyMode.value,
      camera_to_axle_m: Number(el("strategyMargin").value),
      corner_confirm_frames: Number(el("cornerConfirm").value),
      corner_theta_threshold: Number(el("cornerTheta").value),
      corner_turn_angle_rad: Number(el("cornerTurnAngle").value),
      corner_reacquire_angle_rad: Number(el("cornerReacquireAngle").value),
      corner_replay_max_w: Number(el("cornerTurnW").value),
      corner_turn_speed_ratio: Number(el("cornerTurnSpeed").value),
    },
    obstacle: { enabled: el("obstacleEnabled").checked },
  });
  return { message: "selected " + strategyMode.value };
}));

loadStrategy();

// -- Run controls ---------------------------------------------------------
const sshTarget = () => el("ssh").value.trim() || "yahboom";

function setMsg(text, isErr) {
  const m = el("msg");
  m.textContent = text;
  m.classList.toggle("err", !!isErr);
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json();
  if (!res.ok || !data.ok) throw new Error(data.error || res.statusText);
  return data;
}

async function guard(label, fn) {
  setMsg(label + " ...");
  try {
    const d = await fn();
    setMsg(label + ": " + (d.message || "ok") + (d.debug_path ? "\nDebug: " + d.debug_path : ""));
  } catch (e) {
    setMsg(label + " FAILED: " + e.message, true);
  }
}

// Keep the Max-s slider and its number box in sync.
const maxSec = el("maxSec"), maxSecN = el("maxSecN");
maxSec.addEventListener("input", () => (maxSecN.value = maxSec.value));
maxSecN.addEventListener("input", () => (maxSec.value = maxSecN.value));

el("deployBtn").addEventListener("click", () =>
  guard("DEPLOY", () => post("/api/deploy", { ssh_target: sshTarget() }))
);
el("stopBtn").addEventListener("click", () =>
  guard("STOP", () => post("/api/live/stop", { ssh_target: sshTarget() }))
);
el("reloadVideoBtn").addEventListener("click", () => {
  el("video").src = "/video?ssh_target=" + encodeURIComponent(sshTarget()) + "&ts=" + Date.now();
  setMsg("video reconnecting");
});
document.querySelectorAll("button.run").forEach((btn) =>
  btn.addEventListener("click", () =>
    guard(btn.textContent.trim(), () =>
      post("/api/live/start", {
        profile: btn.dataset.profile,
        ssh_target: sshTarget(),
        max_sec: Number(maxSec.value),
        debug: el("debugCapture").checked,
      })
    )
  )
);
