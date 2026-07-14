# 控制流程与 Transition 审计交接报告

> **用途**：本文件是交给后续实现 Agent 的事实基线和修改契约。  
> **审计日期**：2026-07-14  
> **审计基准**：`a580c0ab0acb64fb9b3c17cfcf88e58812fdd36d` + 当前 dirty 工作树  
> **重要边界**：`CURRENT` 是当前代码事实；`TARGET` 是建议目标，尚未实现。不得把目标架构描述成现有能力。

## 2026-07-14 implementation status

Implemented on `codex/control-transition-refactor`:

- explicit `ControlOwner`, `SafetyState`, `StopCause`, owner epoch and a single
  candidate/final command arbiter;
- persistent stop latch with explicit clear for non-recoverable causes;
- typed mission, corner handoff and ring phase events;
- `FINISHED`, route-loss and executor-failure safety stops;
- obstacle hold semantics that do not advance a non-owning cruise controller;
- process-level motor lease and bounded gateway for auto/manual runners;
- manual action generation invalidation and worker join before recorder close;
- consecutive geometry epochs and temporal confirmation for strong exit lines;
- safe boolean config coercion and repository-local pytest collection.

Still not implemented: color classification, fork/terminal executors, complete
return mission graph, shared config schema for every UI save path, and physical
full-course replay/robot validation. The CURRENT sections below remain the
historical audit baseline and should not be read as post-refactor behavior.

---

## 0. 给实现 Agent 的强制指令

1. **先读完本报告，再改代码。** 不要从某个失败 run 出发继续添加特判。
2. **先建立 transition 契约和测试，再改控制逻辑。** 每条 transition 必须有唯一 guard、明确计数器、明确副作用和唯一控制权归属。
3. **不要一次重写整个系统。** 按本报告的迁移阶段逐步替换，并在每阶段保持现有可验证行为。
4. **不要把 `reason`、shape 字符串或 `MotionCommand.state` 当作跨模块控制协议。** 它们当前正是主要冲突源。
5. **每个 control tick 只能有一个 motor owner。** Safety 可以覆盖最终命令，但不能伪造 mission state 或 owner。
6. **不要覆盖或清理当前工作树的既有修改。** 本审计包含未提交的 `mission.py`、`ring_entry.py`、manual drive 和相关测试。
7. **不得以“测试通过”代替状态模型审查。** 当前测试多为单场景回归，尚未形成完整 transition matrix。
8. **禁止再写 run ID 特判。** 历史 run 只能作为失败证据，不能成为 guard 的语义来源。

---

## 1. 一句话结论

当前程序不是一个统一状态机，而是以下系统并行运行，再由 `apps/race_runner.py` 的条件分支临时仲裁：

1. `FixedSessionMission`：课程阶段；
2. `RaceStateMachine`：普通巡线、丢线搜索和部分停止；
3. `CornerCommandDelay`：角点/旧 fork 执行器；
4. `RingEntryExecutor`：环岛入口、环内和出口执行器；
5. `ObstacleMonitor`：最后覆盖 stop/slow；
6. `manual_drive_app.py`：另一个独立进程中的电机写入者。

真正的问题不是“状态太少”，而是：

- mission、owner、safety、executor phase 混在不同对象中；
- `STOPPED` 至少有三种不一致表达；
- `completed` 同时表达阶段完成和执行器完成；
- `reason` 和 `event_shape` 字符串参与控制；
- direction 和 confidence 在不同模块有不同语义；
- 同一 tick 的输出可能被多次替换，却没有结构化仲裁记录；
- manual 与 auto 没有进程级 motor lease；
- `FINISHED` 不会保证停车。

**目标不是继续扩展状态，而是把状态拆成三个正交维度：`MissionState`、`ControlOwner`、`SafetyState`，再通过唯一仲裁器产生唯一命令。**

---

# Part I — CURRENT：当前真实流程

## 2. 当前端到端控制链

```text
Camera frame
  -> crop / black-line preprocessing / occlusion
  -> scan_line_features()
  -> fit_line_trajectory()                  [普通 visual_fit]
  -> analyze_capture_geometry()             [corner/curve/circle/fork observation]
  -> temporal geometry filter               [decision]
  -> FixedSessionMission.gate()             [当前 session 的物理签名 gate]
  -> RingEntryExecutor or CornerCommandDelay or RaceStateMachine
  -> race_runner inline arbitration
  -> obstacle slow override
  -> bot.set_car_motion(v, w)
```

另有一条完全绕开上述链路的路径：

```text
manual_drive_app HTTP action
  -> StepController
  -> bot.set_car_motion(v, w)
```

核心证据：

- 普通视觉、geometry、mission gate：`apps/race_runner.py:451-509`
- ring/corner/cruise 仲裁：`apps/race_runner.py:513-708`
- obstacle slow 最终覆盖：`apps/race_runner.py:709-714`
- 自动模式最终电机写入：`apps/race_runner.py:715-717`
- 手动模式电机写入：`apps/manual_drive_app.py:193-221`

---

## 3. 当前存在的四层状态

### 3.1 Mission 层

`CourseSession`：

```text
OBSTACLE
CORNER
RING_ENTRY
RING_EXIT
FORK
RETURN
FINISHED
STOPPED
```

固定顺序：

```text
OBSTACLE -> CORNER -> RING_ENTRY -> RING_EXIT -> FORK -> RETURN -> FINISHED
```

证据：`transbot_race/mission.py:12-20,44-52`

当前默认配置却从 `CORNER` 开始：

- Python 默认：`transbot_race/config.py:94-101`
- 部署 JSON：`configs/race_config.json:104-109`

`SESSION_MAP` 当前行为：

| Session | Detector | Executor |
|---|---|---|
| `OBSTACLE` | none | cruise |
| `CORNER` | corner | corner |
| `RING_ENTRY` | ring_entry | ring_entry |
| `RING_EXIT` | ring_entry | ring_entry |
| `FORK` | none | cruise |
| `RETURN` | none | cruise |
| `FINISHED` | none | cruise |
| `STOPPED` | none | cruise |

证据：`transbot_race/mission.py:53-62`

**关键问题**：`FINISHED` 仍映射到 `CRUISE`，`advance()` 只改变 session，不触发停车，也不结束 runner。见 `transbot_race/mission.py:181-189`。

---

### 3.2 Cruise 层

正式状态：

```text
TRACK
LOST
STOPPED
```

TRACK 下的日志 mode：

```text
FOLLOW
PREDICT
PIVOT
PLAN
```

证据：`transbot_race/state_machine.py:11-24`

当前 `PIVOT` 不可达：`_track_step()` 只返回 `FOLLOW` 或 `PREDICT`。见 `transbot_race/state_machine.py:353-388`。

`PLAN` 默认关闭：`configs/race_config.json:150-156`。

---

### 3.3 Corner executor 层

`CornerCommandDelay.state` 实际可能为：

```text
armed
approach
waiting
turning
captured
aligning
exit_tracking
seeking
failed
failed_locked
cooldown
```

主要实现：`transbot_race/path_memory.py:178-783`。

这些不是 telemetry-only phase；其中多个状态真实拥有底盘并输出命令。

---

### 3.4 Ring executor 层

`RingEntryExecutor.state`：

```text
waiting
margin
tracking
inside
exiting
completed
```

证据：`transbot_race/ring_entry.py:76-127,182-298`。

---

## 4. 当前 control ownership

### 4.1 runner 中的实际优先级

当前大致优先级为：

```text
OBSTACLE STOP
  > RING EXECUTOR
  > CORNER EXECUTOR
  > CRUISE TRACK/LOST
  > final OBSTACLE SLOW v clamp
```

但它不是一个显式 arbiter，而是分散在 `apps/race_runner.py:531-714` 的分支和后置替换中。

### 4.2 corner ownership 的正确部分

当 corner active 时，runner 不调用 `sm.step()`，而不是仅覆盖其输出：

- `apps/race_runner.py:617-622`

这是当前应保留的正确不变量：

> 一个 executor 拥有底盘时，非 owner 控制器不应在后台推进会影响未来 handoff 的隐藏状态。

### 4.3 当前 ownership 不完整之处

- owner 没有 enum；只能从 runner 分支和 executor state 推导。
- detector activation 依赖 `corner_margin.state in {"armed", "approach"}`，导致 ring detector 生命周期依赖 corner executor 内部状态。见 `apps/race_runner.py:460-465`。
- pending corner takeover 在 runner 和 `CornerCommandDelay.step()` 中重复计算。见 `apps/race_runner.py:536-543` 与 `transbot_race/path_memory.py:267-291`。
- obstacle stop 暂停命令，但没有统一定义 corner/ring executor 应 `hold`、`reset`、`resume` 还是 `abort`。
- obstacle slow 在所有仲裁结束后只限制 `v`，不限制 `w`，并覆盖原始 reason。见 `apps/race_runner.py:709-714`。

---

# Part II — CURRENT：精确 Transition 契约

## 5. RaceStateMachine transitions

### 5.1 首次获取

#### `TRACK(unacquired) -> TRACK(acquired)`

Guard：

```text
fit.found
AND fit.conf >= conf_predict
AND fit.n_bands >= 3
AND NOT fit.disconnected
```

Action：

- `_seed_filter(fit)`；
- `ever_acquired = True`；
- 开始普通 TRACK 输出。

否则：

```text
v=0, w=0, reason=await_first_line
```

证据：`transbot_race/state_machine.py:216-225,247-253`。

### 5.2 TRACK 内 FOLLOW/PREDICT

Pose 可更新条件比 trackable 更宽：

```text
fit.found
AND fit.conf >= 0.60 * conf_predict
AND fit.n_bands >= 2
AND NOT fit.disconnected
```

证据：`transbot_race/state_machine.py:255-291`。

输出 mode：

```text
f_conf < conf_predict  -> PREDICT
otherwise              -> FOLLOW
```

Steering：

```text
w_desired = sign * k_e * filtered_e0
w = slew_limit(w_desired)
v = v_max * max(v_min_ratio, 1 - slowdown(|w|))
PREDICT additionally multiplies predict_speed_factor
```

证据：`transbot_race/state_machine.py:353-388`。

### 5.3 `TRACK -> LOST`

Guard：

```text
no active PLAN command
AND filtered_conf <= conf_lost
```

Action：

```text
_enter(LOST, event="line_lost")
```

本 tick 随即执行 LOST search。

证据：`transbot_race/state_machine.py:235-244`。

### 5.4 `LOST -> TRACK`

Guard：

```text
filtered_conf > conf_predict
```

Action：

```text
_enter(TRACK, event="reacquired")
_track_step()
```

证据：`transbot_race/state_machine.py:390-396`。

### 5.5 `LOST -> STOPPED`

Guard：

```text
now - state_started_at >= search_timeout_sec
```

Action：

```text
_enter(STOPPED, event="search_timeout")
v=0, w=0
```

证据：`transbot_race/state_machine.py:397-399`。

### 5.6 `LOST -> LOST`

Guard：上述恢复和 timeout 条件均不满足。

Action：

```text
v = 0
w = direction(last filtered side) * max(abs(f_e0)*w_search, w_search_min)
```

证据：`transbot_race/state_machine.py:400-404`。

### 5.7 `ANY -> STOPPED(obstacle)`

Guard：

```text
obstacle=True
```

Action：

```text
_enter(STOPPED, event="obstacle")
v=0, w=0
```

证据：`transbot_race/state_machine.py:176-180`。

### 5.8 `STOPPED(search_timeout) -> TRACK`

当前不是显式 stop subtype，而是依赖：

```text
state == STOPPED
AND last_event == "search_timeout"
AND current fit is trackable
AND observations are continuous:
    abs(delta e0) <= 0.32
    abs(delta theta) <= 0.45
AND consecutive frames >= 3
```

Action：`reacquire_from()`。

证据：`transbot_race/state_machine.py:181-214`。

### 5.9 `STOPPED(obstacle) -> STOPPED`

因为 `last_event != "search_timeout"`，任何视觉 fit 都不会自动恢复。

**审计结论**：一个 enum 状态承载了两种恢复策略，实际 subtype 隐藏在字符串 `last_event` 中。

### 5.10 PLAN

PLAN latch：

```text
preview_plan_enabled
AND preview_dir != 0
AND preview_conf >= preview_conf_min
```

PLAN activate：

```text
latched plan exists
AND current fit weak or blind
```

阶段：

```text
preview_forward_sec: v>0, w=0
preview_turn_sec:    v>0, w=fixed signed preview_turn_w
then clear plan
```

证据：`transbot_race/state_machine.py:293-351`。

**当前默认不可达**，因为 `preview_plan_enabled=false`。

---

## 6. Mission transitions 与 gate

### 6.1 Mission advance

唯一通用 transition：

```text
ORDER[index] -> ORDER[index+1]
```

没有 transition guard 内置在 mission；调用者决定何时 `advance()`。

证据：`transbot_race/mission.py:181-189`。

### 6.2 CORNER gate

接受条件：

```text
decision.kind == "corner"
AND NOT decision.is_fork
AND observation.kind in {"corner", "curve"}  [若 observation 存在]
AND abs(incoming_e) <= 0.65
AND abs(incoming_theta) <= 0.45
```

证据：`transbot_race/mission.py:110-135`。

### 6.3 RING_ENTRY gate

共同条件：

```text
decision.kind == "ring_entry"
AND observation.kind in {"curve", "circle"}  [若 observation 存在]
AND 0.35 <= vertex_y_frac <= 0.76
AND incoming_ready 连续 >= 3 帧
AND fit.found / bands>=3 / conf>=0.30         [若 fit 存在]
```

fork entry：接受后强制 direction 使用 `mission.ring_entry_direction`。

single-path tangent entry：

```text
observation.endpoints == 2
AND 25deg <= abs(angle) <= 80deg
```

证据：`transbot_race/mission.py:102-165`。

### 6.4 RING_EXIT gate

```text
decision.kind == "ring_entry"
AND observation.is_fork
AND 0.30 <= vertex_y_frac <= 0.82
```

接受后 direction 强制使用 `mission.ring_exit_direction`。

证据：`transbot_race/mission.py:166-177`。

### 6.5 Mission 的结构问题

- `advance()` 没有 typed completion event，只由 runner 的字符串/布尔结果触发。
- `FORK`、`RETURN`、`FINISHED` 暂无专用 detector/executor。
- `STOPPED` 在 enum/SESSION_MAP 中，但不在 ORDER 中。
- obstacle 同时被建模为 mission session 和并行 safety monitor，职责重叠。

---

## 7. Geometry filter transitions

默认 `confirm_frames=3`，窗口长度至少 3 后才能 decision。

但 turn/session decision 实际要求：

```text
required = max(2, confirm_frames - 1)
```

默认即 3 帧窗口内 2 票即可。

Vote 必须满足：

```text
same direction
same fork flag
confidence >= 0.55
abs(angle) >= 20deg
vertex_y_frac exists
shape belongs to allowed set
```

证据：`transbot_race/capture_geometry.py:441-513`。

**问题**：这是“滑动窗口内任意有效票”，不是“连续 N 帧确认”。旧票可以跨过中间无效帧参与当前 decision，事件边界不明确。

---

## 8. CornerCommandDelay transitions

### 8.1 `armed -> approach`

Guard：

```text
capture geometry enabled
AND state == armed
AND stable_v > 0.01
AND decision exists
AND decision.kind in {"turn", "corner"}
AND direction != 0
AND abs(incoming_e) <= 0.65
AND abs(incoming_theta) <= 0.45
AND, if fork:
    abs(incoming_e) <= corner_reacquire_max_e
    abs(incoming_theta) <= corner_reacquire_max_theta
```

Action：

- 锁定 `candidate_dir`；
- 字符串 `event_shape` 设为 fork/corner/curve；
- 冻结 incoming `stable_v/stable_w`；
- 重置 gate/yaw/reacquire/latch 计数器。

证据：`transbot_race/path_memory.py:215-319`。

### 8.2 `armed -> waiting`（legacy fallback）

仅 `capture_geometry_enabled=false`：

```text
same visual direction confirmed >= max(3, corner_confirm_frames)
AND stable_v > 0.01
```

证据：`transbot_race/path_memory.py:143-171,320-335`。

### 8.3 `approach -> waiting`

持续有效 geometry：

```text
same direction
confidence >= 0.55
shape matches locked event_shape
fork additionally remains incoming-aligned
vertex_y_frac >= corner_gate_y_frac
for gate_frames >= corner_gate_confirm_frames
```

若是普通 corner 或 fork margin enabled，则进入 waiting，并保留 camera-to-axle distance。

证据：`transbot_race/path_memory.py:336-408`。

### 8.4 `approach -> turning`

仅：

```text
event_shape == "fork"
AND roundabout_margin_enabled == false
AND gate confirmed
```

证据：`transbot_race/path_memory.py:396-407`。

### 8.5 `approach -> armed`

任一条件：

```text
approach_travelled_m > max(0.25, camera_to_axle_m)
OR approach_missing_frames > corner_approach_missing_frames
```

Action：`_reset_armed()`，清空事件 epoch。

证据：`transbot_race/path_memory.py:386-409,742-765`。

### 8.6 `waiting -> turning`

```text
remaining_m <= 1e-6
```

`remaining_m` 按正向 travelled distance 消耗。

证据：`transbot_race/path_memory.py:410-415`。

### 8.7 `turning -> failed`

优先 guard：

```text
turned_rad >= turn_limit
```

其中：

```text
fork: target_angle_rad + corner_search_extra_rad
other: corner_max_turn_angle_rad
```

证据：`transbot_race/path_memory.py:415-424,767-778`。

### 8.8 `turning -> exit_tracking`（非 fork）

```text
turned_rad >= corner_reacquire_angle_rad
AND first exit line latched
AND latched exit is trackable
```

证据：`transbot_race/path_memory.py:424-438`。

### 8.9 `turning -> captured`（fork）

```text
turned_rad >= corner_reacquire_angle_rad
AND FirstEntryLineLatch.try_latch() succeeds
```

证据：`transbot_race/path_memory.py:438-445`。

### 8.10 `turning -> seeking`（fork）

```text
minimum yaw reached
AND no exit latch candidate pending
AND turned_rad >= target_angle_rad
```

证据：`transbot_race/path_memory.py:446-452`。

### 8.11 `captured -> aligning`

```text
turn limit not reached
AND latched exit remains complete/connected/continuous
```

证据：`transbot_race/path_memory.py:453-468`。

### 8.12 `captured -> seeking`

```text
align_missing_frames > corner_align_missing_frames
```

保留累计 yaw budget，重置 provisional exit latch。

证据：`transbot_race/path_memory.py:468-487`。

### 8.13 `aligning -> cooldown`

```text
complete line
AND moving_follow_handoff_ready(fit)
for corner_cruise_ready_frames
AND align_travelled_m >= corner_cruise_ready_distance_m
```

Action：`visual_takeover=True`，runner 随后调用 `sm.reacquire_from()`。

证据：`transbot_race/path_memory.py:488-510`。

### 8.14 `exit_tracking -> cooldown`

```text
latched exit remains trackable
AND cruise-ready consecutive frames >= corner_cruise_ready_frames
```

证据：`transbot_race/path_memory.py:511-528`。

### 8.15 `exit_tracking -> failed_locked`

```text
no exit seen
OR now - exit_last_seen_now > corner_exit_predict_sec
```

证据：`transbot_race/path_memory.py:529-536`。

### 8.16 `seeking -> failed`

先累计 yaw，再优先检查 turn limit；一旦到达 limit，不再接受本帧候选。

证据：`transbot_race/path_memory.py:537-547`。

### 8.17 `seeking -> captured`

```text
turn limit not reached
AND first exit latch succeeds
```

证据：同上。

### 8.18 `failed -> cooldown`

```text
safe cruise fit
AND no branches
AND e/theta continuity
for corner_reacquire_confirm_frames
```

Action：visual takeover。

证据：`transbot_race/path_memory.py:548-576`。

### 8.19 `failed_locked`

当前没有内部恢复 transition。它持续输出 `v=0,w=0`，除非外部重建/reset executor。

证据：`transbot_race/path_memory.py:699-714,777-783`。

### 8.20 `cooldown -> armed`

- curve/fork：稳定直线连续 3 帧；
- 普通 corner：exit cleared 连续 2 帧。

证据：`transbot_race/path_memory.py:577-590`。

---

## 9. FirstEntryLineLatch transitions

### Strong candidate

```text
found
AND conf >= 0.30
AND n_bands >= 3
AND abs(theta) <= 0.95
AND direction * e0 > 0
```

当前 strong candidate **单帧即可 latch**。

### Weak candidate

```text
found
AND conf >= 0.15
AND n_bands >= 1
AND abs(e0) >= 0.45
AND direction * e0 > 0
AND abs(delta e0) <= 0.50
for confirm_frames
```

证据：`transbot_race/path_memory.py:42-101`。

### Latch 后连续性

```text
found
AND conf >= handoff min
AND n_bands >= 3
AND connected
AND abs(e0) <= 0.95
AND abs(theta) <= 1.20
AND abs(delta e0) <= 0.32
AND abs(delta theta) <= 0.45
```

证据：`transbot_race/path_memory.py:103-123`。

**问题**：strong 与 weak 的 temporal contract 不一致；“强”被用来绕过时间确认，而不是仅提高观测质量。

---

## 10. RingEntryExecutor transitions

### 10.1 `waiting -> margin`

```text
route_fit exists
AND accepted_entry == true
```

Action：

- 冻结 incoming `v,w`；
- `margin_remaining_m = margin_distance_m`；
- 保存 selected route fit。

证据：`transbot_race/ring_entry.py:208-218`。

### 10.2 `margin -> tracking`

```text
margin_remaining_m <= 1e-6
```

证据：`transbot_race/ring_entry.py:219-230`。

### 10.3 `tracking -> inside`

```text
travelled_m >= min_distance_m
AND coherent_arc consecutive frames >= clear_frames_required
```

其中 `coherent_arc` 只要求 `route_fit` 和 observation 都存在，不要求 fork label 消失。

Action：返回：

```text
completed=True
reason="ring_entry_path_established"
```

证据：`transbot_race/ring_entry.py:239-262`。

这表示 **entry phase 完成**，不是整个 ring executor 完成。

### 10.4 `inside -> exiting`

```text
accepted_entry == true
AND inside_travelled_m >= inside_arm_distance_m
```

Action：返回 `started=True, completed=False`。

证据：`transbot_race/ring_entry.py:263-276`。

### 10.5 `exiting -> completed`

```text
moving_follow_handoff_ready(cruise_fit)
AND observation exists
AND NOT observation.is_fork
for clear_frames_required
AND travelled_m >= exit_distance_m
```

Action：返回：

```text
completed=True
reason="ring_exit_cruise_established"
```

证据：`transbot_race/ring_entry.py:277-290`。

这一次 `completed=True` 表示 **整个 ring executor 可完成并 handoff**。

### 10.6 Missing route bridge

```text
route_fit missing_frames <= 5 -> use last_fit
missing_frames > 5            -> result.fit = None
```

证据：`transbot_race/ring_entry.py:291-298`。

runner 对 committed route loss 的处理：

```text
v=0,w=0, MotionCommand.state=STOPPED
```

但没有修改 `RaceStateMachine.state`。

证据：`apps/race_runner.py:576-586`。

**这是当前最明确的全局状态不一致。**

### 10.7 Ring control 参数问题

`control()` 接收 `approach_max_w`，但实际只使用 `max_w`：

- `transbot_race/ring_entry.py:300-329`

因此调用者认为存在 approach 限幅，实际没有生效。

---

# Part III — 主要问题与严重度

## 11. P0：安全与控制权

### P0-1：两个独立 motor writer，没有进程级 lease

事实：

- auto runner：`apps/race_runner.py:715-717`
- manual app：`apps/manual_drive_app.py:193-221`

进程内 `threading.Lock` 不能防止两个进程同时写底盘。

风险：

- manual emergency stop 后，auto 下一 tick 可重新发运动；
- auto stop 后，manual worker 可再次发旧动作；
- 无法从 telemetry 确认当前真正 owner。

目标：所有 `set_car_motion()` 必须经单一 `MotorGateway` 和进程级 lease。

### P0-2：没有统一 stop latch

当前停止至少有：

1. obstacle：真正修改 `RaceStateMachine.state=STOPPED`；
2. search timeout：同一 state，但可自动恢复；
3. ring route loss：只构造 `MotionCommand(state=STOPPED)`；
4. corner failed/failed_locked：executor 输出零，但全局 state 未必 STOPPED；
5. manual stop：独立路径重复发零；
6. process shutdown：finally stop；
7. mission FINISHED：目前不会自动 stop。

风险：系统无法统一回答：

- 当前是否被锁停？
- 谁要求停？
- 是否允许自动恢复？
- 谁有权限 clear？
- executor 应继续、保持还是 abort？

目标：引入 typed `SafetyState` + `StopCause` + recovery policy。

---

## 12. P1：状态一致性

### P1-1：`MotionCommand.state` 不等于持久状态

ring route loss 时 command 标为 STOPPED，而 `sm.state` 可仍是 TRACK。

影响：telemetry、下一 tick 和恢复逻辑可互相矛盾。

### P1-2：`STOPPED` 恢复策略依赖 `last_event` 字符串

`search_timeout` 可恢复，`obstacle` 不可恢复，但二者共用 enum。

影响：字符串改名、错误覆盖或新 stop reason 都可能改变安全行为。

### P1-3：ring `completed` 双重语义

同一个 bool 同时表示：

- entry phase complete；
- whole executor complete。

runner 必须结合当前 mission session 才能解释。

### P1-4：`FINISHED` 无停车 transition

mission 到 FINISHED 后仍映射到 cruise，runner 可继续行驶。

### P1-5：obstacle interruption 没有 executor contract

active corner/ring 遇到 obstacle 时，没有统一的：

```text
hold / resume / abort / reset
```

解除后，多个状态机可能在不同历史位置继续。

### P1-6：控制协议依赖字符串 reason

runner 使用：

```text
memory_status.reason == "corner_visual_takeover"
```

触发 handoff 和 mission advance。见 `apps/race_runner.py:671-690`。

文案字段承担了 typed transition event 的职责。

---

## 13. P2：Transition vague 与检测漂移

### P2-1：confidence 没有统一质量等级

当前阈值至少包括：

| 阈值 | 当前语义 |
|---:|---|
| `0.15` | weak first exit candidate |
| `0.60 * conf_predict`（默认 `0.21`） | pose filter usable |
| `0.30` | strong exit candidate / ring gate |
| `0.35` | normal trackable / moving handoff min |
| `0.55` | geometry vote / clear |
| `0.65` | stable command / recovery |
| `0.80` | legacy corner observation |

这些语义分散在多个模块，无法通过一个 quality enum 解释。

### P2-2：direction authority 不明确

方向可能来自：

- skeleton endpoint displacement；
- branch side；
- preview direction；
- theta sign；
- mission configured route；
- `invert_turn` 后的 chassis sign。

目标应明确：

```text
observed_direction
route_intent
locked_executor_direction
chassis_turn_sign
```

一旦 executor 接受 takeover，视觉方向不得再替换 locked intent。

### P2-3：shape 字符串跨层承担控制语义

`corner/curve/circle/fork` 同时用于：

- 原始观测；
- temporal filter；
- mission gate；
- executor `event_shape`；
- turn limit；
- cooldown clear 策略；
- telemetry。

raw shape 不应直接等价于 ownership event。

### P2-4：geometry 投票不是连续确认

3 帧窗口默认只需 2 票，旧票可跨无效帧参与新 decision。

目标：明确采用以下二选一，不能继续 vague：

1. 连续 N 帧；或
2. 长度 N 的窗口允许最多 D 个 dropout，并显式记录 dropout policy。

### P2-5：strong exit 单帧 latch

strong candidate 一帧即可锁定，weak 才要求多帧，存在单帧误锁风险。

### P2-6：ring detector 激活依赖 corner state

mission 已进入 ring session 时，geometry 是否运行仍取决于 `corner_margin.state`。

目标：detector lifecycle 由 mission/observation pipeline 管理，不依赖另一个 executor 的内部 phase。

---

## 14. P2：配置来源冲突

### 14.1 Python default 与 JSON default 不一致

| 字段 | Python | JSON | 影响 |
|---|---:|---:|---|
| `camera.crop` | `(300,265,430,455)` | `[300,265,430,442]` | ROI 与 geometry 变化 |
| `path_memory.mode` | `none` | `fixed_sessions` | 构造路径不同 |
| `camera_to_axle_m` | `0.10` | `0.45` | margin 差异巨大 |
| `obstacle.enabled` | `True` | `False` | safety 默认行为相反 |

证据：

- `transbot_race/config.py:6-14,31-78,103-117`
- `configs/race_config.json:2-13,47-84,110-121`

### 14.2 Web bool coercion 错误

```python
bool("false") is True
```

当前 `_deep_update_cfg()` 对 bool 使用 `bool(value)`：`apps/race_debug_app.py:343-366`。

### 14.3 Web save 不复用 runner validation

`/api/defaults/save` 直接 update 和 write：`apps/race_debug_app.py:835-842`。

它没有复用 runner 的完整 `_validate_config()`。

### 14.4 缺失的关键 cross-field validation

至少应验证：

```text
0 <= conf_lost < conf_predict <= 1
corner_reacquire_angle_rad <= corner_turn_angle_rad <= corner_max_turn_angle_rad
0 <= all speed ratios <= 1
all frame counters >= 1 unless zero has explicit semantics
all time/distance values finite and >= 0
corner replay/align/approach limits compatible with tracker/chassis limits
preview hold window compatible with forward + turn duration
mission initial state belongs to enabled course plan
UI bounds contain the deployment profile value
```

---

## 15. P3：文档和可维护性

- README 的 unified tracker / pivot 描述与当前 dedicated corner/ring executor 不符。
- `TrackMode.PIVOT` 和相关参数存在但普通 cruise 不使用。
- `FORK`、`RETURN`、`FINISHED` 是名义状态，尚无完整执行器。
- telemetry `reason` 会被 obstacle slow 后处理覆盖，原 owner reason 丢失。
- `RingEntryExecutor.control()` 的 `approach_max_w` 参数未使用。
- `CourseSession.OBSTACLE` 与并行 `ObstacleMonitor` 职责重叠。

---

# Part IV — TARGET：建议目标模型（尚未实现）

## 16. 三个正交状态维度

### 16.1 MissionState

Mission 只回答“现在在完成哪一个课程阶段”，不回答谁控制电机、是否安全停车。

建议逐步演化为：

```text
START
OBSTACLE_OUTBOUND
CORNER_OUTBOUND
GAP_OUTBOUND
RING_ENTRY_OUTBOUND
RING_CIRCULATE_OUTBOUND
RING_EXIT_OUTBOUND
COLOR_CLASSIFY
FORK_OUTBOUND
UNLOAD_STOP
RETURN_ORIENT
FORK_RETURN
RING_ENTRY_RETURN
RING_CIRCULATE_RETURN
RING_EXIT_RETURN
GAP_RETURN
CORNER_RETURN
OBSTACLE_RETURN
HOME_STOP
FINISHED
```

当前尚未实现的课程功能可以暂不加入第一轮代码，但状态模型必须允许明确扩展，不能继续把它们塞进 corner executor。

Mission transition 只能由 typed `MissionEvent` 触发，例如：

```text
PHASE_COMPLETED
MISSION_COMPLETED
MISSION_RESET
```

### 16.2 ControlOwner

```text
NONE
CRUISE
CORNER_EXECUTOR
RING_EXECUTOR
FORK_EXECUTOR       [future]
TERMINAL_EXECUTOR   [future]
MANUAL
```

不变量：

1. 每 tick 恰好一个 owner 或 NONE；
2. detector 只能发 takeover request，不能直接发 motor command；
3. takeover 原子化：validate -> seed/freeze -> accept owner -> produce command；
4. handoff 原子化：handoff ready -> seed next owner -> switch owner；
5. 非 owner executor 不推进会影响未来控制的隐藏状态；
6. manual 必须先获得进程 lease。

### 16.3 SafetyState

```text
CLEAR
SLOW
STOP_LATCHED
```

`STOP_LATCHED` 必须携带 typed `StopCause`：

```text
OBSTACLE
SEARCH_TIMEOUT
ROUTE_LOST
EXECUTOR_FAILED
MISSION_FINISHED
OPERATOR_STOP
PROCESS_LEASE_LOST
CAMERA_FAILURE
INVALID_COMMAND
```

每个 cause 必须定义：

| Stop cause | 默认可自动恢复 | Clear authority | Executor disposition |
|---|---|---|---|
| `SEARCH_TIMEOUT` | 是 | reliable line policy | hold/reset cruise search |
| `OBSTACLE` | 按策略 | obstacle clear debounce 或 operator | hold active executor |
| `ROUTE_LOST` | 默认否 | executor recovery 或 operator | hold/abort owner |
| `EXECUTOR_FAILED` | 受限 | validated cruise recovery 或 operator | abort executor |
| `MISSION_FINISHED` | 否 | new mission/reset | complete all executors |
| `OPERATOR_STOP` | 否 | operator reset | abort/hold by policy |
| `PROCESS_LEASE_LOST` | 否 | new lease + reset | abort owner |
| `CAMERA_FAILURE` | 按策略 | camera stable + reset | hold owner |
| `INVALID_COMMAND` | 否 | diagnosis + reset | abort owner |

Safety 只覆盖最终命令，不伪造 owner 或 mission：

```text
STOP_LATCHED -> final v=0,w=0
SLOW         -> clamp according to owner-aware policy
CLEAR        -> unchanged
```

---

## 17. 唯一 command pipeline

目标每 tick 严格按以下顺序：

```text
1. Collect observations
2. Classify observation quality using shared typed levels
3. Update MissionState from typed mission events
4. Update only the active ControlOwner executor
5. Evaluate takeover/handoff requests
6. Resolve exactly one ControlOwner
7. Produce exactly one candidate command
8. Apply SafetyState
9. Validate command: finite, bounded, owner epoch valid
10. Single MotorGateway.write(command)
11. Emit one immutable telemetry snapshot
```

Telemetry 必须同时报告：

```text
mission_state
control_owner
executor_phase
safety_state
stop_cause
observation_quality
transition_event
candidate_command
final_command
safety_override
owner_epoch
```

`MotionCommand` 不应继续用 `RaceState` 假装表达全局系统状态。

---

## 18. 建议 typed events

### 18.1 ObservationQuality

```text
INVALID
WEAK
POSE_USABLE
TRACKABLE
HANDOFF_READY
```

每个等级的 guard 必须集中定义，所有模块复用。

### 18.2 ObservationEvent

```text
LINE_TRACKABLE
LINE_WEAK
LINE_LOST
CORNER_CANDIDATE
RING_ENTRY_CANDIDATE
RING_EXIT_CANDIDATE
FORK_CANDIDATE
OBSTACLE_APPROACH
OBSTACLE_STOP
```

### 18.3 OwnerEvent

```text
TAKEOVER_REQUESTED
TAKEOVER_ACCEPTED
HANDOFF_READY
EXECUTOR_HELD
EXECUTOR_RESUMED
EXECUTOR_ABORTED
EXECUTOR_COMPLETED
ROUTE_LOST
```

### 18.4 Ring phase events

用明确事件替代 `completed: bool`：

```text
RING_ENTRY_ESTABLISHED
RING_EXIT_SELECTED
RING_EXIT_HANDOFF_READY
RING_EXECUTOR_COMPLETED
```

### 18.5 SafetyEvent

```text
SLOW_REQUESTED
STOP_REQUESTED
STOP_LATCHED
STOP_CLEAR_REQUESTED
STOP_CLEARED
```

### 18.6 Direction

```text
NONE
LEFT
RIGHT
STRAIGHT
```

只在一个边界函数中转换：

```text
image direction -> route intent -> chassis w sign
```

禁止业务逻辑到处使用 `-1/+1` 和 `invert_turn` 重新解释。

---

# Part V — 实施顺序

## 19. Phase 0：冻结事实和补 transition matrix

**本阶段不改变控制行为。**

任务：

1. 为 CURRENT 四层状态建立合法 predecessor/successor 表；
2. 增加 frame-level ownership test；
3. 记录每次 transition：from、to、event、guard result、counter、owner；
4. 增加当前行为的 characterization tests；
5. 明确哪些当前行为是必须保留，哪些是已知 bug。

必须新增的测试：

- ring route loss 时 command state 与 `sm.state` 不一致的 characterization；
- obstacle 在 corner/ring active 中触发；
- obstacle clear 后 executor 当前行为；
- mission FINISHED 仍继续 cruise 的 characterization；
- geometry 旧票跨 dropout；
- strong single-frame exit latch；
- `failed_locked` 无内部恢复；
- obstacle slow 覆盖 owner reason；
- manual 与 auto 两个 writer 的结构检查。

验收：

- 所有当前 transition 都能从一张 machine-readable matrix 找到；
- 不允许测试名称继续使用具体 run ID 作为唯一语义；
- 运行代码行为未改变。

---

## 20. Phase 1：增加类型和 telemetry，仍不改变仲裁

任务：

1. 引入 `MissionState`/复用现有 session，但明确职责；
2. 新增 `ControlOwner`；
3. 新增 `SafetyState`、`StopCause`；
4. 新增 typed transition events；
5. runner 同时输出新旧 telemetry；
6. 用 assertion 比较“推导 owner”与旧分支实际 winner。

验收：

- 每帧 telemetry 有唯一 owner；
- 每个 zero command 有明确 safety cause 或 executor phase；
- `reason` 仅用于人类说明，不再是新逻辑 guard；
- 旧控制输出逐帧保持等价。

---

## 21. Phase 2：提取唯一 Arbiter

任务：

1. 把 `apps/race_runner.py:531-714` 的内联仲裁提取成单一组件；
2. 输入：owner state、executor result、cruise candidate、safety request；
3. 输出：winner owner、candidate command、final command、override record；
4. corner active 时保持“不推进 cruise hidden state”的不变量；
5. pending takeover 只计算一次，形成 immutable decision。

验收：

- 每 tick 只能产生一个 winner；
- 不再通过后置 `replace()` 隐式改变 owner；
- obstacle slow 不覆盖原始 reason，而写入独立 `safety_override`；
- 与 Phase 1 记录的旧输出进行 replay 等价比较。

---

## 22. Phase 3：统一 StopLatch

任务：

1. obstacle、search timeout、route loss、executor failure、finish、operator stop 全部进入 `SafetyState`；
2. 删除“只在 command 上标 STOPPED”的伪状态；
3. 将 search timeout recovery 从 `last_event` 字符串迁移到 `StopCause` policy；
4. 为 active executor 定义 `hold/resume/abort`；
5. `MISSION_FINISHED` 必须 latch stop；
6. camera failure 和 invalid command 进入明确 stop cause。

验收：

- `command.state` 不再作为全局 safety 真相；
- 每个 stop 都能回答 cause、recoverable、clear authority、owner disposition；
- FINISHED 永远输出并保持 `v=0,w=0`，直到 reset；
- route loss 不再出现 `sm.state=TRACK` / command STOPPED 的矛盾。

---

## 23. Phase 4：规范 executor completion

任务：

1. ring result 用 typed phase event 替代双义 `completed`；
2. mission 只消费 typed mission event；
3. corner handoff 不再依赖 `reason="corner_visual_takeover"`；
4. raw shape 与 navigation event 分离；
5. `event_shape` 替换为 typed locked event identity；
6. detector lifecycle 从 corner executor state 解耦。

验收：

- entry phase complete 与 executor complete 不可能混淆；
- 修改 telemetry 文案不会改变 transition；
- mission advance 的每个调用点都有唯一 typed event；
- locked route direction 不会被后续 raw observation 覆盖。

---

## 24. Phase 5：MotorGateway 与进程 lease

任务：

1. 所有 auto/manual motor write 经过同一个 gateway；
2. gateway 验证 owner lease 和 owner epoch；
3. manual start 必须先获得 lease；
4. auto start 在 lease 冲突时拒绝运行；
5. lease 丢失立即 `STOP_LATCHED(PROCESS_LEASE_LOST)`；
6. shutdown 必须先停止 worker，再关闭 recorder/resources。

同时修复 manual lifecycle：

- `App.close()` 当前没有等待 action worker 完成，见 `apps/manual_drive_app.py:180-240`；
- stop/action start 需要 generation token，防止旧 worker 在 stop 后再次发运动。

验收：

- 任意时刻只有一个进程拥有 motor lease；
- lease 冲突绝不发非零命令；
- manual emergency stop 后旧 worker 无法恢复运动；
- shutdown 日志完整且 recorder 不会被仍运行的 worker 写入。

---

## 25. Phase 6：统一配置 schema、coercion、validation

任务：

1. runner、debug app、Web save 使用同一配置加载和 validation；
2. 明确 Python default 是 schema fallback，JSON 是 deployment profile；
3. telemetry 显示 effective value 和 source；
4. 修复 string bool coercion；
5. 增加 finite/range/cross-field validation；
6. 修复 UI bounds 与部署值冲突；
7. 将 ring executor 的关键 transition 参数显式放入 config；
8. 删除未使用参数或使其真正生效，例如 `approach_max_w`。

验收：

- 同一配置经 CLI、Web、直接加载得到相同 effective config；
- Web 不可保存 runner 会拒绝的配置；
- `"false"` 不会变成 `True`；
- 所有 transition 参数有单位、合法范围和 consumer。

---

## 26. Phase 7：清理旧语义和文档

最后才做：

- 删除或正式实现不可达 `PIVOT`；
- 更新 README 的真实架构；
- 删除旧字符串 guard；
- 清理重复 direction source；
- 完善 `FORK`、`RETURN`、terminal executor；
- 将本报告标记 implementation status，而不是删除历史审计事实。

---

# Part VI — 不可破坏的不变量

## 27. 当前已正确、必须保留

1. 首次完整 line 前不运动、不搜索。
2. corner/ring takeover 前，raw candidate 只观察，不拥有电机。
3. corner active 时不推进 cruise hidden state。
4. session-to-cruise handoff 使用完整、connected、稳定 corridor。
5. turn yaw limit 是硬安全边界，达到 limit 后不能再接受新出口身份。
6. committed ring route 短缺帧可保持，但不能静默切回可能选错 branch 的 generic tracker。
7. handoff 时必须原子 seed 新 owner，不能混入旧 filter/search history。
8. obstacle stop 优先于普通执行器命令。
9. 所有 shutdown path 最终必须发零命令。

## 28. 新架构必须新增的不变量

1. 每 tick 一个 owner、一个 candidate、一个 final command。
2. 每个 transition 有唯一 typed event。
3. 每个 stop 有 typed cause 和 recovery policy。
4. Safety override 不改变 mission 或伪造 owner。
5. 非 owner 不推进控制历史。
6. 所有 motor writes 需要有效 lease。
7. `FINISHED` 必须锁停。
8. `reason` 变化不能改变控制行为。
9. raw geometry shape 不能直接成为跨层 owner event。
10. config 的任何入口必须得到同一 effective semantics。

---

# Part VII — 实现 Agent 的交付要求

## 29. 每个 Phase 必须提交的内容

1. 修改前的 transition 表；
2. 修改后的 transition 表；
3. 新增/删除的 state 与 event；
4. 每条 transition 的 guard、counter、action、owner、safety effect；
5. 对应测试；
6. replay 或逐帧等价证据；
7. 剩余已知不一致；
8. 明确说明哪些 TARGET 尚未实现。

## 30. 禁止的做法

- 禁止把新的布尔字段继续堆进 runner 分支，却不定义 owner/safety 语义；
- 禁止新增 `reason == "..."` 控制判断；
- 禁止新增同义 STOPPED enum 而不统一 stop latch；
- 禁止用 sleep/timer 替代可观测的成功条件，timer 只能是安全上限；
- 禁止因一个 run 失败就放宽所有 confidence/angle gate；
- 禁止让 geometry detector 直接写 motor command；
- 禁止在 obstacle clear 后默认继续旧 executor，除非 resume policy 明确且有测试；
- 禁止同时修改所有 executor 后只依赖集成测试排错；
- 禁止清理用户当前未提交工作。

## 31. 推荐测试结构

### Unit transition tests

每个状态逐一验证：

```text
allowed predecessors
allowed successors
guard just below boundary
guard exactly at boundary
guard just above boundary
counter reset conditions
entry side effects
exit side effects
motor ownership
safety effect
```

### Arbiter matrix

至少覆盖：

```text
owner x safety x executor result x mission phase
```

关键组合：

- cruise + obstacle stop；
- corner active + obstacle stop/clear；
- ring active + route loss + obstacle；
- handoff ready + same-frame stop；
- mission finished + stale executor command；
- manual lease request + auto owner；
- owner epoch stale command。

### Replay tests

输入固定逐帧 observation/motion sequence，断言：

```text
mission_state
control_owner
executor_phase
safety_state
transition_event
final v/w
```

不要只断言最终 reason 字符串。

### Config tests

所有入口：

```text
Python direct
JSON load
Web update
saved reload
CLI runner
```

必须产生相同 effective config 或明确 source override。

---

# Part VIII — 优先修复清单

## 32. 建议执行顺序

### 第一优先：先把“真相”建起来

1. `ControlOwner` telemetry；
2. `SafetyState/StopCause` telemetry；
3. transition event enum；
4. characterization matrix；
5. 不改变输出的唯一 arbiter。

### 第二优先：消除安全不一致

1. ring route-loss pseudo STOP；
2. mission FINISHED 不停车；
3. obstacle interruption contract；
4. manual/auto motor lease；
5. manual stop generation race。

### 第三优先：消除 vague transition

1. ring completed 双义；
2. reason 字符串 handoff；
3. geometry 连续确认；
4. confidence quality levels；
5. direction authority；
6. shape 与 event 分离。

### 第四优先：统一配置和清理残留

1. config source/validation；
2. bool coercion；
3. PIVOT 残留；
4. README/UI；
5. 未使用参数。

---

## 33. 最终目标判断标准

重构完成后，任何一帧都必须能用一条记录回答：

```text
当前任务是什么？
谁拥有底盘？
该 owner 处于哪个 phase？
本帧发生了什么 typed event？
为什么发生 transition？
用了哪个 counter/window？
是否有 safety override？
如果停车，cause 是什么、谁能 clear？
候选命令是什么？
最终写给电机的命令是什么？
```

如果仍需要通过 `reason` 文案、多个对象的隐式 state、上一个 run 的特殊历史或 runner 中的分支顺序才能回答，说明 transition 架构仍未完成。

---

## 34. 审计证据入口

后续 Agent 至少应重新读取以下当前文件，而不是只依赖本报告：

- `apps/race_runner.py`
- `apps/manual_drive_app.py`
- `apps/race_debug_app.py`
- `transbot_race/state_machine.py`
- `transbot_race/path_memory.py`
- `transbot_race/mission.py`
- `transbot_race/ring_entry.py`
- `transbot_race/capture_geometry.py`
- `transbot_race/config.py`
- `configs/race_config.json`
- `tests/test_race_state_machine.py`
- `tests/test_path_memory.py`
- `tests/test_mission.py`
- `tests/test_ring_entry.py`
- `tests/test_capture_geometry.py`
- `tests/test_manual_drive.py`
- `tests/test_debug_config.py`

行号基于 2026-07-14 dirty 工作树，后续修改会漂移。代码事实优先于本报告；若事实变化，Agent 必须更新 transition matrix 和本报告中的 implementation status，而不是继续保留失效结论。
