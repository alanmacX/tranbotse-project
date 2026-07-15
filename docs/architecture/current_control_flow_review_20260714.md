# 当前代码全流程审计与问题报告

> 审计日期：2026-07-14
> 审计对象：`codex/control-transition-refactor` 当前 dirty 工作树
> 审计方式：只读代码审查，并与六个正式 run 的逐帧证据交叉核验
> 重要边界：本文区分“历史 run 中已发生的问题”“当前已修复的问题”“当前仍存在的问题”。历史 run 不能自动代表当前代码仍有同一行为。

## 1. 先说结论

当前系统已经不再是最初完全没有 owner/safety 概念的版本，但仍不是一个真正原子化的统一控制系统。

现在有四套同时存在的状态：

1. `FixedSessionMission`：现在跑到课程哪一段；
2. `RaceStateMachine`：普通巡线、预测、丢线搜索、停止；
3. `CornerCommandDelay` / `RingEntryExecutor`：特殊路段执行到哪一步；
4. `CommandArbiter`：本帧谁拥有底盘、是否被安全层锁停。

它们最终仍由 `apps/race_runner.py` 中一段按固定顺序执行的集成代码拼起来。最大的风险不再只是“没有状态”，而是：

- 同一帧内，mission、executor、owner 的更新不是一个原子事务；
- raw `curve` 仍会被正式转换成可接管底盘的 `corner` 事件；
- ring selected path 可以替自己证明 takeover quality；
- ring route loss 会永久锁停，但 runner 没有 clear 路径；
- obstacle hold 会暂停 executor 的时间更新，恢复时可能一次性吞入过大的运动增量；
- ring telemetry 把“当前图像看见线”和“旧 selected path 仍在预测”混在 `found/conf` 里；
- 当前测试主要验证单个模块，没有验证 runner 的真实执行顺序。

因此，“越改越乱”的本质不是某一个阈值错了，而是**检测事实、任务接受、控制权接管、路径承诺、安全许可仍没有完全分离**。

必须持续遵守：

```text
看见某种形状
≠ mission 接受该事件
≠ executor 获得底盘
≠ 路径已经物理建立
≠ safety 允许继续运动
```

---

## 2. 当前每一帧到底怎么走

当前实际主循环位于 `apps/race_runner.py:459-829`。

### 2.1 相机与普通巡线感知

```text
相机 frame
→ 扩展 crop
→ 黑线阈值与连通域清理
→ 7 个扫描带
→ sliding-window 选择一条路径
→ TrajectoryFit visual_fit
```

代码入口：

- crop / mask：`apps/race_runner.py:465-474`
- 普通轨迹拟合：`apps/race_runner.py:475-482`
- sliding-window 路径：`transbot_race/vision.py:585-654`

`TrajectoryFit` 的几个词不能混淆：

- `found`：算法找到了一个几何候选；
- `has_near_support`：路径到达实用近端 band；
- `control_valid`：这个候选有资格影响控制；
- `path_memory`：这是显式选择的路径记忆，不是普通 scan path。

当前普通 cruise 和 corner 出口判断已使用 `control_valid`。这解决了 174902 一类“屏幕边角反光在远处可见，却被当作近场控制线”的主要穿透路径。

### 2.2 独立 geometry 感知

同时，系统从更大的 floor ROI 中：

```text
threshold mask
→ 选择 chassis 附近连通域
→ 骨架化
→ 找 endpoint 和候选 path
→ 判断 straight / curve / corner / circle / fork
```

代码：`transbot_race/capture_geometry.py:297-438`。

它输出的是 `CaptureGeometryObservation`，只是观察事实。随后 session-specific temporal filter 才产生 `CaptureGeometryDecision`。

### 2.3 Mission gate

runner 根据当前课程阶段选择 detector，并调用：

```python
mission.gate(decision, observation, takeover_fit, ...)
```

代码：`apps/race_runner.py:560-569`。

Mission gate 决定“当前课程阶段是否接受这个事件”，不是直接写电机。

### 2.4 三种控制候选

之后只应有一种候选控制：

- 普通路段：`RaceStateMachine`；
- corner：`CornerCommandDelay`；
- ring：`RingEntryExecutor`。

runner 的实际分支位于 `apps/race_runner.py:610-785`。

### 2.5 Owner 与 Safety

runner 再推导：

```text
RING_EXECUTOR
CORNER_EXECUTOR
CRUISE
```

见 `apps/race_runner.py:786-791`。

然后 `CommandArbiter.resolve()`：

1. 记录 owner/owner_epoch；
2. 校验 candidate 命令是否有限且不越界；
3. 锁存 stop cause；
4. 必要时将最终命令覆盖为 `v=0,w=0`；
5. 或执行 slow clamp。

见 `transbot_race/control.py:95-166`。

最终只有：

```python
bot.set_car_motion(cmd_v, cmd_w)
```

见 `apps/race_runner.py:826-829`。

---

## 3. 普通虚线巡线现在怎么工作

### 3.1 当前 steering 已是 lateral-only

普通 cruise 当前只用横向误差 `e0`：

```text
w = sign × k_e × filtered_e0
```

`theta` 不再直接进入普通 cruise steering。它仍用于：

- telemetry；
- observation continuity；
- handoff readiness；
- corner/ring 几何。

这项修改针对 190744 的证据是合理的：直线段 `e0` 已相对稳定，而局部 dash 的 `theta` 明显更噪。继续把 theta 加入普通 steering 会把每一段虚线的局部方向噪声直接变成左右摆动。

### 3.2 但 gap prediction 仍有累积漂移风险

当前滤波在 observation 不可用时每帧执行：

```python
f_e0 += d_e0
f_e_look += d_e0
conf -= conf_decay
```

见 `transbot_race/state_machine.py:291-318`。

这意味着同一个速度估计 `d_e0` 会在连续 gap 中反复外推，直到 confidence 进入 LOST。它比“直接追逐每一段 dash”平滑，但仍可能把一次错误趋势重复累加到饱和。

这项行为是 owner/filter rollback 后按用户要求恢复的原行为；当前并没有 one-shot gap prediction。

### 3.3 当前可用性契约比旧 run 更安全

普通 cruise 的 trackable 条件现在要求：

```text
control_valid
conf >= conf_predict
n_bands >= 3
not disconnected
```

见 `transbot_race/state_machine.py:274-289`。

因此旧 run 中的：

```text
found=true + 单个远端 band
```

当前不会自动等于：

```text
可以更新 cruise pose / 可以获得移动控制
```

---

## 4. Corner 当前怎么工作

### 4.1 正常流程

```text
armed
→ approach
→ waiting
→ turning
→ exit_tracking / captured / aligning
→ cooldown
→ handoff_ready
```

失败分支包括：

```text
seeking
failed
failed_locked
```

核心代码：`transbot_race/path_memory.py:251-739`。

物理含义：

1. geometry 先看到拐弯；
2. executor 冻结一条可信的入弯直行命令；
3. vertex 到达图像 gate 后，才开始计算 camera-to-axle 距离；
4. 距离消耗完后原地转；
5. 找到同侧第一条出口线；
6. executor 自己低速跟踪出口；
7. 只有完整、连续、适合 cruise 的线稳定若干帧后才 handoff。

这套流程在 160851、185400、201757、203256 中都完成过；因此不是 corner 整体不可用。

### 4.2 当前严重语义错误：curve 会被正式升级为 corner

这不是误会，而是当前代码明确写着允许：

```python
allowed_shapes={"corner", "curve"}
```

见 `transbot_race/capture_geometry.py:549-564`。

随后 mission 也允许 raw observation：

```python
observation.kind in {"corner", "curve"}
```

见 `transbot_race/mission.py:120-145`。

最终 corner executor 又把 raw shape 存进 `event_shape`：

```python
event_shape = geometry.kind
```

见 `transbot_race/path_memory.py:294-303`。

所以实际链路是：

```text
raw shape = curve
→ CornerGeometryFilter 输出 decision.kind = corner
→ mission CORNER 接受
→ CornerCommandDelay 获得底盘
→ event_shape 仍可能等于 curve
```

这不仅是 telemetry 命名问题。`event_shape` 会影响 cooldown 清除规则和 turn limit 分支，见 `transbot_race/path_memory.py:580-592,778-786`。

201757 和 203256 都实际记录了 curve 先出现、随后 corner executor 接管。当前实现的设计理由是“圆角弯与锐角 corner 是同一导航事件”，但这与用户要求的严格 corner/curve 语义冲突，而且跨层标签不一致。

### 4.3 同一帧 handoff 仍不是原子的

runner 在 `corner_margin.step()` 之前计算：

```python
corner_owns_chassis
```

见 `apps/race_runner.py:602-609`。

同一帧内 executor 可能完成 handoff，runner 调用：

```python
sm.reacquire_from(...)
mission.transition(...)
```

见 `apps/race_runner.py:748-768`。

但随后 owner 选择仍复用本帧较早计算的 `corner_owns_chassis`，见 `apps/race_runner.py:786-791`。

结果是：handoff 帧可以出现：

```text
candidate command 已来自 cruise
mission 已进入下一阶段
transition_event = handoff_ready
control_owner 仍记录 corner_executor
```

201757 的最后一帧就是这种证据。原 owner/filter suspend API 已删除，因此当前不会再触发当时的 RuntimeError；但一帧的 owner/command 语义不一致仍在。

### 4.4 failed_locked 是永久停

corner 出口超过 prediction horizon 后进入 `failed_locked`，见 `transbot_race/path_memory.py:532-539`。

runner 将其映射为：

```text
StopCause.EXECUTOR_FAILED
```

见 `apps/race_runner.py:809-812`。

该 stop 是非自动恢复 cause。174902 中它正确保护了底盘，但也说明一旦错误锁定出口身份后丢失，run 会永久停住，必须由明确 reset/人工干预恢复。

---

## 5. Ring 当前怎么工作

### 5.1 正常流程

```text
waiting
→ margin
→ tracking
→ inside
→ exiting
→ completed
```

见 `transbot_race/ring_entry.py:198-321`。

当前配置：

```json
"roundabout_margin_enabled": false
```

见 `configs/race_config.json:72`。

因此正常情况下 ring gate 接受后会直接进入 selected-path tracking，不再执行长 camera-to-axle margin。

### 5.2 历史 margin 旋转已修复

190744 中 ring margin 冻结：

```text
v=0.027, w=-0.24
```

持续约 16.66 秒，这是直接造成持续旋转的原因。

当前代码在进入 margin 时明确：

```python
self.margin_v = max(0.0, incoming_v)
self.margin_w = 0.0
```

见 `transbot_race/ring_entry.py:223-239`。

因此历史的“margin 期间复制旧角速度”已经修复；并且当前部署配置直接关闭 margin。

### 5.3 selected path 的含义被夸大

`selected_path_fit()` 把 skeleton path 转为 `TrajectoryFit` 时固定写：

```text
found = true
n_bands = 6
disconnected = false
path_memory = true
```

见 `transbot_race/ring_entry.py:68-83`。

但这些值不是由普通 6 个扫描带实际观测得出，而是 synthetic metadata。

203256 中出现了最清楚的结果：当前 mask 已只剩底边小区域，telemetry 仍显示：

```text
found=true
conf=0.8
n_bands=6
```

这并不表示当前相机仍看见一条完整六带路线，只表示 selected skeleton/path memory 仍有输出。

### 5.4 route fit 可以替自己证明 takeover readiness

runner 先生成：

```python
route_fit = selected_path_fit(...)
```

再选择：

```python
takeover_fit = route_fit or visual_fit
```

然后同时把它用于：

- `mission.gate()` 的 track quality；
- `sm.can_take_ring_entry()` 的 takeover readiness。

见 `apps/race_runner.py:554-569`。

由于 route fit 自带 `path_memory=True`、`n_bands=6` 和至少 0.35 confidence，一个 geometry candidate 可能同时提供：

1. “这里存在 ring route”的几何证据；
2. “当前控制质量足以接管”的质量证据。

这形成了自验证闭环。更稳妥的契约应是：

```text
geometry 决定候选路线
独立 incoming visual/pose 证明车辆当前可安全接管
```

### 5.5 route loss 会永久锁停

selected route 丢失超过 5 帧后，ring result 返回 `ROUTE_LOST`。runner 产生：

```text
StopCause.ROUTE_LOST
```

见 `apps/race_runner.py:801-808`。

但 `CommandArbiter._AUTO_RECOVERABLE` 不包含 `ROUTE_LOST`，见 `transbot_race/control.py:76-80`；runner 又没有任何：

```python
arbiter.clear_stop(StopCause.ROUTE_LOST)
```

因此当前一旦 route loss 锁存，即使后面路线重新出现，arbiter 仍会持续输出零命令。

这是当前最高优先级的实际控制问题之一。

### 5.6 entry completed 与 executor completed 仍容易混淆

`RingEntryResult.completed=True` 同时用于：

- `tracking→inside`：只表示入口建立；
- `exiting→completed`：表示整个 ring executor 完成。

虽然已有 `RingPhaseEvent` 区分，但 bool 仍保留双义。见 `transbot_race/ring_entry.py:261-306`。

同样，参数 `accepted_entry` 在 `waiting` 表示入口接受，在 `inside` 又表示出口选择被接受，见 `transbot_race/ring_entry.py:223-225,278-291`。

---

## 6. Safety 与 obstacle 当前问题

### 6.1 Safety arbiter 的正确部分

当前已经做到：

- candidate 与 final command 分开记录；
- stop cause typed；
- stop latch 持久化；
- owner 与 safety 分离；
- invalid command 会锁停；
- 最终只有一处写电机。

这些是应保留的架构改进。

### 6.2 obstacle hold 会冻结 executor 时间基准

ring active 且 obstacle stop 时，runner 不调用 `ring_entry.step()`，只构造 held result，见 `apps/race_runner.py:574-593`。

corner active 时，`_corner_control_allowed(...)` 会阻止 `corner_margin.step()`，见 `apps/race_runner.py:705-724`。

两个 executor 内部都用 `last_now` 计算 `dt`。暂停期间不更新 `last_now`，恢复第一帧会将整个暂停间隔压缩成一次 `dt`（ring 上限 1 秒；corner 受 `max_motion_dt_sec` 约束）。

后果可能是：

- margin distance 突然减少；
- travelled distance 突然增加；
- yaw budget 突然消耗；
- phase 提前 transition。

正确 hold contract 应在暂停时继续更新时间基准，但不累计运动，或显式 `pause/resume`。

### 6.3 slow clamp 不处理反向速度幅值

`CommandArbiter` 只在：

```python
candidate.v > slow_v_limit
```

时 clamp，见 `transbot_race/control.py:144-155`。

如果未来 executor 支持负速度，较大的反向速度不会被 slow policy 限制。

### 6.4 invalid candidate 之前 owner 已切换

arbiter 在校验 candidate 前先更新 owner/epoch，见 `transbot_race/control.py:104-114`。

因此一个越界或 NaN candidate 可以造成：

```text
owner epoch 已切换
→ candidate 随后被判 invalid
→ safety 锁停
```

这不会发出危险命令，但 telemetry 会把一个无效 takeover 记录成 owner 已接受。

---

## 7. Mission 与配置仍未闭环

### 7.1 Mission 名义上有、实际未实现的阶段

当前顺序：

```text
OBSTACLE → CORNER → RING_ENTRY → RING_EXIT → FORK → RETURN → FINISHED
```

见 `transbot_race/mission.py:51-59`。

但：

- `OBSTACLE` 映射到普通 cruise，障碍检测不会推进该 mission phase；
- `FORK` 没有专用 detector/executor；
- `RETURN` 没有专用行为；
- `fork_branch` 被配置和校验，却不参与当前 active control；
- 当前实际配置直接从 `corner` 开始。

因此这是一个“声明完整、执行不完整”的课程图。

### 7.2 direction 有多个来源

当前至少有：

- `path_memory.roundabout_direction`；
- `mission.ring_entry_direction`；
- `mission.ring_exit_direction`。

runner 每帧还会修改：

```python
cfg.path_memory.roundabout_direction = ...
```

见 `apps/race_runner.py:504-518`。

配置对象因此既是配置，又变成 runtime state。日志和后续模块很难判断某个 direction 是部署配置、mission intent 还是本帧动态覆盖。

### 7.3 dead / misleading config

当前 `k_theta_cruise` 仍在 JSON 和 dataclass 中，见：

- `configs/race_config.json:128`
- `transbot_race/config.py:188-193`

但普通 cruise 已不读取它。

`incoming_w` 仍传入 ring executor，见 `apps/race_runner.py:591-593`，但 translation-only margin 已不使用它。

`camera_to_axle_m=0.48` 是真实标定参数变化，与架构重构混在同一个 dirty 工作树中；这会让 run 间差异难以归因。

---

## 8. 为什么单元测试通过仍然会在 runner 崩

当前测试重点是：

- 单独给 `RaceStateMachine` synthetic fit；
- 单独给 corner executor 一段 observation；
- 单独给 ring executor route fit；
- 单独给 mission gate decision；
- replay 一张 mask 到 perception contract。

缺失的是一个真正 runner-level 测试，逐帧执行：

```text
visual fit
→ geometry observation/filter
→ mission gate
→ executor step
→ mission transition
→ owner selection
→ arbiter
→ final motor command
```

201757 的 owner/filter 回归正是这种测试缺口：各模块单测都过，但同一帧中 corner 完成 handoff 后，runner 仍复用了 handoff 前的 owner 布尔值。

需要的核心集成断言是：

```text
每一帧：
- candidate producer 必须等于 control_owner
- mission transition、executor transition、owner transition 必须形成一个合法组合
- handoff 帧不能用旧 owner 标注新 controller 命令
- stop cause 清除策略必须可达
- obstacle hold/resume 不得累计暂停时间
```

---

## 9. 已修复、仍有问题、不要混淆

### 9.1 当前已经修复

1. **190744 ring margin 复制旧角速度**：当前 margin 强制 `w=0`，且配置关闭。
2. **旧普通 cruise 直接使用 noisy theta**：当前 generic cruise 是 lateral-only。
3. **174902 far-only 反光拥有普通控制权**：当前新增 `control_valid/near support` 契约。
4. **corner/ring active 时后台推进 cruise**：当前 executor active 分支不调用普通 cruise step。
5. **owner/filter suspend RuntimeError**：相关 suspend/resume/filter epoch API 已按要求删除。
6. **motor owner 和 safety 完全不可观测**：当前已有 typed owner、epoch、stop cause、candidate/final telemetry。

### 9.2 当前仍存在

P0 / 必须先处理：

1. `ROUTE_LOST` 永久锁停、无 clear/recovery path；
2. obstacle hold 后 executor `dt` 过大；
3. curve 被正式升级为 corner；
4. ring takeover route fit 自验证；
5. handoff 帧 command producer 与 owner 可能不一致。

P1 / 直接影响 ring 可靠性：

6. selected path synthetic `n_bands=6` 掩盖实时视觉缺失；
7. ring selected path 在复杂圆环中会换支、饱和；
8. entry phase completed 与 executor completed 双义；
9. mission 与 executor transition 非原子；
10. ring/corner geometry filter reset 分散在多个 runner 分支。

P2 / 维护与可观测性：

11. runtime 修改配置 direction；
12. dead config `k_theta_cruise`；
13. `fork_branch` 未进入 active control；
14. `OBSTACLE/FORK/RETURN` 是名义 mission；
15. 生产 JSON 没有被完整 runner replay 测试加载；
16. unknown config key 仍可能静默忽略。

---

## 10. 建议后续顺序

在再次改控制参数前，建议严格按以下顺序：

### 第一阶段：修 runner 原子性，不改轨迹算法

1. 将 corner/ring `step()` 结果、mission event、next owner 组合成不可变 frame transition；
2. owner 必须在 executor step 后按新状态计算；
3. candidate producer 与 owner 加 assertion；
4. obstacle 增加显式 pause/resume 时间契约；
5. 为 `ROUTE_LOST` 定义明确恢复或人工 clear 路径。

### 第二阶段：修语义，不调阈值

1. 明确课程中的 `curve` 与 `corner` 是否同一个导航事件；
2. 如果不是，禁止 `CornerGeometryFilter` 接受 `curve`；
3. 如果确实是，必须改成一个独立 typed event（例如 `TURN_EVENT`），不能 decision 写 `corner`、executor 又保存 raw `curve`；
4. ring takeover quality 必须来自独立 incoming visual pose；
5. selected path telemetry 必须区分 `observed_now` 与 `predicted/path_memory`。

### 第三阶段：再处理 ring route stability

1. 对每个 selected skeleton 保留 route identity；
2. branch swap 必须经过明确连续确认；
3. 不要用 synthetic `n_bands=6` 表达 route quality；
4. entry established 必须有物理/拓扑证据，不只依赖路径对象连续存在；
5. 为 ring exit 建立生产配置下的完整 replay。

### 第四阶段：最后调虚线与标定

1. 保持 lateral-only cruise；
2. 用正式 run replay 验证 repeated `d_e0` extrapolation 是否仍造成 gap 漂移；
3. calibration 变更与架构变更分开提交、分开 run；
4. 每个 config 变化必须写入 run effective config 和 source。

---

## 11. 最终判断

当前不是“所有修改都无效”。以下方向是正确的：

- owner/safety typed 化；
- candidate/final command 分离；
- far-only 不可控制；
- generic cruise lateral-only；
- corner/ring active 时冻结普通 cruise；
- ring margin translation-only。

但当前代码仍把关键 transition 分散在 runner 的执行顺序中，导致**模块本身正确不等于整帧行为正确**。最应停止的是继续针对某个 run 调 angle/confidence；最应先做的是把 runner 的 transition 变成原子、可测试、每帧唯一的一份事实。