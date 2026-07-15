# 大版本后六个正式 Run 逐帧证据报告

> 审计日期：2026-07-14
> 范围：`20260714-160851_final`、`174902_final`、`185400_final`、`190744_final`、`201757_final`、`203256_final`
> 方法：OpenCV 5.0.0 解码现存全部 artifact；按文件名时间戳与全部 telemetry 对齐；逐保存点检查 frame/crop/mask/overlay/obstacle/geometry。
> 边界：本文报告的是当时 run 所使用代码和配置的行为。之后已经修复的历史问题会明确标注，不能据此声称当前代码仍有完全相同行为。

## 1. 覆盖声明

### 1.1 总量

六个正式 run 合计：

- telemetry：**2,116 行，全部读取和纳入状态时间线**；
- 保存时间点：**542 个，全部对齐检查**；
- frame：542；
- crop：542；
- mask：542；
- overlay：542；
- obstacle：542；
- geometry：344；
- 现存图像 artifact：**3,054 个，全部可解码**。

每个保存点的 frame/crop/mask/overlay/obstacle 都完整。geometry 只在 detector active 或 recorder 有输出的阶段存在；缺口为连续阶段性缺口，不是随机坏文件。

### 1.2 分 run 数量

| Run | Telemetry | 保存点 | frame/crop/mask/overlay/obstacle | geometry |
|---|---:|---:|---:|---:|
| `160851` | 368 | 96 | 各 96 | 67 |
| `174902` | 291 | 60 | 各 60 | 15 |
| `185400` | 437 | 112 | 各 112 | 70 |
| `190744` | 432 | 125 | 各 125 | 99 |
| `201757` | 224 | 48 | 各 48 | 20 |
| `203256` | 364 | 101 | 各 101 | 73 |

所有保存点均与 telemetry 精确或在 1 ms 内对齐。

### 1.3 证据等级

本文明确区分：

- **图像直接证据**：原始 frame/crop、mask、overlay、geometry 中可见；
- **Telemetry 直接记录**：状态、owner、命令、gate reason 等字段明确写出；
- **推断**：根据图像变化和命令推断物理车身行为，没有外部位姿真值。

注意：旧 run telemetry 没有显式 `control_valid` / `has_near_support` 字段。对旧 run 只能用 mask 空间位置、`n_bands`、`found`、`disconnected` 推断“远端支撑”，不能反向声称当时某布尔字段为 true/false。

---

## 2. 跨 run 总结

### 2.1 六个 run 实际证明了什么

1. **Corner 整体流程可以成功**：160851、185400、201757、203256 都完成了 turn 与出口跟踪；
2. **Corner 出口也会失败并永久锁停**：174902；
3. **Far-only 反光/瓷砖干扰真实存在**：160851 和 174902 的 waiting/turning 阶段最清楚；
4. **虚线 zigzag 不只是预处理问题**：160851、190744、203256 都出现 observation 断续→predict/lost/search→角速度换向；
5. **历史 ring margin 确实冻结角速度旋转**：190744，证据非常确定；
6. **无 margin 并不自动解决 ring**：185400 和 203256 仍出现 selected path 换支、实时视觉退化、未完成 ring exit；
7. **runner-level owner/filter sequencing 曾造成进程退出**：201757；该 owner/filter API 后来已回滚；
8. **测试通过不能覆盖真实集成顺序**：201757 是直接反例。

### 2.2 各问题的历史与当前状态

| 证据问题 | 最清楚 run | 当前状态 |
|---|---|---|
| far-only 反光进入普通控制 | `174902` | 已用 `control_valid/near support` 主要修复 |
| generic cruise 用 noisy theta | `190744` | 已改为 lateral-only |
| ring margin 复制旧 `w` | `190744` | 已改 translation-only，当前配置关闭 |
| owner/filter handoff RuntimeError | `201757` | 相关 owner/filter suspend API 已删除 |
| curve 被转为 corner | `201757`,`203256` | **当前仍存在** |
| ring route fit 换支/饱和 | `185400`,`203256` | **当前仍需解决** |
| ring path memory 与当前视觉混淆 | `203256` | **当前仍存在** |
| corner exit lost 永久停 | `174902` | fail-locked 仍是当前策略 |
| route lost 无 clear path | run 中未形成完整当前复现 | **当前代码审计确认存在** |

---

# 3. Run `20260714-160851_final`

## 3.1 完整时间线

```text
0.31–10.60   corner / cruise：普通巡线和 geometry 观察
10.61–15.70  corner_executor / approach
15.93–23.26  corner_executor / waiting margin
23.36–30.09  corner_executor / turning，期间有出口弱候选确认抖动
30.19–31.75  corner_executor / exit_tracking
31.84         handoff_ready，mission→ring_entry
31.94         cruise takeover
31.94–55.34  ring_entry / cruise，虚线 follow/predict/lost 多次切换
55.56–59.85  ring_executor / margin
```

run 结束时没有 stop，仍在 ring margin，未证明 ring entry 完成。

## 3.2 Key frame：curve 只是观察，随后 corner 确认

### `t=9.74`

Telemetry：

```text
geometry_raw_kind=curve
geometry_decision=null
geometry_gate_reason=no_turn_decision
```

### `t=10.61`

Telemetry：

```text
geometry_raw_kind=corner
geometry_decision=corner
corner_signature_accepted
incoming_e=-0.0923
incoming_theta=-0.0269
owner=corner_executor#2
```

这次是 raw curve 先出现，但真正接管发生在 raw corner 持续确认后。它证明 detector 能等待更强 signature；也与后续 201757 中 curve 被直接升级为 corner 形成对比。

## 3.3 Key frames：far-only 与地面干扰

### `masks/00026_0017300.png`

Telemetry：

```text
found=true
n_bands=1
e0=-0.2197
conf=0.1587
```

图像直接证据：mask 支撑集中在远端，近端四带无有效像素。

### `masks/00030_0019470.png`

Telemetry：

```text
found=false
n_bands=0
```

图像直接证据：远端仍有约 4,400 个前景像素，近端为 0。

说明：

```text
图像看见大量暗结构
≠ 形成一条可用主路径
```

### `masks/00033_0021180.png`

Telemetry：

```text
found=true
n_bands=1
disconnected=true
e0=0.0218
e_look=0.8617
preview_theta=1.1777
```

图像仍以远端结构为主。若 generic cruise 使用这些值，会有极端转向风险；但当时 owner 是 corner executor，命令固定直行，干扰没有穿透 owner 边界。

## 3.4 Corner 成功出口

### `overlays/00050_0030380.jpg`

```text
n_bands=6
e0=0.2993
theta=0.6859
conf=0.7482
command=(0.025,-0.0718)
```

### `overlays/00052_0031550.jpg`

```text
n_bands=5
e0=-0.0713
theta=0.0288
conf=0.7852
command=(0.025,+0.0171)
```

图像和 telemetry 共同显示出口线从偏侧逐步收敛为完整近远端路径。随后：

```text
31.84 handoff_ready + phase_completed
31.94 cruise takeover
```

这是成功 handoff 的参考样本。

## 3.5 虚线 zigzag 链

ring_entry 等待 topology 时：

- 40.43–41.08：LOST，`w=+0.16`；
- 44.28：LOST，`w=-0.16`；
- 47.78–50.04：LOST，`w=+0.16`；
- 53.16–53.58：LOST，`w=-0.10`。

典型：

### `overlays/00071_0043860.jpg`

```text
n_bands=1
disconnected=true
e0=0.3824
e_look=1.0
mode=predict
command=(0.0189,-0.24)
```

链路：

```text
虚线 gap / 候选断裂
→ n_bands 降到 0–2，disconnected
→ follow / predict / lost 来回切换
→ 搜索方向继承最后误差
→ 新 dash 重捕获时误差符号和幅值变化
→ w 在正负方向切换
```

## 3.6 Ring margin

### `overlays/00089_0055560.jpg`

```text
ring_entry_fork_accepted
owner=ring_executor#4
phase=margin
command=(0.049,-0.0797)
```

55.56–59.85 期间命令冻结为接管时值。随后视觉丢失，但 command 不变。run 在 margin 完成前结束。

这是旧 ring margin 会冻结角速度的早期样本；190744 给出了更严重、更长的同类证据。

---

# 4. Run `20260714-174902_final`

## 4.1 完整时间线

```text
0.31–4.10    corner / cruise
4.11–6.05    corner decision 已接受，但等待 control prerequisite
6.25–9.22    corner_executor / approach
9.42–17.01   waiting margin
17.10–23.10  turning / exit-confirming 抖动
23.19–27.31  exit_tracking
27.42–33.70  failed_locked + stop_latched(executor_failed)
```

没有 handoff，mission 始终停在 corner。

## 4.2 Geometry 接受不等于立即 owner 接管

4.11 s 起 telemetry 已显示：

```text
geometry_decision=corner
corner_signature_accepted
```

但 owner 仍为 cruise，reason 是：

```text
corner_waiting_control_prerequisite
```

到 6.25 s 才 takeover。这证明 gate acceptance 与 motor ownership 是两个不同步骤；这个分离是正确的。

## 4.3 远端反光/瓷砖候选

### `masks/00017_0011080.png`

```text
n_bands=3
e0=1.0
theta=-1.2281
disconnected=true
```

### `masks/00018_0011620.png`

```text
found=false
n_bands=0
```

图像：近端为 0，远端仍约 2,874 前景像素。

### `masks/00023_0014310.png`

```text
found=true
n_bands=1
e0=0.7785
```

图像：近端 0，远端约 3,746 前景像素。

这些 frame 是 anti-interference fixture 的来源区域。它们证明旧实现中 `found` 可以被远端结构触发，而不代表存在可控制的近场路线。

## 4.4 Exit confirming 抖动

19.05–20.97 s 期间 executor 在：

```text
committed_turn: command=(0,-0.2)
corner_exit_confirming: command=(0,0)
```

之间高频切换。

这不是 owner 冲突，而是 corner executor 内部弱候选反复满足/失去 temporal latch。弱视觉候选会让底盘“转一下、停一下、再转”。

## 4.5 Exit tracking 最终失败

### `overlays/00040_0023470.jpg`

```text
n_bands=4
e0=-1.0
theta=0.9765
command=(0.025,+0.1619)
```

### `overlays/00047_0027310.jpg`

```text
n_bands=2
e0=0.1095
theta=0.0558
```

数值看起来接近中心，但图像支撑仍主要在远端，近端极少，不能视为稳定出口重获。

### 精确 stop：`t=27.42`

```text
executor_phase=failed_locked
path_strategy_reason=corner_exit_lost
safety_state=stop_latched
stop_cause=executor_failed
transition_event=stop_latched
final=(0,0)
```

之后 63 行持续锁停。即使后续 `found=true`，也不会恢复。

故障链：

```text
转弯后只有偏边、远端、断裂候选
→ exit_tracking 保留已锁定出口身份
→ 1 秒 prediction horizon 内未恢复
→ failed_locked
→ arbiter 锁存 EXECUTOR_FAILED
→ 永久 v=0,w=0
```

Safety 最终优先级工作正确；根问题在出口视觉身份没有稳定建立。

---

# 5. Run `20260714-185400_final`

## 5.1 完整时间线

```text
0.32–6.16    corner / cruise
6.39–12.03   corner_executor / approach
12.24–19.89  waiting margin
19.98–24.77  committed turn
24.86–36.03  exit_tracking
36.13         mission→ring_entry
36.22–43.93  cruise
44.18–49.44  ring_executor / selected-path tracking
49.67–65.57  ring_exit mission / executor inside
```

run 结束时仍在 ring inside，没有完成出口。

## 5.2 Corner takeover

### `geometry/00009_0006390.jpg`

```text
raw_kind=corner
decision=corner
angle=53.88°
vertex_y_frac=0.2947
corner_signature_accepted
owner=corner_executor#2
```

Corner 流程随后成功完成。

## 5.3 无 margin 直接 selected path

配置：

```text
roundabout_margin_enabled=false
ring_effective_margin_m=0
```

### `geometry/00073_0044180.jpg`

```text
ring_entry_fork_accepted
owner=ring_executor#4
phase=tracking
fit_source=selected_path
command=(0.0204,-0.08)
```

这说明关闭 margin 后，ring 确实可以立即开始 route control，不再等待 camera-to-axle 距离。

## 5.4 Ring selected path 换支

49.67 s 后 mission 进入 ring_exit，但 executor 保持 inside。

后段 telemetry：

```text
e0: -0.9929 → +1.0
theta: 可到 +0.9281，后又到 -0.8931
selected_theta: -1.35 ↔ +1.35
w: 多次在 -0.2 与 +0.2 附近换向
```

最混乱 mask：

- `masks/00089_0053170.png`
- `masks/00090_0053680.png`
- `masks/00091_0054190.png`
- `masks/00092_0054710.png`
- `masks/00093_0055250.png`

图像直接证据：多连通域、宽结构、行中心换支。Telemetry 与其同步发生 selected route 饱和翻转。

故障链：

```text
环形/分支结构同时可见
→ skeleton 每帧重建
→ selected path identity 不稳定
→ carrot 从一支跳到另一支
→ e_look/theta 饱和翻转
→ ring control w 正负切换
→ 没有形成稳定环内行驶与出口选择
```

这个 run 证明：**关闭 margin 只解决“控制开始太晚”，不解决 selected route identity 不稳定。**

---

# 6. Run `20260714-190744_final`

## 6.1 完整时间线

```text
0.32–18.43   corner / cruise
18.64–18.84  corner approach
19.06–26.34  waiting margin
26.43–31.81  committed turn
31.91–33.65  exit_tracking
33.74–47.46  ring_entry / cruise，含两次 LOST
47.72–64.38  ring_executor / margin
64.62–69.90  selected-path tracking
70.10–73.79  inside
```

run 最后仍在 ring inside，未完成出口。

## 6.2 虚线 zigzag 最明确样本

33.84–47.46 s：

```text
36.37–36.78  LOST / line_search
43.37–44.13  LOST / line_search
```

关键命令：

- predict 可达 `w=+0.2274`；
- LOST 固定 `w=+0.16`；
- 重捕获后 `e0→+1.0`；
- command 变为 `w=-0.24`。

### `overlays/00077_0046960.jpg`

```text
e0=0.9322
w=-0.2357
```

### `overlays/00078_0047460.jpg`

```text
e0=1.0
w=-0.24
```

链路：

```text
虚线段离开 ROI
→ observation 失效
→ predict / LOST 向最后一侧搜索
→ 新 dash 从另一空间位置重现
→ fixed-reference pose 重新建立
→ e0 突变或饱和
→ steering 从正搜索切到负饱和
```

早先统计还显示该直线段 `e0` 的标准差远小于 `theta`。因此将 generic cruise 改为 lateral-only 是有证据支持的修复。

## 6.3 Ring margin 冻结旋转——全套证据

47.72 s takeover：

```text
ring_entry_fork_accepted
owner=ring_executor#4
phase=margin
```

47.72–64.38 s 共 79 行：

```text
v=0.027
w=-0.24
```

完全不变，持续约 **16.66 秒**。

同时：

```text
ring_margin_remaining_m: 0.45 → 0.0001
found: true/false 反复
pose_reference: fixed_bottom / nearest_observed / none
geometry direction: +1 / -1 变化
e0/theta: 大幅变化
candidate_command == final_command
safety_state=clear
```

关键 frame：

- `overlays/00079_0047960.jpg`：`e0=1.0`，仍 `w=-0.24`；
- `overlays/00090_0053880.jpg`：`found=false/pose none`，仍 `w=-0.24`；
- `overlays/00095_0056770.jpg`：direction=-1，仍 `w=-0.24`；
- `overlays/00104_0062320.jpg`：direction=+1，仍 `w=-0.24`；
- `overlays/00107_0063880.jpg`：remaining≈0.0137，仍 `w=-0.24`。

直接结论：

```text
不是预处理让车旋转
不是 safety override 让车旋转
不是实时 ring pursuit 让车旋转
而是 margin executor 冻结了接管瞬间的旧角速度
```

当前该问题已通过 `margin_w=0` 修复，且部署配置关闭 margin。

## 6.4 Margin 后 route control 太晚

64.62 s 才开始 selected path。此时车辆已被旧命令持续旋转 16.66 秒。随后 tracking 又长时间为：

```text
v=0.0204
w=-0.08
```

70.10 s 进入 inside，后续接近 `w=-0.2`。run 结束时仍在环内。

所以该 run 不能证明 route fit 本身能正确进环；它开始控制时初始姿态已经被 margin 破坏。

---

# 7. Run `20260714-201757_final`

## 7.1 完整时间线

```text
0.31–12.46   corner / cruise
12.67–12.88  corner_executor / approach
13.10–20.98  waiting margin
21.09–26.45  turning
26.54–28.38  exit_tracking
28.48         handoff_ready + mission→ring_entry，随后运行结束
```

## 7.2 Curve 被升级成 corner

### `overlays/00018_0012240.jpg`

```text
geometry_raw_kind=curve
geometry_kind=corner
geometry_decision=corner
corner_reject_incoming_stem
```

### `overlays/00019_0012880.jpg`

```text
geometry_kind=corner
corner_signature_accepted
owner=corner_executor
phase=approach
```

这是最清楚的语义证据：raw observation 是 curve，但 session filter 输出 corner decision，随后 corner executor 接管。

这项行为在当前代码仍明确存在，不是历史 telemetry 错标。

## 7.3 Corner 实际完成了出口跟踪

### `overlays/00044_0026640.jpg`

```text
phase=exit_tracking
turn_exit_latched=true
e0=1.0
theta=0.4087
command=(0.025,-0.2)
```

### `overlays/00047_0028380.jpg`

```text
e0=0.3238
theta=0.2481
command=(0.025,-0.0777)
```

图像中出口线从右侧连续向中心移动，命令同步收敛。所谓“转弯之后不动”并不是 corner 没进入 exit tracking；它确实在低速 `v=0.025` 跟踪出口。

## 7.4 Handoff 边界与 owner/filter 回归

最后 telemetry `t=28.48`：

```text
mission_state=ring_entry
mission_transition_event=phase_completed
transition_event=handoff_ready
reason=track_follow
candidate≈(0.0502,-0.0710)
control_owner=corner_executor
```

这帧已经产生 cruise 风格命令和 mission transition，但 owner 仍保留 handoff 前的 corner owner。

当时临时加入的 owner/filter suspend 代码随后在下一帧再次把 cruise suspend，触发 RuntimeError，进程 finally stop。该实现后来按要求整套删除。

当前不再有 RuntimeError 路径，但 runner 仍在 executor step 前计算 `corner_owns_chassis`，所以 handoff 帧 owner 标注落后一帧的问题仍值得修。

## 7.5 运行结束原因

- `max_sec=60`；
- manifest 约 29.57 秒；
- safety clear；
- stop_cause null；
- 最后 command 非零。

所以这不是正常 safety stop，也不是 mission 完成。结合当时代码和 RuntimeError 复盘，终止来自 runner 集成回归，而不是 corner 算法主动停住。

---

# 8. Run `20260714-203256_final`

## 8.1 完整时间线

```text
0.31–11.14   corner / cruise
11.15–11.36  corner_executor / approach
11.58–19.62  waiting margin
19.72–25.27  turning
25.37–27.10  exit_tracking
27.20         handoff_ready
27.29–36.57  ring_entry / cruise
36.83–38.51  LOST / line_search
38.77–42.28  cruise reacquired
42.54–46.78  ring_executor / tracking
47.01–59.99  ring_exit mission / executor inside
```

达到 60 秒上限结束，未完成 ring exit。

## 8.2 Curve→corner 再次复现

### `overlays/00015_0010100.jpg`

```text
mode=track_predict
geometry_raw_kind=curve
```

### `overlays/00017_0011360.jpg`

```text
geometry_kind=corner
geometry_decision=corner
corner_signature_accepted
owner=corner_executor
```

这与 201757 一致，说明不是单次 run 偶发。

## 8.3 Corner handoff 后大幅摆动

27.53–35.27 s，mask 线位置在几秒内：

```text
中心偏右 → 最左 → 中心 → 最右 → 再回左
```

Telemetry：

- `e0≈-0.59` 时 `w≈+0.2224`；
- `e0≈+1.0` 时 `w=-0.24`。

关键：

### `overlays/00050_0030040.jpg`

```text
mode=predict
e0=-0.5442
command=(0.0206,+0.2224)
```

### `overlays/00055_0033330.jpg`

```text
e0=0.9978
command=(0.027,-0.24)
```

这不是只有 telemetry 数值跳；mask 质心也实际横跨 ROI。没有外部位姿真值，不能精确量化车身 zigzag，但控制命令正负饱和的事实明确。

## 8.4 第一次 ring 候选被拒后 LOST

### `overlays/00059_0035790.jpg`

```text
geometry_decision=ring_entry
ring_entry_reject_vertex
e0=-0.5461
command=(0.045,+0.1093)
```

随后：

```text
found=false
→ predict
→ LOST / line_search (v=0,w=+0.16)
→ 38.77 s reacquired
```

该次 gate 拒绝与 tracker 搜索行为一致，没有错误 takeover。

## 8.5 第二次 ring takeover 与“看见/记忆”混淆

42.54 s：

```text
ring_entry_fork_accepted
owner=ring_executor#4
phase=tracking
```

从约 `masks/00076_*` 起，当前 mask 退化为底缘约 3.6%–3.9% 的固定小区域；但 telemetry 持续：

```text
found=true
conf=0.8
n_bands=6
e0 继续变化
```

47.01 s：

```text
ring_entry_established
mission→ring_exit
executor→inside
```

图像事实与 telemetry 事实并不矛盾，但字段语义容易误导：

- 图像当前没有清晰普通 scan line；
- selected/path-memory 仍输出 synthetic fit；
- `found/conf/n_bands` 描述的是路径对象，不是“当前相机六个 band 都看见线”。

## 8.6 环内未完成出口

50.51 s 后真实结构再次进入 mask，raw geometry 多次为 circle/ring-like。Mission gate 先后报告：

```text
ring_exit_waiting_arm_distance
ring_exit_waiting_branch
no_turn_decision
```

到 59.99 s：

```text
mission=ring_exit
owner=ring_executor
phase=inside
command=(0.0217,+0.1843)
stop_cause=null
```

因此 run 是时间上限结束，不是完成出口或 safety stop。

---

# 9. 两个 manual run 为什么不能作为控制验证

同时检查了：

- `artifacts/manual_runs/manual_20260714-204221`
- `artifacts/manual_runs/manual_20260714-204318`

结论：它们不是正式视觉 replay。

### `204221`

- 只有 `meta.json`；
- actions/motion 为空；
- 无 frame；
- 无 manifest。

### `204318`

- `dry_run=true`；
- `camera_enabled=false`；
- 四张 action 图片完全相同，都是 placeholder；
- 两次 forward 0.05 m；
- 49 条 motion 的 `measured` 来自 DryBot 命令回读，不是真实编码器/IMU；
- 没有 perception、owner、mission、control_valid、near support 或 handoff 字段。

因此它们只能验证 manual action recorder 的 smoke path，不能验证任何正式 anti-interference 或控制问题。

---

# 10. 最终故障归因

## 10.1 不是单一预处理问题

预处理确实会保留：

- 反光边缘；
- 瓷砖缝；
- 宽环形结构；
- 虚线的断续组件。

但最终是否变成危险命令，取决于后续契约：

```text
candidate selection
→ support authority
→ temporal continuity
→ mission gate
→ owner
→ executor phase
→ safety
```

174902 的 far-only 问题说明旧 support contract 不够；190744 的 margin 说明即使图像变化，executor 冻结命令也会独立造成旋转；185400/203256 又说明无 margin 时 route identity 仍可能不稳。

## 10.2 虚线问题的本质

```text
每一段 dash 是局部可见的
→ 当前被选中的路径不是同一连续实体
→ fit 在 gap 中预测
→ 新 dash 重捕获时 reference/候选发生变化
→ e0 或旧 theta/preview 大幅变化
→ follow/predict/lost/search 产生方向翻转
```

当前 lateral-only cruise 已消除 theta 直接 steering 的一层问题，但 repeated `d_e0` prediction 与 path identity 连续性仍需 production replay 验证。

## 10.3 Ring 问题的本质

历史：

```text
检测接受
→ margin 冻结 incoming w
→ selected path 控制开始太晚
```

当前：

```text
检测接受
→ selected skeleton 自己证明 takeover quality
→ 每帧重建 route
→ path identity 可能换支
→ synthetic fit 掩盖实时视觉丢失
→ entry established / inside 的物理证据不足
→ exit branch 长期不满足
```

## 10.4 架构问题的本质

```text
模块单测验证了局部 transition
但 runner 在同一帧按旧状态计算 owner
又按新状态推进 mission/executor
最后把两种时间截面的事实写进同一 telemetry
```

201757 是最明确例子。

---

# 11. 下一步必须以这些 key frame 建回归

建议最小 replay corpus：

1. `160851/00026_0017300`、`00030_0019470`、`00033_0021180`：far-only；
2. `174902/00040_0023470` 到 `00047_0027310`：出口错误身份与 loss；
3. `185400/00089_0053170` 到 `00093_0055250`：ring path 换支；
4. `190744/00077_0046960`、`00078_0047460`：虚线重捕获反向；
5. `190744/00079_0047960` 到 `00107_0063880`：margin 冻结角速度；
6. `201757/00018_0012240`、`00019_0012880`：curve→corner；
7. `201757/00044_0026640` 到最后 telemetry：handoff 原子性；
8. `203256/00050_0030040`、`00055_0033330`：handoff 后摆动；
9. `203256/00076` 到 `00084`：当前 mask 丢失但 selected path 仍报告完整 fit；
10. `203256/00085` 到 `00100`：ring exit gate 长期不完成。

Replay 不能只断言 perception。每帧必须断言：

```text
mission_state
raw observation
filtered decision
gate result
executor phase
control_owner
candidate command
safety state
final command
transition event
```

只有这样才能阻止再次出现“115 个单元测试全过，但 runner 下一帧直接退出”的情况。