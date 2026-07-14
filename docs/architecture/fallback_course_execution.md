# 退化版赛道后续执行文件

> 这是后续实现工作的约束与检查表。  
> 分支：`fallback/line-parking-fan`  
> 基线：`be8b7ad`  
> 目标：独立实现“巡线 → 倒车入库 → 风扇 → 停车”，不得污染完整比赛版本。

## 1. 不可违反的边界

- [ ] 只在 `fallback/line-parking-fan` 分支工作。
- [ ] 不修改或切换用户原工作区 `codex/control-transition-refactor`。
- [ ] 不改造 `apps/race_runner.py` 来兼容退化版。
- [ ] 不改完整赛道的 `FixedSessionMission` 任务链。
- [ ] 不删除或重命名原版模块、配置、测试和入口。
- [ ] 不复制原 runner 全文件后再删逻辑；仅按明确依赖复用成熟模块。
- [ ] 不假设测距或风扇 API；实机适配器必须建立在现场证据上。
- [ ] 不使用车灯接口冒充风扇接口。
- [ ] 不以固定时间盲倒作为唯一完成判据。
- [ ] 未经用户明确要求，不提交、不推送、不创建 PR。

## 2. 目标架构

建议新增独立文件（名称可在实施前按现有命名复核）：

```text
apps/fallback_runner.py
configs/fallback_course.json
transbot_race/fallback_mission.py
transbot_race/parking.py
transbot_race/fan.py
transbot_race/range_sensor.py
```

职责：

- `fallback_runner.py`：唯一循环、唯一相机、统一调度和安全收尾；
- `fallback_course.json`：独立参数，不读取完整赛道任务参数；
- `fallback_mission.py`：小型显式状态机和合法转换；
- `parking.py`：库位触发、stage、倒车和完成判据；
- `range_sensor.py`：协议接口、dry-run fake、后续实机适配；
- `fan.py`：协议接口、dry-run fake、后续实机适配；
- 继续复用 `vision.py` 的黑线处理和拟合；
- 继续复用 `state_machine.py` 的普通巡线；
- 继续复用 `control.py`/`motor.py` 的单一命令和安全出口。

若实际实现只需更少文件，应合并简单协议，避免为了形式制造空模块。

## 3. 状态与所有权

目标状态：

```text
STARTUP
FOLLOW_LINE
PARK_TRIGGER
PARK_STAGE
PARK_REVERSE
PARK_VERIFY
FAN_RUN
FINISHED
FAULT
```

所有权规则：

| 状态 | 底盘所有者 | 允许非零运动 | 风扇 |
| --- | --- | --- | --- |
| STARTUP | safety | 否 | 关闭 |
| FOLLOW_LINE | cruise | 是 | 关闭 |
| PARK_TRIGGER | cruise/parking 原子切换 | 仅低速 | 关闭 |
| PARK_STAGE | parking | 受限 | 关闭 |
| PARK_REVERSE | parking | 仅受限倒车/修正 | 关闭 |
| PARK_VERIFY | safety | 否 | 关闭 |
| FAN_RUN | safety | 否 | 开启 |
| FINISHED | safety | 否 | 关闭 |
| FAULT | safety | 否 | 关闭 |

强制不变量：

```text
fan_on => commanded_v == 0 and commanded_w == 0
FAULT or FINISHED => commanded_v == 0 and commanded_w == 0
range_invalid_during_reverse => latch FAULT
camera_failure => latch FAULT
parking_timeout => latch FAULT
```

## 4. 感知契约

### 巡线

输入：单相机帧。  
输出：现有 `TrajectoryFit`。  
要求：使用近场可控支撑，持续丢线停车，不因远端墙缝/反光恢复运动。

### 入库触发

第一版必须是显式、可解释的视觉特征，例如：

- 主线旁出现特定矩形结构；
- 横向入口线与主线构成稳定 T/矩形边界；
- 固定 ROI 内连续多帧满足几何约束。

禁止：

- 仅凭“黑色像素变多”触发；
- 仅凭一帧触发；
- 仅凭运行时间或从起点估算时间触发；
- 让普通巡线器在主线和库位边线之间自由选最大轮廓。

### 测距

协议最小输出：

```python
RangeReading(
    valid: bool,
    distance_m: float | None,
    timestamp: float,
    reason: str,
)
```

必须定义：

- 单位固定为 metre；
- 有效量程；
- 无回波、超时、NaN、零值和离群值行为；
- 最大数据年龄；
- 倒车时连续异常多少帧立即停车；
- 停车阈值、减速阈值和滞回。

没有真实 API 前只实现协议和 fake，不写猜测式反射调用。

### 风扇

协议最小能力：

```python
fan.off()
fan.on()
fan.close()  # 必须保证 off
```

要求：

- 构造后默认关闭；
- 异常、Ctrl-C、超时和正常退出均关闭；
- 只有 `PARK_VERIFY` 成功后才允许开启；
- 达到最大运行时长后无条件关闭；
- dry-run 只记录事件，不访问 GPIO。

## 5. 倒车策略

首选闭环：

1. 视觉连续确认库位；
2. 停车，固定一次 parking target，禁止重新选主线；
3. 以现场标定的低速进入倒车起始姿态；
4. 倒车阶段用库位边线做横向/角度修正；
5. 后向测距进入减速区后进一步限速；
6. 达到停止距离并连续确认后停车；
7. 同时检查最大倒车时间和最大估算距离；任一超限进入 `FAULT`。

若倒车时相机看不到库位边线：

- 先评估是否可通过停车前对准，使倒车段只需直退；
- 使用后向测距完成末端停车；
- 固定时间/距离只能作为上限，不作为成功判据；
- 无稳定闭环证据时不进入风扇状态。

## 6. 配置原则

配置项仅允许保存可调参数，不保存未验证事实。建议分组：

```text
camera
tracker
parking_trigger
parking_motion
range_sensor
fan
safety
logging
```

所有初始数值标记为“采数值”，实车后才能冻结。至少需要：

- 巡线速度/角速度上限；
- 入库触发窗口和确认帧数；
- stage 速度/角速度/最大时长；
- 倒车速度、修正角速度、slew limit；
- 测距减速/停车阈值；
- 最大倒车时长和最大估算距离；
- 风扇运行时长与硬上限；
- 相机失败、丢线、测距失效的停止策略。

## 7. 测试清单

### 单元测试

- [ ] 合法状态转换完整，非法转换拒绝并停车；
- [ ] 普通巡线不能直接跳进 `FAN_RUN`；
- [ ] 入库触发要求连续帧，单帧假矩形不触发；
- [ ] 触发锁定后目标不跳回主线；
- [ ] 倒车命令始终满足 `v <= 0` 和速度上限；
- [ ] 测距 NaN/负值/超时/陈旧数据触发停车；
- [ ] 停止距离确认包含滞回，避免抖动反复启停；
- [ ] parking timeout 进入锁存 `FAULT`；
- [ ] `FAN_RUN` 中底盘命令恒为零；
- [ ] 任意异常路径均调用 `fan.off()` 和底盘停车；
- [ ] dry-run 不加载实机 Transbot/GPIO 库。

### 离线视觉测试

- [ ] 普通直线、弯道和回头弯不误触发入库；
- [ ] 反光、墙缝、瓷砖缝不误触发；
- [ ] 库位矩形在允许视角范围内稳定触发；
- [ ] 主线和库位边线同时出现时目标不跳线；
- [ ] 入口被局部遮挡时拒绝或停车，不盲倒。

### 实车分阶段测试

1. [ ] `--dry-run` 完整状态模拟；
2. [ ] 仅相机 shadow mode，电机永远为零；
3. [ ] 只跑普通巡线，入库检测只记日志；
4. [ ] 人工单步标定 stage 动作；
5. [ ] 单独读测距，车轮悬空/底盘不动；
6. [ ] 单独控制风扇，底盘锁定停车；
7. [ ] 低速倒车，不启动风扇；
8. [ ] 入库完成后人工确认，再允许风扇；
9. [ ] 最后才启用自动串联。

## 8. 每次实车必须记录

- 原始相机帧或视频；
- 黑线 mask、巡线 fit 和入库 overlay；
- 当前状态、状态进入时间、转换原因；
- range 原始值、滤波值、有效性和数据年龄；
- 候选命令、最终命令和控制所有者；
- parking target、误差、累计倒车时间/估算距离；
- 风扇 on/off 时间和关闭原因；
- fault/stop 原因；
- 配置快照和代码 commit。

## 9. 现场资料门槛

以下信息未获得前，不实现对应实机适配器：

- [ ] 测距传感器型号、接线和可运行读取示例；
- [ ] 风扇驱动板/继电器型号、供电、电平和可运行启停示例；
- [ ] 库位实测宽度、深度和目标后向间距；
- [ ] 车体宽度、长度、轴距及相机相对位置；
- [ ] 车载相机完整路线视频；
- [ ] 现场急停方式。

## 10. 完成定义

软件层完成：

- 独立入口和配置存在；
- 原完整比赛入口和测试不受影响；
- 新旧测试全部通过；
- dry-run 可覆盖成功和全部故障路径；
- 所有故障最终都使底盘停、风扇关；
- 文档中的未验证参数没有被描述为“已验证”。

实车层完成：

- 多次普通赛道运行不误触发入库；
- 入库触发在安全位置完成，未越过可停止边界；
- 倒车阶段无越界、碰撞和持续丢感知；
- 停车距离重复性满足现场要求；
- 风扇只在确认停车后启动，且所有退出路径可靠关闭；
- 失败时宁可停车，也不继续猜测动作。
