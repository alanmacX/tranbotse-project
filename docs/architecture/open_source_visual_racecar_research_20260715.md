# 相似赛题纯视觉无人小车开源代码调研总档

> 文档用途：供后续 Agent 进行方案比较、代码定位和算法选型。
> 调研日期：2026-07-14 至 2026-07-15
> 调研对象：GitHub 公开仓库中与“摄像头巡黑线、直角弯、断线、环岛、颜色识别、岔路选道、停车和返程”相关的真实实现。
> 重要边界：本文只总结外部公开源码，不审计本地项目，也不提供本项目实施设计。

---

## 1. Executive Summary

### 1.1 最重要的结论

没有发现一个公开仓库能够直接完整实现以下整条竞赛链路：

```text
黑线巡线
→ 障碍停车或避障
→ 90°直角弯
→ 断线盲驶与重新捕线
→ 1 m 环岛按方向绕半圈
→ 红/绿/蓝物料识别
→ 三岔口选择 A/B/C
→ 原路返回
```

公开项目通常分为四类：

1. **基础巡线类**：能完成黑线提取、质心误差和 PID，但没有可靠任务状态机。
2. **路口导航类**：能识别停止线或交叉口，并执行左/直/右，但没有环岛和返程。
3. **竞赛整车类**：具有较完整的任务状态机和环岛流程，但常依赖 ROS、雷达、里程计或检测模型，不是严格纯视觉。
4. **工程框架类**：状态机、模块化和日志能力成熟，但与本赛道的黑线、环岛和颜色卸载任务相距较远。

因此，正确使用方式不是寻找“整仓复制”的项目，而是按模块参考：

| 本赛道模块 | 最值得研究的仓库 |
| --- | --- |
| 单目巡线、路口状态机、转弯后重捕线 | `Luissalamanca23/duckietown-follow-line-pi` |
| HSV 颜色检测、连续帧确认、任务模式切换 | `ROBOTIS-GIT/turtlebot3_autorace_2020` |
| 环岛实验代码/竞赛阶段流程 | `diaoquesang/smartcar2023`（分支未接通）、`Abaabaxx/UCAR`（多传感器） |
| 路口路径生成与视觉旋转估计 | `duckietown/duckietown-intnav` |
| 大型视觉导航 FSM | `duckietown/dt-core` |
| 视觉停车、目标触发和事件去抖 | `dctian/DeepPiCar` |
| 通用视觉巡线框架 | `autorope/donkeycar` |
| 黑线角度、连续目标选择 | `aryan-02/line_follow` |
| 方向色块与多路径连续性 | `DivyamArora22/RCJLine_Follower` |
| 简单交叉口计数和预设选路 | `Adilnasceng/Line-follower` |

### 1.2 综合推荐排序

综合考虑赛题相似度、代码真实性、视觉模块占比、许可证和可迁移性：

1. **Luissalamanca23/duckietown-follow-line-pi**：最值得参考其单目巡线、路口状态机和转弯后视觉重捕；完整默认返程仍使用编码器/里程计。
2. **ROBOTIS-GIT/turtlebot3_autorace_2020**：最值得参考红/黄/绿交通灯识别结构和路口任务调度；它没有蓝色物料与 A/B/C 映射。
3. **diaoquesang/smartcar2023**：公开代码包含绿色赛道巡线及环岛/“返回基地”实验分支，但核验 commit 每次图像回调都把 `huandao=False`，使该分支不可达，且脚本存在 `Odometry` 缺失导入；只能研究未接通实验逻辑。
4. **duckietown/duckietown-intnav**：适合研究交叉口轨迹和视觉旋转估计；没有环岛实现。
5. **Abaabaxx/UCAR**：公开代码覆盖多个竞赛阶段，但 ROS、雷达/里程计和底盘耦合重，且未发现许可证。
6. **duckietown/dt-core**：工程化状态机优秀，但整体迁移成本高。
7. **dctian/DeepPiCar**：适合参考视觉停车和目标事件处理。
8. **autorope/donkeycar**：适合参考可组合车辆管线，不适合直接解决比赛拓扑。

---

## 2. 调研方法与可信度说明

### 2.1 检索方向

本次检索覆盖以下关键词和代码形态：

- OpenCV camera line follower；
- black line following、HSV line detection；
- intersection、junction、fork、roundabout；
- traffic circle、turn sign、route selection；
- color block detection、traffic light detection；
- Raspberry Pi、Jetson、Arduino、ROS；
- path memory、teach and repeat、return route；
- PID、pure pursuit、state machine、reacquire line。

### 2.2 核验标准

候选仓库不是仅凭 README 入选，而是至少核对以下一项或多项：

- 实际图像处理源码；
- PID 或轨迹控制源码；
- 路口/环岛状态转换源码；
- 颜色分割或目标停车源码；
- 路线动作、任务 FSM 或返程源码；
- GitHub 仓库许可证元数据。

本文的“核验 commit”用于冻结本次判断所对应的源码版本；仓库后续更新可能改变文件路径和行为。

### 2.3 核验版本快照

| 仓库 | 分支 | 核验 commit |
| --- | --- | --- |
| `Luissalamanca23/duckietown-follow-line-pi` | `main` | `5cf0c96652aecd1142ee08f51f1c09f7aa47cdc4` |
| `ROBOTIS-GIT/turtlebot3_autorace_2020` | `main` | `367a24a7228fe9dcd339af50f70b8672df3cc0cc` |
| `diaoquesang/smartcar2023` | `main` | `abce313d298b4b0913e8d6a929f81c50ed766797` |
| `Abaabaxx/UCAR` | `archive` | `30190ad70bde97faa2083ba257bccd856276a19b` |
| `duckietown/duckietown-intnav` | `master` | `74c208fcd99468100b342d5b5624d6d63dbc13ec` |
| `duckietown/dt-core` | `ente` | `e4b6fc0629b86d7080ac66828ba74113a69c52f1` |
| `dctian/DeepPiCar` | `master` | `d01bd627c0bbbf4d167cbfdd9ad86f4741c3496c` |
| `autorope/donkeycar` | `main` | `b074a74ce190827e16f16b345dd69f44387f7b64` |
| `DivyamArora22/RCJLine_Follower` | `main` | `c3a74abba337365b8e21ccc42b8d9fb3be95acc6` |
| `Adilnasceng/Line-follower` | `main` | `b454ae581e5ef734a1f6c8570b158221f463d5b9` |
| `aryan-02/line_follow` | `master` | `b0b4d6fac7b7329f70722c31ba0efc161acdb722` |
| `GarvitTech/LINE_FOLLOWER_CAR` | `main` | `a4ce90907a7676f6fb4ba8b66f9e44b1387edcc5` |
| `d4n93rS4nY0k/Autonomous-Robotic-Device_DP` | `main` | `162ff7a3b63a52ea9208cf6ba94e51b67f400708` |
| `saumyaranjan1111/Line-Following-Robot-Using-OpenCV-and-Arduino` | `main` | `3da0898bd105897512456e060bb28cb03a4338dc` |
| `utiasASRL/vtr3` | `main` | `98c8d8d0eb127cc3ffbb1a05c2152752f5a5fd9e` |

复核源码时，应使用“仓库 URL + 上表 commit + 文中完整相对路径”。这样不会因默认分支更新而把后续代码误认为本次核验代码。

### 2.4 许可证判定边界

必须区分：

- **MIT / Apache-2.0**：通常适合借鉴和复用，但仍需遵守许可证中的版权、归属、修改标示等要求。
- **GPL-3.0 / AGPL-3.0**：可以研究；若复制、修改并分发覆盖代码，可能要求衍生作品按相同许可证提供源码。
- **Other / Duckietown Software Terms**：不是标准宽松许可证，复用前必须人工核对条款。
- **未发现许可证**：虽然 GitHub 上能看到源码，但法律上不等于获得复制、修改和分发授权。此类仓库只建议学习思想，不应直接复制代码。

---

## 3. 全部候选覆盖矩阵

符号定义严格按已核验源码：

- `✓`：源码中有对应任务的明确实现；
- `△`：只有子能力、原型或相邻任务，不能视为完整实现；
- `—`：未发现对应实现。

能力口径：

- **断线恢复**：专指线路暂时消失后的有限保持/搜索/重新捕线，不包括 Teach & Repeat 定位。
- **颜色分类**：专指多类别颜色判定；只用固定颜色巡线不算。
- **返程**：专指公开主流程中存在返回/重复路径行为；路线动作或注释性阶段只记 `△`。
- **视觉/传感器边界**：描述对应功能的完整闭环，而不只描述某个检测节点。

| 仓库 | 巡线 | 直角/路口 | 断线恢复 | 环岛 | 多类颜色 | 岔路选道 | 障碍/停车 | 返程 | 视觉/传感器边界 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `Luissalamanca23/duckietown-follow-line-pi` | ✓ | ✓ | △ | — | — | ✓ | ✓ | △ | 巡线/路口为单目；默认返程用编码器/里程计 |
| `ROBOTIS-GIT/turtlebot3_autorace_2020` | ✓ | ✓ | △ | — | ✓ | ✓ | △ | — | 视觉感知；动作闭环使用 TurtleBot 里程计/底盘栈 |
| `diaoquesang/smartcar2023` | ✓ | ✓ | △ | △ | — | — | △ | △ | 单目绿色赛道；环岛/返程实验分支在核验 commit 中不可达 |
| `Abaabaxx/UCAR` | ✓ | △ | △ | ✓ | ✓ | — | ✓ | △ | 多传感器 ROS 系统，含雷达/里程计 |
| `duckietown/duckietown-intnav` | △ | ✓ | △ | — | — | ✓ | — | — | 单目视觉路口模块；非环岛/返程系统 |
| `duckietown/dt-core` | ✓ | ✓ | △ | — | △ | ✓ | ✓ | △ | 多节点 Duckiebot 栈，常含标志与底盘里程信息 |
| `dctian/DeepPiCar` | ✓ | — | — | — | △ | — | ✓ | — | 车道为单目；目标检测可依赖 TFLite/Coral |
| `autorope/donkeycar` | ✓ | — | △ | — | — | — | △ | △ | 单目框架；具体行为可选编码器/模型等部件 |
| `DivyamArora22/RCJLine_Follower` | ✓ | ✓ | △ | — | △ | ✓ | — | — | 单目视觉 + 定时动作 |
| `Adilnasceng/Line-follower` | ✓ | ✓ | — | — | — | ✓ | — | — | 单目视觉 + 定时动作 |
| `aryan-02/line_follow` | ✓ | △ | △ | — | — | △ | △ | — | 单目视觉；障碍函数使用超声波 |
| `GarvitTech/LINE_FOLLOWER_CAR` | ✓ | ✓ | — | — | — | — | — | — | 单目巡线，急转闭环使用 MPU6050 |
| `d4n93rS4nY0k/Autonomous-Robotic-Device_DP` | ✓ | — | — | — | — | — | △ | — | 单目视觉 + Arduino；绕障为定时原型 |
| `saumyaranjan1111/Line-Following-Robot-Using-OpenCV-and-Arduino` | ✓ | — | △ | — | — | — | — | — | 单目视觉 + Arduino；断线只是零误差直行行为 |
| `utiasASRL/vtr3` | — | — | — | — | — | — | △ | ✓ | 多传感器 Teach & Repeat 框架；视觉 pipeline 通常为双目/GPU |

注意：`✓` 只证明该仓库在自身环境中有对应源码，不证明能原样用于目标赛道。

---

# 4. 高优先级项目详解

## 4.1 Luissalamanca23/duckietown-follow-line-pi

- 仓库：https://github.com/Luissalamanca23/duckietown-follow-line-pi
- 核验 commit：`5cf0c96652aecd1142ee08f51f1c09f7aa47cdc4`（`main`）
- 许可证：MIT（GitHub `licenseInfo` 与根许可证元数据）
- GitHub 描述：Python-only Duckietown line follower for Raspberry Pi 5，包含 Web UI、路线、LED 和 QR 支持。
- GitHub 元数据显示最后更新时间：2026-06-26。
- 推荐级别：**A+（限巡线、路口与重捕架构）**

### 关键源码

- `robot/vision.py`
- `robot/controller.py`
- `robot/state_machine.py`
- `robot/turn_controller.py`
- `robot/maneuver.py`
- `robot/route_manager.py`

### 实际实现机制

#### 视觉

- 使用 HSV 或颜色阈值提取目标线；
- 通过轮廓和图像区域计算横向误差；
- 除近场位置外，还保留前方线路信息；
- 检测横向停止线；
- 转弯过程中检查目标线路是否重新进入视野。

#### 控制

- PID 控制转向；
- 根据弯曲程度调整控制行为；
- 对积分项做限制，避免长时间偏差导致积分饱和；
- 转弯支持定时弧线或预设动作；
- 重新看见线路后可提前结束固定转弯动作。

#### 状态机

典型流程接近：

```text
FOLLOW_LINE
→ 检测停止线或路口
→ STOP_AT_RED / PREPARE_TURN
→ TURNING
→ REACQUIRE_LINE
→ FOLLOW_LINE
```

状态机包含超时、冷却和防重复触发逻辑。`robot/route_manager.py` 使用路线动作描述后续转向，因此比单纯在视觉循环里堆积 `if/else` 更适合竞赛任务扩展。

### 对相似赛题的价值

最值得借鉴：

1. 单相机视觉与电机控制解耦；
2. 路口事件与普通巡线状态分离；
3. 转弯后用视觉重捕线，而不是只靠固定时间；
4. 路线动作通过数据表达，不把全部赛道流程硬编码在一个循环；
5. 冷却机制防止同一路口被连续重复触发。

### 不能直接解决的问题

- 原始赛道不是 25 mm 单黑线；
- 丢线策略偏向停车，没有完整的“保存最后方向并盲驶穿过断线区”；
- 没有 1 m 环岛半圈状态机；
- 没有 RGB 物料到 A/B/C 的业务映射；
- 完整默认返回逻辑使用编码器/里程计，因此不能把整个项目标为严格纯视觉；
- 路线动作不等于仅靠单目黑线完成的原路返回路径记忆。

### 复用建议

- **可以研究并复用架构和部分 MIT 代码**；
- 优先看 `state_machine.py` 和 `turn_controller.py`；
- 不建议照搬颜色阈值和底盘接口；
- 若复制代码，应保留 MIT 版权和许可证声明。

---

## 4.2 ROBOTIS-GIT/turtlebot3_autorace_2020

- 仓库：https://github.com/ROBOTIS-GIT/turtlebot3_autorace_2020
- 核验 commit：`367a24a7228fe9dcd339af50f70b8672df3cc0cc`（`main`）
- 许可证：Apache-2.0（GitHub `licenseInfo`）
- GitHub 描述：TurtleBot3 Autorace 2020 missions。
- GitHub 元数据显示最后更新时间：2026-03-14。
- 推荐级别：**A（颜色检测结构与模式调度）**

### 关键源码

- `turtlebot3_autorace_detect/nodes/detect_traffic_light`
- `turtlebot3_autorace_core/nodes/traffic_light_core_mode_decider`
- `turtlebot3_autorace_detect/nodes/detect_intersection`

### 实际实现机制

#### HSV 颜色检测

交通灯检测代码执行类似流程：

```text
BGR 图像
→ HSV
→ 红/黄/绿色域 inRange
→ Blob/轮廓提取
→ 面积和圆度过滤
→ 连续帧确认
→ 发布颜色状态
```

关键价值不只是 `cv2.inRange()`，而是它没有用单帧最大色块直接做任务决定，而是结合面积、形状和时序确认。

#### 任务模式切换

核心节点根据感知结果，在巡线、交通灯、路口等模式之间切换。检测节点只发布事实，模式决策节点决定何时改变车辆任务，这种拆分适合颜色识别后再进入岔路任务。

#### 路口行为

路口模块接收路线或路口命令，再执行左、右或其他动作。它不是完整自由路径规划，而是固定赛道上的任务动作控制。

### 对相似赛题的价值

该仓库最适合作为**颜色检测结构参考**。原始代码检测红、黄、绿交通灯；没有蓝色类别，也没有物料或 A/B/C 映射。把蓝色加入分类、再把红/绿/蓝映射到目标区域，属于目标任务自行实现的适配，不是该仓库已有能力。

可迁移的关键不是 ROS topic 名称，而是：

- 独立颜色 ROI；
- 每种颜色独立 mask；
- 面积和形状门槛；
- 连续 N 帧确认；
- 识别结果与任务模式分离；
- 识别完成后锁存结果。

### 局限

- 依赖 ROS1 和 TurtleBot3 消息；
- 交通灯是红/黄/绿，赛题是红/绿/蓝色块；
- 交通灯通常位于画面上方，物料色块可能位于地面或车前下方，ROI 与面积规律不同；
- 没有环岛和原路返回。

### 复用建议

- **适合参考或迁移颜色检测的算法结构**；
- 不应为了颜色识别把整个 ROS 栈引入轻量项目；
- Apache-2.0 复用时应提供许可证副本、保留适用的版权/归属声明并标示修改；如果原分发包含 NOTICE，再按许可证要求处理 NOTICE 内容。

---

## 4.3 diaoquesang/smartcar2023

- 仓库：https://github.com/diaoquesang/smartcar2023
- 核验 commit：`abce313d298b4b0913e8d6a929f81c50ed766797`（`main`）
- 许可证：MIT（GitHub `licenseInfo`）
- GitHub 描述：第十八届全国大学生智能汽车竞赛百度智慧交通创意组项目，全国一等奖方案。
- GitHub 元数据显示最后更新时间：2025-06-12。
- 推荐级别：**B，仅研究未接通的视觉环岛/返回实验分支**

### 关键源码

- `midfollow.py`
- `follow.py`
- `newfollow.py`

### 实际实现机制

#### 巡线

- 相机画面转换到 HSV；
- 分割赛道颜色；
- 提取线路中心；
- PID 产生底盘控制；
- 特殊结构通过额外条件覆盖普通巡线。

#### 环岛和阶段流程

`midfollow.py` 包含多阶段入环、绕环，以及注释/标志所表达的“返回基地”实验代码。项目通过多个全局阶段标志表达：

- 何时仍按普通巡线；
- 何时进入特殊环岛动作；
- 何时认为已经进入环内；
- 何时切换到后续视觉模式；
- 何时进入代码所称的返回阶段。

关键限制是：核验 commit 的图像回调中每帧执行 `huandao=False`，使调用 `cal()` 的环岛/返回分支不可达；脚本还使用 `Odometry` 却没有相应导入，不能原样运行。因此这里只能确认仓库保留了环岛/返程实验代码，不能确认主流程已接通，更不能确认完整闭环。

#### 特殊区域

`newfollow.py` 等文件对十字、异常振荡或复杂赛道区域进行特殊处理。它体现了竞赛代码常见模式：普通 PID 只处理稳定单线，拓扑事件由任务状态覆盖。

### 对相似赛题的价值

- 适合研究竞赛调试代码如何表达多个阶段标志；
- 可阅读其环岛和“返回基地”分支中的图像处理与状态意图；
- 不应将这些不可达分支当作可运行环岛或返程实现。

### 局限

- 公开脚本依赖 ROS 图像/速度消息和自定义底盘接口；本次核验没有把“雷达依赖”作为该仓库公开脚本的已证实事实；
- 识别的是绿色赛道，不是 25 mm 黑线；
- 大量全局 flag，状态边界和所有权不够清晰；
- 没有本赛道完整的 RGB→A/B/C 三岔闭环；
- README 的获奖背景不能自动证明公开的几个调试脚本完整复现获奖整车；
- 环岛/返回代码在核验 commit 中未接入可达主流程，且存在缺失导入，不能称为可运行原型。

### 复用建议

- 只将它作为**未接通的竞赛阶段和环岛实验代码参考**；
- 不建议复制其全局变量式控制结构；
- 可在 MIT 条件下复用明确独立的算法，但必须移除传感器假设并重新验证。

---

## 4.4 Abaabaxx/UCAR

- 仓库：https://github.com/Abaabaxx/UCAR
- 核验 commit：`30190ad70bde97faa2083ba257bccd856276a19b`（`archive`）
- 许可证：GitHub `licenseInfo` 未发现许可证；本次没有确认到授予复制/分发权的根许可证。
- GitHub 描述：全国大学生智能汽车竞赛讯飞创意组项目。
- 默认分支：`archive`
- GitHub 元数据显示最后更新时间：2026-06-05。
- 推荐级别：**A- 研究价值，C 直接复用价值**

### 关键源码

- `UCAR-FOLLOW-LINE/z国赛巡线/scripts/国赛全流程_左.py`
- `UCAR/src/follow_line/scripts/guosai/guosai_left.py`
- `UCAR/src/state_machine/scripts/all_guosai.py`

### 实际实现机制

该项目是候选中覆盖任务阶段最多的整车竞赛项目之一：

- 视觉边线或赛道检测；
- PID 控制；
- 多阶段环岛流程；
- 避障；
- 冲刺和停车；
- 外层任务状态机；
- 目标检测或分类流程。

部分源码使用十余个状态表达巡线、环岛、障碍和终点；更外层任务脚本还有更长的任务序列。

### 对相似赛题的价值

- 可研究大型竞赛任务如何分阶段；
- 环岛、避障、停车与全流程状态机集中在同一项目，便于理解竞赛整车的控制顺序；
- 可用于列举“哪些行为必须由任务状态管理，而不是视觉 PID 自己处理”。

### 局限

- UCAR 全向底盘与普通差速小车不同；
- 环岛和避障大量依赖激光雷达、里程计和 ROS；
- 没有完整的单目黑线断线方案；
- 未发现明确许可证，不能默认允许复制或分发；
- 项目规模大，硬件和消息接口耦合重。

### 复用建议

- **只研究流程和状态命名，不直接复制源码**；
- 不能把雷达辅助状态逻辑包装成纯视觉方案；
- 若要复用任何代码，应先联系作者确认授权。

---

## 4.5 duckietown/duckietown-intnav

- 仓库：https://github.com/duckietown/duckietown-intnav
- 核验 commit：`74c208fcd99468100b342d5b5624d6d63dbc13ec`（`master`）
- 许可证：GitHub 标记为 `Other`；仓库许可属于 Duckietown 自有条款，不按标准 MIT/Apache 处理。
- GitHub 描述：Intersection Navigation for Duckietown。
- 推荐级别：**A- 路口算法参考；没有环岛实现**

### 关键源码

- `lib-intnav/src/duckietown_intnav/planner.py`
- `lib-intnav/src/duckietown_intnav/controller.py`
- `lib-intnav/src/duckietown_intnav/vcompass.py`
- `ros-intnav/nodes/interface_stopline.py`

### 实际实现机制

#### 路径规划

`planner.py` 中的路径生成逻辑使用 Bézier 曲线生成：

- 直行；
- 左转；
- 右转。

这是固定结构路口的局部轨迹，不是全局地图规划。

#### 路径控制

`controller.py` 对生成轨迹执行跟踪，思路接近局部 pure pursuit：根据目标轨迹点的横向位置和方向产生控制。

#### Visual Compass

`vcompass.py` 通过相邻图像在地平线附近的图像块位移或 SSD 匹配估计视觉旋转。其价值在于：即使暂时没有清晰线路，也能从图像运动得到短时方向变化代理量。

### 对相似赛题的价值

以下只是从其已实现的左/直/右 Bézier 路口轨迹和 visual compass 得出的**迁移启发**，不是该仓库已有功能：

- 曲线路径生成方法可供研究固定几何轨迹模板；
- visual compass 可供研究短时图像旋转代理量；
- 明确的左/直/右局部轨迹适合比较三支路表达方法；
- 路口控制与普通巡线分模式的架构可作为状态切换参考。

该仓库没有 1 m 环岛、RGB→A/B/C 映射或目标赛道返程实现。

### 局限

- 原项目处理 Duckietown 路口，不是环岛；
- 部分入口触发依赖特定环境或人工接口；
- 视觉罗盘会受纹理不足、光照变化和动态遮挡影响；
- 模板轨迹不能替代当前线路重捕和安全边界；
- 许可证不是标准 MIT/Apache，复制前必须核条款。

### 复用建议

- 只借鉴 Bézier 路径和视觉旋转思想；
- 若要使用源码，先确认 Duckietown Software Terms 是否允许目标使用场景；
- 不应把视觉旋转当作长期精确航向计。

---

## 4.6 duckietown/dt-core

- 仓库：https://github.com/duckietown/dt-core
- 核验 commit：`e4b6fc0629b86d7080ac66828ba74113a69c52f1`（`ente`）
- 许可证：仓库包含 Duckietown 自有许可资料；GitHub 元数据不足以将其简单视为标准 MIT/Apache。
- 推荐级别：**A- 架构参考，C 整体迁移**

### 关键源码

- `packages/lane_control/include/lane_controller/controller.py`
- `packages/fsm/src/fsm_node.py`
- `packages/fsm/config/fsm_node/single_robot_indefinite_navigation.yaml`
- `packages/navigation/src/intersection_type_detector_node.py`

### 实际实现机制

#### 车道控制

- 接收车道位姿；
- 使用横向和航向误差控制；
- 对速度和转向进行配置化限制。

#### YAML 状态机

状态和事件通过 YAML 描述，典型任务链包括：

```text
LANE_FOLLOWING
→ DETECT_INTERSECTION_TYPE
→ STOP_SIGN / TRAFFIC_LIGHT
→ INTERSECTION_CONTROL
→ LANE_FOLLOWING
```

每个状态启停不同视觉和控制节点。FSM 本身不负责图像分割，它负责决定当前哪些模块有权工作。

#### 路口识别

路口类型检测与路口控制被拆分。部分方案使用 AprilTag、停止线或环境标识，因此不等同于只从黑线拓扑自动判断三岔。

### 对相似赛题的价值

- 最适合研究复杂任务的状态、事件和模块启停；
- 颜色识别、环岛、岔路和返程可以建模为独立 session；
- 感知事实与控制状态解耦，避免单个检测结果直接写电机。

### 局限

- ROS/Duckiebot 工程规模大；
- 依赖彩色双边车道、AprilTag、交通灯等特定设施；
- 没有 1 m 黑线环岛；
- 没有比赛所需原路返回逻辑；
- 整体迁移会引入远超需求的依赖。

### 复用建议

- 借鉴 YAML FSM 和事件驱动思想；
- 不建议迁移整个 dt-core；
- 许可证必须单独人工确认。

---

# 5. 中优先级项目详解

## 5.1 dctian/DeepPiCar

- 仓库：https://github.com/dctian/DeepPiCar
- 核验 commit：`d01bd627c0bbbf4d167cbfdd9ad86f4741c3496c`（`master`）
- 许可证：GPL-3.0（GitHub `licenseInfo`）
- GitHub 描述：基于 Raspberry Pi、PiCar-V、TensorFlow 和 Coral EdgeTPU 的自动驾驶小车。
- GitHub 元数据显示最后更新时间：2026-07-10。
- 推荐级别：**B+**

### 关键源码

- `driver/code/hand_coded_lane_follower.py`
- `driver/code/traffic_objects.py`
- `driver/code/objects_on_road_processor.py`

### 实际实现机制

#### 车道

手工车道模块大致使用：

```text
HSV 蓝色区域分割
→ Canny
→ Hough 线段
→ 车道线合并
→ 转向角
```

#### 目标与停车

- 目标检测输出类别和框；
- 通过目标框高度或画面占比估计是否接近；
- 红灯和行人可直接把速度设为 0；
- 停车牌使用计时器；
- 使用防重复计数，避免同一个标志连续触发多次。

### 对相似赛题的价值

- 视觉障碍物进入近场后的停车触发；
- “检测到目标”与“目标足够近必须停车”的两级判定；
- 停车保持、恢复和重复触发冷却；
- 目标处理器与车辆控制器的接口分离。

### 局限

- 原目标检测依赖 TFLite/Coral，RGB 色块任务无需这么重；
- 车道是双边蓝色道路，不是单条黑线；
- 没有环岛、三岔和返程；
- GPL-3.0 不适合未经评估直接复制到非 GPL 项目。

### 复用建议

- 研究事件处理和停车去抖；
- 不复制深度学习依赖；
- 若复用 GPL 源码，必须先评估整个发布项目的许可证义务。

---

## 5.2 autorope/donkeycar

- 仓库：https://github.com/autorope/donkeycar
- 核验 commit：`b074a74ce190827e16f16b345dd69f44387f7b64`（`main`）
- 许可证：MIT（GitHub `licenseInfo`）
- GitHub 描述：构建小型自动驾驶车的开源软硬件平台。
- GitHub 元数据显示最后更新时间：2026-07-13。
- 推荐级别：**B+ 工程框架参考**

### 关键源码

- `donkeycar/parts/line_follower.py`
- `donkeycar/parts/behavior.py`
- `donkeycar/parts/object_detector/stop_sign_detector.py`
- `donkeycar/vehicle.py`

### 实际实现机制

#### Line Follower

- 将图像转换到 HSV；
- 根据配置颜色范围生成 mask；
- 在 ROI 中统计目标像素分布；
- 找到线路中心；
- 使用 PID 控制转向；
- 根据线路可见性和误差控制速度。

#### Vehicle Pipeline

DonkeyCar 的核心价值是 `Vehicle` 管线：摄像头、视觉、行为、控制和执行器可以作为独立 part 串联，并按条件运行。

#### 行为与目标

- `behavior.py` 支持行为状态；
- 停车牌检测器可以把目标检测结果转为停车行为。

### 对相似赛题的价值

- 适合参考模块化车辆主循环；
- HSV 单线追踪适合作为简单基线；
- 行为状态可以承载颜色结果或任务模式；
- MIT 许可证相对友好。

### 局限

- 没有赛题专用直角、断线、环岛和三岔逻辑；
- 原框架大量场景偏向 Ackermann 转向和端到端学习；
- 若只需要少量 OpenCV 模块，引入整个框架成本过高。

### 复用建议

- 参考 `Vehicle` part 化思想和 `line_follower.py` 的配置接口；
- 不建议仅为了 PID 巡线整体迁移 DonkeyCar。

---

## 5.3 DivyamArora22/RCJLine_Follower

- 仓库：https://github.com/DivyamArora22/RCJLine_Follower
- 许可证：未发现。
- GitHub 描述：Raspberry Pi + 相机 + OpenCV 巡线小车。
- GitHub 元数据显示最后更新时间：2020-10-30。
- 推荐级别：**B 算法思想，D 代码复用**

### 关键源码

- `LineFollower/line_functions.py`
- `LineFollower/greenlf.py`

### 实际实现机制

- 检测绿色方向标志；
- 在绿色标志周围采样黑线分布；
- 根据黑线位于标志的哪个方向判断左转、右转或 U 转；
- 普通黑线存在多个轮廓时，结合图像底边和上一目标位置选择连续路线；
- 识别方向后执行固定转向动作，再恢复巡线。

### 对相似赛题的价值

- 方向色块和附近黑线联合判断，可用于环岛左/右标志；
- 上一帧目标连续性可用于减少多路径跳支；
- 对 RoboCup Junior 类复杂线路场景具有直接参考意义。

### 局限

- 使用固定时间转向和阻塞式动作；
- 没有环岛半圈和出口确认；
- 没有红绿蓝物料分类；
- 未发现许可证，不能直接复制。

### 复用建议

- 只学习“方向标志不能脱离附近线路拓扑解释”的思想；
- 将阻塞式转向替换为状态机和视觉闭环；
- 不复制无许可证源码。

---

## 5.4 Adilnasceng/Line-follower

- 仓库：https://github.com/Adilnasceng/Line-follower
- 许可证：未发现。
- GitHub 描述：Raspberry Pi/PiCamera 黑线巡线、交叉口检测和预定义路线 PID 控制。
- GitHub 元数据显示最后更新时间：2026-05-21。
- 推荐级别：**B-**

### 关键源码

- `Line-follower.py`

### 实际实现机制

- 使用 `minAreaRect` 获取线路轮廓宽度和方向角；
- 普通路段根据位置和角度 PID 控制；
- 轮廓宽度突然增大时判断可能进入交叉区域；
- 使用 `intersection_count` 和预设路线数组决定直行、左转或右转；
- 交叉口动作通常以定时或固定命令完成。

### 对相似赛题的价值

- 宽结构可作为“进入交叉拓扑”的廉价候选信号；
- 交叉口计数加路线动作适合固定赛道；
- 位置误差和线路角度联合控制比纯质心更稳定。

### 局限

- 宽轮廓不等于三岔，阴影、环岛、横线和粘连都可能造成误触发；
- 没有 A/B/C 三条候选的独立几何确认；
- 预设计数容易在漏检或重复检测后永久错位；
- 固定 `sleep` 或阻塞动作不适合安全闭环；
- 未发现许可证。

### 复用建议

- 宽度突变只可作为候选触发，不能成为唯一三岔判据；
- 只参考算法思想，不复制代码。

---

## 5.5 aryan-02/line_follow

- 仓库：https://github.com/aryan-02/line_follow
- 许可证：未发现。
- GitHub 描述：OpenCV、Raspberry Pi 和 Pi Camera V2 巡线。
- GitHub 元数据显示最后更新时间：2024-01-28。
- 推荐级别：**B-**

### 关键源码

- `lf.py`
- `line_functions.py`
- `motor.py`

### 实际实现机制

- 黑色阈值分割；
- 腐蚀和膨胀清理 mask；
- 使用 `minAreaRect` 同时估计横向位置和线路角度；
- 多轮廓时根据上一帧位置选择连续目标；
- 包含超声波测距函数。

### 对相似赛题的价值

- 对 25 mm 黑线可建立简单而透明的基线；
- 位置与角度联合控制适合弯道；
- 上一帧连续性比每帧直接选择最大轮廓更稳；
- 多轮廓选择思想可用于直角或岔路前的候选保持。

### 局限

- 丢线恢复分支不完整；
- 超声波障碍函数没有形成完整任务状态机；
- 强绑定 PiCamera、GPIO 和 L298N；
- 没有环岛、颜色和返程；
- 未发现许可证。

### 复用建议

- 只研究 `minAreaRect` 角度和上一帧目标关联；
- 不将最大轮廓作为岔路最终选路依据；
- 不复制无许可证代码。

---

# 6. 低优先级与专项参考项目

## 6.1 GarvitTech/LINE_FOLLOWER_CAR

- 仓库：https://github.com/GarvitTech/LINE_FOLLOWER_CAR
- 许可证：未发现。
- GitHub 描述：Raspberry Pi、OpenCV、PID 和 MPU6050 的高速视觉巡线。
- GitHub 元数据显示最后更新时间：2026-01-09。
- 推荐级别：**C+**

### 关键源码

- `vision_processor.py`
- `main_controller.py`
- `motor_controller.py`
- `imu_filter.py`

### 实现机制

- 小 ROI 中进行自适应阈值；
- 计算线路像素质心；
- PID 加前馈控制；
- 横向误差很大时切换到 90°急转模式；
- 使用 MPU6050 辅助完成闭环转向。

### 价值

- 适合研究光照变化下的简单自适应阈值；
- 展示普通 PID 与急转模式切换；
- 可用于理解为什么直角弯需要特殊状态。

### 局限

- 不符合严格纯视觉，因为 90°动作依赖 IMU；
- MPU6050 长期 yaw 估计容易漂移；
- 转向循环缺少足够的超时和视觉出口确认；
- 无许可证，不能直接复制。

---

## 6.2 d4n93rS4nY0k/Autonomous-Robotic-Device_DP

- 仓库：https://github.com/d4n93rS4nY0k/Autonomous-Robotic-Device_DP
- 许可证：AGPL-3.0
- GitHub 描述：Raspberry Pi 4 + Arduino Uno 的 OpenCV 巡线机器人。
- GitHub 元数据显示最后更新时间：2024-09-04。
- 推荐级别：**C**

### 关键源码

- `Software/OnlyBlackLine.py`
- `Software/OnlyBlackLine/OnlyBlackLine.ino`

### 实现机制

- HSV 黑线轮廓；
- 形态学处理；
- 根据上一目标位置选择邻近黑线；
- Raspberry Pi 将误差通过串口发送给 Arduino；
- Arduino 负责电机 P 控制；
- 代码中存在定时 `Detour()` 绕障轨迹。

### 价值

- 可参考树莓派视觉与 Arduino 实时电机控制分工；
- 有黑线目标连续性和预设绕障动作原型。

### 局限

- 某些无轮廓路径仍可能使用旧变量，鲁棒性不足；
- `Detour()` 未形成完整主流程闭环；
- 预设绕障不保证能重新捕获赛道；
- AGPL-3.0 许可证要求严格。

---

## 6.3 saumyaranjan1111/Line-Following-Robot-Using-OpenCV-and-Arduino

- 仓库：https://github.com/saumyaranjan1111/Line-Following-Robot-Using-OpenCV-and-Arduino
- 许可证：未发现。
- GitHub 描述：OpenCV 黑线巡线机器人。
- GitHub 元数据显示最后更新时间：2025-12-16。
- 推荐级别：**C**

### 关键源码

- `ImageProcessing_LFR-Saumya Ranjan.py`
- `Arduino Code.txt`

### 实现机制

- 选择最大黑色轮廓；
- 计算误差并发送给 Arduino；
- Arduino 执行差速 PID；
- 没有轮廓时，Python 将误差置零，车辆可能短暂保持直行。

### 价值

无轮廓时保持直行可以作为“断线盲驶”的最小概念演示，但它只是偶然行为，不是合格的断线策略。

### 局限

缺少：

- 断线计时；
- 最后可靠方向锁存；
- 速度降级；
- 重捕连续帧确认；
- 最大盲驶距离；
- 超时停车；
- 路口、环岛、颜色和返程。

因此不能把该项目描述为已实现可靠断线恢复。

---

## 6.4 utiasASRL/vtr3

- 仓库：https://github.com/utiasASRL/vtr3
- 核验 commit：`98c8d8d0eb127cc3ffbb1a05c2152752f5a5fd9e`（`main`）
- 许可证：Apache-2.0（GitHub `licenseInfo`）
- 项目类型：多传感器 Teach & Repeat 3，ROS2/C++；提供双目视觉 pipeline，也支持 LiDAR、Radar 与组合传感器。
- 推荐级别：**C，仅作 Teach & Repeat 理论参考**

### 关键源码

- `main/src/vtr_mission_planning/src/state_machine/states/repeat/plan.cpp`
- `main/src/vtr_mission_planning/src/state_machine/states/repeat/follow.cpp`

### 实现机制

- Teach 阶段记录路径和拓扑经验；
- Repeat 阶段进行定位；
- 按路点或拓扑路径重复行驶；
- 逐步移除已经通过的目标；
- 明确判断重复任务完成。

### 对相似赛题的价值

- 提供高层路径记忆和重复行驶的研究参考；
- 说明 Teach & Repeat 需要路径身份、进度和重定位。

这不等同于黑线断线恢复，也不证明该框架能直接完成赛题中的单目黑线原路返回。

### 局限

- ROS2/C++ 体系很重；
- 视觉 pipeline 通常基于双目并需要 GPU，框架也支持 LiDAR、Radar 等非视觉 pipeline；
- 算力、标定和地图要求远超固定 6 m × 4 m 黑线赛道；
- 不适合直接移植到轻量 Raspberry Pi 黑线车。

### 复用建议

只借鉴“Teach/Repeat、拓扑路点和单调进度”概念。固定赛道更适合轻量任务动作记忆，不应为了返程引入完整 VT&R 系统。

---

# 7. 按赛题模块归纳开源实现

> 本章分两层表述：“代表项目/已有机制”来自已核验外部源码；“调研结论”是跨仓库比较后的分析判断，不代表任何单一仓库已经实现目标赛道方案，也不是本地项目实施规格。

## 7.1 黑线巡线

### 常见方法一：ROI 质心

代表项目：

- DonkeyCar；
- GarvitTech；
- Saumya Ranjan；
- 多个 Raspberry Pi/Arduino 示例。

流程：

```text
ROI
→ 灰度或 HSV 阈值
→ 形态学处理
→ 最大轮廓/像素直方图
→ 质心误差
→ PID
```

优点：简单、低算力、容易调试。
缺点：多分支、直角、环岛和阴影下会选择错误目标；只知道“哪里黑”，不知道路线拓扑。

### 常见方法二：位置和角度联合

代表项目：

- `aryan-02/line_follow`；
- `Adilnasceng/Line-follower`。

通过 `minAreaRect` 或轮廓方向获得线角度，再与横向误差共同控制。它比纯质心更适合弯道，但在断线和分叉处仍需状态机。

### 常见方法三：结构化车道/轨迹

代表项目：

- Duckietown；
- DeepPiCar；
- `duckietown-intnav`。

先恢复地面线段、车道位姿或局部轨迹，再控制。工程性最好，但对标定和环境先验要求较高。

### 调研结论

纯质心 PID 只能作为普通路段底层控制，不能独立承担整场比赛。

---

## 7.2 90°直角弯

公开项目主要采用三类方案：

### 固定时间/固定角速度

代表：Adilnasceng、RCJLine_Follower。
简单但受电压、摩擦和速度影响，不能稳定保证出口捕获。

### IMU/编码器转角

代表：GarvitTech。
比固定时间强，但不属于严格纯视觉，且低成本 IMU yaw 易漂移。

### 状态机 + 转弯后视觉重捕

代表：`duckietown-follow-line-pi`。
这是最适合相似赛题的思路：特殊转弯状态持有电机，直到出口线路稳定出现后才交还普通巡线。

### 调研结论

候选对比表明，仅增大普通 PID 参数没有覆盖出口身份、动作超时和重捕条件；`duckietown-follow-line-pi` 提供了“事件—专用执行—视觉重捕—交接”的更完整公开参考。这里是比较结论，不代表它已在目标 25 mm 黑线直角上验证。

---

## 7.3 断线区

检索到的基础项目大多没有可靠断线策略。常见行为包括：

- 无线时停车；
- 保持上一命令；
- 把误差置零继续直行；
- 向最后看到线路的一侧旋转搜索。

从这些项目暴露的缺口推导，一个待验证的完整断线方案通常还需要：

```text
最后可靠姿态/误差
+ 有限时间或距离预测
+ 降速
+ 最大偏航限制
+ 连续帧重捕
+ 超时停车
```

本次候选中，没有一个轻量基础项目完整实现上述全部契约。Duckietown 类项目提供重捕和 FSM 思路；简单 Arduino 项目只能提供概念性基线。

---

## 7.4 环岛

包含竞赛环岛相关代码的主要候选是：

- `diaoquesang/smartcar2023`：绿色赛道环岛实验分支，但在核验 commit 中被每帧 `huandao=False` 阻断，且脚本存在 `Odometry` 缺失导入；
- `Abaabaxx/UCAR`：多传感器 ROS 竞赛系统，环岛和避障流程可见雷达、里程计等依赖。

因此前者只能证明仓库保留了视觉环岛实验逻辑，后者证明多传感器竞赛 FSM 中存在环岛流程；两者都不是可直接运行的目标纯视觉环岛答案。

`duckietown-intnav` 本身只实现路口导航；其 Bézier 路径和 visual compass 可作为曲线路径研究资料，但不是现成环岛代码。

### 调研结论

没有发现可直接下载并用于“25 mm 黑线、1 m 直径环岛、方向标志、绕半圈指定出口”的成熟纯视觉仓库。公开源码分别提供了视觉环岛阶段、竞赛多传感器 FSM、曲线路径生成和视觉旋转估计等子能力；如何组合属于目标系统的后续设计，不是这些仓库已经共同证明的实现。

---

## 7.5 红绿蓝识别

最佳公开参考是 TurtleBot3 Autorace 的交通灯检测，辅以 DeepPiCar 的目标事件处理。

外部源码已直接证明的结构是：

```text
固定/任务相关 ROI
→ HSV
→ 原仓库中的红、黄、绿 mask
→ Blob/轮廓形状与面积过滤
→ 连续帧计数
→ 颜色事件
```

红色双 Hue 区间、蓝色类别、红/绿/蓝投票窗口和类别锁存都需要针对目标物料另行实现与实测，不能写成 TurtleBot3 原仓库已有能力。DeepPiCar 则证明了视觉目标事件可以与停车计时和防重复触发结合，但其目标分类使用 TFLite/Coral 路径。

---

## 7.6 三岔选路

候选项目提供了三种不同级别的参考：

1. `Adilnasceng/Line-follower`：交叉口宽度 + 路口计数 + 预设动作；
2. `RCJLine_Follower`：方向色块 + 多轮廓连续目标；
3. Duckietown：明确左/直/右任务状态和局部轨迹。

没有候选完整实现“先识别红绿蓝，再在一个三分支黑线岔路中保留 A/B/C 三条候选并闭环跟踪目标支路”。

### 调研结论

这些仓库说明了三种已有技术：宽结构触发、方向标志辅助、多条局部轨迹。它们没有共同证明某一种目标赛道三岔算法；尤其没有仓库直接实现“RGB 结果锁存后验证 A/B/C 目标支路并闭环跟踪”。最大轮廓或最长线在候选源码中只是局部启发式，不能据此宣称已解决该完整任务。

---

## 7.7 障碍停车与避障

### 视觉停车

DeepPiCar 最适合作为参考：目标类别 + 接近度 + 停车计时 + 防重复触发。

### 预设绕障

`Autonomous-Robotic-Device_DP` 有定时绕障动作，但没有证明完整接回赛道。

### 多传感器避障

UCAR 有更完整的多传感器竞赛避障行为，并依赖雷达/里程计等接口；本次没有在 smartcar2023 的公开脚本中核实同等雷达避障闭环。

### 调研结论

从候选成熟度看，视觉停车的公开参考比轻量纯视觉绕障闭环更完整。若目标系统选择绕障，以下能力属于尚待自行设计和验证的缺口，而不是候选仓库已共同实现的功能：

- 障碍边界和可通行侧判断；
- 绕障轨迹；
- 黑线被遮挡期间的路径记忆；
- 绕过后重新捕线；
- 最大动作时间和安全停车。

现有轻量候选没有提供完整可直接复用的纯视觉闭环。

---

## 7.8 原路返回

开源项目中的“返回”分为三种：

1. **不可达的返程实验分支**：smartcar2023 公开脚本中的阶段标志和“返回基地”注释；核验 commit 每帧重置 `huandao=False`，不能称为可运行返程系统；
2. **动作/路线管理**：`duckietown-follow-line-pi` 的 route manager，完整默认返回使用编码器/里程计；
3. **Teach & Repeat**：VT&R3 的多传感器路径记忆与重复，视觉 pipeline 通常为双目/GPU。

对于固定小赛道，是否需要视觉 SLAM 属于后续架构选择。候选代码可提供的启发包括：

- 路线动作必须有身份和阶段；
- 去程识别结果要锁存；
- 返程时某些任务跳过，不应再次分类；
- 环岛返程规则可能不是简单左右取反；
- 已通过路段要有单调进度，避免重复触发。

---

# 8. 许可证与代码使用风险

## 8.1 相对容易复用

| 仓库 | 许可证 | 注意事项 |
| --- | --- | --- |
| `Luissalamanca23/duckietown-follow-line-pi` | MIT | 保留版权和 MIT 文本 |
| `ROBOTIS-GIT/turtlebot3_autorace_2020` | Apache-2.0 | 提供许可证副本、保留适用归属并标示修改；原分发含 NOTICE 时再处理 NOTICE |
| `diaoquesang/smartcar2023` | MIT | 保留版权和 MIT 文本 |
| `autorope/donkeycar` | MIT | 保留版权和 MIT 文本 |
| `utiasASRL/vtr3` | Apache-2.0 | 保留许可证和通知 |

## 8.2 强 copyleft

| 仓库 | 许可证 | 风险 |
| --- | --- | --- |
| `dctian/DeepPiCar` | GPL-3.0 | 分发衍生作品时通常需按 GPL 开源 |
| `d4n93rS4nY0k/Autonomous-Robotic-Device_DP` | AGPL-3.0 | 若用户通过网络与修改后的 AGPL 覆盖程序交互，AGPL 第 13 条通常要求向这些用户提供对应源码 |

## 8.3 条款需人工确认

- `duckietown/duckietown-intnav`
- `duckietown/dt-core`

这些项目不应仅凭“GitHub 可访问”就视为标准宽松开源。

## 8.4 未发现许可证，不应复制

- `Abaabaxx/UCAR`
- `DivyamArora22/RCJLine_Follower`
- `Adilnasceng/Line-follower`
- `aryan-02/line_follow`
- `GarvitTech/LINE_FOLLOWER_CAR`
- `saumyaranjan1111/Line-Following-Robot-Using-OpenCV-and-Arduino`

可以阅读、比较和重新实现思想，但不应直接复制其源码进入项目。

---

# 9. 面向后续 Agent 的使用指南

后续 Agent 使用本调研时，应遵守以下规则。

## 9.1 不要声称存在完整现成方案

正确表述：

> 没有发现单仓完整覆盖整条赛道；需要组合借鉴多个项目，并重新实现赛题专用状态机。

错误表述：

> smartcar2023 或 UCAR 可以直接完成本赛道纯视觉任务。

它们都包含明显的硬件、传感器和场景差异。

## 9.2 不要把“源码公开”误写成“可自由复制”

未声明许可证的仓库只可作为研究资料。任何 Agent 推荐复制前，必须先检查仓库根许可证和目标文件版权头。

## 9.3 不要只看 README

如果后续需要进一步使用某个仓库，应继续核对：

1. 关键函数真实输入输出；
2. 状态机是否真的被主程序调用；
3. README 宣称的功能是否接入主流程；
4. 是否依赖隐藏话题、雷达、编码器、IMU 或预训练模型；
5. 许可证是否覆盖目标源码目录；
6. 默认分支是否仍包含被引用文件。

## 9.4 推荐的进一步阅读顺序

### 若任务是颜色识别

1. TurtleBot3 Autorace 的 traffic light detector；
2. DeepPiCar 的 object event processor；
3. Duckietown FSM 的模式切换。

### 若任务是三岔选路

1. `duckietown-intnav` 的 planner/controller；
2. `duckietown-follow-line-pi` 的 route manager/state machine；
3. RCJLine_Follower 的多轮廓连续性；
4. Adilnasceng 的交叉口候选触发。

### 若任务是环岛

1. smartcar2023 的阶段流程；
2. UCAR 的竞赛 FSM；
3. `duckietown-intnav` 的曲线路径和视觉方向估计。

### 若任务是障碍停车

1. DeepPiCar 的目标距离触发；
2. TurtleBot3 的模式决策；
3. UCAR 的复杂避障仅作为多传感器对照。

### 若任务是返程

1. smartcar2023 的不可达返程实验分支；
2. `duckietown-follow-line-pi` 的路线动作和编码器/里程计返回；
3. VT&R3 的多传感器路径身份和单调进度概念。

---

# 10. 最终调研判断

## 10.1 最可直接借鉴的代码

在许可证和轻量化方面，优先级最高的是：

1. `Luissalamanca23/duckietown-follow-line-pi` 的状态机、转弯重捕和路线动作；
2. `ROBOTIS-GIT/turtlebot3_autorace_2020` 的 HSV 颜色检测和连续帧确认；
3. `diaoquesang/smartcar2023` 的未接通环岛实验代码（不能直接复用为运行方案）；
4. `autorope/donkeycar` 的车辆管线和基础视觉巡线。

## 10.2 最值得借鉴但不宜复制的代码

1. `Abaabaxx/UCAR`：完整竞赛 FSM，但无许可证且依赖多传感器；
2. `RCJLine_Follower`：方向标志和多轮廓连续性，但无许可证；
3. `Adilnasceng/Line-follower`：交叉口触发和路线计数，但无许可证；
4. `aryan-02/line_follow`：线路角度和上一目标关联，但无许可证；
5. Duckietown 自有条款项目：应先核准授权范围。

## 10.3 现有开源证据不足以支持的做法

下列做法在本次候选中只有简化示例、局部启发式或没有完整闭环证据，不能仅凭这些仓库宣称已经可靠解决：

- 用单个最大轮廓完成整条赛道；
- 用普通 PID 硬冲 90°直角；
- 把无线时误差置零当作完整断线恢复；
- 仅按固定时间绕环岛；
- 用颜色结果直接生成转向，不检查目标支路是否存在；
- 把去程动作简单全部取反当作原路返回；
- 为识别三种规则色块引入完整 YOLO/Coral 系统；
- 因为仓库在 GitHub 公开就直接复制无许可证代码。

## 10.4 一句话结论

最现实的开源利用方式是：

> 公开源码中，`duckietown-follow-line-pi` 提供较完整的单目巡线/路口状态机，TurtleBot3 Autorace 提供红黄绿 HSV 时序检测，UCAR 提供多传感器竞赛 FSM，`duckietown-intnav` 提供路口轨迹；smartcar2023 只保留未接通的环岛/返回实验分支。没有项目直接提供目标赛道所需的纯视觉环岛、RGB→A/B/C 三岔闭环和规则化返程。
