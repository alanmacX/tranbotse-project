# 开源方案与技术转向参考清单

## 一句话结论

- [x] **不彻底迁移技术栈。**
- [x] 保留 `OpenCV 感知 + 显式状态机 + 差速控制`。
- [x] 从“统一算法处理所有赛道形状”收敛为“固定赛道状态图 + 专用事件执行器”。
- [ ] 优先补齐颜色识别、三岔选择、卸载停车和原路返回，而不是继续扩张角点/环岛通用几何逻辑。

---

## 赛道判断

- 固定 6 m × 4 m 场地。
- 固定 25 mm 黑线。
- 任务节点和顺序已知。
- 不需要未知地图探索。
- 不需要自由路径规划。
- 不需要 SLAM/Nav2。
- 返回路线可由去程事件逆序执行，不需要记录像素级路径。

推荐任务链：

```text
起点
→ 障碍
→ 90°弯
→ 断线
→ 环岛入口
→ 环岛半圈
→ 环岛出口
→ 颜色识别
→ 三岔选路
→ 卸载停车
→ 原路返回
→ 起点停车
```

---

## 当前实现：保留清单

- [x] 自适应黑线分割与反光/噪声过滤。
- [x] 多扫描带、连通域和轨迹拟合。
- [x] 普通循线、短断线预测、持续丢线搜索和安全停车。
- [x] 相机到车轴距离补偿。
- [x] 90°弯出口线锁存和视觉重获。
- [x] 环岛骨架选路和前视目标控制。
- [x] 环岛路径跳变抑制及短时缺帧保持。
- [x] 固定关卡顺序和 detector gating。
- [x] 调试录像、telemetry 和手动驾驶后备工具。

主要入口：

- `apps/race_runner.py`
- `transbot_race/vision.py`
- `transbot_race/state_machine.py`
- `transbot_race/path_memory.py`
- `transbot_race/ring_entry.py`
- `transbot_race/mission.py`

---

## 当前实现：缺口清单

- [ ] 默认配置从障碍物关卡开始。
- [ ] 正式启用并实车验证障碍检测。
- [ ] 灰色物料检测区停车或减速。
- [ ] 红、绿、蓝颜色识别。
- [ ] 多帧颜色投票和结果锁存。
- [ ] `红→A / 绿→B / 蓝→C` 映射。
- [ ] 三岔拓扑检测。
- [ ] 目标分支锁定和控制。
- [ ] 卸载区到达判定与停车。
- [ ] 返回前重新定向。
- [ ] 去程事件记录和逆序执行。
- [ ] 返回环岛、断线、直角弯和障碍处理。
- [ ] 起点识别和最终停车。
- [ ] 完整赛道实车闭环测试。

现状注意：

- `FORK` 当前仅使用普通巡线。
- `RETURN` 当前仅使用普通巡线。
- `fork_branch` 尚未形成完整分支执行逻辑。
- 当前没有颜色分类器。
- 当前没有完整返回路线执行器。

---

## 推荐目标架构

### 1. 基础视觉

仅输出稳定事实：

```text
near_error
heading
confidence
line_present
left_branch
straight_branch
right_branch
```

### 2. 普通巡线控制器

只负责：

- 连续黑线；
- 短断线预测；
- 丢线搜索；
- 安全停车。

### 3. 专用执行器

建议保持独立：

```text
CornerExecutor
RingExecutor
ForkExecutor
TerminalExecutor
```

### 4. 颜色分类器

推荐：

```text
固定曝光/白平衡
→ 物料 ROI
→ HSV 或 LAB 分割
→ 面积过滤
→ 5–10 帧投票
→ 锁存 RED/GREEN/BLUE
→ 映射目标分支
```

不需要神经网络。

### 5. 固定任务状态图

建议状态：

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

---

## 开源项目参考清单

### 优先参考

- [SoniDavid/TE3002B_mcrChallenge](https://github.com/SoniDavid/TE3002B_mcrChallenge)
  - 参考：五带扫描、虚线迟滞、两阶段丢线搜索、90°动作。
  - 注意：未发现明确许可证，只借鉴思想并自行实现。

- [Hiwonder/LanderPi](https://github.com/Hiwonder/LanderPi)
  - 参考：多 ROI 加权质心、最大轮廓、轻量 PID。
  - 用途：可作为当前视觉算法的简单对照基线。

- [osrf/rosbook](https://github.com/osrf/rosbook)
  - 参考：最小扫描带质心控制。
  - 许可证：Apache-2.0。
  - 用途：仅作硬件连通性和普通直线基线。

- [ROBOTIS TurtleBot3 AutoRace](https://github.com/ROBOTIS-GIT/turtlebot3_autorace)
  - 参考：滑窗、可靠度、任务节点切换。
  - 许可证：Apache-2.0。
  - 注意：双车道模型与本赛道单黑线不同，不整体移植。

- [Robo-mat/robocup2025-Jonson](https://github.com/Robo-mat/robocup2025-Jonson)
  - 参考：断线、色标、障碍及任务状态切换。
  - 注意：未声明许可证。

- [xelisce/robawt2023](https://github.com/xelisce/robawt2023)
  - 参考：远端 ROI 跨缺口、障碍后重获线。
  - 注意：未声明许可证。

- [Duckietown/dt-core](https://github.com/duckietown/dt-core)
  - 参考：感知事件、任务状态和动作执行解耦。
  - 注意：借鉴 FSM，不迁移完整 Docker/ROS 栈。

- [Maze-solving-line-following-robot](https://github.com/WithorwithoutU/Maze-solving-line-following-robot)
  - 参考：`L/R/S/U` 事件栈和路线逆序思想。
  - 许可证：MIT。
  - 注意：传感器车方案，只借鉴数据模型。

- [Yahboom TransbotSE](https://github.com/YahboomTechnology/TransbotSE)
  - 参考：底盘协议和硬件接口。
  - 注意：官方视觉主要是最大轮廓和 PID，不足以替代当前实现。

### 开源复用规则

- [x] Apache/MIT/BSD 项目可在核对具体文件后复用。
- [ ] 无 LICENSE 项目不要直接复制代码。
- [x] 无许可证项目只研究算法思想并自行重写。

---

## 不推荐迁移清单

- [ ] **ROS 2 整体迁移**：通信和录包更强，但不直接改善视觉与控制成功率。
- [ ] **Nav2/SLAM**：赛道已给出路径，属于重复求解；运动反馈也不足以支撑高精度定位。
- [ ] **Duckietown 完整栈**：双车道和标定假设不同，依赖过重。
- [ ] **DonkeyCar/端到端学习**：需要大量数据，对光照和相机位姿变化敏感。
- [ ] **F1TENTH**：Ackermann、LiDAR、VESC 与 Transbot 差速底盘不匹配。
- [ ] **行为树替换当前 FSM**：任务严格线性，显式状态机更容易调试。
- [ ] **OpenMV 硬件替换**：可借鉴多 ROI 思想，但没有必要更换现有 Pi/OpenCV。

---

## 应停止扩张的方向

- [ ] 不再尝试让一个统一控制器自然处理直线、直角、环岛和三岔。
- [ ] 不把颜色、三岔、卸载和返回继续塞进角点控制器。
- [ ] 不把命令速度积分当作精确里程。
- [ ] 不仅依靠固定距离或固定时间完成转弯。
- [ ] 不在缺少真实失败录像时继续增加几何阈值和状态分支。

控制原则：

- 视觉重获是动作成功条件；
- 距离和角度只作为安全边界；
- 当前任务状态负责解释几何；
- 每一帧只允许一个控制器拥有底盘。

---

## 实施优先级

### P0：完成计分闭环

- [ ] 颜色识别。
- [ ] 颜色到 A/B/C 映射。
- [ ] 三岔选路。
- [ ] 卸载停车。
- [ ] 返回任务图。
- [ ] 起点停车。

### P1：真实数据闭环

每个场景采集：

- [ ] 白天和夜间。
- [ ] 补光开和关。
- [ ] 高电量和低电量。
- [ ] 左环和右环。
- [ ] 红、绿、蓝物料。
- [ ] 去程和回程。
- [ ] 成功和失败片段。

离线验证：

```text
视频帧
→ 感知结果
→ 状态转移
→ 电机命令
```

### P2：基于证据简化

仅在真实数据证明以下问题时替换视觉方案：

- [ ] 帧率持续不足。
- [ ] 骨架路径频繁跳变。
- [ ] 反光导致大面积误检。
- [ ] CPU 占用破坏控制周期。

候选简化方案：

```text
3–5 条 ROI
+ 连通域连续性
+ 加权质心
+ 专用路口 detector
```

---

## 彻底转向的触发条件

只有满足至少一项，才重新评估技术栈：

- [ ] Pi 上不能稳定达到约 12–15 FPS。
- [ ] 固定曝光和相机安装后，环岛路径仍无法稳定识别。
- [ ] 传统视觉在真实回放评测集上仍无法达到可接受识别率。
- [ ] 比赛改成未知地图、随机障碍或自由导航。
- [ ] 硬件增加可靠编码器、LiDAR 和 IMU，并要求多点导航。
- [ ] 已有覆盖全部光照和姿态的大规模驾驶数据，学习方法在统一测试集上明显胜出。

---

## 最终决策表

| 部分 | 决策 |
|---|---|
| OpenCV 感知 | 保留 |
| 普通循线状态机 | 保留 |
| 角点出口锁存 | 保留 |
| 环岛选路执行器 | 保留并冻结复杂度 |
| 固定任务 session | 保留并补全 |
| 颜色识别 | 新增 |
| 三岔执行器 | 新增 |
| 卸载与返回 | 新增 |
| ROS/SLAM | 不迁移 |
| 端到端学习 | 不作为主控 |
| 当前总体策略 | 局部收敛，不推倒重写 |

## 推荐下一步

> 先完成 `颜色 → 三岔 → 卸载 → 返回 → 起点停车` 的最小闭环，然后用真实赛道录像和实车重复成功率决定是否简化已有角点、环岛算法。
