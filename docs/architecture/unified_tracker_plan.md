# 统一巡线控制器改造方案(Implementation Doc)

状态:已实施(Phase 0-3 落地)。Phase 4 实机标定待现场进行。
基线:branch `unified-tracker`,anchor commit 保存了改造前的旧状态机。
本文既是设计说明,也是标定与回退参考。

## 1. 动机

当前实现的问题(来自 2026-07-09 code review):

1. 拐点被假设为"直角",走 branch 检测 → `TIMED_FORWARD(5s 盲走)` →
   `TIMED_TURN(定时开环转)` → `REACQUIRE`。实际赛道拐点不一定是直角,
   开环定时动作对角度、速度、地面摩擦都敏感,标定成本高且脆。
2. 虚线走独立 `GAP_BLIND` 状态;blind 超时后进入搜索,若丢线时
   `last_err_norm≈0` 则 `v=0, w≈0` 原地死锁(`state_machine.py:83`)。
3. `REACQUIRE` 无超时兜底,转不回线就永远原地转。
4. 圆环没有任何专门能力,靠局部纠偏碰运气。
5. `err_norm` 用 `crop_center` 归一化,但 crop 左扩 20px / 右扩 140px,
   左右不对称,同样物理偏差产生不同控制量(`vision.py:136`)。
6. debug app 的 `_apply_profile()` 直接改全局 CONFIG 并
   `_write_config_file()` 持久化,污染 canonical 配置
   (当前 `race_config.json` 里 `right_turn_dir=1.0` 即此产物)。
7. 当前保存的 crop `[336,391,429,415]` 只有 24px 高,5 个 band 各 4-5px,
   纵向特征(分支、朝向)完全失效,疑似调试误存。

核心结论:**缓弯、锐弯(含直角)、虚线、圆环在几何上都是"线还在视野里,
只是位置/朝向/连续性变化",应该由一个更强的连续跟踪器统一处理,
而不是各配一个离散状态。** 离散状态只保留物理上必须停/必须搜的情形。

## 2. 目标状态机(3 态)

```
TRACK   —— 唯一的行驶状态:连续跟踪线,内部无子状态
LOST    —— 线置信度低于阈值且持续超过 T_lost:朝最后已知方向摆动搜索,
           带超时(t_search_max)→ STOPPED
STOPPED —— 障碍/急停/搜索超时/外部命令
```

删除:`GAP_BLIND`、`TIMED_FORWARD`、`TIMED_TURN`、`REACQUIRE`。
`corner.*` 与 `gap.*` 配置段整体废弃(见 §7 迁移)。

## 3. Vision 升级:从"单点误差"到"轨迹估计"

### 3.1 输入

- 恢复大视野 crop(参考旧默认 `(300,265,430,455)`,高约 190px);
  加载配置时校验 crop 高度 ≥ band_count × 12px,不满足打警告并拒跑。
- band_count 提到 7~9(当前 5),每 band 依然产出 runs。

### 3.2 每 band 输出:中心 + 置信度

对每个 band 的 best run 计算:

```
conf_band = f(面积是否达标, 宽度 vs 期望线宽的偏差, 与上一帧该 band 中心的跳变)
∈ [0, 1]
```

虚线的表现就是若干 band conf=0 —— 不需要任何"虚线状态",
拟合时这些 band 自然缺席。

### 3.3 加权轨迹拟合

对有效 band 的 `(y_i, cx_i, conf_i)` 做加权一次拟合(必要时二次):

```
cx(y) = a + b·y (+ c·y²)
```

导出三个量(均在拟合层做对称归一化,修复问题 5):

- `e0`   :底部延伸处的横向误差,除以 **crop 半宽**(不是 crop_center)
- `theta`:线相对车体的朝向 ≈ atan(b)
- `kappa`:曲率 ≈ 2c(启用二次项时);圆环 = 恒定 kappa,
  直角 = 局部 kappa 很大 / 上部 band 中心剧烈偏移

拟合整体置信度:

```
conf = Σ conf_i 加权(底部 band 权重更高) / 归一化
```

### 3.4 时间滤波(替代 GAP_BLIND)

对 `(e0, theta, kappa)` 做 alpha-beta(或 EMA + 速率外推)滤波:

- conf 高:滤波器正常更新。
- conf 低(虚线间隙、短暂遮挡):**滤波器进入预测模式**,
  按上一帧状态和车速外推,控制器继续用预测值行驶——
  这就是原来 GAP_BLIND 想做的事,但连续、无状态切换、有方向记忆。
- 预测模式下 conf 以固定速率衰减;衰减到 `conf_lost`(约 0.15)
  且持续 `T_lost`(约 0.8s)→ 才切 LOST。

## 4. 控制律(TRACK 内唯一控制器)

```
w = k_e·e0 + k_theta·theta + k_ff·kappa        # 反馈 + 曲率前馈
w = clip(w, ±max_w)
v = v_max · g(|w|/max_w) · h(conf)             # 越弯越慢、越不确定越慢
```

- `g`:速度-转向耦合,如 `g(x) = 1 - slow_gain·x`,下限 `v_min_ratio`。
- `h`:置信度降速,conf 预测模式下额外乘 0.6~0.8。
- **Pivot assist(替代定时直角转)**:当 `|e0| > e_pivot` 或
  `|theta| > theta_pivot` 时,`v → v_pivot(≈0)`,w 保持同一控制律输出的
  符号、幅值取 `w_pivot`。这是连续控制器的饱和分支,不是独立状态:
  条件一旦不满足立即回到正常增益,无需 reacquire 确认帧。
  锐弯/直角 = 上部 band 大偏移 → theta 大 → 自动减速原地对准 → 继续走。
  任意角度的拐点都被同一机制覆盖。
- 圆环:恒定 kappa 由前馈项直接吃掉,e0 反馈只做残差修正;
  进出环岛的分叉选择(比赛要求的 A/B/C 岔路)后续用"偏置项"实现:
  `e0_bias = ±bias_amount`,让控制器主动贴向内/外侧线——仍是同一控制器。

### LOST 行为(修复死锁)

```
w = sign_hint · w_search      # sign_hint = 滤波器最后的 theta/e0 符号;
                              # 若两者都≈0,固定取上一次非零符号,兜底 +1
v = 0
超时 t_search_max(约 3s)→ STOPPED
```

保证 w 永不为 0,且必然终止。

## 5. 配置改动

```
删除: corner.*(整段), gap.*(整段)
新增: tracker:
  band_conf_min_area, expected_line_width_px(或沿用中值估计)
  k_e, k_theta, k_ff
  v_max, v_min_ratio, slow_gain
  filter_alpha, filter_beta, conf_decay, conf_lost, T_lost
  e_pivot, theta_pivot, v_pivot, w_pivot
  t_search_max, w_search
保留: camera.*, vision.*(band/threshold 部分), line.invert_turn
```

`err` 归一化统一为除以 crop 半宽,`invert_turn` 语义不变。

## 6. Debug app / runner 配套修改

1. **修复配置污染**:`_apply_profile()` 不再改 `CONFIG` 也不再写
   `race_config.json`;改为生成 overlay dict,序列化为
   `configs/race_config.live.json` 下发,runner `--config` 指向它。
   canonical config 只被"保存默认"按钮显式写入。
2. profiles 简化:直角/虚线单测按钮删除(不再有对应状态),
   保留 `line`(=TRACK 全功能)与 `final`;新增"限速"滑条方便实测。
3. runner 删除 `--force-turn-dir` / `--stop-after-corner`
   (依附于旧状态机的旁路补丁);日志字段增加
   `e0/theta/kappa/conf/mode(track|predict|pivot)`。
4. overlay 增画:拟合曲线、各 band conf(颜色深浅)、pivot 触发阈值线。

## 7. 迁移步骤(每步可独立验证、可回退)

- **Phase 0 — commit 现状**:当前全部 untracked 代码入库;
  清理 `tests/__pycache__` 中无源文件的 stale pyc。
- **Phase 1 — 纯 bug 修复(不动架构)**:
  归一化对称化;LOST 搜索最小角速度 + 超时;profile 不写 canonical config;
  crop 高度校验。跑现有 7 个测试保持绿。
- **Phase 2 — vision 拟合层**:新增 `fit_line_trajectory()`,
  输出 `(e0, theta, kappa, conf)`;旧接口暂存共用;
  在 debug app 单帧诊断 + 合成图像测试上验证
  (直线/斜线/圆弧/虚线/直角五类 mask)。
- **Phase 3 — 新控制器 + 3 态状态机**:实现滤波器与控制律,
  删除四个旧状态及其配置;测试改为"帧序列回放":
  合成一段虚线序列、一段 90° 拐角序列、一段圆弧序列,
  断言全程 state==TRACK、v>0 占比、e0 收敛。
- **Phase 4 — 实机标定**:先 `line` profile 定 `k_e/k_theta/v_max`,
  再用直角赛段定 `e_pivot/theta_pivot/w_pivot`,
  再虚线段定 `filter_*/conf_*`,最后圆环定 `k_ff`。
  每步用 runner JSON 日志离线回看。

## 8. 风险与对策

- 二次拟合在 band 少/虚线稀疏时不稳 → 有效 band < 4 时退化为一次拟合,
  kappa 沿用滤波器预测值。
- Pivot assist 在阈值附近抖振 → 进入/退出阈值加 10~15% 迟滞。
- 大视野 crop 会看到相邻赛道线 → runs 选择沿用"距预测中心最近 + 面积加分",
  预测中心来自滤波器而非固定 crop_center(比现状更抗干扰)。
- 圆环出入口分叉误导拟合 → 拟合用 RANSAC-lite:残差最大的 1 个 band
  可剔除一次后重拟合。
