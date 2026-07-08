# 今晚计划解读与执行顺序

基于 `/Users/macalan/Downloads/transbot_tonight_debug_plan.md`。

## 总判断

这份计划方向是对的：不要推翻当前 Web Demo，也不要一上来写完整比赛状态机。今晚应该围绕“可观察、可调参、可回退”推进。

核心路线：

```text
先把检测结果标准化成 features
-> 在 Web 上实时看 features/overlay
-> 再做直角、虚线、障碍、环岛这些状态机模块
```

也就是说，今晚第一优先级不是“车跑得更远”，而是“每个判断为什么发生都能看见”。

## 建议执行顺序

### 1. 保住当前稳定功能

不要删：

```text
视频流
相机/机械臂调参
crop
检测一次
巡线 Demo
手动底盘/空转
停车
```

任何新功能都加在新 Tab 或新 section 里。

### 2. 先做 Debug 总览，而不是先做直角

先落地：

```python
features, line_mask, debug_frame = detect_line_features(frame, params)
```

至少输出：

```text
found
confidence
cx
target_center
err
err_norm
area
bbox
wide_ratio
left_area/right_area/top_area/bottom_area/center_area
num_contours
```

原因：直角、虚线、障碍、环岛本质上都依赖这些 features。没有统一 features，后面会到处复制视觉逻辑。

### 3. Web 先加 Tab 框架

推荐顺序：

```text
原有控制
Debug总览
直角弯
虚线短线
障碍检测
环岛入口
环岛绕行
参数管理
```

今晚不一定每个 Tab 都做满，但 Tab 框架要先有，参数别堆在一个巨长页面里。

### 4. 第一个状态机：直角弯

不要看到直角马上转。按计划里的原则：

```text
检测点 != 动作点
```

应该是：

```text
NORMAL_LINE
-> RIGHT_ANGLE_DETECTED
-> PRE_FORWARD
-> TURNING
-> REACQUIRE
-> NORMAL_LINE
```

关键参数：

```text
right_turn_pre_forward_sec
right_turn_pre_forward_v
right_turn_w
right_turn_min_sec
turn_finish_err_th
```

### 5. 第二个状态机：虚线/短线

当 `found=false` 不要立刻停，进入：

```text
DASHED_BLIND
```

短时间低速盲驶：

```text
blind_time
blind_v
blind_w_keep_ratio
```

重新看到线并连续确认后回 NORMAL。

### 6. 障碍检测

先做纯视觉停车，不碰深度学习：

```text
非白地面 + 非黑线 + 大块轮廓
```

用 confirm frames 防误报：

```text
obstacle_area_th
obstacle_confirm_frames
```

### 7. 环岛

先支持手动方向：

```text
manual_left
manual_right
```

`auto_vlm` 之后再接。VLM 只判断方向牌，不参与实时控制。

环岛也必须有 margin：

```text
round_entry_pre_forward_sec
round_entry_pre_forward_v
round_entry_v
round_entry_w
round_entry_sec
round_bias_w
min_round_time
```

## 右拐问题的 debug 判断

如果车一直向右拐，优先判断：

1. 手动左转/右转是否符合预期。
2. `features.err` 是否稳定偏一侧。
3. overlay 里红线 `cx` 是否真的在黑线上。
4. 如果手动方向正常，但巡线方向反，切换 `line_invert`。
5. 如果 `cx` 不在黑线上，是识别问题，调 crop/threshold/features。

## 今晚不要做的事

```text
不要重写整个 Web
不要把 YOLO 放进巡线主链路
不要一口气做完整比赛路线
不要让 VLM 参与实时控制
不要删除当前稳定 demo
```

## 机械臂范围

Web 已放开到硬件完整范围：

```text
j1: 0..225
j2: 30..270
j3: 30..180
```

注意：范围放开不代表每个姿态都安全。调试时仍然建议慢速、小步移动。

## 已落地：直角弯第一版

`robot_tune_app.py` 已加入：

```text
直角弯 / Margin 面板
enable_corner 开关
right_top_area_th
right_area_th
right_wide_ratio_th
right_confirm_frames
right_turn_dir
right_turn_pre_forward_sec
right_turn_pre_forward_v
right_turn_w
right_turn_min_sec
turn_finish_err_th
```

巡线 Demo 内部状态：

```text
NORMAL_LINE
-> PRE_FORWARD
-> REACQUIRE
-> NORMAL_LINE
```

触发逻辑：

```text
right_area >= right_area_th
top_area >= right_top_area_th
wide_ratio >= right_wide_ratio_th
连续 right_confirm_frames 帧
```

触发后动作：

```text
1. pre-forward: right_turn_pre_forward_v / right_turn_pre_forward_sec
2. turn: right_turn_dir * right_turn_w / right_turn_min_sec
3. reacquire: 继续转，直到 err_norm <= turn_finish_err_th 且 bottom_area 足够
```

调参建议：

```text
先不开 enable_corner，只跑线看 overlay/log 里的 R/T/ratio。
记下直角出现时的 right_area、top_area、wide_ratio。
把阈值设到略低于这些值，再打开 enable_corner。
如果转错方向，切换 right_turn_dir。
如果太早转，提高 top/right 阈值或 confirm_frames。
如果太晚转，降低阈值或增加 pre_forward。
```
