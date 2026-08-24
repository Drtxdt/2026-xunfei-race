# ucar_2026_national_competition

国赛总控包：任务一带坡道版本 + 省赛流程（task2~task5）无缝交接。

## 参数管理规则（重要）

**所有参数只存在于 `config/national_competition.yaml`（逐行中文注释），
launch 文件不覆盖任何值。** 调参流程：改 yaml → 重启 roslaunch。
坡道本体参数在 `ucar_2026_upanddown/config/ramp_traverse.yaml`。
launch 仅保留结构性开关：`debug`（rviz）与 `map`（地图文件路径）。

## 与省赛代码的关系

省赛 `ucar_2026_competition`、视觉、语音、LLM 全部原样复用（唯一例外：
省赛 `flow_node.launch` 新增了一个可选 `track_package` 参数，默认仍为省赛
巡线包，省赛行为不变；国赛通过交接参数把它覆盖为国赛巡线包
`ucar_2026_track_end_stop`，即带灰色挡板避障的版本）：

- 导航栈换成 `national_nav.launch`（与 `ucar_amcl_nav.launch` 参数完全
  一致，仅 AMCL/move_base 订阅门控雷达 `/scan_gated`，另起 `scan_gate`）
- 任务一由 `national_flow_node` 执行：语音握手 → move_base 到坡道暂泊点
  → `/ramp_traverse/start` 过坡 → 定位恢复（settle + clear_costmaps）
  → 导航到二维码区 → 原地旋转扫码（省赛 qr_decoder；不足三码自动去
    备用点补扫）→ 星火 X2 推理 → 播报
- 任务二~五：按 `task2 → task3_task4 → task4_task5`（无仿真则
  `task2 → task4 → task5`）顺序子进程启动省赛 `flow_node.launch`，
  task1 结果通过 launch 参数传入；省赛流程 pause 时转发到
  `/national/resume`

## 启动

```bash
catkin_make
source devel/setup.bash

# 全流程（导航+语音+ASR+TTS+LLM+相机+国赛流程）
roslaunch ucar_2026_national_competition national_full_competition.launch

# 只测坡道段：先把 yaml 里 start_stage 改为 ramp，再启动
# 只测任务一：start_stage 改为 task1（含语音+扫码+播报，不交接）
# 联调模式：skip_task4_stop_line_approach: true（车人工摆在停止线前）
```

以上模式切换全部通过修改 `config/national_competition.yaml` 的
`start_stage` / `skip_task4_stop_line_approach` 等字段完成，
**不支持也不允许**用 launch 参数覆盖。

暂停/恢复/终止：`/national/status`（JSON），`/national/resume`，
`/national/abort`（会转发 `/competition/abort` 给省赛子流程）。

## 上场前必须标定（config/national_competition.yaml）

| 参数 | 说明 |
|---|---|
| `start_stage` | 运行模式 full / task1 / ramp |
| `ramp_staging_x/y/yaw` | 坡道暂泊点（坡脚前 20~30cm，朝向=坡道轴向） |
| `ramp_heading_*` | move_base 到点后的坡道轴向闭环对正参数（默认误差不超过 2°） |
| `qr_area_x/y/yaw` | 二维码区导航点 |
| `qr_fallback_x/y/yaw` | 补扫备用点 |
| `traffic_pose_configured/x/y/yaw` | 透传给省赛 task4 的停车线标定 |
| `pitch_sign` 等 | 见 `ucar_2026_upanddown` README 标定步骤 |

## 节点接口速查

| 接口 | 类型 | 说明 |
|---|---|---|
| `/national/status` | String(latch) | `{"stage","state","message","stamp"}` |
| `/national/resume` `/national/abort` | std_srvs/Trigger | 阶段失败暂停后恢复 / 终止 |
| `/competition/task1_result` | String(latch) | 与省赛同名字段兼容 |
| `/ramp_traverse/start` `/abort` | Trigger | 由 national_flow 调用 |
| `/ramp_traverse/status` | String(latch) | segment/pitch/distance JSON |
| `/scan_gate/set_open` | std_srvs/SetBool | 雷达门控 |
