# vision_triggered_navigator

视觉触发三阶段导航节点：巡航 → 视觉触发 → 结束点。

## 功能

1. **巡航阶段**
   - 依次访问 9 个预标巡航点。
   - 发送目标前读取 `/move_base/global_costmap/costmap` 判断目标是否可行；不可行则跳过。
   - 导航过程中通过定时器持续检查当前目标可行性；若中途变不可行，取消并跳到下一目标。
   - 到达每个点后按配置执行自转（通过 `/cmd_vel` 发布角速度）。

2. **视觉触发阶段**
   - 支持两种触发方式，通过 `trigger_mode` 参数选择：
     - `keyboard`：在终端按回车触发（测试用）。
     - `vision`：订阅指定视觉话题（默认 `/vision/detected`，`std_msgs/Bool`）。
   - 触发后打断当前巡航。
   - 任务2覆盖模式先使用 `/odom` 闭环小角度步进，将OCR目标框居中；底盘未起转时角速度从 `0.20` 自适应提升至 `0.35 rad/s`。
   - 比赛总控通过 `/vision_triggered_navigator/trigger_target` 服务完成带确认的一次性触发；`/vision/detected` 仍供独立测试使用。
   - 根据机器人/摄像头位姿向车头方向发射射线，与四个实测角点构成的真实四边形墙段求最近交点，不进行 50cm 网格吸附。
   - 目标点 = 交点沿真实墙内法向量回退 `vision_offset`（默认 0.4 m），**车头垂直指向墙外**（与内法向相反）。
   - 比赛任务2先由 move_base 同时满足距墙预停点的位置和航向约束，再取消目标；随后等待近距OCR精居中。1秒内没有新目标框时保留首次厂牌切向中心并继续停泊；已经开始复居中转动后丢框仍安全失败。
   - 最后约 `0.33m` 使用激光墙线控制实际墙距和车头垂直度、使用 odom 控制沿墙切向位置，不再依赖地图墙距或 TEB。
   - 发布 `arrived` 前，按实测墙面坐标验证完整 footprint 位于 `0.50×0.50m` 停车框内且最小物理余量不少于 `2cm`。

3. **结束点阶段**
   - 直接发送结束点目标，等待到达。

## 启动

```bash
roslaunch vision_triggered_navigator vision_triggered_navigator.launch
```

切换为视觉触发模式：

```bash
roslaunch vision_triggered_navigator vision_triggered_navigator.launch trigger_mode:=vision
```

## 参数

主要参数见 `config/vision_triggered_navigator.yaml`：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `trigger_mode` | 触发模式：`keyboard` / `vision` | `keyboard` |
| `vision_topic` | 视觉触发话题 | `/vision/detected` |
| `trigger_service` | 比赛流程使用的可靠目标触发服务 | `/vision_triggered_navigator/trigger_target` |
| `camera_frame` | 摄像头 TF 帧，作为车头朝向 | `camera_frame` |
| `base_frame` | 机器人基座 TF 帧 | `base_link` |
| `vision_rect_corners` | 长方形围墙 4 个角点 | 见 yaml |
| `vision_offset` | 墙交点回退距离（m） | `0.4` |
| `target_center_coarse_step_deg` / `target_center_fine_step_deg` | 任务2目标居中粗调/细调步长 | `4.0 / 2.0` |
| `target_center_start_speed` / `target_center_step_max_speed` | 居中起始/最大角速度 | `0.20 / 0.35` |
| `parking_staging_offset` | move_base 预停点距墙距离 | 比赛任务2 `0.55` |
| `parking_staging_position_tolerance` / `parking_staging_yaw_tolerance` | move_base控制权交接门限 | `0.10m / 0.10rad` |
| `parking_recenter_initial_wait_sec` | 预停后等待新OCR框；超时则保留首次锁定结果 | `1.0` |
| `parking_goal_offset` | 低速闭环最终点距墙距离 | 独立模式 `0.4`，比赛任务2 `0.26` |
| `parking_docking_timeout_sec` | odom 短距闭环超时 | `15.0` |
| `parking_dock_max_x/y/yaw` | 最终停泊三轴速度上限 | `0.10 / 0.06 / 0.15` |
| `parking_wall_fit_*` | 实际墙线点数、跨度、残差及方向门限 | 见 yaml |
| `parking_normal_offset` / `parking_tangent_offset` | 最终目标沿墙法向/切向的实车标定修正 | `0.0 / 0.0` |
| `coverage_goal_soft_timeout_sec` / `coverage_goal_hard_timeout_sec` | 精确锚点进度感知软/硬时限 | `25 / 40` |
| `validate_parking_box` | 是否要求完整footprint通过50cm框验证 | `false` |
| `costmap_topic` | costmap 话题 | `/move_base/global_costmap/costmap` |
| `cost_threshold` | 代价阈值，>= 则认为不可行 | `100` |
| `feasibility_check_rate` | 导航中可行性检查频率（Hz） | `1.0` |
| `rotation_speed` | 自转角速度（rad/s，左正） | `0.5` |
| `patrol_points` | 巡航点列表（含自转配置） | 9 个点 |
| `end_goal` | 结束点 | 见 yaml |
| `publish_initial_pose` | 是否发布初始位姿给 AMCL | `true` |
| `initial_pose` | 初始位姿 `x/y/yaw` | `0, 0, 0` |

## TF / cmd_vel 出处

- **`/cmd_vel`**：`ucar_controller/config/driver_params_*.yaml` 中 `vel_topic: /cmd_vel`，`base_driver` 订阅该话题执行速度。
- **`base_frame: base_link`**：`ucar_controller/launch/base_driver.launch` 加载 `driver_params_ucarV2.yaml` 设置 `base_frame: base_link`；`ucar_nav/launch/config/move_base/*.yaml` 与 AMCL 也使用 `base_link`。
- **`camera_frame: camera_frame`**：`ucar_controller/launch/tf_server.launch` 中设置 `camera_frame: camera_frame`。注意 `sensor_tf_server.py` 源码默认是 `"cam"`，若直接运行 Python 脚本而不使用 launch 文件，需改回 `"cam"`。
- **注意**：`tf_server.launch` 的 `base_frame` 是 `base_footprint`，与导航实际使用的 `base_link` 不一致；实际使用前建议用 `rosrun tf view_frames` 查看完整 TF 树。

## 依赖

- rospy
- move_base_msgs
- geometry_msgs
- nav_msgs
- actionlib
- tf
- std_msgs
- dynamic_reconfigure
