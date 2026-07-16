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
   - 根据机器人/摄像头位姿向车头方向发射射线，与长方形围墙求最近交点。
   - 目标点 = 交点沿墙内法向量回退 `vision_offset`（默认 0.4 m），**车头垂直指向墙外**（与内法向相反）。

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
| `camera_frame` | 摄像头 TF 帧，作为车头朝向 | `camera_frame` |
| `base_frame` | 机器人基座 TF 帧 | `base_link` |
| `vision_rect_corners` | 长方形围墙 4 个角点 | 见 yaml |
| `vision_offset` | 墙交点回退距离（m） | `0.4` |
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
