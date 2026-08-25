# 2026 U-CAR 智慧工厂赛项

这是 2026 年 U-CAR 智慧工厂赛项的小车端 ROS 工程。仓库以 ROS Noetic catkin 工作空间的形式组织，包含比赛总控、导航与定位、二维码和厂牌识别、语音交互、循迹停车、交通灯识别，以及任务 3 的仿真协同模块。

主流程对应五个子任务：

1. 扫描二维码，结合语音指令和星火大模型，确定实体货品与仿真货品。
2. 导航到目标车间，识别厂牌并完成贴墙停车和入库播报。
3. 通过 TCP 桥接与独立仿真环境协同，完成仿真货品入库。
4. 导航到交通灯位置，识别信号并播报左转、右转或直行。
5. 根据交通决策启动对应循迹路线，在终点停车并播报任务完成。

具体参考 [省赛赛规](赛规.md)

## 环境要求

- Ubuntu + ROS Noetic
- catkin 工作空间
- Python 3（仓库中的 ROS Python 节点按 Python 3 编写）
- USB 摄像头、激光雷达、底盘驱动和里程计
- 小车工作区中的 `speech_command`，用于旧版离线 TTS 和语音输入
- 任务 1 的讯飞星火 APIPassword
- 任务 3 仿真电脑上的独立 ROS Master、仿真程序和 TCP 桥

当前工程的实体车文档以 `/home/ucar/ucar_ws` 为底层语音工作区、以 `/home/ucar/2026-xunfei-race` 为本仓库路径。换到其他目录时，需要相应调整命令和脚本中的路径。

## 获取与编译

将仓库放入 catkin 工作空间的 `src` 目录后编译：

```bash
cd ~/2026-xunfei-race
source /opt/ros/noetic/setup.bash
source /home/ucar/ucar_ws/devel/setup.bash

catkin_make \
  -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m \
  -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.7m.so \
  -DPYTHON_NUMPY_INCLUDE_DIR=/usr/lib/python3/dist-packages/numpy/core/include

source devel/setup.bash
```

上面的 Python 参数针对车上的系统 Python 3.7 和 ARM 环境。不要在同一个构建过程中混入 conda 的 Python，否则 `cv_bridge` 等 Python 绑定可能链接到不兼容的运行库。

只编译某个包时可以使用：

```bash
catkin_make --pkg ucar_2026_line_follow
```

每个新终端都要重新加载环境：

```bash
source /opt/ros/noetic/setup.bash
source ~/2026-xunfei-race/devel/setup.bash
```

## 正式流程

推荐使用仓库根目录的 `run_competition.sh`。脚本会检查 ROS、当前工作区、语音节点、讯飞听写凭据和仿真桥环境，然后依次启动 ROS Master、旧版语音节点、固定命令听写、比赛总控和日志目录。

先设置必要的环境变量：

```bash
export XF_SPARK_API_PASSWORD='讯飞星火 APIPassword'
export SIM_BRIDGE_HOST='仿真电脑 IP'

# 交通灯位置必须经过实车标定；没有设置时，流程会在任务 3 后安全暂停。
export TRAFFIC_X='...'
export TRAFFIC_Y='...'
export TRAFFIC_YAW='...'
```

启动完整流程：

```bash
cd ~/2026-xunfei-race
bash run_competition.sh
```

需要 RViz 时：

```bash
bash run_competition.sh --debug
```

日志保存在 `logs/competition_YYYYMMDD_HHMMSS/`。

如果不使用脚本，也可以直接启动总控：

```bash
roslaunch ucar_2026_competition full_competition.launch \
  enable_simulation:=true \
  sim_bridge_host:=192.168.1.28
```

## 分阶段启动

总控包提供了各阶段的独立入口，适合调试和现场复现：

```bash
# 任务 1：二维码、语音指令和大模型推理
roslaunch ucar_2026_competition task1.launch

# 任务 1 完成后连续联调任务 2
roslaunch ucar_2026_competition task1_task2.launch debug:=true

# 单独启动任务 2，需要显式给出目标参数
roslaunch ucar_2026_competition task2.launch \
  target_category:=daily target_item:=牙膏 target_workshop:=日用品加工车间 \
  sim_target_category:=food sim_item:=香蕉 sim_workshop:=食品加工车间

# 单独启动任务 3
roslaunch ucar_2026_competition task3.launch \
  sim_target_category:=food sim_item:=香蕉 sim_workshop:=食品加工车间 \
  sim_bridge_host:=192.168.1.20

# 单独启动任务 4
roslaunch ucar_2026_competition task4.launch \
  traffic_pose_configured:=true traffic_x:="$TRAFFIC_X" \
  traffic_y:="$TRAFFIC_Y" traffic_yaw:="$TRAFFIC_YAW"

# 单独启动任务 5
roslaunch ucar_2026_competition task5.launch traffic_decision:=left
```

任务 4、5 可以从停止线前直接联调：

```bash
roslaunch ucar_2026_competition task4_task5.launch debug:=true
```

启动前车辆应面向交通灯，距停止线不超过 10 cm，且车轮不能压线或越线。该入口不会替车辆导航到停止线，也不会再次向前靠线。

流程控制服务：

```bash
# 流程暂停后修改交通灯位姿并恢复
rosparam set /competition_flow/traffic_x 1.23
rosparam set /competition_flow/traffic_y 0.45
rosparam set /competition_flow/traffic_yaw 1.57
rosparam set /competition_flow/traffic_pose_configured true
rosservice call /competition/resume

# 在任意阶段安全终止
rosservice call /competition/abort
```

## 仿真协同

任务 3 使用独立 ROS Master，通过 TCP 桥与小车端协同。仿真电脑应在 `fangzhen` 工程根目录启动对应脚本：

```bash
bash run_task3_sim.sh
```

小车端通过 `SIM_BRIDGE_HOST` 指定仿真电脑地址，端口默认为 `26003`，也可以设置 `SIM_BRIDGE_PORT` 覆盖。仿真超时或反复重连失败时，主流程会跳过任务 3，进入任务 4；具体行为由 `src/ucar_2026_competition/config/competition.yaml` 控制。

## 语音和大模型

语音链路分为两层：旧版 `voice_speak_node` 负责真正播放，`ucar_2026_competition_speech` 负责根据比赛事件生成统一文案并发布到 `/speak`。

启动旧版 TTS：

```bash
source /home/ucar/ucar_ws/devel/setup.bash
rosrun speech_command voice_speak_node
```

启动统一播报服务：

```bash
source ~/2026-xunfei-race/devel/setup.bash
roslaunch ucar_2026_competition_speech competition_speech.launch
```

统一服务为 `/competition_speech/announce`，类型为 `ucar_2026_competition_speech/Announce`。支持 `task1` 到 `task5` 五类事件；`wait: true` 时会等待预计播报结束再返回。

例如：

```bash
rosservice call /competition_speech/announce \
  "{event: 'task4', item: '', workshop: '', decision: 'left', text: '', wait: true}"
```

任务 1 的大模型服务为 `/smart_factory_llm/reason_pickup_order`，类型为 `ucar_2026_smart_factory_llm/ReasonPickupOrder`。输入是三个二维码物品名和完整语音指令，输出包括 `pickup_item`、`pickup_workshop`、`sim_item`、`sim_workshop` 以及 `announcement_full`。

```bash
export XF_SPARK_API_PASSWORD='讯飞星火 APIPassword'
roslaunch ucar_2026_smart_factory_llm reason_pickup.launch
```

不要把 APIPassword、AppID、APIKey 或 APISecret 写入仓库。相关凭据应通过环境变量、车上的凭据文件或 launch 参数注入；如果历史文档或提交记录中出现真实密钥，应立即在讯飞控制台吊销并重新生成。

## 主要 ROS 包

| 包 | 用途 |
| --- | --- |
| `ucar_2026_competition` | 五个子任务的总控、阶段切换、导航和仿真协同 |
| `ucar_2026_competition_speech` | 统一比赛播报服务和播报状态 |
| `ucar_2026_smart_factory_llm` | 任务 1 的星火大模型推理服务 |
| `ucar_2026_line_follow` | 视觉循迹、分叉选择和终点停车调试 |
| `ucar_2026_track_end_stop` | 任务 5 的主循迹和终点停车实现 |
| `ucar_2026_track_end_stop_provincial` | 省赛版本循迹实现 |
| `ucar_2026_traffic_light_rknn_test` | RKNN 红绿灯识别测试 |
| `factory_sign_ppocr_rknn_test` | RKNN PP-OCR 厂牌识别测试 |
| `factory_sign_ocr_test` | 厂牌分类识别测试 |
| `ucar_2026_qr_speak_test` | 二维码识别与播报联调 |
| `ucar_2026_nav` | 导航、视觉触发导航和相关测试包 |
| `yolo` | 二维码、交通标志及数据集辅助工具 |
| `ucar_controller`、`ydlidar`、`ucar_map` | 底盘、激光雷达和地图相关包 |

仓库还保留了国赛流程、上下坡、严格任务状态机以及若干模型转换和 PC 测试工具。这些目录用于专项联调，不会全部由 `full_competition.launch` 自动启动。

## 常用专项测试

```bash
# 左、右、中三条循迹路线
roslaunch ucar_2026_track_end_stop track_end_stop.launch
roslaunch ucar_2026_track_end_stop right_track_end_stop.launch
roslaunch ucar_2026_track_end_stop stable_right_track_end_stop.launch

# 红绿灯
roslaunch ucar_2026_traffic_light_rknn_test traffic_light_rknn_x11_speak_test.launch

# 厂牌识别
roslaunch factory_sign_ppocr_rknn_test factory_sign_ppocr_rknn_test.launch \
  recognition_mode:=ppocr_rknn_system

# 视觉循迹测试和调试画面
roslaunch ucar_2026_line_follow line_follow_test.launch
rosrun image_view image_view image:=/line_follow/debug_image
```

循迹节点常用接口：

```text
/usb_cam/image_raw       sensor_msgs/Image       输入图像
/odom                    nav_msgs/Odometry      输入里程计
/cmd_vel                 geometry_msgs/Twist     输出速度
/line_follow/status      std_msgs/String         输出状态
/line_follow/debug_image sensor_msgs/Image       输出调试图像
```

正式任务 5 由比赛总控根据交通灯决策自动选择对应 launch，不需要手动选择路线。

## 目录说明

```text
src/                     ROS 包和第三方依赖
docs/                    语音、大模型、YOLO 等专题文档
debug/                   语音修复、补丁和现场调试记录
picture/                 相机和场地调试图片
run_competition.sh       完整比赛启动脚本
competition_task_interface.py 供外部总控调用的任务接口封装
命令.txt                 现场常用命令汇总
赛规.md                  赛题规则整理
```

主流程参数集中在 `src/ucar_2026_competition/config/competition.yaml`，启动文件位于 `src/ucar_2026_competition/launch`。修改参数时优先通过 launch 参数或 ROS 私有参数覆盖，确认稳定后再更新 YAML 默认值。

## 测试

仓库中各 ROS 包带有 Python 单元测试，至少可以在工作区根目录执行：

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
catkin_make run_tests
```

部分测试不需要真实硬件，覆盖任务状态机、播报模板、OCR/交通灯输入处理和循迹逻辑。需要相机、雷达、底盘或旧版语音节点的测试，必须在对应设备和 ROS 节点已启动后进行。

## 现场排查顺序

遇到整车不动作时，先按下面顺序检查：

1. `roscore`、底盘、雷达、相机和 `/odom` 是否正常。
2. `rostopic info /speak` 是否能看到 `voice_speak_node` 订阅者。
3. `/competition/status` 是否有总控状态输出。
4. 任务 1 是否同时拿到了三个稳定二维码结果和完整语音指令。
5. 任务 2 的 OCR 是否连续命中两次，导航触发服务是否返回确认。
6. 仿真桥地址和端口是否可达，任务 3 是否收到完成反馈。
7. 交通灯坐标、交通决策和对应循迹 launch 是否匹配。

语音启动、统一播报调用和大模型接口的详细说明见：

- `docs/voice_startup_guide.md`
- `docs/competition_speech_call_guide.md`
- `docs/task1_llm_interface.md`
- `src/ucar_2026_competition_speech/README.md`
- `src/ucar_2026_smart_factory_llm/README.md`
- `src/ucar_2026_line_follow/README.md`

## 备注

这是比赛现场工程，不是开箱即用的通用机器人框架。相机外参、交通灯坐标、导航目标、停车框尺寸、循迹速度和语音设备都与具体车辆和场地有关。任何涉及运动控制的修改，都应先在抬轮、低速和空场环境下验证，再进入完整流程。
