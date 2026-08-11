# 智慧工厂比赛总控

## 运行位置

- **Windows**：只用于修改和推送代码，不运行 ROS。
- **Ubuntu 20.04 控制机**：正式比赛唯一的命令入口，同时运行 Gazebo。
- **小车**：运行全部实车节点；由 Ubuntu 通过免密 SSH 自动启动，正式比赛时不要手动登录启动。

Ubuntu 和小车各有自己的 `:11311` ROS Master。两边不共享 ROS 话题，只通过
任务三 TCP 协议通信；不再使用 Ubuntu `:11312`。小车复用厂家随系统启动的
`roscore.service`，不要为了比赛入口手动停止该服务。

## 一次性部署

Ubuntu 使用两个独立的 Catkin 工作空间：

```bash
cd /home/txdt
git clone -b sim git@github.com:Drtxdt/2026-xunfei-race.git
cd /home/txdt/2026-xunfei-race
catkin_make --pkg \
  ucar_2026_competition_speech \
  ucar_2026_smart_factory_llm \
  ucar_2026_competition

cd /home/txdt/2026-race-nav/gazebo_ws
catkin_make
```

在 Ubuntu 的 `~/.bashrc` 写入：

```bash
export UCAR_COMPETITION_WS=/home/txdt/2026-xunfei-race
export UCAR_SIM_WS=/home/txdt/2026-race-nav/gazebo_ws
export UCAR_ROBOT_HOST=ucar@192.168.1.6
export UCAR_ROBOT_WS=/home/ucar/2026-xunfei-race
export UCAR_ROBOT_ENV=/home/ucar/.config/ucar_2026/robot_env.sh

unset ROS_MASTER_URI ROS_IP ROS_HOSTNAME
source /opt/ros/noetic/setup.bash
source "$UCAR_SIM_WS/devel/setup.bash"
source "$UCAR_COMPETITION_WS/devel/setup.bash"
```

上述地址已在当前比赛网络中验证。若路由器重新分配了小车地址，只修改
`UCAR_ROBOT_HOST`，不要把 SSH 密码写入仓库或启动文件。配置完成后重新打开终端，或先执行
`source ~/.bashrc`。

Ubuntu 配置免密 SSH：

```bash
test -f ~/.ssh/id_ed25519 || ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519
ssh-copy-id "$UCAR_ROBOT_HOST"
ssh -o BatchMode=yes "$UCAR_ROBOT_HOST" true
```

最后一条命令必须能够在不询问密码的情况下退出并返回成功；正式入口使用
`BatchMode=yes`，不会在比赛启动过程中等待交互式密码输入。

小车只在赛前部署时通过 SSH 执行：

```bash
cd /home/ucar
git clone -b sim git@github.com:Drtxdt/2026-xunfei-race.git
cd /home/ucar/2026-xunfei-race
catkin_make
```

随后在小车端创建
`~/.config/ucar_2026/robot_env.sh`：

```bash
export XF_SPARK_API_PASSWORD='轮换后的新密码'
export ROBOT_WS=/home/ucar/ucar_ws
export IAT_CREDENTIALS_FILE=/home/ucar/.config/ucar_2026/iat_credentials.json
```

随后执行：

```bash
chmod 600 ~/.config/ucar_2026/robot_env.sh
```

Ubuntu 和小车的比赛仓库必须位于同一 Git commit 且没有未提交改动，否则监督器
会拒绝启动。仿真仓库的 `src/car3/config/task3.yaml` 可保留本机标定修改，不参与该检查。
更新仿真仓库时不得使用 `git reset --hard` 或覆盖该文件。

## 正式比赛

只在 **Ubuntu 控制机**执行：

```bash
roslaunch ucar_2026_competition full_competition.launch
```

该入口自动选择通往小车的 Ubuntu 网卡地址、启动 Gazebo，并通过 SSH 启动小车的
`physical_competition.launch`。Gazebo、SSH、小车实车 launch 任一失败或退出时，
整套比赛都会停止并清理两端进程组。

正式入口会先检查控制机仓库、小车仓库、版本和 SSH；只有预检全部通过才启动
Gazebo。若提示工作区有未提交改动，应先提交并同步两端版本，不要把随后出现的
`Waiting for /clock` 或 controller spawner 退出信息当作 Gazebo 根因。

关闭仿真可使用：

```bash
roslaunch ucar_2026_competition full_competition.launch enable_simulation:=false
```

外部仿真调试使用 `start_local_sim:=false sim_bridge_host:=<外部桥地址>`。
`run_competition.sh` 仅是 Ubuntu 调试包装器，不是正式入口。

以下五个子任务命令只用于调试，均在 **小车 SSH 会话**中执行：

```bash
roslaunch ucar_2026_competition task1.launch
roslaunch ucar_2026_competition task2.launch target_category:=daily target_item:=牙膏 target_workshop:=日用品加工车间 sim_target_category:=food sim_item:=香蕉 sim_workshop:=食品加工车间
roslaunch ucar_2026_competition task3.launch sim_target_category:=food sim_item:=香蕉 sim_workshop:=食品加工车间
roslaunch ucar_2026_competition task4.launch traffic_pose_configured:=true traffic_x:="$TRAFFIC_X" traffic_y:="$TRAFFIC_Y" traffic_yaw:="$TRAFFIC_YAW"
roslaunch ucar_2026_competition task5.launch traffic_decision:=left
```

任务4与任务5可从停止线前连续联调。启动前应将小车车头朝向交通灯，
车头距停止线不超过10cm，且车轮不得压线或越线：

```bash
roslaunch ucar_2026_competition task4_task5.launch debug:=true
```

该入口不会导航到停止线，也不会再次向前靠线。启动后车辆保持停车并识别交通灯；
红灯时继续停车，确认左转、右转或直行后自动播报决策、启动对应巡线控制器，
到达任务5终点后停车并播报“任务完成”。关闭调试画面时可省略 `debug:=true`。

任务1完成后直接联调任务2（复用同一导航栈、相机和AMCL，不重新发布初始位姿）：

```bash
roslaunch ucar_2026_competition task1_task2.launch debug:=true 2>&1 | tee task1_task2.log
```

预停后近距OCR属于可选增强：1秒内没有新的目标框时发布
`parking_recenter_skipped`，保留首次厂牌锁定结果并继续激光墙面停泊。
覆盖锚点采用25秒软时限和40秒硬时限，仅在最近5秒仍前进至少3cm时延长。
只有导航发布`arrived`后，流程才播报“已将[货品名称]放入[仓库类别]”。

任务2在 OCR 两次命中后通过同步服务请求导航锁存目标，并等待导航状态确认；2 秒内没有收到服务和状态双重确认时会停车并报告 `trigger_delivery_failed`。move_base只驶到距墙0.55m的预停点，最后约33cm使用激光拟合的实际墙面控制距离和垂直度、使用odom控制厂牌切向位置。车辆不会将场地吸附为40个固定格点，只有完整footprint具有至少2cm框内余量才发布`arrived`。

流程暂停后，以下命令同样必须在 **小车 SSH 会话**中执行：

```bash
rosparam set /competition_flow/traffic_x 1.23
rosparam set /competition_flow/traffic_y 0.45
rosparam set /competition_flow/traffic_yaw 1.57
rosparam set /competition_flow/traffic_pose_configured true
rosservice call /competition/resume
```

任何阶段均可安全终止：

```bash
rosservice call /competition/abort
```
