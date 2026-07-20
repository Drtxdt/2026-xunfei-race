# 智慧工厂比赛总控

正式比赛推荐从工作空间根目录执行：

```bash
export XF_SPARK_API_PASSWORD='你的 APIPassword'
export SIM_BRIDGE_HOST='仿真电脑 IP'
# 完成实车标定后再设置；未设置时任务3结束后会安全暂停。
export TRAFFIC_X='...'
export TRAFFIC_Y='...'
export TRAFFIC_YAW='...'
bash run_competition.sh
```

调试模式会启动 RViz：

```bash
bash run_competition.sh --debug
```

五个子任务可以独立启动：

```bash
roslaunch ucar_2026_competition task1.launch
roslaunch ucar_2026_competition task2.launch target_category:=food target_item:=苹果 target_workshop:=食品加工车间
roslaunch ucar_2026_competition task3.launch target_category:=food sim_item:=苹果 sim_workshop:=食品加工车间 sim_bridge_host:=192.168.1.20
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

仿真电脑在 `fangzhen` 仓库根目录启动独立 ROS Master、Gazebo、任务3和 TCP 桥：

```bash
bash run_task3_sim.sh
```

流程暂停后，可先修改总控节点私有参数，再恢复当前阶段：

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
