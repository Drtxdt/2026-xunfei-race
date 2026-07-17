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

任务2在 OCR 两次命中后通过同步服务请求导航锁存目标，并等待导航状态确认；2 秒内没有收到服务和状态双重确认时会停车并报告 `trigger_delivery_failed`。最终停车点由实测四边形墙段连续计算，不将场地吸附为 40 个固定格点。实车仅需在确实执行到 `parking_verifying` 后，根据余量日志分别调整 `parking_normal_offset` 或 `parking_tangent_offset`。

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
