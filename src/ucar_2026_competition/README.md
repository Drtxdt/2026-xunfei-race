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
