# 建图
## 编译
```bash
catkin_make
source devel/setup.bash
```
## 1. 遥控
### 安装
```bash
# 用 键盘控制节点 teleop_twist_keyboard，它直接发布 /cmd_vel，你的 base_driver 本来就在监听 /cmd_vel（MOTOR_MODE_CMD 模式），所以不需要改任何代码。
sudo apt install ros-noetic-teleop-twist-keyboard
```
### 启动
```bash
# 目前 joy_node 在 launch 文件里被注释掉了：
<!-- ucar_controller/launch/base_driver.launch 第7行 -->
<!-- <node pkg="joy" name="joy_node" type="joy_node"/> -->
# 取消注释
<node pkg="joy" name="joy_node" type="joy_node"/>
```
### 使用
```bash
# 1. 先正常启动 base_driver.launch，确保小车能正常移动，这时已经启动roscore的ros主节点
roslaunch ucar_controller base_driver.launch
# 2. 另开终端
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```
```
i	前进
,	后退
j	左转
l	右转
u/o/m/.	        斜向移动（麦克纳姆轮适用）
Shift + 方向	横移/斜移（全向模式）
q/z	            整体加速/减速
k / 其他	          停止
```
是**增量式（步进式）**的按一下加0.1，要减速必须按反方向的按键

## 2. 建图指令
```bash
roslaunch ucar_map ucar_gmapping.launch
```
不结束上面的建图进程，运行：
```bash
rosrun map_server map_saver -f my_map
mv my_map.pgm /home/ucar/26nav/src/ucar_nav/maps
```


## 3. 把地图转给导航
把pgm和yaml存到maps目录下，修改下面的参数
```bash
<node name="map_server" pkg="map_server" type="map_server" 
      args="$(find ucar_nav)/maps/map0.05.yaml" output="screen">
```

## 4. 开rviz然后开导航
```bash
# RViz 被注释掉了，所以启动导航后不会自动弹出 RViz。但配置文件是存在的：
<!--<node pkg="rviz" type="rviz" name="rviz" required="true"
    args="-d $(find ucar_nav)/launch/config/rviz/tebrviz.rviz"/>-->
```

```bash
roslaunch ucar_nav ucar_navigation.launch
```

## 5. ros1 主节点
```bash
roscore
```

## 6. p图pgm
- **==将画笔改为铅笔：在界面内按N键即可==**


## 7. 在线调参
```bash
rosrun rqt_reconfigure rqt_reconfigure
```