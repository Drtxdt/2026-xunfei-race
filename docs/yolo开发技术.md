# 小车运动控制与视觉开发技术文档

> 小车有两个工作空间，必须 **先后 source**：
> ```bash
> source ~/ucar_ws/devel/setup.bash          # 基础工作空间（底盘驱动 ucar_controller）
> source ~/2026-xunfei-race/devel/setup.bash  # 比赛工作空间（yolo、巡线、语音等）
> ```

## 一、运动控制

### 1.1 控制链路

```
指令 → /cmd_vel (Twist) → base_driver → 串口(/dev/base_serial_port, 921600) → 底盘MCU → 麦克纳姆轮
```

### 1.2 启动底盘

```bash
source ~/ucar_ws/devel/setup.bash
source ~/2026-xunfei-race/devel/setup.bash
roslaunch ucar_controller base_driver.launch
```

验证：`rostopic list | grep cmd_vel` 应看到 `/cmd_vel`。

### 1.3 Twist 接口

| 字段 | 含义 | 值域 |
|------|------|------|
| `linear.x` | 前后 m/s，正=前进 | [-3.0, 3.0] |
| `linear.y` | 左右 m/s，正=左平移 | [-3.0, 3.0] |
| `angular.z` | 旋转 rad/s，正=左转 | [-3.14, 3.14] |

常用指令：

```bash
# 前进 0.05
rostopic pub /cmd_vel geometry_msgs/Twist "linear: {x: 0.05, y: 0, z: 0} angular: {x: 0, y: 0, z: 0}" -1

# 左转 0.2
rostopic pub /cmd_vel geometry_msgs/Twist "linear: {x: 0, y: 0, z: 0} angular: {x: 0, y: 0, z: 0.2}" -1

# 停车
rostopic pub /cmd_vel geometry_msgs/Twist "linear: {x: 0, y: 0, z: 0} angular: {x: 0, y: 0, z: 0}" -1
```

### 1.4 麦克纳姆轮运动学

```
vw1 = vx - vy - ω×a    # 左前
vw2 = vx + vy + ω×a    # 右前
vw3 = vx - vy + ω×a    # 左后
vw4 = vx + vy - ω×a    # 右后
```
a = 0.2169m，b = 0。

### 1.5 键盘控制 (keyboard_collect_yolo_images.py)

| 按键 | 运动 |
|------|------|
| W/S | 前进/后退 |
| A/D | 左转/右转 |
| Q/E | 左平移/右平移 |
| X/Space | 停车 |
| 1/2 | 减速/加速 |
| P | 暂停/继续保存 |
| C | 手动保存一张 |

```bash
# 确保 source 两个工作空间后再运行
rosrun yolo keyboard_collect_yolo_images.py --cls red --interval 0.3 --start-paused
```

关键参数：`--linear 0.04`（线速度）、`--angular 0.18`（角速度）、`--max-images 200`。

### 1.6 其他 ROS 服务

```bash
rosservice call /get_max_vel          # 查最大速度
rosservice call /set_max_vel "..."    # 设最大速度
rosservice call /stop_move            # 停车
rosservice call /get_battery_state    # 电池
```

---

## 二、视觉系统

### 2.1 摄像头

- 话题：`/usb_cam/image_raw`，配套话题：`/qr_code_data`
- 分辨率 640×480@30FPS，FOV 水平 124.8°

```bash
# 检查摄像头是否发图
rostopic hz /usb_cam/image_raw

# 无数据则重启 usb_cam 节点
rosnode kill /usb_cam_node
rosrun usb_cam usb_cam_node _video_device:=/dev/video0
```

### 2.2 QR 二维码扫图

```bash
rosrun yolo qr_collect_and_decode.py --fetch
```

`--fetch` 表示访问 QR 内 URL 获取物品名。结果实时发布到 `/qr_code_data`。

```bash
# 另开终端查看结果
rostopic echo /qr_code_data/data
```

输出示例：
```json
{"stamp": 1779..., "count": 1, "items": [{"raw": "http://...food", "api": {"code":200,"result":"苹果"}, "ok": true, "result": "苹果"}]}
```

三个 QR 码分别对应 food/daily/electronic 三个品类，API 每次随机返回不同物品名。比赛时 3 个码各取一次结果，结合语音指令送星火 X2 分类。

QR 扫图常见问题：

| 现象 | 原因 | 解决 |
|------|------|------|
| `cv2.QRCodeDetector` 报错 | noetic 无 opencv-contrib | 已改用 `pyzbar` 备选，不影响 |
| `api: null, ok: false` | 没加 `--fetch` | 加上后自动请求 URL |

### 2.3 本地浏览器实时查看小车摄像头

**方案一：camera_mjpeg_server.py（零依赖，已集成在 yolo 包中）**

```bash
source ~/ucar_ws/devel/setup.bash
source ~/2026-xunfei-race/devel/setup.bash
rosrun yolo camera_mjpeg_server.py
```

电脑浏览器打开：**`http://192.168.1.6:8080/stream`**

**方案二：web_video_server（推荐，更稳定）**

```bash
# 确认 ROS 版本
rosversion -d                # 输出 noetic

# 安装
sudo apt install ros-noetic-web-video-server

# 启动
rosrun web_video_server web_video_server
```

电脑浏览器打开：**`http://192.168.1.6:8080/stream?topic=/usb_cam/image_raw`**

### 2.4 YOLO 数据集采集

```bash
# 采集红灯图像
rosrun yolo keyboard_collect_yolo_images.py --cls red --interval 0.3 --start-paused

# 按 P 开始采图，WASD 控制小车移动换角度
# 图片保存到 ~/2026-xunfei-race/src/yolo/yolo_dataset/raw_images/<cls>/
```

---

## 三、常见问题

| 现象 | 排查 |
|------|------|
| `/cmd_vel` 不存在 | 底盘驱动未启动 |
| 小车不动 | `ls /dev/base_serial_port`，检查电机供电 |
| 摄像头无数据 | `rosnode kill /usb_cam_node` 后重启 |
| `api: null` | 加 `--fetch` 参数 |
| web_video_server 黑屏 | 确认 usb_cam 在发图 `rostopic hz /usb_cam/image_raw` |
| pip 包冲突 | 不用 opencv-contrib，pyzbar 已够用 |
| `rosrun yolo ...` 找不到包 | 先 `source ~/2026-xunfei-race/devel/setup.bash`，确认 `catkin_make` 过 |
| `$(find ucar_controller)` 找不到 | 先 `source ~/ucar_ws/devel/setup.bash` |
| `devel/setup.bash: No such file` | 不要在 `src/` 下 source，要在工作空间根目录 `~/2026-xunfei-race/` 下执行 |
| 红绿灯画面左右反转（镜像） | 不得用水平翻转纠正；保持 `flip:=false`，以原始图像中的箭头实际方向标注 |

---

## 四、红绿灯采集完整流程

### 4.1 类别说明

红绿灯共 4 个类别，YOLO 检测到后小车据此决定行驶方向：

| 类别 | 含义 | 小车行为 |
|------|------|----------|
| `red_light` | 红灯 | 停车等待 |
| `green_straight` | 绿灯直行 ↑ | 直行 |
| `green_left` | 绿灯左转 ← | 左转 |
| `green_right` | 绿灯右转 → | 右转 |

### 4.2 左右方向安全约束

红绿灯数据集和部署画面统一使用摄像头原始方向：

- launch 必须使用 `flip:=false`；采集脚本不得加 `--flip`。
- 标注时以图片里箭头实际指向为准，不以观察者的主观“正反”为准。
- YOLOv5 训练设置 `fliplr=0.0` 和 `flipud=0.0`，否则左右类别会被破坏。
- 采集脚本现已默认拒绝 `--flip`，防止误操作。

### 4.3 编译 yolo 包

yolo 包首次使用或修改后需要编译：

```bash
cd ~/2026-xunfei-race
catkin_make --pkg yolo    # 只编译 yolo，跳过其他包
source devel/setup.bash
```

### 4.4 一键启动：运动控制 + 摄像头 + 视频流

```bash
source ~/ucar_ws/devel/setup.bash && source ~/2026-xunfei-race/devel/setup.bash && roslaunch yolo traffic_light_collect.launch flip:=false
```

一条命令同时启动：

| 组件 | launch 来源 | 作用 |
|------|------------|------|
| `base_driver` | `ucar_controller/base_driver.launch`（来自 `~/ucar_ws`） | 底盘驱动，订阅 `/cmd_vel` 控制麦克纳姆轮 |
| `usb_cam` | `usb_cam/usb_cam-test.launch`（ROS 系统包） | USB 摄像头驱动，发布 `/usb_cam/image_raw` |
| `camera_mjpeg_server` | `yolo/camera_mjpeg_server.py`（来自 `~/2026-xunfei-race`） | MJPEG HTTP 视频流，端口 8080 |

参数控制：

```bash
# 红绿灯采集必须不翻转（默认）
roslaunch yolo traffic_light_collect.launch flip:=false

# 只启动摄像头+视频流，不启底盘（调试用）
roslaunch yolo traffic_light_collect.launch start_driver:=false flip:=false

# 只启动底盘+摄像头，不开视频流
roslaunch yolo traffic_light_collect.launch start_mjpeg:=false
```

### 4.5 电脑浏览器实时查看摄像头

在电脑浏览器打开：**`http://192.168.1.6:8080/stream`**

> 浏览器画面与保存图片均保持原始方向。不要在浏览器或采集端另做水平翻转。

### 4.6 键盘采集命令行（一句复制）

**终端 1 — 启动服务：**

```bash
source ~/ucar_ws/devel/setup.bash && source ~/2026-xunfei-race/devel/setup.bash && roslaunch yolo traffic_light_collect.launch start_driver:=false start_inference:=false flip:=false
```

**终端 2 — 按 session 依次采集 4 个正类和 background：**

```bash
rosrun yolo keyboard_collect_yolo_images.py --cls red_light --output ~/traffic_dataset_raw/session_01_normal_light --interval 0.8 --max-images 120 --start-paused
```

```bash
rosrun yolo keyboard_collect_yolo_images.py --cls green_straight --output ~/traffic_dataset_raw/session_01_normal_light --interval 0.8 --max-images 120 --start-paused
```

```bash
rosrun yolo keyboard_collect_yolo_images.py --cls green_left --output ~/traffic_dataset_raw/session_01_normal_light --interval 0.8 --max-images 120 --start-paused
```

```bash
rosrun yolo keyboard_collect_yolo_images.py --cls green_right --output ~/traffic_dataset_raw/session_01_normal_light --interval 0.8 --max-images 120 --start-paused
```

负样本把 `--cls` 替换为 `background`。启动后按 `P` 开始保存；安全起见默认不启动车轮驱动，由人工调整车位。完整的 session 配额、LabelImg 标注和数据集构建流程见 `src/yolo/traffic_yolov5_dataset.md`。

**关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cls` | red_light | 类别名：red_light / green_straight / green_left / green_right / background |
| `--interval` | 0.5 | 自动保存间隔（秒） |
| `--linear` | 0.04 | 基础线速度 m/s |
| `--angular` | 0.18 | 基础角速度 rad/s |
| `--max-images` | 0 | 最大保存张数（0=不限制） |
| `--start-paused` | false | 启动后暂停，需按 P 开始 |
| `--flip` | false | 危险兼容参数；红绿灯数据不得使用 |

图片保存路径：`<output>/<cls>/`，每个 session 使用独立的 `<output>`。

文件名格式：`<cls>_000001_<timestamp>.jpg`

采集完目录结构：

```
traffic_dataset_raw/session_01_normal_light/
├── green_left/
├── green_right/
├── green_straight/
├── red_light/
└── background/
```

### 4.7 yolo 包文件结构

```
~/2026-xunfei-race/src/yolo/
├── CMakeLists.txt                     # catkin 编译
├── package.xml                        # 包声明
├── camera_mjpeg_server.py             # MJPEG 视频流服务器（支持 --flip）
├── keyboard_collect_yolo_images.py    # 键盘采集（CLI + ROS param 双模式）
├── qr_collect_and_decode.py           # QR 码扫图识别
└── launch/
    └── traffic_light_collect.launch   # 一键启动底盘+摄像头+视频流
```

### 4.8 进阶：roscd / rosrun 快捷操作

```bash
# 跳到 yolo 包目录
roscd yolo

# 跳到 launch 目录
roscd yolo/launch

# 直接运行脚本
rosrun yolo camera_mjpeg_server.py _flip:=false
rosrun yolo keyboard_collect_yolo_images.py --cls red_light --output ~/traffic_dataset_raw/session_01_normal_light --start-paused
rosrun yolo qr_collect_and_decode.py --fetch
```
