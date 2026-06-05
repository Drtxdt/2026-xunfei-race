# YOLO 红绿灯推理节点 — 部署与调试说明

> 训练方案见 [yolo训练.md](yolo训练.md)，本文档覆盖部署、ROS API、参数调优、测试步骤和比赛合规说明。

---

## 当前训练结果

| 模型 | mAP@0.5 | mAP@0.5:0.95 | 图片数 | 类别数 |
|---|---|---|---|---|
| YOLOv5s | 0.993 | 0.956 | 2806 (train 2244 / val 562) | 4 |

类别与 class_id 映射：

| class_id | 名称 | 含义 | 比赛对应 |
|---|---|---|---|
| 0 | `green_left` | 绿灯左转 | 左转灯（绿色箭头左向） |
| 1 | `green_right` | 绿灯右转 | 右转灯（绿色箭头右向） |
| 2 | `green_straight` | 绿灯直行 | 直行灯（绿色箭头上向） |
| 3 | `red_light` | 红灯停止 | 红灯（红色停止） |

> 此映射已在校验节点 `traffic_light_inference_node.py`（line 30）和数据集配置 `yolo_dataset/data.yaml` 中保持一致。

---

## 包概览

```
yolo/
  traffic_light_inference_node.py   # 推理节点（核心）
  camera_mjpeg_server.py            # MJPEG 视频流
  keyboard_collect_yolo_images.py   # 键盘采集工具
  qr_collect_and_decode.py          # QR 码采集/解码
  coco2yolo.py                      # COCO→YOLO 格式转换
  config/
    traffic_light.yaml              # 推理参数配置
  launch/
    traffic_light_inference.launch  # 推理独立启动
    traffic_light_collect.launch    # 采集启动（含可选推理）
  models/
    best.pt                          # 训练好的权重（需手动放入）
  yolo_dataset/                     # 训练数据集
```

---

## ROS API

### 订阅话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/usb_cam/image_raw` | `sensor_msgs/Image` | 摄像头输入（BGR8），可通过 `~image_topic` 修改 |

### 发布话题

| 话题 | 类型 | 说明 |
|---|---|---|
| `/traffic_light/detections` | `std_msgs/String` (JSON) | 检测结果 + 共识状态 |
| `/traffic_light/debug_image` | `sensor_msgs/Image` | 标注框的可视化画面 |
| `/traffic_light/status` | `std_msgs/String` (latched) | 节点状态：`tracking` / `no_image` / `error` / `shutdown` |

### `detections` JSON 格式

```json
{
  "header": {"stamp": 1717623456.789},
  "raw_detections": [
    {
      "class_name": "green_straight",
      "class_id": 2,
      "confidence": 0.93,
      "bbox": [120.5, 80.3, 200.1, 160.7]
    }
  ],
  "consensus": {
    "class_name": "green_straight",
    "class_id": 2,
    "confidence": 0.93,
    "active": true,
    "held_frames": 12
  },
  "status": "tracking",
  "diagnostics": {
    "fps": 8.5,
    "inference_ms": 95.2,
    "error_count": 0
  }
}
```

- `raw_detections` — 当前帧所有检测框（受 `confidence_threshold` 过滤）
- `consensus` — **下游节点只读这个**：
  - `active: true` → 信任 `class_name`
  - `active: false` / `class_name: null` → 当前无有效检测
- `diagnostics` — 运行时诊断信息

---

## 共识滤波机制

消除单帧抖动，确保下游节点收到的分类稳定可靠，满足比赛"红灯停、绿灯行"的实时决策需求。

```
                   N 帧连续命中
  无共识  ──────────────────→  锁定 class X
  (null)  ←──────────────────  (active=true)
               M 帧连续未命中

  锁定期间：
  - 同类检测：置信度 EMA 平滑（alpha=0.3），持续保持
  - 异类高置信检测：确认帧数达标后立即切换（不等释放）
  - 超时未检测（1.0s）：强制释放
```

关键参数：

| 参数 | 默认 | 效果 |
|---|---|---|
| `consensus_confirm_frames` | 5 | 越大越稳，延迟越高（5帧@10Hz≈0.5s） |
| `consensus_release_frames` | 3 | 越大越不容易断开 |
| `consensus_timeout` | 1.0s | 最大保持时间 |
| `consensus_ema_alpha` | 0.3 | 置信度平滑，越小越稳定 |

---

## 推理线程模型

```
Main Thread (rospy.spin)      Daemon Thread (inference loop)
  image_cb() ←── 存最新帧        │
       │                         while not shutdown:
       │                           copy latest frame (lock)
       v                           模型推理
  rospy.on_shutdown → cleanup      共识状态机
                                   发布 detections
                                   发布 debug_image
                                   sleep(1/inference_rate)
```

- 摄像头 30fps，推理 10fps — 自动跳帧，始终用最新帧
- GPU 推理约 10-30ms，CPU 推理约 50-200ms
- CPU 模式自动将输入分辨率从 640 降到 320

---

## 部署步骤

### 1. 放入模型

将训练好的 `best.pt` 放到 `yolo/models/` 目录：

```bash
cp ~/yolov5/runs/train/exp/weights/best.pt ~/catkin_ws/src/yolo/models/
```

### 2. 编译

```bash
cd ~/catkin_ws && catkin_make
source devel/setup.bash
```

### 3. 启动

独立推理（比赛模式）：
```bash
roslaunch yolo traffic_light_inference.launch
```

与采集工具同启（调试时有用，默认不启推理）：
```bash
roslaunch yolo traffic_light_collect.launch start_inference:=true
```

---

## 启动参数

### `traffic_light_inference.launch`

| 参数 | 默认 | 说明 |
|---|---|---|
| `start_camera` | `true` | 是否同时启动 usb_cam |
| `camera_topic` | `/usb_cam/image_raw` | 摄像头话题 |
| `config_file` | `$(find yolo)/config/traffic_light.yaml` | 配置文件路径 |
| `flip` | `false` | 水平翻转图像 |

### `traffic_light_collect.launch`

| 参数 | 默认 | 说明 |
|---|---|---|
| `start_inference` | `false` | 是否同时启动推理 |
| `start_driver` | `true` | 是否启动底盘驱动 |
| `start_mjpeg` | `true` | 是否启动 MJPEG 视频流 |
| `mjpeg_port` | `8080` | 视频流端口 |

---

## 数据集清理

训练完成后，`yolo_dataset/` 目录（286MB）可删除。推理只需 `models/best.pt`。

| 文件/目录 | 用途 | 可否删除 |
|---|---|---|
| `2728601_1780572667/` | COCO 原始标注 + 增广图 | 保留（如需重新训练） |
| `2728601_1780572667.zip` | 同上，压缩包 | 可删（解压后已存在） |
| `yolo_dataset/` | YOLO 格式训练/验证集 | **可删**（可用 coco2yolo.py 重新生成） |
| `models/best.pt` | 训练好的模型权重 | **不可删**（推理必需） |

---

## 实机测试步骤（完整命令）

以下每一步都是一个独立可执行的命令块，按顺序跑即可。所有命令在部署用的小车上执行。

### 前置检查

```bash
# 1. 确认模型文件存在
ls -lh ~/catkin_ws/src/yolo/models/best.pt

# 2. 确认配置文件存在
cat ~/catkin_ws/src/yolo/config/traffic_light.yaml | head -5

# 3. 确认编译通过
cd ~/catkin_ws && catkin_make
source devel/setup.bash

# 4. 确认 ROS 环境已加载
echo $ROS_PACKAGE_PATH | grep yolo
```

### 第一步：启动节点

```bash
# 启动推理 + 摄像头（会自动启动 usb_cam）
roslaunch yolo traffic_light_inference.launch

# 如果摄像头已启动，只启动推理
roslaunch yolo traffic_light_inference.launch start_camera:=false
```

### 第二步：确认节点和话题存活

```bash
# 1. 检查节点是否运行
rosnode list | grep traffic_light_inference

# 2. 检查状态话题
rostopic echo /traffic_light/status
# 期望输出: "tracking"

# 3. 检查摄像头频率（应有 ~30Hz）
rostopic hz /usb_cam/image_raw -w5

# 4. 检查检测话题频率（应有 ~10Hz）
rostopic hz /traffic_light/detections -w5

# 5. 确认话题列表
rostopic list | grep traffic_light
# 期望输出:
# /traffic_light/detections
# /traffic_light/debug_image
# /traffic_light/status
```

### 第三步：查看检测结果

```bash
# 1. 查看完整 JSON（每秒输出一条）
rostopic echo /traffic_light/detections -n1 | python3 -m json.tool

# 2. 只看共识结果（持续监控）
rostopic echo /traffic_light/detections | grep -E '"class_name"|"active"|"held_frames"|"confidence"' | grep -A3 consensus

# 3. 只看原始检测框
rostopic echo /traffic_light/detections | grep -A6 raw_detections

# 4. 查看诊断信息（FPS、推理耗时）
rostopic echo /traffic_light/detections | grep -A3 diagnostics
```

### 第四步：查看标注画面

```bash
# 方式一：rqt_image_view（GUI，推荐）
rqt_image_view /traffic_light/debug_image

# 方式二：image_view（无 rqt 时）
rosrun image_view image_view image:=/traffic_light/debug_image

# 方式三：保存一帧到本地检查
rosrun image_view image_saver image:=/traffic_light/debug_image _save_all_image:=false _filename_format:="debug_frame.jpg"
```

### 第五步：逐类别人工验证

依次将红绿灯切换为四种状态，每切换一次观察输出。

```bash
# 持续监控 consensus 字段
rostopic echo /traffic_light/detections | grep -B1 -A4 '"consensus"'
```

验证清单：

| 灯光状态 | 期望 `class_name` | 期望 `active` | 切换延迟 |
|---|---|---|---|
| 红灯亮 | `red_light` | `true` | ≤ 0.5s |
| 直行绿灯亮 | `green_straight` | `true` | ≤ 0.5s |
| 左转绿灯亮 | `green_left` | `true` | ≤ 0.5s |
| 右转绿灯亮 | `green_right` | `true` | ≤ 0.5s |
| 灯全灭 | `null` | `false` | ≤ 0.3s |

如果某一类始终检测不到，用 debug_image 确认：
```bash
# 看标注框是否正确
rostopic echo /traffic_light/detections -n1 | python3 -m json.tool | grep -A5 raw_detections
```

### 第六步：切换响应延迟定量测试

```bash
# 准备一个脚本记录时间戳
rostopic echo /traffic_light/detections -p > /tmp/detections.csv &
# 此时切换红灯→绿灯
# 等几秒后 Ctrl+C 停止
# 查看 class_name 变化的时间差
cat /tmp/detections.csv | grep consensus | cut -d',' -f1
```

更简单的方法——直接肉眼观察 `held_frames` 变化：

```bash
# held_frames 从 0 涨到 5 时锁定，观察这个过程的耗时
rostopic echo /traffic_light/detections | grep held_frames
```

期望：`5 / inference_rate = 5 / 10 = 0.5s`
如果 `inference_rate` 改为 15Hz：`5 / 15 ≈ 0.33s`

### 第七步：抗干扰测试

```bash
# 1. 先确认无干扰时共识为 null
# （红绿灯关闭）

# 2. 用手机屏幕在摄像头前晃动
# 观察是否误报——raw_detections 可能有框，但 consensus 应保持 null
rostopic echo /traffic_light/detections | grep -A4 consensus

# 3. 红绿灯旁边放一个反光物体（如金属板）
# 确认检测框仍然稳定在 LED 屏幕区域
rqt_image_view /traffic_light/debug_image
```

### 第八步：不同距离测试

分别在 20cm、50cm、1m、2m 处放置红绿灯：

```bash
# 每个距离停留 5s，记录检测状态
for dist in 20 50 100 200; do
    echo "Testing at ${dist}cm..."
    rostopic echo /traffic_light/detections -n10 | grep -c '"active": true'
done
```

### 第九步：节点崩溃恢复测试

```bash
# 1. 查看节点 PID
rosnode info /traffic_light_inference | grep Pid

# 2. 杀掉节点进程
kill -9 <PID>

# 3. 3 秒后用 roslaunch 重新启动（如果在 launch 文件里设置了 required="true"，会自动重启）
rostopic echo /traffic_light/status
# 期望: 重新输出 "tracking"

# 4. 确认检测恢复
rostopic echo /traffic_light/detections -n1 | python3 -m json.tool | grep -A4 consensus
```

### 第十步：长期稳定性测试

```bash
# 跑 5 分钟，记录 error_count 和 fps
rostopic echo /traffic_light/detections -p > /tmp/long_run.csv &
sleep 300 && kill %1

# 分析
cat /tmp/long_run.csv | grep consensus | tail -1
python3 -c "
import csv
with open('/tmp/long_run.csv') as f:
    reader = csv.DictReader(f)
    errors = 0; samples = 0
    for row in reader:
        if row.get('field.field.diagnostics.error_count'):
            e = int(row['field.field.diagnostics.error_count'])
            errors = max(errors, e)
            samples += 1
    print(f'Samples: {samples}, Max error_count: {errors}')
"
# 期望: error_count = 0, samples ≈ 3000 (10Hz × 300s)
```

### 停止线联合测试

与巡线节点联动，模拟比赛完整流程：
1. 启动巡线 launch + 推理 launch
2. 小车沿巡线接近停止线
3. 距停止线 ~30cm 开始减速
4. 读取 `/traffic_light/detections` 的 `consensus`
5. 红灯 → 停稳，车头距停止线 <10cm
6. 灯变绿 → 起步，按箭头方向进入巡线路径

> 停止线检测和巡线控制由 `ucar_2026_line_follow` 包负责，不在本包范围内。
> 需要巡线节点订阅 `/traffic_light/detections` 并加入红绿灯状态机逻辑。

---

## 快速诊断脚本

将以下内容保存为 `~/check_yolo.sh`，一键诊断：

```bash
#!/bin/bash
echo "=== YOLO 节点诊断 ==="
echo ""

echo "[1] 节点状态:"
rosnode list 2>/dev/null | grep traffic_light_inference || echo "  节点未运行!"

echo ""
echo "[2] 话题列表:"
rostopic list 2>/dev/null | grep traffic_light || echo "  无 traffic_light 话题"

echo ""
echo "[3] 摄像头频率 (5秒采样):"
rostopic hz /usb_cam/image_raw -w5 2>&1 | tail -1 || echo "  摄像头未出图"

echo ""
echo "[4] 检测频率 (5秒采样):"
rostopic hz /traffic_light/detections -w5 2>&1 | tail -1 || echo "  检测话题无输出"

echo ""
echo "[5] 当前状态:"
rostopic echo /traffic_light/status -n1 2>/dev/null || echo "  无法获取状态"

echo ""
echo "[6] 最新检测结果:"
rostopic echo /traffic_light/detections -n1 2>/dev/null | python3 -m json.tool 2>/dev/null | grep -A4 consensus || echo "  无法获取检测结果"

echo ""
echo "[7] 模型文件:"
ls -lh ~/catkin_ws/src/yolo/models/best.pt 2>/dev/null || echo "  best.pt 不存在!"
```

使用：
```bash
chmod +x ~/check_yolo.sh
~/check_yolo.sh
```

---

## 实机调参指南

所有参数在 `config/traffic_light.yaml`，改完重启节点即可生效，无需重编译。

### 检测不稳定，频繁闪烁

```yaml
consensus_confirm_frames: 8    # 5→8，更稳但延迟 ~0.3s
consensus_release_frames: 5    # 3→5
```

### 反应太慢，车已过灯

```yaml
consensus_confirm_frames: 3    # 5→3
inference_rate: 15.0           # 10→15 Hz
```

### 误检（非红绿灯物体被识别）

```yaml
confidence_threshold: 0.7      # 0.5→0.7
```

### 漏检（明明有灯但检不出）

```yaml
confidence_threshold: 0.3      # 降低门槛
cpu_reduce_input: false        # 保持 640 分辨率
input_size: 640
```

### 小车上 CPU 太慢

```yaml
inference_rate: 5.0            # 降到 5Hz
cpu_reduce_input: true         # 自动用 320 分辨率
```

---

## 集成到巡线节点

下游节点消费 `/traffic_light/detections`，只需关注 `consensus` 字段：

```python
import json
from std_msgs.msg import String

def detection_cb(msg):
    data = json.loads(msg.data)
    consensus = data["consensus"]

    if not consensus["active"]:
        return  # 无有效检测，保持当前动作

    class_name = consensus["class_name"]
    conf = consensus["confidence"]

    if class_name == "red_light":
        stop_robot()
    elif class_name == "green_straight":
        go_straight()
    elif class_name == "green_left":
        prepare_left_turn()
    elif class_name == "green_right":
        prepare_right_turn()
```

要点：
- 只信任 `active: true` 的结果
- 共识滤波已做平滑，不需要额外去抖
- `held_frames` 越大说明检测越稳定，可作为辅助判断
- 停止线识别由巡线节点负责，本节点只输出红绿灯状态

---

## 比赛合规检查

### 已确认 ✅

| 检查项 | 状态 | 说明 |
|---|---|---|
| class_id 映射一致 | ✅ | 推理节点、data.yaml、COCO标注三者一致 |
| 四种状态覆盖 | ✅ | red_light / green_straight / green_left / green_right |
| 模型 mAP@0.5 | ✅ | 0.993，远超比赛实用要求 |
| 摄像头分辨率 | ✅ | 640×480@30fps，符合 ≤1920×1080 规范 |
| 共识滤波抗抖动 | ✅ | 5帧确认 + EMA平滑，稳定可靠 |
| 故障恢复 | ✅ | CvBridge try-catch、模型重载、超时保护 |
| 训练数据来源 | ✅ | 选手自制红绿灯采集，符合"不提供官方数据集"规则 |
| ROS 框架 | ✅ | 使用 ROS，符合要求 |

### 需注意 ⚠️

| 检查项 | 说明 |
|---|---|
| 红绿灯外观一致性 | 训练模型的 LED 点阵与比赛实际 LED 的外观必须一致（颜色、形状、亮度）。建议用同一块 LED 屏采集数据并测试 |
| 近距识别 | 比赛要求车头距停止线 <10cm，LED 屏在极近处可能变形/过曝。建议在 ~10-50cm 距离范围补充训练数据 |
| 巡线节点未集成 | `ucar_2026_line_follow` 尚未订阅 `/traffic_light/detections`，需在巡线状态机中加入红绿灯判断 |
| 停止线触发 | 停止线由巡线节点基于白色胶带识别，本节点不负责"何时开始看灯"的逻辑 |
| LED 供电电池 | 比赛要求红绿灯电池供电，建议部署前确认 LED 亮度和闪烁频率稳定 |

### 比赛扣分相关

| 扣分项 | 扣分值 | YOLO 节点责任 |
|---|---|---|
| 视觉识别或决策错误 | -10 | core: 确保识别准确 |
| 未按要求制作电子红绿灯 | -10 | 不相关（硬件） |
| 停止线和停车规范（超线/压线） | -5 | 不相关（巡线节点） |

---

## 故障排查

| 现象 | 原因 | 解决 |
|---|---|---|
| 节点启动打印 `Model not found` | `models/best.pt` 不存在 | 把训练好的权重放到 `yolo/models/` |
| `PyTorch not installed` | 虚拟环境没装 torch | `pip install torch torchvision` |
| 检测延迟 >2s | 模型太大或 CPU 扛不住 | 用 YOLOv5n，或降 `input_size` 到 320 |
| 状态一直 `no_image` | 摄像头没出图 | 检查 `rostopic hz /usb_cam/image_raw` |
| `error_count` 持续增长 | 推理反复异常 | 查看日志，换用 CPU 模式 `device: cpu` |
| 共识始终 `null` | 阈值太高或模型不匹配 | 降低 `confidence_threshold`，检查 class_id 映射 |
| debug_image 一片黑 | 摄像头供电/固件 | 检查 USB 连接，`rostopic echo /usb_cam/image_raw` |
| 编译报 `cv_bridge not found` | 缺少依赖 | `sudo apt install ros-noetic-cv-bridge` |
| 实机检测不稳定 | 训练数据与实机场景不匹配 | 在实机上用 `keyboard_collect_yolo_images.py` 补充采集，重新训练 |

---

## 依赖

```
ros-noetic-cv-bridge
ros-noetic-sensor-msgs
ros-noetic-std-msgs
ros-noetic-geometry-msgs
ros-noetic-usb-cam
ros-noetic-ucar-controller
pip: torch torchvision rospkg
```
