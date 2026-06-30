# factory_sign_ppocr_test

独立 ROS1 测试包，用本地 PaddleOCR PP-OCRv5 做厂牌文字识别，并复用项目已有语音接口播报结果。

本包不接入导航、避障或任务主流程；不调用除比赛允许的讯飞能力之外的任何云 API。PaddleOCR 在小车本地 Python 环境中离线运行。

## 识别目标

- 食品加工车间
- 日用品加工车间
- 电子产品生产车间

OCR 文本使用关键词归类：

- 食品：`食品` / `食` / `food`
- 日用品：`日用品` / `日用` / `daily`
- 电子产品：`电子` / `电` / `electronic`

## 架构

ROS 节点运行在小车现有 Python3.7.3 环境中，只负责摄像头、ROI、投票、调试图和语音。

PaddleOCR 放到独立本地 Python 环境中，通过 `ppocr_worker.py` 子进程运行。主节点和 worker 用 stdin/stdout JSON Lines 通信，避免 PaddleOCR 3.x 依赖破坏 ROS Python3.7.3。

默认 worker Python：

```bash
/home/ucar/ppocrv6_env/bin/python3
```

可通过 launch 覆盖：

```bash
roslaunch factory_sign_ppocr_test factory_sign_ppocr_test.launch \
  paddle_python:=/path/to/ppocr/env/bin/python3
```

## 编译

```bash
cd ~/2026-xunfei-race
catkin_make
source devel/setup.bash
```

## 检查本地 PaddleOCR 环境

```bash
rosrun factory_sign_ppocr_test check_ppocr_env.py \
  --python /home/ucar/ppocrv6_env/bin/python3
```

如果失败，先在小车本地准备 PaddleOCR 环境。不要把 PaddleOCR 3.x 强行装进 ROS Python3.7.3。

示例方向：

```bash
python3.8 -m venv /home/ucar/ppocrv6_env
/home/ucar/ppocrv6_env/bin/python3 -m pip install paddlepaddle paddleocr
```

具体版本以小车系统、架构和 Paddle 官方 wheel 支持为准。

## 一键启动

默认启动 USB 摄像头、比赛语音服务、PP-OCR 节点和 MobaXterm X11 调试窗口：

```bash
roslaunch factory_sign_ppocr_test factory_sign_ppocr_test.launch
```

如果摄像头或语音服务已启动：

```bash
roslaunch factory_sign_ppocr_test factory_sign_ppocr_test.launch \
  start_camera:=false start_competition_speech:=false
```

## MobaXterm X11 实时调试

MobaXterm SSH Session 勾选 X11 forwarding，登录小车后检查：

```bash
echo $DISPLAY
```

默认窗口显示：

```text
/factory_sign_ppocr_test/debug_image
```

切换窗口：

```bash
roslaunch factory_sign_ppocr_test factory_sign_ppocr_test.launch debug_view:=camera
roslaunch factory_sign_ppocr_test factory_sign_ppocr_test.launch debug_view:=preprocess
```

手动打开：

```bash
rosrun image_view image_view image:=/factory_sign_ppocr_test/debug_image
rosrun image_view image_view image:=/factory_sign_ppocr_test/preprocess_image
```

## 主要配置

`config/factory_sign_ppocr.yaml`：

```yaml
image_topic: /usb_cam/image_raw
paddle_python: /home/ucar/ppocrv6_env/bin/python3
ocr_model_name: PP-OCRv5
ocr_lang: ch
ocr_min_score: 0.45
ocr_timeout_sec: 2.0
inference_rate: 2.0
roi_scale: 0.8
resize_scale: 1.6
vote_window_size: 5
vote_min_count: 2
cooldown_sec: 5.0
speech_service: /competition_speech/announce
speech_topic: /speak
```

## 语音接口

首选：

- service：`/competition_speech/announce`
- type：`ucar_2026_competition_speech/Announce`

兜底：

- topic：`/speak`
- type：`std_msgs/String`

手动测试：

```bash
rosservice call /competition_speech/announce "event: 'custom'
item: ''
workshop: ''
decision: ''
text: '识别到食品加工车间'
wait: false"
```

## 日志

节点会打印：

- OCR 原始文本
- 归类结果
- 投票窗口
- worker 耗时
- 是否播报
- PaddleOCR worker 错误

如果 PaddleOCR 环境不存在或导入失败，ROS 节点不会崩溃，会持续提示本地环境错误并等待修复。
