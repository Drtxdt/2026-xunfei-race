# factory_sign_ocr_test

独立 ROS1 测试包，用于“小车摄像头识别厂牌文字并语音播报”。它不接入导航、避障或任务主流程，只订阅摄像头图像，识别到以下任一厂牌后播报一次：

- 食品加工车间
- 日用品加工车间
- 电子产品生产车间

## 已复用的语音接口

当前工程中已找到并复用：

- 推荐 service：`/competition_speech/announce`
- service 类型：`ucar_2026_competition_speech/Announce`
- 调用方式：`event="custom"`，`text="识别到食品加工车间"`，`wait=false`
- 兜底 topic：`/speak`
- topic 类型：`std_msgs/String`
- 旧 TTS 节点：`speech_command/voice_speak_node`
- 统一播报节点：`ucar_2026_competition_speech/scripts/competition_announcer.py`

## 编译

```bash
cd ~/ucar_ws
catkin_make
source devel/setup.bash
```

如果当前目录就是工作空间根目录：

```bash
catkin_make
source devel/setup.bash
```

## 一键启动

默认会启动 USB 摄像头、旧 TTS、统一播报服务和 OCR 测试节点：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch
```

常用参数：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch image_topic:=/usb_cam/image_raw debug:=true
```

如果摄像头或语音节点已经单独启动：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch start_camera:=false start_tts:=false start_competition_speech:=false
```

## 单独启动摄像头

```bash
roslaunch usb_cam usb_cam-test.launch
```

确认图像：

```bash
rostopic hz /usb_cam/image_raw
```

## 测试方法

把纸牌放在小车摄像头正前方。节点默认裁剪画面中心 80% 区域，连续识别 5 帧；只要某类别出现次数不少于 2 次，就认为识别成功。

关键词归类规则：

- 食品加工车间：命中 `食品`、`食`、`food`
- 日用品加工车间：命中 `日用品`、`日用`、`daily`
- 电子产品生产车间：命中 `电子`、`电`、`electronic`

识别成功后：

- 食品：播报 `识别到食品加工车间`
- 日用品：播报 `识别到日用品加工车间`
- 电子产品：播报 `识别到电子产品生产车间`

同一类别默认冷却 5 秒；类别变化会立即播报。

## OCR 依赖

节点会优先尝试 `rknn_model_path` 指定的 OCR RKNN 模型。当前工程中发现的 `src/yolo/models/factory_sign_3cls.rknn` 是厂牌三分类/检测模型，不是 OCR 文字识别模型，所以默认会进入 CPU OCR fallback。

CPU OCR 会按顺序尝试：

```bash
python3 -m pip install paddleocr paddlepaddle
```

或：

```bash
python3 -m pip install easyocr
```

或：

```bash
sudo apt install tesseract-ocr tesseract-ocr-chi-sim
python3 -m pip install pytesseract
```

如果这些库都不存在，节点不会崩溃，会在 ROS 日志中打印清晰安装提示。

## 切换 RKNN / CPU OCR

配置文件：

```yaml
use_rknn: true
rknn_model_path: ""
cpu_ocr_engine: auto
```

启动时强制 CPU：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch use_rknn:=false
```

指定 OCR RKNN 模型：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch rknn_model_path:=/path/to/ocr_model.rknn
```

## 确认语音是否调用成功

检查统一播报服务：

```bash
rosservice list | grep competition_speech
rosservice call /competition_speech/announce "event: 'custom'
item: ''
workshop: ''
decision: ''
text: '识别到食品加工车间'
wait: false"
```

检查旧 TTS topic：

```bash
rostopic info /speak
rostopic pub -1 /speak std_msgs/String "data: '识别到食品加工车间'"
```

运行 OCR 节点时，终端会打印：

- OCR 原始文本
- 归类结果
- 投票窗口
- 是否播报
