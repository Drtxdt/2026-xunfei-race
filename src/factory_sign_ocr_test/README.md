# factory_sign_ocr_test

独立 ROS1 测试包，用于“小车摄像头识别厂牌并语音播报”。它不接入导航、避障或任务主流程，只订阅摄像头图像，识别到以下任一厂牌后播报一次：

- 食品加工车间
- 日用品加工车间
- 电子产品生产车间

## 当前推荐方案

小车 Python 是 3.7.3，Tesseract 对现场纸牌、字体变化、光照和倾斜比较敏感，所以本包现在默认使用：

1. RKNN 厂牌三分类优先：自动查找 `yolo/models/factory_sign_3cls.rknn`，直接识别 `food/electronic/daily`。
2. OCR 兜底：RKNN 未识别到时再尝试 OCR 文本。
3. RapidOCR 优先于 Tesseract：更适合 Python 3.7.3 的轻量 CPU OCR。
4. Tesseract 仅最后兜底：会用多种预处理图和 `psm 6/7/11` 合并结果。

这比“纯 Tesseract OCR”稳定，因为任务只有 3 类固定厂牌，不需要逐字完整识别。

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

默认会启动 USB 摄像头、旧 TTS、统一播报服务、识别节点和 X11 调试窗口：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch
```

如果摄像头或语音节点已经单独启动：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch \
  start_camera:=false start_tts:=false start_competition_speech:=false
```

强制只用 RKNN 三分类：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch recognition_mode:=rknn_classifier
```

强制只用 RapidOCR/Tesseract：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch recognition_mode:=rapidocr
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch recognition_mode:=tesseract cpu_ocr_engine:=tesseract
```

## MobaXterm X11 实时调试

参考工程里的红绿灯和巡线包，本包会发布 ROS 调试图像，再用 `image_view` 通过 X11 显示。

1. MobaXterm 新建 SSH Session 时勾选 X11 forwarding。
2. SSH 登录小车后检查：

```bash
echo $DISPLAY
```

能看到类似 `localhost:10.0` 才说明 X11 转发可用。

3. 启动一键 launch：

```bash
source ~/ucar_ws/devel/setup.bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch
```

默认窗口显示：

```text
/factory_sign_ocr_test/debug_image
```

图像上会画出中心 ROI、RKNN 检测框、识别来源、类别、置信度、投票窗口、是否播报。

切换查看预处理图：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch debug_view:=preprocess
```

切换查看原始摄像头：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch debug_view:=camera
```

如果 launch 没弹出窗口，可以手动打开：

```bash
rosrun image_view image_view image:=/factory_sign_ocr_test/debug_image
rosrun image_view image_view image:=/factory_sign_ocr_test/preprocess_image
```

确认话题频率：

```bash
rostopic hz /factory_sign_ocr_test/debug_image
rostopic hz /factory_sign_ocr_test/preprocess_image
```

## OCR 依赖

优先推荐 RKNN 分类模型，不依赖 CPU OCR。若要用 OCR fallback，在 Python 3.7.3 上优先安装 RapidOCR：

```bash
python3 -m pip install rapidocr_onnxruntime==1.3.24
```

如果 RapidOCR 装不上，再尝试 EasyOCR：

```bash
python3 -m pip install easyocr
```

Tesseract 兜底：

```bash
sudo apt install tesseract-ocr tesseract-ocr-chi-sim
python3 -m pip install pytesseract
```

不建议在 Python 3.7.3 上直接装最新版 PaddleOCR；新版 PaddleOCR 已经偏向 Python 3.8+ 环境。

## 配置说明

主要参数在 `config/factory_sign_ocr.yaml`：

```yaml
recognition_mode: auto
classifier_model_path: ""
classifier_confidence_threshold: 0.5
cpu_ocr_engine: auto
publish_debug_image: true
debug_image_topic: /factory_sign_ocr_test/debug_image
debug_preprocess_topic: /factory_sign_ocr_test/preprocess_image
```

关键词归类规则：

- 食品加工车间：命中 `食品`、`食`、`food`
- 日用品加工车间：命中 `日用品`、`日用`、`daily`
- 电子产品生产车间：命中 `电子`、`电`、`electronic`

稳定性规则：

- 连续 5 帧投票。
- 某类别出现次数不少于 2 次即确认。
- 同类默认冷却 5 秒。
- 类别变化立即播报。

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

运行识别节点时，终端会打印：

- 识别来源：`rknn` 或 `ocr`
- OCR 原始文本
- 归类结果
- RKNN 置信度
- 投票窗口
- 是否播报
