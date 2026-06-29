# factory_sign_ocr_test

独立 ROS1 测试包，用于“小车摄像头识别厂牌并语音播报”。它不接入导航、避障或任务主流程，只订阅摄像头图像，识别到以下任一厂牌后播报一次：

- 食品加工车间
- 日用品加工车间
- 电子产品生产车间

## 当前推荐方案

本包现在默认只加载一个 RK3588 RKNNLite 三分类模型：

```text
factory_sign_ocr_test/models/factory_sign_cls_rk3588.rknn
```

模型输入为中心区域裁剪后的 RGB 图像，resize 到 `224x224`，输出三类 logits，再用 softmax 得到置信度：

- `daily`：日用品加工车间
- `electronic`：电子产品生产车间
- `food`：食品加工车间

当前默认不加载 OCR，不加载旧 YOLO 检测模型，也不依赖 `yolo/validate_model.py`。

## 已复用的语音接口

当前工程中已找到并复用：

- 推荐 service：`/competition_speech/announce`
- service 类型：`ucar_2026_competition_speech/Announce`
- 调用方式：`event="custom"`，`text="识别到食品加工车间"`，`wait=false`
- 兜底 topic：`/speak`
- topic 类型：`std_msgs/String`
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

默认会启动 USB 摄像头、统一播报服务、RKNNLite 分类识别节点和 X11 调试窗口。OCR fallback 默认关闭，节点不会加载 OCR：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch
```

如果摄像头或语音节点已经单独启动，或只想调识别：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch \
  start_camera:=false start_competition_speech:=false
```

强制只用包内 RKNN 三分类（默认就是这个模式）：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch recognition_mode:=rknn_classifier
```

指定模型路径：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch \
  classifier_model_path:=$(rospack find factory_sign_ocr_test)/models/factory_sign_cls_rk3588.rknn
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

图像上会画出中心 ROI、识别来源、类别、softmax 置信度、投票窗口、是否播报。

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

## NPU 依赖

小车端需要能导入 `rknnlite.api.RKNNLite`。确认方式：

```bash
python3 -c "from rknnlite.api import RKNNLite; print('rknnlite ok')"
```

本包不需要安装 PaddleOCR、EasyOCR、RapidOCR 或 Tesseract。

## 配置说明

主要参数在 `config/factory_sign_ocr.yaml`：

```yaml
recognition_mode: rknn_classifier
classifier_model_path: ""
classifier_input_size: 224
classifier_input_layout: nhwc
classifier_input_color: rgb
classifier_crop_mode: square
classifier_preprocess_mode: rknn
classifier_confidence_threshold: 0.50
publish_debug_image: true
debug_image_topic: /factory_sign_ocr_test/debug_image
debug_preprocess_topic: /factory_sign_ocr_test/preprocess_image
```

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

检查兜底播报 topic：

```bash
rostopic info /speak
rostopic pub -1 /speak std_msgs/String "data: '识别到食品加工车间'"
```

运行识别节点时，终端会打印：

- 识别来源：`rknn_cls`
- RKNN logits/probs
- 归类结果
- softmax 置信度
- 投票窗口
- 是否播报

如果所有画面都稳定识别成同一类，按顺序做 A/B，不要同时改多个参数：

1. 先试手动 PyTorch 归一化，排除 RKNN 内置 mean/std 语义差异：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch \
  classifier_preprocess_mode:=torch classifier_input_layout:=nchw
```

2. 再试 BGR，排除颜色通道差异：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_input_color:=bgr
```

3. 再试全图，排除中心方形裁剪切掉纸牌文字：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_crop_mode:=full
```

4. 最后才试 `nchw` 布局：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_input_layout:=nchw
```







