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

模型输入为中心区域裁剪后的 BGR 图像，resize 到 `224x224`，输出三类 logits。节点会对 `electronic` 类做一个可调 logit 偏置，再用 softmax 得到置信度，并用 top1-top2 margin 过滤弱判断：

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
classifier_input_color: bgr
classifier_crop_mode: square
vote_window_size: 5
vote_min_count: 2
use_score_voting: true
score_vote_min_score: 0.42
score_vote_min_margin: 0.08
classifier_confidence_threshold: 0.50
classifier_min_margin: 0.15
classifier_daily_logit_bias: 0.45
classifier_food_logit_bias: 0.00
classifier_electronic_logit_bias: -0.60
publish_debug_image: true
debug_image_topic: /factory_sign_ocr_test/debug_image
debug_preprocess_topic: /factory_sign_ocr_test/preprocess_image
```

稳定性规则：

- 默认连续 5 帧做 softmax 分数平均，而不是只统计每帧 argmax。
- 平均最高类分数不少于 `score_vote_min_score`，且领先第二名不少于 `score_vote_min_margin` 即确认。
- 可用 `use_score_voting:=false` 回退到原来的硬投票：某类别出现次数不少于 2 次即确认。
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
- RKNN raw logits / 校准后 logits / probs / margin
- score_vote 窗口平均分和平均 margin
- 归类结果
- softmax 置信度
- 投票窗口
- 是否播报

如果所有画面都稳定识别成同一类，按顺序做 A/B，不要同时改多个参数：

1. 如果三类互相抢票，先调 soft vote，不要继续只拧 bias。日用品弱但持续有分数时，降低平均分门限：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch score_vote_min_score:=0.35
```

如果误报变多，提高平均 margin：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch score_vote_min_margin:=0.12
```

2. 如果日用品平均分仍然长期低于另外两类，再抬高 daily：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_daily_logit_bias:=0.70
```

如果日用品开始误报，再回调：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_daily_logit_bias:=0.25
```

3. 如果食品、日用品能出，但电子过强，把电子偏置调得更负：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_electronic_logit_bias:=-1.0
```

如果真实电子牌识别太困难，把偏置往 0 调：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_electronic_logit_bias:=-0.3
```

4. 再试 RGB，排除颜色通道差异。当前小车实测 BGR 更可靠，所以默认是 BGR：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_input_color:=rgb
```

5. 调单帧 classifier margin。误报多就提高，漏报多就降低：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_min_margin:=0.25
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_min_margin:=0.05
```

6. 再试全图，排除中心方形裁剪切掉纸牌文字：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_crop_mode:=full
```

7. 最后才试 `nchw` 布局。RKNNLite 在当前模型上会提示需要 NHWC，默认不要改：

```bash
roslaunch factory_sign_ocr_test factory_sign_ocr_test.launch classifier_input_layout:=nchw
```

注意：`classifier_preprocess_mode:=torch` 已禁用。现场日志已经证明该 float pass-through 路径可能导致 RKNNLite 段错误退出，不能作为一键测试路径。







