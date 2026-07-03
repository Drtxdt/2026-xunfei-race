# factory_sign_ppocr_rknn_test

独立 ROS1 测试包，用 Rockchip RKNNLite 跑 PPOCR 厂牌文字识别，并复用项目已有语音播报接口。

这个包不依赖 PaddleOCR / PaddlePaddle Python 运行时。当前车上的 PaddleOCR 3.7.0 已经出现初始化后推理异常，所以本包改走 `rknnlite.api.RKNNLite`。

## 模型来源

推荐使用 Rockchip 官方 `rknn_model_zoo`：

```bash
git clone https://github.com/airockchip/rknn_model_zoo.git
```

使用 `examples/PPOCR` 里的模型转换流程生成 RK3588 模型：

```bash
python convert.py ../model/ppocrv4_det.onnx rk3588
python convert.py ../model/ppocrv4_rec.onnx rk3588 fp
```

把文件放到：

```text
factory_sign_ppocr_rknn_test/models/ppocrv4_det.rknn
factory_sign_ppocr_rknn_test/models/ppocrv4_rec.rknn
factory_sign_ppocr_rknn_test/models/ppocr_keys_v1.txt
```

如果只用默认 `rec_only` 模式，可以先只放 `ppocrv4_rec.rknn` 和 `ppocr_keys_v1.txt`。

## 编译

```bash
cd ~/2026-xunfei-race
catkin_make
source devel/setup.bash
```

## 检查 RKNN 环境

```bash
rosrun factory_sign_ppocr_rknn_test check_ppocr_rknn_env.py --mode rec_only
```

检测+识别全链路：

```bash
rosrun factory_sign_ppocr_rknn_test check_ppocr_rknn_env.py --mode system
```

## 单张图片测试

```bash
rosrun factory_sign_ppocr_rknn_test test_image_ppocr_rknn.py \
  --image /tmp/factory_sign.jpg \
  --mode ppocr_rknn_rec_only
```

## 一键启动

默认启动 USB 摄像头、比赛语音服务、RKNN OCR 节点和 MobaXterm X11 调试窗口：

```bash
roslaunch factory_sign_ppocr_rknn_test factory_sign_ppocr_rknn_test.launch
```

如果摄像头或语音服务已启动：

```bash
roslaunch factory_sign_ppocr_rknn_test factory_sign_ppocr_rknn_test.launch \
  start_camera:=false start_competition_speech:=false
```

切换到检测+识别：

```bash
roslaunch factory_sign_ppocr_rknn_test factory_sign_ppocr_rknn_test.launch \
  recognition_mode:=ppocr_rknn_system
```

## MobaXterm X11 调试

默认窗口显示：

```text
/factory_sign_ppocr_rknn_test/debug_image
```

切换窗口：

```bash
roslaunch factory_sign_ppocr_rknn_test factory_sign_ppocr_rknn_test.launch debug_view:=camera
roslaunch factory_sign_ppocr_rknn_test factory_sign_ppocr_rknn_test.launch debug_view:=preprocess
```

## 日志重点

节点会输出：

- OCR 原始文本
- 关键词分类结果
- 投票窗口
- `elapsed_ms / det_ms / rec_ms`
- 是否播报
- RKNN 模型或字典缺失错误

成功后应看到类似：

```text
factory_sign_ppocr_rknn: text='食品加工车间' category=food conf=... vote=[...] confirmed=food spoken=True elapsed_ms=...
```
