# 红绿灯 ResNet18 五分类训练

## 数据方向

原始摄像头和 X11 画面是水平镜像的。抽查证明原始 `green_left` 像素中箭头朝右，`green_right` 像素中箭头朝左。

已整理的分类数据集对每张图只做一次水平翻转，使图片箭头与物理灯态标签一致。原始数据保留不动。

```text
E:\2026讯飞\traffic_dataset_raw             原始镜像数据，不要用于训练
E:\2026讯飞\traffic_dataset_cls_corrected   翻正后的五分类数据
```

分类索引固定为：

```text
0 green_left
1 green_right
2 green_straight
3 red_light
4 background
```

## 数据划分

不随机拆散连续帧，而是按采集 session 整体划分：

```text
train: session_01 + session_02 + session_03 = 每类350，共1750
val:   session_04                         = 每类100，共 500
test:  session_05                         = 每类 50，共 250
```

## 裁剪和输入

因为灯牌竖直位置稳定、水平位置不保证，不做左右裁剪：

```text
640×480 翻正原图
→ 保留全宽 x=0:640
→ 裁纵向 y=86:346（归一化 0.18:0.72）
→ 缩放为 width=320, height=160
→ BGR 转 RGB
→ ImageNet mean/std 归一化
```

训练时可以使用小幅亮度、对比度、颜色和仿射增强，但禁止任何水平或垂直随机翻转。

可在以下目录检查实际裁剪效果：

```text
E:\2026讯飞\traffic_dataset_cls_corrected\crop_preview
```

## 开始训练

在 Windows PowerShell 进入项目根目录：

```powershell
python src\yolo\scripts\train_traffic_resnet18.py `
  --data "E:\2026讯飞\traffic_dataset_cls_corrected" `
  --output "E:\2026讯飞\traffic_resnet18_run" `
  --epochs 60 `
  --batch-size 32 `
  --workers 4 `
  --export-onnx
```

默认使用 COCO/ImageNet 无关的 ImageNet ResNet18 预训练权重。首次运行如果本机没有缓存，torchvision 会下载权重。显存不足时先把 `--batch-size` 改为 16，不要降低输入尺寸。

主要输出：

```text
best.pt                   最佳 PyTorch 权重和预处理元数据
traffic_resnet18.onnx     用于后续 RKNN 转换
test_report.json          test 总准确率和每类 precision/recall
test_confusion.csv        五分类混淆矩阵
best_val_confusion.csv    最佳 epoch 的验证集混淆矩阵
history.csv               训练曲线原始数据
training_config.json      类别顺序与预处理参数
```

## 上车必须一致

训练图片已经翻正，而车上摄像头输出仍是镜像。因此部署节点必须按下列顺序：

```python
corrected = cv2.flip(camera_bgr, 1)
height = corrected.shape[0]
roi = corrected[round(height * 0.18):round(height * 0.72), :]
resized = cv2.resize(roi, (320, 160))
rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
```

不得对已翻正的数据集再翻转，也不得在车端翻转两次。
