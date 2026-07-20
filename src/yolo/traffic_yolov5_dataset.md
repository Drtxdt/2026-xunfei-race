# YOLOv5 红绿灯检测数据集 SOP

本流程训练四类目标检测模型，检测框统一覆盖完整的 16×16 cm 显示面板：

```text
0 green_left
1 green_right
2 green_straight
3 red_light
```

禁止水平翻转。水平翻转会把左箭头变成右箭头，但标签不会自动交换。

## 1. 初始化采集目录

同步代码并编译后，在小车执行：

```bash
cd ~/2026-xunfei-race
catkin_make --pkg yolo
source devel/setup.bash

rosrun yolo init_traffic_dataset.py --root ~/traffic_dataset_raw
```

工具会生成五个 session、LabelImg 类别文件、session 划分和每类配额：

```text
session_01_normal_light       每类120张 -> train
session_02_bright            每类120张 -> train
session_03_dim               每类110张 -> train
session_04_background_changed 每类100张 -> val
session_05_final_field       每类 50张 -> test
```

每类总计 500 张；每个 session 还按相同配额采集 `background`，最终共约 2500 张。

## 2. 用 X11 launch 采集

从 MobaXterm 开启 X11 forwarding 的 SSH 会话，确认 `echo $DISPLAY` 非空。每个任务都有独立 launch，例如：

```bash
source ~/ucar_ws/devel/setup.bash
source ~/2026-xunfei-race/devel/setup.bash
roslaunch yolo traffic_collect_s01_green_left.launch
```

启动后会出现两个 X11 窗口：

- `image_view` 显示未镜像的摄像头原图。
- `xterm` 运行采集器。鼠标点中 xterm 后按 `P` 开始/暂停，`C` 单张拍摄，`Esc` 退出。

采集入口共 25 个：

```text
traffic_collect_s01_green_left.launch
traffic_collect_s01_green_right.launch
traffic_collect_s01_green_straight.launch
traffic_collect_s01_red_light.launch
traffic_collect_s01_background.launch

traffic_collect_s02_green_left.launch
traffic_collect_s02_green_right.launch
traffic_collect_s02_green_straight.launch
traffic_collect_s02_red_light.launch
traffic_collect_s02_background.launch

traffic_collect_s03_green_left.launch
traffic_collect_s03_green_right.launch
traffic_collect_s03_green_straight.launch
traffic_collect_s03_red_light.launch
traffic_collect_s03_background.launch

traffic_collect_s04_green_left.launch
traffic_collect_s04_green_right.launch
traffic_collect_s04_green_straight.launch
traffic_collect_s04_red_light.launch
traffic_collect_s04_background.launch

traffic_collect_s05_green_left.launch
traffic_collect_s05_green_right.launch
traffic_collect_s05_green_straight.launch
traffic_collect_s05_red_light.launch
traffic_collect_s05_background.launch
```

`s01/s02/s03/s04/s05` 每类分别自动限制为 `120/120/110/100/50` 张，并保存到对应 session。一个任务结束后先用 `Ctrl-C` 停止当前 roslaunch，再启动下一个，避免重复启动摄像头。

### 手动参数方式（备用）

终端一只启动摄像头，不启动底盘：

```bash
source ~/ucar_ws/devel/setup.bash
source ~/2026-xunfei-race/devel/setup.bash
roslaunch yolo traffic_light_collect.launch \
  start_driver:=false start_inference:=false flip:=false
```

终端二按 session、类别分别采集，例如：

```bash
rosrun yolo keyboard_collect_yolo_images.py \
  --cls green_left \
  --output ~/traffic_dataset_raw/session_01_normal_light \
  --interval 0.8 --max-images 120 --start-paused
```

合法类别：

```text
green_left green_right green_straight red_light background
```

脚本默认阻止 `--flip`，并在 session 根目录记录 `_capture_manifest.json`。使用 `--start-paused` 时启动后处于暂停状态；调整好灯态、距离和车头角度后按 `P` 开始。

采集分布：

- 70% 位于 1.3–1.7 m，重点覆盖 1.5 m。
- 15% 位于 0.8–1.3 m。
- 15% 位于 1.7–2.2 m。
- 约 60% 在画面中心，40% 覆盖上下左右偏移。
- 覆盖约 `-15°、-8°、0°、8°、15°` 车头偏航。
- 分 session 改变亮度、背景、曝光和采集时段。
- 删除灯态切换、LED 扫描残缺、严重模糊和遮挡超过 50% 的帧。

`background` 包含灯牌熄灭、无灯牌、人员、反光、红绿杂物、显示器和严重遮挡，不能包含可正常识别的四种灯态。采集脚本会自动为每张 background 图片建立同名空 `.txt`。

## 3. LabelImg 标注

使用初始化器生成的：

```text
~/traffic_dataset_raw/predefined_classes.txt
```

在 LabelImg 中选择 YOLO 格式。每次打开一个 `session/class` 图片目录，并把标签保存到同一目录，确保图片和 `.txt` 同名相邻。

标注规则：

- 框完整正方形显示面板，不只框发光箭头。
- 不包含支架、墙面、线缆和大面积背景。
- 边缘统一留约 2–4 像素。
- 正样本通常恰好一个框，主灯牌类别必须与目录一致。若意外出现多个真实灯牌，每个都按实际状态标注。
- 左右类按图片中箭头实际指向标注。
- `background` 不画框，但必须保留采集脚本生成的同名空 `.txt`。

## 4. 构建和质检

把原始目录复制到标注电脑后执行：

```bash
rosrun yolo build_traffic_yolo_dataset.py \
  --raw-root ~/traffic_dataset_raw \
  --output ~/traffic_yolov5
```

输出目录必须不存在或为空，工具不会覆盖已有数据。构建前会检查：

- 640×480 图像能否解码。
- 正样本标签是否存在、是否非空、类别是否与文件夹一致。
- class id、列数、归一化坐标和边界是否合法。
- 每张图片是否有同名 `.txt`，background 标签是否为空。
- train/val/test 是否按 session 隔离。
- 是否存在跨 split 的 SHA256 完全重复图片。
- 相邻图片是否为疑似近重复帧。
- 框是否小于 20 px 或大于 200 px，需要人工复核。
- 每个类在 train/val/test 是否分别为 350/100/50 张。

构建成功后重点查看：

```text
~/traffic_yolov5/data.yaml
~/traffic_yolov5/dataset_report.json
~/traffic_yolov5/near_duplicates.csv
~/traffic_yolov5/qa/*.jpg
```

`qa` 会按 split 和类别生成所有图片的分页标注框联系表。左右转联系表必须全量快速检查。

只有在采集中途做草稿质检时，才可传入 `--allow-incomplete-counts`；最终数据集不得使用该参数。

## 5. 训练

在 YOLOv5 仓库中使用 COCO 预训练的 YOLOv5s：

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 150 \
  --data ~/traffic_yolov5/data.yaml \
  --weights yolov5s.pt \
  --hyp ~/2026-xunfei-race/src/yolo/config/traffic_yolov5_hyp.yaml \
  --name traffic_light_yolov5s_640
```

显存不足时只降低 `--batch`，不要降低 `--img`。超参数文件已固定：

- `fliplr=0.0`
- `flipud=0.0`
- 旋转 ±5°
- Mosaic 0.5
- 轻量 HSV、缩放和平移增强

## 6. 验收

- test 集 `mAP@0.5 ≥ 0.95`。
- 四类 precision、recall 分别不低于 0.95。
- background 误检图片比例不超过 2%。
- 单独检查红灯误判为绿灯、左转与右转混淆。
- 实车 1.5 m 下每类连续测试至少 100 帧，正确检测率不低于 95%。
- PyTorch、ONNX、RKNN 使用同一批图片逐级对比类别和框，确认一致后部署。
