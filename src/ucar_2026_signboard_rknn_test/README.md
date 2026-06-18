# RKNN Signboard Test

This package tests `yolo/models/cls_best.rknn` on the car NPU, shows the annotated
camera stream through X11, and announces stable signboard decisions.

## Class Labels

| Class Name | Chinese | Description |
|-----------|---------|-------------|
| `food_processing` | 食品加工车间 | Food processing workshop |
| `daily_necessities` | 日用品加工车间 | Daily necessities workshop |
| `electronics` | 电子产品生产车间 | Electronics production workshop |

## Start

On MobaXterm, enable X11 forwarding, SSH into the car, then run:

```bash
source /home/ucar/ucar_ws/devel/setup.bash
source ~/2026-xunfei-race/devel/setup.bash
roslaunch ucar_2026_signboard_rknn_test signboard_rknn_x11_speak_test.launch
```

If the camera or speech nodes are already running:

```bash
roslaunch ucar_2026_signboard_rknn_test signboard_rknn_x11_speak_test.launch \
  start_camera:=false start_tts:=false start_competition_speech:=false
```

You can also override the model path directly:

```bash
roslaunch ucar_2026_signboard_rknn_test signboard_rknn_x11_speak_test.launch \
  model_path:=/home/ucar/Downloads/cls_best.rknn
```

## Topics

- `/signboard_rknn_test/detections`: JSON detection and consensus state.
- `/signboard_rknn_test/debug_image`: annotated image for `image_view`.
- `/signboard_rknn_test/status`: node status.

## Parameters

Key parameters (see `config/signboard_rknn_test.yaml` for full list):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `~model_path` | `$(find yolo)/models/cls_best.rknn` | RKNN model file |
| `~input_size` | `640` | Model input resolution |
| `~confidence_threshold` | `0.5` | Detection confidence threshold |
| `~nms_iou_threshold` | `0.45` | NMS IoU threshold |
| `~inference_rate` | `10.0` | Inference loop rate (Hz) |
| `~flip` | `false` | Horizontal flip of input image |
| `~consensus_confirm_frames` | `3` | Frames required to lock a class |
| `~consensus_release_frames` | `3` | Frames required to release a locked class |
| `~enable_speech` | `true` | Enable voice announcements |
| `~repeat_same` | `false` | Re-announce the same class repeatedly |
| `~min_speech_interval_sec` | `2.0` | Minimum interval between announcements |

## Diagnostics

```bash
rosrun ucar_2026_signboard_rknn_test check_signboard_rknn_test.py
rostopic echo /signboard_rknn_test/detections -n1
rostopic hz /signboard_rknn_test/detections -w5
```

## Package Structure

```
ucar_2026_signboard_rknn_test/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── signboard_rknn_test.yaml
├── launch/
│   └── signboard_rknn_x11_speak_test.launch
└── scripts/
    ├── check_signboard_rknn_test.py
    └── signboard_rknn_test_node.py
```

## Notes

- The signboard detector uses the same RKNN inference pipeline as the traffic-light test,
  supporting both single-output ONNX-Detect format and 3-head YOLOv5 format.
- Consensus filtering prevents flickering: a class must be detected for `confirm_frames`
  consecutive frames before speech triggers, and persists for `release_frames` frames
  after disappearance.
- Speech uses the competition announcer service (`/competition_speech/announce`) when
  available, falling back to the `/speak` topic otherwise.
