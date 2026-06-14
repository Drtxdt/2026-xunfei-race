# RKNN Traffic Light Test

This package tests `yolo/models/best_640.rknn` on the car NPU, shows the annotated
camera stream through X11, and announces stable traffic-light decisions.

## Start

On MobaXterm, enable X11 forwarding, SSH into the car, then run:

```bash
source /home/ucar/ucar_ws/devel/setup.bash
source ~/2026-xunfei-race/devel/setup.bash
roslaunch ucar_2026_traffic_light_rknn_test traffic_light_rknn_x11_speak_test.launch
```

If the camera or speech nodes are already running:

```bash
roslaunch ucar_2026_traffic_light_rknn_test traffic_light_rknn_x11_speak_test.launch \
  start_camera:=false start_tts:=false start_competition_speech:=false
```

## Topics

- `/traffic_light_rknn_test/detections`: JSON detection and consensus state.
- `/traffic_light_rknn_test/debug_image`: annotated image for `image_view`.
- `/traffic_light_rknn_test/status`: node status.

## Diagnostics

```bash
rosrun ucar_2026_traffic_light_rknn_test check_traffic_light_rknn_test.py
rostopic echo /traffic_light_rknn_test/detections -n1
rostopic hz /traffic_light_rknn_test/detections -w5
```
