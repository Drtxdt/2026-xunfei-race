# ResNet18 RKNN Traffic Light Test

This package runs the five-class `traffic_resnet18_rk3588_int8.rknn` classifier
on the car NPU. Classes are `green_left`, `green_right`, `green_straight`,
`red_light`, and `background`.

The raw USB-camera image is mirrored. The node flips it once, keeps the full
width, crops normalized vertical range `0.18:0.72` (pixel rows `86:346` for a
640x480 frame), resizes to 320x160, and sends RGB NHWC `uint8` input to RKNN.
Mean/std normalization is embedded in the model and must not be repeated in ROS.

## Start

On MobaXterm, enable X11 forwarding, SSH into the car, then run the unchanged
command:

```bash
source /home/ucar/ucar_ws/devel/setup.bash
source ~/2026-xunfei-race/devel/setup.bash
roslaunch ucar_2026_traffic_light_rknn_test traffic_light_rknn_x11_speak_test.launch
```

The X11 view is the corrected, non-mirrored image. It shows the crop, five
probabilities, inference time, rejection reason, and final consensus. The lamp
should be inside the blue full-width crop at the 1.5 m stop line.

If shared camera or speech nodes are already running:

```bash
roslaunch ucar_2026_traffic_light_rknn_test traffic_light_rknn_x11_speak_test.launch \
  start_camera:=false start_tts:=false start_competition_speech:=false
```

To test the FP16 fallback explicitly:

```bash
roslaunch ucar_2026_traffic_light_rknn_test traffic_light_rknn_x11_speak_test.launch \
  model_path:=$(rospack find yolo)/models/traffic_resnet18_rk3588_fp16.rknn \
  model_quantization:=fp16
```

## Tuning and safety

- Default acceptance is confidence >= 0.55 and top1-top2 margin >= 0.12.
- Scores are averaged over five valid frames; at least three are required.
- Red confirms in two fused frames and green directions in three.
- `background`, low-quality frames, and camera timeout immediately make the
  published consensus inactive.
- Only adjust `crop_top`/`crop_bottom` if the physical lamp falls outside the
  blue region. Do not add a second horizontal flip.

The JSON topic `/traffic_light_rknn_test/detections` preserves
`consensus.active` and `consensus.class_name` for competition `task4`.

## Diagnostics

```bash
rosrun ucar_2026_traffic_light_rknn_test check_traffic_light_rknn_test.py
rostopic echo /traffic_light_rknn_test/detections -n1
rostopic hz /traffic_light_rknn_test/detections -w5
```
