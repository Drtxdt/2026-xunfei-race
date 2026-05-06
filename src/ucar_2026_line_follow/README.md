# ucar_2026_line_follow

ROS1 Python package for the 2026 visual line-following subtask.

Rules reference recorded from the task request:
`https://www.iflysse.com/laboratory/intelligent-car-race`

The first version is intentionally vision-only. It subscribes to the USB camera,
tracks the lane between two white borders, prefers the left corridor at a fork,
and stops after detecting the P1 parking square.

## Interfaces

- Subscribes: `/usb_cam/image_raw` (`sensor_msgs/Image`)
- Optional start: `/line_follow/start` (`std_msgs/Bool`)
- Publishes: `/cmd_vel` (`geometry_msgs/Twist`)
- Publishes: `/line_follow/status` (`std_msgs/String`)
- Publishes: `/line_follow/debug_image` (`sensor_msgs/Image`)

Status values:

- `idle`: waiting for `/line_follow/start` when `auto_start` is false
- `searching`: moving slowly while looking for white borders
- `tracking`: following the detected corridor
- `turn_left`: fork-like multi-border area detected; left corridor is selected
- `lost`: line lost longer than `lost_timeout`; command is stopped
- `finish_stop`: P1 parking square detected; command is stopped before reversing
- `finish_reverse`: reversing after the P1 stop
- `finish`: reverse maneuver is complete; command is stopped
- `parking_debug`: debug node has detected P1 but is waiting for manual `Ctrl+C`

## Run

From a sourced catkin workspace:

```bash
roslaunch ucar_2026_line_follow line_follow_test.launch
```

Run only the line-follow node when the base driver and camera are already up:

```bash
roslaunch ucar_2026_line_follow line_follow_test.launch start_driver:=false start_camera:=false
```

View the debug image:

```bash
rosrun image_view image_view image:=/line_follow/debug_image
```

Record a parking calibration snapshot. Let the car run, press `Ctrl+C` at the
position where you want it to stop, and send the printed JSON path or file:

```bash
roslaunch ucar_2026_line_follow parking_debug.launch
```

By default the snapshot is saved under
`~/.ros/ucar_2026_line_follow/parking_debug_*.json`. You can choose a path:

```bash
roslaunch ucar_2026_line_follow parking_debug.launch debug_output_path:=/tmp/parking_debug.json
```

## Field Tuning

Start with `config/line_follow.yaml`.

- Increase `white_v_min` or `gray_white_threshold` if bright floor regions are
  detected as line.
- Increase `white_s_max` if the white border appears slightly colored under the
  camera.
- Tune `lane_width_px` using a straight section of the real track. This is used
  when only one side border is visible.
- Reduce `base_linear_speed` and `turn_linear_speed` for the first real-car test.
- If the P1 square triggers too early, increase `finish_confirm_frames` or
  `finish_horizontal_min_width_ratio`.

## Test Checklist

1. Confirm `/line_follow/debug_image` shows the ROI, border candidates, lane
   center, target center, and current status.
2. With the car lifted, move the track image left and right under the camera and
   verify `/cmd_vel/angular.z` points back toward the lane center.
3. Place the car at the line-follow start point and run at low speed.
4. Verify it stays between borders, chooses the left fork, and stops inside P1.
