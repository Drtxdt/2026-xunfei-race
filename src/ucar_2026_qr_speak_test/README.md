# ucar_2026_qr_speak_test

ROS Noetic test package for QR camera viewing and direct speech playback.

## What It Does

- Starts the USB camera topic `/usb_cam/image_raw`.
- Opens an X11 `image_view` window so MobaXterm can show the live camera image.
- Runs `yolo/qr_collect_and_decode.py --fetch` and listens to `/qr_code_data`.
- Cleans QR JSON, classifies the three item results, and publishes the task
  completion sentence to `/speak`.

The node does not speak status codes, URLs, `ok`, `stamp`, or other debug fields.
For a QR payload like:

```json
{"code": 200, "result": "纸巾"}
```

or the existing `/qr_code_data` payload:

```json
{"items": [{"raw": "http://example/daily", "api": {"code": 200, "result": "纸巾"}, "ok": true, "result": "纸巾"}]}
```

in item mode it publishes:

```text
纸巾
```

By default it waits for three QR results, then publishes the race-format task
sentence:

```text
取得苹果属于食品大类应放置在食品加工车间，仿真环境中取得毛巾属于日用品大类应放置在日用品加工车间
```

The spoken text uses extra punctuation by default so the TTS pauses more slowly.
During field testing, each accepted QR item is also confirmed with a short
`已识别xx` speech by default. Disable it with `speak_each_item:=false` if only
the final task sentence should be spoken.

## Run On The Car

Open MobaXterm SSH with X11 forwarding enabled, then:

```bash
source ~/ucar_ws/devel/setup.bash
source ~/2026-xunfei-race/devel/setup.bash
```

Start the speech node in another terminal:

```bash
rosrun speech_command voice_speak_node
```

Quick TTS check:

```bash
rostopic pub -1 /speak std_msgs/String "data: '你好，这是测试播报'"
```

Launch the QR test:

```bash
roslaunch ucar_2026_qr_speak_test qr_camera_speak_test.launch
```

Or use the sequence launch for field testing. It starts camera, TTS, X11 viewer,
QR decoder, and QR speech bridge with short delays:

```bash
roslaunch ucar_2026_qr_speak_test qr_speak_sequence_test.launch
```

Default target categories are:

```text
pickup_target_major:=食品大类
sim_target_major:=日用品大类
```

Change them from launch args when the voice/task target changes:

```bash
roslaunch ucar_2026_qr_speak_test qr_speak_sequence_test.launch \
  pickup_target_major:=电子产品大类 sim_target_major:=食品大类
```

If you need the old behavior that speaks each QR item directly:

```bash
roslaunch ucar_2026_qr_speak_test qr_speak_sequence_test.launch output_mode:=item
```

If the speech is too slow after this change:

```bash
roslaunch ucar_2026_qr_speak_test qr_speak_sequence_test.launch slow_speech:=false
```

If you only want the final race-format sentence:

```bash
roslaunch ucar_2026_qr_speak_test qr_speak_sequence_test.launch speak_each_item:=false
```

The default `offline_mode:=fallback` keeps the official URL path first. If the
QR website is down, it uses the QR URL category (`food`, `daily`, `electronic`)
to generate a local test result and still publishes the standard JSON shape.

Fully offline rehearsal:

```bash
roslaunch ucar_2026_qr_speak_test qr_speak_sequence_test.launch offline_mode:=force
```

Official server-only behavior:

```bash
roslaunch ucar_2026_qr_speak_test qr_speak_sequence_test.launch offline_mode:=off
```

If the camera image is mirrored:

```bash
roslaunch ucar_2026_qr_speak_test qr_speak_sequence_test.launch flip:=true
```

If the camera is already running:

```bash
roslaunch ucar_2026_qr_speak_test qr_speak_sequence_test.launch start_camera:=false
```

## Topics

- Subscribes: `/qr_code_data` (`std_msgs/String`)
- Publishes: `/speak` (`std_msgs/String`)
- Publishes: `/qr_speak_test/status` (`std_msgs/String`)

## Notes

- The same QR key is spoken once by default to avoid repeated frame-by-frame playback.
- Use `repeat_same:=true` only for manual repeated testing.
- This package assumes the car has `usb_cam`, `image_view`, `yolo`, and a `/speak` TTS subscriber available.
