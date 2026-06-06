# ucar_2026_qr_speak_test

ROS Noetic test package for QR camera viewing and direct speech playback.

## What It Does

- Starts the USB camera topic `/usb_cam/image_raw`.
- Opens an X11 `image_view` window so MobaXterm can show the live camera image.
- Runs `yolo/qr_collect_and_decode.py --fetch` and listens to `/qr_code_data`.
- Cleans QR JSON and publishes only the speakable result text to `/speak`.

The node does not speak status codes, URLs, `ok`, `stamp`, or other debug fields.
For a QR payload like:

```json
{"code": 200, "result": "纸巾"}
```

or the existing `/qr_code_data` payload:

```json
{"items": [{"raw": "http://example/daily", "api": {"code": 200, "result": "纸巾"}, "ok": true, "result": "纸巾"}]}
```

it publishes:

```text
纸巾
```

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

If the camera image is mirrored:

```bash
roslaunch ucar_2026_qr_speak_test qr_camera_speak_test.launch flip:=true
```

If the camera is already running:

```bash
roslaunch ucar_2026_qr_speak_test qr_camera_speak_test.launch start_camera:=false
```

## Topics

- Subscribes: `/qr_code_data` (`std_msgs/String`)
- Publishes: `/speak` (`std_msgs/String`)
- Publishes: `/qr_speak_test/status` (`std_msgs/String`)

## Notes

- The same QR key is spoken once by default to avoid repeated frame-by-frame playback.
- Use `repeat_same:=true` only for manual repeated testing.
- This package assumes the car has `usb_cam`, `image_view`, `yolo`, and a `/speak` TTS subscriber available.
