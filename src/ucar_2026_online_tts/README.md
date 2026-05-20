# ucar_2026_online_tts

ROS `/speak` text-to-speech node using iFlytek online TTS WebAPI:

`wss://tts-api.xfyun.cn/v2/tts`

## Configure

Use the credentials from the iFlytek 在线语音合成 service page:

```bash
export XF_TTS_APPID='your_appid'
export XF_TTS_API_KEY='your_api_key'
export XF_TTS_API_SECRET='your_api_secret'
```

Install the WebSocket client dependency on the robot if needed:

```bash
python3 -m pip install --user websocket-client
```

## Run

```bash
roslaunch ucar_2026_online_tts online_speak.launch
```

Test:

```bash
rostopic pub -1 /speak std_msgs/String "data: '你好，这是测试播报'"
```
