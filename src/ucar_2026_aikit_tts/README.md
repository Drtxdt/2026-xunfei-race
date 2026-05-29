# ucar_2026_aikit_tts

ROS `/speak` text-to-speech node using iFlytek AIKit offline XTTS.

This is a replacement for the old `speech_command/voice_speak_node` when that
node fails with expired MSC TTS errors such as `QTTSAudioGet failed, error code:
11212`.

## Configure

Set the AIKit auth values from the iFlytek app that downloaded the SDK:

```bash
export XF_AIKIT_APPID='your_appid'
export XF_AIKIT_API_KEY='your_api_key'
export XF_AIKIT_API_SECRET='your_api_secret'
```

For permanent use, add those exports to `~/.bashrc`.

## Build

```bash
cd ~/2026-xunfei-race
catkin_make
source devel/setup.bash
```

## Run

```bash
roslaunch ucar_2026_aikit_tts aikit_speak.launch
```

Test:

```bash
rostopic pub -1 /speak std_msgs/String "data: '你好，这是测试播报'"
```

If playback uses the wrong sound card, pass an ALSA device:

```bash
roslaunch ucar_2026_aikit_tts aikit_speak.launch audio_device:=hw:3,0
```
