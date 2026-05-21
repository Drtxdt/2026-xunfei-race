# Offline TTS MSC Patch

The robot image provides the ARM64 `speech_command` package:

```text
/home/ucar/ucar_ws/src/speech_command
```

The package subscribes to `/speak` through `voice_speak_node`, but older images
may still use an expired appid such as `32607f0d`. The current offline TTS appid
is:

```text
97d4e3db
```

Run this on the robot:

```bash
cd /home/ucar/2026-xunfei-race
chmod +x debug/patch_speech_command_offline_tts.sh
./debug/patch_speech_command_offline_tts.sh
```

If TTS fails with `QTTSAudioGet failed, error code: 11210`, the appid and MSC TTS
resources do not match. Copy the SDK resources for appid `97d4e3db` to the robot,
then rerun the script with the resource directory:

```bash
./debug/patch_speech_command_offline_tts.sh 97d4e3db /path/to/Linux_aisound_exp1227_97d4e3db/bin/msc/res/tts
```

Test only the TTS node:

```bash
source /home/ucar/ucar_ws/devel/setup.bash
rosrun speech_command voice_speak_node
```

Publish test text in another terminal:

```bash
source /home/ucar/ucar_ws/devel/setup.bash
rostopic pub -1 /speak std_msgs/String "data: '你好，这是离线语音合成测试'"
```
