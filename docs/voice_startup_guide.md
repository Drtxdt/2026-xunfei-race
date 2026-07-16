# 比赛语音播报启动与调用指南

本文档用于队友在小车上启动和测试比赛语音播报模块。推荐使用 MobaXterm 通过 SSH 连接小车。

小车连接信息：

```text
ssh ucar@192.168.1.6
密码：iflytek
```

## 1. 语音播报整体链路

```text
各任务节点
  -> 调用 /competition_speech/announce
  -> competition_announcer 生成播报文本
  -> 发布到 /speak
  -> voice_speak_node 订阅 /speak
  -> 旧 TTS 合成并播放声音
```

也就是说：

- `/competition_speech/announce`：比赛统一播报服务，负责生成规范文案。
- `/speak`：底层 TTS 文本话题。
- `voice_speak_node`：真正负责合成并播放声音。

如果 `/competition_speech/announce` 返回成功但没有声音，通常是 `voice_speak_node` 没启动，或者 `/speak` 没有订阅者。

## 2. 启动旧 TTS 播放节点

新开一个 MobaXterm 终端，执行：

```bash
source /opt/ros/noetic/setup.bash
/home/ucar/ucar_ws/devel/lib/speech_command/voice_speak_node
```

看到类似输出即可：

```text
语音合成节点已启动
```

这个终端不要关闭。

注意：不建议使用下面这个命令：

```bash
roslaunch speech_command speech_command.launch
```

因为小车工作区可能存在重复 ROS 包，`roslaunch` 扫包时容易报错。直接运行可执行文件最稳。

## 3. 启动统一比赛播报服务

再开一个 MobaXterm 终端，执行：

```bash
cd ~/2026-xunfei-race
source devel/setup.bash
roslaunch ucar_2026_competition_speech competition_speech.launch
```

看到类似输出即可：

```text
competition_announcer ready
```

这个终端也不要关闭。

## 4. 检查 /speak 是否有人订阅

再开一个终端：

```bash
cd ~/2026-xunfei-race
source devel/setup.bash
rostopic info /speak
```

正常应看到：

```text
Subscribers:
 * /voice_speak_node
```

如果 `Subscribers: None`，说明旧 TTS 没启动，回到第 2 步。

## 5. 最底层声音测试

确认 `/speak` 有订阅者后，测试能不能出声：

```bash
rostopic pub -1 /speak std_msgs/String "data: '测试播报'"
```

如果能听到“测试播报”，说明旧 TTS 正常。

## 6. 测试统一播报服务

测试任务完成播报：

```bash
rosservice call /competition_speech/announce \
"{event: 'task5', item: '', workshop: '', decision: '', text: '', wait: true}"
```

成功返回类似：

```text
success: True
speech_text: "\u4EFB\u52A1\u5B8C\u6210"
message: "completed"
```

`\u4EFB\u52A1\u5B8C\u6210` 是 ROS/YAML 的 Unicode 显示方式，实际含义是“任务完成”。

## 7. 各任务调用方式

### 7.1 任务1：智能接单与货品筛选

播报格式：

```text
取得[货品名称]属于[目标大类]应放置在[目标仓库]，仿真环境中取得[货品名称]属于[目标大类]应放置在[目标仓库]
```

调用示例：

```bash
rosservice call /competition_speech/announce \
"{event: 'task1', item: '', workshop: '', decision: '', text: '取得香蕉属于食品大类应放置在食品加工车间，仿真环境中取得牙刷属于日用品大类应放置在日用品加工车间', wait: true}"
```

任务1总控节点应在二维码识别完成、大模型推理完成后调用。

### 7.2 任务2：实体货品入库

播报格式：

```text
已将[货品名称]放入[仓库类别]
```

调用示例：

```bash
rosservice call /competition_speech/announce \
"{event: 'task2', item: '香蕉', workshop: '食品加工车间', decision: '', text: '', wait: true}"
```

实际播报：

```text
已将香蕉放入食品加工车间
```

### 7.3 任务3：仿真系统协同完成

播报格式：

```text
仿真任务已完成，已将[货品名称]放入[仓库类别]
```

调用示例：

```bash
rosservice call /competition_speech/announce \
"{event: 'task3', item: '牙刷', workshop: '日用品加工车间', decision: '', text: '', wait: true}"
```

实际播报：

```text
仿真任务已完成，已将牙刷放入日用品加工车间
```

### 7.4 任务4：交通决策与路径选择

支持的 `decision`：

```text
left      -> 左转
right     -> 右转
straight  -> 直行
stop      -> 停止
```

调用示例：

```bash
rosservice call /competition_speech/announce \
"{event: 'task4', item: '', workshop: '', decision: 'left', text: '', wait: true}"
```

实际播报：

```text
左转
```

### 7.5 任务5：终点抵达

播报格式：

```text
任务完成
```

调用示例：

```bash
rosservice call /competition_speech/announce \
"{event: 'task5', item: '', workshop: '', decision: '', text: '', wait: true}"
```

## 8. 大模型推理服务启动与测试

任务1如果需要单独测试星火大模型推理，先启动大模型服务。

终端 1：

```bash
cd ~/2026-xunfei-race
source devel/setup.bash
export XF_SPARK_API_PASSWORD='你的星火APIPassword'
roslaunch ucar_2026_smart_factory_llm reason_pickup.launch
```

看到类似输出：

```text
smart_factory_llm: 服务已就绪 ~/reason_pickup_order
```

终端 2 测试调用：

```bash
cd ~/2026-xunfei-race
source devel/setup.bash

rosservice call /smart_factory_llm/reason_pickup_order \
"{item_a: '香蕉', item_b: '牙刷', item_c: '手机', voice_instruction: '小车小车，前往物品领取区，取得食品类，放置在对应仓库，并领取仿真环境中需要的日用品类放置在对应仓库'}"
```

成功标准：

```text
success: True
raw_model_reply: ...
```

如果 `raw_model_reply` 有内容，说明确实调用到了星火大模型。

## 9. 推荐启动顺序

比赛或联调时建议按这个顺序启动：

```text
1. 启动 roscore 或底盘总 launch
2. 启动旧 TTS：voice_speak_node
3. 启动统一播报服务：competition_speech.launch
4. 如果任务1需要大模型，启动 reason_pickup.launch
5. 启动各任务控制节点
```

最低限度只测试语音播报时，需要启动：

```text
voice_speak_node
competition_speech.launch
```

## 10. 常见问题

### 10.1 /competition_speech/announce 返回成功但没声音

检查：

```bash
rostopic info /speak
```

如果是：

```text
Subscribers: None
```

说明 `voice_speak_node` 没启动。执行：

```bash
source /opt/ros/noetic/setup.bash
/home/ucar/ucar_ws/devel/lib/speech_command/voice_speak_node
```

### 10.2 roslaunch speech_command speech_command.launch 报错

不要用这个命令。小车工作区可能存在重复包。

直接运行：

```bash
/home/ucar/ucar_ws/devel/lib/speech_command/voice_speak_node
```

### 10.3 任务1能启动，但“小飞小飞”没有反应

先检查 `task1.launch` 的终端日志。以下信息表示旧 ASR 运行环境或资源失效：

```text
package 'speech_command' not found
aiui.cfg 读取错误
globalAgent未创建
IllegalApiKeyError
build grammar error, errcode = 11212
```

仓库提供了实体车恢复脚本。它会备份旧语音源码和配置，从车上的原厂备份恢复在线 AIUI 凭据，关闭已过期的离线试用语法，并恢复硬件唤醒后的 AIUI 会话切换：

```bash
cd ~/2026-xunfei-race
bash debug/repair_speech_command_asr.sh
```

修复后重新启动：

```bash
cd ~/2026-xunfei-race
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch ucar_2026_competition task1.launch
```

正常日志应包含：

```text
EVENT_STATE:READY
请使用唤醒词唤醒
EVENT_STATE:WORKING
```

任务1采用两段式语音握手：

```text
用户：小飞小飞
小车：我在
用户：取得食品（也可说取得日用品、取得电子产品）
小车：好的
小车：开始导航
```

AIUI 只在“我在”播放完毕后开启，并在“好的”播放前关闭，避免小车把自己的播报识别成用户指令。旧 AIUI 云语义链路不再承担固定命令识别：语音节点将麦克风 PCM 发布到 `/speech_command_node/audio_pcm`，`fixed_command_iat.py` 使用讯飞流式听写识别三类命令。一次空 VAD 结束后会自动建立下一轮听写，不会锁死；在收到有效“取得xx”前再次说“小飞小飞”，小车会再次回复“我在”并重置听写会话。

也可分别监视：

```bash
rostopic echo /wakeup
rostopic echo /question
rostopic echo /competition/iat_text
```

### 10.4 中文显示成 Unicode

例如：

```text
\u4EFB\u52A1\u5B8C\u6210
```

这是正常显示问题，实际内容是：

```text
任务完成
```

### 10.5 播报语速偏快

已验证旧 TTS 的合适语速为 `speed = 60`。如果需要重新应用补丁：

```bash
cd ~/2026-xunfei-race
chmod +x debug/patch_speech_command_offline_tts.sh
./debug/patch_speech_command_offline_tts.sh
```

如果要临时调节：

```bash
TTS_SPEED=50 ./debug/patch_speech_command_offline_tts.sh
TTS_SPEED=70 ./debug/patch_speech_command_offline_tts.sh
```

调完后重新启动：

```bash
/home/ucar/ucar_ws/devel/lib/speech_command/voice_speak_node
```

## 11. 判定语音模块通过的标准

满足以下条件即可认为语音播报模块可用：

```text
1. voice_speak_node 正常运行。
2. rostopic info /speak 能看到 /voice_speak_node 订阅者。
3. rostopic pub -1 /speak 能听到声音。
4. /competition_speech/announce 五类 event 均返回 success: True。
5. wait: true 时返回 message: completed。
```
