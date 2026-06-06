# 比赛语音播报调用指南

本文档给各子任务控制同学使用，目标是让所有需要语音播报的比赛环节都统一调用同一个接口：

```text
/competition_speech/announce
```

底层仍然使用旧 TTS 节点订阅的 `/speak`，各任务节点不要再直接拼接并发布 `/speak`，建议统一调用本服务。这样可以保证播报文案符合赛规，并且 `wait: true` 时会等待预计播报完成后才返回，避免后续任务抢跑。

## 1. 启动顺序

### 1.1 编译

```bash
cd ~/2026-xunfei-race
source /opt/ros/noetic/setup.bash
source /home/ucar/ucar_ws/devel/setup.bash
catkin_make
source devel/setup.bash
```

如果刚更新过语音播报包，建议清理旧构建：

```bash
rm -rf build devel
catkin_make
source devel/setup.bash
```

### 1.2 启动旧 TTS 节点

新开终端，保持运行：

```bash
source /home/ucar/ucar_ws/devel/setup.bash
rosrun speech_command voice_speak_node
```

正常应看到类似：

```text
语音合成节点已启动
```

### 1.3 启动统一比赛播报服务

新开终端，保持运行：

```bash
cd ~/2026-xunfei-race
source devel/setup.bash
roslaunch ucar_2026_competition_speech competition_speech.launch
```

正常应看到：

```text
competition_announcer ready
```

## 2. 服务接口

服务名：

```text
/competition_speech/announce
```

服务类型：

```text
ucar_2026_competition_speech/Announce
```

查看接口：

```bash
rossrv show ucar_2026_competition_speech/Announce
```

字段含义：

| 字段 | 用途 |
|---|---|
| `event` | 子任务类型：`task1` / `task2` / `task3` / `task4` / `task5` |
| `item` | 货品名称，子任务2、3需要 |
| `workshop` | 仓库或车间名称，子任务2、3需要 |
| `decision` | 交通决策，子任务4需要 |
| `text` | 完整自定义播报文本，子任务1需要 |
| `wait` | 是否等待播报完成，正式比赛建议固定为 `true` |

返回字段：

| 字段 | 含义 |
|---|---|
| `success` | 是否成功生成并发布播报 |
| `speech_text` | 实际播报文本 |
| `estimated_duration` | 估计播报耗时 |
| `message` | `completed` 表示播报等待结束 |

注意：终端里中文可能显示成 `\u4EFB...`，这是 ROS/YAML 的 Unicode 显示方式，不代表错误。

## 3. 各子任务调用方式

### 3.1 子任务1：智能接单与货品筛选

比赛要求播报：

```text
取得[货品名称]属于[目标大类]应放置在[目标仓库]，仿真环境中取得[货品名称]属于[目标大类]应放置在[目标仓库]
```

调用方式：

```bash
rosservice call /competition_speech/announce \
"{event: 'task1', item: '', workshop: '', decision: '', text: '取得香蕉属于食品大类应放置在食品加工车间，仿真环境中取得毛巾属于日用品大类应放置在日用品加工车间', wait: true}"
```

谁来调用：

```text
任务1总控节点在二维码识别完成、大模型推理完成后调用。
```

如果使用现有 `task1_full_once.py`，它已经优先调用这个服务；服务不可用时才回退到旧 `/speak`。

### 3.2 子任务2：实体货品入库

比赛要求播报：

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

谁来调用：

```text
导航/停车控制节点在实体车抵达正确仓库停车区域后调用。
```

### 3.3 子任务3：仿真系统协同完成

比赛要求播报：

```text
仿真任务已完成，已将[货品名称]放入[仓库类别]
```

调用示例：

```bash
rosservice call /competition_speech/announce \
"{event: 'task3', item: '毛巾', workshop: '日用品加工车间', decision: '', text: '', wait: true}"
```

实际播报：

```text
仿真任务已完成，已将毛巾放入日用品加工车间
```

谁来调用：

```text
仿真协同节点在收到仿真任务完成反馈后调用。
```

### 3.4 子任务4：交通决策与路径选择

比赛要求：

```text
机器人根据交通灯识别结果做出明确路径决策，并进行决策播报。
```

支持的 `decision`：

| decision | 播报 |
|---|---|
| `left` | 左转 |
| `right` | 右转 |
| `straight` | 直行 |
| `stop` | 停止 |
| `红灯` | 停止 |

调用示例：

```bash
rosservice call /competition_speech/announce \
"{event: 'task4', item: '', workshop: '', decision: 'left', text: '', wait: true}"
```

实际播报：

```text
左转
```

谁来调用：

```text
交通灯识别/路径选择节点在锁定最终决策后调用。服务返回 completed 后，再驶入对应巡线入口。
```

### 3.5 子任务5：终点抵达

比赛要求播报：

```text
任务完成
```

调用示例：

```bash
rosservice call /competition_speech/announce \
"{event: 'task5', item: '', workshop: '', decision: '', text: '', wait: true}"
```

实际播报：

```text
任务完成
```

谁来调用：

```text
巡线/终点停车节点在确认终点停稳后10秒内调用。该播报预计约3.3秒完成，小于规则要求的30秒。
```

如果巡线节点会发布状态 `finish`，也可以启动自动监听：

```bash
roslaunch ucar_2026_competition_speech competition_speech.launch \
  finish_status_topic:=/line_follow/status
```

当 `/line_follow/status` 收到 `finish` 时，会自动播报 `任务完成`。

## 4. Python 节点中如何调用

推荐各业务节点直接调用服务：

```python
import rospy
from ucar_2026_competition_speech.srv import Announce

rospy.wait_for_service("/competition_speech/announce")
announce = rospy.ServiceProxy("/competition_speech/announce", Announce)

res = announce(
    "task2",              # event
    "香蕉",               # item
    "食品加工车间",       # workshop
    "",                   # decision
    "",                   # text
    True,                 # wait
)

if not res.success:
    rospy.logerr("播报失败: %s", res.message)
else:
    rospy.loginfo("播报完成: %s", res.speech_text)
```

正式流程里建议：

```text
wait = true
```

这样服务返回后，当前播报基本已经结束，可以安全进入下一个子任务。

## 5. Topic 方式调用

如果某个节点不方便调用 service，也可以发布 JSON 到：

```text
/competition_speech/request
```

示例：

```bash
rostopic pub -1 /competition_speech/request std_msgs/String \
"data: '{\"event\":\"task4\",\"decision\":\"right\",\"wait\":true}'"
```

但正式流程更推荐 service，因为 service 能拿到 `success` 和 `completed` 返回。

## 6. 验证结果

小车端已经验证过以下五项，均返回：

```text
success: True
message: "completed"
```

已验证内容：

| 子任务 | 测试内容 | 结果 |
|---|---|---|
| task1 | 完整推理结果播报 | 通过 |
| task2 | 已将香蕉放入食品加工车间 | 通过 |
| task3 | 仿真任务已完成，已将毛巾放入日用品加工车间 | 通过 |
| task4 | 左转 | 通过 |
| task5 | 任务完成 | 通过 |

其中 `task5` 返回示例：

```text
success: True
speech_text: "\u4EFB\u52A1\u5B8C\u6210"
estimated_duration: 3.3333332538604736
message: "completed"
```

`\u4EFB\u52A1\u5B8C\u6210` 就是 `任务完成`。

## 7. 通过标准

语音播报模块可判定为满足比赛要求，需要同时满足：

```text
1. 旧 TTS 节点 voice_speak_node 正常运行。
2. competition_announcer 正常启动。
3. 五类 event 调用均 success=True。
4. wait=true 时服务在播报完成后返回 completed。
5. 各子任务控制节点在对应完成时机调用本服务。
```

注意：本文档验证的是“语音播报模块能力”。整车比赛还需要导航、识别、仿真反馈、终点停车等模块在正确时机调用本接口。

## 8. 常见问题

### 8.1 ImportError: cannot import name build_announcement

说明使用了旧构建缓存。执行：

```bash
cd ~/2026-xunfei-race
git fetch origin
git checkout feature/competition-announcer
git pull --ff-only

source /opt/ros/noetic/setup.bash
source /home/ucar/ucar_ws/devel/setup.bash
rm -rf build devel
catkin_make
source devel/setup.bash
```

然后测试：

```bash
python3 -c "from ucar_2026_competition_speech.speech_templates import build_announcement; print(build_announcement('task5')[1])"
```

应输出：

```text
任务完成
```

### 8.2 服务不存在

检查播报服务是否启动：

```bash
rosservice list | grep competition_speech
```

应看到：

```text
/competition_speech/announce
```

### 8.3 没有声音

检查旧 TTS 是否订阅 `/speak`：

```bash
rostopic info /speak
```

应看到订阅者类似：

```text
Subscribers:
 * /voice_speak_node
```

如果没有，启动：

```bash
source /home/ucar/ucar_ws/devel/setup.bash
rosrun speech_command voice_speak_node
```

### 8.4 中文显示成 Unicode

例如：

```text
\u4EFB\u52A1\u5B8C\u6210
```

这是正常显示问题，实际含义是：

```text
任务完成
```
