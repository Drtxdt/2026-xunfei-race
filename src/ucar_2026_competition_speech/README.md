# Competition Speech

This package is the single announcement gateway for all five smart-factory
competition subtasks. It keeps the robot's legacy offline TTS interface
unchanged: generated text is published as `std_msgs/String` on `/speak`.

## Official announcement events

| Event | Required fields | Generated text |
|---|---|---|
| `task1` | `text` | The complete LLM reasoning result |
| `task2` | `item`, `workshop` | `已将[货品名称]放入[仓库类别]` |
| `task3` | `item`, `workshop` | `仿真任务已完成，已将[货品名称]放入[仓库类别]` |
| `task4` | `decision` | `左转` / `右转` / `直行` / `停止` |
| `task5` | none | `任务完成` |

Start the old TTS node, then the gateway:

```bash
source /home/ucar/ucar_ws/devel/setup.bash
rosrun speech_command voice_speak_node

source ~/2026-xunfei-race/devel/setup.bash
roslaunch ucar_2026_competition_speech competition_speech.launch
```

The service blocks conservatively until the announcement should have finished,
so task controllers can safely continue after the call returns:

```bash
rosservice call /competition_speech/announce \
  "{event: 'task2', item: '香蕉', workshop: '食品加工车间', decision: '', text: '', wait: true}"

rosservice call /competition_speech/announce \
  "{event: 'task3', item: '毛巾', workshop: '日用品加工车间', decision: '', text: '', wait: true}"

rosservice call /competition_speech/announce \
  "{event: 'task4', item: '', workshop: '', decision: 'left', text: '', wait: true}"

rosservice call /competition_speech/announce \
  "{event: 'task5', item: '', workshop: '', decision: '', text: '', wait: true}"
```

Other controllers may publish JSON to `/competition_speech/request`. Completed
events are published to `/competition_speech/completed`, and detailed JSON logs
are published to `/competition_speech/status`.

Example topic request:

```bash
rostopic pub -1 /competition_speech/request std_msgs/String \
  "data: '{\"event\":\"task4\",\"decision\":\"right\",\"wait\":true}'"
```

To automatically announce task 5 after the line-follow node reports `finish`:

```bash
roslaunch ucar_2026_competition_speech competition_speech.launch \
  finish_status_topic:=/line_follow/status
```
