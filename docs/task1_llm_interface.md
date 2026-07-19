# 任务1大模型接口文档

本文档给二维码、语音识别、导航和播报同学使用。  
目标是说明：三个二维码物品名和语音指令识别出来以后，如何调用大模型模块，如何拿到结果，如何让小车播报。

我的讯飞星火API密钥是：NpYqwKSsiXVuhuTPdSAY:eqNAhCRrUuYToKxWvtTq

APPID： 97d4e3db

APISecret：Y2I2NjQ0YTU4YWNhNWE0ODQwNzg5ZDI4

APIKey：b5e01a18f2a592b26bd06ff3b59ef0f4

## 1. 这个模块做什么

任务1要求小车在物品领取区识别三个二维码，得到三个物品名，例如：

```text
草莓
纸巾
手机
```

同时语音识别会得到一整句任务指令，例如：

```text
小飞小飞，前往物品领取区，取得食品类，放置在对应仓库，并领取仿真环境中需要的日用品类放置在对应仓库
```

其中实体领取类别和仿真环境类别必须不同。总控按“仿真环境”分隔语句：前半句类别用于实体领取及任务2，后半句类别用于任务3；缺失任一类别或两个类别相同都会要求重新说指令。

大模型模块会根据这两个输入做推理：

```text
三个物品名 + 语音指令
↓
判断物理领取区要取哪个物品
判断仿真环境要取哪个物品
判断它们属于哪个大类
判断应该放到哪个车间
生成比赛要求的播报文本
```

例如输出：

```text
取得草莓属于食品大类应放置在食品加工车间，仿真环境中取得纸巾属于日用品大类应放置在日用品加工车间
```

## 2. 启动前准备

小车上需要先配置讯飞星火大模型的 API 密钥。

在小车终端执行：

```bash
export XF_SPARK_API_PASSWORD='你的讯飞星火APIPassword'
```

如果已经写进 `~/.bashrc`，则只需要：

```bash
source ~/.bashrc
```

检查是否配置成功：

```bash
echo $XF_SPARK_API_PASSWORD
```

如果能输出密钥，说明配置成功。

## 3. 启动顺序

### 3.1 启动离线语音播报节点

打开一个终端，执行：

```bash
cd /home/ucar/ucar_ws/src/speech_command/bin
source /home/ucar/ucar_ws/devel/setup.bash
rosrun speech_command voice_speak_node
```

看到类似下面的输出即可：

```text
语音合成节点已启动
```

这个终端不要关闭。

可以先单独测试播报：

```bash
rostopic pub -1 /speak std_msgs/String "data: '你好，这是测试播报'"
```

如果小车能播出这句话，说明 `/speak` 播报链路正常。

### 3.2 启动大模型服务

再打开一个终端，执行：

```bash
cd ~/2026-xunfei-race
source devel/setup.bash
roslaunch ucar_2026_smart_factory_llm reason_pickup.launch
```

看到下面这句说明服务启动成功：

```text
smart_factory_llm: 服务已就绪 ~/reason_pickup_order
```

这个终端也不要关闭。

## 4. 大模型接口

### 4.1 Service 名称

```text
/smart_factory_llm/reason_pickup_order
```

### 4.2 Service 类型

```text
ucar_2026_smart_factory_llm/ReasonPickupOrder
```

查看接口定义：

```bash
rossrv show ucar_2026_smart_factory_llm/ReasonPickupOrder
```

### 4.3 输入字段

| 字段名 | 类型 | 含义 | 示例 |
|---|---|---|---|
| `item_a` | string | 第一个二维码最终稳定识别出的物品名 | `草莓` |
| `item_b` | string | 第二个二维码最终稳定识别出的物品名 | `纸巾` |
| `item_c` | string | 第三个二维码最终稳定识别出的物品名 | `手机` |
| `voice_instruction` | string | 语音识别得到的完整任务指令 | `小飞小飞，前往物品领取区，取得食品类...` |

注意：二维码同学只需要传 `result` 里的物品名，不需要传整个 JSON，也不需要传二维码链接。

二维码原始返回示例：

```json
{"code":200,"result":"纸巾"}
```

传给大模型时只传：

```text
纸巾
```

## 5. 调用示例

手动测试命令：

```bash
rosservice call /smart_factory_llm/reason_pickup_order \
"{item_a: '草莓', item_b: '纸巾', item_c: '手机', voice_instruction: '小飞小飞，前往物品领取区，取得食品类，放置在对应仓库，并领取仿真环境中需要的日用品类放置在对应仓库'}"
```

成功时会看到：

```text
success: True
pickup_item: 草莓
pickup_major: 食品大类
pickup_workshop: 食品加工车间
sim_item: 纸巾
sim_major: 日用品大类
sim_workshop: 日用品加工车间
announcement_full: 取得草莓属于食品大类应放置在食品加工车间，仿真环境中取得纸巾属于日用品大类应放置在日用品加工车间
```

终端里中文有时会显示成 `\u53D6\u5F97...`，这是 ROS/YAML 的中文转义，不代表结果错误。

## 6. 返回字段含义

| 字段名 | 类型 | 含义 |
|---|---|---|
| `success` | bool | 是否推理成功。`True` 表示成功，`False` 表示失败 |
| `error_message` | string | 失败原因。成功时为空字符串 |
| `pickup_item` | string | 物理领取区要取的物品 |
| `pickup_major` | string | 物理领取区物品所属大类 |
| `pickup_workshop` | string | 物理领取区物品对应车间 |
| `sim_item` | string | 仿真环境要取的物品 |
| `sim_major` | string | 仿真环境物品所属大类 |
| `sim_workshop` | string | 仿真环境物品对应车间 |
| `announcement_physical` | string | 物理领取区单句播报 |
| `announcement_simulation` | string | 仿真环境单句播报 |
| `announcement_full` | string | 完整播报文本，可以直接发给 `/speak` |
| `raw_model_reply` | string | 大模型原始返回，主要用于调试 |

## 7. 大类和车间对应关系

| 目标大类 | 对应车间 |
|---|---|
| 食品大类 | 食品加工车间 |
| 日用品大类 | 日用品加工车间 |
| 电子产品大类 | 电子产品生产车间 |

示例：

```text
草莓、苹果、饺子、面条、薯片、馒头 -> 食品大类 -> 食品加工车间
纸巾、毛巾、牙刷、洗衣液、T恤衫 -> 日用品大类 -> 日用品加工车间
手机、耳机、充电器、鼠标、数据线 -> 电子产品大类 -> 电子产品生产车间
```

## 8. 如何播报结果

小车播报话题是：

```text
/speak
```

消息类型：

```text
std_msgs/String
```

大模型返回的 `announcement_full` 可以直接发布到 `/speak`。

命令行测试：

```bash
rostopic pub -1 /speak std_msgs/String "data: '取得草莓属于食品大类应放置在食品加工车间，仿真环境中取得纸巾属于日用品大类应放置在日用品加工车间'"
```

Python 示例：

```python
import rospy
from std_msgs.msg import String

rospy.init_node("speak_test")
pub = rospy.Publisher("/speak", String, queue_size=1)
rospy.sleep(1.0)
pub.publish(String(data="取得草莓属于食品大类应放置在食品加工车间"))
```

## 9. 一次性联调节点

仓库里提供了一个测试节点：

```text
reason_and_speak_once.launch
```

它会自动完成：

```text
调用大模型服务
拿到 announcement_full
发布到 /speak
让小车播报
```

使用示例：

```bash
cd ~/2026-xunfei-race
source devel/setup.bash
roslaunch ucar_2026_smart_factory_llm reason_and_speak_once.launch \
  item_a:=草莓 item_b:=纸巾 item_c:=手机 \
  voice_instruction:=小飞小飞，前往物品领取区，取得食品类，放置在对应仓库，并领取仿真环境中需要的日用品类放置在对应仓库
```

如果终端显示：

```text
Publishing TTS text to /speak
```

说明已经把大模型结果发给播报节点。

## 10. 给二维码同学的要求

二维码同学最终只需要输出三个稳定物品名。

建议格式：

```text
item_a = 草莓
item_b = 纸巾
item_c = 手机
```

重要注意事项：

```text
同一个二维码不要每帧都重复请求接口。
每个二维码识别到后，只请求一次接口并锁定 result。
等三个二维码都拿到稳定 result 后，再传给大模型模块。
```

错误示例：

```text
同一个 raw 链接一直返回草莓、饺子、苹果、葡萄...
```

这种会导致结果不稳定。

正确做法：

```text
第一个二维码锁定一个 result
第二个二维码锁定一个 result
第三个二维码锁定一个 result
最终只传三个物品名
```

## 11. 给语音识别同学的要求

语音识别同学需要输出完整任务文本。

示例：

```text
小飞小飞，前往物品领取区，取得食品类，放置在对应仓库，并领取仿真环境中需要的日用品类放置在对应仓库
```

不要只传关键词：

```text
食品 日用品
```

虽然大模型可能能猜出来，但完整句子更稳。

## 12. 给导航/控制同学的结果

导航/控制同学一般关心这两个字段：

```text
pickup_workshop
sim_workshop
```

例如：

```text
pickup_workshop = 食品加工车间
sim_workshop = 日用品加工车间
```

如果导航内部不用中文，可以约定映射：

```text
食品加工车间 -> FOOD
日用品加工车间 -> DAILY
电子产品生产车间 -> ELECTRONICS
```

## 13. 常见问题

### 13.1 服务找不到

报错类似：

```text
Unable to load type [ucar_2026_smart_factory_llm/ReasonPickupOrder]
```

解决：

```bash
cd ~/2026-xunfei-race
catkin_make
source devel/setup.bash
```

每开一个新终端，都要先：

```bash
source ~/2026-xunfei-race/devel/setup.bash
```

### 13.2 大模型服务启动了但调用失败

如果看到：

```text
api_password 为空
```

说明没有配置星火 API 密钥。

解决：

```bash
export XF_SPARK_API_PASSWORD='你的讯飞星火APIPassword'
```

### 13.3 播报没声音

先确认 `/speak` 有没有订阅者：

```bash
rostopic info /speak
```

正常应该看到：

```text
Subscribers:
 * /voice_speak_node
```

如果没有，启动离线 TTS：

```bash
cd /home/ucar/ucar_ws/src/speech_command/bin
source /home/ucar/ucar_ws/devel/setup.bash
rosrun speech_command voice_speak_node
```

### 13.4 出现 QTTSAudioGet 11212

报错：

```text
QTTSAudioGet failed, error code: 11212
```

说明离线 TTS 授权或资源有问题。需要更新：

```text
common.jet
xiaoyan.jet
```

目前小车上的资源已经更新过，正常情况下不应再报这个错。

## 14. 推荐完整联调流程

1. 启动离线 TTS。
2. 测试 `/speak` 能不能播“你好”。
3. 启动大模型服务。
4. 用 `reason_and_speak_once.launch` 测试固定样例。
5. 二维码节点输出三个稳定物品名。
6. 语音识别节点输出完整语音文本。
7. 总控节点调用 `/smart_factory_llm/reason_pickup_order`。
8. 总控节点把 `announcement_full` 发布到 `/speak`。
9. 小车播报完成后，继续后续导航任务。

## 15. 最小可用链路

必须保证下面三件事同时成立：

```text
二维码：能给出 item_a / item_b / item_c
语音识别：能给出 voice_instruction
播报：能播放 /speak 里的 String 文本
```

只要这三件事成立，大模型模块就能接入任务1。
