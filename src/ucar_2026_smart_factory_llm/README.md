# ucar_2026_smart_factory_llm

讯飞智慧工厂赛项 **子任务 1：智能接单与货品筛选** 中的 **大模型推理** 部分。

## 功能

- 订阅式 **ROS Service** `~reason_pickup_order`，类型 `ucar_2026_smart_factory_llm/ReasonPickupOrder`。
- 输入：三个二维码解析得到的子类名称 + 语音指令全文。
- 通过 **讯飞星火 Spark X2** HTTP 接口（`spark-x`）推理：在三个货品中分别选出「领取区目标大类」与「仿真环境目标大类」对应的一项，并映射到 **食品加工车间 / 日用品加工车间 / 电子产品生产车间**。
- 输出：结构化字段 + 赛题要求格式的两句播报（物理领取、仿真领取）。

## 依赖

- ROS Noetic（或兼容的 ROS1 + `rospy` Python3）。
- 仅使用 Python 标准库发起 HTTPS 请求（无需安装 `openai` 包）。

## 密钥配置（勿提交到 Git）

在讯飞开放平台控制台获取 HTTP 协议的 **APIPassword**，任选其一：

1. 环境变量（推荐）  
   `export XF_SPARK_API_PASSWORD='你的APIPassword'`
2. Launch 参数  
   `roslaunch ucar_2026_smart_factory_llm reason_pickup.launch api_password:=你的APIPassword`

参考文档：[Spark-X2 http 协议](https://www.xfyun.cn/doc/spark/X1http.html)

## 编译与运行

```bash
cd ~/catkin_ws   # 你的工作空间根目录，且已包含本仓库 src
catkin_make
source devel/setup.bash
roslaunch ucar_2026_smart_factory_llm reason_pickup.launch
```

Start the shared competition announcement gateway before running the task-1
controllers. They will use its blocking service and only fall back to direct
`/speak` publishing when the gateway is unavailable:

```bash
roslaunch ucar_2026_competition_speech competition_speech.launch
```

## 调用示例

```bash
rosservice call /smart_factory_llm/reason_pickup_order \
  "{item_a: '香蕉', item_b: 'T恤衫', item_c: '充电器', \
    voice_instruction: '小飞小飞，前往物品领取区，取得食品类，放置在对应仓库，并领取仿真环境中需要的日用品类放置在对应仓库'}"
```

成功时 `success: True`，`announcement_full` 为两句播报合并字符串（中间为中文逗号），便于直接送给 TTS 节点。

## 与队友集成

- **二维码节点**：将三次 `result` 填入 `item_a/b/c`（顺序与车队约定一致即可）。
- **语音节点**：将 ASR 文本填入 `voice_instruction`。
- **语音播报**：使用 `announcement_physical` 与 `announcement_simulation`，或整段 `announcement_full`。

## 播报联调

车上 TTS 节点订阅 `/speak`（`std_msgs/String`）。先启动语音包：

```bash
roslaunch speech_command speech_command.launch
```

再启动大模型服务：

```bash
roslaunch ucar_2026_smart_factory_llm reason_pickup.launch
```

可以用一次性联调节点测试“大模型推理 → /speak 播报”：

```bash
roslaunch ucar_2026_smart_factory_llm reason_and_speak_once.launch \
  item_a:=草莓 item_b:=纸巾 item_c:=手机 \
  voice_instruction:=小飞小飞，前往物品领取区，取得食品类，放置在对应仓库，并领取仿真环境中需要的日用品类放置在对应仓库
```
