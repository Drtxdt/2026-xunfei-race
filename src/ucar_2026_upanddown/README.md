# ucar_2026_upanddown

国赛坡道通过包（上坡22° — 平路 — 下坡25°，总长约1.5m）。

## 参数管理规则（重要）

**所有参数只存在于 config/*.yaml，launch 文件不覆盖任何值。**
调参流程：改 yaml → 重启 roslaunch。yaml 是本项目唯一权威参数来源。

| 文件 | 内容 |
|---|---|
| `config/ramp_traverse.yaml` | 坡道通过全部参数（分段阈值/速度/航向/看门狗/校验），逐行中文注释 |
| `config/scan_gate.yaml` | 雷达门控话题与初始状态 |

每个 yaml 首行有 `config_loaded: true` 标记；节点启动时若检测不到该标记，
说明 yaml 未被加载，会在日志显著告警（防止以内置默认值误跑比赛）。

## 组成

1. `scan_gate_node`（`/scan_gate`）：`/scan` → `/scan_gated` 中继。
   国赛导航栈（`ucar_2026_national_competition/launch/national_nav.launch`）
   的 AMCL 与 move_base 订阅门控话题。关门丢帧 → `map→odom` 冻结，
   定位退化为纯里程计推算，且局部代价地图不会把坡道标成障碍。
2. `ramp_traverse_node`（`/ramp_traverse`）：
   - 服务 `~start` / `~abort`，状态 JSON 发 `/ramp_traverse/status`
   - 流程：关门 → 低速直行（软加减速防滑移）→ IMU pitch 分段
     （level→up→plateau→down→complete，迟滞+确认帧）→ 下坡结束后
     多走 `exit_extra_m` 轴距余量 → 停稳 → 开门（并确认扫描恢复流动）→ DONE
   - 航向保持：目标航向取自**同一来源**起步时刻读数（默认 imu_orientation）
   - 看门狗：总里程预算、分段超时、IMU/odom 失联、坡度签名校验
   - 任何 FAULT：立即停车并重新开门，交还控制权

## 上车标定（务必先做）

```bash
catkin_make --pkg ucar_2026_upanddown   # 或整体 catkin_make
source devel/setup.bash

# 1. 单独起门控 + 坡道节点（不起导航栈）
roslaunch ucar_2026_upanddown ramp_traverse.launch

# 2. 车放到 22° 上坡上，观察（pitch_deg 必须为正，约 +22）
rostopic echo /ramp_traverse/status | grep -E "pitch_deg|max_pitch"

# 3. 若 pitch 为负 → config/ramp_traverse.yaml 改 pitch_sign: -1.0
# 4. 平地上 pitch_deg 应≈0，否则把稳定偏差填入 level_offset_deg
# 5. 手动遥控测试：先取消 move_base 目标，再
rosservice call /ramp_traverse/start
```

## 手动控制门控

```bash
rosservice call /scan_gate/set_open "data: false"   # 关门（冻结AMCL）
rosservice call /scan_gate/set_open "data: true"    # 开门（恢复AMCL）
rostopic echo /scan_gate/status                     # state/forwarded/dropped
```
