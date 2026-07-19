# ucar_2026_traffic_stop_line

任务4停止线专用控制器。`move_base` 只负责抵达停止线后方预停点；本节点获得可靠
服务触发后，使用相机完成横线搜索、角度对齐、低速接近和停车验证。

## 首次相机标定

将车头人工放在距白色停止线 `6cm` 且垂直于白线的位置，然后运行：

```bash
roslaunch ucar_2026_traffic_stop_line traffic_stop_line.launch \
  start_camera:=true calibrate_only:=true publish_debug:=true
```

节点累计30帧有效横线后写入：

```text
~/.ros/traffic_stop_line_calibration.yaml
```

标定过程中节点不会发布非零速度。可通过
`rqt_image_view /traffic_stop_line/debug_image` 检查候选框和目标行。

## 单独测试任务4

完整启动导航、相机和语音：

```bash
roslaunch ucar_2026_competition task4.launch debug:=true
```

如果任务1、2的导航栈、相机和语音仍在运行，则只附加任务4总控，避免重复节点：

```bash
roslaunch ucar_2026_competition task4.launch \
  start_nav:=false start_camera:=false start_speech:=false \
  start_external_voice:=false debug:=true
```

视觉停稳并发布 `/traffic_stop_line/status=stopped` 前，总控不会启动交通灯识别。
