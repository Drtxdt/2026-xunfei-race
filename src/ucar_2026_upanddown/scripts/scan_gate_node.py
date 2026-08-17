#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""雷达门控中继: 过坡期间冻结 AMCL / 局部代价地图的扫描输入。

订阅原始雷达话题, 转发到门控话题（国赛导航栈的 AMCL 与 move_base
订阅门控话题）。开门时原样转发; 关门时丢帧, AMCL 的 map->odom 冻结
（坡上退化为纯里程计推算）, 局部代价地图不会再把坡道标成障碍。

日志约定:
  【雷达门控·初始化】 启动参数
  【雷达门控·切换】   开/关门事件
  【雷达门控·统计】   周期性转发/丢弃计数
"""

from __future__ import annotations

import json
import threading

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolResponse


class ScanGateNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.input_topic = rospy.get_param("~input_topic", "/scan")
        self.output_topic = rospy.get_param("~output_topic", "/scan_gated")
        self.status_topic = rospy.get_param("~status_topic", "/scan_gate/status")
        self.stats_log_interval_sec = max(
            0.0, float(rospy.get_param("~stats_log_interval_sec", 10.0)))
        self.open_state = bool(rospy.get_param("~initial_open", True))
        self.forwarded_count = 0
        self.dropped_count = 0
        self.last_forward_at = 0.0
        self._last_stats_log_at = 0.0

        if not bool(rospy.get_param("~config_loaded", False)):
            rospy.logwarn(
                "【雷达门控·初始化】警告: 未检测到 config_loaded 标记! "
                "scan_gate.yaml 没有被加载, 当前使用内置默认参数!")

        self.scan_pub = rospy.Publisher(
            self.output_topic, LaserScan, queue_size=10)
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=5, latch=True)
        rospy.Service("~set_open", SetBool, self.set_open_service)
        rospy.Subscriber(
            self.input_topic, LaserScan, self.scan_callback, queue_size=10)
        self.publish_status("门控就绪")
        rospy.loginfo(
            "【雷达门控·初始化】%s -> %s | 初始状态=%s | 统计日志间隔=%.0fs",
            self.input_topic, self.output_topic,
            "开(转发)" if self.open_state else "关(丢帧)",
            self.stats_log_interval_sec)

    def set_open_service(self, request):
        with self.lock:
            previous = self.open_state
            self.open_state = bool(request.data)
            state = self.open_state
            forwarded = self.forwarded_count
            dropped = self.dropped_count
        changed = previous != state
        if changed:
            rospy.loginfo(
                "【雷达门控·切换】%s -> %s | 历史累计: 转发=%d帧 丢弃=%d帧 | "
                "%s",
                "开(转发)" if previous else "关(丢帧)",
                "开(转发)" if state else "关(丢帧)",
                forwarded, dropped,
                "AMCL 定位已冻结, 坡上改为里程计推算" if not state
                else "AMCL 定位恢复更新")
        else:
            rospy.loginfo(
                "【雷达门控·切换】请求状态与当前一致(仍为%s), 无动作",
                "开" if state else "关")
        self.publish_status(
            "open" if state else "closed", changed=changed)
        return SetBoolResponse(success=True,
                               message="open" if state else "closed")

    def scan_callback(self, msg):
        forward = False
        log_stats = False
        with self.lock:
            if self.open_state:
                self.forwarded_count += 1
                self.last_forward_at = msg.header.stamp.to_sec() or rospy.get_time()
                forward = True
                if self.stats_log_interval_sec > 0.0:
                    now = rospy.get_time()
                    if now - self._last_stats_log_at >= self.stats_log_interval_sec:
                        self._last_stats_log_at = now
                        log_stats = True
            else:
                self.dropped_count += 1
        if forward:
            self.scan_pub.publish(msg)
            if log_stats:
                rospy.loginfo(
                    "【雷达门控·统计】开门运行中: 累计转发=%d帧 累计丢弃=%d帧",
                    self.forwarded_count, self.dropped_count)

    def publish_status(self, detail="", changed=None):
        with self.lock:
            payload = {
                "state": "open" if self.open_state else "closed",
                "forwarded": self.forwarded_count,
                "dropped": self.dropped_count,
                "last_forward_at": self.last_forward_at,
                "detail": detail,
                "changed": changed,
                "stamp": rospy.get_time(),
            }
        self.status_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False, sort_keys=True)))


if __name__ == "__main__":
    rospy.init_node("scan_gate")
    ScanGateNode()
    rospy.spin()
