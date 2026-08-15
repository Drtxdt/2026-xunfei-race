#!/usr/bin/env python3
"""Exclusive cmd_vel mux for the national-finals patrol-line board."""

from __future__ import annotations

import json
import math
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from ucar_2026_track_end_stop.obstacle_logic import ObstacleAvoidanceController


def sector_min(msg, center, half_width):
    values = []
    angle = float(msg.angle_min)
    for value in msg.ranges:
        delta = (angle - center + math.pi) % (2.0 * math.pi) - math.pi
        if abs(delta) <= half_width and math.isfinite(value):
            if msg.range_min <= value <= msg.range_max:
                values.append(float(value))
        angle += float(msg.angle_increment)
    return min(values) if values else float("inf")


class ObstacleAvoidanceMux(object):
    def __init__(self):
        self.lock = threading.RLock()
        self.raw = Twist()
        self.raw_at = 0.0
        self.pose = None
        self.odom_at = 0.0
        self.clearances = {}
        self.scan_at = 0.0
        self.controller = ObstacleAvoidanceController(
            trigger_distance=rospy.get_param("~obstacle_trigger_distance_m", 0.58),
            clear_distance=rospy.get_param("~obstacle_clear_distance_m", 0.72),
            confirm_scans=rospy.get_param("~obstacle_confirm_scans", 3),
            min_side_clearance=rospy.get_param("~obstacle_min_side_clearance_m", 0.45),
            min_shift=rospy.get_param("~obstacle_min_shift_m", 0.24),
            max_shift=rospy.get_param("~obstacle_max_shift_m", 0.82),
            shift_speed=rospy.get_param("~obstacle_shift_speed_mps", 0.12),
            pass_distance=rospy.get_param("~obstacle_pass_distance_m", 0.58),
            pass_speed=rospy.get_param("~obstacle_pass_speed_mps", 0.12),
            return_tolerance=rospy.get_param("~obstacle_return_tolerance_m", 0.025),
            stop_hold=rospy.get_param("~obstacle_stop_hold_sec", 0.20),
            reacquire_duration=rospy.get_param("~obstacle_reacquire_sec", 1.2),
            reacquire_speed=rospy.get_param("~obstacle_reacquire_speed_mps", 0.07),
            enable_delay=rospy.get_param("~obstacle_enable_delay_sec", 4.0),
            emergency_distance=rospy.get_param("~obstacle_emergency_distance_m", 0.16),
        )
        self.stale_sec = float(rospy.get_param("~obstacle_sensor_stale_sec", 0.35))
        self.raw_stale_sec = float(rospy.get_param("~obstacle_raw_cmd_stale_sec", 0.30))
        self.front_half = math.radians(float(rospy.get_param(
            "~obstacle_front_half_angle_deg", 14.0)))
        self.side_half = math.radians(float(rospy.get_param(
            "~obstacle_side_half_angle_deg", 32.0)))
        self.cmd_pub = rospy.Publisher(
            rospy.get_param("~cmd_vel_topic", "/cmd_vel"), Twist, queue_size=2)
        self.status_pub = rospy.Publisher(
            rospy.get_param("~status_topic", "/national_obstacle_avoidance/status"),
            String, queue_size=5, latch=True)
        rospy.Subscriber(
            rospy.get_param("~raw_cmd_vel_topic", "/track_end_stop/raw_cmd_vel"),
            Twist, self.raw_cb, queue_size=2)
        rospy.Subscriber(
            rospy.get_param("~odom_topic", "/odom"),
            Odometry, self.odom_cb, queue_size=10)
        rospy.Subscriber(
            rospy.get_param("~scan_topic", "/scan"),
            LaserScan, self.scan_cb, queue_size=1)
        rospy.on_shutdown(self.stop)

    def raw_cb(self, msg):
        with self.lock:
            self.raw = msg
            self.raw_at = time.monotonic()

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.pose = (
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y), yaw)
            self.odom_at = time.monotonic()

    def scan_cb(self, msg):
        with self.lock:
            self.clearances = {
                "front": sector_min(msg, 0.0, self.front_half),
                "left": sector_min(msg, math.pi / 2.0, self.side_half),
                "right": sector_min(msg, -math.pi / 2.0, self.side_half),
                "rear": sector_min(msg, math.pi, self.front_half),
            }
            self.scan_at = time.monotonic()

    def stop(self):
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass

    def run(self):
        rate = rospy.Rate(max(10.0, float(rospy.get_param("~control_rate_hz", 30.0))))
        last_state = None
        while not rospy.is_shutdown():
            now = time.monotonic()
            with self.lock:
                pose = self.pose
                clearances = dict(self.clearances)
                raw = (self.raw.linear.x, self.raw.linear.y, self.raw.angular.z)
                odom_age = now - self.odom_at
                scan_age = now - self.scan_at
                raw_age = now - self.raw_at
            if (pose is None or odom_age > self.stale_sec or
                    scan_age > self.stale_sec or raw_age > self.raw_stale_sec):
                command = (0.0, 0.0, 0.0)
                detail = "stale_sensor_or_command"
            else:
                command = self.controller.update(now, pose, clearances, raw)
                detail = self.controller.fault
            output = Twist()
            output.linear.x, output.linear.y, output.angular.z = command
            self.cmd_pub.publish(output)
            state = self.controller.state
            if state != last_state or state == "FAULT":
                payload = {
                    "state": state,
                    "detail": detail,
                    "front_m": clearances.get("front"),
                    "left_m": clearances.get("left"),
                    "right_m": clearances.get("right"),
                    "stamp": time.time(),
                }
                self.status_pub.publish(String(data=json.dumps(payload)))
                rospy.loginfo("national obstacle mux: %s %s", state, detail)
                last_state = state
            rate.sleep()


def main():
    rospy.init_node("national_obstacle_avoidance_mux")
    ObstacleAvoidanceMux().run()


if __name__ == "__main__":
    main()
