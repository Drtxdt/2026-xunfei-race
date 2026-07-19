#!/usr/bin/env python3
"""Fail-safe visual controller that stops before the task4 white line."""

from __future__ import annotations

import json
import math
import os
import threading
from collections import deque

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from stop_line_logic import (
    approach_speed,
    clamp,
    confirmed_window,
    detect_stop_line,
    load_calibration,
    safety_failure,
    save_calibration,
    target_position_state,
)


class TrafficStopLineNode:
    TERMINAL = frozenset(("stopped", "failed", "calibrated", "calibration_failed"))

    def __init__(self):
        rospy.init_node("traffic_stop_line")
        self.lock = threading.RLock()
        self.bridge = CvBridge()
        self._read_params()

        self.state = "calibrating" if self.calibrate_only else "ready"
        self.failure_reason = ""
        self.active = False
        self.started_at = 0.0
        self.start_pose = None
        self.last_pose = None
        self.travelled = 0.0
        self.odom_pose = None
        self.odom_twist = None
        self.odom_at = 0.0
        self.scan_at = 0.0
        self.front_distance = float("inf")
        self.image_at = 0.0
        self.image_shape = None
        self.latest_detection = None
        self.latest_detection_at = 0.0
        self.hit_window = deque(maxlen=self.window_size)
        self.align_since = None
        self.verify_since = None
        self.calibration_detections = []
        self.last_mask = None
        self.last_frame = None

        self.calibration = None if self.calibrate_only else load_calibration(
            self.calibration_file)
        if not self.calibrate_only and self.calibration is None:
            self.state = "calibration_missing"
            self.failure_reason = "missing or invalid calibration: {}".format(
                self.calibration_file)

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=5, latch=True)
        self.result_pub = rospy.Publisher(
            self.result_topic, String, queue_size=10, latch=True)
        self.debug_pub = rospy.Publisher(
            self.debug_image_topic, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self._image_cb,
            queue_size=1, buff_size=2 ** 24)
        self.odom_sub = rospy.Subscriber(
            self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        self.scan_sub = rospy.Subscriber(
            self.scan_topic, LaserScan, self._scan_cb, queue_size=1)
        self.start_srv = rospy.Service(
            self.start_service, Trigger, self._start_cb)
        self.timer = rospy.Timer(rospy.Duration(0.05), self._control_timer)
        rospy.on_shutdown(self._shutdown)
        self._publish_status(self.state)
        rospy.loginfo(
            "traffic_stop_line ready calibrate_only=%s calibration=%s",
            self.calibrate_only, self.calibration_file)

    def _read_params(self):
        gp = rospy.get_param
        self.image_topic = gp("~image_topic", "/usb_cam/image_raw")
        self.odom_topic = gp("~odom_topic", "/odom")
        self.scan_topic = gp("~scan_topic", "/scan")
        self.cmd_vel_topic = gp("~cmd_vel_topic", "/cmd_vel")
        self.status_topic = gp("~status_topic", "/traffic_stop_line/status")
        self.result_topic = gp("~result_topic", "/traffic_stop_line/result")
        self.debug_image_topic = gp(
            "~debug_image_topic", "/traffic_stop_line/debug_image")
        self.start_service = gp("~start_service", "/traffic_stop_line/start")
        self.calibrate_only = bool(gp("~calibrate_only", False))
        self.calibration_samples = max(3, int(gp("~calibration_samples", 30)))
        self.calibration_file = os.path.abspath(os.path.expanduser(
            str(gp("~calibration_file", "~/.ros/traffic_stop_line_calibration.yaml"))))
        self.target_front_gap = float(gp("~target_front_gap_m", 0.06))
        self.publish_debug = bool(gp("~publish_debug", True))

        names = (
            "roi_y_start_ratio", "roi_y_end_ratio", "white_s_max",
            "white_v_min", "gray_white_threshold", "morph_kernel_size",
            "min_area", "max_area_ratio", "min_fill_ratio",
            "min_aspect_ratio", "min_width_ratio", "max_height_ratio",
            "max_detection_angle_deg",
        )
        self.detection_params = {name: gp("~" + name) for name in names}
        self.window_size = max(1, int(gp("~window_size", 5)))
        self.required_hits = max(1, int(gp("~required_hits", 3)))
        if self.required_hits > self.window_size:
            raise ValueError("required_hits cannot exceed window_size")

        self.search_speed = min(0.05, abs(float(gp("~search_speed", 0.05))))
        self.search_timeout = float(gp("~search_timeout_sec", 15.0))
        self.max_search_distance = float(gp("~max_search_distance", 0.60))
        self.speed_far = min(0.06, abs(float(gp("~approach_speed_far", 0.06))))
        self.speed_mid = min(0.035, abs(float(gp("~approach_speed_mid", 0.035))))
        self.speed_near = min(0.02, abs(float(gp("~approach_speed_near", 0.02))))
        self.far_error = float(gp("~approach_far_error_ratio", 0.12))
        self.mid_error = float(gp("~approach_mid_error_ratio", 0.04))
        self.target_y_tolerance = float(gp("~target_y_tolerance_ratio", 0.012))
        self.overshoot_margin = float(gp("~overshoot_margin_ratio", 0.02))
        self.angle_tolerance = float(gp("~angle_tolerance_deg", 3.0))
        self.align_stable_sec = float(gp("~align_stable_sec", 0.4))
        self.verify_stable_sec = float(gp("~verify_stable_sec", 0.5))
        self.angular_kp = float(gp("~angular_kp", 0.035))
        self.angular_min = abs(float(gp("~angular_min_speed", 0.15)))
        self.angular_max = abs(float(gp("~angular_max_speed", 0.18)))
        self.steering_sign = 1.0 if float(gp("~steering_sign", 1.0)) >= 0 else -1.0
        self.sensor_timeout = float(gp("~sensor_timeout_sec", 0.5))
        self.line_lost_timeout = float(gp("~line_lost_timeout_sec", 0.3))
        self.front_obstacle_distance = float(gp("~front_obstacle_distance", 0.20))
        self.scan_half_angle = math.radians(
            float(gp("~scan_front_half_angle_deg", 15.0)))
        self.stopped_linear_tolerance = float(
            gp("~stopped_linear_tolerance", 0.01))
        self.stopped_angular_tolerance = float(
            gp("~stopped_angular_tolerance", 0.03))

    def _publish_status(self, status):
        if status != self.state:
            self.state = status
        self.status_pub.publish(String(data=self.state))

    def _publish_result(self, now):
        detection = self.latest_detection or {}
        front_distance = (
            self.front_distance if math.isfinite(self.front_distance) else None)
        payload = {
            "stamp": now,
            "state": self.state,
            "active": self.active,
            "confirmed": confirmed_window(self.hit_window, self.required_hits),
            "confidence": detection.get("confidence", 0.0),
            "angle_deg": detection.get("angle_deg"),
            "angle_error_deg": detection.get("angle_error_deg"),
            "center_y_ratio": detection.get("center_y_ratio"),
            "width_ratio": detection.get("width_ratio"),
            "travelled_m": self.travelled,
            "front_distance_m": front_distance,
            "failure_reason": self.failure_reason,
        }
        self.result_pub.publish(String(data=json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"))))

    def _odom_cb(self, msg):
        now = rospy.get_time()
        pose = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))
        with self.lock:
            if self.active and self.last_pose is not None:
                step = math.hypot(
                    pose[0] - self.last_pose[0], pose[1] - self.last_pose[1])
                if step < 0.20:
                    self.travelled += step
            self.odom_pose = pose
            self.last_pose = pose
            self.odom_twist = (
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.linear.y),
                float(msg.twist.twist.angular.z),
            )
            self.odom_at = now

    def _scan_cb(self, msg):
        front = []
        for index, value in enumerate(msg.ranges):
            if not math.isfinite(value):
                continue
            angle = msg.angle_min + index * msg.angle_increment
            angle = (angle + math.pi) % (2.0 * math.pi) - math.pi
            if abs(angle) <= self.scan_half_angle and value >= msg.range_min:
                front.append(float(value))
        with self.lock:
            self.front_distance = min(front) if front else float("inf")
            self.scan_at = rospy.get_time()

    def _image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(2.0, "stop-line cv_bridge failed: %s", exc)
            return
        now = rospy.get_time()
        target_angle = 0.0
        with self.lock:
            if self.calibration is not None:
                target_angle = self.calibration["target_angle_deg"]
        detection, mask = detect_stop_line(
            frame, self.detection_params, target_angle_deg=target_angle)
        with self.lock:
            self.image_at = now
            self.image_shape = frame.shape
            self.last_frame = frame
            self.last_mask = mask
            self.hit_window.append(detection is not None)
            if detection is not None:
                self.latest_detection = detection
                self.latest_detection_at = now
            if self.calibrate_only and detection is not None and self.state == "calibrating":
                self.calibration_detections.append(detection)
                if len(self.calibration_detections) >= self.calibration_samples:
                    try:
                        payload = save_calibration(
                            self.calibration_file,
                            self.calibration_detections,
                            self.target_front_gap,
                            frame.shape,
                        )
                        self.failure_reason = ""
                        self._publish_status("calibrated")
                        rospy.loginfo(
                            "stop-line calibration saved: %s target_y=%.6f angle=%.3f samples=%d",
                            self.calibration_file, payload["target_y_ratio"],
                            payload["target_angle_deg"], payload["sample_count"])
                    except Exception as exc:
                        self.failure_reason = str(exc)
                        self._publish_status("calibration_failed")
            self._publish_result(now)
        self._publish_debug(frame, mask, detection)

    def _publish_debug(self, frame, mask, detection):
        if not self.publish_debug:
            return
        debug = frame.copy()
        height, width = debug.shape[:2]
        y0 = int(float(self.detection_params["roi_y_start_ratio"]) * height)
        y1 = int(float(self.detection_params["roi_y_end_ratio"]) * height)
        cv2.rectangle(debug, (0, y0), (width - 1, min(height - 1, y1)),
                      (255, 120, 0), 1)
        if detection is not None:
            x, y, box_w, box_h = detection["bbox"]
            cv2.rectangle(debug, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
        if self.calibration is not None:
            target_y = int(self.calibration["target_y_ratio"] * height)
            cv2.line(debug, (0, target_y), (width - 1, target_y), (0, 0, 255), 2)
        text = "state={} hits={}/{} travel={:.2f}m".format(
            self.state, sum(self.hit_window), self.required_hits, self.travelled)
        cv2.putText(debug, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
        except CvBridgeError:
            pass

    def _start_cb(self, _request):
        now = rospy.get_time()
        with self.lock:
            if self.calibrate_only:
                return TriggerResponse(False, "calibration-only node cannot move")
            if self.active:
                return TriggerResponse(True, "stop-line docking already active")
            if self.state == "stopped":
                return TriggerResponse(True, "already stopped at line")
            if self.calibration is None:
                self._publish_status("calibration_missing")
                return TriggerResponse(False, self.failure_reason)
            calibrated_gap = float(
                self.calibration.get("target_front_gap_m", self.target_front_gap))
            if abs(calibrated_gap - self.target_front_gap) > 0.005:
                reason = (
                    "calibration gap {:.3f}m does not match requested {:.3f}m; "
                    "recalibrate at the requested gap".format(
                        calibrated_gap, self.target_front_gap))
                self.failure_reason = reason
                self._publish_status("calibration_failed")
                return TriggerResponse(False, reason)
            stale = self._sensor_failure(now)
            if stale:
                if stale in ("image_stale", "odom_stale", "scan_stale"):
                    return TriggerResponse(False, stale)
                self.failure_reason = stale
                self._publish_status("failed")
                return TriggerResponse(False, stale)
            expected_w = int(self.calibration.get("image_width", 0))
            expected_h = int(self.calibration.get("image_height", 0))
            if (expected_w and expected_h and self.image_shape is not None and
                    (self.image_shape[1] != expected_w or self.image_shape[0] != expected_h)):
                reason = "calibration image {}x{} does not match camera {}x{}".format(
                    expected_w, expected_h, self.image_shape[1], self.image_shape[0])
                self.failure_reason = reason
                self._publish_status("failed")
                return TriggerResponse(False, reason)
            self.failure_reason = ""
            self.active = True
            self.started_at = now
            self.start_pose = self.odom_pose
            self.last_pose = self.odom_pose
            self.travelled = 0.0
            self.align_since = None
            self.verify_since = None
            self.hit_window.clear()
            self.latest_detection = None
            self.latest_detection_at = 0.0
            self._publish_status("line_searching")
        return TriggerResponse(True, "stop-line docking started")

    def _sensor_failure(self, now):
        return safety_failure(
            now, self.image_at, self.odom_at, self.scan_at,
            self.sensor_timeout, self.front_distance,
            self.front_obstacle_distance)

    def _line(self, now):
        if (self.latest_detection is None or
                now - self.latest_detection_at > self.line_lost_timeout):
            return None
        return self.latest_detection

    def _angular_command(self, angle_error):
        if abs(angle_error) <= self.angle_tolerance:
            return 0.0
        command = self.steering_sign * self.angular_kp * float(angle_error)
        magnitude = clamp(abs(command), self.angular_min, self.angular_max)
        return math.copysign(magnitude, command)

    def _fail(self, reason):
        self.active = False
        self.failure_reason = str(reason)
        self.cmd_pub.publish(Twist())
        self._publish_status("failed")
        rospy.logerr("traffic stop-line docking failed: %s", reason)

    def _control_timer(self, _event):
        now = rospy.get_time()
        with self.lock:
            if not self.active:
                if (self.calibrate_only or self.state in self.TERMINAL or
                        self.state == "calibration_missing"):
                    self.cmd_pub.publish(Twist())
                return
            failure = self._sensor_failure(now)
            if failure:
                self._fail(failure)
                return
            command = Twist()
            confirmed = confirmed_window(self.hit_window, self.required_hits)
            line = self._line(now)

            if self.state == "line_searching":
                if confirmed and line is not None:
                    self._publish_status("line_aligning")
                    self.align_since = None
                elif now - self.started_at >= self.search_timeout:
                    self._fail("line_search_timeout")
                    return
                elif self.travelled >= self.max_search_distance:
                    self._fail("line_search_distance_exceeded")
                    return
                else:
                    command.linear.x = self.search_speed

            elif self.state == "line_aligning":
                if line is None:
                    self._fail("line_lost_while_aligning")
                    return
                error = float(line["angle_error_deg"])
                if abs(error) <= self.angle_tolerance:
                    if self.align_since is None:
                        self.align_since = now
                    elif now - self.align_since >= self.align_stable_sec:
                        self._publish_status("line_approaching")
                else:
                    self.align_since = None
                    command.angular.z = self._angular_command(error)

            elif self.state == "line_approaching":
                if line is None:
                    self._fail("line_lost_while_approaching")
                    return
                target_y = float(self.calibration["target_y_ratio"])
                y_error = target_y - float(line["center_y_ratio"])
                position_state = target_position_state(
                    y_error, self.target_y_tolerance, self.overshoot_margin)
                if position_state == "overshoot":
                    self._fail("stop_line_overshoot")
                    return
                angle_error = float(line["angle_error_deg"])
                if position_state == "target":
                    self._publish_status("line_verifying")
                    self.verify_since = None
                elif abs(angle_error) > 2.0 * self.angle_tolerance:
                    command.angular.z = self._angular_command(angle_error)
                else:
                    command.linear.x = approach_speed(
                        y_error, self.speed_far, self.speed_mid, self.speed_near,
                        self.far_error, self.mid_error)
                    command.angular.z = self._angular_command(angle_error)

            elif self.state == "line_verifying":
                if line is None:
                    self._fail("line_lost_while_verifying")
                    return
                target_y = float(self.calibration["target_y_ratio"])
                y_error = target_y - float(line["center_y_ratio"])
                angle_ok = abs(float(line["angle_error_deg"])) <= self.angle_tolerance
                stopped = (self.odom_twist is not None and
                           math.hypot(self.odom_twist[0], self.odom_twist[1]) <=
                           self.stopped_linear_tolerance and
                           abs(self.odom_twist[2]) <= self.stopped_angular_tolerance)
                position_state = target_position_state(
                    y_error, self.target_y_tolerance, self.overshoot_margin)
                if position_state == "overshoot":
                    self._fail("stop_line_overshoot")
                    return
                if position_state == "target" and angle_ok and stopped:
                    if self.verify_since is None:
                        self.verify_since = now
                    elif now - self.verify_since >= self.verify_stable_sec:
                        self.active = False
                        self.failure_reason = ""
                        self._publish_status("stopped")
                        rospy.loginfo(
                            "traffic stop-line verified gap=%.3fm y_error=%+.4f angle_error=%+.2f",
                            self.calibration["target_front_gap_m"], y_error,
                            line["angle_error_deg"])
                else:
                    self.verify_since = None
                    if y_error > self.target_y_tolerance:
                        self._publish_status("line_approaching")

            self.cmd_pub.publish(command)
            self._publish_result(now)

    def _shutdown(self):
        self.active = False
        if hasattr(self, "cmd_pub"):
            for _ in range(3):
                self.cmd_pub.publish(Twist())


def main():
    TrafficStopLineNode()
    rospy.spin()


if __name__ == "__main__":
    main()
