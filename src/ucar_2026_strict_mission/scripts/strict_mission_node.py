#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail-safe mission coordinator from warehouse completion to track finish."""

from __future__ import annotations

import json
import math
import subprocess
import threading
import time

import actionlib
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from ucar_2026_strict_mission.logic import (
    ApproachPolicy,
    ConsecutiveBandFilter,
    DistanceCalibration,
    lowest_horizontal_band,
    forward_progress,
    track_launch_for_decision,
    traffic_decision_from_payload,
    valid_stop_line_geometry,
)


TERMINAL_STATES = frozenset(("DONE", "FAULT"))


def quaternion_from_yaw(yaw):
    half = float(yaw) * 0.5
    return math.sin(half), math.cos(half)


class StrictMissionNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.bridge = CvBridge()
        self.state = "WAIT_START"
        self.fault_reason = ""
        self.started = False
        self.last_image_at = 0.0
        self.line_missing_since = None
        self.last_distance_m = None
        self.odom_pose = None
        self.odom_received_at = 0.0
        self.traffic_hits = 0
        self.last_traffic_decision = None
        self.selected_decision = None
        self.track_status = {}
        self.track_process = None
        self.start_event = threading.Event()
        self.parked_event = threading.Event()
        self.traffic_event = threading.Event()
        self.shutdown_event = threading.Event()

        calibration_points = rospy.get_param(
            "~distance_calibration",
            [[0.55, 0.50], [0.65, 0.32], [0.75, 0.20],
             [0.85, 0.11], [0.90, 0.07], [0.94, 0.03]],
        )
        self.calibration = DistanceCalibration(calibration_points)
        self.target_min_m = float(rospy.get_param("~target_min_m", 0.05))
        self.target_max_m = float(rospy.get_param("~target_max_m", 0.07))
        self.policy = ApproachPolicy(
            self.target_min_m,
            self.target_max_m,
            float(rospy.get_param("~absolute_max_m", 0.10)),
            float(rospy.get_param("~calibration_error_m", 0.03)),
            speed_far=float(rospy.get_param("~speed_far", 0.10)),
            speed_medium=float(rospy.get_param("~speed_medium", 0.06)),
            speed_near=float(rospy.get_param("~speed_near", 0.05)),
            speed_creep=float(rospy.get_param("~speed_creep", 0.045)),
        )
        self.band_filter = ConsecutiveBandFilter(
            int(rospy.get_param("~stop_confirm_frames", 5)),
            self.target_min_m,
            self.target_max_m,
        )

        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.status_topic = rospy.get_param(
            "~status_topic", "/strict_mission/status")
        self.traffic_topic = rospy.get_param(
            "~traffic_topic", "/traffic_light_rknn_test/detections")
        self.competition_status_topic = rospy.get_param(
            "~competition_status_topic", "/competition/status")
        self.auto_start = bool(rospy.get_param(
            "~auto_start_on_warehouse_status", False))
        self.warehouse_complete_stage = str(rospy.get_param(
            "~warehouse_complete_stage", "task3"))
        self.required_traffic_frames = max(
            1, int(rospy.get_param("~traffic_confirm_frames", 3)))

        self.cmd_pub = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=10, latch=True)
        self.debug_pub = rospy.Publisher(
            "~debug_image", Image, queue_size=1)
        rospy.Subscriber(
            self.image_topic, Image, self.image_callback, queue_size=1,
            buff_size=2 ** 24,
        )
        rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=5)
        rospy.Subscriber(
            self.traffic_topic, String, self.traffic_callback, queue_size=10)
        rospy.Subscriber(
            self.competition_status_topic, String,
            self.competition_status_callback, queue_size=10,
        )
        for topic in (
            "/track_end_stop/status",
            "/right_track_end_stop/status",
            "/stable_right_track_end_stop/status",
        ):
            rospy.Subscriber(
                topic, String, self.track_status_callback,
                callback_args=topic, queue_size=10,
            )
        rospy.Service("~start", Trigger, self.start_service)
        rospy.Service("~abort", Trigger, self.abort_service)
        self.move_base = actionlib.SimpleActionClient(
            rospy.get_param("~move_base_action", "move_base"),
            MoveBaseAction,
        )
        self.watchdog = rospy.Timer(
            rospy.Duration(0.05), self.watchdog_callback)
        rospy.on_shutdown(self.shutdown)
        self.publish_status("waiting for explicit start")
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def publish_status(self, detail="", **extra):
        if self.state == "FAULT" and "error" not in extra:
            extra["error"] = self.fault_reason
        payload = {
            "state": self.state,
            "detail": detail,
            "distance_m": self.last_distance_m,
            "decision": self.selected_decision,
            "stamp": rospy.Time.now().to_sec(),
        }
        payload.update(extra)
        self.status_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False, sort_keys=True)))

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def set_fault(self, reason):
        with self.lock:
            if self.state in TERMINAL_STATES:
                return
            self.state = "FAULT"
            self.fault_reason = str(reason)
            self.shutdown_event.set()
        self.move_base.cancel_all_goals()
        self.publish_stop()
        self.publish_status("fail-safe stop", error=self.fault_reason)
        rospy.logerr("strict mission fault: %s", self.fault_reason)

    def start_service(self, _request):
        with self.lock:
            if self.started:
                return TriggerResponse(
                    success=False, message="mission already started")
            self.started = True
            self.start_event.set()
        return TriggerResponse(success=True, message="strict mission started")

    def abort_service(self, _request):
        self.set_fault("operator abort")
        return TriggerResponse(success=True, message="vehicle stopped")

    def competition_status_callback(self, msg):
        if not self.auto_start or self.started:
            return
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        stage = str(payload.get("stage") or payload.get("task") or "")
        state = str(payload.get("state") or payload.get("status") or "")
        if stage == self.warehouse_complete_stage and state == "completed":
            with self.lock:
                if not self.started:
                    self.started = True
                    self.start_event.set()
                    self.publish_status("warehouse completion trigger accepted")

    def odom_callback(self, msg):
        orientation = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z
                   + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y
                         + orientation.z * orientation.z),
        )
        position = msg.pose.pose.position
        with self.lock:
            self.odom_pose = (position.x, position.y, yaw)
            self.odom_received_at = time.monotonic()

    def detect_stop_line(self, frame):
        height, width = frame.shape[:2]
        roi_start = float(rospy.get_param("~line_roi_start_ratio", 0.45))
        y0 = max(0, min(height - 1, int(height * roi_start)))
        roi = frame[y0:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = (
            0,
            0,
            int(rospy.get_param("~white_v_min", 165)),
        )
        upper = (
            180,
            int(rospy.get_param("~white_s_max", 85)),
            255,
        )
        mask = cv2.inRange(hsv, lower, upper)
        kernel_size = max(3, int(rospy.get_param("~morph_kernel_size", 5)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width <= 0 or box_height <= 0:
                continue
            area = float(cv2.contourArea(contour))
            width_ratio = float(box_width) / float(width)
            height_ratio = float(box_height) / float(height)
            fill_ratio = area / float(box_width * box_height)
            bottom_ratio = float(y0 + y + box_height) / float(height)
            if valid_stop_line_geometry(
                width_ratio,
                height_ratio,
                fill_ratio,
                bottom_ratio,
                min_width_ratio=float(rospy.get_param(
                    "~line_min_width_ratio", 0.45)),
                max_height_ratio=float(rospy.get_param(
                    "~line_max_height_ratio", 0.12)),
                min_fill_ratio=float(rospy.get_param(
                    "~line_min_fill_ratio", 0.55)),
                min_bottom_ratio=roi_start,
            ):
                candidates.append((
                    width_ratio * bottom_ratio,
                    bottom_ratio,
                    (x, y0 + y, box_width, box_height),
                ))
        if not candidates:
            row_occupancies = np.count_nonzero(mask, axis=1) / float(width)
            band = lowest_horizontal_band(
                row_occupancies,
                float(rospy.get_param("~line_min_width_ratio", 0.45)),
                int(round(height * float(rospy.get_param(
                    "~line_max_height_ratio", 0.12)))),
            )
            if band is None:
                return None, mask, None
            start, end = band
            bottom_ratio = float(y0 + end + 1) / float(height)
            return bottom_ratio, mask, (0, y0 + start, width, end - start + 1)
        _, bottom_ratio, box = max(candidates, key=lambda item: item[0])
        return bottom_ratio, mask, box

    def image_callback(self, msg):
        now = time.monotonic()
        with self.lock:
            self.last_image_at = now
            if self.state != "APPROACH_LINE":
                return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.set_fault("cv_bridge failed: {}".format(exc))
            return
        bottom_ratio, mask, box = self.detect_stop_line(frame)
        if bottom_ratio is None:
            self.publish_stop()
            self.band_filter.reset()
            if self.line_missing_since is None:
                self.line_missing_since = now
            self.publish_status("stop line not trusted; holding stop")
            return
        self.line_missing_since = None
        distance = self.calibration.distance_for_ratio(bottom_ratio)
        self.last_distance_m = distance
        if distance is None:
            self.publish_stop()
            self.band_filter.reset()
            self.publish_status(
                "line outside calibrated range; holding stop",
                line_bottom_ratio=bottom_ratio,
            )
            return
        command = Twist()
        command.linear.x = self.policy.command_for_distance(distance)
        self.cmd_pub.publish(command)
        if self.band_filter.push(distance):
            self.publish_stop()
            with self.lock:
                self.state = "FINAL_ADVANCE"
                self.parked_event.set()
            self.publish_status(
                "visual stop band confirmed; arming odometry final advance",
                line_bottom_ratio=bottom_ratio,
            )
        else:
            self.publish_status(
                "closed-loop line approach",
                line_bottom_ratio=bottom_ratio,
                commanded_speed_mps=command.linear.x,
            )
        if box is not None and self.debug_pub.get_num_connections() > 0:
            x, y, box_width, box_height = box
            cv2.rectangle(
                frame, (x, y), (x + box_width, y + box_height),
                (0, 0, 255), 2,
            )
            cv2.putText(
                frame, "distance={:.3f}m".format(distance), (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
            self.debug_pub.publish(
                self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))

    def traffic_callback(self, msg):
        with self.lock:
            if self.state != "WAIT_TRAFFIC":
                return
        try:
            decision = traffic_decision_from_payload(json.loads(msg.data))
        except (TypeError, ValueError):
            decision = None
        if decision is None:
            self.last_traffic_decision = None
            self.traffic_hits = 0
            return
        self.publish_stop()
        if decision == "stop":
            self.last_traffic_decision = "stop"
            self.traffic_hits = 0
            self.publish_status("red light; holding strict stop")
            return
        if decision == self.last_traffic_decision:
            self.traffic_hits += 1
        else:
            self.last_traffic_decision = decision
            self.traffic_hits = 1
        if self.traffic_hits >= self.required_traffic_frames:
            with self.lock:
                self.selected_decision = decision
                self.traffic_event.set()
            self.publish_status("traffic direction confirmed")

    def track_status_callback(self, msg, topic):
        self.track_status[topic] = str(msg.data).strip()

    def watchdog_callback(self, _event):
        with self.lock:
            state = self.state
            last_image_at = self.last_image_at
        if state in ("STOP_CONFIRM", "WAIT_TRAFFIC", "FAULT"):
            self.publish_stop()
        if state != "APPROACH_LINE":
            return
        now = time.monotonic()
        stale_stop_sec = float(rospy.get_param("~image_stale_stop_sec", 0.25))
        stale_fault_sec = float(rospy.get_param("~image_stale_fault_sec", 1.0))
        if last_image_at <= 0.0 or now - last_image_at >= stale_stop_sec:
            self.publish_stop()
        if last_image_at > 0.0 and now - last_image_at >= stale_fault_sec:
            self.set_fault("camera image timeout")
        missing_timeout = float(rospy.get_param(
            "~line_missing_fault_sec", 2.0))
        if (self.line_missing_since is not None
                and now - self.line_missing_since >= missing_timeout):
            self.set_fault("stop line lost during approach")

    def navigate_to_staging_pose(self):
        if not bool(rospy.get_param("~traffic_pose_configured", False)):
            raise RuntimeError(
                "traffic_pose_configured is false; set staging coordinates")
        timeout = float(rospy.get_param("~navigation_timeout_sec", 120.0))
        if not self.move_base.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError("move_base action server unavailable")
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = rospy.get_param(
            "~traffic_frame", "map")
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = float(rospy.get_param(
            "~traffic_staging_x"))
        goal.target_pose.pose.position.y = float(rospy.get_param(
            "~traffic_staging_y"))
        sin_half, cos_half = quaternion_from_yaw(
            float(rospy.get_param("~traffic_staging_yaw")))
        goal.target_pose.pose.orientation.z = sin_half
        goal.target_pose.pose.orientation.w = cos_half
        self.move_base.send_goal(goal)
        if not self.move_base.wait_for_result(rospy.Duration(timeout)):
            self.move_base.cancel_goal()
            raise RuntimeError("navigation to stop-line staging pose timed out")
        if self.move_base.get_state() != 3:
            raise RuntimeError(
                "navigation failed with action state {}".format(
                    self.move_base.get_state()))

    def final_advance(self):
        target = float(rospy.get_param("~final_advance_m", 0.0))
        if target <= 0.0:
            return
        speed = float(rospy.get_param("~final_advance_speed", 0.045))
        timeout = float(rospy.get_param("~final_advance_timeout_sec", 6.0))
        stale = float(rospy.get_param("~final_advance_odom_stale_sec", 0.5))
        if speed <= 0.0 or timeout <= 0.0 or stale <= 0.0:
            raise RuntimeError("final advance parameters must be positive")

        wait_deadline = time.monotonic() + min(2.0, timeout)
        start_pose = None
        while not rospy.is_shutdown() and time.monotonic() < wait_deadline:
            with self.lock:
                pose = self.odom_pose
                age = time.monotonic() - self.odom_received_at
            if pose is not None and age <= stale:
                start_pose = pose
                break
            self.publish_stop()
            time.sleep(0.02)
        if start_pose is None:
            raise RuntimeError("fresh odometry unavailable for final advance")

        deadline = time.monotonic() + timeout
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                pose = self.odom_pose
                age = time.monotonic() - self.odom_received_at
            if pose is None or age > stale:
                self.publish_stop()
                raise RuntimeError("odometry became stale during final advance")
            progress = forward_progress(start_pose, pose)
            if progress >= target:
                self.publish_stop()
                self.publish_status(
                    "odometry final advance completed",
                    final_advance_m=progress,
                )
                return
            command = Twist()
            command.linear.x = speed
            self.cmd_pub.publish(command)
            self.publish_status(
                "odometry final advance",
                final_advance_m=progress,
                final_advance_target_m=target,
                commanded_speed_mps=speed,
            )
            rate.sleep()
        self.publish_stop()
        raise RuntimeError("odometry final advance timed out")

    def launch_track(self, decision):
        launch_file, status_topic, finish_value = track_launch_for_decision(
            decision)
        command = [
            "roslaunch", "ucar_2026_track_end_stop", launch_file,
            "start_driver:=false", "start_camera:=false",
            "start_viewer:=false",
        ]
        self.track_process = subprocess.Popen(command)
        return status_topic, finish_value

    def wait_event(self, event, timeout, description):
        deadline = time.monotonic() + float(timeout)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.state == "FAULT":
                raise RuntimeError(self.fault_reason)
            if event.wait(0.05):
                return
        raise RuntimeError("{} timed out".format(description))

    def run(self):
        self.start_event.wait()
        if rospy.is_shutdown():
            return
        try:
            with self.lock:
                self.state = "NAVIGATING"
            self.publish_status("navigating to calibrated staging pose")
            self.navigate_to_staging_pose()
            self.publish_stop()
            with self.lock:
                self.state = "APPROACH_LINE"
                self.last_image_at = time.monotonic()
            self.publish_status("visual stop-line approach armed")
            self.wait_event(
                self.parked_event,
                float(rospy.get_param("~line_approach_timeout_sec", 75.0)),
                "strict line approach",
            )
            self.final_advance()
            with self.lock:
                self.state = "STOP_CONFIRM"
            settle = float(rospy.get_param("~stop_settle_sec", 0.6))
            settle_deadline = time.monotonic() + settle
            while time.monotonic() < settle_deadline:
                self.publish_stop()
                time.sleep(0.02)
            with self.lock:
                self.state = "WAIT_TRAFFIC"
            self.publish_status("vehicle held; waiting for traffic consensus")
            self.wait_event(
                self.traffic_event,
                float(rospy.get_param("~traffic_timeout_sec", 180.0)),
                "traffic recognition",
            )
            self.publish_stop()
            with self.lock:
                self.state = "TRACKING"
            status_topic, finish_value = self.launch_track(
                self.selected_decision)
            self.publish_status(
                "matching track controller launched",
                track_status_topic=status_topic,
                expected_finish=finish_value,
            )
            deadline = time.monotonic() + float(rospy.get_param(
                "~track_timeout_sec", 420.0))
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                if self.track_process.poll() is not None:
                    raise RuntimeError(
                        "track controller exited before finish")
                if self.track_status.get(status_topic) == finish_value:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("line following timed out")
            self.publish_stop()
            with self.lock:
                self.state = "DONE"
            self.publish_status("strict post-warehouse mission completed")
        except Exception as exc:
            self.set_fault(str(exc))

    def shutdown(self):
        self.shutdown_event.set()
        try:
            self.move_base.cancel_all_goals()
        except Exception:
            pass
        for _ in range(10):
            self.publish_stop()
        if self.track_process and self.track_process.poll() is None:
            self.track_process.terminate()
            try:
                self.track_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.track_process.kill()


def main():
    rospy.init_node("strict_mission")
    StrictMissionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
