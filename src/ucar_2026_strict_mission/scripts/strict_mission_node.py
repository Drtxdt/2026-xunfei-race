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
import dynamic_reconfigure.client
import numpy as np
import rospy
import tf2_ros
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
    StableLineDistanceFilter,
    forward_progress,
    heading_alignment_command,
    lateral_displacement,
    line_alignment_command,
    lowest_horizontal_band,
    select_final_advance,
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
        self.line_search_origin_yaw = None
        self.line_search_direction = 1.0
        self.line_search_reversals = 0
        self.last_distance_m = None
        self.last_distance_at = 0.0
        self.last_stop_line_color = None
        self.last_distance_color = None
        self.visual_stop_distance_m = None
        self.visual_stop_distance_at = 0.0
        self.visual_stop_line_color = None
        self.planned_final_advance_m = 0.0
        self.final_advance_source = "unplanned"
        self.final_progress_m = 0.0
        self.final_lateral_drift_m = 0.0
        self.final_yaw_drift_deg = 0.0
        self.final_visual_verified = False
        self.final_stop_distance_m = None
        self.final_stop_line_color = None
        self.final_stop_confirm_hits = 0
        self.final_stop_source = "unconfirmed"
        self.odom_pose = None
        self.odom_received_at = 0.0
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.traffic_hits = 0
        self.last_traffic_decision = None
        self.selected_decision = None
        self.track_status = {}
        self.track_process = None
        self.start_event = threading.Event()
        self.parked_event = threading.Event()
        self.final_parked_event = threading.Event()
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
            int(rospy.get_param("~stop_confirm_frames", 8)),
            self.target_min_m,
            self.target_max_m,
        )
        self.final_distance_filter = StableLineDistanceFilter(
            int(rospy.get_param("~final_visual_confirm_frames", 3)),
            float(rospy.get_param("~final_visual_max_spread_m", 0.02)),
        )
        self.final_advance_m = float(rospy.get_param(
            "~final_advance_m", 0.0))
        if not 0.0 <= self.final_advance_m <= 0.20:
            raise ValueError(
                "final_advance_m must be within [0.0, 0.20]")
        self.final_target_clearance_m = float(rospy.get_param(
            "~final_advance_target_clearance_m", 0.05))
        self.final_no_vision_fallback_m = float(rospy.get_param(
            "~final_advance_no_vision_m", 0.155))
        self.final_visual_max_age_sec = float(rospy.get_param(
            "~final_advance_visual_max_age_sec", 0.75))
        self.final_minimum_command_m = float(rospy.get_param(
            "~final_advance_min_command_m", 0.015))
        self.final_visual_bias_m = float(rospy.get_param(
            "~final_advance_visual_bias_m", 0.03))
        select_final_advance(
            None,
            None,
            self.final_target_clearance_m,
            self.final_advance_m,
            self.final_no_vision_fallback_m,
            self.final_visual_max_age_sec,
            self.final_minimum_command_m,
            self.final_visual_bias_m,
        )
        self.precision_start_m = float(rospy.get_param(
            "~precision_start_m", 0.14))
        self.line_yaw_tolerance_rad = math.radians(float(rospy.get_param(
            "~line_yaw_tolerance_deg", 3.0)))
        self.line_center_tolerance = float(rospy.get_param(
            "~line_center_tolerance_ratio", 0.06))
        self.final_yaw_tolerance_rad = math.radians(float(rospy.get_param(
            "~final_yaw_tolerance_deg", 1.5)))
        self.final_center_tolerance = float(rospy.get_param(
            "~final_center_tolerance_ratio", 0.03))
        if self.precision_start_m <= self.target_max_m:
            raise ValueError(
                "precision_start_m must exceed the target stop distance")
        if self.final_yaw_tolerance_rad > self.line_yaw_tolerance_rad:
            raise ValueError(
                "final yaw tolerance must not exceed coarse tolerance")
        if self.final_center_tolerance > self.line_center_tolerance:
            raise ValueError(
                "final center tolerance must not exceed coarse tolerance")

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
            "line_color": self.last_distance_color,
            "visual_stop_distance_m": self.visual_stop_distance_m,
            "visual_stop_line_color": self.visual_stop_line_color,
            "final_advance_m": self.planned_final_advance_m,
            "final_advance_limit_m": self.final_advance_m,
            "final_advance_source": self.final_advance_source,
            "final_progress_m": self.final_progress_m,
            "final_lateral_drift_m": self.final_lateral_drift_m,
            "final_yaw_drift_deg": self.final_yaw_drift_deg,
            "final_visual_verified": self.final_visual_verified,
            "final_stop_distance_m": self.final_stop_distance_m,
            "final_stop_line_color": self.final_stop_line_color,
            "final_stop_confirm_hits": self.final_stop_confirm_hits,
            "final_stop_source": self.final_stop_source,
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
        """Detect only the competition's yellow stop line."""
        height, width = frame.shape[:2]
        roi_start = float(rospy.get_param("~line_roi_start_ratio", 0.45))
        y0 = max(0, min(height - 1, int(height * roi_start)))
        roi = frame[y0:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = (
            int(rospy.get_param("~yellow_h_min", 12)),
            int(rospy.get_param("~yellow_s_min", 70)),
            int(rospy.get_param("~yellow_v_min", 70)),
        )
        upper = (
            int(rospy.get_param("~yellow_h_max", 42)),
            int(rospy.get_param("~yellow_s_max", 255)),
            int(rospy.get_param("~yellow_v_max", 255)),
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
        max_abs_angle = max(1.0, float(rospy.get_param(
            "~line_max_abs_angle_deg", 35.0)))
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width <= 0 or box_height <= 0:
                continue
            area = float(cv2.contourArea(contour))
            bottom_ratio = float(y0 + y + box_height) / float(height)
            rect = cv2.minAreaRect(contour)
            (center_x, _center_y), (rect_width, rect_height), angle_deg = rect
            long_side = max(float(rect_width), float(rect_height))
            short_side = min(float(rect_width), float(rect_height))
            if long_side <= 0.0 or short_side <= 0.0:
                continue
            oriented_width_ratio = long_side / float(width)
            oriented_height_ratio = short_side / float(height)
            oriented_fill_ratio = area / (long_side * short_side)
            if rect_width < rect_height:
                angle_deg += 90.0
            while angle_deg > 90.0:
                angle_deg -= 180.0
            while angle_deg <= -90.0:
                angle_deg += 180.0
            if valid_stop_line_geometry(
                oriented_width_ratio,
                oriented_height_ratio,
                oriented_fill_ratio,
                bottom_ratio,
                min_width_ratio=float(rospy.get_param(
                    "~line_min_width_ratio", 0.45)),
                max_height_ratio=float(rospy.get_param(
                    "~line_max_height_ratio", 0.12)),
                min_fill_ratio=float(rospy.get_param(
                    "~line_min_fill_ratio", 0.55)),
                min_bottom_ratio=roi_start,
            ) and abs(angle_deg) <= max_abs_angle:
                center_error = (
                    float(center_x) - 0.5 * float(width)
                ) / (0.5 * float(width))
                candidates.append((
                    oriented_width_ratio * bottom_ratio
                    * (1.0 - 0.5 * abs(angle_deg) / max_abs_angle),
                    bottom_ratio,
                    (x, y0 + y, box_width, box_height),
                    center_error,
                    math.radians(angle_deg),
                ))
        if not candidates:
            row_occupancies = np.count_nonzero(mask, axis=1) / float(width)
            band = lowest_horizontal_band(
                row_occupancies,
                float(rospy.get_param("~line_min_width_ratio", 0.45)),
                max(2, int(round(float(rospy.get_param(
                    "~line_max_height_ratio", 0.12)) * height))),
            )
            if band is None:
                self.last_stop_line_color = None
                return None, mask, None, None, None
            band_start, band_end = band
            band_mask = mask[band_start:band_end + 1, :]
            ys, xs = np.nonzero(band_mask)
            if len(xs) < 8:
                self.last_stop_line_color = None
                return None, mask, None, None, None
            points = np.column_stack((xs, ys + band_start)).astype(np.float32)
            vx, vy, _fit_x, _fit_y = cv2.fitLine(
                points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            angle_deg = math.degrees(math.atan2(float(vy), float(vx)))
            while angle_deg > 90.0:
                angle_deg -= 180.0
            while angle_deg <= -90.0:
                angle_deg += 180.0
            if abs(angle_deg) > max_abs_angle:
                self.last_stop_line_color = None
                return None, mask, None, None, None
            x_min = int(np.min(xs))
            x_max = int(np.max(xs))
            center_x = float(np.median(xs))
            center_error = (
                center_x - 0.5 * float(width)
            ) / (0.5 * float(width))
            bottom_ratio = float(y0 + band_end + 1) / float(height)
            box = (
                x_min,
                y0 + band_start,
                x_max - x_min + 1,
                band_end - band_start + 1,
            )
            self.last_stop_line_color = "yellow"
            return (
                bottom_ratio,
                mask,
                box,
                center_error,
                math.radians(angle_deg),
            )
        _, bottom_ratio, box, center_error, angle_rad = max(
            candidates, key=lambda item: item[0])
        self.last_stop_line_color = "yellow"
        return bottom_ratio, mask, box, center_error, angle_rad

    def image_callback(self, msg):
        now = time.monotonic()
        with self.lock:
            self.last_image_at = now
            state = self.state
            if state not in ("APPROACH_LINE", "FINAL_VISUAL_APPROACH"):
                return
        final_visual_approach = state == "FINAL_VISUAL_APPROACH"
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            if final_visual_approach:
                self.publish_stop()
                self.publish_status(
                    "final image conversion failed; holding for timeout fallback",
                    visual_error=str(exc),
                )
                return
            self.set_fault("cv_bridge failed: {}".format(exc))
            return
        bottom_ratio, mask, box, center_error, angle_error = \
            self.detect_stop_line(frame)
        if bottom_ratio is None:
            self.band_filter.reset()
            self.final_distance_filter.reset()
            self.final_stop_confirm_hits = 0
            if self.line_missing_since is None:
                self.line_missing_since = now
                with self.lock:
                    pose = self.odom_pose
                self.line_search_origin_yaw = pose[2] if pose else None
                self.line_search_direction = 1.0
                self.line_search_reversals = 0
            if final_visual_approach:
                self.publish_stop()
                self.publish_status(
                    "final yellow stop line missing; holding stop",
                    stop_confirm_hits=0,
                )
            else:
                command = self.missing_line_search_command(now)
                self.cmd_pub.publish(command)
                self.publish_status(
                    "yellow stop line not trusted; bounded yaw reacquisition",
                    commanded_yaw_rps=command.angular.z,
                    search_reversals=self.line_search_reversals,
                )
            return
        self.line_missing_since = None
        self.line_search_origin_yaw = None
        self.line_search_direction = 1.0
        self.line_search_reversals = 0
        distance = self.calibration.distance_for_ratio(bottom_ratio)
        with self.lock:
            self.last_distance_m = distance
            self.last_distance_at = now if distance is not None else 0.0
            self.last_distance_color = (
                self.last_stop_line_color if distance is not None else None)
        if distance is None:
            self.publish_stop()
            self.band_filter.reset()
            self.final_distance_filter.reset()
            self.final_stop_confirm_hits = 0
            if final_visual_approach:
                self.publish_status(
                    "final yellow line outside calibrated range; "
                    "holding for timeout fallback",
                    line_bottom_ratio=bottom_ratio,
                )
                return
            self.publish_status(
                "yellow line outside calibrated range; holding stop",
                line_bottom_ratio=bottom_ratio,
            )
            return
        if final_visual_approach and distance < self.target_min_m:
            self.publish_stop()
            self.band_filter.reset()
            self.final_stop_confirm_hits = 0
            self.publish_status(
                "final yellow line is closer than the visual target; "
                "holding for timeout fallback",
                distance_m=distance,
                minimum_target_m=self.target_min_m,
            )
            return
        precision_mode = distance <= self.precision_start_m
        yaw_tolerance = (
            self.final_yaw_tolerance_rad
            if precision_mode else self.line_yaw_tolerance_rad)
        center_tolerance = (
            self.final_center_tolerance
            if precision_mode else self.line_center_tolerance)
        yaw_limit = float(rospy.get_param(
            "~final_yaw_max_speed", 0.10)) if precision_mode else \
            float(rospy.get_param("~line_yaw_max_speed", 0.16))
        lateral_limit = float(rospy.get_param(
            "~final_lateral_max_speed", 0.03)) if precision_mode else \
            float(rospy.get_param("~line_lateral_max_speed", 0.045))
        alignment_state, lateral_speed, yaw_speed, aligned = \
            line_alignment_command(
                angle_error,
                center_error,
                yaw_tolerance,
                center_tolerance,
                float(rospy.get_param("~line_yaw_kp", 0.8)),
                yaw_limit,
                float(rospy.get_param("~line_yaw_command_sign", -1.0)),
                float(rospy.get_param("~line_lateral_kp", 0.10)),
                lateral_limit,
                float(rospy.get_param(
                    "~line_lateral_command_sign", -1.0)),
                yaw_min=float(rospy.get_param(
                    "~line_yaw_min_speed", 0.04)),
                lateral_min=float(rospy.get_param(
                    "~line_lateral_min_speed", 0.015)),
            )
        command = Twist()
        calibrated_fallback = float(rospy.get_param(
            "~calibrated_final_advance_fallback_sec", 3.0))
        if alignment_state == "yaw":
            command.angular.z = yaw_speed
            self.band_filter.reset()
            self.final_distance_filter.reset()
        elif alignment_state == "lateral":
            command.linear.y = lateral_speed
            self.band_filter.reset()
            self.final_distance_filter.reset()
        else:
            command.linear.x = self.policy.command_for_distance(distance) \
                if final_visual_approach else 0.0
        with self.lock:
            if self.state != state:
                self.publish_stop()
                return
        self.cmd_pub.publish(command)

        if final_visual_approach:
            self.final_distance_filter.reset()
            final_confirmed = aligned and self.band_filter.push(distance)
            self.final_stop_confirm_hits = self.band_filter.hits
            if final_confirmed:
                self.publish_stop()
                with self.lock:
                    if self.state != "FINAL_VISUAL_APPROACH":
                        return
                    self.final_visual_verified = True
                    self.final_stop_distance_m = distance
                    self.final_stop_line_color = "yellow"
                    self.final_stop_source = "yellow_visual"
                    self.state = "FINAL_VISUAL_CONFIRM"
                    self.final_parked_event.set()
                self.publish_status(
                    "final yellow stop-line clearance confirmed",
                    line_bottom_ratio=bottom_ratio,
                    distance_m=distance,
                    line_center_error_ratio=center_error,
                    line_angle_deg=math.degrees(angle_error),
                    line_color="yellow",
                    stop_confirm_hits=self.final_stop_confirm_hits,
                )
            else:
                self.publish_status(
                    "final yellow stop-line precision approach",
                    line_bottom_ratio=bottom_ratio,
                    distance_m=distance,
                    line_center_error_ratio=center_error,
                    line_angle_deg=math.degrees(angle_error),
                    line_color="yellow",
                    yaw_tolerance_deg=math.degrees(yaw_tolerance),
                    center_tolerance_ratio=center_tolerance,
                    stop_confirm_hits=self.final_stop_confirm_hits,
                    alignment_state=alignment_state,
                    commanded_speed_mps=command.linear.x,
                    commanded_lateral_mps=command.linear.y,
                    commanded_yaw_rps=command.angular.z,
                )
            self.publish_stop_line_debug(
                frame, box, distance, angle_error, center_error,
                alignment_state)
            return

        confirmed_distance = None
        if aligned and calibrated_fallback > 0.0:
            self.band_filter.reset()
            confirmed_distance = self.final_distance_filter.push(
                distance, self.last_stop_line_color)
        elif aligned and self.band_filter.push(distance):
            self.final_distance_filter.reset()
            confirmed_distance = distance
        stop_confirm_hits = (
            self.final_distance_filter.hits
            if calibrated_fallback > 0.0 else self.band_filter.hits)
        if confirmed_distance is not None:
            self.publish_stop()
            with self.lock:
                if self.state != "APPROACH_LINE":
                    return
                self.visual_stop_distance_m = confirmed_distance
                self.visual_stop_distance_at = now
                self.visual_stop_line_color = self.last_stop_line_color
                self.state = "VISUAL_CONFIRM"
                self.parked_event.set()
            self.publish_status(
                "stable visual stop-line distance confirmed; "
                "final odometry advance armed",
                line_bottom_ratio=bottom_ratio,
                distance_m=confirmed_distance,
                line_center_error_ratio=center_error,
                line_angle_deg=math.degrees(angle_error),
                line_color=self.last_stop_line_color,
                final_advance_limit_m=self.final_advance_m,
                stop_confirm_hits=stop_confirm_hits,
            )
        else:
            self.publish_status(
                "precision line approach" if precision_mode
                else "closed-loop line approach",
                line_bottom_ratio=bottom_ratio,
                line_center_error_ratio=center_error,
                line_angle_deg=math.degrees(angle_error),
                line_color=self.last_stop_line_color,
                precision_mode=precision_mode,
                yaw_tolerance_deg=math.degrees(yaw_tolerance),
                center_tolerance_ratio=center_tolerance,
                stop_confirm_hits=stop_confirm_hits,
                alignment_state=alignment_state,
                commanded_speed_mps=command.linear.x,
                commanded_lateral_mps=command.linear.y,
                commanded_yaw_rps=command.angular.z,
            )
        self.publish_stop_line_debug(
            frame, box, distance, angle_error, center_error,
            alignment_state)

    def publish_stop_line_debug(
            self, frame, box, distance, angle_error, center_error,
            alignment_state):
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
            cv2.putText(
                frame,
                "angle={:+.1f} center={:+.2f} {}".format(
                    math.degrees(angle_error), center_error, alignment_state),
                (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 255), 2,
            )
            self.debug_pub.publish(
                self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))

    @staticmethod
    def normalized_angle(angle):
        return math.atan2(math.sin(float(angle)), math.cos(float(angle)))

    def missing_line_search_command(self, now):
        """Sweep in place within a small yaw window; never advance blind."""
        command = Twist()
        delay = max(0.0, float(rospy.get_param(
            "~line_search_delay_sec", 0.40)))
        if self.line_missing_since is None or now - self.line_missing_since < delay:
            return command
        with self.lock:
            pose = self.odom_pose
            odom_age = now - self.odom_received_at
        stale = float(rospy.get_param("~line_search_odom_stale_sec", 0.50))
        if pose is None or odom_age > stale:
            return command
        if self.line_search_origin_yaw is None:
            self.line_search_origin_yaw = pose[2]
        delta = self.normalized_angle(pose[2] - self.line_search_origin_yaw)
        limit = math.radians(max(1.0, float(rospy.get_param(
            "~line_search_yaw_limit_deg", 8.0))))
        if self.line_search_direction > 0.0 and delta >= limit:
            self.line_search_direction = -1.0
            self.line_search_reversals += 1
        elif self.line_search_direction < 0.0 and delta <= -limit:
            self.line_search_direction = 1.0
            self.line_search_reversals += 1
        speed = max(0.01, float(rospy.get_param(
            "~line_search_yaw_speed", 0.10)))
        command.angular.z = self.line_search_direction * speed
        return command

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
        if state in (
                "VISUAL_CONFIRM", "FINAL_VISUAL_CONFIRM", "STOP_CONFIRM",
                "WAIT_TRAFFIC", "FAULT"):
            self.publish_stop()
        if state not in ("APPROACH_LINE", "FINAL_VISUAL_APPROACH"):
            return
        now = time.monotonic()
        stale_stop_sec = float(rospy.get_param("~image_stale_stop_sec", 0.25))
        stale_fault_sec = float(rospy.get_param("~image_stale_fault_sec", 1.0))
        if last_image_at <= 0.0 or now - last_image_at >= stale_stop_sec:
            self.publish_stop()
        if (state == "APPROACH_LINE" and last_image_at > 0.0 and
                now - last_image_at >= stale_fault_sec):
            self.set_fault("camera image timeout")
        missing_timeout = float(rospy.get_param(
            "~final_visual_line_missing_fault_sec", 5.0)) \
            if state == "FINAL_VISUAL_APPROACH" else float(rospy.get_param(
                "~line_missing_fault_sec", 2.0))
        if (state == "APPROACH_LINE" and self.line_missing_since is not None
                and now - self.line_missing_since >= missing_timeout):
            self.set_fault(
                "yellow stop line lost during {}".format(
                    "final visual approach"
                    if state == "FINAL_VISUAL_APPROACH" else "approach"))

    def navigate_to_staging_pose(self):
        if not bool(rospy.get_param("~traffic_pose_configured", False)):
            raise RuntimeError(
                "traffic_pose_configured is false; set staging coordinates")
        timeout = float(rospy.get_param("~navigation_timeout_sec", 120.0))
        if not self.move_base.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError("move_base action server unavailable")
        planner_client, saved_tolerances = self.tighten_staging_tolerances()
        try:
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
                raise RuntimeError(
                    "navigation to stop-line staging pose timed out")
            if self.move_base.get_state() != 3:
                raise RuntimeError(
                    "navigation failed with action state {}".format(
                        self.move_base.get_state()))
        finally:
            self.restore_staging_tolerances(
                planner_client, saved_tolerances)

    @staticmethod
    def restore_staging_tolerances(planner_client, saved_tolerances):
        if planner_client is None or saved_tolerances is None:
            return
        try:
            restored = planner_client.update_configuration(saved_tolerances)
            rospy.loginfo(
                "restored TEB goal tolerances: xy=%.3f yaw=%.3f",
                float(restored.get(
                    "xy_goal_tolerance",
                    saved_tolerances["xy_goal_tolerance"])),
                float(restored.get(
                    "yaw_goal_tolerance",
                    saved_tolerances["yaw_goal_tolerance"])),
            )
        except Exception as exc:
            rospy.logerr("failed to restore TEB goal tolerances: %s", exc)

    @staticmethod
    def tighten_staging_tolerances():
        if not bool(rospy.get_param(
                "~tighten_staging_goal_tolerance", True)):
            return None, None
        namespace = str(rospy.get_param(
            "~staging_planner_reconfigure_ns",
            "/move_base/TebLocalPlannerROS"))
        xy_tolerance = float(rospy.get_param(
            "~staging_xy_goal_tolerance", 0.04))
        yaw_tolerance = float(rospy.get_param(
            "~staging_yaw_goal_tolerance", 0.08))
        if not 0.01 <= xy_tolerance <= 0.15:
            raise RuntimeError(
                "staging_xy_goal_tolerance must be within [0.01, 0.15]")
        if not 0.02 <= yaw_tolerance <= 0.20:
            raise RuntimeError(
                "staging_yaw_goal_tolerance must be within [0.02, 0.20]")
        try:
            planner_client = dynamic_reconfigure.client.Client(
                namespace, timeout=5.0)
            current = planner_client.get_configuration(timeout=3.0)
            saved_tolerances = {
                "xy_goal_tolerance": current.get(
                    "xy_goal_tolerance", 0.15),
                "yaw_goal_tolerance": current.get(
                    "yaw_goal_tolerance", 0.10),
                "free_goal_vel": current.get("free_goal_vel", False),
            }
            updated = planner_client.update_configuration({
                "xy_goal_tolerance": xy_tolerance,
                "yaw_goal_tolerance": yaw_tolerance,
                "free_goal_vel": False,
            })
            rospy.logwarn(
                "TASK4_TOLERANCE_GUARD applied: xy=%.3f yaw=%.3f",
                float(updated.get("xy_goal_tolerance", xy_tolerance)),
                float(updated.get("yaw_goal_tolerance", yaw_tolerance)),
            )
            return planner_client, saved_tolerances
        except Exception as exc:
            raise RuntimeError(
                "unable to tighten task4 TEB goal tolerances: {}".format(
                    exc))

    def align_to_staging_heading(self):
        target_yaw = float(rospy.get_param("~traffic_staging_yaw"))
        target_frame = str(rospy.get_param("~traffic_frame", "map"))
        base_frame = str(rospy.get_param(
            "~staging_heading_base_frame", "base_link"))
        tolerance = math.radians(float(rospy.get_param(
            "~staging_heading_tolerance_deg", 6.0)))
        timeout = max(1.0, float(rospy.get_param(
            "~staging_heading_timeout_sec", 20.0)))
        kp = float(rospy.get_param("~staging_heading_kp", 0.9))
        min_speed = float(rospy.get_param(
            "~staging_heading_min_speed", 0.20))
        max_speed = float(rospy.get_param(
            "~staging_heading_max_speed", 0.30))
        deadline = time.monotonic() + timeout
        rate = rospy.Rate(30)
        last_error = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    base_frame,
                    rospy.Time(0),
                    rospy.Duration(0.10),
                )
            except tf2_ros.TransformException:
                self.publish_stop()
                rate.sleep()
                continue
            orientation = transform.transform.rotation
            current_yaw = math.atan2(
                2.0 * (orientation.w * orientation.z
                       + orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y * orientation.y
                             + orientation.z * orientation.z),
            )
            error = self.normalized_angle(target_yaw - current_yaw)
            last_error = error
            angular = heading_alignment_command(
                error, tolerance, kp, min_speed, max_speed)
            if angular == 0.0:
                self.publish_stop()
                self.publish_status(
                    "staging heading aligned",
                    heading_error_deg=math.degrees(error),
                )
                return
            command = Twist()
            command.angular.z = angular
            self.cmd_pub.publish(command)
            self.publish_status(
                "aligning staging heading before line search",
                heading_error_deg=math.degrees(error),
                commanded_yaw_rps=angular,
            )
            rate.sleep()
        self.publish_stop()
        if bool(rospy.get_param(
                "~staging_heading_fallback_to_vision", True)):
            error_text = (
                "unknown" if last_error is None
                else "{:.2f}deg".format(math.degrees(last_error)))
            rospy.logwarn(
                "staging heading alignment timed out at %s; vehicle stopped, "
                "continuing with visual stop-line alignment",
                error_text,
            )
            self.publish_status(
                "staging heading incomplete; visual alignment taking over",
                heading_error_deg=(
                    None if last_error is None
                    else math.degrees(last_error)),
            )
            return
        raise RuntimeError("staging heading alignment timed out")

    def plan_final_advance(self):
        now = time.monotonic()
        with self.lock:
            measured_distance = self.visual_stop_distance_m
            measured_at = self.visual_stop_distance_at
            measured_color = self.visual_stop_line_color
            candidate_distance = self.last_distance_m
            candidate_at = self.last_distance_at
            candidate_color = self.last_distance_color
        measurement_age = (
            None if measured_distance is None or measured_at <= 0.0
            else max(0.0, now - measured_at))
        candidate_age = (
            None if candidate_distance is None or candidate_at <= 0.0
            else max(0.0, now - candidate_at))
        distance, source = select_final_advance(
            measured_distance,
            measurement_age,
            self.final_target_clearance_m,
            self.final_advance_m,
            self.final_no_vision_fallback_m,
            self.final_visual_max_age_sec,
            self.final_minimum_command_m,
            self.final_visual_bias_m,
        )
        with self.lock:
            self.planned_final_advance_m = distance
            self.final_advance_source = source
        rospy.logwarn(
            "TASK4_FINAL_ADVANCE planned=%.3fm source=%s "
            "confirmed=%s confirmed_color=%s age=%s "
            "candidate=%s candidate_color=%s candidate_age=%s "
            "target=%.3fm visual_bias=%.3fm",
            distance,
            source,
            "none" if measured_distance is None
            else "{:.3f}m".format(measured_distance),
            measured_color or "none",
            "none" if measurement_age is None
            else "{:.3f}s".format(measurement_age),
            "none" if candidate_distance is None
            else "{:.3f}m".format(candidate_distance),
            candidate_color or "none",
            "none" if candidate_age is None
            else "{:.3f}s".format(candidate_age),
            self.final_target_clearance_m,
            self.final_visual_bias_m,
        )
        return distance

    def advance_final_offset(self):
        distance = self.planned_final_advance_m
        with self.lock:
            self.final_progress_m = 0.0
            self.final_lateral_drift_m = 0.0
            self.final_yaw_drift_deg = 0.0
        if distance <= 0.0:
            self.publish_stop()
            self.publish_status(
                "final advance not required; target clearance already met",
                final_advance_m=distance,
                final_progress_m=0.0,
            )
            return
        stale_limit = max(0.05, float(rospy.get_param(
            "~final_advance_odom_stale_sec", 0.30)))
        wait_deadline = time.monotonic() + max(0.5, float(rospy.get_param(
            "~final_advance_odom_wait_sec", 2.0)))
        start_pose = None
        while not rospy.is_shutdown() and time.monotonic() < wait_deadline:
            now = time.monotonic()
            with self.lock:
                pose = self.odom_pose
                age = now - self.odom_received_at
            if pose is not None and age <= stale_limit:
                start_pose = pose
                break
            self.publish_stop()
            time.sleep(0.02)
        if start_pose is None:
            raise RuntimeError("fresh odometry unavailable for final advance")

        speed = max(0.005, float(rospy.get_param(
            "~final_advance_speed_mps", 0.045)))
        creep_speed = min(speed, max(0.005, float(rospy.get_param(
            "~final_advance_creep_speed_mps", 0.030))))
        creep_distance = max(0.005, float(rospy.get_param(
            "~final_advance_creep_distance_m", 0.03)))
        max_yaw_drift = math.radians(max(1.0, float(rospy.get_param(
            "~final_advance_max_yaw_drift_deg", 4.0))))
        max_lateral_drift = max(0.005, float(rospy.get_param(
            "~final_advance_max_lateral_drift_m", 0.025)))
        timeout = max(1.0, float(rospy.get_param(
            "~final_advance_timeout_sec", 10.0)))
        deadline = time.monotonic() + timeout
        rate = rospy.Rate(30)

        while not rospy.is_shutdown() and time.monotonic() < deadline:
            now = time.monotonic()
            with self.lock:
                pose = self.odom_pose
                age = now - self.odom_received_at
            if pose is None or age > stale_limit:
                self.publish_stop()
                raise RuntimeError("odometry became stale during final advance")
            progress = forward_progress(start_pose, pose)
            lateral_drift = lateral_displacement(start_pose, pose)
            yaw_drift = abs(self.normalized_angle(pose[2] - start_pose[2]))
            with self.lock:
                self.final_progress_m = progress
                self.final_lateral_drift_m = lateral_drift
                self.final_yaw_drift_deg = math.degrees(yaw_drift)
            if yaw_drift > max_yaw_drift:
                self.publish_stop()
                raise RuntimeError("heading drift exceeded final advance limit")
            if abs(lateral_drift) > max_lateral_drift:
                self.publish_stop()
                raise RuntimeError("lateral drift exceeded final advance limit")
            if progress < -0.01:
                self.publish_stop()
                raise RuntimeError("vehicle moved backward during final advance")
            remaining = distance - progress
            if remaining <= 0.002:
                self.publish_stop()
                self.publish_status(
                    "final odometry advance completed",
                    final_advance_m=distance,
                    final_progress_m=progress,
                    final_remaining_m=max(0.0, remaining),
                )
                return
            command = Twist()
            command.linear.x = (
                creep_speed if remaining <= creep_distance else speed)
            self.cmd_pub.publish(command)
            self.publish_status(
                "guarded final advance toward stop line",
                final_advance_m=distance,
                final_progress_m=progress,
                final_remaining_m=remaining,
                commanded_speed_mps=command.linear.x,
                lateral_drift_m=lateral_drift,
                yaw_drift_deg=math.degrees(yaw_drift),
            )
            rate.sleep()
        self.publish_stop()
        raise RuntimeError("final odometry advance timed out")

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
                self.state = "ALIGN_STAGING_HEADING"
            self.publish_status("correcting staging heading")
            self.align_to_staging_heading()
            with self.lock:
                self.state = "APPROACH_LINE"
                self.last_image_at = time.monotonic()
                self.last_distance_m = None
                self.last_distance_at = 0.0
                self.last_distance_color = None
                self.visual_stop_distance_m = None
                self.visual_stop_distance_at = 0.0
                self.visual_stop_line_color = None
                self.planned_final_advance_m = 0.0
                self.final_advance_source = "unplanned"
                self.final_visual_verified = False
                self.final_stop_distance_m = None
                self.final_stop_line_color = None
                self.final_stop_confirm_hits = 0
                self.final_stop_source = "unconfirmed"
                self.parked_event.clear()
                self.final_parked_event.clear()
                self.final_distance_filter.reset()
                self.band_filter.reset()
                self.line_missing_since = None
                self.line_search_origin_yaw = (
                    self.odom_pose[2] if self.odom_pose else None)
                self.line_search_direction = 1.0
                self.line_search_reversals = 0
            self.publish_status("visual stop-line approach armed")
            fallback_timeout = max(0.0, float(rospy.get_param(
                "~calibrated_final_advance_fallback_sec", 3.0)))
            try:
                self.wait_event(
                    self.parked_event,
                    fallback_timeout or float(rospy.get_param(
                        "~line_approach_timeout_sec", 75.0)),
                    "strict line approach",
                )
            except RuntimeError:
                with self.lock:
                    faulted = self.state == "FAULT"
                if faulted or fallback_timeout <= 0.0:
                    raise
                self.publish_stop()
                rospy.logwarn(
                    "visual stop-line confirmation did not finish within %.2fs; "
                    "using calibrated guarded final advance",
                    fallback_timeout,
                )
                self.publish_status(
                    "visual alignment window complete; calibrated final "
                    "advance armed",
                    visual_candidate_distance_m=self.last_distance_m,
                    visual_candidate_color=self.last_distance_color,
                    visual_confirmed_distance_m=self.visual_stop_distance_m,
                    visual_confirmed_color=self.visual_stop_line_color,
                    fallback_timeout_sec=fallback_timeout,
                )
            planned_advance = self.plan_final_advance()
            with self.lock:
                self.state = "FINAL_ADVANCE"
            self.publish_status(
                "starting guarded final stop-line advance",
                final_advance_m=planned_advance,
                final_advance_source=self.final_advance_source,
            )
            self.advance_final_offset()
            self.publish_stop()
            with self.lock:
                self.state = "FINAL_VISUAL_APPROACH"
                self.last_image_at = time.monotonic()
                self.last_distance_m = None
                self.last_distance_at = 0.0
                self.last_distance_color = None
                self.final_visual_verified = False
                self.final_stop_distance_m = None
                self.final_stop_line_color = None
                self.final_stop_confirm_hits = 0
                self.final_stop_source = "unconfirmed"
                self.final_parked_event.clear()
                self.band_filter.reset()
                self.final_distance_filter.reset()
                self.line_missing_since = None
                self.line_search_origin_yaw = None
                self.line_search_direction = 1.0
                self.line_search_reversals = 0
            self.publish_status(
                "hard advance complete; fresh yellow-line precision stop armed")
            final_visual_timeout = float(rospy.get_param(
                "~final_visual_approach_timeout_sec", 3.0))
            try:
                self.wait_event(
                    self.final_parked_event,
                    final_visual_timeout,
                    "final yellow stop-line approach",
                )
            except RuntimeError:
                with self.lock:
                    if self.state == "FAULT":
                        raise
                    self.state = "FINAL_VISUAL_TIMEOUT"
                    self.final_visual_verified = False
                    self.final_stop_distance_m = self.last_distance_m
                    self.final_stop_line_color = self.last_distance_color
                    self.final_stop_source = "hard_advance_timeout"
                self.publish_stop()
                rospy.logwarn(
                    "final yellow-line confirmation timed out after %.2fs; "
                    "accepting the completed guarded hard advance",
                    final_visual_timeout,
                )
                self.publish_status(
                    "final visual timeout; completed hard advance accepted",
                    fallback_timeout_sec=final_visual_timeout,
                )
            with self.lock:
                visual_valid = (
                    self.final_visual_verified
                    and self.final_stop_source == "yellow_visual"
                    and self.final_stop_line_color == "yellow"
                    and self.final_stop_distance_m is not None
                    and self.policy.in_target_band(
                        self.final_stop_distance_m)
                )
                hard_advance_fallback = (
                    not self.final_visual_verified
                    and self.final_stop_source == "hard_advance_timeout"
                )
                if not (visual_valid or hard_advance_fallback):
                    raise RuntimeError(
                        "final stop completion source is invalid")
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
