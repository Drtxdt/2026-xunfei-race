#!/usr/bin/env python3
"""
直行+右转巡线节点
路径：第一个岔路口直行，第二个岔路口右转，末端停车
"""
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


@dataclass
class Segment:
    left: int
    right: int
    center: float
    width: int


@dataclass
class RowObservation:
    y: int
    segments: List[Segment]
    left_x: Optional[float]
    right_x: Optional[float]
    center_x: Optional[float]
    multi_candidate: bool


@dataclass
class FinishDetectionResult:
    detected: bool
    candidate_box: Optional[Tuple[int, int, int, int]]
    horizontal_width_ratio: float
    vertical_left_height_ratio: float
    vertical_right_height_ratio: float
    inner_fill_ratio: float
    inner_component_count: int


class PidController:
    def __init__(self, kp: float, ki: float, kd: float, max_integral: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = abs(max_integral)
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def update(self, error: float, now: float) -> float:
        if self.last_time is None:
            dt = 0.0
        else:
            dt = max(now - self.last_time, 1e-3)

        if dt > 0.0:
            self.integral += error * dt
            self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0

        self.last_error = error
        self.last_time = now
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class LineFollowStraightRightNode:
    def __init__(self):
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("~image_topic", rospy.get_param("image_topic", "/usb_cam/image_raw"))
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", rospy.get_param("cmd_vel_topic", "/cmd_vel"))
        self.odom_topic = rospy.get_param("~odom_topic", rospy.get_param("odom_topic", "/odom"))
        self.status_topic = rospy.get_param("~status_topic", rospy.get_param("status_topic", "/line_follow/status"))
        self.debug_image_topic = rospy.get_param(
            "~debug_image_topic", rospy.get_param("debug_image_topic", "/line_follow/debug_image")
        )
        self.start_topic = rospy.get_param("~start_topic", rospy.get_param("start_topic", "/line_follow/start"))

        self.auto_start = bool(rospy.get_param("~auto_start", rospy.get_param("auto_start", True)))
        self.started = self.auto_start
        self.publish_debug = bool(rospy.get_param("~publish_debug", rospy.get_param("publish_debug", True)))

        # 视觉参数
        self.lane_width_px_init = float(rospy.get_param("~lane_width_px_init", rospy.get_param("lane_width_px_init", 230)))
        self.lane_width_px_min = float(rospy.get_param("~lane_width_px_min", rospy.get_param("lane_width_px_min", 150)))
        self.lane_width_px_max = float(rospy.get_param("~lane_width_px_max", rospy.get_param("lane_width_px_max", 320)))
        self.lane_width_adapt_alpha = float(rospy.get_param("~lane_width_adapt_alpha", rospy.get_param("lane_width_adapt_alpha", 0.2)))
        self.enable_lane_width_adapt = bool(rospy.get_param("~enable_lane_width_adapt", rospy.get_param("enable_lane_width_adapt", False)))
        self.estimated_lane_width_px = float(self.lane_width_px_init)
        self.single_line_hold_frames = int(rospy.get_param("~single_line_hold_frames", rospy.get_param("single_line_hold_frames", 12)))

        self.roi_y_start_ratio = float(rospy.get_param("~roi_y_start_ratio", rospy.get_param("roi_y_start_ratio", 0.45)))
        self.roi_y_end_ratio = float(rospy.get_param("~roi_y_end_ratio", rospy.get_param("roi_y_end_ratio", 1.0)))
        self.white_s_max = int(rospy.get_param("~white_s_max", rospy.get_param("white_s_max", 85)))
        self.white_v_min = int(rospy.get_param("~white_v_min", rospy.get_param("white_v_min", 150)))
        self.gray_white_threshold = int(rospy.get_param("~gray_white_threshold", rospy.get_param("gray_white_threshold", 185)))
        self.morph_kernel_size = int(rospy.get_param("~morph_kernel_size", rospy.get_param("morph_kernel_size", 5)))
        self.min_line_width_px = int(rospy.get_param("~min_line_width_px", rospy.get_param("min_line_width_px", 6)))
        self.min_segment_gap_px = int(rospy.get_param("~min_segment_gap_px", rospy.get_param("min_segment_gap_px", 12)))
        self.min_contour_area = float(rospy.get_param("~min_contour_area", rospy.get_param("min_contour_area", 60.0)))
        self.scan_row_ratios = self._get_float_list("scan_row_ratios", [0.20, 0.35, 0.50, 0.65, 0.80, 0.92])
        self.target_row_weight_bottom = float(rospy.get_param("~target_row_weight_bottom", rospy.get_param("target_row_weight_bottom", 1.5)))

        # PID
        kp = float(rospy.get_param("~kp", rospy.get_param("kp", 0.0045)))
        ki = float(rospy.get_param("~ki", rospy.get_param("ki", 0.0)))
        kd = float(rospy.get_param("~kd", rospy.get_param("kd", 0.0015)))
        max_integral = float(rospy.get_param("~max_integral", rospy.get_param("max_integral", 80.0)))
        self.pid = PidController(kp, ki, kd, max_integral)

        # 运动控制
        self.base_linear_speed = float(rospy.get_param("~base_linear_speed", rospy.get_param("base_linear_speed", 0.16)))
        self.min_linear_speed = float(rospy.get_param("~min_linear_speed", rospy.get_param("min_linear_speed", 0.06)))
        self.search_linear_speed = float(rospy.get_param("~search_linear_speed", rospy.get_param("search_linear_speed", 0.035)))
        self.max_angular_speed = float(rospy.get_param("~max_angular_speed", rospy.get_param("max_angular_speed", 0.8)))
        self.error_slowdown_px = float(rospy.get_param("~error_slowdown_px", rospy.get_param("error_slowdown_px", 160.0)))

        self.search_angular_speed = float(rospy.get_param("~search_angular_speed", rospy.get_param("search_angular_speed", 0.25)))
        self.lost_timeout = float(rospy.get_param("~lost_timeout", rospy.get_param("lost_timeout", 1.0)))
        self.stop_on_lost = bool(rospy.get_param("~stop_on_lost", rospy.get_param("stop_on_lost", False)))

        # 岔路参数
        self.fork_candidate_count = int(rospy.get_param("~fork_candidate_count", rospy.get_param("fork_candidate_count", 3)))
        self.fork_center_tolerance_px = float(rospy.get_param("~fork_center_tolerance_px", rospy.get_param("fork_center_tolerance_px", 180.0)))
        self.fork_cooldown_sec = float(rospy.get_param("~fork_cooldown_sec", rospy.get_param("fork_cooldown_sec", 1.0)))
        self.fork_latch_time = float(rospy.get_param("~fork_latch_time", rospy.get_param("fork_latch_time", 0.35)))
        self.turn_bias_px = float(rospy.get_param("~turn_bias_px", rospy.get_param("turn_bias_px", 55.0)))   # 右转正偏移
        self.turn_hold_time = float(rospy.get_param("~turn_hold_time", rospy.get_param("turn_hold_time", 1.2)))
        self.turn_linear_speed = float(rospy.get_param("~turn_linear_speed", rospy.get_param("turn_linear_speed", 0.08)))

        # 直行+右转状态机
        self.fork_handled_count = 0
        self.right_turn_active = False
        self.fork_cool_until = 0.0

        # 停车相关
        self.finish_enable_delay = float(rospy.get_param("~finish_enable_delay", rospy.get_param("finish_enable_delay", 6.0)))
        self.finish_confirm_frames = int(rospy.get_param("~finish_confirm_frames", rospy.get_param("finish_confirm_frames", 5)))
        self.finish_release_frames = int(rospy.get_param("~finish_release_frames", rospy.get_param("finish_release_frames", 1)))
        self.finish_stop_time = float(rospy.get_param("~finish_stop_time", rospy.get_param("finish_stop_time", 1.0)))
        self.finish_auto_stop = bool(rospy.get_param("~finish_auto_stop", rospy.get_param("finish_auto_stop", True)))
        self.finish_use_odom_approach = bool(rospy.get_param("~finish_use_odom_approach", rospy.get_param("finish_use_odom_approach", True)))
        self.finish_odom_approach_distance_m = float(rospy.get_param("~finish_odom_approach_distance_m", rospy.get_param("finish_odom_approach_distance_m", 0.50)))
        self.finish_odom_approach_speed = abs(float(rospy.get_param("~finish_odom_approach_speed", rospy.get_param("finish_odom_approach_speed", 0.05))))
        self.finish_odom_min_trigger_frames = int(rospy.get_param("~finish_odom_min_trigger_frames", rospy.get_param("finish_odom_min_trigger_frames", 2)))
        self.finish_odom_timeout_sec = float(rospy.get_param("~finish_odom_timeout_sec", rospy.get_param("finish_odom_timeout_sec", 8.0)))
        self.finish_parking_target_bottom_y_ratio = float(rospy.get_param("~finish_parking_target_bottom_y_ratio", rospy.get_param("finish_parking_target_bottom_y_ratio", 0.955)))
        self.finish_parking_slow_bottom_y_ratio = float(rospy.get_param("~finish_parking_slow_bottom_y_ratio", rospy.get_param("finish_parking_slow_bottom_y_ratio", 0.90)))
        self.finish_parking_confirm_frames = int(rospy.get_param("~finish_parking_confirm_frames", rospy.get_param("finish_parking_confirm_frames", 2)))
        self.finish_parking_min_horizontal_width_ratio = float(rospy.get_param("~finish_parking_min_horizontal_width_ratio", rospy.get_param("finish_parking_min_horizontal_width_ratio", 0.70)))
        self.finish_parking_min_vertical_side_height_ratio = float(rospy.get_param("~finish_parking_min_vertical_side_height_ratio", rospy.get_param("finish_parking_min_vertical_side_height_ratio", 0.30)))
        self.finish_parking_min_box_width_ratio = float(rospy.get_param("~finish_parking_min_box_width_ratio", rospy.get_param("finish_parking_min_box_width_ratio", 0.70)))
        self.finish_parking_min_box_height_ratio = float(rospy.get_param("~finish_parking_min_box_height_ratio", rospy.get_param("finish_parking_min_box_height_ratio", 0.09)))
        self.finish_bottom_ratio = float(rospy.get_param("~finish_bottom_ratio", rospy.get_param("finish_bottom_ratio", 0.72)))
        self.finish_horizontal_min_width_ratio = float(rospy.get_param("~finish_horizontal_min_width_ratio", rospy.get_param("finish_horizontal_min_width_ratio", 0.45)))
        self.finish_horizontal_min_rows = int(rospy.get_param("~finish_horizontal_min_rows", rospy.get_param("finish_horizontal_min_rows", 4)))
        self.finish_vertical_side_min_height_ratio = float(rospy.get_param("~finish_vertical_side_min_height_ratio", rospy.get_param("finish_vertical_side_min_height_ratio", 0.18)))
        self.finish_box_min_fill_ratio = float(rospy.get_param("~finish_box_min_fill_ratio", rospy.get_param("finish_box_min_fill_ratio", 0.03)))
        self.finish_box_max_components = int(rospy.get_param("~finish_box_max_components", rospy.get_param("finish_box_max_components", 4)))
        self.finish_box_min_area_ratio = float(rospy.get_param("~finish_box_min_area_ratio", rospy.get_param("finish_box_min_area_ratio", 0.06)))
        self.finish_box_bottom_touch_ratio = float(rospy.get_param("~finish_box_bottom_touch_ratio", rospy.get_param("finish_box_bottom_touch_ratio", 0.92)))
        self.finish_box_min_height_ratio = float(rospy.get_param("~finish_box_min_height_ratio", rospy.get_param("finish_box_min_height_ratio", 0.22)))
        self.finish_box_center_tolerance_ratio = float(rospy.get_param("~finish_box_center_tolerance_ratio", rospy.get_param("finish_box_center_tolerance_ratio", 0.28)))
        self.finish_approach_center_alpha = float(rospy.get_param("~finish_approach_center_alpha", rospy.get_param("finish_approach_center_alpha", 0.75)))
        self.finish_approach_max_angular_speed = float(rospy.get_param("~finish_approach_max_angular_speed", rospy.get_param("finish_approach_max_angular_speed", 0.45)))
        self.finish_approach_linear_speed_scale = float(rospy.get_param("~finish_approach_linear_speed_scale", rospy.get_param("finish_approach_linear_speed_scale", 0.78)))
        self.finish_final_approach_frames = int(rospy.get_param("~finish_final_approach_frames", rospy.get_param("finish_final_approach_frames", 2)))
        self.finish_final_linear_speed = float(rospy.get_param("~finish_final_linear_speed", rospy.get_param("finish_final_linear_speed", 0.03)))
        self.finish_center_jump_reject_px = float(rospy.get_param("~finish_center_jump_reject_px", rospy.get_param("finish_center_jump_reject_px", 90.0)))
        self.finish_profile = rospy.get_param("~finish_profile", rospy.get_param("finish_profile", "default"))
        self._load_finish_profile_overrides()

        # ROS 接口
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)

        # 状态变量
        self.status = "idle" if not self.started else "searching"
        self.last_detection_time = None
        self.last_lane_center = None
        self.last_error_px = 0.0
        self.single_line_frames = 0
        self.turn_until = 0.0
        self.last_fork_time = -1e9
        self.fork_latch_until = 0.0
        self.finish_frames = 0
        self.finish_lost_frames = 0
        self.finish_time = None
        self.finish_parking_candidate_frames = 0
        self.finish_parking_reached_frames = 0
        self.finish_parking_bottom_y_ratio = 0.0
        self.current_odom_xy: Optional[Tuple[float, float]] = None
        self.last_odom_time = None
        self.finish_odom_active = False
        self.finish_odom_start_xy: Optional[Tuple[float, float]] = None
        self.finish_odom_start_time = None
        self.finish_odom_distance_m = 0.0
        self.finish_phase = "search"
        self.last_finish_result: Optional[FinishDetectionResult] = None
        self.last_debug_snapshot: Optional[Dict] = None
        self.start_time = time.time()
        self.finish_detection_enabled = False
        self.dual_line_stable_frames = 0
        self.nonfork_stable_frames = 0

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.start_sub = rospy.Subscriber(self.start_topic, Bool, self.start_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10)

        rospy.on_shutdown(self.stop_robot)
        self.publish_status(force=True)
        rospy.loginfo("Straight+Right node started. image=%s cmd_vel=%s", self.image_topic, self.cmd_vel_topic)

    # ---------- 工具函数 ----------
    def _get_float_list(self, name, default):
        value = rospy.get_param("~" + name, rospy.get_param(name, list(default)))
        return [float(item) for item in value]

    def _load_finish_profile_overrides(self):
        profiles = rospy.get_param("~finish_profiles", rospy.get_param("finish_profiles", {}))
        if not isinstance(profiles, dict):
            return
        cfg = profiles.get(self.finish_profile, {})
        if not isinstance(cfg, dict):
            return
        self.finish_confirm_frames = int(cfg.get("finish_confirm_frames", self.finish_confirm_frames))
        self.finish_horizontal_min_width_ratio = float(cfg.get("finish_horizontal_min_width_ratio", self.finish_horizontal_min_width_ratio))
        self.finish_vertical_side_min_height_ratio = float(cfg.get("finish_vertical_side_min_height_ratio", self.finish_vertical_side_min_height_ratio))

    def start_callback(self, msg: Bool):
        self.started = bool(msg.data)
        if self.started and self.status in ("idle", "finish"):
            self.start_time = time.time()
            self.finish_detection_enabled = False
            self.finish_frames = 0
            self.finish_lost_frames = 0
            self.finish_time = None
            self.finish_parking_candidate_frames = 0
            self.finish_parking_reached_frames = 0
            self.finish_parking_bottom_y_ratio = 0.0
            self.reset_finish_odom_approach()
            self.finish_phase = "search"
            self.fork_handled_count = 0
            self.right_turn_active = False
            self.fork_cool_until = 0.0
            self.turn_until = 0.0
            self.dual_line_stable_frames = 0
            self.nonfork_stable_frames = 0
            self.set_status("searching")
        if not self.started:
            self.pid.reset()
            self.finish_time = None
            self.finish_parking_candidate_frames = 0
            self.finish_parking_reached_frames = 0
            self.finish_parking_bottom_y_ratio = 0.0
            self.reset_finish_odom_approach()
            self.finish_phase = "search"
            self.stop_robot()
            self.set_status("idle")

    def odom_callback(self, msg: Odometry):
        position = msg.pose.pose.position
        self.current_odom_xy = (float(position.x), float(position.y))
        self.last_odom_time = time.time()

    def reset_finish_odom_approach(self):
        self.finish_odom_active = False
        self.finish_odom_start_xy = None
        self.finish_odom_start_time = None
        self.finish_odom_distance_m = 0.0

    # ---------- 主回调 ----------
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "cv_bridge conversion failed: %s", exc)
            return

        now = time.time()
        if self.finish_time is None and not self.finish_detection_enabled:
            self.finish_detection_enabled = (now - self.start_time) >= self.finish_enable_delay
        if self.finish_time is not None:
            self.handle_finish_maneuver(now)
            return

        mask, roi_origin_y = self.extract_white_mask(frame)
        observations = self.observe_lane(mask, frame.shape[1], self.right_turn_active)
        lane_center = self.estimate_lane_center(observations, frame.shape[1])
        self.update_lane_width_estimate(observations)

        fork_rows = sum(1 for obs in observations if obs.multi_candidate)
        if fork_rows > 0:
            self.nonfork_stable_frames = 0
        else:
            self.nonfork_stable_frames += 1

        image_center = frame.shape[1] / 2.0
        lane_center_offset = None if lane_center is None else abs(lane_center - image_center)
        fork_geometry_ok = lane_center_offset is not None and lane_center_offset <= self.fork_center_tolerance_px
        fork_detected = fork_rows >= self.fork_candidate_count and fork_geometry_ok
        if fork_detected:
            self.fork_latch_until = max(self.fork_latch_until, now + self.fork_latch_time)
        fork_detected_latched = now < self.fork_latch_until

        # 两岔路状态机
        if fork_detected_latched and now > self.fork_cool_until:
            if self.fork_handled_count == 0:
                self.fork_handled_count = 1
                self.fork_cool_until = now + self.fork_cooldown_sec
                rospy.loginfo("First fork passed (straight)")
            elif self.fork_handled_count == 1:
                self.right_turn_active = True
                self.turn_until = max(self.turn_until, now + self.turn_hold_time)
                self.last_fork_time = now
                self.fork_handled_count = 2
                self.fork_cool_until = now + self.fork_cooldown_sec
                rospy.loginfo("Second fork detected, turning right")

        # 停车检测
        finish_result = self.detect_finish(mask)
        finish_detected = finish_result.detected
        self.last_finish_result = finish_result
        parking_candidate, parking_reached, parking_bottom_y_ratio = self.evaluate_parking_target(
            finish_result, roi_origin_y, frame.shape[0], frame.shape[1]
        )
        if parking_candidate and lane_center is None:
            lane_center = frame.shape[1] / 2.0

        if not self.finish_detection_enabled:
            finish_detected = False
            parking_candidate = False
            parking_reached = False

        self.finish_parking_bottom_y_ratio = parking_bottom_y_ratio if parking_candidate else 0.0
        if parking_candidate:
            self.finish_parking_candidate_frames += 1
        else:
            self.finish_parking_candidate_frames = 0
        if parking_reached:
            self.finish_parking_reached_frames += 1
        else:
            self.finish_parking_reached_frames = 0

        # odom 辅助接近
        if self.finish_use_odom_approach:
            if (not self.finish_odom_active and
                    self.finish_parking_candidate_frames >= max(1, self.finish_odom_min_trigger_frames)):
                self.start_finish_odom_approach(now)
            if self.finish_odom_active:
                self.update_finish_odom_distance()
                if self.finish_odom_distance_m >= self.finish_odom_approach_distance_m:
                    if self.finish_auto_stop:
                        rospy.loginfo("Odom parking target reached")
                        self.finish_time = now
                        self.finish_phase = "finish_stop"
                        self.set_status("finish_stop")
                        self.hard_stop_robot()
                        self.publish_debug_image(frame, mask, roi_origin_y, observations, lane_center,
                                                 finish_result, fork_rows, fork_detected_latched, now)
                        self.publish_status()
                        return
                if self.finish_auto_stop and self.finish_odom_timed_out(now):
                    rospy.logwarn("Odom parking timed out")
                    self.finish_time = now
                    self.finish_phase = "finish_stop"
                    self.set_status("finish_stop")
                    self.hard_stop_robot()
                    self.publish_debug_image(frame, mask, roi_origin_y, observations, lane_center,
                                             finish_result, fork_rows, fork_detected_latched, now)
                    self.publish_status()
                    return

        # 停车状态机
        if finish_detected or parking_candidate or self.finish_odom_active:
            self.finish_frames += 1
            self.finish_lost_frames = 0
            if parking_reached:
                self.finish_phase = "parking_ready"
            elif parking_candidate:
                self.finish_phase = "parking_approach"
            else:
                self.finish_phase = "approach_finish"
        else:
            self.finish_lost_frames += 1
            self.finish_frames = 0
            self.finish_phase = "search"

        if (not self.finish_use_odom_approach) and self.finish_parking_reached_frames >= self.finish_parking_confirm_frames:
            if self.finish_auto_stop:
                rospy.loginfo("Visual parking target reached")
                self.finish_time = now
                self.finish_phase = "finish_stop"
                self.set_status("finish_stop")
                self.hard_stop_robot()
                self.publish_debug_image(frame, mask, roi_origin_y, observations, lane_center,
                                         finish_result, fork_rows, fork_detected_latched, now)
                self.publish_status()
                return

        if not self.started:
            self.stop_robot()
            self.publish_debug_image(frame, mask, roi_origin_y, observations, lane_center,
                                     finish_result, fork_rows, fork_detected_latched, now)
            return

        # 控制信号
        if lane_center is not None:
            self.last_detection_time = now
            raw = lane_center
            if self.finish_frames > 0 and self.last_lane_center is not None:
                if abs(raw - self.last_lane_center) > self.finish_center_jump_reject_px:
                    raw = self.last_lane_center
                alpha = max(0.0, min(1.0, self.finish_approach_center_alpha))
                lane_center = alpha * self.last_lane_center + (1.0 - alpha) * raw
            self.last_lane_center = lane_center

            two_sided = any(obs.left_x is not None and obs.right_x is not None for obs in observations)
            if two_sided:
                self.single_line_frames = 0
            else:
                self.single_line_frames += 1

            target = lane_center
            if self.right_turn_active and now < self.turn_until:
                target += self.turn_bias_px
                self.set_status("turn_right")
            elif not two_sided and self.single_line_frames > self.single_line_hold_frames:
                self.set_status("searching")
            else:
                self.set_status("tracking")

            self.publish_control(target, frame.shape[1], now, two_sided)
        else:
            self.single_line_frames += 1
            self.handle_lost_or_search(now)

        self.publish_debug_image(frame, mask, roi_origin_y, observations, lane_center,
                                 finish_result, fork_rows, fork_detected_latched, now)
        self.publish_status()

    # ---------- 图像处理 ----------
    def extract_white_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        height = frame.shape[0]
        y0 = int(height * self.roi_y_start_ratio)
        y1 = int(height * self.roi_y_end_ratio)
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        roi = frame[y0:y1, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_hsv = cv2.inRange(hsv, (0, 0, self.white_v_min), (179, self.white_s_max, 255))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white_gray = cv2.threshold(gray, self.gray_white_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(white_hsv, white_gray)
        mask = self.remove_small_components(mask)

        kernel_size = max(3, self.morph_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask, y0

    def remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
        filtered = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_contour_area:
                cv2.drawContours(filtered, [contour], -1, 255, thickness=cv2.FILLED)
        return filtered

    def observe_lane(self, mask: np.ndarray, image_width: int, force_right: bool = False) -> List[RowObservation]:
        observations = []
        roi_height = mask.shape[0]
        for ratio in self.scan_row_ratios:
            y = int(max(0, min(roi_height - 1, roi_height * ratio)))
            segments = self.find_segments(mask[y, :])
            left_x, right_x, center_x, multi_candidate = self.choose_lane_pair(segments, image_width, force_right)
            observations.append(RowObservation(y, segments, left_x, right_x, center_x, multi_candidate))
        return observations

    def find_segments(self, row: np.ndarray) -> List[Segment]:
        active = row > 0
        segments = []
        start = None
        for idx, val in enumerate(active):
            if val and start is None:
                start = idx
            elif not val and start is not None:
                self._append_segment(segments, start, idx - 1)
                start = None
        if start is not None:
            self._append_segment(segments, start, len(active) - 1)
        return self.merge_close_segments(segments)

    def _append_segment(self, segments: List[Segment], start: int, end: int):
        width = end - start + 1
        if width >= self.min_line_width_px:
            segments.append(Segment(start, end, (start + end) / 2.0, width))

    def merge_close_segments(self, segments: List[Segment]) -> List[Segment]:
        if not segments:
            return []
        merged = [segments[0]]
        for seg in segments[1:]:
            prev = merged[-1]
            if seg.left - prev.right <= self.min_segment_gap_px:
                merged[-1] = Segment(prev.left, seg.right,
                                     (prev.left + seg.right) / 2.0,
                                     seg.right - prev.left + 1)
            else:
                merged.append(seg)
        return merged

    def choose_lane_pair(self, segments: List[Segment], image_width: int, force_right: bool = False):
        lane_width_px = self.current_lane_width_px()
        if len(segments) >= 2:
            multi = len(segments) >= self.fork_candidate_count
            if force_right:
                left = segments[-2]
                right = segments[-1]
                return left.center, right.center, (left.center + right.center) / 2.0, multi
            else:
                return self.best_pair_near_image_center(segments, image_width)
        if len(segments) == 1:
            seg = segments[0]
            if force_right:
                center = seg.center - lane_width_px / 2.0
                return None, seg.center, center, False
            else:
                if seg.center < image_width / 2.0:
                    center = seg.center + lane_width_px / 2.0
                    return seg.center, None, center, False
                else:
                    center = seg.center - lane_width_px / 2.0
                    return None, seg.center, center, False
        return None, None, None, False

    def best_pair_near_image_center(self, segments: List[Segment], image_width: int):
        image_center = image_width / 2.0
        lane_width_px = self.current_lane_width_px()
        best_pair = (segments[0], segments[1])
        best_score = float("inf")
        for left, right in zip(segments, segments[1:]):
            center = (left.center + right.center) / 2.0
            width = right.center - left.center
            width_penalty = abs(width - lane_width_px) * 0.25
            score = abs(center - image_center) + width_penalty
            if score < best_score:
                best_pair = (left, right)
                best_score = score
        return best_pair

    def current_lane_width_px(self) -> float:
        return self.estimated_lane_width_px if self.enable_lane_width_adapt else self.lane_width_px_init

    def update_lane_width_estimate(self, observations: Sequence[RowObservation]):
        if not self.enable_lane_width_adapt:
            return
        samples = []
        for obs in observations:
            if obs.left_x and obs.right_x and not obs.multi_candidate:
                w = obs.right_x - obs.left_x
                if self.lane_width_px_min <= w <= self.lane_width_px_max:
                    samples.append(w)
        if samples:
            avg = float(np.mean(samples))
            alpha = self.lane_width_adapt_alpha
            self.estimated_lane_width_px = (1.0 - alpha) * self.estimated_lane_width_px + alpha * avg
            self.estimated_lane_width_px = max(self.lane_width_px_min, min(self.lane_width_px_max, self.estimated_lane_width_px))

    def estimate_lane_center(self, observations: Sequence[RowObservation], image_width: int) -> Optional[float]:
        centers, weights = [], []
        total = len(observations)
        for i, obs in enumerate(observations):
            if obs.center_x is None:
                continue
            weight = 1.0 + (float(i) / max(total - 1, 1)) * (self.target_row_weight_bottom - 1.0)
            centers.append(obs.center_x)
            weights.append(weight)
        if not centers:
            return None
        center = float(np.average(np.array(centers), weights=np.array(weights)))
        return max(0.0, min(float(image_width - 1), center))

    # ---------- 停车检测函数 ----------
    def detect_finish(self, mask: np.ndarray) -> FinishDetectionResult:
        height, width = mask.shape[:2]
        y0 = int(height * self.finish_bottom_ratio)
        bottom = mask[y0:, :]
        if bottom.size == 0:
            return FinishDetectionResult(False, None, 0.0, 0.0, 0.0, 0.0, 0)

        row_min_width = int(width * self.finish_horizontal_min_width_ratio)
        wide_rows = 0
        for row in bottom:
            segs = self.find_segments(row)
            if any(seg.width >= row_min_width for seg in segs):
                wide_rows += 1
        has_horizontal_edge = wide_rows >= self.finish_horizontal_min_rows
        horizontal_width_ratio = float(max([0] + [max([s.width for s in self.find_segments(row)] + [0]) for row in bottom])) / max(1.0, float(width))

        col_projection = bottom > 0
        min_side_height = int(bottom.shape[0] * self.finish_vertical_side_min_height_ratio)
        left_band = col_projection[:, : width // 3]
        right_band = col_projection[:, (width * 2) // 3 :]
        left_cols = np.sum(left_band, axis=0) if left_band.size else np.array([])
        right_cols = np.sum(right_band, axis=0) if right_band.size else np.array([])
        left_h = int(np.max(left_cols)) if left_cols.size else 0
        right_h = int(np.max(right_cols)) if right_cols.size else 0
        has_left_side = bool(left_cols.size and left_h >= min_side_height)
        has_right_side = bool(right_cols.size and right_h >= min_side_height)
        left_h_ratio = float(left_h) / max(1.0, float(bottom.shape[0]))
        right_h_ratio = float(right_h) / max(1.0, float(bottom.shape[0]))

        contour_result = cv2.findContours(bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
        if not contours:
            return FinishDetectionResult(False, None, horizontal_width_ratio, left_h_ratio, right_h_ratio, 0.0, 0)

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        box = (x, y0 + y, w, h)
        box_region = bottom[y : y + h, x : x + w]
        if box_region.size == 0:
            return FinishDetectionResult(False, box, horizontal_width_ratio, left_h_ratio, right_h_ratio, 0.0, 0)
        fill_ratio = float(np.count_nonzero(box_region)) / float(box_region.size)
        n_labels, _, _, _ = cv2.connectedComponentsWithStats((box_region > 0).astype(np.uint8), connectivity=8)
        component_count = max(0, int(n_labels) - 1)
        good_fill = fill_ratio >= self.finish_box_min_fill_ratio
        good_conn = component_count <= self.finish_box_max_components

        box_area_ratio = float(w * h) / max(1.0, float(bottom.shape[0] * width))
        good_box_area = box_area_ratio >= self.finish_box_min_area_ratio
        box_height_ratio = float(h) / max(1.0, float(bottom.shape[0]))
        good_box_height = box_height_ratio >= self.finish_box_min_height_ratio
        box_bottom = y + h
        good_bottom_touch = float(box_bottom) / max(1.0, float(bottom.shape[0])) >= self.finish_box_bottom_touch_ratio
        box_center_x = x + w / 2.0
        center_tolerance_px = self.finish_box_center_tolerance_ratio * float(width)
        good_center = abs(box_center_x - (width / 2.0)) <= center_tolerance_px

        detected = (has_horizontal_edge and has_left_side and has_right_side and
                    good_fill and good_conn and good_box_area and
                    good_box_height and good_bottom_touch and good_center)
        return FinishDetectionResult(detected, box, horizontal_width_ratio, left_h_ratio, right_h_ratio, fill_ratio, component_count)

    def evaluate_parking_target(self, finish_result: FinishDetectionResult, roi_origin_y: int,
                                image_height: int, image_width: int) -> Tuple[bool, bool, float]:
        box = finish_result.candidate_box
        if box is None:
            return False, False, 0.0
        x, y, w, h = box
        bottom_y_ratio = float(roi_origin_y + y + h) / max(1.0, float(image_height))
        box_width_ratio = float(w) / max(1.0, float(image_width))
        box_height_ratio = float(h) / max(1.0, float(image_height))

        candidate = (finish_result.horizontal_width_ratio >= self.finish_parking_min_horizontal_width_ratio and
                     finish_result.vertical_left_height_ratio >= self.finish_parking_min_vertical_side_height_ratio and
                     finish_result.vertical_right_height_ratio >= self.finish_parking_min_vertical_side_height_ratio and
                     box_width_ratio >= self.finish_parking_min_box_width_ratio and
                     box_height_ratio >= self.finish_parking_min_box_height_ratio and
                     finish_result.inner_fill_ratio >= self.finish_box_min_fill_ratio and
                     finish_result.inner_component_count <= self.finish_box_max_components)
        reached = candidate and bottom_y_ratio >= self.finish_parking_target_bottom_y_ratio
        return candidate, reached, bottom_y_ratio

    def start_finish_odom_approach(self, now: float) -> bool:
        if self.current_odom_xy is None:
            rospy.logwarn_throttle(1.0, "No odom available for parking approach")
            return False
        if self.odom_age(now) > self.finish_odom_timeout_sec:
            rospy.logwarn_throttle(1.0, "Odom stale for parking approach")
            return False
        self.finish_odom_active = True
        self.finish_odom_start_xy = self.current_odom_xy
        self.finish_odom_start_time = now
        self.finish_odom_distance_m = 0.0
        rospy.loginfo("Parking odom approach started")
        return True

    def update_finish_odom_distance(self):
        if self.current_odom_xy is None or self.finish_odom_start_xy is None:
            return
        dx = self.current_odom_xy[0] - self.finish_odom_start_xy[0]
        dy = self.current_odom_xy[1] - self.finish_odom_start_xy[1]
        self.finish_odom_distance_m = math.hypot(dx, dy)

    def odom_age(self, now: float) -> float:
        if self.last_odom_time is None:
            return float("inf")
        return max(0.0, now - self.last_odom_time)

    def finish_odom_timed_out(self, now: float) -> bool:
        return self.finish_odom_active and self.odom_age(now) > self.finish_odom_timeout_sec

    # ---------- 运动控制 ----------
    def publish_control(self, lane_center: float, image_width: int, now: float, two_sided: bool):
        error = lane_center - image_width / 2.0
        self.last_error_px = error
        angular = -self.pid.update(error, now)
        limit = self.max_angular_speed
        if self.finish_frames > 0 or self.finish_parking_candidate_frames > 0:
            limit = min(limit, self.finish_approach_max_angular_speed)
        angular = max(-limit, min(limit, angular))

        if self.right_turn_active and now < self.turn_until:
            linear = self.turn_linear_speed
        elif not two_sided and self.single_line_frames > self.single_line_hold_frames:
            linear = self.search_linear_speed
        else:
            slowdown = min(abs(error) / max(self.error_slowdown_px, 1.0), 1.0)
            linear = self.base_linear_speed - slowdown * (self.base_linear_speed - self.min_linear_speed)
            linear = max(self.min_linear_speed, min(self.base_linear_speed, linear))
            if self.finish_frames > 0 or self.finish_parking_candidate_frames > 0:
                linear *= max(0.2, min(1.0, self.finish_approach_linear_speed_scale))

        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def handle_lost_or_search(self, now: float):
        if self.last_detection_time is None or now - self.last_detection_time <= self.lost_timeout:
            self.set_status("searching")
            twist = Twist()
            twist.linear.x = self.search_linear_speed
            twist.angular.z = 0.0
            self.cmd_pub.publish(twist)
            return
        if self.stop_on_lost:
            self.pid.reset()
            self.stop_robot()
            self.set_status("lost")
            return
        self.set_status("searching")
        twist = Twist()
        twist.linear.x = self.search_linear_speed
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def handle_finish_maneuver(self, now: float):
        if now - self.finish_time < self.finish_stop_time:
            self.stop_robot()
            self.set_status("finish_stop")
        else:
            self.stop_robot()
            self.set_status("finish")

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def hard_stop_robot(self):
        for _ in range(4):
            self.cmd_pub.publish(Twist())

    # ---------- 状态与调试 ----------
    def set_status(self, status: str):
        if self.status != status:
            self.status = status
            self.publish_status(force=True)

    def publish_status(self, force: bool = False):
        self.status_pub.publish(String(data=self.status))

    def update_debug_snapshot(self, frame, roi_origin_y, observations, lane_center, finish_result,
                              fork_rows, fork_detected, now):
        height, width = frame.shape[:2]
        box = finish_result.candidate_box
        box_info = None
        if box is not None:
            x, y, w, h = box
            full_y = roi_origin_y + y
            box_info = {
                "roi_x": x, "roi_y": y,
                "full_x": x, "full_y": full_y,
                "width_px": w, "height_px": h,
                "center_x_px": x + w / 2.0,
                "center_y_full_px": full_y + h / 2.0,
                "bottom_y_full_px": full_y + h,
                "center_x_ratio": (x + w / 2.0) / max(1.0, float(width)),
                "center_y_full_ratio": (full_y + h / 2.0) / max(1.0, float(height)),
                "bottom_y_full_ratio": (full_y + h) / max(1.0, float(height)),
                "width_ratio": w / max(1.0, float(width)),
                "height_full_ratio": h / max(1.0, float(height)),
            }
        self.last_debug_snapshot = {
            "timestamp_sec": now,
            "status": self.status,
            "finish_phase": self.finish_phase,
            "started": self.started,
            "image_width": width,
            "image_height": height,
            "roi_origin_y": roi_origin_y,
            "lane_center_px": lane_center,
            "lane_center_ratio": None if lane_center is None else lane_center / max(1.0, float(width)),
            "last_error_px": self.last_error_px,
            "finish_detection_enabled": self.finish_detection_enabled,
            "finish_frames": self.finish_frames,
            "finish_parking_candidate_frames": self.finish_parking_candidate_frames,
            "finish_parking_reached_frames": self.finish_parking_reached_frames,
            "finish_parking_bottom_y_ratio": self.finish_parking_bottom_y_ratio,
            "finish_odom_active": self.finish_odom_active,
            "finish_odom_start_xy": self.finish_odom_start_xy,
            "finish_odom_current_xy": self.current_odom_xy,
            "finish_odom_distance_m": self.finish_odom_distance_m,
            "finish_detected": finish_result.detected,
            "finish_candidate_box": box_info,
            "finish_metrics": {
                "horizontal_width_ratio": finish_result.horizontal_width_ratio,
                "vertical_left_height_ratio": finish_result.vertical_left_height_ratio,
                "vertical_right_height_ratio": finish_result.vertical_right_height_ratio,
                "inner_fill_ratio": finish_result.inner_fill_ratio,
                "inner_component_count": finish_result.inner_component_count,
            },
            "fork_rows": fork_rows,
            "fork_detected": fork_detected,
            "fork_handled_count": self.fork_handled_count,
            "estimated_lane_width_px": self.current_lane_width_px(),
            "observations": [
                {
                    "roi_y": obs.y,
                    "left_x": obs.left_x,
                    "right_x": obs.right_x,
                    "center_x": obs.center_x,
                    "multi_candidate": obs.multi_candidate,
                    "segments": [{"left": s.left, "right": s.right, "center": s.center, "width": s.width} for s in obs.segments],
                }
                for obs in observations
            ],
            "control_params": {
                "turn_bias_px": self.turn_bias_px,
                "turn_hold_time": self.turn_hold_time,
            }
        }

    def publish_debug_image(self, frame, mask, roi_origin_y, observations, lane_center,
                            finish_result, fork_rows, fork_detected, now):
        if not self.publish_debug or self.debug_pub.get_num_connections() == 0:
            return
        debug = frame.copy()
        height, width = debug.shape[:2]
        cv2.rectangle(debug, (0, roi_origin_y), (width - 1, height - 1), (80, 80, 0), 1)
        cv2.line(debug, (width//2, roi_origin_y), (width//2, height-1), (255,0,0), 1)
        for obs in observations:
            y = roi_origin_y + obs.y
            cv2.line(debug, (0, y), (width-1, y), (45,45,45), 1)
            for seg in obs.segments:
                cv2.circle(debug, (int(seg.center), y), 4, (0,255,255), -1)
                cv2.line(debug, (seg.left, y), (seg.right, y), (0,255,255), 2)
            if obs.left_x is not None:
                cv2.circle(debug, (int(obs.left_x), y), 5, (0,255,0), 1)
            if obs.right_x is not None:
                cv2.circle(debug, (int(obs.right_x), y), 5, (0,255,0), 1)
            if obs.center_x is not None:
                cv2.circle(debug, (int(obs.center_x), y), 5, (0,0,255), -1)
        if lane_center is not None:
            cv2.line(debug, (int(lane_center), roi_origin_y), (int(lane_center), height-1), (0,0,255), 2)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (width//3, max(1, mask_bgr.shape[0]//3)))
        mh, mw = mask_bgr.shape[:2]
        debug[0:mh, 0:mw] = mask_bgr
        if finish_result.candidate_box is not None:
            x, y, w, h = finish_result.candidate_box
            full_y = roi_origin_y + y
            cv2.rectangle(debug, (x, full_y), (x+w, full_y+h), (255,120,0), 2)
        turn_sec = max(0.0, self.turn_until - now)
        text = f"status={self.status} phase={self.finish_phase} finish={self.finish_frames} forks={self.fork_handled_count} turn={turn_sec:.1f}s"
        cv2.putText(debug, text, (10, max(mh+25,30)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,0), 2)
        text2 = f"bias={self.turn_bias_px} h={finish_result.horizontal_width_ratio:.2f} vl={finish_result.vertical_left_height_ratio:.2f} vr={finish_result.vertical_right_height_ratio:.2f}"
        cv2.putText(debug, text2, (10, max(mh+50,55)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,200,255), 2)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "Debug image conversion failed: %s", exc)


def main():
    rospy.init_node("line_follow_straight_right")
    LineFollowStraightRightNode()
    rospy.spin()


if __name__ == "__main__":
    main()
    