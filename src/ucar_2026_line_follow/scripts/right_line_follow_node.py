#!/usr/bin/env python3
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
    selected_pair: Optional[Tuple[int, int]]


@dataclass
class ParkingDetectionResult:
    detected: bool
    reached: bool
    candidate_box: Optional[Tuple[int, int, int, int]]
    center_x: Optional[float]
    bottom_y_ratio: float
    area_ratio: float
    width_ratio: float
    height_ratio: float
    fill_ratio: float
    component_count: int


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


class RightLineFollowNode:
    def __init__(self):
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("~image_topic", rospy.get_param("image_topic", "/usb_cam/image_raw"))
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", rospy.get_param("cmd_vel_topic", "/cmd_vel"))
        self.odom_topic = rospy.get_param("~odom_topic", rospy.get_param("odom_topic", "/odom"))
        self.status_topic = rospy.get_param("~status_topic", rospy.get_param("status_topic", "/right_line_follow/status"))
        self.debug_image_topic = rospy.get_param(
            "~debug_image_topic", rospy.get_param("debug_image_topic", "/right_line_follow/debug_image")
        )
        self.start_topic = rospy.get_param("~start_topic", rospy.get_param("start_topic", "/right_line_follow/start"))

        self.auto_start = bool(rospy.get_param("~auto_start", rospy.get_param("auto_start", True)))
        self.started = self.auto_start
        self.publish_debug = bool(rospy.get_param("~publish_debug", rospy.get_param("publish_debug", True)))

        self.lane_width_px_init = float(
            rospy.get_param("~lane_width_px_init", rospy.get_param("lane_width_px_init", 230))
        )
        self.lane_width_px_min = float(
            rospy.get_param("~lane_width_px_min", rospy.get_param("lane_width_px_min", 150))
        )
        self.lane_width_px_max = float(
            rospy.get_param("~lane_width_px_max", rospy.get_param("lane_width_px_max", 320))
        )
        self.lane_width_adapt_alpha = float(
            rospy.get_param("~lane_width_adapt_alpha", rospy.get_param("lane_width_adapt_alpha", 0.2))
        )
        self.enable_lane_width_adapt = bool(
            rospy.get_param("~enable_lane_width_adapt", rospy.get_param("enable_lane_width_adapt", False))
        )
        self.estimated_lane_width_px = self.lane_width_px_init
        self.single_line_hold_frames = int(
            rospy.get_param("~single_line_hold_frames", rospy.get_param("single_line_hold_frames", 12))
        )

        self.roi_y_start_ratio = float(rospy.get_param("~roi_y_start_ratio", rospy.get_param("roi_y_start_ratio", 0.45)))
        self.roi_y_end_ratio = float(rospy.get_param("~roi_y_end_ratio", rospy.get_param("roi_y_end_ratio", 1.0)))
        self.white_s_max = int(rospy.get_param("~white_s_max", rospy.get_param("white_s_max", 85)))
        self.white_v_min = int(rospy.get_param("~white_v_min", rospy.get_param("white_v_min", 150)))
        self.gray_white_threshold = int(
            rospy.get_param("~gray_white_threshold", rospy.get_param("gray_white_threshold", 185))
        )
        self.morph_kernel_size = int(rospy.get_param("~morph_kernel_size", rospy.get_param("morph_kernel_size", 5)))
        self.min_line_width_px = int(rospy.get_param("~min_line_width_px", rospy.get_param("min_line_width_px", 6)))
        self.min_segment_gap_px = int(
            rospy.get_param("~min_segment_gap_px", rospy.get_param("min_segment_gap_px", 12))
        )
        self.min_contour_area = float(rospy.get_param("~min_contour_area", rospy.get_param("min_contour_area", 60.0)))
        self.scan_row_ratios = self._get_float_list("scan_row_ratios", [0.20, 0.35, 0.50, 0.65, 0.80, 0.92])
        self.target_row_weight_bottom = float(
            rospy.get_param("~target_row_weight_bottom", rospy.get_param("target_row_weight_bottom", 1.5))
        )

        kp = float(rospy.get_param("~kp", rospy.get_param("kp", 0.0045)))
        ki = float(rospy.get_param("~ki", rospy.get_param("ki", 0.0)))
        kd = float(rospy.get_param("~kd", rospy.get_param("kd", 0.0015)))
        max_integral = float(rospy.get_param("~max_integral", rospy.get_param("max_integral", 80.0)))
        self.pid = PidController(kp, ki, kd, max_integral)

        self.base_linear_speed = float(
            rospy.get_param("~base_linear_speed", rospy.get_param("base_linear_speed", 0.15))
        )
        self.min_linear_speed = float(rospy.get_param("~min_linear_speed", rospy.get_param("min_linear_speed", 0.055)))
        self.search_linear_speed = float(
            rospy.get_param("~search_linear_speed", rospy.get_param("search_linear_speed", 0.035))
        )
        self.max_angular_speed = float(
            rospy.get_param("~max_angular_speed", rospy.get_param("max_angular_speed", 0.8))
        )
        self.error_slowdown_px = float(
            rospy.get_param("~error_slowdown_px", rospy.get_param("error_slowdown_px", 160.0))
        )
        self.search_angular_speed = float(
            rospy.get_param("~search_angular_speed", rospy.get_param("search_angular_speed", 0.25))
        )
        self.lost_timeout = float(rospy.get_param("~lost_timeout", rospy.get_param("lost_timeout", 1.0)))
        self.stop_on_lost = bool(rospy.get_param("~stop_on_lost", rospy.get_param("stop_on_lost", False)))

        self.fork_candidate_count = int(
            rospy.get_param("~fork_candidate_count", rospy.get_param("fork_candidate_count", 3))
        )
        self.fork_center_tolerance_px = float(
            rospy.get_param("~fork_center_tolerance_px", rospy.get_param("fork_center_tolerance_px", 220.0))
        )
        self.fork_cooldown_sec = float(rospy.get_param("~fork_cooldown_sec", rospy.get_param("fork_cooldown_sec", 0.8)))
        self.fork_latch_time = float(rospy.get_param("~fork_latch_time", rospy.get_param("fork_latch_time", 0.45)))
        self.right_turn_bias_px = float(
            rospy.get_param("~right_turn_bias_px", rospy.get_param("right_turn_bias_px", 65.0))
        )
        self.right_turn_hold_time = float(
            rospy.get_param("~right_turn_hold_time", rospy.get_param("right_turn_hold_time", 1.15))
        )
        self.right_turn_linear_speed = float(
            rospy.get_param("~right_turn_linear_speed", rospy.get_param("right_turn_linear_speed", 0.08))
        )
        self.startup_right_bias_duration = float(
            rospy.get_param("~startup_right_bias_duration", rospy.get_param("startup_right_bias_duration", 1.7))
        )
        self.startup_force_right_until_dual_frames = int(
            rospy.get_param(
                "~startup_force_right_until_dual_frames",
                rospy.get_param("startup_force_right_until_dual_frames", 8),
            )
        )
        self.startup_force_right_clear_nonfork_frames = int(
            rospy.get_param(
                "~startup_force_right_clear_nonfork_frames",
                rospy.get_param("startup_force_right_clear_nonfork_frames", 18),
            )
        )
        self.startup_force_right_min_duration = float(
            rospy.get_param(
                "~startup_force_right_min_duration",
                rospy.get_param("startup_force_right_min_duration", 3.2),
            )
        )
        self.startup_force_right_bias_px = float(
            rospy.get_param("~startup_force_right_bias_px", rospy.get_param("startup_force_right_bias_px", 115.0))
        )
        self.right_anchor_ratio = float(rospy.get_param("~right_anchor_ratio", rospy.get_param("right_anchor_ratio", 0.72)))

        self.parking_enable_delay = float(
            rospy.get_param("~parking_enable_delay", rospy.get_param("parking_enable_delay", 6.0))
        )
        self.parking_roi_y_start_ratio = float(
            rospy.get_param("~parking_roi_y_start_ratio", rospy.get_param("parking_roi_y_start_ratio", 0.58))
        )
        self.parking_roi_y_end_ratio = float(
            rospy.get_param("~parking_roi_y_end_ratio", rospy.get_param("parking_roi_y_end_ratio", 1.0))
        )
        self.black_v_max = int(rospy.get_param("~black_v_max", rospy.get_param("black_v_max", 80)))
        self.black_gray_max = int(rospy.get_param("~black_gray_max", rospy.get_param("black_gray_max", 75)))
        self.black_s_min = int(rospy.get_param("~black_s_min", rospy.get_param("black_s_min", 0)))
        self.black_morph_kernel_size = int(
            rospy.get_param("~black_morph_kernel_size", rospy.get_param("black_morph_kernel_size", 7))
        )
        self.parking_min_area_ratio = float(
            rospy.get_param("~parking_min_area_ratio", rospy.get_param("parking_min_area_ratio", 0.035))
        )
        self.parking_min_width_ratio = float(
            rospy.get_param("~parking_min_width_ratio", rospy.get_param("parking_min_width_ratio", 0.25))
        )
        self.parking_min_height_ratio = float(
            rospy.get_param("~parking_min_height_ratio", rospy.get_param("parking_min_height_ratio", 0.08))
        )
        self.parking_max_height_ratio = float(
            rospy.get_param("~parking_max_height_ratio", rospy.get_param("parking_max_height_ratio", 0.55))
        )
        self.parking_min_aspect_ratio = float(
            rospy.get_param("~parking_min_aspect_ratio", rospy.get_param("parking_min_aspect_ratio", 0.9))
        )
        self.parking_max_aspect_ratio = float(
            rospy.get_param("~parking_max_aspect_ratio", rospy.get_param("parking_max_aspect_ratio", 5.0))
        )
        self.parking_min_fill_ratio = float(
            rospy.get_param("~parking_min_fill_ratio", rospy.get_param("parking_min_fill_ratio", 0.45))
        )
        self.parking_max_components = int(
            rospy.get_param("~parking_max_components", rospy.get_param("parking_max_components", 3))
        )
        self.parking_center_tolerance_ratio = float(
            rospy.get_param(
                "~parking_center_tolerance_ratio",
                rospy.get_param("parking_center_tolerance_ratio", 0.34),
            )
        )
        self.parking_reached_bottom_y_ratio = float(
            rospy.get_param(
                "~parking_reached_bottom_y_ratio",
                rospy.get_param("parking_reached_bottom_y_ratio", 0.94),
            )
        )
        self.parking_candidate_confirm_frames = int(
            rospy.get_param(
                "~parking_candidate_confirm_frames",
                rospy.get_param("parking_candidate_confirm_frames", 3),
            )
        )
        self.parking_reached_confirm_frames = int(
            rospy.get_param(
                "~parking_reached_confirm_frames",
                rospy.get_param("parking_reached_confirm_frames", 2),
            )
        )
        self.parking_release_frames = int(
            rospy.get_param("~parking_release_frames", rospy.get_param("parking_release_frames", 2))
        )
        self.parking_auto_stop = bool(
            rospy.get_param("~parking_auto_stop", rospy.get_param("parking_auto_stop", True))
        )
        self.parking_stop_time = float(
            rospy.get_param("~parking_stop_time", rospy.get_param("parking_stop_time", 1.0))
        )
        self.parking_use_odom_approach = bool(
            rospy.get_param(
                "~parking_use_odom_approach",
                rospy.get_param("parking_use_odom_approach", True),
            )
        )
        self.parking_odom_approach_distance_m = float(
            rospy.get_param(
                "~parking_odom_approach_distance_m",
                rospy.get_param("parking_odom_approach_distance_m", 0.24),
            )
        )
        self.parking_odom_approach_speed = abs(
            float(
                rospy.get_param(
                    "~parking_odom_approach_speed",
                    rospy.get_param("parking_odom_approach_speed", 0.055),
                )
            )
        )
        self.parking_odom_min_trigger_frames = int(
            rospy.get_param(
                "~parking_odom_min_trigger_frames",
                rospy.get_param("parking_odom_min_trigger_frames", 3),
            )
        )
        self.parking_odom_timeout_sec = float(
            rospy.get_param(
                "~parking_odom_timeout_sec",
                rospy.get_param("parking_odom_timeout_sec", 8.0),
            )
        )
        self.parking_approach_center_alpha = float(
            rospy.get_param(
                "~parking_approach_center_alpha",
                rospy.get_param("parking_approach_center_alpha", 0.70),
            )
        )
        self.parking_center_weight = float(
            rospy.get_param("~parking_center_weight", rospy.get_param("parking_center_weight", 0.55))
        )
        self.parking_approach_max_angular_speed = float(
            rospy.get_param(
                "~parking_approach_max_angular_speed",
                rospy.get_param("parking_approach_max_angular_speed", 0.38),
            )
        )
        self.parking_approach_linear_speed_scale = float(
            rospy.get_param(
                "~parking_approach_linear_speed_scale",
                rospy.get_param("parking_approach_linear_speed_scale", 0.65),
            )
        )
        self.parking_final_linear_speed = float(
            rospy.get_param("~parking_final_linear_speed", rospy.get_param("parking_final_linear_speed", 0.03))
        )

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)

        self.status = "idle" if not self.started else "searching"
        self.last_detection_time = None
        self.last_lane_center = None
        self.last_error_px = 0.0
        self.single_line_frames = 0
        self.right_turn_until = 0.0
        self.last_fork_time = -1e9
        self.fork_latch_until = 0.0
        self.parking_detection_enabled = False
        self.parking_candidate_frames = 0
        self.parking_lost_frames = 0
        self.parking_reached_frames = 0
        self.parking_time = None
        self.parking_phase = "search"
        self.last_parking_result: Optional[ParkingDetectionResult] = None
        self.last_debug_snapshot: Optional[Dict] = None
        self.start_time = time.time()
        self.startup_force_right_mode = True
        self.dual_line_stable_frames = 0
        self.nonfork_stable_frames = 0
        if self.started:
            self.right_turn_until = self.start_time + self.startup_right_bias_duration

        self.current_odom_xy: Optional[Tuple[float, float]] = None
        self.last_odom_time = None
        self.parking_odom_active = False
        self.parking_odom_start_xy: Optional[Tuple[float, float]] = None
        self.parking_odom_start_time = None
        self.parking_odom_distance_m = 0.0

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.start_sub = rospy.Subscriber(self.start_topic, Bool, self.start_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10)

        rospy.on_shutdown(self.stop_robot)
        self.publish_status(force=True)
        rospy.loginfo("right_line_follow_node started. image=%s cmd_vel=%s", self.image_topic, self.cmd_vel_topic)

    def _get_float_list(self, name: str, default: Sequence[float]) -> List[float]:
        value = rospy.get_param("~" + name, rospy.get_param(name, list(default)))
        return [float(item) for item in value]

    def start_callback(self, msg: Bool):
        self.started = bool(msg.data)
        if self.started and self.status in ("idle", "finish"):
            self.start_time = time.time()
            self.parking_detection_enabled = False
            self.parking_candidate_frames = 0
            self.parking_lost_frames = 0
            self.parking_reached_frames = 0
            self.parking_time = None
            self.parking_phase = "search"
            self.startup_force_right_mode = True
            self.dual_line_stable_frames = 0
            self.nonfork_stable_frames = 0
            self.right_turn_until = self.start_time + self.startup_right_bias_duration
            self.reset_parking_odom_approach()
            self.pid.reset()
            self.set_status("searching")
        if not self.started:
            self.pid.reset()
            self.parking_time = None
            self.reset_parking_odom_approach()
            self.stop_robot()
            self.set_status("idle")

    def odom_callback(self, msg: Odometry):
        position = msg.pose.pose.position
        self.current_odom_xy = (float(position.x), float(position.y))
        self.last_odom_time = time.time()

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "cv_bridge conversion failed: %s", exc)
            return

        now = time.time()
        if self.parking_time is not None:
            self.handle_parking_stop(now)
            return
        if not self.parking_detection_enabled:
            self.parking_detection_enabled = (now - self.start_time) >= self.parking_enable_delay

        white_mask, roi_origin_y = self.extract_white_mask(frame)
        force_right = self.startup_force_right_mode or now < self.fork_latch_until
        observations = self.observe_lane(white_mask, frame.shape[1], force_right)
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

        parking_mask, parking_roi_origin_y = self.extract_black_mask(frame)
        parking_result = self.detect_parking(parking_mask, parking_roi_origin_y, frame.shape[0], frame.shape[1])
        if not self.parking_detection_enabled:
            parking_result = ParkingDetectionResult(False, False, parking_result.candidate_box, parking_result.center_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        self.last_parking_result = parking_result

        if parking_result.detected:
            self.parking_candidate_frames += 1
            self.parking_lost_frames = 0
        else:
            self.parking_lost_frames += 1
            if self.parking_lost_frames >= self.parking_release_frames:
                self.parking_candidate_frames = 0
                self.parking_reached_frames = 0
                self.parking_phase = "search"

        if parking_result.reached:
            self.parking_reached_frames += 1
        else:
            self.parking_reached_frames = 0

        if lane_center is not None and parking_result.detected and parking_result.center_x is not None:
            weight = max(0.0, min(1.0, self.parking_center_weight))
            lane_center = (1.0 - weight) * lane_center + weight * parking_result.center_x
        elif lane_center is None and parking_result.detected and parking_result.center_x is not None:
            lane_center = parking_result.center_x

        if self.parking_use_odom_approach:
            if (
                not self.parking_odom_active
                and self.parking_candidate_frames >= max(1, self.parking_odom_min_trigger_frames)
            ):
                self.start_parking_odom_approach(now)

            if self.parking_odom_active:
                self.update_parking_odom_distance()
                self.parking_phase = "parking_odom_approach"
                self.set_status("parking_odom_approach")
                if lane_center is None:
                    lane_center = image_center
                if self.parking_odom_distance_m >= self.parking_odom_approach_distance_m:
                    self.finish_parking(now, frame, white_mask, parking_mask, roi_origin_y, parking_roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected_latched)
                    return
                if self.parking_auto_stop and self.parking_odom_timed_out(now):
                    rospy.logwarn(
                        "parking odom timed out: last_odom_age=%.2fs distance=%.3fm",
                        self.odom_age(now),
                        self.parking_odom_distance_m,
                    )
                    self.finish_parking(now, frame, white_mask, parking_mask, roi_origin_y, parking_roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected_latched)
                    return

        if self.parking_candidate_frames >= self.parking_candidate_confirm_frames:
            if self.parking_odom_active:
                self.parking_phase = "parking_odom_approach"
                self.set_status("parking_odom_approach")
            elif parking_result.reached:
                self.parking_phase = "parking_ready"
                self.set_status("parking_ready")
            else:
                self.parking_phase = "parking_approach"
                self.set_status("parking_approach")

        if (
            (not self.parking_use_odom_approach or not self.parking_odom_active)
            and self.parking_reached_frames >= self.parking_reached_confirm_frames
        ):
            self.finish_parking(now, frame, white_mask, parking_mask, roi_origin_y, parking_roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected_latched)
            return

        self.update_debug_snapshot(
            frame,
            roi_origin_y,
            parking_roi_origin_y,
            observations,
            lane_center,
            parking_result,
            fork_rows,
            fork_detected_latched,
            now,
        )

        if not self.started:
            self.stop_robot()
            self.publish_debug_image(frame, white_mask, parking_mask, roi_origin_y, parking_roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected_latched, now)
            self.publish_status()
            return

        if lane_center is not None:
            self.last_detection_time = now
            if self.parking_candidate_frames > 0 and self.last_lane_center is not None:
                alpha = max(0.0, min(1.0, self.parking_approach_center_alpha))
                lane_center = alpha * self.last_lane_center + (1.0 - alpha) * lane_center
            self.last_lane_center = lane_center

            two_sided_tracking = any(obs.left_x is not None and obs.right_x is not None for obs in observations)
            if two_sided_tracking:
                self.single_line_frames = 0
                self.dual_line_stable_frames += 1
            else:
                self.single_line_frames += 1
                self.dual_line_stable_frames = 0

            startup_elapsed = now - self.start_time
            if self.startup_force_right_mode and self.dual_line_stable_frames >= self.startup_force_right_until_dual_frames:
                if (
                    startup_elapsed >= self.startup_force_right_min_duration
                    and self.nonfork_stable_frames >= self.startup_force_right_clear_nonfork_frames
                ):
                    self.startup_force_right_mode = False

            if fork_detected_latched and (now - self.last_fork_time) >= self.fork_cooldown_sec:
                self.right_turn_until = max(self.right_turn_until, now + self.right_turn_hold_time)
                self.last_fork_time = now

            target_center = lane_center
            if self.startup_force_right_mode:
                target_center += self.startup_force_right_bias_px
                self.set_status("startup_right")
            if now < self.right_turn_until:
                target_center += self.right_turn_bias_px
                self.set_status("turn_right")
            elif self.parking_candidate_frames >= self.parking_candidate_confirm_frames:
                self.set_status(self.parking_phase)
            elif not two_sided_tracking and self.single_line_frames > self.single_line_hold_frames:
                self.set_status("searching")
            else:
                self.set_status("tracking")

            self.publish_control(target_center, frame.shape[1], now, two_sided_tracking)
        else:
            self.single_line_frames += 1
            self.handle_lost_or_search(now)

        self.publish_debug_image(frame, white_mask, parking_mask, roi_origin_y, parking_roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected_latched, now)
        self.publish_status()

    def extract_white_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        height = frame.shape[0]
        y0 = int(height * self.roi_y_start_ratio)
        y1 = int(height * self.roi_y_end_ratio)
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        roi = frame[y0:y1, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv_mask = cv2.inRange(hsv, np.array([0, 0, self.white_v_min]), np.array([179, self.white_s_max, 255]))
        _, gray_mask = cv2.threshold(gray, self.gray_white_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(hsv_mask, gray_mask)
        if self.morph_kernel_size > 1:
            kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return self.remove_small_components(mask, self.min_contour_area), y0

    def extract_black_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        height = frame.shape[0]
        y0 = int(height * self.parking_roi_y_start_ratio)
        y1 = int(height * self.parking_roi_y_end_ratio)
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        roi = frame[y0:y1, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv_mask = cv2.inRange(hsv, np.array([0, self.black_s_min, 0]), np.array([179, 255, self.black_v_max]))
        _, gray_mask = cv2.threshold(gray, self.black_gray_max, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.bitwise_and(hsv_mask, gray_mask)
        if self.black_morph_kernel_size > 1:
            kernel = np.ones((self.black_morph_kernel_size, self.black_morph_kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return self.remove_small_components(mask, self.min_contour_area), y0

    def remove_small_components(self, mask: np.ndarray, min_area: float) -> np.ndarray:
        contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
        filtered = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(filtered, [contour], -1, 255, thickness=cv2.FILLED)
        return filtered

    def observe_lane(self, mask: np.ndarray, image_width: int, force_right_mode: bool = False) -> List[RowObservation]:
        observations = []
        roi_height = mask.shape[0]
        for ratio in self.scan_row_ratios:
            y = int(max(0, min(roi_height - 1, roi_height * ratio)))
            segments = self.find_segments(mask[y, :])
            left_x, right_x, center_x, multi_candidate, selected_pair = self.choose_lane_pair(
                segments, image_width, force_right_mode
            )
            observations.append(RowObservation(y, segments, left_x, right_x, center_x, multi_candidate, selected_pair))
        return observations

    def find_segments(self, row: np.ndarray) -> List[Segment]:
        active = row > 0
        segments = []
        start = None

        for idx, value in enumerate(active):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                self._append_segment(segments, start, idx - 1)
                start = None
        if start is not None:
            self._append_segment(segments, start, len(active) - 1)
        return self.merge_close_segments(segments)

    def _append_segment(self, segments: List[Segment], start: int, end: int):
        width = end - start + 1
        if width >= self.min_line_width_px:
            center = (start + end) / 2.0
            segments.append(Segment(start, end, center, width))

    def merge_close_segments(self, segments: List[Segment]) -> List[Segment]:
        if not segments:
            return []

        merged = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            if segment.left - previous.right <= self.min_segment_gap_px:
                left = previous.left
                right = segment.right
                width = right - left + 1
                merged[-1] = Segment(left, right, (left + right) / 2.0, width)
            else:
                merged.append(segment)
        return merged

    def choose_lane_pair(
        self, segments: List[Segment], image_width: int, force_right_mode: bool = False
    ) -> Tuple[Optional[float], Optional[float], Optional[float], bool, Optional[Tuple[int, int]]]:
        lane_width_px = self.current_lane_width_px()
        if len(segments) >= 2:
            multi_candidate = len(segments) >= self.fork_candidate_count
            if force_right_mode or multi_candidate:
                left_index = len(segments) - 2
                right_index = len(segments) - 1
                left = segments[left_index]
                right = segments[right_index]
                selected_pair = (left_index, right_index)
            else:
                left_index, right_index = self.best_pair_near_image_center(segments, image_width)
                left = segments[left_index]
                right = segments[right_index]
                selected_pair = (left_index, right_index)
            return left.center, right.center, (left.center + right.center) / 2.0, multi_candidate, selected_pair

        if len(segments) == 1:
            segment = segments[0]
            center = self.estimate_single_line_center(segment.center, image_width, force_right_mode)
            if segment.center < center:
                return segment.center, None, center, False, None
            return None, segment.center, center, False, None

        return None, None, None, False, None

    def estimate_single_line_center(self, segment_center: float, image_width: int, force_right_mode: bool) -> float:
        lane_width_px = self.current_lane_width_px()
        if force_right_mode:
            right_anchor = image_width * self.right_anchor_ratio
            candidates = [
                segment_center + lane_width_px / 2.0,
                segment_center - lane_width_px / 2.0,
            ]
            if self.last_lane_center is not None:
                anchor = 0.65 * self.last_lane_center + 0.35 * right_anchor
            else:
                anchor = right_anchor
            return min(candidates, key=lambda value: abs(value - anchor))

        if segment_center < image_width / 2.0:
            return segment_center + lane_width_px / 2.0
        return segment_center - lane_width_px / 2.0

    def best_pair_near_image_center(self, segments: List[Segment], image_width: int) -> Tuple[int, int]:
        image_center = image_width / 2.0
        lane_width_px = self.current_lane_width_px()
        best_pair = (0, 1)
        best_score = float("inf")
        for index, (left, right) in enumerate(zip(segments, segments[1:])):
            center = (left.center + right.center) / 2.0
            width = right.center - left.center
            width_penalty = abs(width - lane_width_px) * 0.25
            score = abs(center - image_center) + width_penalty
            if score < best_score:
                best_pair = (index, index + 1)
                best_score = score
        return best_pair

    def current_lane_width_px(self) -> float:
        if self.enable_lane_width_adapt:
            return self.estimated_lane_width_px
        return self.lane_width_px_init

    def update_lane_width_estimate(self, observations: Sequence[RowObservation]):
        if not self.enable_lane_width_adapt:
            self.estimated_lane_width_px = self.lane_width_px_init
            return

        width_samples = []
        for obs in observations:
            if obs.left_x is None or obs.right_x is None or obs.multi_candidate:
                continue
            sample_width = obs.right_x - obs.left_x
            if self.lane_width_px_min <= sample_width <= self.lane_width_px_max:
                width_samples.append(sample_width)

        if not width_samples:
            return

        sample_mean = float(np.mean(width_samples))
        alpha = max(0.0, min(1.0, self.lane_width_adapt_alpha))
        updated = (1.0 - alpha) * self.estimated_lane_width_px + alpha * sample_mean
        self.estimated_lane_width_px = max(self.lane_width_px_min, min(self.lane_width_px_max, updated))
        rospy.loginfo_throttle(1.0, "right estimated_lane_width_px=%.2f", self.estimated_lane_width_px)

    def estimate_lane_center(self, observations: Sequence[RowObservation], image_width: int) -> Optional[float]:
        centers = []
        weights = []
        total = len(observations)

        for index, obs in enumerate(observations):
            if obs.center_x is None:
                continue
            weight = 1.0 + (float(index) / max(total - 1, 1)) * (self.target_row_weight_bottom - 1.0)
            centers.append(obs.center_x)
            weights.append(weight)

        if not centers:
            return None
        return float(np.average(centers, weights=weights))

    def detect_parking(
        self, mask: np.ndarray, roi_origin_y: int, image_height: int, image_width: int
    ) -> ParkingDetectionResult:
        contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
        if not contours:
            return ParkingDetectionResult(False, False, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        min_area = self.parking_min_area_ratio * float(max(1, mask.shape[0] * image_width))
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue
            box_region = mask[y : y + h, x : x + w]
            fill_ratio = float(np.count_nonzero(box_region)) / float(max(1, box_region.size))
            aspect = float(w) / float(max(1, h))
            width_ratio = float(w) / float(max(1, image_width))
            height_ratio = float(h) / float(max(1, image_height))
            full_bottom_y = roi_origin_y + y + h
            bottom_y_ratio = float(full_bottom_y) / float(max(1, image_height))
            center_x = x + w / 2.0
            center_ok = abs(center_x - image_width / 2.0) <= self.parking_center_tolerance_ratio * image_width
            shape_ok = (
                width_ratio >= self.parking_min_width_ratio
                and self.parking_min_height_ratio <= height_ratio <= self.parking_max_height_ratio
                and self.parking_min_aspect_ratio <= aspect <= self.parking_max_aspect_ratio
                and fill_ratio >= self.parking_min_fill_ratio
                and center_ok
            )
            if not shape_ok:
                continue
            candidates.append((area, x, y, w, h, center_x, bottom_y_ratio, width_ratio, height_ratio, fill_ratio))

        if not candidates:
            return ParkingDetectionResult(False, False, None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

        area, x, y, w, h, center_x, bottom_y_ratio, width_ratio, height_ratio, fill_ratio = max(
            candidates, key=lambda item: item[0]
        )
        box_region = mask[y : y + h, x : x + w]
        n_labels, _, _, _ = cv2.connectedComponentsWithStats((box_region > 0).astype(np.uint8), connectivity=8)
        component_count = max(0, int(n_labels) - 1)
        detected = component_count <= self.parking_max_components
        reached = detected and bottom_y_ratio >= self.parking_reached_bottom_y_ratio
        area_ratio = area / float(max(1, mask.shape[0] * image_width))
        return ParkingDetectionResult(
            detected,
            reached,
            (x, y, w, h),
            center_x,
            bottom_y_ratio,
            area_ratio,
            width_ratio,
            height_ratio,
            fill_ratio,
            component_count,
        )

    def start_parking_odom_approach(self, now: float) -> bool:
        if self.current_odom_xy is None:
            rospy.logwarn_throttle(1.0, "parking candidate locked but no odom is available on %s", self.odom_topic)
            return False
        if self.odom_age(now) > self.parking_odom_timeout_sec:
            rospy.logwarn_throttle(
                1.0,
                "parking candidate locked but odom on %s is stale: age=%.2fs",
                self.odom_topic,
                self.odom_age(now),
            )
            return False

        self.parking_odom_active = True
        self.parking_odom_start_xy = self.current_odom_xy
        self.parking_odom_start_time = now
        self.parking_odom_distance_m = 0.0
        rospy.loginfo(
            "right parking odom approach started: distance=%.3fm speed=%.3fm/s",
            self.parking_odom_approach_distance_m,
            self.parking_odom_approach_speed,
        )
        return True

    def reset_parking_odom_approach(self):
        self.parking_odom_active = False
        self.parking_odom_start_xy = None
        self.parking_odom_start_time = None
        self.parking_odom_distance_m = 0.0

    def update_parking_odom_distance(self):
        if self.current_odom_xy is None or self.parking_odom_start_xy is None:
            return
        dx = self.current_odom_xy[0] - self.parking_odom_start_xy[0]
        dy = self.current_odom_xy[1] - self.parking_odom_start_xy[1]
        self.parking_odom_distance_m = math.hypot(dx, dy)

    def odom_age(self, now: float) -> float:
        if self.last_odom_time is None:
            return float("inf")
        return max(0.0, now - self.last_odom_time)

    def parking_odom_timed_out(self, now: float) -> bool:
        return self.parking_odom_active and self.odom_age(now) > self.parking_odom_timeout_sec

    def publish_control(self, lane_center: float, image_width: int, now: float, two_sided_tracking: bool):
        image_center = image_width / 2.0
        error = lane_center - image_center
        self.last_error_px = error
        angular = -self.pid.update(error, now)
        angular_limit = self.max_angular_speed
        if self.parking_candidate_frames > 0 or self.parking_odom_active:
            angular_limit = min(angular_limit, self.parking_approach_max_angular_speed)
        angular = max(-angular_limit, min(angular_limit, angular))

        if now < self.right_turn_until:
            linear = self.right_turn_linear_speed
        elif not two_sided_tracking and self.single_line_frames > self.single_line_hold_frames:
            linear = self.search_linear_speed
        else:
            slowdown = min(abs(error) / max(self.error_slowdown_px, 1.0), 1.0)
            linear = self.base_linear_speed - slowdown * (self.base_linear_speed - self.min_linear_speed)
            linear = max(self.min_linear_speed, min(self.base_linear_speed, linear))
            if self.parking_candidate_frames > 0:
                linear *= max(0.2, min(1.0, self.parking_approach_linear_speed_scale))
            if self.parking_reached_frames > 0:
                linear = min(linear, self.parking_final_linear_speed)

        if self.parking_odom_active:
            linear = self.parking_odom_approach_speed

        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    def handle_lost_or_search(self, now: float):
        if self.last_detection_time is None or now - self.last_detection_time <= self.lost_timeout:
            self.set_status("searching")
            twist = Twist()
            twist.linear.x = self.search_linear_speed
            if abs(self.last_error_px) > 1.0:
                direction = -1.0 if self.last_error_px > 0.0 else 1.0
            else:
                direction = -1.0
            twist.angular.z = direction * self.search_angular_speed
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
        twist.angular.z = -self.search_angular_speed
        self.cmd_pub.publish(twist)

    def finish_parking(
        self,
        now: float,
        frame: np.ndarray,
        white_mask: np.ndarray,
        parking_mask: np.ndarray,
        roi_origin_y: int,
        parking_roi_origin_y: int,
        observations: Sequence[RowObservation],
        lane_center: Optional[float],
        parking_result: ParkingDetectionResult,
        fork_rows: int,
        fork_detected: bool,
    ):
        self.update_debug_snapshot(
            frame,
            roi_origin_y,
            parking_roi_origin_y,
            observations,
            lane_center,
            parking_result,
            fork_rows,
            fork_detected,
            now,
        )
        if self.parking_auto_stop:
            rospy.loginfo(
                "right parking target reached: odom=%.3fm bottom=%.3f area=%.3f box=%s",
                self.parking_odom_distance_m,
                parking_result.bottom_y_ratio,
                parking_result.area_ratio,
                parking_result.candidate_box,
            )
            self.parking_time = now
            self.parking_phase = "finish_stop"
            self.set_status("finish_stop")
            self.hard_stop_robot()
            self.publish_debug_image(frame, white_mask, parking_mask, roi_origin_y, parking_roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected, now)
            self.publish_status()
            return

        self.parking_phase = "parking_debug"
        self.set_status("parking_debug")

    def handle_parking_stop(self, now: float):
        if self.parking_time is None:
            return
        if now - self.parking_time < self.parking_stop_time:
            self.parking_phase = "finish_stop"
            self.stop_robot()
            self.set_status("finish_stop")
            return
        self.parking_phase = "finish"
        self.stop_robot()
        self.set_status("finish")

    def update_debug_snapshot(
        self,
        frame: np.ndarray,
        roi_origin_y: int,
        parking_roi_origin_y: int,
        observations: Sequence[RowObservation],
        lane_center: Optional[float],
        parking_result: ParkingDetectionResult,
        fork_rows: int,
        fork_detected: bool,
        now: float,
    ):
        height, width = frame.shape[:2]
        box_info = None
        if parking_result.candidate_box is not None:
            x, y, w, h = parking_result.candidate_box
            full_y = parking_roi_origin_y + y
            box_info = {
                "roi_x": x,
                "roi_y": y,
                "full_x": x,
                "full_y": full_y,
                "width_px": w,
                "height_px": h,
                "center_x_px": x + w / 2.0,
                "bottom_y_full_px": full_y + h,
                "center_x_ratio": (x + w / 2.0) / max(1.0, float(width)),
                "bottom_y_full_ratio": (full_y + h) / max(1.0, float(height)),
                "width_ratio": w / max(1.0, float(width)),
                "height_full_ratio": h / max(1.0, float(height)),
            }

        self.last_debug_snapshot = {
            "timestamp_sec": now,
            "status": self.status,
            "parking_phase": self.parking_phase,
            "started": self.started,
            "image_width": width,
            "image_height": height,
            "white_roi_origin_y": roi_origin_y,
            "parking_roi_origin_y": parking_roi_origin_y,
            "lane_center_px": lane_center,
            "lane_center_ratio": None if lane_center is None else lane_center / max(1.0, float(width)),
            "last_error_px": self.last_error_px,
            "startup_force_right_mode": self.startup_force_right_mode,
            "parking_detection_enabled": self.parking_detection_enabled,
            "parking_candidate_frames": self.parking_candidate_frames,
            "parking_reached_frames": self.parking_reached_frames,
            "parking_odom_active": self.parking_odom_active,
            "parking_odom_start_xy": self.parking_odom_start_xy,
            "parking_odom_current_xy": self.current_odom_xy,
            "parking_odom_distance_m": self.parking_odom_distance_m,
            "parking_odom_age_sec": self.odom_age(now),
            "parking_detected": parking_result.detected,
            "parking_reached": parking_result.reached,
            "parking_candidate_box": box_info,
            "parking_metrics": {
                "bottom_y_ratio": parking_result.bottom_y_ratio,
                "area_ratio": parking_result.area_ratio,
                "width_ratio": parking_result.width_ratio,
                "height_ratio": parking_result.height_ratio,
                "fill_ratio": parking_result.fill_ratio,
                "component_count": parking_result.component_count,
            },
            "fork_rows": fork_rows,
            "fork_detected": fork_detected,
            "right_turn_latched": now < self.right_turn_until,
            "estimated_lane_width_px": self.current_lane_width_px(),
            "observations": [
                {
                    "roi_y": obs.y,
                    "full_y": roi_origin_y + obs.y,
                    "left_x": obs.left_x,
                    "right_x": obs.right_x,
                    "center_x": obs.center_x,
                    "multi_candidate": obs.multi_candidate,
                    "selected_pair": obs.selected_pair,
                    "segments": [
                        {
                            "left": segment.left,
                            "right": segment.right,
                            "center": segment.center,
                            "width": segment.width,
                        }
                        for segment in obs.segments
                    ],
                }
                for obs in observations
            ],
        }

    def publish_debug_image(
        self,
        frame: np.ndarray,
        white_mask: np.ndarray,
        parking_mask: np.ndarray,
        roi_origin_y: int,
        parking_roi_origin_y: int,
        observations: Sequence[RowObservation],
        lane_center: Optional[float],
        parking_result: ParkingDetectionResult,
        fork_rows: int,
        fork_detected: bool,
        now: float,
    ):
        if not self.publish_debug or self.debug_pub.get_num_connections() == 0:
            return

        debug = frame.copy()
        height, width = debug.shape[:2]
        image_center = width // 2

        cv2.rectangle(debug, (0, roi_origin_y), (width - 1, height - 1), (80, 80, 0), 1)
        cv2.rectangle(debug, (0, parking_roi_origin_y), (width - 1, height - 1), (0, 80, 80), 1)
        cv2.line(debug, (image_center, roi_origin_y), (image_center, height - 1), (255, 0, 0), 1)

        for obs in observations:
            y = roi_origin_y + obs.y
            cv2.line(debug, (0, y), (width - 1, y), (45, 45, 45), 1)
            for index, segment in enumerate(obs.segments):
                color = (0, 255, 255)
                if obs.selected_pair is not None and index in obs.selected_pair:
                    color = (0, 180, 255)
                cv2.circle(debug, (int(segment.center), y), 4, color, -1)
                cv2.line(debug, (segment.left, y), (segment.right, y), color, 2)
            if obs.left_x is not None:
                cv2.circle(debug, (int(obs.left_x), y), 5, (0, 255, 0), 1)
            if obs.right_x is not None:
                cv2.circle(debug, (int(obs.right_x), y), 5, (0, 255, 0), 1)
            if obs.center_x is not None:
                cv2.circle(debug, (int(obs.center_x), y), 5, (0, 0, 255), -1)

        if lane_center is not None:
            cv2.line(debug, (int(lane_center), roi_origin_y), (int(lane_center), height - 1), (0, 0, 255), 2)

        if parking_result.candidate_box is not None:
            x, y, w, h = parking_result.candidate_box
            full_y = parking_roi_origin_y + y
            color = (0, 255, 255) if parking_result.detected else (0, 120, 255)
            cv2.rectangle(debug, (x, full_y), (x + w, full_y + h), color, 2)
            if parking_result.center_x is not None:
                cv2.line(debug, (int(parking_result.center_x), full_y), (int(parking_result.center_x), full_y + h), color, 1)

        white_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        black_bgr = cv2.cvtColor(parking_mask, cv2.COLOR_GRAY2BGR)
        white_bgr = cv2.resize(white_bgr, (width // 4, max(1, white_bgr.shape[0] // 4)))
        black_bgr = cv2.resize(black_bgr, (width // 4, max(1, black_bgr.shape[0] // 4)))
        wh, ww = white_bgr.shape[:2]
        bh, bw = black_bgr.shape[:2]
        debug[0:wh, 0:ww] = white_bgr
        debug[0:bh, ww : ww + bw] = black_bgr

        turn_left_sec = max(0.0, self.right_turn_until - now)
        text = "status={} phase={} right_mode={} fork_rows={} fork={} turn={:.2f}s park={} reached={} odom={}".format(
            self.status,
            self.parking_phase,
            int(self.startup_force_right_mode),
            fork_rows,
            int(fork_detected),
            turn_left_sec,
            int(parking_result.detected),
            int(parking_result.reached),
            int(self.parking_odom_active),
        )
        cv2.putText(debug, text, (10, max(wh + 25, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
        text2 = "black bottom={:.2f} area={:.3f} fill={:.2f} cc={} odom_dist={:.3f}".format(
            parking_result.bottom_y_ratio,
            parking_result.area_ratio,
            parking_result.fill_ratio,
            parking_result.component_count,
            self.parking_odom_distance_m,
        )
        cv2.putText(debug, text2, (10, max(wh + 50, 55)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 2)

        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "debug image conversion failed: %s", exc)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def hard_stop_robot(self):
        for _ in range(4):
            self.cmd_pub.publish(Twist())

    def set_status(self, status: str):
        if self.status != status:
            self.status = status
            self.publish_status(force=True)

    def publish_status(self, force: bool = False):
        if force:
            self.status_pub.publish(String(data=self.status))
        else:
            self.status_pub.publish(String(data=self.status))


def main():
    rospy.init_node("right_line_follow_node")
    RightLineFollowNode()
    rospy.spin()


if __name__ == "__main__":
    main()
