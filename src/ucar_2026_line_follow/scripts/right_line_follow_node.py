#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
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
    selection: str


@dataclass
class ParkingBoxResult:
    detected: bool
    box: Optional[Tuple[int, int, int, int]]
    horizontal_width_ratio: float
    vertical_left_height_ratio: float
    vertical_right_height_ratio: float
    bottom_y_ratio: float


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
        self.status_topic = rospy.get_param("~status_topic", rospy.get_param("status_topic", "/right_line_follow/status"))
        self.debug_image_topic = rospy.get_param(
            "~debug_image_topic", rospy.get_param("debug_image_topic", "/right_line_follow/debug_image")
        )
        self.debug_info_topic = rospy.get_param(
            "~debug_info_topic", rospy.get_param("debug_info_topic", "/right_line_follow/debug_info")
        )
        self.start_topic = rospy.get_param("~start_topic", rospy.get_param("start_topic", "/right_line_follow/start"))

        self.auto_start = bool(rospy.get_param("~auto_start", rospy.get_param("auto_start", True)))
        self.started = self.auto_start
        self.publish_debug = bool(rospy.get_param("~publish_debug", rospy.get_param("publish_debug", True)))

        self.lane_width_m = float(rospy.get_param("~lane_width_m", rospy.get_param("lane_width_m", 0.42)))
        self.line_width_m = float(rospy.get_param("~line_width_m", rospy.get_param("line_width_m", 0.02)))
        self.parking_box_size_m = float(
            rospy.get_param("~parking_box_size_m", rospy.get_param("parking_box_size_m", 0.50))
        )
        self.lane_width_px = float(rospy.get_param("~lane_width_px", rospy.get_param("lane_width_px", 230.0)))
        self.lane_width_px_min = float(rospy.get_param("~lane_width_px_min", rospy.get_param("lane_width_px_min", 150)))
        self.lane_width_px_max = float(rospy.get_param("~lane_width_px_max", rospy.get_param("lane_width_px_max", 340)))
        self.lane_width_adapt_alpha = float(
            rospy.get_param("~lane_width_adapt_alpha", rospy.get_param("lane_width_adapt_alpha", 0.15))
        )
        self.enable_lane_width_adapt = bool(
            rospy.get_param("~enable_lane_width_adapt", rospy.get_param("enable_lane_width_adapt", True))
        )
        self.estimated_lane_width_px = self.lane_width_px

        self.roi_y_start_ratio = float(rospy.get_param("~roi_y_start_ratio", rospy.get_param("roi_y_start_ratio", 0.45)))
        self.roi_y_end_ratio = float(rospy.get_param("~roi_y_end_ratio", rospy.get_param("roi_y_end_ratio", 1.0)))
        self.white_s_max = int(rospy.get_param("~white_s_max", rospy.get_param("white_s_max", 85)))
        self.white_v_min = int(rospy.get_param("~white_v_min", rospy.get_param("white_v_min", 150)))
        self.gray_white_threshold = int(
            rospy.get_param("~gray_white_threshold", rospy.get_param("gray_white_threshold", 185))
        )
        self.morph_kernel_size = int(rospy.get_param("~morph_kernel_size", rospy.get_param("morph_kernel_size", 5)))
        self.min_contour_area = float(rospy.get_param("~min_contour_area", rospy.get_param("min_contour_area", 60.0)))
        self.min_line_width_px = int(rospy.get_param("~min_line_width_px", rospy.get_param("min_line_width_px", 6)))
        self.min_segment_gap_px = int(
            rospy.get_param("~min_segment_gap_px", rospy.get_param("min_segment_gap_px", 12))
        )
        self.scan_row_ratios = self.get_float_list("scan_row_ratios", [0.20, 0.35, 0.50, 0.65, 0.80, 0.92])
        self.target_row_weight_bottom = float(
            rospy.get_param("~target_row_weight_bottom", rospy.get_param("target_row_weight_bottom", 1.6))
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
        self.turn_linear_speed = float(
            rospy.get_param("~turn_linear_speed", rospy.get_param("turn_linear_speed", 0.075))
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
        self.single_line_hold_frames = int(
            rospy.get_param("~single_line_hold_frames", rospy.get_param("single_line_hold_frames", 12))
        )

        self.fork_candidate_count = int(
            rospy.get_param("~fork_candidate_count", rospy.get_param("fork_candidate_count", 3))
        )
        self.fork_center_tolerance_px = float(
            rospy.get_param("~fork_center_tolerance_px", rospy.get_param("fork_center_tolerance_px", 220.0))
        )
        self.fork_latch_time = float(rospy.get_param("~fork_latch_time", rospy.get_param("fork_latch_time", 0.45)))
        self.fork_cooldown_sec = float(rospy.get_param("~fork_cooldown_sec", rospy.get_param("fork_cooldown_sec", 1.0)))
        self.right_turn_bias_px = abs(
            float(rospy.get_param("~right_turn_bias_px", rospy.get_param("right_turn_bias_px", 25.0)))
        )
        self.turn_hold_time = float(rospy.get_param("~turn_hold_time", rospy.get_param("turn_hold_time", 1.2)))
        self.right_route_lock_duration = float(
            rospy.get_param("~right_route_lock_duration", rospy.get_param("right_route_lock_duration", 0.0))
        )
        self.right_route_relock_duration = float(
            rospy.get_param("~right_route_relock_duration", rospy.get_param("right_route_relock_duration", 3.0))
        )
        self.right_route_pair_width_slack = float(
            rospy.get_param("~right_route_pair_width_slack", rospy.get_param("right_route_pair_width_slack", 1.45))
        )
        self.startup_right_bias_duration = float(
            rospy.get_param("~startup_right_bias_duration", rospy.get_param("startup_right_bias_duration", 0.0))
        )
        self.startup_right_bias_px = abs(
            float(rospy.get_param("~startup_right_bias_px", rospy.get_param("startup_right_bias_px", 0.0)))
        )
        self.startup_maneuver_enabled = bool(
            rospy.get_param("~startup_maneuver_enabled", rospy.get_param("startup_maneuver_enabled", True))
        )
        self.startup_forward1_distance_m = float(
            rospy.get_param("~startup_forward1_distance_m", rospy.get_param("startup_forward1_distance_m", 0.70))
        )
        self.startup_turn_angle_deg = float(
            rospy.get_param("~startup_turn_angle_deg", rospy.get_param("startup_turn_angle_deg", 60.0))
        )
        self.startup_forward2_distance_m = float(
            rospy.get_param("~startup_forward2_distance_m", rospy.get_param("startup_forward2_distance_m", 0.50))
        )
        self.startup_forward_speed = abs(
            float(rospy.get_param("~startup_forward_speed", rospy.get_param("startup_forward_speed", 0.12)))
        )
        self.startup_turn_angular_speed = abs(
            float(rospy.get_param("~startup_turn_angular_speed", rospy.get_param("startup_turn_angular_speed", 0.35)))
        )
        self.rightmost_line_only_duration = float(
            rospy.get_param("~rightmost_line_only_duration", rospy.get_param("rightmost_line_only_duration", 3.0))
        )

        self.finish_enable_delay = float(rospy.get_param("~finish_enable_delay", rospy.get_param("finish_enable_delay", 5.5)))
        self.finish_confirm_frames = int(
            rospy.get_param("~finish_confirm_frames", rospy.get_param("finish_confirm_frames", 8))
        )
        self.finish_release_frames = int(
            rospy.get_param("~finish_release_frames", rospy.get_param("finish_release_frames", 2))
        )
        self.finish_stop_time = float(
            rospy.get_param("~finish_stop_time", rospy.get_param("finish_stop_time", 1.0))
        )
        self.finish_bottom_roi_ratio = float(
            rospy.get_param("~finish_bottom_roi_ratio", rospy.get_param("finish_bottom_roi_ratio", 0.55))
        )
        self.finish_min_horizontal_width_ratio = float(
            rospy.get_param(
                "~finish_min_horizontal_width_ratio",
                rospy.get_param("finish_min_horizontal_width_ratio", 0.50),
            )
        )
        self.finish_min_horizontal_rows = int(
            rospy.get_param("~finish_min_horizontal_rows", rospy.get_param("finish_min_horizontal_rows", 4))
        )
        self.finish_min_vertical_height_ratio = float(
            rospy.get_param(
                "~finish_min_vertical_height_ratio",
                rospy.get_param("finish_min_vertical_height_ratio", 0.18),
            )
        )
        self.finish_min_box_width_ratio = float(
            rospy.get_param("~finish_min_box_width_ratio", rospy.get_param("finish_min_box_width_ratio", 0.55))
        )
        self.finish_min_bottom_y_ratio = float(
            rospy.get_param("~finish_min_bottom_y_ratio", rospy.get_param("finish_min_bottom_y_ratio", 0.82))
        )
        self.finish_approach_speed_scale = float(
            rospy.get_param("~finish_approach_speed_scale", rospy.get_param("finish_approach_speed_scale", 0.65))
        )
        self.finish_final_linear_speed = float(
            rospy.get_param("~finish_final_linear_speed", rospy.get_param("finish_final_linear_speed", 0.03))
        )

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.debug_info_pub = rospy.Publisher(self.debug_info_topic, String, queue_size=1)

        self.status = "searching" if self.started else "idle"
        self.start_time = time.time()
        self.last_detection_time = None
        self.last_lane_center = None
        self.last_error_px = 0.0
        self.single_line_frames = 0
        self.turn_until = self.start_time + self.startup_right_bias_duration if self.started else 0.0
        self.right_route_lock_until = self.start_time + self.right_route_lock_duration if self.started else 0.0
        self.startup_sequence_start = self.start_time
        self.startup_maneuver_done = not (self.started and self.startup_maneuver_enabled)
        self.rightmost_line_only_until = 0.0
        self.startup_phase = "none"
        self.fork_latch_until = 0.0
        self.last_fork_time = -1e9
        self.finish_frames = 0
        self.finish_lost_frames = 0
        self.finish_time = None
        self.finish_detection_enabled = False
        self.last_parking_result = ParkingBoxResult(False, None, 0.0, 0.0, 0.0, 0.0)
        self.last_target_center = None
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        self.last_route_locked = self.started
        self.last_two_sided = False

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.start_sub = rospy.Subscriber(self.start_topic, Bool, self.start_callback, queue_size=1)

        rospy.on_shutdown(self.stop_robot)
        self.publish_status(force=True)
        rospy.loginfo("right_line_follow started. image=%s cmd_vel=%s", self.image_topic, self.cmd_vel_topic)

    def get_float_list(self, name: str, default: Sequence[float]) -> List[float]:
        value = rospy.get_param("~" + name, rospy.get_param(name, list(default)))
        return [float(item) for item in value]

    def start_callback(self, msg: Bool):
        self.started = bool(msg.data)
        if self.started:
            self.start_time = time.time()
            self.finish_detection_enabled = False
            self.finish_frames = 0
            self.finish_lost_frames = 0
            self.finish_time = None
            self.turn_until = self.start_time + self.startup_right_bias_duration
            self.right_route_lock_until = self.start_time + self.right_route_lock_duration
            self.startup_sequence_start = self.start_time
            self.startup_maneuver_done = not self.startup_maneuver_enabled
            self.rightmost_line_only_until = 0.0
            self.startup_phase = "startup_forward1" if self.startup_maneuver_enabled else "none"
            self.pid.reset()
            self.set_status("searching")
        else:
            self.pid.reset()
            self.stop_robot()
            self.set_status("idle")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "cv_bridge conversion failed: %s", exc)
            return

        now = time.time()
        if self.finish_time is not None:
            self.handle_finish(now)
            return

        mask, roi_origin_y = self.extract_white_mask(frame)
        startup_active = self.handle_startup_maneuver(now)
        rightmost_only = self.is_rightmost_line_only(now)
        if not self.finish_detection_enabled:
            self.finish_detection_enabled = (
                self.startup_maneuver_done
                and not rightmost_only
                and (now - self.start_time) >= self.finish_enable_delay
            )
        route_locked = self.is_right_route_locked(now) or rightmost_only or startup_active
        observations = self.observe_lane(mask, frame.shape[1], self.selection_mode(route_locked, rightmost_only))
        self.update_lane_width_estimate(observations)
        lane_center = self.estimate_lane_center(observations, frame.shape[1])

        fork_rows = sum(1 for obs in observations if obs.multi_candidate)
        image_center = frame.shape[1] / 2.0
        lane_offset_ok = lane_center is not None and abs(lane_center - image_center) <= self.fork_center_tolerance_px
        fork_detected = fork_rows >= self.fork_candidate_count and lane_offset_ok
        if fork_detected:
            self.fork_latch_until = max(self.fork_latch_until, now + self.fork_latch_time)
            self.right_route_lock_until = max(self.right_route_lock_until, now + self.right_route_relock_duration)
        fork_detected_latched = now < self.fork_latch_until
        route_locked = self.is_right_route_locked(now) or rightmost_only or startup_active
        if route_locked and not any(obs.selection.startswith("right") for obs in observations):
            observations = self.observe_lane(mask, frame.shape[1], self.selection_mode(route_locked, rightmost_only))
            self.update_lane_width_estimate(observations)
            lane_center = self.estimate_lane_center(observations, frame.shape[1])
            fork_rows = sum(1 for obs in observations if obs.multi_candidate)

        if fork_detected_latched and (now - self.last_fork_time) >= self.fork_cooldown_sec:
            self.turn_until = max(self.turn_until, now + self.turn_hold_time)
            self.last_fork_time = now

        parking_result = self.detect_parking_box(mask)
        self.last_parking_result = parking_result
        if self.finish_detection_enabled and parking_result.detected:
            self.finish_frames += 1
            self.finish_lost_frames = 0
        else:
            self.finish_lost_frames += 1
            if self.finish_lost_frames >= self.finish_release_frames:
                self.finish_frames = 0
                self.finish_lost_frames = 0

        if self.finish_frames >= self.finish_confirm_frames:
            rospy.loginfo(
                "P1 parking box reached: frames=%d width=%.2f vl=%.2f vr=%.2f bottom=%.2f",
                self.finish_frames,
                parking_result.horizontal_width_ratio,
                parking_result.vertical_left_height_ratio,
                parking_result.vertical_right_height_ratio,
                parking_result.bottom_y_ratio,
            )
            self.finish_time = now
            self.set_status("finish_stop")
            self.hard_stop_robot()
            self.publish_debug_image(frame, mask, roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected_latched, now)
            self.publish_status()
            return

        if startup_active:
            self.last_target_center = lane_center
            self.last_route_locked = route_locked
            self.last_two_sided = any(obs.left_x is not None and obs.right_x is not None for obs in observations)
            self.publish_debug_image(frame, mask, roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected_latched, now)
            self.publish_debug_info(now, lane_center, parking_result, fork_rows, fork_detected_latched, route_locked)
            self.publish_status()
            return

        if not self.started:
            self.stop_robot()
            self.publish_debug_image(frame, mask, roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected_latched, now)
            self.publish_status()
            return

        if lane_center is None:
            self.single_line_frames += 1
            self.last_target_center = None
            self.last_route_locked = route_locked
            self.last_two_sided = False
            self.handle_lost_or_search(now)
        else:
            self.last_detection_time = now
            self.last_lane_center = lane_center
            two_sided = any(obs.left_x is not None and obs.right_x is not None for obs in observations)
            self.last_two_sided = two_sided
            if two_sided:
                self.single_line_frames = 0
            else:
                self.single_line_frames += 1

            target_center = lane_center
            if now < self.turn_until:
                target_center += self.right_turn_bias_px
                self.set_status("turn_right")
            elif route_locked:
                self.set_status("rightmost_line_only" if rightmost_only else "right_route_lock")
            elif not two_sided and self.single_line_frames > self.single_line_hold_frames:
                self.set_status("searching")
            elif self.finish_frames > 0:
                self.set_status("parking_approach")
            else:
                self.set_status("tracking")

            if now - self.start_time < self.startup_right_bias_duration:
                target_center += self.startup_right_bias_px
                self.set_status("turn_right")

            self.last_target_center = target_center
            self.last_route_locked = route_locked
            self.publish_control(target_center, frame.shape[1], now, two_sided)

        self.publish_debug_image(frame, mask, roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected_latched, now)
        self.publish_debug_info(now, lane_center, parking_result, fork_rows, fork_detected_latched, route_locked)
        self.publish_status()

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

    def observe_lane(self, mask: np.ndarray, image_width: int, selection_mode: str) -> List[RowObservation]:
        observations = []
        roi_height = mask.shape[0]
        for ratio in self.scan_row_ratios:
            y = int(max(0, min(roi_height - 1, roi_height * ratio)))
            segments = self.find_segments(mask[y, :])
            left_x, right_x, center_x, multi_candidate, selection = self.choose_right_lane_pair(
                segments, image_width, selection_mode
            )
            observations.append(RowObservation(y, segments, left_x, right_x, center_x, multi_candidate, selection))
        return observations

    def find_segments(self, row: np.ndarray) -> List[Segment]:
        active = row > 0
        segments = []
        start = None
        for idx, value in enumerate(active):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                self.append_segment(segments, start, idx - 1)
                start = None
        if start is not None:
            self.append_segment(segments, start, len(active) - 1)
        return self.merge_close_segments(segments)

    def append_segment(self, segments: List[Segment], start: int, end: int):
        width = end - start + 1
        if width >= self.min_line_width_px:
            segments.append(Segment(start, end, (start + end) / 2.0, width))

    def merge_close_segments(self, segments: List[Segment]) -> List[Segment]:
        if not segments:
            return []
        merged = [segments[0]]
        for segment in segments[1:]:
            previous = merged[-1]
            if segment.left - previous.right <= self.min_segment_gap_px:
                left = previous.left
                right = segment.right
                merged[-1] = Segment(left, right, (left + right) / 2.0, right - left + 1)
            else:
                merged.append(segment)
        return merged

    def choose_right_lane_pair(
        self, segments: List[Segment], image_width: int, selection_mode: str
    ) -> Tuple[Optional[float], Optional[float], Optional[float], bool, str]:
        lane_width_px = self.current_lane_width_px()
        if selection_mode == "rightmost_line" and segments:
            segment = segments[-1]
            center = segment.center - lane_width_px / 2.0
            multi_candidate = len(segments) >= self.fork_candidate_count
            return None, segment.center, center, multi_candidate, "rightmost_line"

        if len(segments) >= 2:
            multi_candidate = len(segments) >= self.fork_candidate_count
            if multi_candidate or selection_mode == "right_route":
                left, right = self.best_right_route_pair(segments)
                selection = "right_pair_fork" if multi_candidate else "right_pair_lock"
            else:
                left, right = self.best_pair_near_image_center(segments, image_width)
                selection = "center_pair"
            return left.center, right.center, (left.center + right.center) / 2.0, multi_candidate, selection

        if len(segments) == 1:
            segment = segments[0]
            if segment.center < image_width / 2.0:
                center = segment.center + lane_width_px / 2.0
                return segment.center, None, center, False, "single_left_border"
            center = segment.center - lane_width_px / 2.0
            return None, segment.center, center, False, "single_right_border"

        return None, None, None, False, "none"

    def best_right_route_pair(self, segments: List[Segment]) -> Tuple[Segment, Segment]:
        lane_width_px = self.current_lane_width_px()
        min_width = max(self.lane_width_px_min * 0.65, lane_width_px * 0.45)
        max_width = min(self.lane_width_px_max * self.right_route_pair_width_slack, lane_width_px * 1.8)
        candidates = []
        for left, right in zip(segments, segments[1:]):
            width = right.center - left.center
            center = (left.center + right.center) / 2.0
            if min_width <= width <= max_width:
                candidates.append((center, left, right))
        if candidates:
            _, left, right = max(candidates, key=lambda item: item[0])
            return left, right
        return segments[-2], segments[-1]

    def best_pair_near_image_center(self, segments: List[Segment], image_width: int) -> Tuple[Segment, Segment]:
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
        if self.enable_lane_width_adapt:
            return self.estimated_lane_width_px
        return self.lane_width_px

    def update_lane_width_estimate(self, observations: Sequence[RowObservation]):
        if not self.enable_lane_width_adapt:
            self.estimated_lane_width_px = self.lane_width_px
            return

        samples = []
        for obs in observations:
            if obs.left_x is None or obs.right_x is None or obs.multi_candidate:
                continue
            width = obs.right_x - obs.left_x
            if self.lane_width_px_min <= width <= self.lane_width_px_max:
                samples.append(width)
        if not samples:
            return

        sample_mean = float(np.mean(samples))
        alpha = max(0.0, min(1.0, self.lane_width_adapt_alpha))
        updated = (1.0 - alpha) * self.estimated_lane_width_px + alpha * sample_mean
        self.estimated_lane_width_px = max(self.lane_width_px_min, min(self.lane_width_px_max, updated))

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
        center = float(np.average(np.array(centers), weights=np.array(weights)))
        return max(0.0, min(float(image_width - 1), center))

    def detect_parking_box(self, mask: np.ndarray) -> ParkingBoxResult:
        height, width = mask.shape[:2]
        y0 = int(height * self.finish_bottom_roi_ratio)
        y0 = max(0, min(height - 1, y0))
        bottom = mask[y0:, :]
        if bottom.size == 0:
            return ParkingBoxResult(False, None, 0.0, 0.0, 0.0, 0.0)

        horizontal_width_ratio = 0.0
        horizontal_rows = 0
        best_span = None
        for row_index in range(bottom.shape[0]):
            xs = np.flatnonzero(bottom[row_index, :] > 0)
            if xs.size == 0:
                continue
            span = int(xs[-1] - xs[0] + 1)
            ratio = span / float(width)
            if ratio > horizontal_width_ratio:
                horizontal_width_ratio = ratio
                best_span = (int(xs[0]), int(xs[-1]), y0 + row_index)
            if ratio >= self.finish_min_horizontal_width_ratio:
                horizontal_rows += 1

        contour_result = cv2.findContours(bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
        best_box = None
        best_box_score = -1.0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_width_ratio = w / float(width)
            box_height_ratio = h / float(height)
            bottom_y_ratio = (y0 + y + h) / float(height)
            if box_width_ratio < self.finish_min_box_width_ratio:
                continue
            if bottom_y_ratio < self.finish_min_bottom_y_ratio:
                continue
            score = box_width_ratio + box_height_ratio + bottom_y_ratio
            if score > best_box_score:
                best_box = (x, y0 + y, w, h)
                best_box_score = score

        left_h_ratio = 0.0
        right_h_ratio = 0.0
        bottom_y_ratio = 0.0
        if best_span is not None:
            left_x, right_x, span_y = best_span
            margin = max(4, int(width * 0.025))
            left_slice = bottom[:, max(0, left_x - margin) : min(width, left_x + margin + 1)]
            right_slice = bottom[:, max(0, right_x - margin) : min(width, right_x + margin + 1)]
            left_h_ratio = self.vertical_presence_ratio(left_slice)
            right_h_ratio = self.vertical_presence_ratio(right_slice)
            bottom_y_ratio = span_y / float(height)

        if best_box is not None:
            _, box_y, _, box_h = best_box
            bottom_y_ratio = max(bottom_y_ratio, (box_y + box_h) / float(height))

        detected = (
            horizontal_rows >= self.finish_min_horizontal_rows
            and horizontal_width_ratio >= self.finish_min_horizontal_width_ratio
            and bottom_y_ratio >= self.finish_min_bottom_y_ratio
            and (
                best_box is not None
                or min(left_h_ratio, right_h_ratio) >= self.finish_min_vertical_height_ratio
            )
        )
        return ParkingBoxResult(detected, best_box, horizontal_width_ratio, left_h_ratio, right_h_ratio, bottom_y_ratio)

    def vertical_presence_ratio(self, image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        row_hits = np.any(image > 0, axis=1)
        if not np.any(row_hits):
            return 0.0
        ys = np.flatnonzero(row_hits)
        return float(ys[-1] - ys[0] + 1) / float(image.shape[0])

    def publish_control(self, lane_center: float, image_width: int, now: float, two_sided_tracking: bool):
        image_center = image_width / 2.0
        error = lane_center - image_center
        self.last_error_px = error
        angular = -self.pid.update(error, now)
        angular = max(-self.max_angular_speed, min(self.max_angular_speed, angular))

        if now < self.turn_until:
            linear = self.turn_linear_speed
        elif not two_sided_tracking and self.single_line_frames > self.single_line_hold_frames:
            linear = self.search_linear_speed
        else:
            slowdown = min(abs(error) / max(self.error_slowdown_px, 1.0), 1.0)
            linear = self.base_linear_speed - slowdown * (self.base_linear_speed - self.min_linear_speed)
            linear = max(self.min_linear_speed, min(self.base_linear_speed, linear))

        if self.finish_frames > 0:
            linear *= max(0.2, min(1.0, self.finish_approach_speed_scale))
            if self.finish_frames >= max(1, self.finish_confirm_frames - 2):
                linear = min(linear, self.finish_final_linear_speed)

        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.last_cmd_linear = linear
        self.last_cmd_angular = angular
        self.cmd_pub.publish(twist)

    def handle_lost_or_search(self, now: float):
        if self.last_detection_time is not None and now - self.last_detection_time > self.lost_timeout:
            if self.stop_on_lost:
                self.pid.reset()
                self.stop_robot()
                self.set_status("lost")
                return

        self.set_status("searching")
        twist = Twist()
        twist.linear.x = self.search_linear_speed
        if abs(self.last_error_px) > 1.0:
            direction = -1.0 if self.last_error_px > 0.0 else 1.0
        else:
            direction = -1.0
        twist.angular.z = direction * self.search_angular_speed
        self.last_cmd_linear = twist.linear.x
        self.last_cmd_angular = twist.angular.z
        self.cmd_pub.publish(twist)

    def handle_finish(self, now: float):
        self.hard_stop_robot()
        if now - self.finish_time >= self.finish_stop_time:
            self.set_status("finish")
        else:
            self.set_status("finish_stop")
        self.publish_status()

    def handle_startup_maneuver(self, now: float) -> bool:
        if not self.started or self.startup_maneuver_done:
            return False

        forward_speed = max(self.startup_forward_speed, 1e-3)
        turn_speed = max(self.startup_turn_angular_speed, 1e-3)
        forward1_duration = max(0.0, self.startup_forward1_distance_m) / forward_speed
        turn_duration = math.radians(max(0.0, self.startup_turn_angle_deg)) / turn_speed
        forward2_duration = max(0.0, self.startup_forward2_distance_m) / forward_speed

        elapsed = now - self.startup_sequence_start
        twist = Twist()

        if elapsed < forward1_duration:
            self.startup_phase = "startup_forward1"
            self.set_status(self.startup_phase)
            twist.linear.x = forward_speed
        elif elapsed < forward1_duration + turn_duration:
            self.startup_phase = "startup_turn_right_60"
            self.set_status(self.startup_phase)
            twist.angular.z = -turn_speed
        elif elapsed < forward1_duration + turn_duration + forward2_duration:
            self.startup_phase = "startup_forward2"
            self.set_status(self.startup_phase)
            twist.linear.x = forward_speed
        else:
            self.startup_maneuver_done = True
            self.startup_phase = "rightmost_line_only"
            self.rightmost_line_only_until = now + max(0.0, self.rightmost_line_only_duration)
            self.right_route_lock_until = max(self.right_route_lock_until, self.rightmost_line_only_until)
            self.pid.reset()
            self.hard_stop_robot()
            rospy.loginfo(
                "startup maneuver finished: forward1=%.2fm turn=%.1fdeg forward2=%.2fm rightmost_only=%.2fs",
                self.startup_forward1_distance_m,
                self.startup_turn_angle_deg,
                self.startup_forward2_distance_m,
                self.rightmost_line_only_duration,
            )
            return False

        self.last_cmd_linear = twist.linear.x
        self.last_cmd_angular = twist.angular.z
        self.cmd_pub.publish(twist)
        return True

    def publish_debug_image(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        roi_origin_y: int,
        observations: Sequence[RowObservation],
        lane_center: Optional[float],
        parking_result: ParkingBoxResult,
        fork_rows: int,
        fork_detected: bool,
        now: float,
    ):
        if not self.publish_debug:
            return

        debug = frame.copy()
        height, width = debug.shape[:2]
        image_center = width // 2
        cv2.line(debug, (image_center, roi_origin_y), (image_center, height - 1), (255, 0, 0), 1)

        for obs in observations:
            y = roi_origin_y + obs.y
            cv2.line(debug, (0, y), (width - 1, y), (45, 45, 45), 1)
            for segment in obs.segments:
                cv2.line(debug, (segment.left, y), (segment.right, y), (0, 255, 255), 2)
            if obs.left_x is not None:
                cv2.circle(debug, (int(obs.left_x), y), 4, (255, 80, 80), -1)
            if obs.right_x is not None:
                cv2.circle(debug, (int(obs.right_x), y), 4, (80, 80, 255), -1)
            if obs.center_x is not None:
                cv2.circle(debug, (int(obs.center_x), y), 5, (0, 0, 255), -1)

        if lane_center is not None:
            cv2.line(debug, (int(lane_center), roi_origin_y), (int(lane_center), height - 1), (0, 0, 255), 2)

        if parking_result.box is not None:
            x, y, w, h = parking_result.box
            cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 120, 0), 2)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (width // 3, max(1, mask_bgr.shape[0] // 3)))
        mh, mw = mask_bgr.shape[:2]
        debug[0:mh, 0:mw] = mask_bgr

        turn_sec = max(0.0, self.turn_until - now)
        text = "status={} finish_frames={} enabled={} fork_rows={} fork={} turn_right={:.2f}s".format(
            self.status,
            self.finish_frames,
            int(self.finish_detection_enabled),
            fork_rows,
            int(fork_detected),
            turn_sec,
        )
        cv2.putText(debug, text, (10, max(mh + 25, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
        route_sec = max(0.0, self.right_route_lock_until - now)
        target = -1.0 if self.last_target_center is None else self.last_target_center
        rightmost_sec = max(0.0, self.rightmost_line_only_until - now)
        text2 = "phase={} route_lock={:.2f}s rightmost={:.2f}s target={:.1f} err={:.1f} cmd=({:.2f},{:.2f})".format(
            self.startup_phase,
            route_sec,
            rightmost_sec,
            target,
            self.last_error_px,
            self.last_cmd_linear,
            self.last_cmd_angular,
        )
        cv2.putText(debug, text2, (10, max(mh + 50, 55)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)
        text3 = "park_w={:.2f} vl={:.2f} vr={:.2f} bottom={:.2f} lane_w={:.1f}px sel={}".format(
            parking_result.horizontal_width_ratio,
            parking_result.vertical_left_height_ratio,
            parking_result.vertical_right_height_ratio,
            parking_result.bottom_y_ratio,
            self.current_lane_width_px(),
            self.selection_summary(observations),
        )
        cv2.putText(debug, text3, (10, max(mh + 75, 80)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)

        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "debug image conversion failed: %s", exc)

    def set_status(self, status: str):
        if self.status != status:
            self.status = status
            self.publish_status(force=True)

    def publish_status(self, force: bool = False):
        if force:
            self.status_pub.publish(String(data=self.status))
            return
        self.status_pub.publish(String(data=self.status))

    def stop_robot(self):
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        self.cmd_pub.publish(Twist())

    def hard_stop_robot(self):
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        for _ in range(4):
            self.cmd_pub.publish(Twist())

    def is_right_route_locked(self, now: float) -> bool:
        return now < self.right_route_lock_until

    def is_rightmost_line_only(self, now: float) -> bool:
        return self.startup_maneuver_done and now < self.rightmost_line_only_until

    def selection_mode(self, route_locked: bool, rightmost_only: bool) -> str:
        if rightmost_only:
            return "rightmost_line"
        if route_locked:
            return "right_route"
        return "normal"

    def selection_summary(self, observations: Sequence[RowObservation]) -> str:
        counts = {}
        for obs in observations:
            counts[obs.selection] = counts.get(obs.selection, 0) + 1
        if not counts:
            return "none"
        return ",".join("{}:{}".format(key, counts[key]) for key in sorted(counts.keys()))

    def publish_debug_info(
        self,
        now: float,
        lane_center: Optional[float],
        parking_result: ParkingBoxResult,
        fork_rows: int,
        fork_detected: bool,
        route_locked: bool,
    ):
        target = None if self.last_target_center is None else round(float(self.last_target_center), 2)
        lane = None if lane_center is None else round(float(lane_center), 2)
        msg = (
            "status={status} startup_phase={startup_phase} route_locked={route_locked} "
            "route_lock_left={route_left:.2f} rightmost_left={rightmost_left:.2f} "
            "lane_center={lane_center} target_center={target_center} error_px={error:.2f} "
            "cmd_linear={linear:.3f} cmd_angular={angular:.3f} two_sided={two_sided} "
            "fork_rows={fork_rows} fork={fork} finish_frames={finish_frames} "
            "parking_detected={parking_detected} parking_width={parking_width:.2f} "
            "parking_bottom={parking_bottom:.2f}"
        ).format(
            status=self.status,
            startup_phase=self.startup_phase,
            route_locked=int(route_locked),
            route_left=max(0.0, self.right_route_lock_until - now),
            rightmost_left=max(0.0, self.rightmost_line_only_until - now),
            lane_center=lane,
            target_center=target,
            error=self.last_error_px,
            linear=self.last_cmd_linear,
            angular=self.last_cmd_angular,
            two_sided=int(self.last_two_sided),
            fork_rows=fork_rows,
            fork=int(fork_detected),
            finish_frames=self.finish_frames,
            parking_detected=int(parking_result.detected),
            parking_width=parking_result.horizontal_width_ratio,
            parking_bottom=parking_result.bottom_y_ratio,
        )
        self.debug_info_pub.publish(String(data=msg))
        rospy.loginfo_throttle(0.5, "right_line_debug: %s", msg)


def main():
    rospy.init_node("right_line_follow_node")
    RightLineFollowNode()
    rospy.spin()


if __name__ == "__main__":
    main()
