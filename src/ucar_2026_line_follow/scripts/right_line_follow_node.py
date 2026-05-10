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
    horizontal_rows: int = 0
    horizontal_left_x_ratio: float = 0.0
    horizontal_right_x_ratio: float = 0.0
    full_box_detected: bool = False
    stop_pose_detected: bool = False
    closed_shape_detected: bool = False
    closed_shape_box: Optional[Tuple[int, int, int, int]] = None
    closed_shape_score: float = 0.0
    closed_top_ratio: float = 0.0
    closed_bottom_ratio: float = 0.0
    closed_left_ratio: float = 0.0
    closed_right_ratio: float = 0.0


@dataclass
class SeniorFollowResult:
    found: bool
    target_x: Optional[float]
    target_y: Optional[float]
    error: float
    linear_x: float
    linear_y: float
    angular_z: float
    right_count: int
    left_count: int
    path: List[Tuple[float, float]]


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


class SeniorRightLineTracker:
    RESULT_ROW = 480
    RESULT_COL = 640
    POINTS_MAX_LEN = 300
    DIR_FRONT = ((0, -1), (1, 0), (0, 1), (-1, 0))
    DIR_FRONTLEFT = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    DIR_FRONTRIGHT = ((1, -1), (1, 1), (-1, 1), (-1, -1))
    CHANGE_UN_MAT = np.array(
        [
            [-2.897018, 2.446196, -388.368977],
            [-0.061836, 1.194630, -756.140464],
            [-0.000272, 0.008324, -4.335235],
        ],
        dtype=np.float32,
    )

    def __init__(self):
        self.inv_mat = np.linalg.inv(self.CHANGE_UN_MAT)
        self.begin_x = float(rospy.get_param("~senior_begin_x", rospy.get_param("senior_begin_x", 25.0)))
        self.begin_y = float(rospy.get_param("~senior_begin_y", rospy.get_param("senior_begin_y", 400.0)))
        self.thres = float(rospy.get_param("~senior_thres", rospy.get_param("senior_thres", 30.0)))
        self.block_size = int(rospy.get_param("~senior_block_size", rospy.get_param("senior_block_size", 7)))
        if self.block_size % 2 == 0:
            self.block_size += 1
        self.clip_value = float(rospy.get_param("~senior_clip_value", rospy.get_param("senior_clip_value", 1.0)))
        self.line_blur_kernel = int(
            rospy.get_param("~senior_line_blur_kernel", rospy.get_param("senior_line_blur_kernel", 7))
        )
        if self.line_blur_kernel % 2 == 0:
            self.line_blur_kernel += 1
        self.pixel_per_meter = float(
            rospy.get_param("~senior_pixel_per_meter", rospy.get_param("senior_pixel_per_meter", 500.0))
        )
        self.road_width_m = float(rospy.get_param("~senior_road_width_m", rospy.get_param("senior_road_width_m", 0.36)))
        self.sample_dist_m = float(
            rospy.get_param("~senior_sample_dist_m", rospy.get_param("senior_sample_dist_m", 0.01))
        )
        self.aim_dist_m = float(rospy.get_param("~senior_aim_dist_m", rospy.get_param("senior_aim_dist_m", 0.10)))
        self.forward_bias_m = float(
            rospy.get_param("~senior_forward_bias_m", rospy.get_param("senior_forward_bias_m", 0.20))
        )
        self.base_speed = float(rospy.get_param("~senior_base_speed", rospy.get_param("senior_base_speed", 0.24)))
        self.speed_error_scale = float(
            rospy.get_param("~senior_speed_error_scale", rospy.get_param("senior_speed_error_scale", 0.24))
        )
        self.min_speed = float(rospy.get_param("~senior_min_speed", rospy.get_param("senior_min_speed", 0.08)))
        self.max_speed = float(rospy.get_param("~senior_max_speed", rospy.get_param("senior_max_speed", 0.28)))
        self.max_angular = float(
            rospy.get_param("~senior_max_angular_speed", rospy.get_param("senior_max_angular_speed", 0.75))
        )
        self.lost_speed = float(rospy.get_param("~senior_lost_speed", rospy.get_param("senior_lost_speed", 0.12)))
        self.lost_error = float(rospy.get_param("~senior_lost_error", rospy.get_param("senior_lost_error", -0.30)))
        self.lost_lateral_speed = float(
            rospy.get_param("~senior_lost_lateral_speed", rospy.get_param("senior_lost_lateral_speed", -0.05))
        )

    def compute(self, frame: np.ndarray) -> SeniorFollowResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[:2] != (self.RESULT_ROW, self.RESULT_COL):
            gray = cv2.resize(gray, (self.RESULT_COL, self.RESULT_ROW))
        img = 255 - gray
        left_path, right_path = self.process_image(img)
        path = right_path if right_path else left_path
        if not path:
            angular = max(-self.max_angular, min(self.max_angular, -self.lost_error))
            return SeniorFollowResult(
                False, None, None, self.lost_error, self.lost_speed, self.lost_lateral_speed,
                angular, len(right_path), len(left_path), []
            )

        aim_idx = max(0, min(int(round(self.aim_dist_m / self.sample_dist_m)), len(path) - 1))
        target_x, target_y = path[aim_idx]
        dx = target_x - self.RESULT_COL / 2.0
        dy = 490.0 - target_y + self.forward_bias_m * self.pixel_per_meter
        error = -math.atan2(dx, max(dy, 1e-3))
        linear = self.base_speed - abs(error) * self.speed_error_scale
        linear = max(self.min_speed, min(self.max_speed, linear))
        angular = max(-self.max_angular, min(self.max_angular, -error))
        return SeniorFollowResult(
            True, target_x, target_y, error, linear, 0.0, angular, len(right_path), len(left_path), path
        )

    def process_image(self, img: np.ndarray) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        left_raw = self.find_seed_and_trace_left(img)
        right_raw = self.find_seed_and_trace_right(img)
        left_path = self.make_center_path([self.map_point(x, y) for x, y in left_raw], is_right=False)
        right_path = self.make_center_path([self.map_point(x, y) for x, y in right_raw], is_right=True)
        return left_path, right_path

    def find_seed_and_trace_left(self, img: np.ndarray) -> List[Tuple[int, int]]:
        half = self.block_size // 2
        d = 6
        for idx in range(5):
            y = int(self.begin_y - idx * 25)
            for x in range(int(self.RESULT_COL / 2 - self.begin_x), 0, -1):
                local = 0.0
                for dy in range(-half, half + 1):
                    local += self.at(img, x + d, y + dy) - self.at(img, x - d, y + dy)
                if local / self.block_size >= self.thres:
                    return self.findline_adaptive(img, x - d, y, self.DIR_FRONTLEFT, left_hand=True)
        return []

    def find_seed_and_trace_right(self, img: np.ndarray) -> List[Tuple[int, int]]:
        half = self.block_size // 2
        d = 6
        for idx in range(5):
            y = int(self.begin_y - idx * 25)
            for x in range(int(self.RESULT_COL / 2 + self.begin_x), self.RESULT_COL - 1):
                local = 0.0
                for dy in range(-half, half + 1):
                    local -= self.at(img, x + d, y + dy) - self.at(img, x - d, y + dy)
                if local / self.block_size >= self.thres:
                    return self.findline_adaptive(img, x + d, y, self.DIR_FRONTRIGHT, left_hand=False)
        return []

    def findline_adaptive(
        self, img: np.ndarray, x: int, y: int, side_dirs: Sequence[Tuple[int, int]], left_hand: bool
    ) -> List[Tuple[int, int]]:
        half = self.block_size // 2
        step, direction, turn = 0, 0, 0
        pts: List[Tuple[int, int]] = []
        while step < self.POINTS_MAX_LEN and 0 < x < self.RESULT_COL - 1 and 0 < y < self.RESULT_ROW - 1 and turn < 4:
            local_thres = 0.0
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    local_thres += self.at(img, x + dx, y + dy)
            local_thres = local_thres / float(self.block_size * self.block_size) - self.clip_value

            fx, fy = self.DIR_FRONT[direction]
            sx, sy = side_dirs[direction]
            if self.at(img, x + fx, y + fy) < local_thres:
                direction = (direction + (1 if left_hand else 3)) % 4
                turn += 1
            elif self.at(img, x + sx, y + sy) < local_thres:
                x += fx
                y += fy
                pts.append((x, y))
                step += 1
                turn = 0
            else:
                x += sx
                y += sy
                direction = (direction + (3 if left_hand else 1)) % 4
                pts.append((x, y))
                step += 1
                turn = 0
        return pts

    def make_center_path(self, mapped: List[Tuple[float, float]], is_right: bool) -> List[Tuple[float, float]]:
        if len(mapped) < 3:
            return []
        blurred = self.blur_points(mapped, self.line_blur_kernel)
        sampled = self.resample_points(blurred, self.sample_dist_m * self.pixel_per_meter, self.POINTS_MAX_LEN)
        if len(sampled) < 3:
            return []
        approx_num = int(round(0.2 / self.sample_dist_m))
        dist = self.pixel_per_meter * self.road_width_m / 2.0
        tracked = self.track_rightline(sampled, approx_num, dist) if is_right else self.track_leftline(sampled, approx_num, dist)
        return self.resample_points(tracked, self.sample_dist_m * self.pixel_per_meter, self.POINTS_MAX_LEN)

    def track_leftline(self, pts: Sequence[Tuple[float, float]], approx_num: int, dist: float) -> List[Tuple[float, float]]:
        out = [(self.RESULT_COL / 2.0, self.RESULT_ROW + 50.0)]
        for i in range(1, len(pts)):
            dx, dy, dn = self.tangent(pts, i, approx_num)
            if dn > 1e-6:
                out.append((pts[i][0] - dy * dist, pts[i][1] + dx * dist))
        return out

    def track_rightline(self, pts: Sequence[Tuple[float, float]], approx_num: int, dist: float) -> List[Tuple[float, float]]:
        out = [(self.RESULT_COL / 2.0, self.RESULT_ROW + 50.0)]
        for i in range(1, len(pts)):
            dx, dy, dn = self.tangent(pts, i, approx_num)
            if dn > 1e-6:
                out.append((pts[i][0] + dy * dist, pts[i][1] - dx * dist))
        return out

    def tangent(self, pts: Sequence[Tuple[float, float]], index: int, approx_num: int) -> Tuple[float, float, float]:
        left = max(0, min(len(pts) - 1, index - approx_num))
        right = max(0, min(len(pts) - 1, index + approx_num))
        dx = pts[right][0] - pts[left][0]
        dy = pts[right][1] - pts[left][1]
        dn = math.hypot(dx, dy)
        if dn > 1e-6:
            dx /= dn
            dy /= dn
        return dx, dy, dn

    def blur_points(self, pts: Sequence[Tuple[float, float]], kernel: int) -> List[Tuple[float, float]]:
        half = kernel // 2
        denom = (2 * half + 2) * (half + 1) / 2.0
        out = []
        for i in range(len(pts)):
            sx = 0.0
            sy = 0.0
            for j in range(-half, half + 1):
                idx = max(0, min(len(pts) - 1, i + j))
                weight = half + 1 - abs(j)
                sx += pts[idx][0] * weight
                sy += pts[idx][1] * weight
            out.append((sx / denom, sy / denom))
        return out

    def resample_points(self, pts: Sequence[Tuple[float, float]], dist: float, max_len: int) -> List[Tuple[float, float]]:
        out = []
        remain = 0.0
        for i in range(len(pts) - 1):
            if len(out) >= max_len:
                break
            x0, y0 = pts[i]
            dx = pts[i + 1][0] - x0
            dy = pts[i + 1][1] - y0
            dn = math.hypot(dx, dy)
            if dn <= 1e-6:
                continue
            dx /= dn
            dy /= dn
            while remain < dn and len(out) < max_len:
                x0 += dx * remain
                y0 += dy * remain
                out.append((x0, y0))
                dn -= remain
                remain = dist
            remain -= dn
        return out

    def map_point(self, x: int, y: int) -> Tuple[float, float]:
        denom = self.inv_mat[2, 0] * x + self.inv_mat[2, 1] * y + self.inv_mat[2, 2]
        if abs(denom) <= 1e-6:
            return float(x), float(y)
        map_x = (self.inv_mat[0, 0] * x + self.inv_mat[0, 1] * y + self.inv_mat[0, 2]) / denom
        map_y = (self.inv_mat[1, 0] * x + self.inv_mat[1, 1] * y + self.inv_mat[1, 2]) / denom
        return float(map_x), float(map_y)

    def at(self, img: np.ndarray, x: int, y: int) -> int:
        x = max(0, min(self.RESULT_COL - 1, int(x)))
        y = max(0, min(self.RESULT_ROW - 1, int(y)))
        return int(img[y, x])


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
        self.max_lane_segment_width_px = int(
            rospy.get_param("~max_lane_segment_width_px", rospy.get_param("max_lane_segment_width_px", 90))
        )
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
            rospy.get_param("~startup_forward1_distance_m", rospy.get_param("startup_forward1_distance_m", 0.40))
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
        self.rightmost_line_only_speed = float(
            rospy.get_param("~rightmost_line_only_speed", rospy.get_param("rightmost_line_only_speed", 0.045))
        )
        self.rightmost_line_target_offset_px = float(
            rospy.get_param(
                "~rightmost_line_target_offset_px",
                rospy.get_param("rightmost_line_target_offset_px", 115.0),
            )
        )
        self.rightmost_line_target_offset_ratio = float(
            rospy.get_param(
                "~rightmost_line_target_offset_ratio",
                rospy.get_param("rightmost_line_target_offset_ratio", 0.50),
            )
        )
        self.rightmost_max_angular_speed = float(
            rospy.get_param("~rightmost_max_angular_speed", rospy.get_param("rightmost_max_angular_speed", 0.35))
        )
        self.post_rightmost_route_lock_duration = float(
            rospy.get_param(
                "~post_rightmost_route_lock_duration",
                rospy.get_param("post_rightmost_route_lock_duration", 8.0),
            )
        )

        self.finish_enable_delay = float(rospy.get_param("~finish_enable_delay", rospy.get_param("finish_enable_delay", 5.5)))
        self.finish_detection_start_delay = float(
            rospy.get_param(
                "~finish_detection_start_delay",
                rospy.get_param("finish_detection_start_delay", 15.0),
            )
        )
        self.finish_confirm_frames = int(
            rospy.get_param("~finish_confirm_frames", rospy.get_param("finish_confirm_frames", 8))
        )
        self.finish_release_frames = int(
            rospy.get_param("~finish_release_frames", rospy.get_param("finish_release_frames", 2))
        )
        self.finish_stop_time = float(
            rospy.get_param("~finish_stop_time", rospy.get_param("finish_stop_time", 1.0))
        )
        self.finish_auto_stop = bool(rospy.get_param("~finish_auto_stop", rospy.get_param("finish_auto_stop", True)))
        self.finish_forward_after_stop_enabled = bool(
            rospy.get_param(
                "~finish_forward_after_stop_enabled",
                rospy.get_param("finish_forward_after_stop_enabled", True),
            )
        )
        self.finish_forward_distance_m = float(
            rospy.get_param("~finish_forward_distance_m", rospy.get_param("finish_forward_distance_m", 0.50))
        )
        self.finish_forward_speed = abs(
            float(rospy.get_param("~finish_forward_speed", rospy.get_param("finish_forward_speed", 0.10)))
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
        self.finish_require_vertical_sides = bool(
            rospy.get_param(
                "~finish_require_vertical_sides",
                rospy.get_param("finish_require_vertical_sides", True),
            )
        )
        self.finish_use_full_box_stop = bool(
            rospy.get_param("~finish_use_full_box_stop", rospy.get_param("finish_use_full_box_stop", False))
        )
        self.finish_closed_shape_enabled = bool(
            rospy.get_param("~finish_closed_shape_enabled", rospy.get_param("finish_closed_shape_enabled", True))
        )
        self.finish_closed_ignore_route_lock = bool(
            rospy.get_param(
                "~finish_closed_ignore_route_lock",
                rospy.get_param("finish_closed_ignore_route_lock", True),
            )
        )
        self.finish_closed_instant_stop = bool(
            rospy.get_param(
                "~finish_closed_instant_stop",
                rospy.get_param("finish_closed_instant_stop", True),
            )
        )
        self.finish_closed_min_width_ratio = float(
            rospy.get_param(
                "~finish_closed_min_width_ratio",
                rospy.get_param("finish_closed_min_width_ratio", 0.35),
            )
        )
        self.finish_closed_min_height_ratio = float(
            rospy.get_param(
                "~finish_closed_min_height_ratio",
                rospy.get_param("finish_closed_min_height_ratio", 0.25),
            )
        )
        self.finish_closed_min_horizontal_presence = float(
            rospy.get_param(
                "~finish_closed_min_horizontal_presence",
                rospy.get_param("finish_closed_min_horizontal_presence", 0.50),
            )
        )
        self.finish_closed_min_vertical_presence = float(
            rospy.get_param(
                "~finish_closed_min_vertical_presence",
                rospy.get_param("finish_closed_min_vertical_presence", 0.45),
            )
        )
        self.finish_closed_band_ratio = float(
            rospy.get_param("~finish_closed_band_ratio", rospy.get_param("finish_closed_band_ratio", 0.14))
        )
        self.finish_closed_morph_kernel = int(
            rospy.get_param("~finish_closed_morph_kernel", rospy.get_param("finish_closed_morph_kernel", 11))
        )
        self.finish_stop_min_horizontal_width_ratio = float(
            rospy.get_param(
                "~finish_stop_min_horizontal_width_ratio",
                rospy.get_param("finish_stop_min_horizontal_width_ratio", 0.62),
            )
        )
        self.finish_stop_max_horizontal_width_ratio = float(
            rospy.get_param(
                "~finish_stop_max_horizontal_width_ratio",
                rospy.get_param("finish_stop_max_horizontal_width_ratio", 0.88),
            )
        )
        self.finish_stop_min_horizontal_rows = int(
            rospy.get_param(
                "~finish_stop_min_horizontal_rows",
                rospy.get_param("finish_stop_min_horizontal_rows", 5),
            )
        )
        self.finish_stop_max_horizontal_rows = int(
            rospy.get_param(
                "~finish_stop_max_horizontal_rows",
                rospy.get_param("finish_stop_max_horizontal_rows", 40),
            )
        )
        self.finish_stop_min_right_edge_ratio = float(
            rospy.get_param(
                "~finish_stop_min_right_edge_ratio",
                rospy.get_param("finish_stop_min_right_edge_ratio", 0.90),
            )
        )
        self.finish_stop_bottom_y_min_ratio = float(
            rospy.get_param(
                "~finish_stop_bottom_y_min_ratio",
                rospy.get_param("finish_stop_bottom_y_min_ratio", 0.75),
            )
        )
        self.finish_stop_bottom_y_max_ratio = float(
            rospy.get_param(
                "~finish_stop_bottom_y_max_ratio",
                rospy.get_param("finish_stop_bottom_y_max_ratio", 0.88),
            )
        )
        self.finish_approach_speed_scale = float(
            rospy.get_param("~finish_approach_speed_scale", rospy.get_param("finish_approach_speed_scale", 0.65))
        )
        self.finish_final_linear_speed = float(
            rospy.get_param("~finish_final_linear_speed", rospy.get_param("finish_final_linear_speed", 0.03))
        )
        self.use_senior_follow_after_entry = bool(
            rospy.get_param(
                "~use_senior_follow_after_entry",
                rospy.get_param("use_senior_follow_after_entry", True),
            )
        )
        self.senior_tracker = SeniorRightLineTracker()

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
        self.startup_maneuver_done_time = self.start_time if self.startup_maneuver_done else None
        self.startup_phase = "none"
        self.fork_latch_until = 0.0
        self.last_fork_time = -1e9
        self.finish_frames = 0
        self.finish_lost_frames = 0
        self.finish_time = None
        self.finish_detection_enabled = False
        self.finish_forward_active = False
        self.finish_forward_start_time = None
        self.finish_forward_duration = 0.0
        self.last_parking_result = ParkingBoxResult(False, None, 0.0, 0.0, 0.0, 0.0)
        self.last_target_center = None
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        self.last_route_locked = self.started
        self.last_two_sided = False
        self.last_senior_path: List[Tuple[float, float]] = []
        self.last_debug_snapshot = {}

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
            self.reset_finish_forward()
            self.turn_until = self.start_time + self.startup_right_bias_duration
            self.right_route_lock_until = self.start_time + self.right_route_lock_duration
            self.startup_sequence_start = self.start_time
            self.startup_maneuver_done = not self.startup_maneuver_enabled
            self.rightmost_line_only_until = 0.0
            self.startup_maneuver_done_time = self.start_time if self.startup_maneuver_done else None
            self.startup_phase = "startup_forward1" if self.startup_maneuver_enabled else "none"
            self.pid.reset()
            self.set_status("searching")
        else:
            self.reset_finish_forward()
            self.pid.reset()
            self.stop_robot()
            self.set_status("idle")

    def reset_finish_forward(self):
        self.finish_forward_active = False
        self.finish_forward_start_time = None
        self.finish_forward_duration = 0.0

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
        if self.finish_forward_active:
            self.handle_finish_forward(now)
            return

        mask, roi_origin_y = self.extract_white_mask(frame)
        startup_active = self.handle_startup_maneuver(now)
        rightmost_only = self.is_rightmost_line_only(now)
        if not self.finish_detection_enabled:
            self.finish_detection_enabled = (
                self.startup_maneuver_done
                and self.startup_maneuver_done_time is not None
                and (now - self.startup_maneuver_done_time) >= self.finish_enable_delay
                and not rightmost_only
                and not self.is_right_route_locked(now)
            )
        route_locked = self.is_right_route_locked(now) or rightmost_only or startup_active
        observations = self.observe_lane(mask, frame.shape[1], self.selection_mode(route_locked, rightmost_only))
        self.update_lane_width_estimate(observations)
        lane_center = self.estimate_lane_center(observations, frame.shape[1])

        fork_rows = sum(1 for obs in observations if obs.multi_candidate)
        image_center = frame.shape[1] / 2.0
        lane_offset_ok = lane_center is not None and abs(lane_center - image_center) <= self.fork_center_tolerance_px
        fork_detected = fork_rows >= self.fork_candidate_count and lane_offset_ok
        if fork_detected and not startup_active and not rightmost_only and not self.is_right_route_locked(now):
            self.fork_latch_until = max(self.fork_latch_until, now + self.fork_latch_time)
            self.right_route_lock_until = max(self.right_route_lock_until, now + self.right_route_relock_duration)
        fork_detected_latched = now < self.fork_latch_until
        route_locked = self.is_right_route_locked(now) or rightmost_only or startup_active
        if route_locked and not any(obs.selection.startswith("right") for obs in observations):
            observations = self.observe_lane(mask, frame.shape[1], self.selection_mode(route_locked, rightmost_only))
            self.update_lane_width_estimate(observations)
            lane_center = self.estimate_lane_center(observations, frame.shape[1])
            fork_rows = sum(1 for obs in observations if obs.multi_candidate)

        if (
            fork_detected_latched
            and not rightmost_only
            and not route_locked
            and (now - self.last_fork_time) >= self.fork_cooldown_sec
        ):
            self.turn_until = max(self.turn_until, now + self.turn_hold_time)
            self.last_fork_time = now

        parking_result = self.detect_parking_box(mask)
        self.last_parking_result = parking_result
        parking_should_count = self.should_count_finish_detection(parking_result, route_locked)
        if parking_should_count:
            self.finish_frames += 1
            self.finish_lost_frames = 0
        else:
            self.finish_lost_frames += 1
            if self.finish_lost_frames >= self.finish_release_frames:
                self.finish_frames = 0
                self.finish_lost_frames = 0

        if self.finish_frames >= self.finish_confirm_frames:
            rospy.loginfo_throttle(
                0.5,
                "P1 parking box reached: frames=%d width=%.2f vl=%.2f vr=%.2f bottom=%.2f auto_stop=%d",
                self.finish_frames,
                parking_result.horizontal_width_ratio,
                parking_result.vertical_left_height_ratio,
                parking_result.vertical_right_height_ratio,
                parking_result.bottom_y_ratio,
                int(self.finish_auto_stop),
            )
            if self.finish_auto_stop:
                if self.finish_forward_after_stop_enabled and self.finish_forward_distance_m > 0.0:
                    self.start_finish_forward(now)
                else:
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

        if self.use_senior_follow_after_entry and self.startup_maneuver_done and not rightmost_only:
            self.handle_senior_follow(frame, mask, roi_origin_y, parking_result, fork_rows, fork_detected_latched, route_locked, now)
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
            if rightmost_only:
                self.set_status("rightmost_line_only")
            elif now < self.turn_until and not route_locked:
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
        lane_segments = [segment for segment in segments if segment.width <= self.max_lane_segment_width_px]
        if lane_segments:
            segments = lane_segments
        lane_width_px = self.current_lane_width_px()
        if selection_mode == "rightmost_line" and segments:
            segment = segments[-1]
            center = self.center_from_right_boundary(segment.center)
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
            center = self.center_from_right_boundary(segment.center)
            return None, segment.center, center, False, "single_right_border"

        return None, None, None, False, "none"

    def center_from_right_boundary(self, right_x: float) -> float:
        lane_width_px = self.current_lane_width_px()
        configured_offset = self.rightmost_line_target_offset_px
        if configured_offset <= 0.0:
            configured_offset = lane_width_px * self.rightmost_line_target_offset_ratio
        min_offset = max(55.0, lane_width_px * 0.35)
        max_offset = min(150.0, lane_width_px * 0.65)
        offset = max(min_offset, min(max_offset, configured_offset))
        return right_x - offset

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
        closed_shape = self.detect_closed_parking_shape(mask)
        y0 = int(height * self.finish_bottom_roi_ratio)
        y0 = max(0, min(height - 1, y0))
        bottom = mask[y0:, :]
        if bottom.size == 0:
            return ParkingBoxResult(
                closed_shape[0], None, 0.0, 0.0, 0.0, 0.0,
                closed_shape_detected=closed_shape[0],
                closed_shape_box=closed_shape[1],
                closed_shape_score=closed_shape[2],
                closed_top_ratio=closed_shape[3],
                closed_bottom_ratio=closed_shape[4],
                closed_left_ratio=closed_shape[5],
                closed_right_ratio=closed_shape[6],
            )

        horizontal_width_ratio = 0.0
        horizontal_rows = 0
        best_span = None
        best_span_left_ratio = 0.0
        best_span_right_ratio = 0.0
        for row_index in range(bottom.shape[0]):
            xs = np.flatnonzero(bottom[row_index, :] > 0)
            if xs.size == 0:
                continue
            span = int(xs[-1] - xs[0] + 1)
            ratio = span / float(width)
            if ratio > horizontal_width_ratio:
                horizontal_width_ratio = ratio
                best_span = (int(xs[0]), int(xs[-1]), y0 + row_index)
                best_span_left_ratio = float(xs[0]) / float(width)
                best_span_right_ratio = float(xs[-1]) / float(width)
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

        vertical_sides_ok = min(left_h_ratio, right_h_ratio) >= self.finish_min_vertical_height_ratio
        box_shape_ok = best_box is not None and (vertical_sides_ok or not self.finish_require_vertical_sides)
        full_box_detected = (
            horizontal_rows >= self.finish_min_horizontal_rows
            and horizontal_width_ratio >= self.finish_min_horizontal_width_ratio
            and bottom_y_ratio >= self.finish_min_bottom_y_ratio
            and (box_shape_ok or vertical_sides_ok)
        )
        stop_pose_detected = (
            self.finish_stop_min_horizontal_rows <= horizontal_rows <= self.finish_stop_max_horizontal_rows
            and self.finish_stop_min_horizontal_width_ratio
            <= horizontal_width_ratio
            <= self.finish_stop_max_horizontal_width_ratio
            and self.finish_stop_bottom_y_min_ratio <= bottom_y_ratio <= self.finish_stop_bottom_y_max_ratio
            and best_span_right_ratio >= self.finish_stop_min_right_edge_ratio
        )
        return ParkingBoxResult(
            full_box_detected or stop_pose_detected or closed_shape[0],
            best_box,
            horizontal_width_ratio,
            left_h_ratio,
            right_h_ratio,
            bottom_y_ratio,
            horizontal_rows,
            best_span_left_ratio,
            best_span_right_ratio,
            full_box_detected,
            stop_pose_detected,
            closed_shape[0],
            closed_shape[1],
            closed_shape[2],
            closed_shape[3],
            closed_shape[4],
            closed_shape[5],
            closed_shape[6],
        )

    def detect_closed_parking_shape(
        self, mask: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]], float, float, float, float, float]:
        if not self.finish_closed_shape_enabled:
            return False, None, 0.0, 0.0, 0.0, 0.0, 0.0

        height, width = mask.shape[:2]
        if height <= 0 or width <= 0:
            return False, None, 0.0, 0.0, 0.0, 0.0, 0.0

        kernel_size = max(3, int(self.finish_closed_morph_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        closed = cv2.dilate(closed, kernel, iterations=1)

        contour_result = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]

        best = (False, None, 0.0, 0.0, 0.0, 0.0, 0.0)
        best_score = 0.0
        min_w = width * self.finish_closed_min_width_ratio
        min_h = height * self.finish_closed_min_height_ratio
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < min_w or h < min_h:
                continue
            if w <= 0 or h <= 0:
                continue

            roi = closed[y : y + h, x : x + w]
            band_y = max(3, int(round(h * self.finish_closed_band_ratio)))
            band_x = max(3, int(round(w * self.finish_closed_band_ratio)))

            top_ratio = self.column_presence_ratio(roi[:band_y, :])
            bottom_ratio = self.column_presence_ratio(roi[h - band_y :, :])
            left_ratio = self.vertical_presence_ratio(roi[:, :band_x])
            right_ratio = self.vertical_presence_ratio(roi[:, w - band_x :])
            side_ratio = max(left_ratio, right_ratio)

            width_ratio = w / float(width)
            height_ratio = h / float(height)
            shape_ok = (
                top_ratio >= self.finish_closed_min_horizontal_presence
                and bottom_ratio >= self.finish_closed_min_horizontal_presence
                and side_ratio >= self.finish_closed_min_vertical_presence
            )
            if not shape_ok:
                continue

            score = top_ratio + bottom_ratio + side_ratio + 0.5 * width_ratio + 0.5 * height_ratio
            if score > best_score:
                best_score = score
                best = (True, (x, y, w, h), score, top_ratio, bottom_ratio, left_ratio, right_ratio)

        return best

    def should_count_finish_detection(self, parking_result: ParkingBoxResult, route_locked: bool) -> bool:
        now = time.time()
        if now - self.start_time < self.finish_detection_start_delay:
            return False

        if not self.startup_maneuver_done:
            return False

        if route_locked and not (
            self.finish_closed_instant_stop
            and self.finish_closed_ignore_route_lock
            and parking_result.closed_shape_detected
        ):
            return False

        if parking_result.stop_pose_detected or parking_result.closed_shape_detected:
            return True

        return self.finish_use_full_box_stop and self.finish_detection_enabled and parking_result.full_box_detected

    def column_presence_ratio(self, image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        col_hits = np.any(image > 0, axis=0)
        if not np.any(col_hits):
            return 0.0
        xs = np.flatnonzero(col_hits)
        return float(xs[-1] - xs[0] + 1) / float(image.shape[1])

    def vertical_presence_ratio(self, image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        row_hits = np.any(image > 0, axis=1)
        if not np.any(row_hits):
            return 0.0
        ys = np.flatnonzero(row_hits)
        return float(ys[-1] - ys[0] + 1) / float(image.shape[0])

    def handle_senior_follow(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        roi_origin_y: int,
        parking_result: ParkingBoxResult,
        fork_rows: int,
        fork_detected_latched: bool,
        route_locked: bool,
        now: float,
    ):
        result = self.senior_tracker.compute(frame)
        twist = Twist()
        twist.linear.x = result.linear_x
        twist.linear.y = result.linear_y
        twist.angular.z = result.angular_z
        self.cmd_pub.publish(twist)

        self.last_cmd_linear = twist.linear.x
        self.last_cmd_angular = twist.angular.z
        self.last_senior_path = result.path
        self.last_target_center = result.target_x
        self.last_lane_center = result.target_x
        self.last_error_px = 0.0 if result.target_x is None else result.target_x - frame.shape[1] / 2.0
        self.last_route_locked = route_locked
        self.last_two_sided = False
        self.single_line_frames = 0 if result.found else self.single_line_frames + 1
        if result.found:
            self.last_detection_time = now
            self.set_status("senior_tracking")
        else:
            self.set_status("senior_searching")

        rospy.loginfo_throttle(
            0.5,
            "senior_right_follow: found=%d right_count=%d left_count=%d target=(%s,%s) error=%.3f cmd=(%.3f,%.3f,%.3f)",
            int(result.found),
            result.right_count,
            result.left_count,
            "None" if result.target_x is None else "%.1f" % result.target_x,
            "None" if result.target_y is None else "%.1f" % result.target_y,
            result.error,
            twist.linear.x,
            twist.linear.y,
            twist.angular.z,
        )
        self.publish_debug_image(frame, mask, roi_origin_y, [], result.target_x, parking_result, fork_rows, fork_detected_latched, now)
        self.publish_debug_info(now, result.target_x, parking_result, fork_rows, fork_detected_latched, route_locked)
        self.publish_status()

    def publish_control(self, lane_center: float, image_width: int, now: float, two_sided_tracking: bool):
        image_center = image_width / 2.0
        error = lane_center - image_center
        self.last_error_px = error
        angular = -self.pid.update(error, now)
        angular = max(-self.max_angular_speed, min(self.max_angular_speed, angular))
        if self.status == "rightmost_line_only":
            angular_limit = max(0.05, min(self.max_angular_speed, self.rightmost_max_angular_speed))
            angular = max(-angular_limit, min(angular_limit, angular))

        if self.status == "turn_right" and now < self.turn_until:
            linear = self.turn_linear_speed
        elif self.status == "rightmost_line_only":
            linear = self.rightmost_line_only_speed
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

    def start_finish_forward(self, now: float):
        self.pid.reset()
        self.finish_frames = 0
        self.finish_lost_frames = 0
        self.finish_detection_enabled = False
        self.finish_forward_active = True
        self.finish_forward_start_time = now
        speed = max(self.finish_forward_speed, 1e-3)
        self.finish_forward_duration = max(0.0, self.finish_forward_distance_m) / speed
        self.set_status("finish_forward")
        self.hard_stop_robot()
        self.publish_finish_forward_cmd()
        rospy.loginfo(
            "finish forward started: distance=%.2fm speed=%.3fm/s duration=%.2fs",
            self.finish_forward_distance_m,
            speed,
            self.finish_forward_duration,
        )

    def handle_finish_forward(self, now: float):
        elapsed = 0.0 if self.finish_forward_start_time is None else max(0.0, now - self.finish_forward_start_time)
        if elapsed >= self.finish_forward_duration:
            rospy.loginfo(
                "finish forward done: elapsed=%.2fs duration=%.2fs",
                elapsed,
                self.finish_forward_duration,
            )
            self.reset_finish_forward()
            self.finish_time = now
            self.set_status("finish_stop")
            self.hard_stop_robot()
            self.publish_status()
            return

        self.set_status("finish_forward")
        self.publish_finish_forward_cmd()
        self.publish_status()

    def publish_finish_forward_cmd(self):
        twist = Twist()
        twist.linear.x = self.finish_forward_speed
        twist.angular.z = 0.0
        self.last_cmd_linear = twist.linear.x
        self.last_cmd_angular = twist.angular.z
        self.cmd_pub.publish(twist)

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
            self.startup_maneuver_done_time = now
            self.startup_phase = "rightmost_line_only"
            rightmost_duration = max(0.0, self.rightmost_line_only_duration)
            post_lock_duration = max(0.0, self.post_rightmost_route_lock_duration)
            self.rightmost_line_only_until = now + rightmost_duration
            self.right_route_lock_until = max(
                self.right_route_lock_until,
                self.rightmost_line_only_until + post_lock_duration,
            )
            self.turn_until = 0.0
            self.fork_latch_until = 0.0
            self.pid.reset()
            self.hard_stop_robot()
            rospy.loginfo(
                "startup maneuver finished: forward1=%.2fm turn=%.1fdeg forward2=%.2fm rightmost_only=%.2fs post_route_lock=%.2fs",
                self.startup_forward1_distance_m,
                self.startup_turn_angle_deg,
                self.startup_forward2_distance_m,
                self.rightmost_line_only_duration,
                self.post_rightmost_route_lock_duration,
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
        self.last_debug_snapshot = self.build_debug_snapshot(
            frame, mask, roi_origin_y, observations, lane_center, parking_result, fork_rows, fork_detected, now
        )

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

        if self.last_senior_path:
            for x, y in self.last_senior_path[::5]:
                px = max(0, min(width - 1, int(round(x))))
                py = max(0, min(height - 1, int(round(y))))
                cv2.circle(debug, (px, py), 2, (255, 0, 255), -1)

        if parking_result.box is not None:
            x, y, w, h = parking_result.box
            cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 120, 0), 2)

        if parking_result.closed_shape_box is not None:
            x, y, w, h = parking_result.closed_shape_box
            cv2.rectangle(debug, (x, roi_origin_y + y), (x + w, roi_origin_y + y + h), (0, 255, 0), 2)

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
        text3 = "park_w={:.2f} vl={:.2f} vr={:.2f} bottom={:.2f} closed={} lane_w={:.1f}px sel={}".format(
            parking_result.horizontal_width_ratio,
            parking_result.vertical_left_height_ratio,
            parking_result.vertical_right_height_ratio,
            parking_result.bottom_y_ratio,
            int(parking_result.closed_shape_detected),
            self.current_lane_width_px(),
            self.selection_summary(observations),
        )
        cv2.putText(debug, text3, (10, max(mh + 75, 80)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)

        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding="bgr8"))
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "debug image conversion failed: %s", exc)

    def build_debug_snapshot(
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
        height, width = frame.shape[:2]
        box = parking_result.box
        box_width_ratio = 0.0
        box_height_ratio = 0.0
        box_bottom_y_ratio = 0.0
        if box is not None:
            _, y, w, h = box
            box_width_ratio = w / float(max(1, width))
            box_height_ratio = h / float(max(1, height))
            box_bottom_y_ratio = (y + h) / float(max(1, height))
        closed_box = parking_result.closed_shape_box

        return {
            "timestamp": now,
            "status": self.status,
            "startup_phase": self.startup_phase,
            "image_width": width,
            "image_height": height,
            "roi_origin_y": roi_origin_y,
            "mask_height": int(mask.shape[0]),
            "mask_width": int(mask.shape[1]),
            "lane_center_px": None if lane_center is None else float(lane_center),
            "target_center_px": None if self.last_target_center is None else float(self.last_target_center),
            "image_center_px": width / 2.0,
            "error_px": float(self.last_error_px),
            "cmd_linear": float(self.last_cmd_linear),
            "cmd_angular": float(self.last_cmd_angular),
            "route_locked": bool(self.last_route_locked),
            "route_lock_left_sec": max(0.0, self.right_route_lock_until - now),
            "rightmost_left_sec": max(0.0, self.rightmost_line_only_until - now),
            "fork_rows": int(fork_rows),
            "fork_detected": bool(fork_detected),
            "two_sided": bool(self.last_two_sided),
            "selection_summary": self.selection_summary(observations),
            "finish_detection_enabled": bool(self.finish_detection_enabled),
            "finish_auto_stop": bool(self.finish_auto_stop),
            "finish_detection_start_delay": float(self.finish_detection_start_delay),
            "finish_detection_start_delay_left": max(0.0, self.finish_detection_start_delay - (now - self.start_time)),
            "finish_frames": int(self.finish_frames),
            "finish_confirm_frames": int(self.finish_confirm_frames),
            "finish_release_frames": int(self.finish_release_frames),
            "finish_forward_active": bool(self.finish_forward_active),
            "finish_forward_after_stop_enabled": bool(self.finish_forward_after_stop_enabled),
            "finish_forward_distance_m": float(self.finish_forward_distance_m),
            "finish_forward_speed": float(self.finish_forward_speed),
            "finish_forward_duration": float(self.finish_forward_duration),
            "finish_candidate_box": None if box is None else [int(v) for v in box],
            "finish_closed_shape_box": None if closed_box is None else [int(v) for v in closed_box],
            "finish_metrics": {
                "parking_detected": bool(parking_result.detected),
                "full_box_detected": bool(parking_result.full_box_detected),
                "stop_pose_detected": bool(parking_result.stop_pose_detected),
                "closed_shape_detected": bool(parking_result.closed_shape_detected),
                "closed_shape_score": float(parking_result.closed_shape_score),
                "closed_top_ratio": float(parking_result.closed_top_ratio),
                "closed_bottom_ratio": float(parking_result.closed_bottom_ratio),
                "closed_left_ratio": float(parking_result.closed_left_ratio),
                "closed_right_ratio": float(parking_result.closed_right_ratio),
                "horizontal_width_ratio": float(parking_result.horizontal_width_ratio),
                "horizontal_rows": int(parking_result.horizontal_rows),
                "horizontal_left_x_ratio": float(parking_result.horizontal_left_x_ratio),
                "horizontal_right_x_ratio": float(parking_result.horizontal_right_x_ratio),
                "vertical_left_height_ratio": float(parking_result.vertical_left_height_ratio),
                "vertical_right_height_ratio": float(parking_result.vertical_right_height_ratio),
                "bottom_y_ratio": float(parking_result.bottom_y_ratio),
                "box_width_ratio": float(box_width_ratio),
                "box_height_ratio": float(box_height_ratio),
                "box_bottom_y_ratio": float(box_bottom_y_ratio),
                "min_horizontal_width_ratio": float(self.finish_min_horizontal_width_ratio),
                "min_horizontal_rows": int(self.finish_min_horizontal_rows),
                "min_vertical_height_ratio": float(self.finish_min_vertical_height_ratio),
                "min_box_width_ratio": float(self.finish_min_box_width_ratio),
                "min_bottom_y_ratio": float(self.finish_min_bottom_y_ratio),
                "require_vertical_sides": bool(self.finish_require_vertical_sides),
                "use_full_box_stop": bool(self.finish_use_full_box_stop),
                "closed_shape_enabled": bool(self.finish_closed_shape_enabled),
                "closed_ignore_route_lock": bool(self.finish_closed_ignore_route_lock),
                "closed_instant_stop": bool(self.finish_closed_instant_stop),
                "closed_min_width_ratio": float(self.finish_closed_min_width_ratio),
                "closed_min_height_ratio": float(self.finish_closed_min_height_ratio),
                "closed_min_horizontal_presence": float(self.finish_closed_min_horizontal_presence),
                "closed_min_vertical_presence": float(self.finish_closed_min_vertical_presence),
                "closed_band_ratio": float(self.finish_closed_band_ratio),
                "closed_morph_kernel": int(self.finish_closed_morph_kernel),
                "stop_min_horizontal_width_ratio": float(self.finish_stop_min_horizontal_width_ratio),
                "stop_max_horizontal_width_ratio": float(self.finish_stop_max_horizontal_width_ratio),
                "stop_min_horizontal_rows": int(self.finish_stop_min_horizontal_rows),
                "stop_max_horizontal_rows": int(self.finish_stop_max_horizontal_rows),
                "stop_min_right_edge_ratio": float(self.finish_stop_min_right_edge_ratio),
                "stop_bottom_y_min_ratio": float(self.finish_stop_bottom_y_min_ratio),
                "stop_bottom_y_max_ratio": float(self.finish_stop_bottom_y_max_ratio),
            },
            "observations": [
                {
                    "y": int(obs.y),
                    "left_x": None if obs.left_x is None else float(obs.left_x),
                    "right_x": None if obs.right_x is None else float(obs.right_x),
                    "center_x": None if obs.center_x is None else float(obs.center_x),
                    "multi_candidate": bool(obs.multi_candidate),
                    "selection": obs.selection,
                    "segments": [
                        {
                            "left": int(segment.left),
                            "right": int(segment.right),
                            "center": float(segment.center),
                            "width": int(segment.width),
                        }
                        for segment in obs.segments
                    ],
                }
                for obs in observations
            ],
        }

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
            "fork_rows={fork_rows} fork={fork} finish_frames={finish_frames} finish_delay_left={finish_delay_left:.1f} "
            "parking_detected={parking_detected} parking_width={parking_width:.2f} "
            "parking_bottom={parking_bottom:.2f} parking_rows={parking_rows} parking_x=({parking_left:.2f},{parking_right:.2f}) "
            "parking_full_box={parking_full_box} parking_stop_pose={parking_stop_pose} "
            "parking_closed={parking_closed} closed_score={closed_score:.2f}"
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
            finish_delay_left=max(0.0, self.finish_detection_start_delay - (now - self.start_time)),
            parking_detected=int(parking_result.detected),
            parking_width=parking_result.horizontal_width_ratio,
            parking_bottom=parking_result.bottom_y_ratio,
            parking_rows=parking_result.horizontal_rows,
            parking_left=parking_result.horizontal_left_x_ratio,
            parking_right=parking_result.horizontal_right_x_ratio,
            parking_full_box=int(parking_result.full_box_detected),
            parking_stop_pose=int(parking_result.stop_pose_detected),
            parking_closed=int(parking_result.closed_shape_detected),
            closed_score=parking_result.closed_shape_score,
        )
        self.debug_info_pub.publish(String(data=msg))
        rospy.loginfo_throttle(0.5, "right_line_debug: %s", msg)


def main():
    rospy.init_node("right_line_follow_node")
    RightLineFollowNode()
    rospy.spin()


if __name__ == "__main__":
    main()
