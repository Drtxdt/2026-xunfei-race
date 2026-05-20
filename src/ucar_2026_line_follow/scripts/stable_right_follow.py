#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


@dataclass
class Segment:
    left: int
    right: int
    center: float
    width: int


class PID:
    def __init__(self, kp, ki, kd, max_integral):
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

    def update(self, error, now):
        if self.last_time is None:
            dt = 0.0
        else:
            dt = max(now - self.last_time, 1e-3)
        if dt > 0:
            self.integral += error * dt
            self.integral = max(
                -self.max_integral,
                min(self.max_integral, self.integral)
            )
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0
        self.last_error = error
        self.last_time = now
        return (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )


class StableRightFollowNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_topic = "/usb_cam/image_raw"
        self.cmd_vel_topic = "/cmd_vel"
        self.status_topic = "/line_follow/status"
        self.start_topic = "/line_follow/start"

        # =========================
        # ROI
        # =========================
        self.roi_y_start_ratio = 0.42
        self.roi_y_end_ratio = 1.0

        # =========================
        # 白线参数
        # =========================
        self.white_v_min = 185
        self.white_s_max = 70
        self.gray_white_threshold = 195
        self.min_contour_area = 120
        self.min_line_width_px = 8
        self.min_segment_gap_px = 12

        # =========================
        # 巡线核心参数
        # =========================
        self.right_offset_px = 240
        self.base_linear_speed = 0.16
        self.curve_linear_speed = 0.18
        self.search_linear_speed = 0.03
        self.max_angular_speed = 0.72
        self.search_angular_speed = 0.20
        self.straight_threshold = 18
        self.curve_threshold = 60

        # 平滑
        self.angular_smooth_alpha = 0.86

        # 丢线容忍
        self.max_lost_count = 6

        # PID
        self.pid = PID(
            0.0035,
            0.0,
            0.0011,
            100
        )

        # 起步
        self.start_straight_duration = 1.6

        # 停车
        self.first_line_y_threshold = 0.75
        self.after_first_line_speed = 0.04
        self.after_first_line_duration = 1.8
        self.max_run_time = 60.0

        self.state = "START_STRAIGHT"
        self.start_time = time.time()
        self.first_line_stop_time = None
        self.after_first_move_start_time = None
        self.last_error_px = 0.0
        self.last_angular = 0.0
        self.line_lost_count = 0
        self.last_valid_right_x = None
        self.started = True

        self.cmd_pub = rospy.Publisher(
            self.cmd_vel_topic,
            Twist,
            queue_size=1
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24
        )
        self.start_sub = rospy.Subscriber(
            self.start_topic,
            Bool,
            self.start_callback,
            queue_size=1
        )
        rospy.on_shutdown(self.stop_robot)
        rospy.loginfo("Stable Right Follow Node Started")

    # =====================================================
    def start_callback(self, msg):
        self.started = bool(msg.data)
        if self.started:
            self.state = "START_STRAIGHT"
            self.start_time = time.time()
            self.pid.reset()
        else:
            self.stop_robot()

    # =====================================================
    def image_callback(self, msg):
        if not self.started:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception:
            return

        now = time.time()
        elapsed = now - self.start_time
        if elapsed > self.max_run_time:
            self.stop_robot()
            return

        mask = self.extract_white_mask(frame)

        # =========================
        # 起步
        # =========================
        if self.state == "START_STRAIGHT":
            if elapsed < self.start_straight_duration:
                twist = Twist()
                twist.linear.x = 0.12
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return
            else:
                self.state = "FOLLOW_RIGHT"

        horizontal_detected, line_y_ratio = \
            self.detect_horizontal_line(mask)

        # =========================
        # 第一根横线停车
        # =========================
        if self.state == "FOLLOW_RIGHT":
            if horizontal_detected and \
                    line_y_ratio > self.first_line_y_threshold:
                self.stop_robot()
                self.state = "FIRST_LINE_STOP"
                self.first_line_stop_time = now
                return
            angular, speed = self.compute_control(
                mask,
                frame.shape[1]
            )
            twist = Twist()
            twist.linear.x = speed
            twist.angular.z = angular
            self.cmd_pub.publish(twist)
            return

        # =========================
        # 停0.5秒
        # =========================
        if self.state == "FIRST_LINE_STOP":
            self.stop_robot()
            if now - self.first_line_stop_time > 0.5:
                self.state = "AFTER_FIRST_MOVE"
                self.after_first_move_start_time = now
            return

        # =========================
        # 前进25cm
        # =========================
        if self.state == "AFTER_FIRST_MOVE":
            move_elapsed = now - self.after_first_move_start_time
            if move_elapsed > self.after_first_line_duration:
                self.stop_robot()
                return
            angular, _ = self.compute_control(
                mask,
                frame.shape[1]
            )
            twist = Twist()
            twist.linear.x = self.after_first_line_speed
            twist.angular.z = angular * 0.4
            self.cmd_pub.publish(twist)

    # =====================================================
    def extract_white_mask(self, frame):
        h = frame.shape[0]
        y0 = int(h * self.roi_y_start_ratio)
        y1 = int(h * self.roi_y_end_ratio)
        roi = frame[y0:y1, :]

        # =========================
        # 降低反光影响
        # =========================
        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        # 动态亮度阈值
        dynamic_v = max(
            self.white_v_min,
            int(np.mean(v_channel) + 35)
        )

        mask_hsv = cv2.inRange(
            hsv,
            (0, 0, dynamic_v),
            (179, self.white_s_max, 255)
        )

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        _, mask_gray = cv2.threshold(
            gray,
            self.gray_white_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # 双重融合
        mask = cv2.bitwise_and(mask_hsv, mask_gray)

        # =========================
        # 去除反光亮斑
        # =========================
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # 去除小噪声
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        clean_mask = np.zeros_like(mask)
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # 过滤反光点
            if w < 10 and h < 10:
                continue
            # 过滤巨大反光块
            if area > mask.shape[0] * mask.shape[1] * 0.55:
                continue
            cv2.drawContours(
                clean_mask,
                [c],
                -1,
                255,
                thickness=cv2.FILLED
            )
        return clean_mask

    # =====================================================
    def find_segments(self, row):
        active = row > 0
        segments = []
        start = None
        for i, val in enumerate(active):
            if val and start is None:
                start = i
            elif not val and start is not None:
                end = i - 1
                width = end - start + 1
                if width >= self.min_line_width_px:
                    segments.append(
                        Segment(
                            start,
                            end,
                            (start + end) / 2.0,
                            width
                        )
                    )
                start = None
        if start is not None:
            end = len(active) - 1
            width = end - start + 1
            if width >= self.min_line_width_px:
                segments.append(
                    Segment(
                        start,
                        end,
                        (start + end) / 2.0,
                        width
                    )
                )
        return segments

    # =====================================================
    def find_rightmost_line_x(self, mask):
        h = mask.shape[0]
        weighted_points = []
        scan_ratios = [
            0.96,
            0.92,
            0.88,
            0.84,
            0.80,
            0.76
        ]
        weights = [6, 5, 4, 3, 2, 1]
        for ratio, weight in zip(scan_ratios, weights):
            y = int(h * ratio)
            row = mask[y, :]
            segments = self.find_segments(row)
            if segments:
                weighted_points.append(
                    segments[-1].center * weight
                )
        if not weighted_points:
            return None
        return sum(weighted_points) / sum(weights[:len(weighted_points)])

    # =====================================================
    def compute_control(self, mask, image_width):
        right_x = self.find_rightmost_line_x(mask)

        # =========================
        # 丢线恢复
        # =========================
        if right_x is None:
            self.line_lost_count += 1
            twist = Twist()
            twist.linear.x = self.search_linear_speed
            # 连续缓慢右转
            twist.angular.z = -self.search_angular_speed
            self.cmd_pub.publish(twist)
            return -self.search_angular_speed, self.search_linear_speed

        self.line_lost_count = 0
        self.last_valid_right_x = right_x

        target_x = right_x - self.right_offset_px
        error = target_x - image_width / 2.0
        raw_angular = -self.pid.update(
            error,
            time.time()
        )

        # =========================
        # 平滑滤波
        # =========================
        angular = (
            self.angular_smooth_alpha * self.last_angular +
            (1.0 - self.angular_smooth_alpha) * raw_angular
        )
        self.last_angular = angular
        angular = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, angular)
        )

        abs_error = abs(error)

        # =========================
        # 直线
        # =========================
        if abs_error < self.straight_threshold:
            speed = self.base_linear_speed
            angular *= 0.35
        # =========================
        # 小弯
        # =========================
        elif abs_error < self.curve_threshold:
            speed = 0.145
            angular *= 0.72
        # =========================
        # 大弯
        # =========================
        else:
            speed = self.curve_linear_speed
            angular *= 0.90

        return angular, speed

    # =====================================================
    def detect_horizontal_line(self, mask):
        h, w = mask.shape[:2]
        bottom_start = int(h * 0.82)
        bottom = mask[bottom_start:, :]
        if bottom.size == 0:
            return False, 0.0
        min_width = int(w * 0.55)
        hit_count = 0
        best_ratio = 0.0
        for r in range(bottom.shape[0] - 1, -1, -1):
            row = bottom[r, :]
            segments = self.find_segments(row)
            for seg in segments:
                if seg.width >= min_width:
                    hit_count += 1
                    best_ratio = (
                        bottom_start + r
                    ) / float(h)
                    # 防误检
        if hit_count >= 4:
            return True, best_ratio
        return False, 0.0

    # =====================================================
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)


def main():
    rospy.init_node("stable_right_follow")
    StableRightFollowNode()
    rospy.spin()


if __name__ == "__main__":
    main()