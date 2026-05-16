#!/usr/bin/env python3
# stable_right_follow.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

        self.image_topic = rospy.get_param(
            "~image_topic",
            "/usb_cam/image_raw"
        )

        self.cmd_vel_topic = rospy.get_param(
            "~cmd_vel_topic",
            "/cmd_vel"
        )

        self.status_topic = rospy.get_param(
            "~status_topic",
            "/line_follow/status"
        )

        self.start_topic = rospy.get_param(
            "~start_topic",
            "/line_follow/start"
        )

        self.roi_y_start_ratio = rospy.get_param(
            "~roi_y_start_ratio",
            0.45
        )

        self.roi_y_end_ratio = rospy.get_param(
            "~roi_y_end_ratio",
            1.0
        )

        self.white_v_min = rospy.get_param(
            "~white_v_min",
            200
        )

        self.white_v_max = rospy.get_param(
            "~white_v_max",
            255
        )

        self.white_s_max = rospy.get_param(
            "~white_s_max",
            80
        )

        self.gray_white_threshold = rospy.get_param(
            "~gray_white_threshold",
            205
        )

        self.gray_white_max = rospy.get_param(
            "~gray_white_max",
            255
        )

        self.morph_kernel_size = rospy.get_param(
            "~morph_kernel_size",
            5
        )

        self.min_contour_area = rospy.get_param(
            "~min_contour_area",
            80
        )

        self.min_line_width_px = rospy.get_param(
            "~min_line_width_px",
            6
        )

        self.min_segment_gap_px = rospy.get_param(
            "~min_segment_gap_px",
            10
        )

        self.right_offset_px = rospy.get_param(
            "~right_offset_px",
            150
        )

        self.base_linear_speed = rospy.get_param(
            "~base_linear_speed",
            0.12
        )

        self.max_angular_speed = rospy.get_param(
            "~max_angular_speed",
            1.0
        )

        self.search_linear_speed = rospy.get_param(
            "~search_linear_speed",
            0.03
        )

        self.search_angular_speed = rospy.get_param(
            "~search_angular_speed",
            0.40
        )

        kp = rospy.get_param("~kp", 0.0050)
        ki = rospy.get_param("~ki", 0.0)
        kd = rospy.get_param("~kd", 0.0018)
        max_integral = rospy.get_param("~max_integral", 100)

        self.pid = PID(kp, ki, kd, max_integral)

        self.start_straight_duration = rospy.get_param(
            "~start_straight_duration",
            2.5
        )

        self.first_line_y_threshold = rospy.get_param(
            "~first_line_y_threshold",
            0.72
        )

        self.second_line_y_threshold = rospy.get_param(
            "~second_line_y_threshold",
            0.88
        )

        self.after_first_line_speed = rospy.get_param(
            "~after_first_line_speed",
            0.035
        )

        self.after_first_line_duration = rospy.get_param(
            "~after_first_line_duration",
            7.0
        )

        self.max_run_time = rospy.get_param(
            "~max_run_time",
            50.0
        )

        self.state = "START_STRAIGHT"

        self.start_time = time.time()

        self.first_line_stop_time = None
        self.after_first_move_start_time = None

        self.last_error_px = -1.0

        self.started = True

        self.cmd_pub = rospy.Publisher(
            self.cmd_vel_topic,
            Twist,
            queue_size=1
        )

        self.status_pub = rospy.Publisher(
            self.status_topic,
            String,
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

    def start_callback(self, msg):

        self.started = bool(msg.data)

        if self.started:

            self.state = "START_STRAIGHT"

            self.start_time = time.time()

            self.first_line_stop_time = None
            self.after_first_move_start_time = None

            self.pid.reset()

        else:

            self.stop_robot()

    def image_callback(self, msg):

        if not self.started:
            return

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except CvBridgeError:
            return

        now = time.time()

        elapsed = now - self.start_time

        if elapsed > self.max_run_time:

            self.stop_robot()

            self.state = "DONE"

            return

        mask = self.extract_white_mask(frame)

        if self.state == "START_STRAIGHT":

            if elapsed < self.start_straight_duration:

                twist = Twist()

                twist.linear.x = self.base_linear_speed
                twist.angular.z = 0.0

                self.cmd_pub.publish(twist)

                return

            else:

                self.state = "FOLLOW_RIGHT"

        horizontal_detected, line_y_ratio = \
            self.detect_horizontal_line(mask)

        if self.state == "FOLLOW_RIGHT":

            if horizontal_detected and \
               line_y_ratio > self.second_line_y_threshold:

                self.stop_robot()

                self.state = "DONE"

                rospy.loginfo("Second line stop")

                return

            if horizontal_detected and \
               line_y_ratio > self.first_line_y_threshold:

                self.stop_robot()

                self.state = "FIRST_LINE_STOP"

                self.first_line_stop_time = now

                rospy.loginfo("First line detected")

                return

            angular = self.compute_right_follow_angular(
                mask,
                frame.shape[1]
            )

            twist = Twist()

            twist.linear.x = self.base_linear_speed
            twist.angular.z = angular

            self.cmd_pub.publish(twist)

            return

        if self.state == "FIRST_LINE_STOP":

            self.stop_robot()

            if now - self.first_line_stop_time > 0.8:

                self.state = "AFTER_FIRST_MOVE"

                self.after_first_move_start_time = now

            return

        if self.state == "AFTER_FIRST_MOVE":

            move_elapsed = now - self.after_first_move_start_time

            if horizontal_detected and \
               line_y_ratio > self.second_line_y_threshold:

                self.stop_robot()

                self.state = "DONE"

                rospy.loginfo("Second line final stop")

                return

            if move_elapsed > self.after_first_line_duration:

                self.stop_robot()

                self.state = "DONE"

                rospy.loginfo("Distance stop")

                return

            angular = self.compute_right_follow_angular(
                mask,
                frame.shape[1]
            )

            twist = Twist()

            twist.linear.x = self.after_first_line_speed
            twist.angular.z = angular

            self.cmd_pub.publish(twist)

            return

    def extract_white_mask(self, frame):

        h = frame.shape[0]

        y0 = int(h * self.roi_y_start_ratio)
        y1 = int(h * self.roi_y_end_ratio)

        roi = frame[y0:y1, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_hsv = cv2.inRange(
            hsv,
            (0, 0, self.white_v_min),
            (179, self.white_s_max, self.white_v_max)
        )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        mask_gray = cv2.inRange(
            gray,
            self.gray_white_threshold,
            self.gray_white_max
        )

        mask = cv2.bitwise_or(mask_hsv, mask_gray)

        mask = self.remove_small_components(mask)

        k = self.morph_kernel_size

        if k % 2 == 0:
            k += 1

        kernel = np.ones((k, k), np.uint8)

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

        return mask

    def remove_small_components(self, mask):

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result = np.zeros_like(mask)

        for c in contours:

            area = cv2.contourArea(c)

            if area >= self.min_contour_area:

                cv2.drawContours(
                    result,
                    [c],
                    -1,
                    255,
                    thickness=cv2.FILLED
                )

        return result

    def find_segments(self, row) -> List[Segment]:

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

        return self.merge_close_segments(segments)

    def merge_close_segments(self, segments):

        if not segments:
            return []

        merged = [segments[0]]

        for seg in segments[1:]:

            prev = merged[-1]

            if seg.left - prev.right <= self.min_segment_gap_px:

                merged[-1] = Segment(
                    prev.left,
                    seg.right,
                    (prev.left + seg.right) / 2.0,
                    seg.right - prev.left + 1
                )

            else:

                merged.append(seg)

        return merged

    def find_rightmost_line_x(self, mask) -> Optional[float]:

        h = mask.shape[0]

        scan_ratios = [
            0.95,
            0.92,
            0.88,
            0.84,
            0.80,
            0.75,
            0.70
        ]

        for ratio in scan_ratios:

            y = int(h * ratio)

            row = mask[y, :]

            segments = self.find_segments(row)

            if segments:

                return segments[-1].center

        return None

    def compute_right_follow_angular(
            self,
            mask,
            image_width):

        right_x = self.find_rightmost_line_x(mask)

        if right_x is None:

            if self.last_error_px < 0:

                return -self.search_angular_speed * 1.2

            return -self.search_angular_speed

        target_x = right_x - self.right_offset_px

        error = target_x - image_width / 2.0

        self.last_error_px = error

        angular = -self.pid.update(
            error,
            time.time()
        )

        angular = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, angular)
        )

        return angular

    def detect_horizontal_line(
            self,
            mask) -> Tuple[bool, float]:

        h, w = mask.shape[:2]

        bottom_start = int(h * 0.80)

        bottom = mask[bottom_start:, :]

        if bottom.size == 0:
            return False, 0.0

        min_width = int(w * 0.45)

        for r in range(bottom.shape[0] - 1, -1, -1):

            row = bottom[r, :]

            segments = self.find_segments(row)

            for seg in segments:

                if seg.width >= min_width and \
                   r > bottom.shape[0] * 0.45:

                    y_ratio = (
                        bottom_start + r
                    ) / float(h)

                    return True, y_ratio

        return False, 0.0

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
