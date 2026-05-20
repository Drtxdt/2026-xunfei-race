#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import cv2
import numpy as np
import rospy

from collections import deque

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


class PID:

    def __init__(self, kp, ki, kd):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def reset(self):

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def update(self, error):

        now = time.time()

        if self.last_time is None:

            dt = 0.01

        else:

            dt = now - self.last_time

            if dt <= 0:
                dt = 0.01

        self.integral += error * dt

        self.integral = max(
            -300,
            min(300, self.integral)
        )

        derivative = (
            error - self.last_error
        ) / dt

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.last_error = error
        self.last_time = now

        return output


class StableRightFollowNode:

    def __init__(self):

        rospy.init_node("stable_right_follow")

        self.bridge = CvBridge()

        # =====================================================
        # 发布
        # =====================================================

        self.cmd_pub = rospy.Publisher(
            "/cmd_vel",
            Twist,
            queue_size=1
        )

        # =====================================================
        # 订阅
        # =====================================================

        self.image_sub = rospy.Subscriber(
            "/usb_cam/image_raw",
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24
        )

        # =====================================================
        # PID
        # =====================================================

        self.pid = PID(
            kp=0.0042,
            ki=0.0000,
            kd=0.0014
        )

        # =====================================================
        # 参数
        # =====================================================

        self.base_speed = 0.18

        self.curve_speed = 0.22

        self.search_speed = 0.03

        self.max_angular = 0.72

        self.search_angular = 0.20

        self.right_offset = 235

        self.start_time = time.time()

        self.state = "START"

        self.last_angular = 0.0

        self.angular_alpha = 0.86

        self.error_buffer = deque(maxlen=6)

        self.lost_count = 0

        self.max_lost = 8

        self.last_valid_x = None

        self.horizontal_counter = 0

        self.stop_time = None

        self.forward_time = None

        self.forward_duration = 1.8

        self.search_direction = -1

        rospy.loginfo("Stable Right Follow Node Started")

    # =====================================================
    # 图像回调
    # =====================================================

    def image_callback(self, msg):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

        except:
            return

        h, w = frame.shape[:2]

        roi = frame[int(h * 0.42):, :]

        mask = self.extract_white_mask(roi)

        elapsed = time.time() - self.start_time

        # =====================================================
        # 起步直行
        # =====================================================

        if self.state == "START":

            if elapsed < 1.4:

                twist = Twist()

                twist.linear.x = 0.12
                twist.angular.z = 0.0

                self.cmd_pub.publish(twist)

                return

            else:

                self.state = "FOLLOW"

        # =====================================================
        # 横线停车
        # =====================================================

        if self.state == "FOLLOW":

            if self.detect_horizontal_line(mask):

                self.horizontal_counter += 1

            else:

                self.horizontal_counter = 0

            if self.horizontal_counter >= 4:

                self.state = "WAIT"

                self.stop_time = time.time()

                self.stop_robot()

                return

        # =====================================================
        # 等待
        # =====================================================

        if self.state == "WAIT":

            self.stop_robot()

            if time.time() - self.stop_time > 0.5:

                self.state = "FORWARD"

                self.forward_time = time.time()

            return

        # =====================================================
        # 前进25cm
        # =====================================================

        if self.state == "FORWARD":

            if time.time() - self.forward_time > self.forward_duration:

                self.state = "STOP"

                self.stop_robot()

                return

        # =====================================================
        # STOP
        # =====================================================

        if self.state == "STOP":

            self.stop_robot()

            return

        # =====================================================
        # 巡线
        # =====================================================

        right_x = self.find_right_line(mask)

        twist = Twist()

        # =====================================================
        # 丢线恢复
        # =====================================================

        if right_x is None:

            self.lost_count += 1

            twist.linear.x = self.search_speed

            twist.angular.z = (
                self.search_direction *
                self.search_angular
            )

            self.cmd_pub.publish(twist)

            return

        self.lost_count = 0

        self.last_valid_x = right_x

        # =====================================================
        # 目标点
        # =====================================================

        target_x = right_x - self.right_offset

        error = target_x - (w / 2)

        self.error_buffer.append(error)

        avg_error = np.mean(self.error_buffer)

        # =====================================================
        # PID
        # =====================================================

        angular = -self.pid.update(avg_error)

        # =====================================================
        # 平滑
        # =====================================================

        angular = (
            self.angular_alpha *
            self.last_angular
            +
            (1.0 - self.angular_alpha)
            *
            angular
        )

        self.last_angular = angular

        angular = max(
            -self.max_angular,
            min(self.max_angular, angular)
        )

        # =====================================================
        # 动态速度
        # =====================================================

        abs_error = abs(avg_error)

        # 直线

        if abs_error < 18:

            speed = self.base_speed

            angular *= 0.35

        # 小弯

        elif abs_error < 55:

            speed = 0.17

            angular *= 0.75

        # 大弯

        else:

            speed = self.curve_speed

            angular *= 0.92

        # FORWARD阶段

        if self.state == "FORWARD":

            speed = 0.04

            angular *= 0.4

        twist.linear.x = speed
        twist.angular.z = angular

        self.cmd_pub.publish(twist)

    # =====================================================
    # 白线提取
    # =====================================================

    def extract_white_mask(self, frame):

        # =====================================================
        # CLAHE
        # =====================================================

        lab = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        frame = cv2.cvtColor(
            lab,
            cv2.COLOR_LAB2BGR
        )

        # =====================================================
        # 高斯滤波
        # =====================================================

        blur = cv2.GaussianBlur(
            frame,
            (5, 5),
            0
        )

        # =====================================================
        # HSV
        # =====================================================

        hsv = cv2.cvtColor(
            blur,
            cv2.COLOR_BGR2HSV
        )

        gray = cv2.cvtColor(
            blur,
            cv2.COLOR_BGR2GRAY
        )

        v_mean = np.mean(hsv[:, :, 2])

        dynamic_v = max(
            185,
            int(v_mean + 35)
        )

        hsv_mask = cv2.inRange(
            hsv,
            (0, 0, dynamic_v),
            (179, 70, 255)
        )

        gray_mask = cv2.inRange(
            gray,
            195,
            255
        )

        mask = cv2.bitwise_and(
            hsv_mask,
            gray_mask
        )

        # =====================================================
        # 去反光
        # =====================================================

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

        # =====================================================
        # 面积过滤
        # =====================================================

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        clean = np.zeros_like(mask)

        for c in contours:

            area = cv2.contourArea(c)

            if area < 120:
                continue

            x, y, w, h = cv2.boundingRect(c)

            # 过滤小亮点

            if w < 8 and h < 8:
                continue

            # 过滤大反光

            if area > (
                mask.shape[0] *
                mask.shape[1] *
                0.55
            ):
                continue

            cv2.drawContours(
                clean,
                [c],
                -1,
                255,
                thickness=cv2.FILLED
            )

        return clean

    # =====================================================
    # 找右边线
    # =====================================================

    def find_right_line(self, mask):

        h = mask.shape[0]

        scan_rows = [
            int(h * 0.96),
            int(h * 0.92),
            int(h * 0.88),
            int(h * 0.84),
            int(h * 0.80),
            int(h * 0.76)
        ]

        weights = [
            6,
            5,
            4,
            3,
            2,
            1
        ]

        points = []

        total_weight = 0

        for i, y in enumerate(scan_rows):

            row = mask[y]

            idx = np.where(row > 0)[0]

            if len(idx) < 10:
                continue

            right_x = idx[-1]

            points.append(
                right_x * weights[i]
            )

            total_weight += weights[i]

        if total_weight == 0:
            return None

        return sum(points) / total_weight

    # =====================================================
    # 横线检测
    # =====================================================

    def detect_horizontal_line(self, mask):

        h, w = mask.shape

        roi = mask[int(h * 0.82):, :]

        rows = roi.shape[0]

        count = 0

        for y in range(rows):

            row = roi[y]

            white = np.sum(row > 0)

            if white > w * 0.55:

                count += 1

        return count >= 4

    # =====================================================
    # 停车
    # =====================================================

    def stop_robot(self):

        twist = Twist()

        twist.linear.x = 0.0
        twist.angular.z = 0.0

        self.cmd_pub.publish(twist)


if __name__ == "__main__":

    StableRightFollowNode()

    rospy.spin()