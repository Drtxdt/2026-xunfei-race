#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
import rospy

from collections import deque

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


class PID:
    def __init__(self, kp, ki, kd, max_integral=300):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
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
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        derivative = (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        self.last_time = now
        return output

        self.integral += error * dt

class StableRightFollowNode:
    def __init__(self):
        rospy.init_node("stable_right_follow", anonymous=False)
        self.bridge = CvBridge()

        # 发布与订阅
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image,
                                          self.image_callback, queue_size=1, buff_size=2**24)

        # ---------- 从参数服务器读取所有参数（如果 YAML 加载了） ----------
        # 巡线核心参数
        self.right_offset = rospy.get_param("~right_offset", 235)
        self.base_speed = rospy.get_param("~base_speed", 0.18)
        self.curve_speed = rospy.get_param("~curve_speed", 0.22)
        self.search_speed = rospy.get_param("~search_speed", 0.03)
        self.max_angular = rospy.get_param("~max_angular", 0.72)
        self.search_angular = rospy.get_param("~search_angular", 0.20)

        # 分段速度 / 转角增益
        self.straight_threshold = rospy.get_param("~straight_threshold", 18)
        self.curve_threshold = rospy.get_param("~curve_threshold", 55)

        # 平滑滤波
        self.angular_alpha = rospy.get_param("~angular_alpha", 0.86)
        self.error_buffer_size = rospy.get_param("~error_buffer_size", 6)
        self.error_buffer = deque(maxlen=self.error_buffer_size)

        # 丢线恢复
        self.max_lost = rospy.get_param("~max_lost", 8)
        self.lost_count = 0
        self.search_direction = -1
        self.last_valid_x = None

        # 横线停车流程
        self.state = "START"
        self.start_time = time.time()
        self.stop_time = None
        self.forward_time = None
        self.forward_duration = rospy.get_param("~forward_duration", 1.8)
        self.horizontal_counter = 0

        # 白线提取参数
        self.white_v_min = rospy.get_param("~white_v_min", 185)
        self.white_s_max = rospy.get_param("~white_s_max", 70)
        self.gray_threshold = rospy.get_param("~gray_threshold", 195)
        self.min_contour_area = rospy.get_param("~min_contour_area", 120)
        self.max_reflect_ratio = rospy.get_param("~max_reflect_ratio", 0.55)
        self.min_light_size = rospy.get_param("~min_light_size", 8)

        # 起步参数
        self.start_duration = rospy.get_param("~start_duration", 1.4)
        self.start_speed = rospy.get_param("~start_speed", 0.12)

        # PID 参数
        kp = rospy.get_param("~kp", 0.0042)
        ki = rospy.get_param("~ki", 0.0)
        kd = rospy.get_param("~kd", 0.0014)
        max_integral = rospy.get_param("~max_integral", 300)
        self.pid = PID(kp, ki, kd, max_integral)

        # 控制量缓存
        self.last_angular = 0.0

        rospy.loginfo("Stable Right Follow Node Started (with YAML support)")
        rospy.loginfo("Parameters: offset=%d, base_speed=%.2f, curve_speed=%.2f",
                      self.right_offset, self.base_speed, self.curve_speed)

    # ------------------------------------------------------------
    # 图像回调主逻辑（与之前完全相同，无需修改）
    # ------------------------------------------------------------
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(5, "Image conversion error: %s", e)
            return

        h, w = frame.shape[:2]
        roi = frame[int(h * 0.42):, :]
        mask = self.extract_white_mask(roi)

        elapsed = time.time() - self.start_time

        if self.state == "START":
            if elapsed < self.start_duration:
                twist = Twist()
                twist.linear.x = self.start_speed
                twist.angular.z = 0.0
                self.cmd_pub.publish(twist)
                return
            else:
                self.state = "FOLLOW"
                rospy.loginfo("Start finished, entering FOLLOW mode")

        if self.state == "FOLLOW":
            if self.detect_horizontal_line(mask):
                self.horizontal_counter += 1
            else:
                self.horizontal_counter = 0

            if self.horizontal_counter >= 4:
                rospy.loginfo("Horizontal line detected, stopping...")
                self.state = "WAIT"
                self.stop_time = time.time()
                self.stop_robot()
                return

        if self.state == "WAIT":
            self.stop_robot()
            if time.time() - self.stop_time > 0.5:
                self.state = "FORWARD"
                self.forward_time = time.time()
                rospy.loginfo("Moving forward 25cm after stop")
            return

        if self.state == "FORWARD":
            if time.time() - self.forward_time > self.forward_duration:
                self.state = "STOP"
                self.stop_robot()
                rospy.loginfo("Final stop")
                return

        if self.state == "STOP":
            self.stop_robot()
            return

        # 巡线控制
        right_x = self.find_right_line(mask)
        twist = Twist()

        if right_x is None:
            self.lost_count += 1
            if self.lost_count > self.max_lost:
                self.search_direction *= -1
                self.lost_count = 0
                rospy.loginfo_throttle(2, "Lost line too long, reverse search direction")
            twist.linear.x = self.search_speed
            twist.angular.z = self.search_direction * self.search_angular
            self.cmd_pub.publish(twist)
            return

        self.lost_count = 0
        self.last_valid_x = right_x

        target_x = right_x - self.right_offset
        error = target_x - (mask.shape[1] / 2.0)
        self.error_buffer.append(error)
        avg_error = np.mean(self.error_buffer)

        raw_angular = -self.pid.update(avg_error)
        angular = self.angular_alpha * self.last_angular + (1.0 - self.angular_alpha) * raw_angular
        self.last_angular = angular
        angular = max(-self.max_angular, min(self.max_angular, angular))

        abs_err = abs(avg_error)
        if abs_err < self.straight_threshold:
            speed = self.base_speed
            angular *= 0.35
        elif abs_err < self.curve_threshold:
            speed = 0.17
            angular *= 0.75
        else:
            speed = self.curve_speed
            angular *= 0.92

        if self.state == "FORWARD":
            speed = 0.04
            angular *= 0.4

        twist.linear.x = speed
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

    # ------------------------------------------------------------
    # 白线提取（增强抗反光，参数从 self 读取）
    # ------------------------------------------------------------
    def extract_white_mask(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        v_mean = np.mean(hsv[:, :, 2])
        dynamic_v = max(self.white_v_min, int(v_mean + 35))
        hsv_mask = cv2.inRange(hsv, (0, 0, dynamic_v), (179, self.white_s_max, 255))

        gray_mask = cv2.inRange(gray, self.gray_threshold, 255)

        mask = cv2.bitwise_and(hsv_mask, gray_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean = np.zeros_like(mask)
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < self.min_light_size and h < self.min_light_size:
                continue
            if area > mask.shape[0] * mask.shape[1] * self.max_reflect_ratio:
                continue
            cv2.drawContours(clean, [c], -1, 255, thickness=cv2.FILLED)
        return clean

    # ------------------------------------------------------------
    # 寻找右侧白线
    # ------------------------------------------------------------
    def find_right_line(self, mask):
        h = mask.shape[0]
        scan_rows = [
            int(h * 0.96), int(h * 0.92), int(h * 0.88),
            int(h * 0.84), int(h * 0.80), int(h * 0.76)
        ]
        weights = [6, 5, 4, 3, 2, 1]
        points = []
        total_weight = 0
        for i, y in enumerate(scan_rows):
            if y >= h:
                continue
            row = mask[y, :]
            idx = np.where(row > 0)[0]
            if len(idx) < 10:
                continue
            right_x = idx[-1]
            points.append(right_x * weights[i])
            total_weight += weights[i]
        if total_weight == 0:
            return None
        return sum(points) / total_weight

    # ------------------------------------------------------------
    # 横线检测
    # ------------------------------------------------------------
    def detect_horizontal_line(self, mask):
        h, w = mask.shape
        roi_start = int(h * 0.82)
        if roi_start >= h:
            return False
        bottom_roi = mask[roi_start:, :]
        rows = bottom_roi.shape[0]
        count = 0
        for y in range(rows):
            row = bottom_roi[y, :]
            white_pixels = np.sum(row > 0)
            if white_pixels > w * 0.55:
                count += 1
        return count >= 4

    # ------------------------------------------------------------
    # 停车
    # ------------------------------------------------------------
    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)


if __name__ == "__main__":
    try:
        node = StableRightFollowNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
