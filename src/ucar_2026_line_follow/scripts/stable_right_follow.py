#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import time

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


class StableRightFollowNode:
    def __init__(self):
        rospy.init_node("stable_right_follow", anonymous=False)
        self.bridge = CvBridge()

        self.cmd_pub = rospy.Publisher(
            "/cmd_vel",
            Twist,
            queue_size=1
        )

        rospy.Subscriber(
            "/usb_cam/image_raw",
            Image,
            self.image_callback,
            queue_size=1
        )

        # =============================
        # 参数
        # =============================

        self.target_right_x = 235

        self.base_speed = 0.18
        self.curve_speed = 0.24
        self.search_speed = 0.08

        self.kp = 0.0035
        self.kd = 0.0010

        self.last_error = 0

        self.line_lost_count = 0

        self.stop_count = 0
        self.stop_done = False

        self.cross_detected = False

        self.forward_after_stop = False
        self.forward_start_time = None

        rospy.loginfo("Stable Right Follow Node Started")

        rospy.loginfo("Stable Right Follow Node Started (with YAML support)")
        rospy.loginfo("Parameters: offset=%d, base_speed=%.2f, curve_speed=%.2f",
                      self.right_offset, self.base_speed, self.curve_speed)

    # ------------------------------------------------------------
    # 图像回调主逻辑（与之前完全相同，无需修改）
    # ------------------------------------------------------------
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                "bgr8"
            )

        except Exception as e:
            rospy.logerr(e)
            return

        h, w = frame.shape[:2]

        roi = frame[int(h * 0.60):h, :]

        mask = self.extract_white_mask(roi)

        # ========================================
        # 横线检测（停车）
        # ========================================

        if not self.stop_done:

            cross_area = np.sum(mask > 0)

            if cross_area > 45000:

                if not self.cross_detected:

                    self.cross_detected = True

                    rospy.loginfo("FIRST CROSS DETECTED")

                    self.stop_car()

                    rospy.sleep(0.5)

                    self.forward_after_stop = True

                    self.forward_start_time = time.time()

                return

        # ========================================
        # 停车后前进25cm
        # ========================================

        if self.forward_after_stop:

            now = time.time()

            if now - self.forward_start_time < 1.5:

                twist = Twist()
                twist.linear.x = 0.08
                twist.angular.z = 0.0

                self.cmd_pub.publish(twist)

            else:

                self.stop_car()

                self.forward_after_stop = False
                self.stop_done = True

                rospy.loginfo("FINAL STOP")

            return

        # ========================================
        # 右边线检测
        # ========================================

        right_x = self.find_right_line(mask)
        twist = Twist()

        # ========================================
        # 丢线处理
        # ========================================

        if right_x is None:

            self.line_lost_count += 1

            twist.linear.x = self.search_speed

            # 慢速向右寻找
            twist.angular.z = -0.25

            self.cmd_pub.publish(twist)
            return

        self.line_lost_count = 0

        # ========================================
        # PID
        # ========================================

        error = self.target_right_x - right_x

        # 低通滤波，防止左右摇摆
        error = 0.7 * self.last_error + 0.3 * error

        d_error = error - self.last_error

        self.last_error = error

        angular = self.kp * error + self.kd * d_error

        # ========================================
        # 弯道判断
        # ========================================

        if abs(error) > 45:

            linear_speed = self.curve_speed

            angular *= 1.2

        else:

            linear_speed = self.base_speed

        # 限幅
        angular = max(min(angular, 0.45), -0.45)

        twist.linear.x = linear_speed
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

        # 调试窗口
        debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if right_x is not None:

            cv2.circle(
                debug,
                (right_x, int(mask.shape[0] / 2)),
                6,
                (0, 0, 255),
                -1
            )

        cv2.imshow("right_follow_mask", debug)
        cv2.waitKey(1)

    # =====================================================
    # 提取白线
    # =====================================================

    def extract_white_mask(self, roi):

        blur = cv2.GaussianBlur(roi, (5, 5), 0)

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # 抗反光
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 70, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((3, 3), np.uint8)

        # 去噪
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # 连线
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # 中值滤波
        mask = cv2.medianBlur(mask, 5)

        # ==================================================
        # OpenCV3/OpenCV4兼容
        # ==================================================

        contours_info = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours_info) == 3:

            _, contours, _ = contours_info

        else:

            contours, _ = contours_info

        clean_mask = np.zeros_like(mask)

        for cnt in contours:

            area = cv2.contourArea(cnt)

            # 去除反光小区域
            if area > 200:

                cv2.drawContours(
                    clean_mask,
                    [cnt],
                    -1,
                    255,
                    -1
                )

        return clean_mask

    # ------------------------------------------------------------
    # 寻找右侧白线
    # ------------------------------------------------------------
    def find_right_line(self, mask):

        h, w = mask.shape

        search_y = int(h * 0.5)

        row = mask[search_y]

        white_points = np.where(row > 0)[0]

        if len(white_points) == 0:

            return None

        # 最右边白线
        right_x = np.max(white_points)

        return right_x

    # =====================================================
    # 停车
    # =====================================================

    def stop_car(self):

        twist = Twist()

        twist.linear.x = 0
        twist.angular.z = 0

        self.cmd_pub.publish(twist)

    # =====================================================
    # 主循环
    # =====================================================

    def run(self):

        rospy.spin()


if __name__ == "__main__":

    try:

        node = StableRightFollowNode()

        node.run()

    except rospy.ROSInterruptException:

        pass