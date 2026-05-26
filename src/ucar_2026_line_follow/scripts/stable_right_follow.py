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

        rospy.init_node("stable_right_follow")

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

        # 右边线目标位置（距离更远）
        self.target_right_x = 180

        # 速度
        self.base_speed = 0.17
        self.curve_speed = 0.20
        self.search_speed = 0.05

        # PID
        self.kp = 0.0028
        self.kd = 0.0008

        self.last_error = 0

        # 状态机
        self.stage = 0

        """
        0 = 起步直冲120cm
        1 = 低速右转找线
        2 = 稳定右巡线
        3 = 检测到横线后前进25cm
        4 = 最终停车
        """

        self.start_time = time.time()

        self.forward_25_start = None

        self.cross_detected = False

        rospy.loginfo("Stable Right Follow Node Started")


    # ==================================================
    # 图像回调
    # ==================================================

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

        twist = Twist()

        # ==================================================
        # 阶段0：强制直行120cm
        # ==================================================

        if self.stage == 0:

            elapsed = time.time() - self.start_time

            # 约1.9秒 ≈ 120cm
            if elapsed < 3.2:

                twist.linear.x = 0.42
                twist.angular.z = 0.0

                self.cmd_pub.publish(twist)

                return

            else:

                rospy.loginfo("ENTER SEARCH MODE")

                self.stage = 1

        # ==================================================
        # 阶段1：慢速右转找线
        # ==================================================

        right_x = self.find_right_line(mask)

        if self.stage == 1:

            if right_x is None:

                twist.linear.x = 0.05
                twist.angular.z = -0.22

                self.cmd_pub.publish(twist)

                return

            else:

                rospy.loginfo("RIGHT LINE FOUND")

                self.stage = 2

        # ==================================================
        # 阶段2：稳定右巡线
        # ==================================================

        if self.stage == 2:

            # 横线检测
            cross_area = np.sum(mask > 0)

            if cross_area > 42000:

                rospy.loginfo("STOP LINE DETECTED")

                self.stage = 3

                self.forward_25_start = time.time()

                return

            # 丢线处理
            if right_x is None:

                twist.linear.x = 0.04
                twist.angular.z = -0.18

                self.cmd_pub.publish(twist)

                return

            # ==========================
            # PID
            # ==========================

            error = self.target_right_x - right_x

            # 增大容错，避免左右摆
            error = (
                0.82 * self.last_error +
                0.18 * error
            )

            d_error = error - self.last_error

            self.last_error = error

            angular = (
                self.kp * error +
                self.kd * d_error
            )

            # 弯道
            if abs(error) > 45:

                linear_speed = self.curve_speed

                angular *= 1.05

            else:

                linear_speed = self.base_speed

            # 限制角速度
            angular = max(min(angular, 0.32), -0.32)

            twist.linear.x = linear_speed
            twist.angular.z = angular

            self.cmd_pub.publish(twist)

            # 调试窗口
            debug = cv2.cvtColor(
                mask,
                cv2.COLOR_GRAY2BGR
            )

            cv2.circle(
                debug,
                (right_x, int(mask.shape[0] / 2)),
                5,
                (0, 0, 255),
                -1
            )

            cv2.imshow("right_follow", debug)

            cv2.waitKey(1)

            return

        # ==================================================
        # 阶段3：再前进25cm
        # ==================================================

        if self.stage == 3:

            elapsed = time.time() - self.forward_25_start

            if elapsed < 1.0:

                twist.linear.x = 0.09
                twist.angular.z = 0.0

                self.cmd_pub.publish(twist)

                return

            else:

                self.stage = 4

        # ==================================================
        # 阶段4：停车
        # ==================================================

        if self.stage == 4:

            self.stop_car()

            rospy.loginfo("FINAL STOP")

            return


    # ==================================================
    # 白线提取
    # ==================================================

    def extract_white_mask(self, roi):

        blur = cv2.GaussianBlur(
            roi,
            (5, 5),
            0
        )

        hsv = cv2.cvtColor(
            blur,
            cv2.COLOR_BGR2HSV
        )

        # 抗反光
        lower_white = np.array([0, 0, 185])

        upper_white = np.array([180, 60, 255])

        mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        kernel = np.ones((3, 3), np.uint8)

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

        mask = cv2.medianBlur(mask, 5)

        # OpenCV3/4兼容
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

            # 去掉小反光
            if area > 180:

                cv2.drawContours(
                    clean_mask,
                    [cnt],
                    -1,
                    255,
                    -1
                )

        return clean_mask


    # ==================================================
    # 找右边线
    # ==================================================

    def find_right_line(self, mask):

        h, w = mask.shape

        search_y = int(h * 0.55)

        row = mask[search_y]

        white_points = np.where(row > 0)[0]

        if len(white_points) == 0:

            return None

        return np.max(white_points)


    # ==================================================
    # 停车
    # ==================================================

    def stop_car(self):

        twist = Twist()

        twist.linear.x = 0.0
        twist.angular.z = 0.0

        self.cmd_pub.publish(twist)


    # ==================================================
    # 主循环
    # ==================================================

    def run(self):

        rospy.spin()


if __name__ == "__main__":

    try:

        node = StableRightFollowNode()

        node.run()

    except rospy.ROSInterruptException:

        pass