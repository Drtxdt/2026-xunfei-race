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
            queue_size=1,
            buff_size=2**24
        )

        # ==================================================
        # 参数
        # ==================================================

        # =========================
        # 巡线目标
        # =========================

        # 往左偏一点
        # 防止贴右边线
        self.target_right_x = 155

        # =========================
        # 速度参数
        # =========================

        # 直线速度
        self.base_speed = 0.32

        # 弯道速度
        self.curve_speed = 0.28

        # 丢线搜索速度
        self.search_speed = 0.10

        # =========================
        # PID参数
        # =========================

        # 更激进
        self.kp = 0.0052
        self.kd = 0.0018

        # PID缓存
        self.last_error = 0
        self.filtered_error = 0

        # =========================
        # 状态机
        # =========================

        self.stage = 0

        """
        0 = 起步直行
        1 = 右转找线
        2 = 稳定右巡线
        3 = 检测横线后前进
        4 = 停车
        """

        self.start_time = time.time()

        self.forward_25_start = None

        # 丢线缓存
        self.last_right_x = None

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

        # ==================================================
        # ROI
        # ==================================================

        roi = frame[int(h * 0.60):h, :]

        # ==================================================
        # 白线提取
        # ==================================================

        mask = self.extract_white_mask(roi)

        twist = Twist()

        # ==================================================
        # 阶段0：起步直行
        # ==================================================

        if self.stage == 0:

            elapsed = time.time() - self.start_time

            if elapsed < 2.8:

                twist.linear.x = 0.45
                twist.angular.z = 0.0

                self.cmd_pub.publish(twist)

                return

            else:

                rospy.loginfo("ENTER SEARCH MODE")

                self.stage = 1

        # ==================================================
        # 找右边线
        # ==================================================

        right_x = self.find_right_line(mask)

        # ==================================================
        # 阶段1：快速右转找线
        # ==================================================

        if self.stage == 1:

            if right_x is None:

                twist.linear.x = 0.10
                twist.angular.z = -0.26

                self.cmd_pub.publish(twist)

                return

            else:

                rospy.loginfo("RIGHT LINE FOUND")

                self.last_right_x = right_x

                self.stage = 2

        # ==================================================
        # 阶段2：稳定右巡线
        # ==================================================

        if self.stage == 2:

            # ==============================================
            # 横线检测
            # ==============================================

            cross_area = np.sum(mask > 0)

            if cross_area > 48000:

                rospy.loginfo("STOP LINE DETECTED")

                self.stage = 3

                self.forward_25_start = time.time()

                return

            # ==============================================
            # 丢线处理
            # ==============================================

            if right_x is None:

                # 使用上一次位置继续推测
                if self.last_right_x is not None:

                    twist.linear.x = 0.14
                    twist.angular.z = -0.22

                else:

                    twist.linear.x = 0.10
                    twist.angular.z = -0.24

                self.cmd_pub.publish(twist)

                return

            self.last_right_x = right_x

            # ==============================================
            # PID
            # ==============================================

            error = self.target_right_x - right_x

            # ==============================================
            # 低通滤波
            # ==============================================

            alpha = 0.22

            self.filtered_error = (
                (1 - alpha) * self.filtered_error +
                alpha * error
            )

            d_error = (
                self.filtered_error -
                self.last_error
            )

            self.last_error = self.filtered_error

            # ==============================================
            # PID输出
            # ==============================================

            angular = (
                self.kp * self.filtered_error +
                self.kd * d_error
            )

            # ==============================================
            # 弯道增强
            # ==============================================

            if abs(self.filtered_error) > 38:

                linear_speed = self.curve_speed

                angular *= 1.18

            else:

                linear_speed = self.base_speed

            # ==============================================
            # 限制角速度
            # ==============================================

            angular = max(
                min(angular, 0.55),
                -0.55
            )

            twist.linear.x = linear_speed
            twist.angular.z = angular

            self.cmd_pub.publish(twist)

            # ==============================================
            # 调试显示
            # ==============================================

            debug = cv2.cvtColor(
                mask,
                cv2.COLOR_GRAY2BGR
            )

            cv2.line(
                debug,
                (self.target_right_x, 0),
                (self.target_right_x, mask.shape[0]),
                (255, 0, 0),
                2
            )

            cv2.circle(
                debug,
                (right_x, int(mask.shape[0] / 2)),
                5,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                debug,
                "ERR:{:.1f}".format(self.filtered_error),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow("right_follow", debug)

            cv2.waitKey(1)

            return

        # ==================================================
        # 阶段3：前进25cm
        # ==================================================

        if self.stage == 3:

            elapsed = time.time() - self.forward_25_start

            if elapsed < 0.9:

                twist.linear.x = 0.12
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

        # ==============================================
        # 高斯滤波
        # ==============================================

        blur = cv2.GaussianBlur(
            roi,
            (5, 5),
            0
        )

        # ==============================================
        # HSV
        # ==============================================

        hsv = cv2.cvtColor(
            blur,
            cv2.COLOR_BGR2HSV
        )

        # ==============================================
        # 更强抗灯光
        # ==============================================

        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 45, 255])

        mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        # ==============================================
        # 形态学滤波
        # ==============================================

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

        # ==============================================
        # 中值滤波
        # ==============================================

        mask = cv2.medianBlur(mask, 5)

        # ==============================================
        # 再次平滑
        # ==============================================

        mask = cv2.GaussianBlur(
            mask,
            (5, 5),
            0
        )

        # ==============================================
        # 轮廓提取
        # ==============================================

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

            # 去除小反光
            if area > 260:

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

        # ==============================================
        # 多行采样
        # ==============================================

        rows = [
            int(h * 0.50),
            int(h * 0.60),
            int(h * 0.70)
        ]

        points = []

        for y in rows:

            row = mask[y]

            white_points = np.where(row > 0)[0]

            if len(white_points) > 0:

                points.append(np.max(white_points))

        if len(points) == 0:

            return None

        # ==============================================
        # 多行均值
        # ==============================================

        return int(np.mean(points))

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