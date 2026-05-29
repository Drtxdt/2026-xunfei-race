#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyboard controlled YOLO dataset collector for ROS U-CAR.
功能：
1. 订阅相机话题 /usb_cam/image_raw
2. 按固定频率自动保存图片
3. 用键盘 WASD/QE 控制小车移动，让摄像头获得不同距离、角度、光照的数据

按键：
  W/S：前进/后退
  A/D：左转/右转
  Q/E：左平移/右平移，麦克纳姆轮可用
  Space 或 X：停车
  P：暂停/继续自动保存
  C：手动保存当前一张
  + / -：提高/降低拍照频率
  1 / 2：降低/提高移动速度
  H：显示帮助
  ESC 或 Ctrl-C：退出并停车
"""

import os
import sys
import time
import tty
import termios
import select
import argparse
import threading

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class KeyboardDatasetCollector:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_img = None
        self.count = 0
        self.last_save_time = 0.0
        self.save_enabled = not args.start_paused
        self.interval = args.interval

        self.base_linear = args.linear
        self.base_angular = args.angular
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0

        self.out_dir = os.path.join(os.path.expanduser(args.output), args.cls)
        os.makedirs(self.out_dir, exist_ok=True)

        self.cmd_pub = rospy.Publisher(args.cmd_topic, Twist, queue_size=1)
        self.image_sub = rospy.Subscriber(args.topic, Image, self.image_cb, queue_size=1)
        rospy.on_shutdown(self.on_shutdown)
    def print_help(self):
        print("""
================ YOLO 数据集键盘采集工具 ================
当前类别: {cls}
保存目录: {out}
图像话题: {topic}
速度话题: {cmd_topic}
基础速度: linear={lin:.3f} m/s, angular={ang:.3f} rad/s
拍照间隔: {interval:.2f} s
自动保存: {saving}

移动控制：
  W : 前进，改变目标在画面中的距离
  S : 后退，改变目标在画面中的距离
  A : 左转，改变摄像头朝向/左侧视角
  D : 右转，改变摄像头朝向/右侧视角
  Q : 左平移，麦克纳姆轮可用，用于改变横向视角
  E : 右平移，麦克纳姆轮可用，用于改变横向视角
  X / Space : 停车

采集控制：
  P : 暂停/继续自动按频率保存图片
  C : 手动保存当前一张图片
  + / = : 提高拍照频率，减小间隔
  - / _ : 降低拍照频率，增大间隔
  1 : 降低移动速度
  2 : 提高移动速度
  H : 显示帮助
  ESC / Ctrl+C : 退出并停车

建议：先按 P 暂停，摆好目标后再按 P 开始。采集过程中用 WASD 缓慢变换距离和角度。
==========================================================
""".format(
            cls=self.args.cls,
            out=self.out_dir,
            topic=self.args.topic,
            cmd_topic=self.args.cmd_topic,
            lin=self.base_linear,
            ang=self.base_angular,
            interval=self.interval,
            saving="ON" if self.save_enabled else "OFF"
        ))

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.args.flip:
                img = cv2.flip(img, 1)
            with self.lock:
                self.latest_img = img
        except Exception as e:
            rospy.logerr("image_cb error: %s", str(e))

    def save_latest(self, manual=False):
        with self.lock:
            if self.latest_img is None:
                rospy.logwarn("No image received yet. Check camera topic: %s", self.args.topic)
                return False
            img = self.latest_img.copy()

        now = time.time()
        prefix = "manual" if manual else self.args.cls
        filename = "%s_%06d_%d.jpg" % (prefix, self.count, int(now * 1000))
        path = os.path.join(self.out_dir, filename)
        ok = cv2.imwrite(path, img)
        if ok:
            self.count += 1
            self.last_save_time = now
            rospy.loginfo("saved %-45s total=%d", filename, self.count)
            return True
        rospy.logwarn("failed to save image: %s", path)
        return False

    def save_latest_if_needed(self):
        if not self.save_enabled:
            return
        if self.args.max_images > 0 and self.count >= self.args.max_images:
            rospy.loginfo("Reached max_images=%d. Auto saving paused and robot stopped.", self.args.max_images)
            self.save_enabled = False
            self.stop_robot()
            return
        now = time.time()
        if now - self.last_save_time >= self.interval:
            self.save_latest(manual=False)

    def publish_cmd(self):
        cmd = Twist()
        cmd.linear.x = self.linear_x
        cmd.linear.y = self.linear_y
        cmd.angular.z = self.angular_z
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0
        self.cmd_pub.publish(Twist())

    def on_shutdown(self):
        self.stop_robot()
        rospy.loginfo("shutdown, robot stopped. total saved=%d", self.count)

    def get_key(self, timeout=0.05):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None

    def handle_key(self, key):
        if key is None:
            return True
        if key == '\x1b':
            return False
        key = key.lower()

        if key == 'w':
            self.linear_x = self.base_linear
            self.linear_y = 0.0
            self.angular_z = 0.0
        elif key == 's':
            self.linear_x = -self.base_linear
            self.linear_y = 0.0
            self.angular_z = 0.0
        elif key == 'a':
            self.linear_x = 0.0
            self.linear_y = 0.0
            self.angular_z = self.base_angular
        elif key == 'd':
            self.linear_x = 0.0
            self.linear_y = 0.0
            self.angular_z = -self.base_angular
        elif key == 'q':
            self.linear_x = 0.0
            self.linear_y = self.base_linear
            self.angular_z = 0.0
        elif key == 'e':
            self.linear_x = 0.0
            self.linear_y = -self.base_linear
            self.angular_z = 0.0
        elif key == 'x' or key == ' ':
            self.stop_robot()
        elif key == 'p':
            self.save_enabled = not self.save_enabled
            rospy.loginfo("auto save: %s", "ON" if self.save_enabled else "OFF")
        elif key == 'c':
            self.save_latest(manual=True)
        elif key in ['+', '=']:
            self.interval = max(0.05, self.interval * 0.8)
            rospy.loginfo("interval=%.2fs", self.interval)
        elif key in ['-', '_']:
            self.interval = min(10.0, self.interval * 1.25)
            rospy.loginfo("interval=%.2fs", self.interval)
        elif key == '1':
            self.base_linear = max(0.01, self.base_linear * 0.8)
            self.base_angular = max(0.03, self.base_angular * 0.8)
            rospy.loginfo("speed down: linear=%.3f angular=%.3f", self.base_linear, self.base_angular)
        elif key == '2':
            self.base_linear = min(0.20, self.base_linear * 1.25)
            self.base_angular = min(0.80, self.base_angular * 1.25)
            rospy.loginfo("speed up: linear=%.3f angular=%.3f", self.base_linear, self.base_angular)
        elif key == 'h':
            self.print_help()
        return True

    def loop(self):
        self.print_help()
        old_attr = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        try:
            rate = rospy.Rate(self.args.rate)
            while not rospy.is_shutdown():
                key = self.get_key(timeout=0.01)
                if not self.handle_key(key):
                    break
                self.publish_cmd()
                self.save_latest_if_needed()
                rate.sleep()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attr)
            self.stop_robot()


def main():
    parser = argparse.ArgumentParser(description='Keyboard teleop + auto image collector for YOLO dataset')
    parser.add_argument('--cls', required=True, help='类别/采集会话名，例如 red_light、straight、left、right')
    parser.add_argument('--output', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo_dataset', 'raw_images'), help='图片保存根目录')
    parser.add_argument('--topic', default='/usb_cam/image_raw', help='相机图像话题')
    parser.add_argument('--cmd-topic', default='/cmd_vel', help='速度控制话题')
    parser.add_argument('--interval', type=float, default=0.5, help='自动保存间隔，单位秒')
    parser.add_argument('--max-images', type=int, default=0, help='最多保存张数，0表示不限制')
    parser.add_argument('--linear', type=float, default=0.04, help='基础线速度，建议0.03~0.06')
    parser.add_argument('--angular', type=float, default=0.18, help='基础角速度，建议0.15~0.25')
    parser.add_argument('--rate', type=float, default=20.0, help='控制循环频率')
    parser.add_argument('--flip', action='store_true', help='是否水平翻转图片，若正式检测节点翻转则建议加上')
    parser.add_argument('--start-paused', action='store_true', help='启动后先暂停保存，需要按P开始')
    args = parser.parse_args()

    rospy.init_node('keyboard_collect_yolo_images')
    node = KeyboardDatasetCollector(args)
    node.loop()


if __name__ == '__main__':
    main()
