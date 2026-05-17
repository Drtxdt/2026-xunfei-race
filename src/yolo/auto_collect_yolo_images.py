#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS YOLO dataset image auto collector for U-CAR.
Subscribe /usb_cam/image_raw, save frames, optionally publish /cmd_vel to sweep camera angles.
"""
import os, time, argparse, math
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class AutoCollector:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.last_save = 0.0
        self.count = 0
        self.start_time = time.time()
        self.out_dir = os.path.join(args.output, args.cls)
        os.makedirs(self.out_dir, exist_ok=True)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber(args.topic, Image, self.image_cb, queue_size=1)
        rospy.on_shutdown(self.stop_robot)
        rospy.loginfo('Saving images to: %s', self.out_dir)

    def image_cb(self, msg):
        now = time.time()
        if now - self.last_save < self.args.interval:
            return
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.args.flip:
            img = cv2.flip(img, 1)
        filename = '%s_%06d_%d.jpg' % (self.args.cls, self.count, int(now * 1000))
        cv2.imwrite(os.path.join(self.out_dir, filename), img)
        self.count += 1
        self.last_save = now
        rospy.loginfo('saved %s, total=%d', filename, self.count)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def move_loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.args.max_images > 0 and self.count >= self.args.max_images:
                rospy.loginfo('Reached max_images=%d, stop.', self.args.max_images)
                break
            cmd = Twist()
            if self.args.move:
                t = time.time() - self.start_time
                if self.args.mode == 'yaw':
                    cmd.angular.z = self.args.angular * math.sin(2 * math.pi * t / self.args.period)
                elif self.args.mode == 'forward_back':
                    cmd.linear.x = self.args.linear * math.sin(2 * math.pi * t / self.args.period)
                elif self.args.mode == 'circle':
                    cmd.linear.x = self.args.linear
                    cmd.angular.z = self.args.angular
            self.cmd_pub.publish(cmd)
            rate.sleep()
        self.stop_robot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', required=True, help='class/session name, e.g. red_light or food_sign')
    parser.add_argument('--output', default=os.path.expanduser('~/yolo_dataset/raw_images'))
    parser.add_argument('--topic', default='/usb_cam/image_raw')
    parser.add_argument('--interval', type=float, default=0.5, help='seconds between saved images')
    parser.add_argument('--max-images', type=int, default=300)
    parser.add_argument('--move', action='store_true', help='enable cmd_vel auto movement')
    parser.add_argument('--mode', choices=['yaw','forward_back','circle'], default='yaw')
    parser.add_argument('--linear', type=float, default=0.05)
    parser.add_argument('--angular', type=float, default=0.25)
    parser.add_argument('--period', type=float, default=8.0)
    parser.add_argument('--flip', action='store_true', help='same horizontal flip as yolov5_detect.py')
    args = parser.parse_args()
    rospy.init_node('auto_collect_yolo_images')
    AutoCollector(args).move_loop()
