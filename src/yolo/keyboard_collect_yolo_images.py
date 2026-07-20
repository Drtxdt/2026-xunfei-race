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
import json
from datetime import datetime
from pathlib import Path

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


VALID_CAPTURE_CLASSES = (
    'green_left',
    'green_right',
    'green_straight',
    'red_light',
    'background',
)


def record_capture_run(args):
    """Append reproducibility metadata without mixing legacy ROS workspaces."""
    root = Path(os.path.expanduser(args.output)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / '_capture_manifest.json'
    data = {'runs': []}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding='utf-8'))
            if isinstance(loaded, dict) and isinstance(loaded.get('runs'), list):
                data = loaded
        except Exception as exc:
            rospy.logwarn("Ignoring invalid capture manifest %s: %s", manifest_path, exc)
    data['runs'].append({
        'started_at': datetime.now().astimezone().isoformat(),
        'class_name': args.cls,
        'topic': args.topic,
        'cmd_topic': args.cmd_topic,
        'interval_sec': args.interval,
        'max_images': args.max_images,
        'flip': bool(args.flip),
        'output': str(root),
    })
    temp_path = manifest_path.with_suffix('.json.tmp')
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(str(temp_path), str(manifest_path))


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
红绿灯方向数据默认禁止水平翻转；左右箭头会因翻转交换语义。
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
            if self.args.cls == 'background':
                label_path = os.path.splitext(path)[0] + '.txt'
                try:
                    with open(label_path, 'w'):
                        pass
                except OSError as exc:
                    rospy.logerr("Failed to create empty background label %s: %s", label_path, exc)
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    return False
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


def _resolve_arg(name, cli_val):
    """Return cli_val if explicitly set, otherwise fall back to ROS param ~name."""
    if cli_val is not None:
        return cli_val
    return rospy.get_param('~' + name, None)


def main():
    parser = argparse.ArgumentParser(description='Keyboard teleop + auto image collector for YOLO dataset')
    parser.add_argument('--cls', default=None, help='类别/采集会话名，例如 red_light')
    parser.add_argument('--output', default=None, help='图片保存根目录')
    parser.add_argument('--topic', default=None, help='相机图像话题')
    parser.add_argument('--cmd-topic', default=None, help='速度控制话题')
    parser.add_argument('--interval', type=float, default=None, help='自动保存间隔，单位秒')
    parser.add_argument('--max-images', type=int, default=None, help='最多保存张数，0表示不限制')
    parser.add_argument('--linear', type=float, default=None, help='基础线速度，建议0.03~0.06')
    parser.add_argument('--angular', type=float, default=None, help='基础角速度，建议0.15~0.25')
    parser.add_argument('--rate', type=float, default=None, help='控制循环频率')
    parser.add_argument('--flip', action='store_true', default=None,
                        help='危险兼容参数；红绿灯左右方向数据禁止使用')
    parser.add_argument('--allow-horizontal-flip', action='store_true', default=None,
                        help='明确允许水平翻转；红绿灯四分类数据禁止使用')
    parser.add_argument('--start-paused', action='store_true', default=None, help='启动后先暂停保存，按P开始')
    cli, _ = parser.parse_known_args()

    rospy.init_node('keyboard_collect_yolo_images')

    # Resolve each arg: CLI takes priority, else ROS param, else default below
    script_dir = os.path.dirname(os.path.abspath(__file__))
    args = argparse.Namespace(
        cls=_resolve_arg('cls', cli.cls) or 'red_light',
        output=_resolve_arg('output', cli.output)
            or os.path.join(script_dir, 'yolo_dataset', 'raw_images'),
        topic=_resolve_arg('topic', cli.topic) or '/usb_cam/image_raw',
        cmd_topic=_resolve_arg('cmd_topic', cli.cmd_topic) or '/cmd_vel',
        interval=float(_resolve_arg('interval', cli.interval) or 0.5),
        max_images=int(_resolve_arg('max_images', cli.max_images) or 0),
        linear=float(_resolve_arg('linear', cli.linear) or 0.04),
        angular=float(_resolve_arg('angular', cli.angular) or 0.18),
        rate=float(_resolve_arg('rate', cli.rate) or 20.0),
        flip=bool(_resolve_arg('flip', cli.flip) or False),
        allow_horizontal_flip=bool(
            _resolve_arg('allow_horizontal_flip', cli.allow_horizontal_flip) or False
        ),
        start_paused=bool(_resolve_arg('start_paused', cli.start_paused) or False),
    )

    if args.cls not in VALID_CAPTURE_CLASSES:
        rospy.logfatal(
            "Unsupported --cls=%s. Expected one of: %s",
            args.cls,
            ', '.join(VALID_CAPTURE_CLASSES),
        )
        return 2
    if args.flip and not args.allow_horizontal_flip:
        rospy.logfatal(
            "Horizontal flip is blocked for traffic-light capture because it swaps left/right semantics. "
            "Use flip=false. Only pass --allow-horizontal-flip for a deliberately mirrored end-to-end pipeline."
        )
        return 2

    record_capture_run(args)
    node = KeyboardDatasetCollector(args)
    node.loop()
    return 0


if __name__ == '__main__':
    sys.exit(main())
