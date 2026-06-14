#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
厂区标识牌 YOLO 数据集采集脚本
===============================
用途：为子任务2的厂区图文标识牌识别模型采集训练数据

三类厂区标识牌：
  1. factory_food        — 食品加工车间
  2. factory_daily       — 日用品加工车间
  3. factory_electronic  — 电子产品生产车间

============================================================================
使用方式
============================================================================

【终端1 — 启动底盘+摄像头+视频流（一键）】
  source ~/ucar_ws/devel/setup.bash
  source ~/2026-xunfei-race/devel/setup.bash
  roslaunch yolo traffic_light_collect.launch flip:=true

【终端2 — 启动采集脚本】
  source ~/ucar_ws/devel/setup.bash
  source ~/2026-xunfei-race/devel/setup.bash
  rosrun yolo collect_factory_sign.py

============================================================================
键盘操作
============================================================================

  移动控制（WASD，与键盘采集工具一致）：
    W/S    — 前进/后退
    A/D    — 左转/右转
    Q/E    — 左平移/右平移
    X/Space — 停车

  采集控制：
    P      — 暂停/继续自动保存
    C      — 手动保存当前一张
    R      — 开启/关闭自动旋转采集（车原地慢转，覆盖多角度）
    1/2/3  — 切换当前类别：1=食品车间 2=日用品车间 3=电子产品车间
    +/-    — 提高/降低拍照频率
    7/8    — 降低/提高移动速度
    H      — 显示帮助
    ESC / Ctrl+C — 退出并停车

============================================================================
建议采集流程
============================================================================

  1. 将一块厂区标识牌立在实机前的不同距离（0.5m、1m、1.5m、2m）
  2. 按对应数字键选择类别
  3. 先按 P 暂停，用 WASD 移动到合适位置
  4. 按 R 开启自动旋转，小车原地慢转覆盖多角度
  5. 旋转一圈后按 R 关闭，换下一个距离
  6. 换标识牌，按数字键切换到下一类别，重复以上步骤
  7. 每个类别建议采集 150~300 张

============================================================================
输出目录结构
============================================================================

  yolo_dataset/raw_images/
  ├── factory_food/        # 食品加工车间标识牌
  ├── factory_daily/       # 日用品加工车间标识牌
  └── factory_electronic/  # 电子产品生产车间标识牌
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


# ── 三类厂区标识牌 ──────────────────────────────────────────
FACTORY_CLASSES = {
    '1': 'factory_food',
    '2': 'factory_daily',
    '3': 'factory_electronic',
}

CLASS_LABELS = {
    'factory_food':       '食品加工车间',
    'factory_daily':      '日用品加工车间',
    'factory_electronic': '电子产品生产车间',
}


class FactorySignCollector:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # ── 图像 ──
        self.latest_img = None
        self.count = 0
        self.last_save_time = 0.0

        # ── 采集状态 ──
        self.save_enabled = not args.start_paused
        self.interval = args.interval
        self.current_cls = args.cls
        self.out_dir = self._make_out_dir()

        # ── 自动旋转模式 ──
        self.auto_rotate = False
        self.rotate_angular = 0.22       # 原地慢转角速度 rad/s

        # ── 速度 ──
        self.base_linear = args.linear
        self.base_angular = args.angular
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.angular_z = 0.0

        # ── ROS ──
        self.cmd_pub = rospy.Publisher(args.cmd_topic, Twist, queue_size=1)
        self.image_sub = rospy.Subscriber(args.topic, Image, self.image_cb, queue_size=1)
        rospy.on_shutdown(self.on_shutdown)

    # ── 目录 ──────────────────────────────────────────────

    def _make_out_dir(self):
        d = os.path.join(os.path.expanduser(self.args.output), self.current_cls)
        os.makedirs(d, exist_ok=True)
        return d

    def _switch_class(self, cls_key):
        if cls_key not in FACTORY_CLASSES:
            return
        self.current_cls = FACTORY_CLASSES[cls_key]
        self.out_dir = self._make_out_dir()
        rospy.loginfo("========================================")
        rospy.loginfo("切换到类别: %s (%s)", self.current_cls, CLASS_LABELS[self.current_cls])
        rospy.loginfo("保存目录:   %s", self.out_dir)
        rospy.loginfo("========================================")

    # ── 相机回调 ──────────────────────────────────────────

    def image_cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.args.flip:
                img = cv2.flip(img, 1)
            with self.lock:
                self.latest_img = img
        except Exception as e:
            rospy.logerr("image_cb error: %s", str(e))

    # ── 保存 ──────────────────────────────────────────────

    def save_latest(self, manual=False):
        with self.lock:
            if self.latest_img is None:
                rospy.logwarn("未收到图像，请检查摄像头话题: %s", self.args.topic)
                return False
            img = self.latest_img.copy()

        now = time.time()
        prefix = "manual" if manual else self.current_cls
        filename = "%s_%06d_%d.jpg" % (prefix, self.count, int(now * 1000))
        path = os.path.join(self.out_dir, filename)
        ok = cv2.imwrite(path, img)
        if ok:
            self.count += 1
            self.last_save_time = now
            rospy.loginfo("[%s] saved %-45s total=%d",
                          self.current_cls, filename, self.count)
            return True
        rospy.logwarn("保存失败: %s", path)
        return False

    def save_latest_if_needed(self):
        if not self.save_enabled:
            return
        if self.args.max_images > 0 and self.count >= self.args.max_images:
            rospy.loginfo("已达 max_images=%d，自动保存暂停并停车。", self.args.max_images)
            self.save_enabled = False
            self.stop_robot()
            return
        now = time.time()
        if now - self.last_save_time >= self.interval:
            self.save_latest(manual=False)

    # ── 运动控制 ──────────────────────────────────────────

    def publish_cmd(self):
        cmd = Twist()

        if self.auto_rotate:
            # 自动旋转模式：覆盖 angular，忽略 A/D 输入
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = self.rotate_angular
        else:
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
        rospy.loginfo("采集结束。本次共保存 %d 张 [类别: %s]", self.count, self.current_cls)

    # ── 键盘 ──────────────────────────────────────────────

    def get_key(self, timeout=0.05):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            return sys.stdin.read(1)
        return None

    def handle_key(self, key):
        if key is None:
            return True
        if key == '\x1b':          # ESC
            return False
        key = key.lower()

        # ── 类别切换 ──
        if key in FACTORY_CLASSES:
            self._switch_class(key)

        # ── 移动 ──
        elif key == 'w':
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

        # ── 自动旋转 ──
        elif key == 'r':
            self.auto_rotate = not self.auto_rotate
            if self.auto_rotate:
                self.stop_robot()   # 先停车再开始旋转
                rospy.loginfo(" 自动旋转模式开启，小车将原地慢转覆盖多角度")
            else:
                self.stop_robot()
                rospy.loginfo(" 自动旋转模式关闭")

        # ── 采集控制 ──
        elif key == 'p':
            self.save_enabled = not self.save_enabled
            rospy.loginfo("自动保存: %s", "开启" if self.save_enabled else "暂停")
        elif key == 'c':
            self.save_latest(manual=True)

        # ── 频率 ──
        elif key in ['+', '=']:
            self.interval = max(0.05, self.interval * 0.8)
            rospy.loginfo("拍照间隔=%.2fs", self.interval)
        elif key in ['-', '_']:
            self.interval = min(10.0, self.interval * 1.25)
            rospy.loginfo("拍照间隔=%.2fs", self.interval)

        # ── 速度 ──
        elif key == '7':
            self.base_linear = max(0.01, self.base_linear * 0.8)
            self.base_angular = max(0.03, self.base_angular * 0.8)
            rospy.loginfo("减速: linear=%.3f angular=%.3f", self.base_linear, self.base_angular)
        elif key == '8':
            self.base_linear = min(0.20, self.base_linear * 1.25)
            self.base_angular = min(0.60, self.base_angular * 1.25)
            rospy.loginfo("加速: linear=%.3f angular=%.3f", self.base_linear, self.base_angular)

        # ── 帮助 ──
        elif key == 'h':
            self.print_help()

        return True

    # ── 帮助 ──────────────────────────────────────────────

    def print_help(self):
        print("""
╔══════════════════════════════════════════════════════════════╗
║         厂区标识牌 YOLO 数据集采集工具                       ║
╠══════════════════════════════════════════════════════════════╣
║  当前类别 : {cls:<18s} ({label})           ║
║  保存目录 : {out}                                         ║
║  自动保存 : {saving}                                         ║
║  拍照间隔 : {interval:.2f}s                                       ║
║  自动旋转 : {rotate}                                         ║
║  已保存张数: {total}                                           ║
╠══════════════════════════════════════════════════════════════╣
║  移动控制:                                                   ║
║    W/S    前进/后退      A/D    左转/右转                    ║
║    Q/E    左平移/右平移   X/Space 停车                       ║
╠══════════════════════════════════════════════════════════════╣
║  采集控制:                                                   ║
║    1      切换到 食品加工车间 标识牌                          ║
║    2      切换到 日用品加工车间 标识牌                        ║
║    3      切换到 电子产品生产车间 标识牌                      ║
║    P      暂停/继续自动保存                                  ║
║    C      手动保存当前帧                                     ║
║    R      开启/关闭自动旋转（原地慢转，覆盖多角度）           ║
║    +/-    增减拍照频率                                       ║
║    7/8    减速/加速                                          ║
║    H      显示本帮助                                         ║
║    ESC    退出并停车                                         ║
╠══════════════════════════════════════════════════════════════╣
║  建议操作:                                                   ║
║    ① 标识牌立在车前 0.5~2m                                  ║
║    ② 按数字键 1/2/3 选类别                                  ║
║    ③ 按 R 开启自动旋转，覆盖多角度                           ║
║    ④ 圈满后按 R 关闭，移动距离，重复                         ║
║    ⑤ 换标识牌，换类别，重复①~④                              ║
╚══════════════════════════════════════════════════════════════╝
""".format(
            cls=self.current_cls,
            label=CLASS_LABELS.get(self.current_cls, '?'),
            out=self.out_dir,
            saving="开启" if self.save_enabled else "暂停",
            interval=self.interval,
            rotate="开启 (%.2f rad/s)" % self.rotate_angular if self.auto_rotate else "关闭",
            total=self.count,
        ))

    # ── 主循环 ────────────────────────────────────────────

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


# ── 参数解析 ──────────────────────────────────────────────

def _resolve_arg(name, cli_val):
    if cli_val is not None:
        return cli_val
    return rospy.get_param('~' + name, None)


def main():
    parser = argparse.ArgumentParser(
        description='厂区标识牌 YOLO 数据集键盘采集工具'
    )
    parser.add_argument('--cls', default='factory_food',
                        help='起始类别: factory_food / factory_daily / factory_electronic')
    parser.add_argument('--output', default=None,
                        help='图片保存根目录（默认 yolo_dataset/raw_images/）')
    parser.add_argument('--topic', default=None,
                        help='相机图像话题（默认 /usb_cam/image_raw）')
    parser.add_argument('--cmd-topic', default=None,
                        help='速度控制话题（默认 /cmd_vel）')
    parser.add_argument('--interval', type=float, default=None,
                        help='自动保存间隔秒（默认 0.5）')
    parser.add_argument('--max-images', type=int, default=None,
                        help='最多保存张数，0=不限制')
    parser.add_argument('--linear', type=float, default=None,
                        help='基础线速度 m/s（默认 0.04）')
    parser.add_argument('--angular', type=float, default=None,
                        help='基础角速度 rad/s（默认 0.18）')
    parser.add_argument('--rate', type=float, default=None,
                        help='控制循环频率 Hz（默认 20）')
    parser.add_argument('--flip', action='store_true', default=None,
                        help='水平翻转图像（默认开启）')
    parser.add_argument('--no-flip', action='store_true',
                        help='不翻转图像')
    parser.add_argument('--start-paused', action='store_true', default=None,
                        help='启动后先暂停，按 P 开始')
    cli, _ = parser.parse_known_args()

    rospy.init_node('collect_factory_sign')

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # flip: --no-flip 优先，否则默认开启
    if cli.no_flip:
        default_flip = False
    else:
        default_flip = (_resolve_arg('flip', cli.flip) is not False)  # 默认 True

    args = argparse.Namespace(
        cls=_resolve_arg('cls', cli.cls) or 'factory_food',
        output=_resolve_arg('output', cli.output)
            or os.path.join(script_dir, 'yolo_dataset', 'raw_images'),
        topic=_resolve_arg('topic', cli.topic) or '/usb_cam/image_raw',
        cmd_topic=_resolve_arg('cmd_topic', cli.cmd_topic) or '/cmd_vel',
        interval=float(_resolve_arg('interval', cli.interval) or 0.5),
        max_images=int(_resolve_arg('max_images', cli.max_images) or 0),
        linear=float(_resolve_arg('linear', cli.linear) or 0.04),
        angular=float(_resolve_arg('angular', cli.angular) or 0.18),
        rate=float(_resolve_arg('rate', cli.rate) or 20.0),
        flip=default_flip,
        start_paused=bool(_resolve_arg('start_paused', cli.start_paused) or False),
    )

    node = FactorySignCollector(args)
    node.loop()


if __name__ == '__main__':
    main()
