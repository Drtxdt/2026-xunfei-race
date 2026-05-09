#!/usr/bin/env python3
import time

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


class RightDebugRelayNode:
    def __init__(self):
        self.bridge = CvBridge()

        self.debug_image_topic = rospy.get_param(
            "~debug_image_topic", rospy.get_param("debug_image_topic", "/right_line_follow/debug_image")
        )
        self.raw_image_topic = rospy.get_param(
            "~raw_image_topic", rospy.get_param("raw_image_topic", "/usb_cam/image_raw")
        )
        self.compressed_debug_topic = rospy.get_param(
            "~compressed_debug_topic",
            rospy.get_param("compressed_debug_topic", "/right_line_follow/debug_image/compressed"),
        )
        self.compressed_raw_topic = rospy.get_param(
            "~compressed_raw_topic",
            rospy.get_param("compressed_raw_topic", "/right_line_follow/raw_image/compressed"),
        )
        self.status_topic = rospy.get_param(
            "~debug_relay_status_topic",
            rospy.get_param("debug_relay_status_topic", "/right_line_follow/debug_relay/status"),
        )

        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", rospy.get_param("jpeg_quality", 75)))
        self.jpeg_quality = max(20, min(95, self.jpeg_quality))
        self.resize_width = int(rospy.get_param("~debug_resize_width", rospy.get_param("debug_resize_width", 0)))
        self.max_publish_fps = float(
            rospy.get_param("~debug_max_publish_fps", rospy.get_param("debug_max_publish_fps", 12.0))
        )
        self.publish_raw_compressed = bool(
            rospy.get_param("~publish_raw_compressed", rospy.get_param("publish_raw_compressed", True))
        )

        self.debug_pub = rospy.Publisher(self.compressed_debug_topic, CompressedImage, queue_size=1)
        self.raw_pub = rospy.Publisher(self.compressed_raw_topic, CompressedImage, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1)

        self.last_debug_publish = 0.0
        self.last_raw_publish = 0.0
        self.debug_frames = 0
        self.raw_frames = 0

        self.debug_sub = rospy.Subscriber(self.debug_image_topic, Image, self.debug_callback, queue_size=1, buff_size=2**24)
        if self.publish_raw_compressed:
            self.raw_sub = rospy.Subscriber(self.raw_image_topic, Image, self.raw_callback, queue_size=1, buff_size=2**24)
        else:
            self.raw_sub = None

        rospy.Timer(rospy.Duration(1.0), self.publish_status)
        rospy.loginfo(
            "right_debug_relay started. debug=%s -> %s raw=%s -> %s quality=%d resize_width=%d",
            self.debug_image_topic,
            self.compressed_debug_topic,
            self.raw_image_topic,
            self.compressed_raw_topic,
            self.jpeg_quality,
            self.resize_width,
        )

    def debug_callback(self, msg: Image):
        now = time.time()
        if not self.should_publish(now, self.last_debug_publish):
            return
        compressed = self.compress_image(msg, now)
        if compressed is None:
            return
        self.last_debug_publish = now
        self.debug_frames += 1
        self.debug_pub.publish(compressed)

    def raw_callback(self, msg: Image):
        now = time.time()
        if not self.should_publish(now, self.last_raw_publish):
            return
        compressed = self.compress_image(msg, now)
        if compressed is None:
            return
        self.last_raw_publish = now
        self.raw_frames += 1
        self.raw_pub.publish(compressed)

    def should_publish(self, now: float, last_publish: float) -> bool:
        if self.max_publish_fps <= 0.0:
            return True
        return now - last_publish >= 1.0 / self.max_publish_fps

    def compress_image(self, msg: Image, now: float):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "debug relay cv_bridge conversion failed: %s", exc)
            return None

        if self.resize_width > 0 and frame.shape[1] > self.resize_width:
            scale = float(self.resize_width) / float(frame.shape[1])
            frame = cv2.resize(frame, (self.resize_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        ok, data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            rospy.logwarn_throttle(2.0, "debug relay JPEG encode failed")
            return None

        out = CompressedImage()
        out.header = msg.header
        out.header.stamp = rospy.Time.from_sec(now)
        out.format = "jpeg"
        out.data = data.tobytes()
        return out

    def publish_status(self, _event):
        self.status_pub.publish(
            String(
                data=(
                    "debug_frames={} raw_frames={} debug_topic={} raw_topic={} quality={} resize_width={}"
                ).format(
                    self.debug_frames,
                    self.raw_frames,
                    self.compressed_debug_topic,
                    self.compressed_raw_topic if self.publish_raw_compressed else "disabled",
                    self.jpeg_quality,
                    self.resize_width,
                )
            )
        )


def main():
    rospy.init_node("right_debug_relay_node")
    RightDebugRelayNode()
    rospy.spin()


if __name__ == "__main__":
    main()
