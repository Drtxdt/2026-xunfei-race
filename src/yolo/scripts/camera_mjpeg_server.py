#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serve a ROS Image topic as an MJPEG stream for browser debugging."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class FrameStore:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.jpeg: Optional[bytes] = None
        self.seq = 0
        self.last_stamp = 0.0

    def update(self, jpeg: bytes) -> None:
        with self.cond:
            self.jpeg = jpeg
            self.seq += 1
            self.last_stamp = time.time()
            self.cond.notify_all()

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0):
        with self.cond:
            if self.seq == last_seq:
                self.cond.wait(timeout=timeout)
            return self.seq, self.jpeg, self.last_stamp


class MjpegServerNode:
    def __init__(self) -> None:
        rospy.init_node("camera_mjpeg_server")
        self.topic = rospy.get_param("~topic", "/usb_cam/image_raw")
        self.port = int(rospy.get_param("~port", 8080))
        self.host = rospy.get_param("~host", "0.0.0.0")
        self.quality = int(rospy.get_param("~quality", 80))
        self.max_fps = float(rospy.get_param("~max_fps", 15.0))
        self.flip = self._parse_bool(rospy.get_param("~flip", False))
        self.bridge = CvBridge()
        self.frames = FrameStore()
        self.last_encoded_at = 0.0
        self.min_interval = 1.0 / max(self.max_fps, 1.0)
        self.sub = rospy.Subscriber(self.topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
        self.httpd = ThreadingHTTPServer((self.host, self.port), self.make_handler())
        self.thread = threading.Thread(target=self.httpd.serve_forever)
        self.thread.daemon = True
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("MJPEG server ready: topic=%s url=http://%s:%d/",
                      self.topic, self.host, self.port)

    @staticmethod
    def _parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def image_cb(self, msg: Image) -> None:
        now = time.time()
        if now - self.last_encoded_at < self.min_interval:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s", exc)
            return
        if self.flip:
            frame = cv2.flip(frame, 1)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            rospy.logwarn_throttle(2.0, "JPEG encode failed")
            return
        self.last_encoded_at = now
        self.frames.update(encoded.tobytes())

    def make_handler(self):
        frames = self.frames
        topic = self.topic

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                rospy.logdebug("MJPEG HTTP: " + fmt, *args)

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = ("<html><head><title>ROS MJPEG</title>"
                            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                            "<style>body{font-family:sans-serif;background:#111;color:#eee;}"
                            "img{max-width:100%;height:auto;border:1px solid #555;}</style>"
                            "</head><body>"
                            "<h2>ROS MJPEG Stream</h2>"
                            "<p>topic: <code>%s</code></p>"
                            "<img src='/stream.mjpg'>"
                            "</body></html>" % topic).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path not in ("/stream", "/stream.mjpg"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_seq = -1
                while not rospy.is_shutdown():
                    seq, jpeg, _ = frames.wait_for_frame(last_seq, timeout=1.0)
                    if jpeg is None or seq == last_seq:
                        continue
                    last_seq = seq
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    except Exception:
                        break

        return Handler

    def run(self) -> None:
        self.thread.start()
        rospy.spin()

    def shutdown(self) -> None:
        try:
            self.httpd.shutdown()
            self.httpd.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    MjpegServerNode().run()
