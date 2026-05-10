#!/usr/bin/env python3
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


class CameraWindowDebugNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.image_topic = rospy.get_param("~image_topic", rospy.get_param("image_topic", "/usb_cam/image_raw"))
        self.compressed_image_topic = rospy.get_param(
            "~compressed_image_topic",
            rospy.get_param("compressed_raw_topic", "/right_line_follow/raw_image/compressed"),
        )
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", rospy.get_param("cmd_vel_topic", "/cmd_vel"))
        self.status_topic = rospy.get_param(
            "~status_topic", rospy.get_param("status_topic", "/camera_window_debug/status")
        )
        self.use_compressed = bool(rospy.get_param("~use_compressed", rospy.get_param("use_compressed", False)))
        default_output_dir = os.path.join(
            os.environ.get("ROS_HOME", os.path.expanduser("~/.ros")), "ucar_2026_line_follow"
        )
        configured_output_dir = rospy.get_param(
            "~output_dir",
            rospy.get_param("camera_debug_output_dir", default_output_dir),
        )
        self.output_dir = configured_output_dir or default_output_dir

        self.roi_y_start_ratio = float(rospy.get_param("~roi_y_start_ratio", rospy.get_param("roi_y_start_ratio", 0.45)))
        self.roi_y_end_ratio = float(rospy.get_param("~roi_y_end_ratio", rospy.get_param("roi_y_end_ratio", 1.0)))
        self.white_s_max = int(rospy.get_param("~white_s_max", rospy.get_param("white_s_max", 85)))
        self.white_v_min = int(rospy.get_param("~white_v_min", rospy.get_param("white_v_min", 150)))
        self.gray_white_threshold = int(
            rospy.get_param("~gray_white_threshold", rospy.get_param("gray_white_threshold", 185))
        )
        self.morph_kernel_size = int(rospy.get_param("~morph_kernel_size", rospy.get_param("morph_kernel_size", 5)))
        self.min_contour_area = float(rospy.get_param("~min_contour_area", rospy.get_param("min_contour_area", 60.0)))

        self.zero_publish_hz = float(rospy.get_param("~zero_publish_hz", rospy.get_param("zero_publish_hz", 20.0)))
        self.stop_publish_count = int(rospy.get_param("~stop_publish_count", rospy.get_param("stop_publish_count", 20)))
        self.stop_publish_interval = float(
            rospy.get_param("~stop_publish_interval", rospy.get_param("stop_publish_interval", 0.02))
        )

        self.latest_raw = None
        self.latest_processed = None
        self.latest_stamp = 0.0
        self.frames = 0
        self.saved_count = 0

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1)

        if self.use_compressed:
            self.image_sub = rospy.Subscriber(
                self.compressed_image_topic, CompressedImage, self.compressed_callback, queue_size=1, buff_size=2**24
            )
            source = self.compressed_image_topic
        else:
            self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
            source = self.image_topic

        period = 1.0 / max(self.zero_publish_hz, 1.0)
        rospy.Timer(rospy.Duration(period), self.publish_zero)
        rospy.Timer(rospy.Duration(1.0), self.publish_status)
        rospy.on_shutdown(self.shutdown)

        self.input_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.input_thread.start()

        rospy.loginfo(
            "camera_window_debug_node started in SSH snapshot mode. source=%s use_compressed=%d output_dir=%s",
            source,
            int(self.use_compressed),
            self.resolve_output_dir(),
        )
        rospy.loginfo("Press Enter in this SSH terminal to save raw, processed, and combined images on the car.")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "camera debug cv_bridge conversion failed: %s", exc)
            return
        self.update_latest_frame(frame)

    def compressed_callback(self, msg: CompressedImage):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn_throttle(2.0, "camera debug JPEG decode failed")
            return
        self.update_latest_frame(frame)

    def update_latest_frame(self, frame: np.ndarray):
        processed = self.build_processed_view(frame)
        with self.lock:
            self.latest_raw = frame.copy()
            self.latest_processed = processed
            self.latest_stamp = time.time()
            self.frames += 1

    def keyboard_loop(self):
        while not rospy.is_shutdown():
            try:
                line = sys.stdin.readline()
            except Exception as exc:
                rospy.logwarn("camera debug stdin read failed: %s", exc)
                time.sleep(0.5)
                continue

            if rospy.is_shutdown():
                return
            if line == "":
                time.sleep(0.2)
                continue
            self.save_snapshot()

    def save_snapshot(self):
        with self.lock:
            raw = None if self.latest_raw is None else self.latest_raw.copy()
            processed = None if self.latest_processed is None else self.latest_processed.copy()
            stamp_sec = self.latest_stamp

        if raw is None or processed is None:
            rospy.logwarn("no camera frame is available yet; wait for /usb_cam/image_raw and press Enter again")
            return

        out_dir = self.resolve_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = "camera_debug_%s_%03d" % (stamp, self.saved_count + 1)
        raw_path = os.path.join(out_dir, base + "_raw.jpg")
        processed_path = os.path.join(out_dir, base + "_processed.jpg")
        combined_path = os.path.join(out_dir, base + "_combined.jpg")

        combined = self.make_combined(raw, processed)
        ok_raw = cv2.imwrite(raw_path, raw)
        ok_processed = cv2.imwrite(processed_path, processed)
        ok_combined = cv2.imwrite(combined_path, combined)
        if ok_raw and ok_processed and ok_combined:
            self.saved_count += 1
            age = max(0.0, time.time() - stamp_sec)
            rospy.loginfo(
                "saved camera snapshot #%d age=%.2fs raw=%s processed=%s combined=%s",
                self.saved_count,
                age,
                raw_path,
                processed_path,
                combined_path,
            )
        else:
            rospy.logerr(
                "failed to save camera snapshot: raw_ok=%d processed_ok=%d combined_ok=%d dir=%s",
                int(ok_raw),
                int(ok_processed),
                int(ok_combined),
                out_dir,
            )

    def resolve_output_dir(self) -> str:
        return os.path.abspath(os.path.expanduser(self.output_dir))

    def make_combined(self, raw: np.ndarray, processed: np.ndarray) -> np.ndarray:
        if raw.shape[:2] != processed.shape[:2]:
            processed = cv2.resize(processed, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
        return np.hstack((raw, processed))

    def build_processed_view(self, frame: np.ndarray) -> np.ndarray:
        mask, roi_origin_y = self.extract_white_mask(frame)
        height, width = frame.shape[:2]

        mask_full = np.zeros((height, width), dtype=np.uint8)
        mask_full[roi_origin_y : roi_origin_y + mask.shape[0], :] = mask
        processed = frame.copy()
        processed[mask_full > 0] = (0, 255, 255)

        contour_result = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
        contour_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            contour_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(processed, (x, y), (x + w, y + h), (0, 180, 255), 1)

        cv2.line(processed, (0, roi_origin_y), (width - 1, roi_origin_y), (255, 0, 0), 2)
        text = "opencv: S<={} V>={} gray>{} morph={} contours={}".format(
            self.white_s_max,
            self.white_v_min,
            self.gray_white_threshold,
            self.morph_kernel_size,
            contour_count,
        )
        cv2.putText(processed, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        cv2.putText(processed, "SSH Enter saves snapshot; zero cmd_vel active", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        return processed

    def extract_white_mask(self, frame: np.ndarray):
        height = frame.shape[0]
        y0 = int(height * self.roi_y_start_ratio)
        y1 = int(height * self.roi_y_end_ratio)
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        roi = frame[y0:y1, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white_hsv = cv2.inRange(hsv, (0, 0, self.white_v_min), (179, self.white_s_max, 255))

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, white_gray = cv2.threshold(gray, self.gray_white_threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(white_hsv, white_gray)
        mask = self.remove_small_components(mask)
        kernel_size = max(3, self.morph_kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask, y0

    def remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
        filtered = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_contour_area:
                cv2.drawContours(filtered, [contour], -1, 255, thickness=cv2.FILLED)
        return filtered

    def publish_zero(self, _event=None):
        self.cmd_pub.publish(Twist())

    def publish_status(self, _event=None):
        with self.lock:
            frames = self.frames
            saved = self.saved_count
            stamp = self.latest_stamp
        age = float("inf") if stamp <= 0.0 else max(0.0, time.time() - stamp)
        self.status_pub.publish(
            String(data="frames={} saved={} last_frame_age={:.2f}s output_dir={}".format(frames, saved, age, self.resolve_output_dir()))
        )

    def shutdown(self):
        for _ in range(max(0, self.stop_publish_count)):
            self.publish_zero()
            time.sleep(max(0.0, self.stop_publish_interval))


def main():
    rospy.init_node("camera_window_debug_node")
    CameraWindowDebugNode()
    rospy.spin()


if __name__ == "__main__":
    main()
