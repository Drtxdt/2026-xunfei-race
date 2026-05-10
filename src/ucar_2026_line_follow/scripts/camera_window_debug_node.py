#!/usr/bin/env python3
import time

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
        self.window_prefix = rospy.get_param("~window_prefix", rospy.get_param("window_prefix", "ucar camera debug"))

        self.roi_y_start_ratio = float(rospy.get_param("~roi_y_start_ratio", rospy.get_param("roi_y_start_ratio", 0.45)))
        self.roi_y_end_ratio = float(rospy.get_param("~roi_y_end_ratio", rospy.get_param("roi_y_end_ratio", 1.0)))
        self.white_s_max = int(rospy.get_param("~white_s_max", rospy.get_param("white_s_max", 85)))
        self.white_v_min = int(rospy.get_param("~white_v_min", rospy.get_param("white_v_min", 150)))
        self.gray_white_threshold = int(
            rospy.get_param("~gray_white_threshold", rospy.get_param("gray_white_threshold", 185))
        )
        self.morph_kernel_size = int(rospy.get_param("~morph_kernel_size", rospy.get_param("morph_kernel_size", 5)))
        self.min_contour_area = float(rospy.get_param("~min_contour_area", rospy.get_param("min_contour_area", 60.0)))

        self.resize_width = int(rospy.get_param("~window_resize_width", rospy.get_param("window_resize_width", 960)))
        self.zero_publish_hz = float(rospy.get_param("~zero_publish_hz", rospy.get_param("zero_publish_hz", 20.0)))
        self.max_display_fps = float(rospy.get_param("~max_display_fps", rospy.get_param("max_display_fps", 15.0)))
        self.stop_publish_count = int(rospy.get_param("~stop_publish_count", rospy.get_param("stop_publish_count", 20)))
        self.stop_publish_interval = float(
            rospy.get_param("~stop_publish_interval", rospy.get_param("stop_publish_interval", 0.02))
        )

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1)
        self.last_display_time = 0.0
        self.frames = 0
        self.last_frame_time = 0.0
        self.windows_ready = False

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

        rospy.loginfo(
            "camera_window_debug_node started. source=%s use_compressed=%d cmd_vel=%s",
            source,
            int(self.use_compressed),
            self.cmd_vel_topic,
        )

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "camera debug cv_bridge conversion failed: %s", exc)
            return
        self.display_frame(frame)

    def compressed_callback(self, msg: CompressedImage):
        data = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            rospy.logwarn_throttle(2.0, "camera debug JPEG decode failed")
            return
        self.display_frame(frame)

    def display_frame(self, frame: np.ndarray):
        now = time.time()
        if self.max_display_fps > 0.0 and now - self.last_display_time < 1.0 / self.max_display_fps:
            return
        self.last_display_time = now
        self.last_frame_time = now
        self.frames += 1

        raw = self.resize_for_window(frame)
        processed = self.build_processed_view(frame)

        try:
            if not self.windows_ready:
                cv2.namedWindow(self.raw_window_name, cv2.WINDOW_NORMAL)
                cv2.namedWindow(self.processed_window_name, cv2.WINDOW_NORMAL)
                self.windows_ready = True
            cv2.imshow(self.raw_window_name, raw)
            cv2.imshow(self.processed_window_name, processed)
            cv2.waitKey(1)
        except cv2.error as exc:
            rospy.logerr_throttle(2.0, "OpenCV window failed. Run this node on a machine with a GUI display: %s", exc)

    @property
    def raw_window_name(self) -> str:
        return "%s - raw" % self.window_prefix

    @property
    def processed_window_name(self) -> str:
        return "%s - opencv processed" % self.window_prefix

    def build_processed_view(self, frame: np.ndarray) -> np.ndarray:
        mask, roi_origin_y = self.extract_white_mask(frame)
        height, width = frame.shape[:2]

        mask_full = np.zeros((height, width), dtype=np.uint8)
        mask_full[roi_origin_y : roi_origin_y + mask.shape[0], :] = mask
        processed = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)

        overlay = frame.copy()
        overlay[mask_full > 0] = (0, 255, 255)
        processed = cv2.addWeighted(overlay, 0.45, processed, 0.55, 0.0)

        contour_result = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(processed, (x, y), (x + w, y + h), (0, 180, 255), 1)

        cv2.line(processed, (0, roi_origin_y), (width - 1, roi_origin_y), (255, 0, 0), 2)
        text = "mask: hsv(S<={},V>={}) OR gray>{}; morph={}; contours={}".format(
            self.white_s_max,
            self.white_v_min,
            self.gray_white_threshold,
            self.morph_kernel_size,
            len(contours),
        )
        cv2.putText(processed, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        cv2.putText(processed, "publishing zero cmd_vel", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
        return self.resize_for_window(processed)

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

    def resize_for_window(self, frame: np.ndarray) -> np.ndarray:
        if self.resize_width <= 0 or frame.shape[1] <= self.resize_width:
            return frame
        scale = self.resize_width / float(frame.shape[1])
        return cv2.resize(frame, (self.resize_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    def publish_zero(self, _event=None):
        self.cmd_pub.publish(Twist())

    def publish_status(self, _event=None):
        age = float("inf") if self.last_frame_time <= 0.0 else max(0.0, time.time() - self.last_frame_time)
        self.status_pub.publish(
            String(data="frames={} last_frame_age={:.2f}s zero_cmd_hz={:.1f}".format(self.frames, age, self.zero_publish_hz))
        )

    def shutdown(self):
        for _ in range(max(0, self.stop_publish_count)):
            self.publish_zero()
            time.sleep(max(0.0, self.stop_publish_interval))
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def main():
    rospy.init_node("camera_window_debug_node")
    CameraWindowDebugNode()
    rospy.spin()


if __name__ == "__main__":
    main()
