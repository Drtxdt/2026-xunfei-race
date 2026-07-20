#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResNet18 RKNN traffic-light classifier with debug image and speech."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

from ucar_2026_traffic_light_rknn_test.classifier import (
    CLASS_NAMES,
    ScoreConsensus,
    make_detection_payload,
    preprocess_frame,
    stable_softmax,
)

try:
    from ucar_2026_competition_speech.srv import Announce
except Exception:
    Announce = None


CLASS_COLORS = {
    "green_left": (0, 255, 0),
    "green_right": (255, 255, 0),
    "green_straight": (0, 255, 255),
    "red_light": (0, 0, 255),
    "background": (160, 160, 160),
}
SPEECH_TEXT = {
    "green_left": "绿灯，左转。",
    "green_right": "绿灯，右转。",
    "green_straight": "绿灯，直行。",
    "red_light": "红灯，停止。",
}
ANNOUNCE_DECISION = {
    "green_left": "left",
    "green_right": "right",
    "green_straight": "straight",
    "red_light": "stop",
}


def repair_logging_levels():
    for name, value in {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }.items():
        logging.addLevelName(value, name)
    try:
        import rosgraph.roslogging as roslogging

        mapping = getattr(roslogging, "_logging_to_rospy_names", None)
        if isinstance(mapping, dict):
            if "WARNING" in mapping:
                mapping["WARN"] = mapping["WARNING"]
            for short_name, full_name in {
                "D": "DEBUG",
                "I": "INFO",
                "W": "WARNING",
                "E": "ERROR",
                "F": "FATAL",
            }.items():
                source_name = full_name
                if source_name not in mapping and source_name == "FATAL":
                    source_name = "CRITICAL"
                if source_name in mapping:
                    mapping[short_name] = mapping[source_name]
    except Exception:
        pass


def resolve_model_path(param_path):
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(param_path))) if param_path else ""
    if path and os.path.isfile(path):
        return path
    try:
        package_dir = rospkg.RosPack().get_path("yolo")
        for filename in (
            "traffic_resnet18_rk3588_int8.rknn",
            "traffic_resnet18_rk3588_fp16.rknn",
        ):
            candidate = os.path.join(package_dir, "models", filename)
            if os.path.isfile(candidate):
                rospy.logwarn("Configured RKNN missing; using %s", candidate)
                return candidate
    except Exception:
        pass
    rospy.logfatal(
        "Traffic ResNet18 RKNN not found. Set ~model_path; old YOLO models are not compatible."
    )
    raise SystemExit(1)


class TrafficLightRknnTestNode(object):
    def __init__(self):
        repair_logging_levels()
        rospy.init_node("traffic_light_rknn_test_node")
        self._read_params()
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_stamp = 0.0
        self.latest_id = 0
        self.processed_id = 0
        self.last_frame_walltime = time.time()
        self.last_spoken_class = None
        self.last_spoken_at = 0.0
        self.timeout_reported = False

        self.consensus = ScoreConsensus(
            class_names=self.class_names,
            window_size=self.window_size,
            min_valid_samples=self.min_valid_samples,
            confidence_threshold=self.confidence_threshold,
            margin_threshold=self.margin_threshold,
            red_confirm_frames=self.red_confirm_frames,
            green_confirm_frames=self.green_confirm_frames,
            release_frames=self.release_frames,
        )
        self.rknn = self._load_rknn()
        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=1)
        self.det_pub = rospy.Publisher(self.detections_topic, String, queue_size=1)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self._image_cb, queue_size=1, buff_size=2 ** 24
        )
        rospy.on_shutdown(self._on_shutdown)
        self._publish_status("tracking")
        rospy.loginfo(
            "Traffic ResNet18 ready: model=%s quant=%s input=1x%dx%dx3 flip=%s crop=%.2f:%.2f",
            self.model_path,
            self.model_quantization,
            self.input_height,
            self.input_width,
            self.flip,
            self.crop_top,
            self.crop_bottom,
        )

    def _read_params(self):
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.detections_topic = rospy.get_param(
            "~detections_topic", "/traffic_light_rknn_test/detections"
        )
        self.debug_image_topic = rospy.get_param(
            "~debug_image_topic", "/traffic_light_rknn_test/debug_image"
        )
        self.status_topic = rospy.get_param("~status_topic", "/traffic_light_rknn_test/status")
        self.model_path = resolve_model_path(rospy.get_param("~model_path", ""))
        names = rospy.get_param("~class_names", list(CLASS_NAMES))
        self.class_names = tuple(str(name) for name in names)
        if self.class_names != CLASS_NAMES:
            rospy.logfatal("class_names must be exactly: %s", ", ".join(CLASS_NAMES))
            raise SystemExit(1)
        self.model_quantization = str(rospy.get_param("~model_quantization", "int8"))
        self.input_width = int(rospy.get_param("~input_width", 320))
        self.input_height = int(rospy.get_param("~input_height", 160))
        self.crop_top = float(rospy.get_param("~crop_top", 0.18))
        self.crop_bottom = float(rospy.get_param("~crop_bottom", 0.72))
        self.flip = bool(rospy.get_param("~flip", True))
        self.inference_rate = float(rospy.get_param("~inference_rate", 10.0))
        self.confidence_threshold = float(rospy.get_param("~confidence_threshold", 0.55))
        self.margin_threshold = float(rospy.get_param("~margin_threshold", 0.12))
        self.window_size = int(rospy.get_param("~score_window_size", 5))
        self.min_valid_samples = int(rospy.get_param("~min_valid_samples", 3))
        self.red_confirm_frames = int(rospy.get_param("~red_confirm_frames", 2))
        self.green_confirm_frames = int(rospy.get_param("~green_confirm_frames", 3))
        self.release_frames = int(rospy.get_param("~consensus_release_frames", 1))
        self.image_timeout = float(rospy.get_param("~image_timeout", 1.0))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        self.enable_speech = bool(rospy.get_param("~enable_speech", True))
        self.announce_service = rospy.get_param(
            "~announce_service", "/competition_speech/announce"
        )
        self.announce_timeout = float(rospy.get_param("~announce_service_timeout_sec", 0.5))
        self.speak_topic = rospy.get_param("~speak_topic", "/speak")
        self.use_announce_service = bool(rospy.get_param("~use_announce_service", True))
        self.announce_event = str(rospy.get_param("~announce_event", "custom")).strip() or "custom"
        self.fallback_to_speak_topic = bool(rospy.get_param("~fallback_to_speak_topic", True))
        self.slow_speech = bool(rospy.get_param("~slow_speech", True))
        self.repeat_same = bool(rospy.get_param("~repeat_same", False))
        self.min_speech_interval = float(rospy.get_param("~min_speech_interval_sec", 2.0))
        self.speech_wait = bool(rospy.get_param("~speech_wait", False))

    def _load_rknn(self):
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:
            repair_logging_levels()
            rospy.logfatal("Failed to import rknnlite.api: %s", exc)
            raise SystemExit(1)
        repair_logging_levels()
        rknn = RKNNLite()
        repair_logging_levels()
        rospy.loginfo("Loading RKNN classifier: %s", self.model_path)
        ret = rknn.load_rknn(self.model_path)
        repair_logging_levels()
        if ret != 0:
            rospy.logfatal("load_rknn failed: %s", ret)
            raise SystemExit(ret)
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        repair_logging_levels()
        if ret != 0:
            rospy.logfatal("init_runtime failed: %s", ret)
            raise SystemExit(ret)
        return rknn

    def _image_cb(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr_throttle(2.0, "cv_bridge failed: %s", exc)
            return
        with self.lock:
            self.latest_frame = frame
            self.latest_stamp = message.header.stamp.to_sec() or rospy.Time.now().to_sec()
            self.latest_id += 1
            self.last_frame_walltime = time.time()
        self.timeout_reported = False

    def _next_frame(self):
        with self.lock:
            if self.latest_frame is None or self.latest_id == self.processed_id:
                return None
            self.processed_id = self.latest_id
            return self.latest_frame.copy(), self.latest_stamp

    def run(self):
        rate = rospy.Rate(max(0.5, self.inference_rate))
        while not rospy.is_shutdown():
            item = self._next_frame()
            if item is not None:
                self._process_frame(item[0], item[1])
            elif self.last_frame_walltime and time.time() - self.last_frame_walltime > self.image_timeout:
                if not self.timeout_reported:
                    self.consensus.reset()
                    self._publish_status("image_timeout")
                    rospy.logwarn("Camera image timeout; consensus released")
                    self.timeout_reported = True
            rate.sleep()

    def _process_frame(self, frame, stamp):
        try:
            corrected, batch, roi_bbox = preprocess_frame(
                frame,
                flip=self.flip,
                crop_top=self.crop_top,
                crop_bottom=self.crop_bottom,
                input_width=self.input_width,
                input_height=self.input_height,
            )
            started = time.perf_counter()
            outputs = self.rknn.inference(inputs=[batch])
            inference_ms = (time.perf_counter() - started) * 1000.0
            if outputs is None or len(outputs) == 0:
                raise RuntimeError("RKNN classifier returned no outputs")
            probabilities = stable_softmax(outputs[0], len(self.class_names))
            consensus_state = self.consensus.update(probabilities)
            payload = make_detection_payload(
                stamp,
                self.class_names,
                probabilities,
                roi_bbox,
                consensus_state,
                inference_ms,
                self.model_quantization,
            )
            self.det_pub.publish(
                String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            )
            self._publish_debug(corrected, roi_bbox, probabilities, consensus_state, inference_ms)
            self._maybe_speak(consensus_state)
            self._publish_status("tracking")
        except Exception as exc:
            self.consensus.update(np.full(len(self.class_names), np.nan, dtype=np.float32))
            self._publish_status("inference_error")
            rospy.logerr_throttle(2.0, "RKNN classifier inference failed: %s", exc)

    def _publish_debug(self, frame, roi_bbox, probabilities, consensus_state, inference_ms):
        if not self.publish_debug:
            return
        image = frame.copy()
        x1, y1, x2, y2 = roi_bbox
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (255, 180, 0), 2)
        cv2.putText(
            image,
            "corrected view | crop y=%.2f:%.2f | %.1f ms" % (
                self.crop_top,
                self.crop_bottom,
                inference_ms,
            ),
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (255, 255, 255),
            2,
        )
        for index, name in enumerate(self.class_names):
            color = CLASS_COLORS[name]
            cv2.putText(
                image,
                "%s: %.3f" % (name, probabilities[index]),
                (8, 46 + index * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
            )
        label = "consensus: %s (%s)" % (
            consensus_state["class_name"] if consensus_state["active"] else "inactive",
            consensus_state["reason"],
        )
        cv2.putText(
            image,
            label,
            (8, max(24, image.shape[0] - 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
        )
        try:
            message = self.bridge.cv2_to_imgmsg(image, "bgr8")
            message.header.stamp = rospy.Time.now()
            self.debug_pub.publish(message)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "debug image publish failed: %s", exc)

    def _maybe_speak(self, state):
        if not self.enable_speech or not state["active"]:
            return
        class_name = state["class_name"]
        now = time.time()
        if not self.repeat_same and class_name == self.last_spoken_class:
            return
        if now - self.last_spoken_at < self.min_speech_interval:
            return
        self.last_spoken_class = class_name
        self.last_spoken_at = now
        text = SPEECH_TEXT[class_name]
        if self.slow_speech:
            text = text.replace("，", "， ")
        if self._try_announce_service(class_name, text):
            return
        if self.fallback_to_speak_topic:
            self.speak_pub.publish(String(data=text))

    def _try_announce_service(self, class_name, text):
        if not self.use_announce_service or Announce is None:
            return False
        try:
            rospy.wait_for_service(self.announce_service, timeout=self.announce_timeout)
            announce = rospy.ServiceProxy(self.announce_service, Announce)
            event = self.announce_event
            response = announce(
                event,
                "",
                "",
                ANNOUNCE_DECISION[class_name],
                text if event == "custom" else "",
                self.speech_wait,
            )
            if bool(response.success):
                rospy.loginfo("Announced via %s: %s", self.announce_service, response.speech_text)
                return True
            rospy.logwarn("Announce service failed: %s", response.message)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Announce service unavailable: %s", exc)
        return False

    def _publish_status(self, status):
        self.status_pub.publish(String(data=status))

    def _on_shutdown(self):
        try:
            self._publish_status("shutdown")
            self.rknn.release()
        except Exception:
            pass


def main():
    TrafficLightRknnTestNode().run()


if __name__ == "__main__":
    main()
