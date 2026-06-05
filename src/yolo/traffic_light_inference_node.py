#!/usr/bin/env python3
"""YOLOv5 traffic-light inference node.

Subscribes to /usb_cam/image_raw, runs YOLOv5 every N Hz in a background
thread, applies consensus filtering to eliminate flicker, and publishes
structured detection results as JSON.
"""

import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    import torch
except ImportError:
    rospy.logfatal("PyTorch not installed. Run: pip install torch torchvision")
    sys.exit(1)

CLASS_NAMES = ["green_left", "green_right", "green_straight", "red_light"]

# BGR colors for drawing
CLASS_COLORS = [
    (0, 255, 0),    # green_left  -> green
    (255, 255, 0),  # green_right -> cyan
    (0, 255, 255),  # green_straight -> yellow
    (0, 0, 255),    # red_light -> red
]


def _resolve_model_path(param_path):
    if param_path and os.path.isfile(param_path):
        return param_path
    try:
        share_dir = rospkg.RosPack().get_path("yolo")
        default = os.path.join(share_dir, "models", "best.pt")
        if os.path.isfile(default):
            rospy.loginfo("Using model: %s", default)
            return default
    except Exception:
        pass
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dev_path = os.path.join(script_dir, "models", "best.pt")
    if os.path.isfile(dev_path):
        return os.path.abspath(dev_path)
    rospy.logfatal(
        "Model not found. Place best.pt in yolo/models/ or set ~model_path"
    )
    sys.exit(1)


class TrafficLightInference:
    def __init__(self):
        self._init_params()
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._shutdown = False
        self._latest_frame = None
        self._last_frame_stamp = None
        self._error_count = 0
        self._frame_counter = 0

        # Consensus state
        self._class_hits = {i: 0 for i in range(len(CLASS_NAMES))}
        self._consensus_class = None
        self._consensus_confidence = 0.0
        self._consensus_held = 0
        self._consensus_start_time = None
        self._release_counter = 0

        # Load model
        self._model_path = _resolve_model_path(self._model_path_param)
        self._model = self._load_model()
        self._set_status("initializing")

        # image subscriber
        self._sub = rospy.Subscriber(
            self._image_topic,
            Image,
            self._image_cb,
            queue_size=1,
            buff_size=2 ** 24,
        )

        # publishers
        self._det_pub = rospy.Publisher(
            self._detections_topic, String, queue_size=1
        )
        self._debug_pub = rospy.Publisher(
            self._debug_image_topic, Image, queue_size=1
        )
        self._status_pub = rospy.Publisher(
            self._status_topic, String, queue_size=1, latch=True
        )

        # Diagnostics
        self._inference_ms = 0.0

        rospy.on_shutdown(self._on_shutdown)
        self._set_status("tracking")
        rospy.loginfo("TrafficLightInference ready (model=%s, device=%s, rate=%.1fHz)",
                      self._model_path, self._device, self._inference_rate)

    # ── params ────────────────────────────────────────────

    def _init_params(self):
        self._image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self._detections_topic = rospy.get_param(
            "~detections_topic", "/traffic_light/detections"
        )
        self._debug_image_topic = rospy.get_param(
            "~debug_image_topic", "/traffic_light/debug_image"
        )
        self._status_topic = rospy.get_param(
            "~status_topic", "/traffic_light/status"
        )

        self._model_path_param = rospy.get_param("~model_path", "")
        self._conf_thresh = rospy.get_param("~confidence_threshold", 0.5)
        self._nms_iou = rospy.get_param("~nms_iou_threshold", 0.45)
        self._input_size = int(rospy.get_param("~input_size", 640))
        self._cpu_reduce_input = rospy.get_param("~cpu_reduce_input", True)
        self._inference_rate = float(rospy.get_param("~inference_rate", 10.0))
        self._device_param = rospy.get_param("~device", "")

        self._confirm_frames = int(rospy.get_param("~consensus_confirm_frames", 5))
        self._release_frames = int(rospy.get_param("~consensus_release_frames", 3))
        self._consensus_timeout = rospy.get_param("~consensus_timeout", 1.0)
        self._ema_alpha = rospy.get_param("~consensus_ema_alpha", 0.3)

        self._max_errors = int(rospy.get_param("~max_consecutive_errors", 3))
        self._reload_cooldown = rospy.get_param("~reload_cooldown", 5.0)
        self._image_timeout = rospy.get_param("~image_timeout", 5.0)

        self._publish_debug = rospy.get_param("~publish_debug", True)
        self._flip = rospy.get_param("~flip", False)

        self._device = self._resolve_device(self._device_param)
        if self._device == "cpu" and self._cpu_reduce_input:
            self._input_size = 320
            rospy.loginfo("CPU mode: reduced input_size to 320")

    @staticmethod
    def _resolve_device(param):
        if param:
            return param
        if torch.cuda.is_available():
            rospy.loginfo("CUDA detected, using GPU")
            return "cuda:0"
        rospy.loginfo("CUDA not available, using CPU")
        return "cpu"

    # ── model ──────────────────────────────────────────────

    def _load_model(self):
        rospy.loginfo("Loading YOLOv5 model: %s", self._model_path)
        try:
            model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=self._model_path,
                force_reload=False,
                device=self._device,
            )
            model.conf = self._conf_thresh
            model.iou = self._nms_iou
            rospy.loginfo("Model loaded successfully")
            return model
        except Exception as exc:
            rospy.logerr("Failed to load YOLO model: %s", exc)
            raise

    def _reload_model(self):
        rospy.logerr(
            "Attempting model reload after %d consecutive errors",
            self._error_count,
        )
        try:
            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._model = self._load_model()
            self._error_count = 0
            self._set_status("tracking")
            rospy.loginfo("Model reload successful")
        except Exception as exc:
            rospy.logerr("Model reload failed: %s. Retry in %.1fs",
                         exc, self._reload_cooldown)
            self._set_status("error")

    # ── image callback ─────────────────────────────────────

    def _image_cb(self, msg):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s", exc)
            return
        if self._flip:
            img = cv2.flip(img, 1)
        t = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
        with self._lock:
            self._latest_frame = img
            self._last_frame_stamp = t

    # ── inference loop (runs in daemon thread) ─────────────

    def run(self):
        rate = rospy.Rate(self._inference_rate)
        while not rospy.is_shutdown() and not self._shutdown:
            t_start = time.time()
            self._infer_once()
            self._inference_ms = (time.time() - t_start) * 1000.0
            rate.sleep()

    def _infer_once(self):
        with self._lock:
            if self._latest_frame is None:
                return
            frame = self._latest_frame.copy()
            stamp = self._last_frame_stamp

        # Check image timeout
        if time.time() - stamp > self._image_timeout:
            self._set_status("no_image")
            self._reset_consensus()
            return

        self._frame_counter += 1

        try:
            results = self._model(frame, size=self._input_size)
            dets = results.xyxy[0]
            if dets is not None and len(dets):
                dets = dets.cpu().numpy()
            else:
                dets = np.empty((0, 6))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Inference error: %s", exc)
            self._error_count += 1
            if self._error_count >= self._max_errors:
                self._reload_model()
                self._error_count = 0
            return

        self._error_count = max(0, self._error_count - 1)  # decay
        self._update_consensus(dets, stamp)
        self._publish_detections(dets, stamp)
        self._publish_debug_image(frame, dets)

    # ── consensus filtering ────────────────────────────────

    def _update_consensus(self, dets, stamp):
        """Apply hysteresis: N hits to confirm, M misses to release."""
        det_classes = {}
        for d in dets:
            cls_id = int(d[5])
            conf = float(d[4])
            if cls_id not in det_classes or conf > det_classes[cls_id]:
                det_classes[cls_id] = conf

        # Update per-class counters
        for cls_id in range(len(CLASS_NAMES)):
            if cls_id in det_classes:
                self._class_hits[cls_id] += 1
            else:
                self._class_hits[cls_id] = 0

        # Check timeout on current consensus
        if self._consensus_class is not None:
            if time.time() - self._consensus_start_time > self._consensus_timeout:
                self._reset_consensus()

        # Find best candidate
        best_cls = None
        best_conf = 0.0
        for cls_id, hits in self._class_hits.items():
            if hits >= self._confirm_frames and cls_id in det_classes:
                if det_classes[cls_id] > best_conf:
                    best_cls = cls_id
                    best_conf = det_classes[cls_id]

        if best_cls is not None:
            if self._consensus_class is None:
                # First lock
                self._consensus_class = best_cls
                self._consensus_confidence = best_conf
                self._consensus_start_time = time.time()
                self._consensus_held = 1
                self._set_status("tracking")
            elif best_cls == self._consensus_class:
                self._consensus_held += 1
                self._consensus_confidence = (
                    self._ema_alpha * best_conf
                    + (1.0 - self._ema_alpha) * self._consensus_confidence
                )
                self._release_counter = 0
            else:
                # Different class locked — switch immediately with confidence
                self._consensus_class = best_cls
                self._consensus_confidence = best_conf
                self._consensus_start_time = time.time()
                self._consensus_held = 1
            return

        # No candidate — count releases
        if self._consensus_class is not None:
            self._release_counter += 1
            if self._release_counter >= self._release_frames:
                self._reset_consensus()

    def _reset_consensus(self):
        self._consensus_class = None
        self._consensus_confidence = 0.0
        self._consensus_held = 0
        self._consensus_start_time = None
        self._release_counter = 0

    # ── publish ────────────────────────────────────────────

    def _publish_detections(self, dets, stamp):
        raw = []
        for d in dets:
            raw.append({
                "class_name": CLASS_NAMES[int(d[5])],
                "class_id": int(d[5]),
                "confidence": round(float(d[4]), 4),
                "bbox": [round(float(x), 1) for x in d[:4]],
            })

        msg = {
            "header": {"stamp": stamp},
            "raw_detections": raw,
            "consensus": {
                "class_name": (
                    CLASS_NAMES[self._consensus_class]
                    if self._consensus_class is not None
                    else None
                ),
                "class_id": self._consensus_class,
                "confidence": round(self._consensus_confidence, 4),
                "active": self._consensus_class is not None,
                "held_frames": self._consensus_held,
            },
            "status": self._status_text,
            "diagnostics": {
                "fps": round(1.0 / max(self._inference_ms / 1000.0, 0.001), 1),
                "inference_ms": round(self._inference_ms, 1),
                "error_count": self._error_count,
            },
        }
        self._det_pub.publish(String(data=json.dumps(msg)))

    def _publish_debug_image(self, frame, dets):
        if not self._publish_debug:
            return
        if self._debug_pub.get_num_connections() == 0:
            return

        annot = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = map(int, d[:4])
            cls_id = int(d[5])
            conf = float(d[4])
            color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
            cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annot, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                annot, label, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )

        # Draw consensus info
        if self._consensus_class is not None:
            info = "{} {:.2f}".format(
                CLASS_NAMES[self._consensus_class], self._consensus_confidence
            )
        else:
            info = "none"
        cv2.putText(
            annot, "TL: " + info, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

        try:
            img_msg = self._bridge.cv2_to_imgmsg(annot, "bgr8")
            img_msg.header.stamp = rospy.Time.now()
            self._debug_pub.publish(img_msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to publish debug image: %s", exc)

    def _set_status(self, text):
        self._status_text = text
        try:
            self._status_pub.publish(String(data=text))
        except Exception:
            pass

    # ── lifecycle ──────────────────────────────────────────

    def _on_shutdown(self):
        self._shutdown = True
        self._set_status("shutdown")
        self._reset_consensus()
        msg = {
            "header": {"stamp": time.time()},
            "raw_detections": [],
            "consensus": {"class_name": None, "class_id": None,
                          "confidence": 0.0, "active": False, "held_frames": 0},
            "status": "shutdown",
            "diagnostics": {"fps": 0.0, "inference_ms": 0.0, "error_count": 0},
        }
        try:
            self._det_pub.publish(String(data=json.dumps(msg)))
        except Exception:
            pass


def main():
    rospy.init_node("traffic_light_inference")
    node = TrafficLightInference()
    node.run()


if __name__ == "__main__":
    main()
