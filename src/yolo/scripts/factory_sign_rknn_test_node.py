#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
厂区标识牌 RKNN/NPU 推理测试节点
=================================
仿照 ucar_2026_traffic_light_rknn_test 结构:
  - 加载 factory_sign_3cls_fp16.rknn 到 NPU
  - 订阅摄像头话题，YOLOv5 推理
  - 发布检测结果 / 调试图像 / 状态
  - 共识滤波 + 语音播报
  - 按 S 键保存调试截图到 scripts/ 目录

用法:
  roslaunch yolo factory_sign_rknn_test.launch
  roslaunch yolo factory_sign_rknn_test.launch start_camera:=false
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from ucar_2026_competition_speech.srv import Announce
except Exception:
    Announce = None

# ============================================================
# 厂区标识牌 — 3 类
# ============================================================
CLASS_NAMES = ["food", "electronic", "daily"]
CLASS_LABELS = ["食品加工车间", "电子产品生产车间", "日用品加工车间"]
CLASS_COLORS = {
    "food": (0, 255, 0),        # 绿
    "electronic": (255, 191, 0),  # 蓝
    "daily": (0, 255, 255),      # 黄
}
SPEECH_TEXT = {
    "food": "前方，食品加工车间。",
    "electronic": "前方，电子产品生产车间。",
    "daily": "前方，日用品加工车间。",
}
ANNOUNCE_DECISION = {
    "food": "food",
    "electronic": "electronic",
    "daily": "daily",
}

# YOLOv5 后处理常量
ANCHORS = np.array(
    [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
     [59, 119], [116, 90], [156, 198], [373, 326]],
    dtype=np.float32,
)
MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

# 调试截图保存目录 (脚本所在路径)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_SAVE_DIR = SCRIPT_DIR


# ============================================================
# 工具函数
# ============================================================

def repair_logging_levels() -> None:
    levels = {
        "CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR,
        "WARNING": logging.WARNING, "WARN": logging.WARNING,
        "INFO": logging.INFO, "DEBUG": logging.DEBUG, "NOTSET": logging.NOTSET,
    }
    for name, value in levels.items():
        logging.addLevelName(value, name)
    try:
        import rosgraph.roslogging as roslogging
        mapping = getattr(roslogging, "_logging_to_rospy_names", None)
        if isinstance(mapping, dict):
            for sn, fn in {"D": "DEBUG", "I": "INFO", "W": "WARN",
                           "E": "ERROR", "F": "FATAL"}.items():
                if fn == "FATAL" and fn not in mapping:
                    fn = "CRITICAL"
                if sn not in mapping and fn in mapping:
                    mapping[sn] = mapping[fn]
    except Exception:
        pass


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(im: np.ndarray, new_shape: int, color=(0, 0, 0)):
    shape = im.shape[:2]
    r = min(float(new_shape) / shape[0], float(new_shape) / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape - new_unpad[0]) / 2.0, (new_shape - new_unpad[1]) / 2.0
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), r, (dw, dh)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2.0
    y[:, 1] = x[:, 1] - x[:, 3] / 2.0
    y[:, 2] = x[:, 0] + x[:, 2] / 2.0
    y[:, 3] = x[:, 1] + x[:, 3] / 2.0
    return y


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep, dtype=np.int64)


def infer_yolov5_input_size_from_count(count: int) -> Optional[int]:
    for size in (320, 416, 512, 640):
        grids = (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2
        if count == grids * 3:
            return size
    return None


def resolve_model_path(param_path: str) -> str:
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(param_path)))
    if path and os.path.isfile(path):
        return path
    try:
        yolo_dir = rospkg.RosPack().get_path("yolo")
        default = os.path.join(yolo_dir, "models", "factory_sign_3cls_fp16.rknn")
        if os.path.isfile(default):
            return default
    except Exception:
        pass
    rospy.logfatal("RKNN model not found. Set ~model_path or place factory_sign_3cls_fp16.rknn under yolo/models")
    sys.exit(1)


# ============================================================
# ROS 节点
# ============================================================

class FactorySignRknnTestNode:
    def __init__(self) -> None:
        repair_logging_levels()
        rospy.init_node("factory_sign_rknn_test_node")
        self.read_params()
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_stamp = 0.0
        self.shutdown = False

        # 共识滤波器状态
        self.class_hits = {idx: 0 for idx in range(len(CLASS_NAMES))}
        self.consensus_class: Optional[int] = None
        self.consensus_confidence = 0.0
        self.consensus_held = 0
        self.consensus_started_at = 0.0
        self.release_counter = 0

        # 语音节流
        self.last_spoken_class: Optional[str] = None
        self.last_spoken_at = 0.0
        self.inference_ms = 0.0
        self.output_shapes_logged = False

        self.rknn = self.load_rknn()
        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=1)
        self.det_pub = rospy.Publisher(self.detections_topic, String, queue_size=1)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.on_shutdown(self.on_shutdown)
        self.publish_status("tracking")
        rospy.loginfo("factory_sign_rknn_test ready: model=%s image=%s",
                       self.model_path, self.image_topic)

    # ---- 参数 ----
    def read_params(self) -> None:
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.detections_topic = rospy.get_param("~detections_topic",
                                                 "/factory_sign_rknn_test/detections")
        self.debug_image_topic = rospy.get_param("~debug_image_topic",
                                                  "/factory_sign_rknn_test/debug_image")
        self.status_topic = rospy.get_param("~status_topic",
                                             "/factory_sign_rknn_test/status")
        self.model_path = resolve_model_path(rospy.get_param("~model_path", ""))
        self.input_size = int(rospy.get_param("~input_size", 640))
        self.conf_thresh = float(rospy.get_param("~confidence_threshold", 0.3))
        self.nms_iou = float(rospy.get_param("~nms_iou_threshold", 0.45))
        self.inference_rate = float(rospy.get_param("~inference_rate", 10.0))
        self.flip = bool(rospy.get_param("~flip", False))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))
        self.image_timeout = float(rospy.get_param("~image_timeout", 5.0))
        self.confirm_frames = int(rospy.get_param("~consensus_confirm_frames", 3))
        self.release_frames = int(rospy.get_param("~consensus_release_frames", 3))
        self.consensus_timeout = float(rospy.get_param("~consensus_timeout", 1.0))
        self.ema_alpha = float(rospy.get_param("~consensus_ema_alpha", 0.3))
        self.enable_speech = bool(rospy.get_param("~enable_speech", True))
        self.announce_service = rospy.get_param("~announce_service",
                                                 "/competition_speech/announce")
        self.announce_timeout = float(rospy.get_param("~announce_service_timeout_sec", 0.5))
        self.speak_topic = rospy.get_param("~speak_topic", "/speak")
        self.use_announce_service = bool(rospy.get_param("~use_announce_service", True))
        self.announce_event = rospy.get_param("~announce_event", "custom").strip() or "custom"
        self.fallback_to_speak_topic = bool(rospy.get_param("~fallback_to_speak_topic", True))
        self.slow_speech = bool(rospy.get_param("~slow_speech", True))
        self.repeat_same = bool(rospy.get_param("~repeat_same", True))
        self.min_speech_interval = float(rospy.get_param("~min_speech_interval_sec", 2.0))
        self.speech_wait = bool(rospy.get_param("~speech_wait", False))
        self.save_debug = bool(rospy.get_param("~save_debug_images", True))

    # ---- 模型加载 ----
    def load_rknn(self):
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:
            repair_logging_levels()
            rospy.logfatal("Failed to import rknnlite.api: %s", exc)
            sys.exit(1)
        repair_logging_levels()
        rknn = RKNNLite()
        repair_logging_levels()
        rospy.loginfo("Loading RKNN model: %s", self.model_path)
        ret = rknn.load_rknn(self.model_path)
        repair_logging_levels()
        if ret != 0:
            rospy.logfatal("load_rknn failed: %s", ret)
            sys.exit(ret)
        rospy.loginfo("Initializing RKNN runtime on all NPU cores")
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        repair_logging_levels()
        if ret != 0:
            rospy.logfatal("init_runtime failed: %s", ret)
            sys.exit(ret)
        # Warmup
        try:
            dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            rknn.inference(inputs=[dummy])
            rospy.loginfo("NPU warmup inference completed")
        except Exception:
            pass
        return rknn

    # ---- 摄像头回调 ----
    def image_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s", exc)
            return
        if self.flip:
            frame = cv2.flip(frame, 1)
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
        with self.lock:
            self.latest_frame = frame
            self.latest_stamp = stamp

    # ---- 主循环 ----
    def run(self) -> None:
        rate = rospy.Rate(self.inference_rate)
        while not rospy.is_shutdown() and not self.shutdown:
            started = time.time()
            self.infer_once()
            self.inference_ms = (time.time() - started) * 1000.0
            rate.sleep()

    def infer_once(self) -> None:
        with self.lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
            stamp = self.latest_stamp

        if time.time() - stamp > self.image_timeout:
            self.publish_status("no_image")
            self.reset_consensus()
            return

        try:
            boxes, classes, scores = self.infer_frame(frame)
        except Exception as exc:
            repair_logging_levels()
            rospy.logerr_throttle(2.0, "RKNN inference failed: %s", exc)
            self.publish_status("error")
            return

        dets = self.build_detections(boxes, classes, scores)
        previous = self.consensus_class
        self.update_consensus(dets)
        self.publish_detections(dets, stamp)
        self.publish_debug_image(frame, dets)
        if self.consensus_class != previous or self.repeat_same:
            self.maybe_speak()
        self.publish_status("tracking")

    # ---- 推理 ----
    def infer_frame(self, frame: np.ndarray):
        img, ratio, pad = letterbox(frame, self.input_size)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = self.rknn.inference(inputs=[rgb])
        repair_logging_levels()
        if not self.output_shapes_logged:
            rospy.loginfo("RKNN output shapes: %s",
                          [getattr(o, "shape", None) for o in outputs])
            self.output_shapes_logged = True

        # 单输出路径 (Detect 层直接输出)
        arr = np.asarray(outputs[0])
        arr = np.squeeze(arr)
        if arr.ndim == 2 and arr.shape[-1] == 5 + len(CLASS_NAMES):
            boxes, classes, scores, model_size = self.single_output_post(arr)
            if boxes is not None and model_size and model_size != self.input_size:
                boxes *= float(self.input_size) / float(model_size)
        elif arr.ndim == 3:
            # 3-head 输出
            processed = [self.normalize_output(o) for o in outputs[:3]]
            boxes, classes, scores = self.three_head_post(processed)
        else:
            rospy.logerr("Unexpected RKNN output shape: %s", arr.shape)
            return None, None, None

        if boxes is None:
            return None, None, None
        return self.scale_boxes_to_frame(boxes, ratio, pad, frame.shape)

    def single_output_post(self, arr: np.ndarray):
        model_size = infer_yolov5_input_size_from_count(arr.shape[0])
        boxes = arr[:, :4].astype(np.float32)
        obj = arr[:, 4].astype(np.float32)
        cls_probs = arr[:, 5:].astype(np.float32)
        if obj.size and (obj.max() > 1.0 or obj.min() < 0.0):
            obj = sigmoid(obj)
        if cls_probs.size and (cls_probs.max() > 1.0 or cls_probs.min() < 0.0):
            cls_probs = sigmoid(cls_probs)
        class_ids = np.argmax(cls_probs, axis=1)
        scores = obj * cls_probs[np.arange(len(cls_probs)), class_ids]
        keep = np.where(scores >= self.conf_thresh)[0]
        if keep.size == 0:
            return None, None, None, model_size
        boxes = xywh2xyxy(boxes[keep])
        class_ids = class_ids[keep]
        scores = scores[keep]
        kept_boxes, kept_cls, kept_scores = [], [], []
        for cls_id in set(class_ids.tolist()):
            idx = np.where(class_ids == cls_id)
            nms_idx = nms_boxes(boxes[idx], scores[idx], self.nms_iou)
            kept_boxes.append(boxes[idx][nms_idx])
            kept_cls.append(np.full(len(nms_idx), cls_id, dtype=np.int64))
            kept_scores.append(scores[idx][nms_idx])
        if not kept_boxes:
            return None, None, None, model_size
        return np.concatenate(kept_boxes), np.concatenate(kept_cls), np.concatenate(kept_scores), model_size

    def normalize_output(self, output: np.ndarray) -> np.ndarray:
        arr = np.asarray(output)
        arr = np.squeeze(arr)
        if arr.ndim != 3:
            raise ValueError("unsupported RKNN output ndim=%d shape=%s" % (arr.ndim, arr.shape))
        if arr.shape[0] % 3 == 0 and arr.shape[1] == arr.shape[2]:
            c, h, w = arr.shape
            return np.transpose(arr.reshape(3, c // 3, h, w), (2, 3, 0, 1))
        if arr.shape[-1] % 3 == 0 and arr.shape[0] == arr.shape[1]:
            h, w, c = arr.shape
            return arr.reshape(h, w, 3, c // 3)
        raise ValueError("unsupported RKNN output shape=%s" % (arr.shape,))

    def three_head_post(self, processed: List[np.ndarray]):
        boxes_list, classes_list, scores_list = [], [], []
        for output, mask in zip(processed, MASKS):
            b, bc, bp = self.process_output_head(output, mask)
            b, c, s = self.filter_boxes_head(b, bc, bp)
            if b.size:
                boxes_list.append(b)
                classes_list.append(c)
                scores_list.append(s)
        if not boxes_list:
            return None, None, None
        boxes = xywh2xyxy(np.concatenate(boxes_list))
        classes = np.concatenate(classes_list)
        scores = np.concatenate(scores_list)
        kept_boxes, kept_cls, kept_scores = [], [], []
        for cls_id in set(classes.tolist()):
            idx = np.where(classes == cls_id)
            keep = nms_boxes(boxes[idx], scores[idx], self.nms_iou)
            kept_boxes.append(boxes[idx][keep])
            kept_cls.append(np.full(len(keep), cls_id, dtype=np.int64))
            kept_scores.append(scores[idx][keep])
        if not kept_boxes:
            return None, None, None
        return np.concatenate(kept_boxes), np.concatenate(kept_cls), np.concatenate(kept_scores)

    def process_output_head(self, output: np.ndarray, mask: List[int]):
        anchors = ANCHORS[mask].reshape(1, 1, 3, 2)
        gh, gw = output.shape[:2]
        box_confidence = output[..., 4:5]
        box_class_probs = output[..., 5:]
        if box_confidence.max() > 1.0 or box_class_probs.max() > 1.0:
            box_confidence = sigmoid(box_confidence)
            box_class_probs = sigmoid(box_class_probs)
        gy, gx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grid = np.stack((gx, gy), axis=-1).reshape(gh, gw, 1, 2)
        box_xy = (output[..., 0:2] * 2.0 - 0.5 + grid) * (self.input_size / float(gh))
        box_wh = np.power(output[..., 2:4] * 2.0, 2.0) * anchors
        return np.concatenate((box_xy, box_wh), axis=-1), box_confidence, box_class_probs

    def filter_boxes_head(self, boxes, bc, bp):
        boxes = boxes.reshape(-1, 4)
        bc = bc.reshape(-1)
        bp = bp.reshape(-1, bp.shape[-1])
        pos = np.where(bc >= self.conf_thresh)
        boxes, bc, bp = boxes[pos], bc[pos], bp[pos]
        if boxes.size == 0:
            return np.empty((0, 4)), np.empty((0,), dtype=np.int64), np.empty((0,))
        class_scores = np.max(bp, axis=-1)
        classes = np.argmax(bp, axis=-1)
        pos = np.where(class_scores >= self.conf_thresh)
        return boxes[pos], classes[pos], (class_scores * bc)[pos]

    def scale_boxes_to_frame(self, boxes, ratio, pad, frame_shape):
        dw, dh = pad
        boxes[:, [0, 2]] -= dw
        boxes[:, [1, 3]] -= dh
        boxes[:, :4] /= ratio
        h, w = frame_shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        return boxes

    def build_detections(self, boxes, classes, scores) -> List[Dict[str, Any]]:
        if boxes is None:
            return []
        dets = []
        for box, cls_id, score in zip(boxes, classes, scores):
            cls_id = int(cls_id)
            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                continue
            dets.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id],
                "label": CLASS_LABELS[cls_id],
                "confidence": float(score),
                "bbox": [float(x) for x in box],
            })
        dets.sort(key=lambda d: d["confidence"], reverse=True)
        return dets

    # ---- 共识滤波 ----
    def update_consensus(self, dets: List[Dict[str, Any]]) -> None:
        best_by_class: Dict[int, float] = {}
        for det in dets:
            cls_id = int(det["class_id"])
            conf = float(det["confidence"])
            if conf > best_by_class.get(cls_id, 0.0):
                best_by_class[cls_id] = conf
        for cls_id in range(len(CLASS_NAMES)):
            self.class_hits[cls_id] = self.class_hits[cls_id] + 1 if cls_id in best_by_class else 0
        if self.consensus_class is not None and \
           time.time() - self.consensus_started_at > self.consensus_timeout:
            self.reset_consensus()
        best_cls, best_conf = None, 0.0
        for cls_id, hits in self.class_hits.items():
            if hits >= self.confirm_frames and best_by_class.get(cls_id, 0.0) > best_conf:
                best_cls, best_conf = cls_id, best_by_class[cls_id]
        if best_cls is not None:
            if self.consensus_class == best_cls:
                self.consensus_confidence = (self.ema_alpha * best_conf +
                                             (1.0 - self.ema_alpha) * self.consensus_confidence)
                self.consensus_held += 1
                self.release_counter = 0
            else:
                self.consensus_class = best_cls
                self.consensus_confidence = best_conf
                self.consensus_held = 1
                self.consensus_started_at = time.time()
                self.release_counter = 0
            return
        if self.consensus_class is not None:
            self.release_counter += 1
            if self.release_counter >= self.release_frames:
                self.reset_consensus()

    def reset_consensus(self) -> None:
        self.consensus_class = None
        self.consensus_confidence = 0.0
        self.consensus_held = 0
        self.consensus_started_at = 0.0
        self.release_counter = 0

    # ---- 发布 ----
    def publish_detections(self, dets: List[Dict[str, Any]], stamp: float) -> None:
        payload = {
            "header": {"stamp": stamp},
            "raw_detections": [
                {"class_name": d["class_name"], "class_id": d["class_id"],
                 "label": d["label"], "confidence": round(d["confidence"], 4),
                 "bbox": [round(x, 1) for x in d["bbox"]]}
                for d in dets
            ],
            "consensus": {
                "class_name": CLASS_NAMES[self.consensus_class] if self.consensus_class is not None else None,
                "label": CLASS_LABELS[self.consensus_class] if self.consensus_class is not None else None,
                "class_id": self.consensus_class,
                "confidence": round(self.consensus_confidence, 4),
                "active": self.consensus_class is not None,
                "held_frames": self.consensus_held,
            },
            "diagnostics": {
                "fps": round(1000.0 / max(self.inference_ms, 1.0), 1),
                "inference_ms": round(self.inference_ms, 1),
            },
        }
        self.det_pub.publish(String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))

    def publish_debug_image(self, frame: np.ndarray, dets: List[Dict[str, Any]]) -> None:
        annot = frame.copy()

        # 画检测框
        for det in dets:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            name = det["class_name"]
            color = CLASS_COLORS.get(name, (255, 255, 255))
            cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)
            label = "%s %.2f" % (CLASS_LABELS[CLASS_NAMES.index(name)], det["confidence"])
            cv2.putText(annot, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 共识状态
        if self.consensus_class is None:
            text = "FS: none"
        else:
            text = "FS: %s %.2f" % (CLASS_LABELS[self.consensus_class], self.consensus_confidence)
        cv2.putText(annot, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # 推理耗时
        cv2.putText(annot, "%.0fms" % self.inference_ms, (8, annot.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 保存调试截图到 scripts/ 目录
        if self.save_debug and self.consensus_class is not None:
            try:
                self._last_detection_time = getattr(self, '_last_detection_time', 0)
                now = time.time()
                if now - self._last_detection_time > 1.0:  # 每秒最多存一张
                    self._last_detection_time = now
                    fname = os.path.join(DEBUG_SAVE_DIR,
                                         "factory_sign_%s.jpg" % time.strftime("%Y%m%d_%H%M%S"))
                    cv2.imwrite(fname, annot)
                    rospy.loginfo("Debug image saved: %s", fname)
            except Exception as exc:
                rospy.logwarn_throttle(5.0, "Failed to save debug image: %s", exc)

        # 发布 ROS 话题
        if self.publish_debug:
            try:
                msg = self.bridge.cv2_to_imgmsg(annot, "bgr8")
                msg.header.stamp = rospy.Time.now()
                self.debug_pub.publish(msg)
            except Exception as exc:
                rospy.logwarn_throttle(2.0, "debug image publish failed: %s", exc)

    # ---- 语音 ----
    def maybe_speak(self) -> None:
        if not self.enable_speech or self.consensus_class is None:
            return
        class_name = CLASS_NAMES[self.consensus_class]
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
            if not text.endswith("。"):
                text += "。"
        if self.try_announce_service(class_name, text):
            return
        if self.fallback_to_speak_topic:
            rospy.logwarn("Fallback publish to %s: %s", self.speak_topic, text)
            self.speak_pub.publish(String(data=text))

    def try_announce_service(self, class_name: str, text: str) -> bool:
        if not self.use_announce_service or Announce is None:
            return False
        try:
            rospy.wait_for_service(self.announce_service, timeout=self.announce_timeout)
            announce = rospy.ServiceProxy(self.announce_service, Announce)
            event = self.announce_event
            req_text = text if event == "custom" else ""
            res = announce(event, "", "", ANNOUNCE_DECISION[class_name], req_text, self.speech_wait)
            if bool(res.success):
                rospy.loginfo("Announced via %s: %s", self.announce_service, res.speech_text)
                return True
            rospy.logwarn("Announce service failed: %s", res.message)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Announce service unavailable: %s", exc)
        return False

    def publish_status(self, status: str) -> None:
        self.status_pub.publish(String(data=status))

    def on_shutdown(self) -> None:
        self.shutdown = True
        self.publish_status("shutdown")
        try:
            if self.rknn is not None:
                self.rknn.release()
        except Exception:
            pass


def main() -> None:
    node = FactorySignRknnTestNode()
    node.run()


if __name__ == "__main__":
    main()
