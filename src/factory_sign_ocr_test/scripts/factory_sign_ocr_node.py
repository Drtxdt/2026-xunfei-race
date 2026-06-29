#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone ROS1 node for factory sign recognition and speech."""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple


CATEGORY_NAMES = {
    "food": "食品加工车间",
    "daily": "日用品加工车间",
    "electronic": "电子产品生产车间",
}

SPEECH_TEXTS = {
    "food": "识别到食品加工车间",
    "daily": "识别到日用品加工车间",
    "electronic": "识别到电子产品生产车间",
}

RKNN_CLASS_NAMES = ["food", "electronic", "daily"]

def _repair_ros_logging() -> None:
    """Repair rospy logging after RKNN libraries rename level names to I/W/E/F."""
    levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    for name, value in levels.items():
        logging.addLevelName(value, name)
    try:
        import rosgraph.roslogging as roslogging

        mapping = getattr(roslogging, "_logging_to_rospy_names", None)
        if isinstance(mapping, dict):
            for short, full in {"D": "DEBUG", "I": "INFO", "W": "WARNING", "E": "ERROR", "F": "CRITICAL"}.items():
                if short not in mapping and full in mapping:
                    mapping[short] = mapping[full]
            if "WARNING" in mapping and "WARN" not in mapping:
                mapping["WARN"] = mapping["WARNING"]
            if "CRITICAL" in mapping and "FATAL" not in mapping:
                mapping["FATAL"] = mapping["CRITICAL"]
    except Exception:
        pass


def _safe_rospy_log(logger, level: str, message: str, *args) -> None:
    if logger is None:
        return
    _repair_ros_logging()
    fn = getattr(logger, "log" + level, None) or getattr(logger, level, None)
    if fn is None:
        return
    try:
        fn(message, *args)
    except KeyError:
        _repair_ros_logging()
        fn(message, *args)


@dataclass
class RecognitionResult:
    category: Optional[str] = None
    confidence: float = 0.0
    source: str = "none"
    raw_text: str = ""
    detections: List[Dict[str, object]] = field(default_factory=list)
    error: str = ""


class FactorySignClassifier:
    """Map OCR text to the three requested factory sign categories."""

    KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("daily", ("日用品", "日用", "daily")),
        ("electronic", ("电子", "电", "electronic")),
        ("food", ("食品", "食", "food")),
    )

    def classify(self, text: str) -> Optional[str]:
        normalized = self._normalize(text)
        if not normalized:
            return None
        for category, keywords in self.KEYWORDS:
            if any(keyword in normalized for keyword in keywords):
                return category
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", text or "").lower()


class VoteWindow:
    """Sliding vote window for stable multi-frame recognition decisions."""

    def __init__(self, size: int, min_count: int) -> None:
        self.size = max(1, int(size))
        self.min_count = max(1, int(min_count))
        self._items: Deque[Optional[str]] = deque(maxlen=self.size)

    def push(self, category: Optional[str]) -> Optional[str]:
        self._items.append(category)
        counts = Counter(item for item in self._items if item)
        if not counts:
            return None
        top_count = max(counts.values())
        if top_count < self.min_count:
            return None
        winners = {name for name, count in counts.items() if count == top_count}
        for item in reversed(self._items):
            if item in winners:
                return item
        return None

    def snapshot(self) -> List[Optional[str]]:
        return list(self._items)


class FactorySignRecognizer:
    """Prefer RKNN factory-sign classification and fall back to OCR text."""

    def __init__(self, classifier, rknn_backend=None, ocr_backend=None, mode: str = "auto") -> None:
        self.classifier = classifier
        self.rknn_backend = rknn_backend
        self.ocr_backend = ocr_backend
        self.mode = (mode or "auto").strip().lower()

    def recognize(self, frame) -> RecognitionResult:
        if self.mode in ("auto", "rknn_classifier") and self.rknn_backend is not None:
            result = self.rknn_backend.recognize(frame)
            if result.category:
                return result
            if self.mode == "rknn_classifier":
                return result

        if self.mode in ("auto", "rapidocr", "tesseract", "ocr") and self.ocr_backend is not None:
            text = self.ocr_backend.recognize(frame)
            return RecognitionResult(
                category=self.classifier.classify(text),
                confidence=0.0,
                source="ocr",
                raw_text=text or "",
            )
        return RecognitionResult(source="none", error="no recognition backend available")

    def release(self) -> None:
        for backend in (self.rknn_backend, self.ocr_backend):
            release = getattr(backend, "release", None)
            if callable(release):
                release()


class RknnFactorySignClassifierBackend:
    """Wrapper around the existing YOLOv5 RKNN factory-sign model."""

    def __init__(self, model_path: str, confidence: float, nms: float, input_size: int, logger=None) -> None:
        self.logger = logger
        self.model_path = self._resolve_model_path(model_path)
        self.confidence = float(confidence)
        self.nms = float(nms)
        self.input_size = int(input_size)
        self.rknn = None
        self.vm = None
        self.output_shapes_logged = False
        self.available = False
        self.last_detections: List[Dict[str, object]] = []
        self._load()

    def recognize(self, frame) -> RecognitionResult:
        if not self.available:
            return RecognitionResult(source="rknn", error="rknn backend unavailable")
        try:
            boxes, classes, scores, self.output_shapes_logged = self.vm.infer_frame(
                self.rknn,
                frame,
                RKNN_CLASS_NAMES,
                self.confidence,
                self.nms,
                self.input_size,
                self.output_shapes_logged,
            )
            _repair_ros_logging()
            dets = self.vm.build_detections(boxes, classes, scores, RKNN_CLASS_NAMES)
            self.last_detections = dets
            if not dets:
                return RecognitionResult(source="rknn", detections=[])
            best = dets[0]
            category = str(best["class_name"])
            return RecognitionResult(
                category=category,
                confidence=float(best["confidence"]),
                source="rknn",
                detections=dets,
            )
        except Exception as exc:
            self._log("warn", "RKNN factory sign inference failed: %s", exc)
            return RecognitionResult(source="rknn", error=str(exc))

    def release(self) -> None:
        if self.rknn is not None:
            try:
                self.rknn.release()
            except Exception:
                pass

    def _load(self) -> None:
        if not self.model_path or not os.path.isfile(self.model_path):
            self._log("warn", "Factory sign RKNN model not found: %s", self.model_path)
            return
        try:
            self.vm = self._load_validate_model_module()
            loaded = self.vm.load_rknn_model(self.model_path)
            self.rknn = loaded[0] if isinstance(loaded, tuple) else loaded
            self.available = True
            self._log("info", "Factory sign RKNN classifier loaded: %s", self.model_path)
        except Exception as exc:
            self._log("warn", "Factory sign RKNN classifier unavailable: %s", exc)

    def _resolve_model_path(self, model_path: str) -> str:
        path = os.path.expanduser(os.path.expandvars(model_path or ""))
        if path and os.path.isfile(path):
            return os.path.abspath(path)
        try:
            import rospkg

            yolo_dir = rospkg.RosPack().get_path("yolo")
            default = os.path.join(yolo_dir, "models", "factory_sign_3cls.rknn")
            if os.path.isfile(default):
                return default
        except Exception:
            pass
        here = os.path.abspath(os.path.dirname(__file__))
        workspace = os.path.abspath(os.path.join(here, "..", ".."))
        default = os.path.join(workspace, "yolo", "models", "factory_sign_3cls.rknn")
        return default

    def _load_validate_model_module(self):
        try:
            import rospkg

            yolo_dir = rospkg.RosPack().get_path("yolo")
        except Exception:
            yolo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "yolo"))
        module_path = os.path.join(yolo_dir, "validate_model.py")
        if not os.path.isfile(module_path):
            raise RuntimeError("missing yolo/validate_model.py at {}".format(module_path))
        if yolo_dir not in sys.path:
            sys.path.insert(0, yolo_dir)
        spec = importlib.util.spec_from_file_location("factory_sign_validate_model", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    def _log(self, level: str, message: str, *args) -> None:
        if self.logger is None:
            return
        fn = getattr(self.logger, "log" + level, None) or getattr(self.logger, level, None)
        if fn is not None:
            fn(message, *args)


class FactorySignOCR:
    """CPU OCR backend. RapidOCR is preferred; Tesseract is the final fallback."""

    def __init__(self, cpu_engine: str = "auto", logger=None) -> None:
        self.cpu_engine = (cpu_engine or "auto").strip().lower()
        self.logger = logger
        self.cpu_reader: Optional[Callable[[object], str]] = None
        self.engine_name = "none"
        self._try_init_cpu_ocr()

    def recognize(self, image) -> str:
        if self.cpu_reader is None:
            self._log(
                "err",
                "No OCR backend available. For Python 3.7.3 try: python3 -m pip install rapidocr_onnxruntime==1.3.24; "
                "or install tesseract-ocr + pytesseract as fallback.",
            )
            return ""
        try:
            return (self.cpu_reader(image) or "").strip()
        except Exception as exc:
            self._log("err", "CPU OCR inference failed: %s", exc)
            return ""

    def _try_init_cpu_ocr(self) -> None:
        engines = [self.cpu_engine] if self.cpu_engine != "auto" else ["rapidocr", "easyocr", "tesseract"]
        for engine in engines:
            if engine == "rapidocr" and self._init_rapidocr():
                return
            if engine == "easyocr" and self._init_easyocr():
                return
            if engine in ("tesseract", "pytesseract") and self._init_tesseract():
                return
        self._log(
            "err",
            "CPU OCR backend not found. Recommended for Python 3.7.3: python3 -m pip install rapidocr_onnxruntime==1.3.24. "
            "Fallback: sudo apt install tesseract-ocr tesseract-ocr-chi-sim && python3 -m pip install pytesseract.",
        )

    def _init_rapidocr(self) -> bool:
        try:
            from rapidocr_onnxruntime import RapidOCR

            engine = RapidOCR()

            def reader(image) -> str:
                result, _elapsed = engine(image)
                texts = []
                for item in result or []:
                    if len(item) >= 2:
                        texts.append(str(item[1]))
                return " ".join(texts)

            self.cpu_reader = reader
            self.engine_name = "rapidocr"
            self._log("info", "CPU OCR backend: RapidOCR")
            return True
        except Exception as exc:
            self._log("warn", "RapidOCR unavailable: %s", exc)
            return False

    def _init_easyocr(self) -> bool:
        try:
            import easyocr

            reader_obj = easyocr.Reader(["ch_sim", "en"], gpu=False)

            def reader(image) -> str:
                result = reader_obj.readtext(image, detail=0, paragraph=True)
                return " ".join(str(item) for item in result or [])

            self.cpu_reader = reader
            self.engine_name = "easyocr"
            self._log("info", "CPU OCR backend: EasyOCR")
            return True
        except Exception as exc:
            self._log("warn", "EasyOCR unavailable: %s", exc)
            return False

    def _init_tesseract(self) -> bool:
        try:
            import pytesseract
            import cv2

            configs = [
                "--oem 3 --psm 6",
                "--oem 3 --psm 7",
                "--oem 3 --psm 11",
            ]

            def reader(image) -> str:
                variants = self._tesseract_variants(image, cv2)
                texts = []
                for variant in variants:
                    for config in configs:
                        try:
                            texts.append(pytesseract.image_to_string(variant, lang="chi_sim+eng", config=config))
                        except Exception:
                            texts.append(pytesseract.image_to_string(variant, config=config))
                return " ".join(t.strip() for t in texts if t and t.strip())

            self.cpu_reader = reader
            self.engine_name = "tesseract"
            self._log("info", "CPU OCR backend: pytesseract multi-psm")
            return True
        except Exception as exc:
            self._log("warn", "pytesseract unavailable: %s", exc)
            return False

    @staticmethod
    def _tesseract_variants(image, cv2):
        gray = image
        if len(getattr(image, "shape", [])) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variants = [gray]
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        variants.append(cv2.bitwise_not(otsu))
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        variants.append(adaptive)
        return variants

    def _log(self, level: str, message: str, *args) -> None:
        if self.logger is None:
            return
        fn = getattr(self.logger, "log" + level, None) or getattr(self.logger, level, None)
        if fn is not None:
            fn(message, *args)


class FactorySignOCRNode:
    def __init__(self) -> None:
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self.rospy = rospy
        self.Image = Image
        self.String = String
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "/factory_sign_ocr_test/debug_image")
        self.preprocess_image_topic = rospy.get_param("~debug_preprocess_topic", "/factory_sign_ocr_test/preprocess_image")
        self.publish_debug_image = bool(rospy.get_param("~publish_debug_image", True))
        self.inference_rate = float(rospy.get_param("~inference_rate", 5.0))
        self.cooldown_sec = float(rospy.get_param("~cooldown_sec", 5.0))
        self.roi_scale = float(rospy.get_param("~roi_scale", 0.8))
        self.resize_scale = float(rospy.get_param("~resize_scale", 1.8))
        self.use_adaptive_threshold = bool(rospy.get_param("~use_adaptive_threshold", True))
        self.use_sharpen = bool(rospy.get_param("~use_sharpen", True))
        self.debug_show_image = bool(rospy.get_param("~debug_show_image", False))
        self.recognition_mode = rospy.get_param("~recognition_mode", "rknn_classifier").strip().lower()
        self.cpu_ocr_engine = rospy.get_param("~cpu_ocr_engine", "auto")
        self.enable_ocr_fallback = bool(rospy.get_param("~enable_ocr_fallback", False))
        self.classifier_model_path = rospy.get_param("~classifier_model_path", "")
        self.classifier_confidence = float(rospy.get_param("~classifier_confidence_threshold", 0.25))
        self.classifier_nms = float(rospy.get_param("~classifier_nms_iou_threshold", 0.45))
        self.classifier_input_size = int(rospy.get_param("~classifier_input_size", 640))
        self.speech_mode = rospy.get_param("~speech_mode", "service").strip().lower()
        self.speech_service = rospy.get_param("~speech_service", "/competition_speech/announce")
        self.speech_timeout = float(rospy.get_param("~speech_service_timeout_sec", 0.5))
        self.speech_topic = rospy.get_param("~speech_topic", "/speak")
        self.fallback_to_topic = bool(rospy.get_param("~fallback_to_speech_topic", True))
        self.speech_wait = bool(rospy.get_param("~speech_wait", False))

        self.classifier = FactorySignClassifier()
        self.vote = VoteWindow(
            int(rospy.get_param("~vote_window_size", 5)),
            int(rospy.get_param("~vote_min_count", 2)),
        )
        rknn_backend = None
        if self.recognition_mode in ("auto", "rknn_classifier"):
            rknn_backend = RknnFactorySignClassifierBackend(
                self.classifier_model_path,
                self.classifier_confidence,
                self.classifier_nms,
                self.classifier_input_size,
                logger=rospy,
            )
        ocr_backend = None
        if self.recognition_mode in ("rapidocr", "tesseract", "ocr") or (
            self.recognition_mode == "auto" and self.enable_ocr_fallback
        ):
            ocr_backend = FactorySignOCR(cpu_engine=self.cpu_ocr_engine, logger=rospy)
        self.recognizer = FactorySignRecognizer(self.classifier, rknn_backend, ocr_backend, self.recognition_mode)

        self.latest_image = None
        self.latest_stamp = 0.0
        self.last_result = RecognitionResult()
        self.last_confirmed: Optional[str] = None
        self.last_spoken_category: Optional[str] = None
        self.last_spoken_at_by_category: Dict[str, float] = {}
        self.last_roi_box = (0, 0, 0, 0)

        self.speak_pub = rospy.Publisher(self.speech_topic, String, queue_size=1)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.preprocess_pub = rospy.Publisher(self.preprocess_image_topic, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.on_shutdown(self._on_shutdown)
        _repair_ros_logging()
        rospy.loginfo(
            "factory_sign_ocr_node ready: image=%s mode=%s ocr_fallback=%s debug=%s preprocess=%s speech_service=%s speech_topic=%s",
            self.image_topic,
            self.recognition_mode,
            self.enable_ocr_fallback,
            self.debug_image_topic,
            self.preprocess_image_topic,
            self.speech_service,
            self.speech_topic,
        )

    def _image_cb(self, msg) -> None:
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_stamp = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
        except Exception as exc:
            _repair_ros_logging()
            self.rospy.logwarn_throttle(2.0, "cv_bridge conversion failed: %s", exc)

    def run(self) -> None:
        rate = self.rospy.Rate(self.inference_rate)
        while not self.rospy.is_shutdown():
            if self.latest_image is not None:
                self._process_once(self.latest_image.copy())
            rate.sleep()

    def _process_once(self, frame) -> None:
        processed = self._preprocess(frame)
        source_frame = frame if self.recognition_mode in ("auto", "rknn_classifier") else processed
        result = self.recognizer.recognize(source_frame)
        if (
            self.enable_ocr_fallback
            and not result.category
            and self.recognition_mode in ("auto", "rknn_classifier")
        ):
            # If RKNN did not lock, run OCR on the preprocessed ROI when available.
            ocr_backend = getattr(self.recognizer, "ocr_backend", None)
            if ocr_backend is not None:
                text = ocr_backend.recognize(processed)
                result.raw_text = text or result.raw_text
                result.category = self.classifier.classify(text)
                if result.category:
                    result.source = "ocr"
        confirmed = self.vote.push(result.category)
        self.last_result = result
        self.last_confirmed = confirmed
        spoken = self._maybe_speak(confirmed) if confirmed else False

        _repair_ros_logging()
        self.rospy.loginfo(
            "factory_sign: source=%s text=%r category=%s conf=%.3f vote=%s confirmed=%s spoken=%s error=%s",
            result.source,
            result.raw_text,
            result.category,
            result.confidence,
            self.vote.snapshot(),
            confirmed,
            spoken,
            result.error,
        )
        self._publish_debug_images(frame, processed, spoken)
        if self.debug_show_image:
            self._show_debug(processed)

    def _preprocess(self, frame):
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        scale = min(max(self.roi_scale, 0.1), 1.0)
        roi_w = int(w * scale)
        roi_h = int(h * scale)
        x0 = max(0, (w - roi_w) // 2)
        y0 = max(0, (h - roi_h) // 2)
        self.last_roi_box = (x0, y0, x0 + roi_w, y0 + roi_h)
        roi = frame[y0 : y0 + roi_h, x0 : x0 + roi_w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.resize_scale and self.resize_scale > 1.0:
            gray = cv2.resize(gray, None, fx=self.resize_scale, fy=self.resize_scale, interpolation=cv2.INTER_CUBIC)
        if self.use_sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            gray = cv2.filter2D(gray, -1, kernel)
        if self.use_adaptive_threshold:
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return processed

    def _publish_debug_images(self, frame, processed, spoken: bool) -> None:
        if not self.publish_debug_image:
            return
        try:
            debug = self._draw_debug(frame, spoken)
            msg = self.bridge.cv2_to_imgmsg(debug, "bgr8")
            msg.header.stamp = self.rospy.Time.now()
            self.debug_pub.publish(msg)
            prep_msg = self.bridge.cv2_to_imgmsg(processed, "mono8")
            prep_msg.header.stamp = msg.header.stamp
            self.preprocess_pub.publish(prep_msg)
        except Exception as exc:
            _repair_ros_logging()
            self.rospy.logwarn_throttle(2.0, "debug image publish failed: %s", exc)

    def _draw_debug(self, frame, spoken: bool):
        import cv2

        out = frame.copy()
        x1, y1, x2, y2 = self.last_roi_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for det in self.last_result.detections:
            bx1, by1, bx2, by2 = [int(v) for v in det.get("bbox", [0, 0, 0, 0])]
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(out, "{} {:.2f}".format(det.get("class_name", "?"), float(det.get("confidence", 0.0))),
                        (bx1, max(20, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        lines = [
            "mode={} source={}".format(self.recognition_mode, self.last_result.source),
            "category={} confirmed={} conf={:.2f}".format(self.last_result.category, self.last_confirmed, self.last_result.confidence),
            "vote={}".format(self.vote.snapshot()),
            "spoken={}".format(spoken),
        ]
        if self.last_result.raw_text:
            lines.append("ocr={}".format(self._ascii_preview(self.last_result.raw_text)))
        if self.last_result.error:
            lines.append("err={}".format(self._ascii_preview(self.last_result.error)))
        y = 24
        for line in lines:
            cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            y += 24
        return out

    @staticmethod
    def _ascii_preview(text: str) -> str:
        return text.encode("ascii", "replace").decode("ascii")[:80]

    def _maybe_speak(self, category: str) -> bool:
        now = time.time()
        last_at = self.last_spoken_at_by_category.get(category, 0.0)
        if category == self.last_spoken_category and now - last_at < self.cooldown_sec:
            return False
        text = SPEECH_TEXTS[category]
        self.last_spoken_category = category
        self.last_spoken_at_by_category[category] = now
        if self.speech_mode in ("service", "auto") and self._try_speech_service(category, text):
            return True
        if self.speech_mode == "topic" or self.fallback_to_topic:
            _repair_ros_logging()
            self.rospy.logwarn("Publishing speech fallback to %s: %s", self.speech_topic, text)
            self.speak_pub.publish(self.String(data=text))
            return True
        return False

    def _try_speech_service(self, category: str, text: str) -> bool:
        try:
            from ucar_2026_competition_speech.srv import Announce
        except Exception as exc:
            _repair_ros_logging()
            self.rospy.logwarn_throttle(2.0, "Announce service type unavailable: %s", exc)
            return False
        try:
            self.rospy.wait_for_service(self.speech_service, timeout=self.speech_timeout)
            announce = self.rospy.ServiceProxy(self.speech_service, Announce)
            res = announce("custom", "", "", category, text, self.speech_wait)
            if bool(res.success):
                _repair_ros_logging()
                self.rospy.loginfo("Speech announced via %s: %s", self.speech_service, res.speech_text)
                return True
            _repair_ros_logging()
            self.rospy.logwarn("Speech service returned failure: %s", res.message)
        except Exception as exc:
            _repair_ros_logging()
            self.rospy.logwarn_throttle(2.0, "Speech service unavailable: %s", exc)
        return False

    def _show_debug(self, image) -> None:
        try:
            import cv2

            cv2.imshow("factory_sign_ocr_preprocessed", image)
            cv2.waitKey(1)
        except Exception as exc:
            _repair_ros_logging()
            self.rospy.logwarn_throttle(2.0, "debug_show_image failed: %s", exc)

    def _on_shutdown(self) -> None:
        self.recognizer.release()
        try:
            import cv2

            cv2.destroyWindow("factory_sign_ocr_preprocessed")
        except Exception:
            pass


def main() -> None:
    import rospy

    _repair_ros_logging()
    rospy.init_node("factory_sign_ocr_node")
    _repair_ros_logging()
    node = FactorySignOCRNode()
    node.run()


if __name__ == "__main__":
    main()







