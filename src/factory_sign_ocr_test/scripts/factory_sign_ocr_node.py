#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone ROS1 node for factory sign OCR recognition and speech."""

from __future__ import annotations

import os
import re
import time
from collections import Counter, deque
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple


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
    """Sliding vote window for stable multi-frame OCR decisions."""

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


class FactorySignOCR:
    """OCR engine wrapper with RKNN skeleton and CPU fallback."""

    def __init__(
        self,
        use_rknn: bool = True,
        rknn_model_path: str = "",
        cpu_engine: str = "auto",
        logger: Optional[object] = None,
    ) -> None:
        self.use_rknn = bool(use_rknn)
        self.rknn_model_path = os.path.expanduser(os.path.expandvars(rknn_model_path or ""))
        self.cpu_engine = (cpu_engine or "auto").strip().lower()
        self.logger = logger
        self.rknn = None
        self.rknn_available = False
        self._rknn_warned = False
        self.cpu_reader: Optional[Callable[[object], str]] = None
        self.engine_name = "none"

        if self.use_rknn:
            self._try_init_rknn()
        self._try_init_cpu_ocr()

    def recognize(self, image) -> str:
        if self.rknn_available:
            text = self._recognize_rknn(image)
            if text:
                return text
        if self.cpu_reader is None:
            self._log(
                "err",
                "No OCR backend is available. Install one of: "
                "python3 -m pip install paddleocr paddlepaddle, "
                "python3 -m pip install easyocr, or install tesseract-ocr + pytesseract.",
            )
            return ""
        try:
            return self.cpu_reader(image).strip()
        except Exception as exc:
            self._log("err", "CPU OCR inference failed: %s", exc)
            return ""

    def release(self) -> None:
        if self.rknn is not None:
            try:
                self.rknn.release()
            except Exception:
                pass

    def _try_init_rknn(self) -> None:
        if not self.rknn_model_path or not os.path.isfile(self.rknn_model_path):
            self._log(
                "warn",
                "OCR RKNN model not found; rknn_model_path is empty or invalid. "
                "Falling back to CPU OCR.",
            )
            return
        try:
            try:
                from rknnlite.api import RKNNLite

                rknn = RKNNLite()
                core_mask = RKNNLite.NPU_CORE_0_1_2
                target = None
            except Exception:
                from rknn.api import RKNN

                rknn = RKNN(verbose=False)
                core_mask = None
                target = "rk3588"
            ret = rknn.load_rknn(self.rknn_model_path)
            if ret != 0:
                self._log("warn", "load_rknn(%s) failed: %s", self.rknn_model_path, ret)
                return
            try:
                ret = rknn.init_runtime(core_mask=core_mask) if core_mask is not None else rknn.init_runtime(target=target)
            except TypeError:
                ret = rknn.init_runtime()
            if ret != 0:
                self._log("warn", "init_runtime for OCR RKNN failed: %s", ret)
                try:
                    rknn.release()
                except Exception:
                    pass
                return
            self.rknn = rknn
            self.rknn_available = True
            self.engine_name = "rknn+cpu-fallback"
            self._log("info", "OCR RKNN runtime loaded: %s", self.rknn_model_path)
        except Exception as exc:
            self._log("warn", "RKNN OCR unavailable: %s. Falling back to CPU OCR.", exc)

    def _recognize_rknn(self, image) -> str:
        # The project currently has factory-sign RKNN detection models, not an OCR
        # recognizer/decoder pair. Keep the NPU hook here and use CPU OCR until a
        # real OCR RKNN model plus decoder is provided.
        if not self._rknn_warned:
            self._log(
                "warn",
                "OCR RKNN runtime is present, but no OCR decoder is implemented for this model. "
                "Using CPU OCR fallback for text recognition.",
            )
            self._rknn_warned = True
        return ""

    def _try_init_cpu_ocr(self) -> None:
        engines = [self.cpu_engine] if self.cpu_engine != "auto" else ["paddleocr", "easyocr", "tesseract"]
        for engine in engines:
            if engine == "paddleocr" and self._init_paddleocr():
                return
            if engine == "easyocr" and self._init_easyocr():
                return
            if engine in ("tesseract", "pytesseract") and self._init_tesseract():
                return
        self._log(
            "err",
            "CPU OCR backend not found. Suggested installs: "
            "python3 -m pip install paddleocr paddlepaddle; "
            "or python3 -m pip install easyocr; "
            "or sudo apt install tesseract-ocr tesseract-ocr-chi-sim && python3 -m pip install pytesseract.",
        )

    def _init_paddleocr(self) -> bool:
        try:
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)

            def reader(image) -> str:
                result = ocr.ocr(image, cls=True)
                texts: List[str] = []
                for page in result or []:
                    for line in page or []:
                        if len(line) >= 2 and line[1]:
                            texts.append(str(line[1][0]))
                return " ".join(texts)

            self.cpu_reader = reader
            self.engine_name = "paddleocr"
            self._log("info", "CPU OCR backend: PaddleOCR")
            return True
        except Exception as exc:
            self._log("warn", "PaddleOCR unavailable: %s", exc)
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

            def reader(image) -> str:
                try:
                    return pytesseract.image_to_string(image, lang="chi_sim+eng")
                except Exception:
                    return pytesseract.image_to_string(image)

            self.cpu_reader = reader
            self.engine_name = "tesseract"
            self._log("info", "CPU OCR backend: pytesseract")
            return True
        except Exception as exc:
            self._log("warn", "pytesseract unavailable: %s", exc)
            return False

    def _log(self, level: str, message: str, *args) -> None:
        if self.logger is None:
            return
        fn = getattr(self.logger, "log" + level, None)
        if fn is None:
            fn = getattr(self.logger, level, None)
        if fn is not None:
            fn(message, *args)


class FactorySignOCRNode:
    def __init__(self) -> None:
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self.rospy = rospy
        self.String = String
        self.bridge = CvBridge()
        self.image_msg_type = Image

        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.inference_rate = float(rospy.get_param("~inference_rate", 3.0))
        self.cooldown_sec = float(rospy.get_param("~cooldown_sec", 5.0))
        self.roi_scale = float(rospy.get_param("~roi_scale", 0.8))
        self.resize_scale = float(rospy.get_param("~resize_scale", 1.8))
        self.use_adaptive_threshold = bool(rospy.get_param("~use_adaptive_threshold", True))
        self.use_sharpen = bool(rospy.get_param("~use_sharpen", True))
        self.debug_show_image = bool(rospy.get_param("~debug_show_image", False))
        self.speech_mode = rospy.get_param("~speech_mode", "service").strip().lower()
        self.speech_service = rospy.get_param("~speech_service", "/competition_speech/announce")
        self.speech_timeout = float(rospy.get_param("~speech_service_timeout_sec", 0.5))
        self.speech_topic = rospy.get_param("~speech_topic", "/speak")
        self.fallback_to_topic = bool(rospy.get_param("~fallback_to_speech_topic", True))
        self.speech_wait = bool(rospy.get_param("~speech_wait", False))

        use_rknn = bool(rospy.get_param("~use_rknn", True))
        rknn_model_path = rospy.get_param("~rknn_model_path", "")
        cpu_ocr_engine = rospy.get_param("~cpu_ocr_engine", "auto")

        self.classifier = FactorySignClassifier()
        self.vote = VoteWindow(
            int(rospy.get_param("~vote_window_size", 5)),
            int(rospy.get_param("~vote_min_count", 2)),
        )
        self.ocr = FactorySignOCR(
            use_rknn=use_rknn,
            rknn_model_path=rknn_model_path,
            cpu_engine=cpu_ocr_engine,
            logger=rospy,
        )

        self.latest_image = None
        self.latest_stamp = 0.0
        self.last_spoken_category: Optional[str] = None
        self.last_spoken_at_by_category: Dict[str, float] = {}

        self.speak_pub = rospy.Publisher(self.speech_topic, String, queue_size=1)
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self._image_cb,
            queue_size=1,
            buff_size=2 ** 24,
        )
        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo(
            "factory_sign_ocr_node ready: image_topic=%s ocr=%s speech_mode=%s service=%s topic=%s",
            self.image_topic,
            self.ocr.engine_name,
            self.speech_mode,
            self.speech_service,
            self.speech_topic,
        )

    def _image_cb(self, msg) -> None:
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_stamp = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "cv_bridge conversion failed: %s", exc)

    def run(self) -> None:
        rate = self.rospy.Rate(self.inference_rate)
        while not self.rospy.is_shutdown():
            if self.latest_image is not None:
                self._process_once(self.latest_image.copy())
            rate.sleep()

    def _process_once(self, frame) -> None:
        processed = self._preprocess(frame)
        raw_text = self.ocr.recognize(processed)
        category = self.classifier.classify(raw_text)
        confirmed = self.vote.push(category)
        spoken = False

        if confirmed:
            spoken = self._maybe_speak(confirmed)

        self.rospy.loginfo(
            "factory_sign_ocr: text=%r category=%s vote=%s confirmed=%s spoken=%s",
            raw_text,
            category,
            self.vote.snapshot(),
            confirmed,
            spoken,
        )

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
        roi = frame[y0 : y0 + roi_h, x0 : x0 + roi_w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.resize_scale and self.resize_scale > 1.0:
            gray = cv2.resize(gray, None, fx=self.resize_scale, fy=self.resize_scale, interpolation=cv2.INTER_CUBIC)
        if self.use_sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            gray = cv2.filter2D(gray, -1, kernel)
        if self.use_adaptive_threshold:
            processed = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                9,
            )
        else:
            _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return processed

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
            self.rospy.logwarn("Publishing speech fallback to %s: %s", self.speech_topic, text)
            self.speak_pub.publish(self.String(data=text))
            return True
        return False

    def _try_speech_service(self, category: str, text: str) -> bool:
        try:
            from ucar_2026_competition_speech.srv import Announce
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "Announce service type unavailable: %s", exc)
            return False
        try:
            self.rospy.wait_for_service(self.speech_service, timeout=self.speech_timeout)
            announce = self.rospy.ServiceProxy(self.speech_service, Announce)
            res = announce("custom", "", "", category, text, self.speech_wait)
            if bool(res.success):
                self.rospy.loginfo("Speech announced via %s: %s", self.speech_service, res.speech_text)
                return True
            self.rospy.logwarn("Speech service returned failure: %s", res.message)
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "Speech service unavailable: %s", exc)
        return False

    def _show_debug(self, image) -> None:
        try:
            import cv2

            cv2.imshow("factory_sign_ocr_preprocessed", image)
            cv2.waitKey(1)
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "debug_show_image failed: %s", exc)

    def _on_shutdown(self) -> None:
        self.ocr.release()
        try:
            import cv2

            cv2.destroyWindow("factory_sign_ocr_preprocessed")
        except Exception:
            pass


def main() -> None:
    import rospy

    rospy.init_node("factory_sign_ocr_node")
    node = FactorySignOCRNode()
    node.run()


if __name__ == "__main__":
    main()
