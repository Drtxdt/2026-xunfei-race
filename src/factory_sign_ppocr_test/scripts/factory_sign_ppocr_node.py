#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS1 node for local PP-OCRv5 factory sign OCR and speech."""

from __future__ import annotations

import base64
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

JSON_PREFIX = "__PPOCR_JSON__"


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


@dataclass
class RecognitionResult:
    category: Optional[str] = None
    confidence: float = 0.0
    raw_text: str = ""
    texts: List[Dict[str, object]] = field(default_factory=list)
    error: str = ""
    elapsed_ms: int = 0


class FactorySignKeywordClassifier:
    KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("food", ("食品", "食", "food")),
        ("daily", ("日用品", "日用", "daily")),
        ("electronic", ("电子", "电", "electronic")),
    )

    def classify(self, text: str) -> Optional[str]:
        normalized = re.sub(r"\s+", "", text or "").lower()
        if not normalized:
            return None
        for category, keywords in self.KEYWORDS:
            if any(keyword in normalized for keyword in keywords):
                return category
        return None


class VoteWindow:
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


class LocalPPOCRClient:
    """Synchronous JSON-lines client for the local PaddleOCR worker."""

    def __init__(
        self,
        paddle_python: str,
        worker_path: str,
        lang: str,
        model_name: str,
        min_score: float,
        api: str,
        timeout_sec: float,
        startup_timeout_sec: float,
        restart_sec: float,
        logger,
    ) -> None:
        self.paddle_python = os.path.expanduser(os.path.expandvars(paddle_python or "python3"))
        self.worker_path = worker_path
        self.lang = lang
        self.model_name = model_name
        self.min_score = float(min_score)
        self.api = (api or "legacy").strip().lower()
        if self.api not in ("legacy", "predict", "auto"):
            self.api = "legacy"
        self.timeout_sec = float(timeout_sec)
        self.startup_timeout_sec = max(1.0, float(startup_timeout_sec))
        self.restart_sec = float(restart_sec)
        self.logger = logger
        self.proc = None
        self.responses = queue.Queue()
        self.stderr_lines = queue.Queue()
        self.request_id = 0
        self.pending_id = None
        self.pending_started_at = 0.0
        self.last_timeout_warn_at = 0.0
        self.last_start_attempt = 0.0
        self.last_error = ""

    def submit(self, image_b64: str) -> Tuple[bool, str]:
        if not self._ensure_worker():
            return False, self.last_error or "PaddleOCR worker unavailable"
        if self.pending_id is not None:
            return False, "PaddleOCR worker busy with request {}".format(self.pending_id)
        self.request_id += 1
        req_id = self.request_id
        payload = {"id": req_id, "image": image_b64}
        try:
            self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            self.pending_id = req_id
            self.pending_started_at = time.time()
            self.last_timeout_warn_at = 0.0
            return True, ""
        except Exception as exc:
            self._kill_worker()
            return False, "worker write failed: {}".format(exc)

    def poll(self) -> Optional[Dict[str, object]]:
        while True:
            try:
                response = self.responses.get_nowait()
            except queue.Empty:
                break
            if self.pending_id is not None and response.get("id") == self.pending_id:
                self.pending_id = None
                self.pending_started_at = 0.0
                self.last_timeout_warn_at = 0.0
                return response
        if self.pending_id is not None:
            elapsed = time.time() - self.pending_started_at
            if elapsed > self.timeout_sec and time.time() - self.last_timeout_warn_at > 5.0:
                self.last_timeout_warn_at = time.time()
                self.last_error = "PaddleOCR request {} still running for {:.1f}s".format(self.pending_id, elapsed)
                self._log_warn("%s; waiting instead of restarting worker", self.last_error)
        return None

    def is_busy(self) -> bool:
        return self.pending_id is not None

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(json.dumps({"cmd": "shutdown", "id": -1}) + "\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        self._kill_worker()

    def _ensure_worker(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return True
        now = time.time()
        if now - self.last_start_attempt < self.restart_sec:
            return False
        self.last_start_attempt = now
        return self._start_worker()

    def _start_worker(self) -> bool:
        if not os.path.isfile(self.worker_path):
            self.last_error = "worker script not found: {}".format(self.worker_path)
            self._log_warn(self.last_error)
            return False
        self._drain_responses()
        cmd = [
            self.paddle_python,
            self.worker_path,
            "--lang",
            self.lang,
            "--model-name",
            self.model_name,
            "--min-score",
            str(self.min_score),
            "--api",
            self.api,
        ]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )
            threading.Thread(target=self._read_stdout, daemon=True).start()
            threading.Thread(target=self._read_stderr, daemon=True).start()
            ready = self.responses.get(timeout=self.startup_timeout_sec)
            if ready.get("type") == "ready" and ready.get("ok"):
                self.last_error = ""
                self._log_info(
                    "Local PaddleOCR worker ready: python=%s model=%s lang=%s api=%s init=%s",
                    self.paddle_python,
                    self.model_name,
                    self.lang,
                    ready.get("api", self.api),
                    ready.get("init_kwargs", {}),
                )
                return True
            self.last_error = ready.get("error", "worker failed to initialize")
            self._log_warn("Local PaddleOCR worker failed: %s", self.last_error)
            self._kill_worker()
            return False
        except queue.Empty:
            returncode = self.proc.poll() if self.proc is not None else None
            if returncode is None:
                self.last_error = "worker startup timeout after {:.1f}s; PaddleOCR model initialization is still not ready".format(
                    self.startup_timeout_sec
                )
            else:
                self.last_error = "worker exited before ready: returncode={}".format(returncode)
            self._log_warn("Local PaddleOCR worker failed: %s", self.last_error)
            self._kill_worker()
            return False
        except Exception as exc:
            self.last_error = "failed to start worker: {}: {}".format(type(exc).__name__, exc)
            self._log_warn(self.last_error)
            self._kill_worker()
            return False

    def _drain_responses(self) -> None:
        while True:
            try:
                self.responses.get_nowait()
            except queue.Empty:
                return

    def _read_stdout(self) -> None:
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                if not line.startswith(JSON_PREFIX):
                    continue
                line = line[len(JSON_PREFIX):]
                try:
                    self.responses.put(json.loads(line))
                except Exception:
                    self._log_warn("Ignoring malformed worker JSON: %s", line[:160])
        except Exception:
            pass

    def _read_stderr(self) -> None:
        try:
            for line in self.proc.stderr:
                line = line.strip()
                if line:
                    self._log_warn("PaddleOCR worker stderr: %s", line[:220])
        except Exception:
            pass

    def _kill_worker(self) -> None:
        proc = self.proc
        self.proc = None
        self.pending_id = None
        self.pending_started_at = 0.0
        self.last_timeout_warn_at = 0.0
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _log_info(self, msg: str, *args) -> None:
        if self.logger is not None:
            self.logger.loginfo(msg, *args)

    def _log_warn(self, msg: str, *args) -> None:
        if self.logger is not None:
            self.logger.logwarn_throttle(2.0, msg, *args)


def maybe_flip_frame(frame, flip_horizontal: bool):
    if not flip_horizontal:
        return frame
    return frame[:, ::-1].copy()


def ros_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


class FactorySignPPOCRNode:
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
        self.flip_image = ros_bool(rospy.get_param("~flip", False), False)
        self.inference_rate = float(rospy.get_param("~inference_rate", 0.2))
        self.roi_scale = float(rospy.get_param("~roi_scale", 0.75))
        self.resize_scale = float(rospy.get_param("~resize_scale", 1.0))
        self.debug_publish_rate = float(rospy.get_param("~debug_publish_rate", 5.0))
        self.use_sharpen = ros_bool(rospy.get_param("~use_sharpen", True), True)
        self.use_adaptive_threshold = ros_bool(rospy.get_param("~use_adaptive_threshold", False), False)
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 90))
        self.debug_show_image = ros_bool(rospy.get_param("~debug_show_image", False), False)
        self.publish_debug_image = ros_bool(rospy.get_param("~publish_debug_image", True), True)
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "/factory_sign_ppocr_test/debug_image")
        self.preprocess_topic = rospy.get_param("~debug_preprocess_topic", "/factory_sign_ppocr_test/preprocess_image")

        self.cooldown_sec = float(rospy.get_param("~cooldown_sec", 5.0))
        self.vote = VoteWindow(
            int(rospy.get_param("~vote_window_size", 5)),
            int(rospy.get_param("~vote_min_count", 2)),
        )

        worker_path = rospy.get_param("~worker_path", "")
        if not worker_path:
            worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppocr_worker.py")
        self.ocr_client = LocalPPOCRClient(
            paddle_python=rospy.get_param("~paddle_python", "/home/ucar/ppocrv6_env/bin/python3"),
            worker_path=worker_path,
            lang=rospy.get_param("~ocr_lang", "ch"),
            model_name=rospy.get_param("~ocr_model_name", "PP-OCRv5"),
            min_score=float(rospy.get_param("~ocr_min_score", 0.45)),
            api=rospy.get_param("~ocr_api", "legacy"),
            timeout_sec=float(rospy.get_param("~ocr_timeout_sec", 120.0)),
            startup_timeout_sec=float(rospy.get_param("~worker_startup_timeout_sec", 60.0)),
            restart_sec=float(rospy.get_param("~worker_restart_sec", 3.0)),
            logger=rospy,
        )

        self.speech_mode = rospy.get_param("~speech_mode", "service").strip().lower()
        self.speech_service = rospy.get_param("~speech_service", "/competition_speech/announce")
        self.speech_timeout = float(rospy.get_param("~speech_service_timeout_sec", 0.5))
        self.speech_topic = rospy.get_param("~speech_topic", "/speak")
        self.fallback_to_topic = ros_bool(rospy.get_param("~fallback_to_speech_topic", True), True)
        self.speech_wait = ros_bool(rospy.get_param("~speech_wait", False), False)

        self.classifier = FactorySignKeywordClassifier()
        self.latest_image = None
        self.last_result = RecognitionResult()
        self.last_confirmed = None
        self.last_spoken_category = None
        self.last_spoken_at_by_category: Dict[str, float] = {}
        self.last_roi_box = (0, 0, 0, 0)
        self.last_ocr_image = None
        self.last_debug_publish_at = 0.0
        self.last_ocr_submit_error = ""

        self.speak_pub = rospy.Publisher(self.speech_topic, String, queue_size=1)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.preprocess_pub = rospy.Publisher(self.preprocess_topic, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "factory_sign_ppocr_node ready: image=%s local_paddle_python=%s ocr_api=%s debug=%s preprocess=%s speech_service=%s speech_topic=%s",
            self.image_topic,
            self.ocr_client.paddle_python,
            self.ocr_client.api,
            self.debug_image_topic,
            self.preprocess_topic,
            self.speech_service,
            self.speech_topic,
        )

    def _image_cb(self, msg) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = maybe_flip_frame(frame, self.flip_image)
            self._publish_live_debug(self.latest_image)
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "cv_bridge conversion failed: %s", exc)

    def run(self) -> None:
        rate = self.rospy.Rate(self.inference_rate)
        while not self.rospy.is_shutdown():
            if self.latest_image is not None:
                self._process_once(self.latest_image.copy())
            rate.sleep()

    def _process_once(self, frame) -> None:
        ocr_image, debug_preprocess = self._make_ocr_image(frame)
        self.last_ocr_image = debug_preprocess
        self._publish_debug_images(frame, debug_preprocess, False)
        response = self.ocr_client.poll()
        spoken = False
        if response is not None:
            result = self._result_from_response(response)
            confirmed = self.vote.push(result.category)
            self.last_result = result
            self.last_confirmed = confirmed
            spoken = self._maybe_speak(confirmed) if confirmed else False
        submitted = False
        if not self.ocr_client.is_busy():
            submitted, self.last_ocr_submit_error = self._submit_ocr(ocr_image)
        self.rospy.loginfo(
            "factory_sign_ppocr: text=%r category=%s vote=%s confirmed=%s spoken=%s elapsed_ms=%d submitted=%s busy=%s error=%s",
            self.last_result.raw_text,
            self.last_result.category,
            self.vote.snapshot(),
            self.last_confirmed,
            spoken,
            self.last_result.elapsed_ms,
            submitted,
            self.ocr_client.is_busy(),
            self.last_result.error or self.last_ocr_submit_error or self.ocr_client.last_error,
        )
        self._publish_debug_images(frame, debug_preprocess, spoken)
        if self.debug_show_image:
            self._show_debug(debug_preprocess)

    def _submit_ocr(self, image) -> Tuple[bool, str]:
        import cv2

        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return False, "cv2.imencode failed"
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return self.ocr_client.submit(image_b64)

    def _result_from_response(self, response) -> RecognitionResult:
        raw_text = str(response.get("raw_text") or "")
        texts = response.get("texts") or []
        category = self.classifier.classify(raw_text)
        confidence = 0.0
        if texts:
            try:
                confidence = max(float(item.get("score", 0.0)) for item in texts)
            except Exception:
                confidence = 0.0
        return RecognitionResult(
            category=category,
            confidence=confidence,
            raw_text=raw_text,
            texts=texts,
            error="" if response.get("ok") else str(response.get("error", "")),
            elapsed_ms=int(response.get("elapsed_ms", 0) or 0),
        )

    def _make_ocr_image(self, frame):
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
        if self.resize_scale and self.resize_scale > 1.0:
            roi = cv2.resize(roi, None, fx=self.resize_scale, fy=self.resize_scale, interpolation=cv2.INTER_CUBIC)
        if self.use_sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            roi = cv2.filter2D(roi, -1, kernel)
        if self.use_adaptive_threshold:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            debug = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
            return cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR), debug
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return roi, gray

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
            self.rospy.logwarn_throttle(2.0, "debug image publish failed: %s", exc)

    def _publish_live_debug(self, frame) -> None:
        if not self.publish_debug_image or self.debug_publish_rate <= 0:
            return
        now = time.time()
        if now - self.last_debug_publish_at < 1.0 / self.debug_publish_rate:
            return
        self.last_debug_publish_at = now
        try:
            debug = self._draw_debug(frame, False)
            msg = self.bridge.cv2_to_imgmsg(debug, "bgr8")
            msg.header.stamp = self.rospy.Time.now()
            self.debug_pub.publish(msg)
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "live debug image publish failed: %s", exc)

    def _draw_debug(self, frame, spoken: bool):
        import cv2

        out = frame.copy()
        x1, y1, x2, y2 = self.last_roi_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        lines = [
            "source=local_PP-OCRv5",
            "category={} confirmed={} conf={:.2f}".format(self.last_result.category, self.last_confirmed, self.last_result.confidence),
            "vote={}".format(self.vote.snapshot()),
            "ocr_busy={} pending={}".format(self.ocr_client.is_busy(), self.ocr_client.pending_id),
            "elapsed={}ms spoken={}".format(self.last_result.elapsed_ms, spoken),
        ]
        if self.last_result.raw_text:
            lines.append("text={}".format(self._ascii_preview(self.last_result.raw_text)))
        error = self.last_result.error or self.last_ocr_submit_error or self.ocr_client.last_error
        if error:
            lines.append("err={}".format(self._ascii_preview(error)))
        y = 24
        for line in lines:
            cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            y += 24
        return out

    @staticmethod
    def _ascii_preview(text: str) -> str:
        return (text or "").encode("ascii", "replace").decode("ascii")[:100]

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

            cv2.imshow("factory_sign_ppocr_preprocessed", image)
            cv2.waitKey(1)
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "debug_show_image failed: %s", exc)

    def _on_shutdown(self) -> None:
        self.ocr_client.shutdown()
        try:
            import cv2

            cv2.destroyWindow("factory_sign_ppocr_preprocessed")
        except Exception:
            pass


def main() -> None:
    import rospy

    rospy.init_node("factory_sign_ppocr_node")
    node = FactorySignPPOCRNode()
    node.run()


if __name__ == "__main__":
    main()
