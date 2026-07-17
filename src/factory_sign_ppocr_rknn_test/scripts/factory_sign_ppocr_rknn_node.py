#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS1 RKNNLite PPOCR factory sign OCR and speech test node."""

from __future__ import annotations

import logging
import json
import os
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple


CATEGORY_NAMES = {
    "food": "食品加工车间",
    "daily": "日用品加工车间",
    "electronic": "电子产品加工车间",
}

SPEECH_TEXTS = {
    "food": "识别到食品加工车间",
    "daily": "识别到日用品加工车间",
    "electronic": "识别到电子产品加工车间",
}


def repair_ros_logging() -> None:
    """RKNN libraries can rename Python logging levels to I/W/E/F."""
    for name, value in {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARN": logging.WARNING,
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
            for short, full in {"D": "DEBUG", "I": "INFO", "W": "WARNING", "E": "ERROR", "F": "CRITICAL"}.items():
                if short not in mapping and full in mapping:
                    mapping[short] = mapping[full]
            if "WARNING" in mapping and "WARN" not in mapping:
                mapping["WARN"] = mapping["WARNING"]
            if "CRITICAL" in mapping and "FATAL" not in mapping:
                mapping["FATAL"] = mapping["CRITICAL"]
    except Exception:
        pass


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


def maybe_flip_frame(frame, flip_horizontal: bool):
    if not flip_horizontal:
        return frame
    return frame[:, ::-1].copy()


def normalize_ocr_target(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in ("electronic", "electronics", "电子", "电子产品"):
        return "electronic"
    if text in ("daily", "日用", "日用品"):
        return "daily"
    if text in ("food", "食品"):
        return "food"
    return ""


def parse_view_scales(value, fallback: float = 0.8) -> List[float]:
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = re.split(r"[,;\s]+", str(value or "").strip())
    scales = []
    for item in raw:
        try:
            scale = min(1.0, max(0.1, float(item)))
        except (TypeError, ValueError):
            continue
        if scale not in scales:
            scales.append(scale)
    return scales or [min(1.0, max(0.1, float(fallback)))]


def map_box_to_frame(box, roi_box, scale_factor: float = 1.0):
    x0, y0, _x1, _y1 = roi_box
    factor = max(1e-6, float(scale_factor))
    return [[float(point[0]) / factor + x0, float(point[1]) / factor + y0] for point in box]


@dataclass
class OCRText:
    text: str
    score: float = 0.0
    box: List[List[float]] = field(default_factory=list)


@dataclass
class RecognitionResult:
    category: Optional[str] = None
    confidence: float = 0.0
    raw_text: str = ""
    texts: List[OCRText] = field(default_factory=list)
    match_debug: str = ""
    error: str = ""
    elapsed_ms: int = 0
    det_ms: int = 0
    rec_ms: int = 0
    category_score: float = 0.0
    evidence: str = ""
    view_scale: float = 1.0
    candidate_count: int = 0


@dataclass
class OCRCandidate:
    crop: object
    box: List[List[float]]
    det_score: float = 0.0
    center_score: float = 0.0
    sign_score: float = 0.0
    rank_score: float = 0.0
    height: float = 0.0


class FactorySignKeywordClassifier:
    """Strict factory-sign classifier; generic wall text is never evidence."""

    FEATURES: Dict[str, Tuple[Tuple[str, float], ...]] = {
        "food": (("食品", 1.0), ("food", 1.0)),
        "daily": (("日用品", 1.0), ("日用", 0.92), ("用品", 0.88), ("daily", 1.0)),
        "electronic": (
            ("电子产品", 1.0),
            ("electronics", 1.0),
            ("electronic", 1.0),
            ("电子", 0.90),
        ),
    }

    def classify(self, text: str) -> Optional[str]:
        category, _score, _debug = self.classify_texts([OCRText(text=text or "", score=1.0)])
        return category

    def classify_texts(self, texts: Sequence[OCRText]) -> Tuple[Optional[str], float, str]:
        category, score, _evidence, debug = self.classify_evidence(texts)
        return category, score, debug

    def classify_evidence(self, texts: Sequence[OCRText]) -> Tuple[Optional[str], float, str, str]:
        scores = {category: 0.0 for category in self.FEATURES}
        hits: Dict[str, List[str]] = {category: [] for category in self.FEATURES}
        for item in texts:
            text = self._normalize(item.text)
            if not text:
                continue
            confidence = max(0.0, min(1.0, float(item.score or 0.0)))
            for category, tokens in self.FEATURES.items():
                for token, strength in tokens:
                    if token in text:
                        value = strength * max(0.5, confidence)
                        if value > scores[category]:
                            scores[category] = value
                        hits[category].append(token)
                        break

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_category, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        debug = ",".join(
            "{}:{:.2f}[{}]".format(k, scores[k], "|".join(hits[k][:3]))
            for k in ("food", "daily", "electronic")
        )
        # A frame containing two complete, competing categories is ambiguous.
        if best_score <= 0.0 or second_score > 0.0:
            return None, best_score, "", debug
        evidence = hits[best_category][0] if hits[best_category] else ""
        return best_category, best_score, evidence, debug

    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").lower()
        text = text.replace("車間", "车间").replace("工間", "车间")
        text = text.replace("晶", "品").replace("吕", "品").replace("曰", "日")
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text)
        return text


def select_category_box(texts: Sequence[OCRText], category: Optional[str], classifier):
    """Pick the strongest OCR box that independently supports the chosen category."""
    if not category:
        return None
    matches = []
    for item in texts:
        if len(item.box) != 4:
            continue
        observed, score, _evidence, _debug = classifier.classify_evidence([item])
        if observed == category:
            matches.append((float(score), float(item.score or 0.0), item))
    if not matches:
        return None
    matches.sort(key=lambda value: (value[0], value[1]), reverse=True)
    return matches[0][2]


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


class CTCLabelDecoder:
    def __init__(self, keys_path: str, add_space: bool = True) -> None:
        if not os.path.isfile(keys_path):
            raise RuntimeError("PPOCR keys file missing: {}".format(keys_path))
        with open(keys_path, "r", encoding="utf-8") as fh:
            raw_lines = [line.rstrip("\n\r") for line in fh if line.rstrip("\n\r")]
        if len(raw_lines) == 1 and len(raw_lines[0]) > 256:
            raw_lines = list(raw_lines[0])
        self.characters = raw_lines + ([" "] if add_space else [])
        if not self.characters:
            raise RuntimeError("PPOCR keys file is empty: {}".format(keys_path))

    def decode(self, output) -> Tuple[str, float]:
        import numpy as np

        arr = np.asarray(output)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise RuntimeError("unexpected recognition output shape: {}".format(arr.shape))
        class_count = len(self.characters) + 1
        if arr.shape[1] == class_count:
            pass
        elif arr.shape[0] == class_count:
            arr = arr.T
        elif arr.shape[0] > arr.shape[1] and arr.shape[0] > 128:
            arr = arr.T
        indexes = np.argmax(arr, axis=1).astype("int32").tolist()
        scores = np.max(arr, axis=1).astype("float32").tolist()
        chars = []
        confs = []
        last_idx = None
        for idx, score in zip(indexes, scores):
            if idx == 0 or idx == last_idx:
                last_idx = idx
                continue
            char_pos = idx - 1
            if 0 <= char_pos < len(self.characters):
                chars.append(self.characters[char_pos])
                confs.append(float(score))
            last_idx = idx
        text = "".join(chars)
        confidence = float(sum(confs) / len(confs)) if confs else 0.0
        return text, confidence


class RknnRuntime:
    def __init__(self, model_path: str, logger=None) -> None:
        self.model_path = model_path
        self.logger = logger
        self.rknn = self._load()

    def _load(self):
        if not os.path.isfile(self.model_path):
            raise RuntimeError("RKNN model missing: {}".format(self.model_path))
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:
            raise RuntimeError("rknnlite.api.RKNNLite unavailable: {}".format(exc))
        repair_ros_logging()
        rknn = RKNNLite()
        ret = rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError("load_rknn failed for {}: {}".format(self.model_path, ret))
        core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", None)
        if core_mask is not None:
            ret = rknn.init_runtime(core_mask=core_mask)
        else:
            ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("init_runtime failed for {}: {}".format(self.model_path, ret))
        repair_ros_logging()
        return rknn

    def infer(self, image, data_format: str = "nhwc"):
        if self.rknn is None:
            raise RuntimeError("RKNN runtime has already been released")
        try:
            return self.rknn.inference(inputs=[image], data_format=[data_format])
        except TypeError:
            return self.rknn.inference(inputs=[image])

    def release(self) -> None:
        rknn = self.rknn
        self.rknn = None
        if rknn is None:
            return
        try:
            rknn.release()
        except Exception:
            pass


class PPOCRRknnRecognizer:
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        keys_path: str,
        mode: str,
        min_score: float,
        det_binary_thresh: float,
        det_box_thresh: float,
        det_input_size: int,
        rec_image_height: int,
        rec_image_width: int,
        rec_resize_mode: str,
        max_rec_crops: int,
        use_global_rec_candidates: bool,
        box_padding_x: float = 0.15,
        box_padding_y: float = 0.35,
        global_fallback_crops: int = 1,
        small_crop_retry: bool = True,
        small_crop_max_height: int = 20,
        target_category: str = "",
        logger=None,
    ) -> None:
        self.logger = logger
        self.mode = (mode or "ppocr_rknn_rec_only").strip().lower()
        if self.mode not in ("ppocr_rknn_rec_only", "ppocr_rknn_system"):
            self.mode = "ppocr_rknn_rec_only"
        self.min_score = float(min_score)
        self.det_binary_thresh = float(det_binary_thresh)
        self.det_box_thresh = float(det_box_thresh)
        self.det_input_size = int(det_input_size or 480)
        self.rec_h = int(rec_image_height or 48)
        self.rec_w = int(rec_image_width or 320)
        self.rec_resize_mode = (rec_resize_mode or "stretch").strip().lower()
        if self.rec_resize_mode not in ("stretch", "pad"):
            self.rec_resize_mode = "stretch"
        self.max_rec_crops = max(1, int(max_rec_crops or 6))
        self.use_global_rec_candidates = bool(use_global_rec_candidates)
        self.box_padding_x = max(0.0, float(box_padding_x))
        self.box_padding_y = max(0.0, float(box_padding_y))
        self.global_fallback_crops = max(0, int(global_fallback_crops))
        self.small_crop_retry = bool(small_crop_retry)
        self.small_crop_max_height = max(1, int(small_crop_max_height))
        self.target_category = normalize_ocr_target(target_category)
        self.classifier = FactorySignKeywordClassifier()
        self.decoder = CTCLabelDecoder(keys_path)
        self.rec = RknnRuntime(rec_model_path, logger=logger)
        self.det = None
        if self.mode == "ppocr_rknn_system":
            self.det = RknnRuntime(det_model_path, logger=logger)

    def recognize(self, image) -> RecognitionResult:
        started = time.time()
        texts: List[OCRText] = []
        det_ms = 0
        rec_ms = 0
        try:
            if self.mode == "ppocr_rknn_system" and self.det is not None:
                detections, det_ms = self._detect_boxes(image)
                candidates = self._rank_candidates(image, detections)[: self.max_rec_crops]
                if not candidates and self.use_global_rec_candidates:
                    candidates = self._fallback_candidates(image)
            else:
                h, w = image.shape[:2]
                boxes = [[[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]]]
                raw_crops = self._global_rec_crops(image) if self.use_global_rec_candidates else [(image, boxes[0])]
                candidates = [OCRCandidate(crop=crop, box=box, height=float(crop.shape[0])) for crop, box in raw_crops[: self.max_rec_crops]]
            for candidate in candidates:
                text, score, one_rec_ms = self._recognize_crop(candidate.crop)
                rec_ms += one_rec_ms
                if text and score >= self.min_score:
                    texts.append(OCRText(text=text, score=score, box=candidate.box))
                    category, _score, _evidence, _debug = self.classifier.classify_evidence(texts)
                    if category and (not self.target_category or category == self.target_category):
                        break
            if (
                self.small_crop_retry
                and candidates
                and candidates[0].height <= self.small_crop_max_height
                and not self._has_target_evidence(texts)
            ):
                retry_crop = self._enhance_small_crop(candidates[0].crop)
                text, score, one_rec_ms = self._recognize_crop(retry_crop)
                rec_ms += one_rec_ms
                if text and score >= self.min_score:
                    texts.append(OCRText(text=text, score=score, box=candidates[0].box))
            raw_text = " ".join(item.text for item in texts)
            return RecognitionResult(
                raw_text=raw_text,
                texts=texts,
                confidence=max([item.score for item in texts] or [0.0]),
                elapsed_ms=int((time.time() - started) * 1000),
                det_ms=det_ms,
                rec_ms=rec_ms,
                candidate_count=len(candidates),
            )
        except Exception as exc:
            return RecognitionResult(
                error=str(exc),
                elapsed_ms=int((time.time() - started) * 1000),
                det_ms=det_ms,
                rec_ms=rec_ms,
            )

    def _has_target_evidence(self, texts: Sequence[OCRText]) -> bool:
        category, _score, _evidence, _debug = self.classifier.classify_evidence(texts)
        return bool(category and (not self.target_category or category == self.target_category))

    def _fallback_candidates(self, image) -> List[OCRCandidate]:
        if self.global_fallback_crops <= 0:
            return []
        return [
            OCRCandidate(crop=crop, box=box, height=float(crop.shape[0]))
            for crop, box in self._global_rec_crops(image)[: self.global_fallback_crops]
        ]

    @staticmethod
    def _enhance_small_crop(crop):
        import cv2

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blurred = cv2.GaussianBlur(clahe, (0, 0), 1.0)
        sharpened = cv2.addWeighted(clahe, 1.5, blurred, -0.5, 0)
        return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    def _recognize_crop(self, crop) -> Tuple[str, float, int]:
        import cv2
        import numpy as np

        if crop is None or crop.size == 0:
            return "", 0.0, 0
        image = self._resize_rec_image(crop)
        image = np.expand_dims(image, axis=0).astype("float32") / 255.0
        started = time.time()
        outputs = self.rec.infer(image, data_format="nhwc")
        elapsed_ms = int((time.time() - started) * 1000)
        if not outputs:
            raise RuntimeError("empty recognition RKNN output")
        text, score = self.decoder.decode(outputs[0])
        return text, score, elapsed_ms

    def _resize_rec_image(self, crop):
        import cv2
        import numpy as np

        h, w = crop.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((self.rec_h, self.rec_w, 3), dtype=np.uint8)
        if self.rec_resize_mode == "stretch":
            return cv2.resize(crop, (self.rec_w, self.rec_h), interpolation=cv2.INTER_CUBIC)
        ratio = min(float(self.rec_w) / float(w), float(self.rec_h) / float(h))
        new_w = max(1, min(self.rec_w, int(round(w * ratio))))
        new_h = max(1, min(self.rec_h, int(round(h * ratio))))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((self.rec_h, self.rec_w, 3), 255, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        return canvas

    def _global_rec_crops(self, image):
        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return []
        crops = []
        for x0, y0, x1, y1 in self._center_crop_boxes(w, h):
            crop = image[y0:y1, x0:x1]
            box = [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]
            crops.append((crop, box))
        return crops

    @staticmethod
    def _center_crop_boxes(width: int, height: int) -> List[Tuple[int, int, int, int]]:
        if width <= 0 or height <= 0:
            return []
        specs = [
            (1.00, 1.00),
            (0.78, 0.78),
            (0.62, 0.62),
            (1.00, 0.62),
            (1.00, 0.42),
        ]
        boxes = []
        seen = set()
        for sx, sy in specs:
            crop_w = max(8, min(width, int(round(width * sx))))
            crop_h = max(8, min(height, int(round(height * sy))))
            x0 = max(0, (width - crop_w) // 2)
            y0 = max(0, (height - crop_h) // 2)
            x1 = min(width, x0 + crop_w)
            y1 = min(height, y0 + crop_h)
            key = (x0, y0, x1, y1)
            if key not in seen and x1 > x0 and y1 > y0:
                boxes.append(key)
                seen.add(key)
        return boxes

    def _detect_boxes(self, image):
        import cv2
        import numpy as np

        h, w = image.shape[:2]
        resized = cv2.resize(image, (self.det_input_size, self.det_input_size), interpolation=cv2.INTER_LINEAR)
        tensor = np.expand_dims(resized, axis=0).astype("uint8")
        started = time.time()
        outputs = self.det.infer(tensor, data_format="nhwc")
        det_ms = int((time.time() - started) * 1000)
        if not outputs:
            return [], det_ms
        pred = np.asarray(outputs[0])
        pred = np.squeeze(pred)
        if pred.ndim == 3:
            pred = pred[0] if pred.shape[0] <= 4 else pred[:, :, 0]
        pred = pred.astype("float32")
        if pred.max() > 1.5:
            pred = pred / 255.0
        bitmap = (pred > self.det_binary_thresh).astype("uint8") * 255
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < 16:
                continue
            mask = np.zeros(bitmap.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            score = float(cv2.mean(pred, mask=mask)[0])
            if score < self.det_box_thresh:
                continue
            rect = cv2.minAreaRect(contour)
            points = cv2.boxPoints(rect)
            points[:, 0] *= float(w) / float(self.det_input_size)
            points[:, 1] *= float(h) / float(self.det_input_size)
            box = self._order_points_clockwise(points).tolist()
            detections.append((box, score))
        return detections, det_ms

    @staticmethod
    def _order_points_clockwise(points):
        import numpy as np

        pts = np.asarray(points, dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        return np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")

    @staticmethod
    def _expanded_box(box, image_width: int, image_height: int, padding_x: float, padding_y: float):
        import numpy as np

        pts = np.asarray(box, dtype="float32")
        span_x = float(np.ptp(pts[:, 0]))
        span_y = float(np.ptp(pts[:, 1]))
        x0 = max(0.0, float(pts[:, 0].min()) - span_x * padding_x)
        x1 = min(float(image_width - 1), float(pts[:, 0].max()) + span_x * padding_x)
        y0 = max(0.0, float(pts[:, 1].min()) - span_y * padding_y)
        y1 = min(float(image_height - 1), float(pts[:, 1].max()) + span_y * padding_y)
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    @staticmethod
    def _candidate_scores(image, box, det_score: float) -> Tuple[float, float, float, float]:
        import cv2
        import numpy as np

        h, w = image.shape[:2]
        pts = np.asarray(box, dtype="float32")
        cx, cy = pts.mean(axis=0)
        dx = (float(cx) - 0.5 * w) / max(1.0, 0.5 * w)
        dy = (float(cy) - 0.5 * h) / max(1.0, 0.5 * h)
        center_score = max(0.0, 1.0 - min(1.0, (dx * dx + dy * dy) ** 0.5))
        x0, x1 = max(0, int(pts[:, 0].min())), min(w, int(pts[:, 0].max()) + 1)
        y0, y1 = max(0, int(pts[:, 1].min())), min(h, int(pts[:, 1].max()) + 1)
        sign_score = 0.0
        if x1 > x0 and y1 > y0:
            patch = image[y0:y1, x0:x1]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
            white_ratio = float(np.mean(gray >= 180))
            dark_ratio = float(np.mean(gray <= 90))
            sign_score = min(1.0, 0.7 * white_ratio + 0.3 * min(1.0, dark_ratio / 0.15))
        rank_score = 0.65 * float(det_score) + 0.20 * center_score + 0.15 * sign_score
        return rank_score, center_score, sign_score, float(max(0.0, np.ptp(pts[:, 1])))

    def _rank_candidates(self, image, detections) -> List[OCRCandidate]:
        candidates = []
        h, w = image.shape[:2]
        for box, det_score in detections:
            rank_score, center_score, sign_score, original_height = self._candidate_scores(image, box, det_score)
            expanded = self._expanded_box(box, w, h, self.box_padding_x, self.box_padding_y)
            crop_items = self._crop_boxes(image, [expanded])
            if not crop_items:
                continue
            crop, mapped_box = crop_items[0]
            candidates.append(OCRCandidate(
                crop=crop,
                box=mapped_box,
                det_score=float(det_score),
                center_score=center_score,
                sign_score=sign_score,
                rank_score=rank_score,
                height=original_height,
            ))
        candidates.sort(key=lambda item: item.rank_score, reverse=True)
        return candidates

    def _crop_boxes(self, image, boxes: Sequence[Sequence[Sequence[float]]]):
        import cv2
        import numpy as np

        crops = []
        for box in boxes:
            pts = self._order_points_clockwise(np.asarray(box, dtype="float32"))
            w1 = np.linalg.norm(pts[0] - pts[1])
            w2 = np.linalg.norm(pts[2] - pts[3])
            h1 = np.linalg.norm(pts[0] - pts[3])
            h2 = np.linalg.norm(pts[1] - pts[2])
            dst_w = max(8, int(round(max(w1, w2))))
            dst_h = max(8, int(round(max(h1, h2))))
            dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")
            matrix = cv2.getPerspectiveTransform(pts, dst)
            crop = cv2.warpPerspective(image, matrix, (dst_w, dst_h), flags=cv2.INTER_CUBIC)
            if dst_h > dst_w * 1.5:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            crops.append((crop, pts.tolist()))
        return crops

    def release(self) -> None:
        if getattr(self, "_released", False):
            return
        self._released = True
        self.rec.release()
        if self.det is not None:
            self.det.release()


class FactorySignPPOCRRknnNode:
    def __init__(self) -> None:
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        repair_ros_logging()
        self.rospy = rospy
        self.Image = Image
        self.String = String
        self.bridge = CvBridge()

        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.flip_image = ros_bool(rospy.get_param("~flip", True), True)
        self.inference_rate = float(rospy.get_param("~inference_rate", 5.0))
        self.roi_scale = float(rospy.get_param("~roi_scale", 0.8))
        self.view_scales = parse_view_scales(rospy.get_param("~view_scales", "0.55,0.75,0.95"), self.roi_scale)
        self.view_index = 0
        self.resize_scale = float(rospy.get_param("~resize_scale", 1.0))
        self.use_sharpen = ros_bool(rospy.get_param("~use_sharpen", False), False)
        self.use_adaptive_threshold = ros_bool(rospy.get_param("~use_adaptive_threshold", False), False)
        self.debug_show_image = ros_bool(rospy.get_param("~debug_show_image", False), False)
        self.publish_debug_image = ros_bool(rospy.get_param("~publish_debug_image", True), True)
        self.debug_publish_rate = float(rospy.get_param("~debug_publish_rate", 8.0))
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "/factory_sign_ppocr_rknn_test/debug_image")
        self.preprocess_topic = rospy.get_param("~debug_preprocess_topic", "/factory_sign_ppocr_rknn_test/preprocess_image")

        self.cooldown_sec = float(rospy.get_param("~cooldown_sec", 5.0))
        self.classifier = FactorySignKeywordClassifier()

        self.recognizer = self._create_recognizer()

        self.speech_mode = rospy.get_param("~speech_mode", "service").strip().lower()
        self.speech_service = rospy.get_param("~speech_service", "/competition_speech/announce")
        self.speech_timeout = float(rospy.get_param("~speech_service_timeout_sec", 0.5))
        self.speech_topic = rospy.get_param("~speech_topic", "/speak")
        self.fallback_to_topic = ros_bool(rospy.get_param("~fallback_to_speech_topic", True), True)
        self.speech_wait = ros_bool(rospy.get_param("~speech_wait", False), False)
        self.enable_speech = ros_bool(rospy.get_param("~enable_speech", True), True)
        self.result_topic = rospy.get_param(
            "~result_topic", "/factory_sign_ppocr_rknn_test/result")

        self.latest_image = None
        self.latest_image_seq = 0
        self.processed_image_seq = -1
        self.last_result = RecognitionResult()
        self.last_confirmed = None
        self.last_spoken_category = None
        self.last_spoken_at_by_category: Dict[str, float] = {}
        self.last_roi_box = (0, 0, 0, 0)
        self.last_ocr_scale_factor = 1.0
        self.last_debug_publish_at = 0.0
        self.shutdown_requested = False
        self.resources_released = False

        self.speak_pub = rospy.Publisher(self.speech_topic, String, queue_size=1)
        self.result_pub = rospy.Publisher(
            self.result_topic, String, queue_size=10, latch=True)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=1)
        self.preprocess_pub = rospy.Publisher(self.preprocess_topic, Image, queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "factory_sign_ppocr_rknn_node ready: image=%s mode=%s debug=%s preprocess=%s speech_service=%s speech_topic=%s",
            self.image_topic,
            self.recognizer.mode,
            self.debug_image_topic,
            self.preprocess_topic,
            self.speech_service,
            self.speech_topic,
        )

    def _create_recognizer(self) -> PPOCRRknnRecognizer:
        det_path = self._resolve_asset_path("~det_model_path", "ppocrv4_det.rknn")
        rec_path = self._resolve_asset_path("~rec_model_path", "ppocrv4_rec.rknn")
        keys_path = self._resolve_asset_path("~keys_path", "ppocr_keys_v1.txt")
        mode = self.rospy.get_param("~recognition_mode", "ppocr_rknn_rec_only")
        if (
            str(mode).strip().lower() == "ppocr_rknn_system"
            and not os.path.isfile(det_path)
            and ros_bool(self.rospy.get_param("~rec_only_if_det_missing", True), True)
        ):
            self.rospy.logwarn("det RKNN model missing, falling back to ppocr_rknn_rec_only: %s", det_path)
            mode = "ppocr_rknn_rec_only"
        return PPOCRRknnRecognizer(
            det_model_path=det_path,
            rec_model_path=rec_path,
            keys_path=keys_path,
            mode=mode,
            min_score=float(self.rospy.get_param("~ocr_min_score", 0.45)),
            det_binary_thresh=float(self.rospy.get_param("~det_binary_thresh", 0.30)),
            det_box_thresh=float(self.rospy.get_param("~det_box_thresh", 0.55)),
            det_input_size=int(self.rospy.get_param("~det_input_size", 480)),
            rec_image_height=int(self.rospy.get_param("~rec_image_height", 48)),
            rec_image_width=int(self.rospy.get_param("~rec_image_width", 320)),
            rec_resize_mode=self.rospy.get_param("~rec_resize_mode", "stretch"),
            max_rec_crops=int(self.rospy.get_param("~max_rec_crops", 3)),
            use_global_rec_candidates=ros_bool(self.rospy.get_param("~use_global_rec_candidates", True), True),
            box_padding_x=float(self.rospy.get_param("~box_padding_x", 0.15)),
            box_padding_y=float(self.rospy.get_param("~box_padding_y", 0.35)),
            global_fallback_crops=int(self.rospy.get_param("~global_fallback_crops", 1)),
            small_crop_retry=ros_bool(self.rospy.get_param("~small_crop_retry", True), True),
            small_crop_max_height=int(self.rospy.get_param("~small_crop_max_height", 20)),
            target_category=self.rospy.get_param("~target_category", ""),
            logger=self.rospy,
        )

    @staticmethod
    def _package_dir() -> str:
        try:
            import rospkg

            return rospkg.RosPack().get_path("factory_sign_ppocr_rknn_test")
        except Exception:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _resolve_asset_path(self, param_name: str, filename: str) -> str:
        value = self.rospy.get_param(param_name, "")
        if value:
            return os.path.abspath(os.path.expanduser(os.path.expandvars(value)))
        return os.path.join(self._package_dir(), "models", filename)

    def _image_cb(self, msg) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = maybe_flip_frame(frame, self.flip_image)
            self.latest_image_seq += 1
            self._publish_live_debug(self.latest_image)
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "cv_bridge conversion failed: %s", exc)

    def run(self) -> None:
        rate = self.rospy.Rate(self.inference_rate)
        while not self.rospy.is_shutdown():
            if self.latest_image is not None and self.latest_image_seq != self.processed_image_seq:
                self.processed_image_seq = self.latest_image_seq
                self._process_once(self.latest_image.copy())
            rate.sleep()

    def _process_once(self, frame) -> None:
        view_scale = self.view_scales[self.view_index % len(self.view_scales)]
        self.view_index += 1
        ocr_image, debug_preprocess = self._make_ocr_image(frame, view_scale)
        result = self.recognizer.recognize(ocr_image)
        result.category, result.category_score, result.evidence, match_debug = self.classifier.classify_evidence(result.texts)
        result.view_scale = view_scale
        result.match_debug = "score={:.2f} evidence={} {}".format(result.category_score, result.evidence or "-", match_debug)
        confirmed = result.category
        self.last_result = result
        self.last_confirmed = confirmed
        spoken = self._maybe_speak(confirmed) if confirmed and self.enable_speech else False
        target_item = select_category_box(result.texts, confirmed, self.classifier)
        target_box = []
        target_center_x = None
        target_center_y = None
        if target_item is not None:
            target_box = map_box_to_frame(
                target_item.box, self.last_roi_box, self.last_ocr_scale_factor)
            target_center_x = sum(point[0] for point in target_box) / len(target_box)
            target_center_y = sum(point[1] for point in target_box) / len(target_box)
        frame_height, frame_width = frame.shape[:2]
        self.result_pub.publish(self.String(data=json.dumps({
            "category": confirmed or "",
            "workshop": CATEGORY_NAMES.get(confirmed, ""),
            "confidence": float(result.confidence),
            "category_score": float(result.category_score),
            "evidence": result.evidence,
            "view_scale": float(result.view_scale),
            "candidate_count": int(result.candidate_count),
            "raw_text": result.raw_text,
            "match_debug": result.match_debug,
            "target_bbox": target_box,
            "target_center_x": target_center_x,
            "target_center_y": target_center_y,
            "image_width": int(frame_width),
            "image_height": int(frame_height),
            "error": result.error,
            "stamp": time.time(),
        }, ensure_ascii=False)))
        self.rospy.loginfo(
            "factory_sign_ppocr_rknn: text=%r category=%s conf=%.3f evidence=%s view=%.2f candidates=%d match=%s texts=%s decision=%s spoken=%s elapsed_ms=%d det_ms=%d rec_ms=%d error=%s",
            result.raw_text,
            result.category,
            result.confidence,
            result.evidence,
            result.view_scale,
            result.candidate_count,
            result.match_debug,
            self._texts_debug(result.texts),
            confirmed,
            spoken,
            result.elapsed_ms,
            result.det_ms,
            result.rec_ms,
            result.error,
        )
        self._publish_debug_images(frame, debug_preprocess, spoken)
        if self.debug_show_image:
            self._show_debug(debug_preprocess)

    def _make_ocr_image(self, frame, view_scale=None):
        import cv2
        import numpy as np

        h, w = frame.shape[:2]
        scale = min(max(float(self.roi_scale if view_scale is None else view_scale), 0.1), 1.0)
        roi_w = int(w * scale)
        roi_h = int(h * scale)
        x0 = max(0, (w - roi_w) // 2)
        y0 = max(0, (h - roi_h) // 2)
        self.last_roi_box = (x0, y0, x0 + roi_w, y0 + roi_h)
        roi = frame[y0 : y0 + roi_h, x0 : x0 + roi_w]
        self.last_ocr_scale_factor = 1.0
        if self.resize_scale and self.resize_scale > 1.0:
            roi = cv2.resize(roi, None, fx=self.resize_scale, fy=self.resize_scale, interpolation=cv2.INTER_CUBIC)
            self.last_ocr_scale_factor = self.resize_scale
        if self.use_sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            roi = cv2.filter2D(roi, -1, kernel)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.use_adaptive_threshold:
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray
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
        for item in self.last_result.texts:
            if len(item.box) == 4:
                pts = [
                    (int(p[0]), int(p[1]))
                    for p in map_box_to_frame(item.box, self.last_roi_box, self.last_ocr_scale_factor)
                ]
                for idx in range(4):
                    cv2.line(out, pts[idx], pts[(idx + 1) % 4], (0, 255, 0), 2)
        lines = [
            "source=ppocr_rknn_{}".format(self.recognizer.mode),
            "category={} confirmed={} conf={:.2f}".format(self.last_result.category, self.last_confirmed, self.last_result.confidence),
            "view={:.2f} candidates={} evidence={}".format(self.last_result.view_scale, self.last_result.candidate_count, self._ascii_preview(self.last_result.evidence)),
            "time={}ms det={} rec={} spoken={}".format(self.last_result.elapsed_ms, self.last_result.det_ms, self.last_result.rec_ms, spoken),
        ]
        if self.last_result.raw_text:
            lines.append("text={}".format(self._ascii_preview(self.last_result.raw_text)))
        if self.last_result.match_debug:
            lines.append("match={}".format(self._ascii_preview(self.last_result.match_debug)))
        if self.last_result.error:
            lines.append("err={}".format(self._ascii_preview(self.last_result.error)))
        y = 24
        for line in lines:
            cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            y += 24
        return out

    @staticmethod
    def _ascii_preview(text: str) -> str:
        return (text or "").encode("ascii", "replace").decode("ascii")[:110]

    @staticmethod
    def _texts_debug(texts: Sequence[OCRText]) -> List[str]:
        out = []
        for item in texts[:8]:
            if len(item.box) == 4:
                width = max(p[0] for p in item.box) - min(p[0] for p in item.box)
                height = max(p[1] for p in item.box) - min(p[1] for p in item.box)
                out.append("{:.2f}:{}({:.0f}x{:.0f})".format(item.score, item.text, width, height))
            else:
                out.append("{:.2f}:{}".format(item.score, item.text))
        return out

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

            cv2.imshow("factory_sign_ppocr_rknn_preprocessed", image)
            cv2.waitKey(1)
        except Exception as exc:
            self.rospy.logwarn_throttle(2.0, "debug_show_image failed: %s", exc)

    def _on_shutdown(self) -> None:
        self.shutdown_requested = True
        try:
            self.image_sub.unregister()
        except Exception:
            pass

    def _release_resources(self) -> None:
        if self.resources_released:
            return
        self.resources_released = True
        try:
            self.recognizer.release()
        except Exception:
            pass
        try:
            import cv2

            cv2.destroyWindow("factory_sign_ppocr_rknn_preprocessed")
        except Exception:
            pass


def main() -> None:
    import rospy

    repair_ros_logging()
    rospy.init_node("factory_sign_ppocr_rknn_node")
    try:
        node = FactorySignPPOCRRknnNode()
    except Exception as exc:
        rospy.logfatal("factory_sign_ppocr_rknn_node failed to initialize: %s", exc)
        raise
    node.run()


if __name__ == "__main__":
    main()
