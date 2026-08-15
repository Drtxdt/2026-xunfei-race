#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS-independent preprocessing, softmax and temporal consensus."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from PIL import Image


CLASS_NAMES = (
    "green_left",
    "green_right",
    "green_straight",
    "red_light",
    "background",
)
BACKGROUND_CLASS = "background"
YELLOW_CLASS = "yellow_light"


class BinarySignalConsensus(object):
    """Debounce a color-only signal without changing the RKNN class layout."""

    def __init__(self, confirm_frames=2, release_frames=1):
        self.confirm_frames = max(1, int(confirm_frames))
        self.release_frames = max(1, int(release_frames))
        self.reset()

    def reset(self):
        self.hits = 0
        self.misses = 0
        self.active = False

    def update(self, detected):
        if bool(detected):
            self.hits += 1
            self.misses = 0
            if self.hits >= self.confirm_frames:
                self.active = True
        else:
            self.hits = 0
            self.misses += 1
            if self.misses >= self.release_frames:
                self.active = False
        return self.active


def detect_yellow_signal(
    bgr_frame,
    roi_bbox,
    x_min_ratio=0.25,
    x_max_ratio=0.75,
    h_min=16,
    h_max=40,
    s_min=110,
    v_min=140,
    min_area_ratio=0.0008,
    max_area_ratio=0.12,
):
    """Detect a compact yellow lamp inside the classifier crop."""
    if bgr_frame is None or not isinstance(bgr_frame, np.ndarray):
        return False, 0.0, None
    height, width = bgr_frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in roi_bbox)
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))
    span = x2 - x1
    sx1 = x1 + int(round(span * float(x_min_ratio)))
    sx2 = x1 + int(round(span * float(x_max_ratio)))
    sx1 = max(x1, min(x2 - 1, sx1))
    sx2 = max(sx1 + 1, min(x2, sx2))
    roi = bgr_frame[y1:y2, sx1:sx2]
    if roi.size == 0:
        return False, 0.0, None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([int(h_min), int(s_min), int(v_min)], dtype=np.uint8),
        np.array([int(h_max), 255, 255], dtype=np.uint8),
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    search_area = float(max(1, roi.shape[0] * roi.shape[1]))
    best = None
    for index in range(1, count):
        left, top, comp_width, comp_height, area = (
            int(value) for value in stats[index]
        )
        ratio = float(area) / search_area
        if not float(min_area_ratio) <= ratio <= float(max_area_ratio):
            continue
        if comp_width < 3 or comp_height < 3:
            continue
        aspect = float(comp_width) / float(comp_height)
        fill = float(area) / float(comp_width * comp_height)
        if not 0.35 <= aspect <= 2.8 or fill < 0.30:
            continue
        if best is None or area > best[0]:
            best = (area, ratio, left, top, comp_width, comp_height)
    if best is None:
        return False, 0.0, None
    _, ratio, left, top, comp_width, comp_height = best
    confidence = min(1.0, ratio / max(float(min_area_ratio), 1e-9))
    bbox = [sx1 + left, y1 + top, sx1 + left + comp_width, y1 + top + comp_height]
    return True, confidence, bbox


def yellow_override_state(base_state, active, confidence=0.0):
    """Return a yellow consensus state while preserving diagnostics shape."""
    if not active:
        return base_state
    state = dict(base_state)
    state.update({
        "active": True,
        "class_name": YELLOW_CLASS,
        "confidence": float(confidence),
        "reason": "yellow_hsv_override",
    })
    return state


def stable_softmax(logits, class_count=5):
    values = np.asarray(logits)
    if values.size != class_count:
        raise ValueError("expected {} logits, got shape {}".format(class_count, values.shape))
    values = values.reshape(-1).astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("classifier output contains non-finite values")
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    denominator = float(np.sum(exp_values))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("invalid softmax denominator")
    probabilities = exp_values / denominator
    return probabilities.astype(np.float32)


def preprocess_frame(
    bgr_frame,
    flip=True,
    crop_top=0.18,
    crop_bottom=0.72,
    input_width=320,
    input_height=160,
):
    if bgr_frame is None or not isinstance(bgr_frame, np.ndarray):
        raise ValueError("frame is empty")
    if bgr_frame.ndim != 3 or bgr_frame.shape[2] != 3:
        raise ValueError("expected HxWx3 BGR frame, got {}".format(bgr_frame.shape))
    height, width = bgr_frame.shape[:2]
    if height < 2 or width < 2 or not 0.0 <= crop_top < crop_bottom <= 1.0:
        raise ValueError("invalid frame or crop bounds")
    corrected = cv2.flip(bgr_frame, 1) if flip else bgr_frame.copy()
    y1 = max(0, min(height - 1, int(round(height * crop_top))))
    y2 = max(y1 + 1, min(height, int(round(height * crop_bottom))))
    roi_bgr = corrected[y1:y2, 0:width]
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    # Match torchvision/Pillow evaluation and RKNN calibration exactly. OpenCV
    # INTER_LINEAR differs enough on the small LED arrows to reduce accuracy.
    rgb = np.asarray(
        Image.fromarray(roi_rgb).resize(
            (int(input_width), int(input_height)), resample=Image.BILINEAR
        ),
        dtype=np.uint8,
    )
    batch = np.ascontiguousarray(rgb[np.newaxis, ...], dtype=np.uint8)
    return corrected, batch, [0, y1, width, y2]


class ScoreConsensus(object):
    """Five-frame probability fusion with class-specific confirmation."""

    def __init__(
        self,
        class_names=CLASS_NAMES,
        window_size=5,
        min_valid_samples=3,
        confidence_threshold=0.55,
        margin_threshold=0.12,
        red_confirm_frames=2,
        green_confirm_frames=3,
        release_frames=3,
    ):
        self.class_names = tuple(class_names)
        if len(self.class_names) != 5 or BACKGROUND_CLASS not in self.class_names:
            raise ValueError("class_names must contain the fixed five traffic classes")
        self.background_index = self.class_names.index(BACKGROUND_CLASS)
        self.scores = deque(maxlen=max(1, int(window_size)))
        self.min_valid_samples = max(1, int(min_valid_samples))
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)
        self.red_confirm_frames = max(1, int(red_confirm_frames))
        self.green_confirm_frames = max(1, int(green_confirm_frames))
        self.release_frames = max(1, int(release_frames))
        self.active_class = None
        self.active_confidence = 0.0
        self.candidate_class = None
        self.candidate_hits = 0
        self.invalid_hits = 0

    def reset(self):
        self.scores.clear()
        self.active_class = None
        self.active_confidence = 0.0
        self.candidate_class = None
        self.candidate_hits = 0
        self.invalid_hits = 0

    @staticmethod
    def _top_two(probabilities):
        order = np.argsort(probabilities)[::-1]
        top1, top2 = int(order[0]), int(order[1])
        return top1, float(probabilities[top1]), float(probabilities[top1] - probabilities[top2])

    def _invalid(self, reason, frame_top1=None, frame_confidence=0.0, frame_margin=0.0):
        self.invalid_hits += 1
        self.candidate_class = None
        self.candidate_hits = 0
        if self.invalid_hits >= self.release_frames:
            self.scores.clear()
            self.active_class = None
            self.active_confidence = 0.0
        return self.snapshot(reason, frame_top1, frame_confidence, frame_margin)

    def update(self, probabilities):
        values = np.asarray(probabilities, dtype=np.float32).reshape(-1)
        if values.size != len(self.class_names) or not np.all(np.isfinite(values)):
            return self._invalid("invalid_output")
        top1, confidence, margin = self._top_two(values)
        if top1 == self.background_index:
            return self._invalid("background", self.class_names[top1], confidence, margin)
        if confidence < self.confidence_threshold:
            return self._invalid("low_confidence", self.class_names[top1], confidence, margin)
        if margin < self.margin_threshold:
            return self._invalid("low_margin", self.class_names[top1], confidence, margin)

        self.invalid_hits = 0
        self.scores.append(values.copy())
        if len(self.scores) < self.min_valid_samples:
            return self.snapshot("warming_up", self.class_names[top1], confidence, margin)

        mean_scores = np.mean(np.stack(tuple(self.scores), axis=0), axis=0)
        mean_top1, mean_confidence, mean_margin = self._top_two(mean_scores)
        mean_class = self.class_names[mean_top1]
        if mean_top1 == self.background_index:
            return self._invalid("mean_background", self.class_names[top1], confidence, margin)
        if mean_confidence < self.confidence_threshold or mean_margin < self.margin_threshold:
            return self._invalid("mean_rejected", self.class_names[top1], confidence, margin)

        if self.candidate_class == mean_class:
            self.candidate_hits += 1
        else:
            self.candidate_class = mean_class
            self.candidate_hits = 1
        required = self.red_confirm_frames if mean_class == "red_light" else self.green_confirm_frames
        if self.candidate_hits >= required:
            self.active_class = mean_class
            self.active_confidence = mean_confidence
        return self.snapshot("accepted", self.class_names[top1], confidence, margin)

    def snapshot(self, reason="idle", frame_top1=None, frame_confidence=0.0, frame_margin=0.0):
        if self.scores:
            average = np.mean(np.stack(tuple(self.scores), axis=0), axis=0)
        else:
            average = np.zeros(len(self.class_names), dtype=np.float32)
        return {
            "active": self.active_class is not None,
            "class_name": self.active_class or "",
            "confidence": float(self.active_confidence),
            "reason": reason,
            "valid_samples": len(self.scores),
            "candidate_class": self.candidate_class or "",
            "candidate_hits": int(self.candidate_hits),
            "frame_top1": frame_top1 or "",
            "frame_confidence": float(frame_confidence),
            "frame_margin": float(frame_margin),
            "average_scores": [float(value) for value in average],
        }


def make_detection_payload(
    stamp,
    class_names,
    probabilities,
    roi_bbox,
    consensus,
    inference_ms,
    model_quantization,
):
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    order = np.argsort(probabilities)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    raw = {
        "class_id": top1,
        "class_name": class_names[top1],
        "confidence": float(probabilities[top1]),
        "margin": float(probabilities[top1] - probabilities[top2]),
        "bbox": [int(value) for value in roi_bbox],
    }
    return {
        "header": {"stamp": float(stamp)},
        "raw_detections": [raw],
        "consensus": {
            "active": bool(consensus.get("active", False)),
            "class_name": consensus.get("class_name") or None,
            "confidence": float(consensus.get("confidence", 0.0)),
        },
        "diagnostics": {
            "probabilities": {
                name: float(probabilities[index]) for index, name in enumerate(class_names)
            },
            "margin": raw["margin"],
            "background": raw["class_name"] == BACKGROUND_CLASS,
            "roi": raw["bbox"],
            "inference_ms": float(inference_ms),
            "model_quantization": str(model_quantization),
            "consensus_reason": consensus.get("reason", ""),
            "valid_samples": int(consensus.get("valid_samples", 0)),
        },
    }
