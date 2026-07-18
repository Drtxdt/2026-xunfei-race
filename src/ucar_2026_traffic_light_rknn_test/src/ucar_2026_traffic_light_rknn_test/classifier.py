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
