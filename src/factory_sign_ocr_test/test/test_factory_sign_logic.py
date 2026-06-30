#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PKG_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from factory_sign_ocr_node import FactorySignClassifier, FactorySignRecognizer, RecognitionResult, ScoreVoteWindow, VoteWindow, _maybe_flip_frame, _repair_ros_logging


def test_classifier_matches_requested_keywords():
    classifier = FactorySignClassifier()

    assert classifier.classify("食品加工") == "food"
    assert classifier.classify("食 品") == "food"
    assert classifier.classify("FOOD workshop") == "food"
    assert classifier.classify("日用品加工车间") == "daily"
    assert classifier.classify("daily goods") == "daily"
    assert classifier.classify("电子产品生产车间") == "electronic"
    assert classifier.classify("电 子") == "electronic"
    assert classifier.classify("Electronic") == "electronic"
    assert classifier.classify("其他文字") is None


def test_vote_window_confirms_two_hits_in_five_frames():
    vote = VoteWindow(size=5, min_count=2)

    assert vote.push(None) is None
    assert vote.push("food") is None
    assert vote.push("daily") is None
    assert vote.push("food") == "food"

    assert vote.snapshot() == [None, "food", "daily", "food"]


def test_score_vote_window_confirms_average_daily_evidence():
    vote = ScoreVoteWindow(size=5, min_score=0.40, min_margin=0.04)

    assert vote.push({"daily": 0.39, "electronic": 0.36, "food": 0.25}) is None
    assert vote.push({"daily": 0.43, "electronic": 0.34, "food": 0.23}) == "daily"

    average = vote.average()
    assert average["daily"] > average["electronic"]
    assert "daily:" in vote.summary()


class FakeRknnBackend:
    def __init__(self, result):
        self.result = result

    def recognize(self, _frame):
        return self.result


class FakeOcrBackend:
    def __init__(self, text):
        self.text = text

    def recognize(self, _frame):
        return self.text


def test_recognizer_prefers_rknn_classifier_result():
    recognizer = FactorySignRecognizer(
        classifier=FactorySignClassifier(),
        rknn_backend=FakeRknnBackend(RecognitionResult(category="electronic", confidence=0.88, source="rknn")),
        ocr_backend=FakeOcrBackend("食品加工车间"),
        mode="auto",
    )

    result = recognizer.recognize(object())

    assert result.category == "electronic"
    assert result.source == "rknn"
    assert result.raw_text == ""


def test_recognizer_does_not_load_or_use_ocr_fallback():
    recognizer = FactorySignRecognizer(
        classifier=FactorySignClassifier(),
        rknn_backend=FakeRknnBackend(RecognitionResult(category=None, confidence=0.0, source="rknn")),
        ocr_backend=FakeOcrBackend("daily goods"),
        mode="auto",
    )

    result = recognizer.recognize(object())

    assert result.category is None
    assert result.raw_text == ""
    assert result.source == "none"
    assert "RKNN" in result.error


def test_repair_ros_logging_accepts_rknn_short_level_names(monkeypatch):
    import logging
    import sys
    import types

    roslogging = types.SimpleNamespace(
        _logging_to_rospy_names={
            "DEBUG": ("debug", ""),
            "INFO": ("info", ""),
            "WARNING": ("warn", ""),
            "ERROR": ("error", ""),
            "CRITICAL": ("fatal", ""),
        }
    )
    monkeypatch.setitem(sys.modules, "rosgraph", types.SimpleNamespace(roslogging=roslogging))
    monkeypatch.setitem(sys.modules, "rosgraph.roslogging", roslogging)
    logging.addLevelName(logging.INFO, "I")
    logging.addLevelName(logging.WARNING, "W")

    _repair_ros_logging()

    assert logging.getLevelName(logging.INFO) == "INFO"
    assert logging.getLevelName(logging.WARNING) == "WARNING"
    assert roslogging._logging_to_rospy_names["I"] == roslogging._logging_to_rospy_names["INFO"]
    assert roslogging._logging_to_rospy_names["W"] == roslogging._logging_to_rospy_names["WARNING"]


def test_maybe_flip_frame_flips_horizontally():
    import numpy as np

    frame = np.array([[[1], [2], [3]], [[4], [5], [6]]], dtype=np.uint8)

    flipped = _maybe_flip_frame(frame, True)
    unchanged = _maybe_flip_frame(frame, False)

    assert flipped[:, :, 0].tolist() == [[3, 2, 1], [6, 5, 4]]
    assert unchanged[:, :, 0].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert unchanged is frame


def test_rknn_classifier_preprocess_returns_batched_nhwc_uint8_by_default():
    import numpy as np
    from factory_sign_ocr_node import RknnFactorySignClassifierBackend

    backend = RknnFactorySignClassifierBackend.__new__(RknnFactorySignClassifierBackend)
    backend.input_size = 224
    backend.input_layout = "nhwc"
    backend.input_color = "bgr"
    backend.crop_mode = "square"
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 20
    frame[:, :, 2] = 30

    tensor = backend._preprocess_for_rknn(frame)

    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.uint8
    assert int(tensor[0, 0, 0, 0]) == 10


def test_rknn_classifier_preprocess_can_return_batched_nchw_uint8():
    import numpy as np
    from factory_sign_ocr_node import RknnFactorySignClassifierBackend

    backend = RknnFactorySignClassifierBackend.__new__(RknnFactorySignClassifierBackend)
    backend.input_size = 224
    backend.input_layout = "nchw"
    backend.input_color = "bgr"
    backend.crop_mode = "square"
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tensor = backend._preprocess_for_rknn(frame)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.uint8


def test_rknn_classifier_can_still_rgb_swap_for_ab_debug():
    import numpy as np
    from factory_sign_ocr_node import RknnFactorySignClassifierBackend

    backend = RknnFactorySignClassifierBackend.__new__(RknnFactorySignClassifierBackend)
    backend.input_size = 224
    backend.input_layout = "nhwc"
    backend.input_color = "rgb"
    backend.crop_mode = "full"
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = 10
    frame[:, :, 1] = 20
    frame[:, :, 2] = 30

    tensor = backend._preprocess_for_rknn(frame)

    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.uint8
    assert int(tensor[0, 0, 0, 0]) == 30


def test_rknn_classifier_logit_biases_apply_by_class_name():
    import numpy as np
    from factory_sign_ocr_node import RknnFactorySignClassifierBackend

    backend = RknnFactorySignClassifierBackend.__new__(RknnFactorySignClassifierBackend)
    backend.logit_biases = {
        "daily": 0.45,
        "electronic": -0.6,
        "food": 0.1,
    }

    logits = backend._apply_logit_bias(np.array([0.0, 1.0, 0.0], dtype=np.float32))

    assert abs(float(logits[0]) - 0.45) < 1e-6
    assert abs(float(logits[1]) - 0.4) < 1e-6
    assert abs(float(logits[2]) - 0.1) < 1e-6


def test_pick_best_detection_returns_top_category():
    from factory_sign_ocr_node import _pick_best_detection

    result = _pick_best_detection([
        {"class_name": "food", "confidence": 0.12},
        {"class_name": "daily", "confidence": 0.22},
    ], source="rknn_low")

    assert result.category == "daily"
    assert result.confidence == 0.22
    assert result.source == "rknn_low"
