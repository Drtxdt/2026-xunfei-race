#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PKG_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from factory_sign_ocr_node import FactorySignClassifier, FactorySignRecognizer, RecognitionResult, VoteWindow, _maybe_flip_frame, _repair_ros_logging


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


def test_recognizer_falls_back_to_ocr_text_when_rknn_has_no_category():
    recognizer = FactorySignRecognizer(
        classifier=FactorySignClassifier(),
        rknn_backend=FakeRknnBackend(RecognitionResult(category=None, confidence=0.0, source="rknn")),
        ocr_backend=FakeOcrBackend("daily goods"),
        mode="auto",
    )

    result = recognizer.recognize(object())

    assert result.category == "daily"
    assert result.raw_text == "daily goods"
    assert result.source == "ocr"


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
