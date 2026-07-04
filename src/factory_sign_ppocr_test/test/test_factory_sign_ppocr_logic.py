#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys


PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(PKG_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from factory_sign_ppocr_node import FactorySignKeywordClassifier, VoteWindow


def test_keyword_classifier_maps_factory_signs():
    classifier = FactorySignKeywordClassifier()

    assert classifier.classify("食品加工车间") == "food"
    assert classifier.classify("食 品 加 工") == "food"
    assert classifier.classify("FOOD") == "food"
    assert classifier.classify("日用品加工车间") == "daily"
    assert classifier.classify("日用 产品") == "daily"
    assert classifier.classify("daily goods") == "daily"
    assert classifier.classify("电子产品生产车间") == "electronic"
    assert classifier.classify("电 子") == "electronic"
    assert classifier.classify("Electronic") == "electronic"
    assert classifier.classify("未知车间") is None


def test_vote_window_requires_two_hits_in_five_frames():
    vote = VoteWindow(size=5, min_count=2)

    assert vote.push(None) is None
    assert vote.push("daily") is None
    assert vote.push("food") is None
    assert vote.push("daily") == "daily"
    assert vote.snapshot() == [None, "daily", "food", "daily"]
