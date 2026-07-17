import os
import sys

import numpy as np

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from factory_sign_ppocr_rknn_node import (
    CTCLabelDecoder,
    FactorySignKeywordClassifier,
    OCRText,
    PPOCRRknnRecognizer,
    VoteWindow,
    map_box_to_frame,
    parse_view_scales,
)


def test_keyword_classifier_maps_requested_categories():
    classifier = FactorySignKeywordClassifier()
    assert classifier.classify("食品加工车间") == "food"
    assert classifier.classify("日用品加工车间") == "daily"
    assert classifier.classify("电子产品加工车间") == "electronic"
    assert classifier.classify("FOOD") == "food"
    assert classifier.classify("品") is None
    assert classifier.classify("车间") is None


def test_keyword_classifier_uses_only_complete_category_evidence():
    classifier = FactorySignKeywordClassifier()

    assert classifier.classify("食品") == "food"
    assert classifier.classify("日用晶") == "daily"
    assert classifier.classify("用品") == "daily"
    assert classifier.classify("用") is None
    assert classifier.classify("电子产生产") == "electronic"
    for noise in ("食", "日", "电", "电品062093", "navigation", "ROS", "robot map"):
        assert classifier.classify(noise) is None


def test_keyword_classifier_scores_ocr_candidates_not_concatenated_noise():
    classifier = FactorySignKeywordClassifier()
    category, score, debug = classifier.classify_texts(
        [
            OCRText("品", 0.95),
            OCRText("车间", 0.95),
            OCRText("日用品", 0.72),
        ]
    )

    assert category == "daily"
    assert score > 0.0
    assert "daily" in debug


def test_keyword_classifier_merges_one_and_two_line_candidates():
    classifier = FactorySignKeywordClassifier()
    category, score, evidence, _debug = classifier.classify_evidence(
        [OCRText("电子产品", 0.82), OCRText("加工车间", 0.90)]
    )
    assert category == "electronic"
    assert score > 0.0
    assert evidence == "电子产品"


def test_competing_complete_categories_are_ambiguous():
    classifier = FactorySignKeywordClassifier()
    category, _score, evidence, _debug = classifier.classify_evidence(
        [OCRText("食品", 0.9), OCRText("日用品", 0.9)]
    )
    assert category is None
    assert evidence == ""


def test_vote_window_requires_min_count():
    vote = VoteWindow(size=5, min_count=2)
    assert vote.push("food") is None
    assert vote.push(None) is None
    assert vote.push("food") == "food"


def test_ctc_decoder_handles_time_major_and_class_major():
    keys = os.path.join(os.path.dirname(__file__), "ppocr_test_keys.txt")
    decoder = CTCLabelDecoder(keys, add_space=False)

    time_major = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.9, 0.05, 0.05],
            [0.1, 0.1, 0.8],
        ],
        dtype=np.float32,
    )
    assert decoder.decode(time_major)[0] == "食品"
    assert decoder.decode(time_major.T)[0] == "食品"


def test_recognizer_resize_defaults_to_ppocr_stretch_shape():
    recognizer = PPOCRRknnRecognizer.__new__(PPOCRRknnRecognizer)
    recognizer.rec_h = 48
    recognizer.rec_w = 320
    recognizer.rec_resize_mode = "stretch"
    crop = np.zeros((80, 180, 3), dtype=np.uint8)

    resized = recognizer._resize_rec_image(crop)

    assert resized.shape == (48, 320, 3)
    assert resized.dtype == np.uint8


def test_global_rec_crops_include_center_bands():
    recognizer = PPOCRRknnRecognizer.__new__(PPOCRRknnRecognizer)
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    crops = recognizer._global_rec_crops(image)

    assert len(crops) == 5
    assert crops[0][0].shape == (100, 200, 3)
    assert crops[1][0].shape == (78, 156, 3)
    assert crops[2][0].shape == (62, 124, 3)
    assert crops[4][0].shape == (42, 200, 3)


def test_center_crop_boxes_are_unique_and_centered():
    boxes = PPOCRRknnRecognizer._center_crop_boxes(200, 100)

    assert len(boxes) == len(set(boxes))
    assert boxes[0] == (0, 0, 200, 100)
    assert boxes[3] == (0, 19, 200, 81)


def test_view_scales_are_clamped_unique_and_keep_order():
    assert parse_view_scales("0.55, 0.75,0.95") == [0.55, 0.75, 0.95]
    assert parse_view_scales([0.55, 0.55, 2.0]) == [0.55, 1.0]


def test_detected_box_expansion_and_center_ranking():
    box = [[80.0, 45.0], [120.0, 45.0], [120.0, 55.0], [80.0, 55.0]]
    expanded = PPOCRRknnRecognizer._expanded_box(box, 200, 100, 0.15, 0.35)
    assert expanded == [[74.0, 41.5], [126.0, 41.5], [126.0, 58.5], [74.0, 58.5]]

    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    centered = PPOCRRknnRecognizer._candidate_scores(image, box, 0.8)
    edge = PPOCRRknnRecognizer._candidate_scores(
        image, [[0.0, 0.0], [40.0, 0.0], [40.0, 10.0], [0.0, 10.0]], 0.8
    )
    assert centered[0] > edge[0]


def test_debug_box_mapping_accounts_for_view_origin_and_resize():
    mapped = map_box_to_frame([[20.0, 10.0], [40.0, 30.0]], (100, 50, 300, 150), 2.0)
    assert mapped == [[110.0, 55.0], [120.0, 65.0]]
