import os
import sys

import numpy as np

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from factory_sign_ppocr_rknn_node import CTCLabelDecoder, FactorySignKeywordClassifier, OCRText, PPOCRRknnRecognizer, VoteWindow


def test_keyword_classifier_maps_requested_categories():
    classifier = FactorySignKeywordClassifier()
    assert classifier.classify("食品加工车间") == "food"
    assert classifier.classify("日用品加工车间") == "daily"
    assert classifier.classify("电子产品生产车间") == "electronic"
    assert classifier.classify("FOOD") == "food"
    assert classifier.classify("品") is None
    assert classifier.classify("车间") is None


def test_keyword_classifier_uses_discriminative_fuzzy_features():
    classifier = FactorySignKeywordClassifier()

    assert classifier.classify("食品") == "food"
    assert classifier.classify("日用晶") == "daily"
    assert classifier.classify("电子产生产") == "electronic"


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
    assert score > 1.6
    assert "daily" in debug


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
