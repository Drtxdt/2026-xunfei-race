#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


PKG_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PKG_DIR / "src"
SCRIPTS_DIR = PKG_DIR / "scripts"
for path in (SOURCE_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from yolo_tools.traffic_dataset import (  # noqa: E402
    CAPTURE_CLASSES,
    CLASS_NAMES,
    difference_hash,
    hamming_distance,
    load_split_manifest,
    parse_yolo_label,
    write_data_yaml,
)
from build_traffic_yolo_dataset import collect_records  # noqa: E402
from prepare_traffic_cls_dataset import (  # noqa: E402
    CLASS_NAMES as CLS_CLASS_NAMES,
    read_image,
    vertical_band,
    write_jpeg,
)


def assert_raises_value_error(callable_object):
    try:
        callable_object()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_fixed_class_order():
    assert CLASS_NAMES == ("green_left", "green_right", "green_straight", "red_light")
    assert CAPTURE_CLASSES[-1] == "background"


def test_parse_valid_yolo_label():
    boxes = parse_yolo_label("3 0.5 0.5 0.1 0.2\n", "sample.txt")
    assert boxes == [(3, 0.5, 0.5, 0.1, 0.2)]


def test_rejects_invalid_labels():
    assert_raises_value_error(lambda: parse_yolo_label("4 0.5 0.5 0.1 0.1"))
    assert_raises_value_error(lambda: parse_yolo_label("0 0.02 0.5 0.1 0.1"))
    assert_raises_value_error(lambda: parse_yolo_label("0 0.5 0.5 -0.1 0.1"))
    assert_raises_value_error(lambda: parse_yolo_label("left 0.5 0.5 0.1 0.1"))


def test_split_manifest_rejects_session_leakage():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "splits.json"
        path.write_text(
            json.dumps({"train": ["s1"], "val": ["s2"], "test": ["s1"]}),
            encoding="utf-8",
        )
        assert_raises_value_error(lambda: load_split_manifest(path))


def test_difference_hash_is_stable_and_detects_change():
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    right = left.copy()
    right[:, 320:] = 255
    assert hamming_distance(difference_hash(left), difference_hash(left.copy())) == 0
    assert hamming_distance(difference_hash(left), difference_hash(right)) > 0


def make_raw_dataset(root):
    manifest = {
        "train": ["session_train"],
        "val": ["session_val"],
        "test": ["session_test"],
    }
    for session_index, session in enumerate(("session_train", "session_val", "session_test")):
        for class_index, class_name in enumerate(CAPTURE_CLASSES):
            directory = root / session / class_name
            directory.mkdir(parents=True)
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            image[20 + session_index * 30:80 + session_index * 30,
                  30 + class_index * 40:70 + class_index * 40] = (
                      20 + class_index * 35,
                      50 + session_index * 45,
                      120 + class_index * 20,
                  )
            image_path = directory / "{}_{}.png".format(session, class_name)
            assert cv2.imwrite(str(image_path), image)
            label_path = directory / (image_path.stem + ".txt")
            label_path.write_text(
                "" if class_name == "background" else
                "{} 0.500000 0.500000 0.100000 0.120000\n".format(class_index),
                encoding="utf-8",
            )
    return manifest


def test_collect_records_accepts_session_dataset_with_empty_background_labels():
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_root = Path(temp_dir) / "raw"
        manifest = make_raw_dataset(raw_root)
        records, errors, warnings, near_rows = collect_records(
            raw_root, manifest, (640, 480), False
        )
        assert not errors
        assert len(records) == 15
        assert len([item for item in records if item["capture_class"] == "background"]) == 3
        assert len(warnings) == 0
        assert len(near_rows) == 0


def test_collect_records_rejects_missing_background_label():
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_root = Path(temp_dir) / "raw"
        manifest = make_raw_dataset(raw_root)
        missing = raw_root / "session_train" / "background" / "session_train_background.txt"
        missing.unlink()
        _, errors, _, _ = collect_records(raw_root, manifest, (640, 480), False)
        assert any("missing label file" in message for message in errors)


def test_collect_records_rejects_folder_label_mismatch():
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_root = Path(temp_dir) / "raw"
        manifest = make_raw_dataset(raw_root)
        wrong = raw_root / "session_train" / "green_left" / "session_train_green_left.txt"
        wrong.write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        _, errors, _, _ = collect_records(raw_root, manifest, (640, 480), False)
        assert any("has no object matching folder" in message for message in errors)


def test_generated_data_yaml_keeps_class_order():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        write_data_yaml(temp_path)
        text = (temp_path / "data.yaml").read_text(encoding="utf-8")
        assert "0: green_left" in text
        assert "1: green_right" in text
        assert "2: green_straight" in text
        assert "3: red_light" in text


def test_training_hyp_disables_direction_destroying_flips():
    text = (PKG_DIR / "config" / "traffic_yolov5_hyp.yaml").read_text(encoding="utf-8")
    assert "fliplr: 0.0" in text
    assert "flipud: 0.0" in text


def test_all_25_capture_launch_files_are_valid_and_unflipped():
    launch_dir = PKG_DIR / "launch"
    tasks = sorted(launch_dir.glob("traffic_collect_s??_*.launch"))
    assert len(tasks) == 25
    expected_classes = {"green_left", "green_right", "green_straight", "red_light", "background"}
    seen = set()
    for path in tasks:
        root = ET.parse(str(path)).getroot()
        values = {
            element.attrib["name"]: element.attrib["value"]
            for element in root.findall(".//arg")
        }
        assert values["class_name"] in expected_classes
        assert values["max_images"] in {"120", "110", "100", "50"}
        seen.add((values["session"], values["class_name"]))
    assert len(seen) == 25

    common = (launch_dir / "traffic_capture_x11.launch").read_text(encoding="utf-8")
    ET.parse(str(launch_dir / "traffic_capture_x11.launch"))
    assert '<param name="flip" value="false" />' in common
    assert 'launch-prefix="xterm ' in common


def test_classifier_crop_preserves_full_width_and_expected_shape():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, 0] = (10, 20, 30)
    image[:, -1] = (40, 50, 60)
    cropped = vertical_band(image, 0.18, 0.72, (320, 160))
    assert cropped.shape == (160, 320, 3)
    assert int(cropped[:, 0].mean()) > 0
    assert int(cropped[:, -1].mean()) > 0


def test_classifier_class_order_and_training_has_no_horizontal_augmentation():
    assert CLS_CLASS_NAMES == (
        "green_left", "green_right", "green_straight", "red_light", "background"
    )
    training_source = (PKG_DIR / "scripts" / "train_traffic_resnet18.py").read_text(
        encoding="utf-8"
    )
    assert "transforms.RandomHorizontalFlip(" not in training_source
    assert '"runtime_horizontal_flip_required": True' in training_source


def test_unicode_image_io_round_trip():
    with tempfile.TemporaryDirectory() as temp_dir:
        directory = Path(temp_dir) / "讯飞数据"
        directory.mkdir()
        path = directory / "左转.jpg"
        image = np.full((24, 32, 3), 127, dtype=np.uint8)
        assert write_jpeg(path, image, 95)
        decoded = read_image(path)
        assert decoded is not None
        assert decoded.shape == image.shape
