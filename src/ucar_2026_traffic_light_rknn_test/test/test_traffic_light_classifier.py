#!/usr/bin/env python3

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np


PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from ucar_2026_traffic_light_rknn_test.classifier import (  # noqa: E402
    CLASS_NAMES,
    ScoreConsensus,
    make_detection_payload,
    preprocess_frame,
    stable_softmax,
)


def scores(winner, confidence=0.80):
    values = np.full(5, (1.0 - confidence) / 4.0, dtype=np.float32)
    values[winner] = confidence
    return values


def test_preprocess_flips_once_and_builds_exact_nhwc_input():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Two source columns remain exact after the 640 -> 320 linear resize.
    frame[:, 0:2] = (10, 20, 30)
    corrected, batch, bbox = preprocess_frame(frame, flip=True)
    assert bbox == [0, 86, 640, 346]
    assert corrected.shape == (480, 640, 3)
    assert corrected[100, 639].tolist() == [10, 20, 30]
    assert corrected[100, 0].tolist() == [0, 0, 0]
    assert batch.shape == (1, 160, 320, 3)
    assert batch.dtype == np.uint8
    # Pillow antialiases the edge, but RGB channel order must still be reversed
    # from the ascending BGR source values.
    right_edge_rgb = batch[0, 40, 319].tolist()
    assert right_edge_rgb[0] > right_edge_rgb[1] > right_edge_rgb[2] > 0


def test_softmax_shape_mapping_and_invalid_values():
    probabilities = stable_softmax(np.asarray([[0.0, 1.0, 2.0, 3.0, -1.0]]))
    assert probabilities.shape == (5,)
    assert CLASS_NAMES[int(np.argmax(probabilities))] == "red_light"
    assert abs(float(probabilities.sum()) - 1.0) < 1e-6
    for invalid in (np.zeros((1, 4)), np.asarray([0, 1, 2, 3, np.nan])):
        try:
            stable_softmax(invalid)
            assert False, "invalid output was accepted"
        except ValueError:
            pass


def test_red_and_green_use_different_confirmation_counts():
    red = ScoreConsensus(min_valid_samples=3, red_confirm_frames=2, green_confirm_frames=3)
    states = [red.update(scores(3)) for _ in range(4)]
    assert not states[2]["active"]
    assert states[3]["active"] and states[3]["class_name"] == "red_light"

    green = ScoreConsensus(min_valid_samples=3, red_confirm_frames=2, green_confirm_frames=3)
    states = [green.update(scores(0)) for _ in range(5)]
    assert not states[3]["active"]
    assert states[4]["active"] and states[4]["class_name"] == "green_left"


def test_low_quality_and_background_release_active_state():
    consensus = ScoreConsensus(min_valid_samples=1, green_confirm_frames=1, release_frames=3)
    assert consensus.update(scores(1))["active"]
    assert consensus.update(scores(4))["active"]
    assert consensus.update(scores(4))["active"]
    released = consensus.update(scores(4))
    assert not released["active"]
    assert released["reason"] == "background"

    low_margin = np.asarray([0.30, 0.29, 0.14, 0.14, 0.13], dtype=np.float32)
    state = consensus.update(low_margin)
    assert not state["active"] and state["reason"] == "low_confidence"


def test_json_payload_keeps_competition_consensus_contract():
    probabilities = scores(2)
    state = {
        "active": True,
        "class_name": "green_straight",
        "confidence": 0.8,
        "reason": "accepted",
        "valid_samples": 5,
    }
    payload = make_detection_payload(
        12.5, CLASS_NAMES, probabilities, [0, 86, 640, 346], state, 4.2, "int8"
    )
    decoded = json.loads(json.dumps(payload))
    assert decoded["consensus"]["active"] is True
    assert decoded["consensus"]["class_name"] == "green_straight"
    assert decoded["raw_detections"][0]["bbox"] == [0, 86, 640, 346]
    assert set(decoded["diagnostics"]["probabilities"]) == set(CLASS_NAMES)


def test_launch_uses_proven_external_tts_wrapper_not_legacy_rospack_lookup():
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    launch_path = os.path.join(
        package_root, "launch", "traffic_light_rknn_x11_speak_test.launch"
    )
    root = ET.parse(launch_path).getroot()
    nodes = list(root.iter("node"))
    assert not any(node.get("pkg") == "speech_command" for node in nodes)
    wrappers = [
        node
        for node in nodes
        if node.get("pkg") == "ucar_2026_competition"
        and node.get("type") == "external_voice_nodes.py"
    ]
    assert len(wrappers) == 1
    params = {item.get("name"): item.get("value") for item in wrappers[0].findall("param")}
    assert params == {"start_asr": "false", "start_tts": "true"}
