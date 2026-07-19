#!/usr/bin/env python3
import math
import os
import sys

import cv2
import numpy as np

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from stop_line_logic import (  # noqa: E402
    approach_speed,
    confirmed_window,
    detect_stop_line,
    load_calibration,
    save_calibration,
    safety_failure,
    staging_pose,
    target_position_state,
)


def blank():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_detects_wide_horizontal_white_line():
    image = blank()
    cv2.rectangle(image, (120, 350), (520, 380), (255, 255, 255), -1)
    result, _ = detect_stop_line(image)
    assert result is not None
    assert result["width_ratio"] >= 0.60
    assert abs(result["angle_deg"]) < 1.0


def test_rejects_longitudinal_track_line():
    image = blank()
    cv2.rectangle(image, (300, 220), (330, 470), (255, 255, 255), -1)
    result, _ = detect_stop_line(image)
    assert result is None


def test_rejects_short_or_steep_white_marks():
    short = blank()
    cv2.rectangle(short, (220, 350), (400, 375), (255, 255, 255), -1)
    assert detect_stop_line(short)[0] is None
    steep = blank()
    cv2.line(steep, (80, 450), (560, 250), (255, 255, 255), 25)
    assert detect_stop_line(steep)[0] is None


def test_rejects_large_white_reflection_region():
    image = blank()
    cv2.rectangle(image, (0, 220), (639, 479), (255, 255, 255), -1)
    assert detect_stop_line(image)[0] is None


def test_confirmation_and_speed_tiers():
    assert confirmed_window([True, False, True, False, True], 3)
    assert not confirmed_window([True, False, True, False, False], 3)
    assert approach_speed(0.20) == 0.06
    assert approach_speed(0.08) == 0.035
    assert approach_speed(0.02) == 0.02


def test_sensor_staleness_and_obstacle_fail_closed():
    assert safety_failure(10.0, 9.0, 10.0, 10.0, 0.5, 1.0, 0.2) == "image_stale"
    assert safety_failure(10.0, 10.0, 9.0, 10.0, 0.5, 1.0, 0.2) == "odom_stale"
    assert safety_failure(10.0, 10.0, 10.0, 9.0, 0.5, 1.0, 0.2) == "scan_stale"
    assert safety_failure(10.0, 10.0, 10.0, 10.0, 0.5, 0.19, 0.2).startswith(
        "front_obstacle")
    assert safety_failure(10.0, 10.0, 10.0, 10.0, 0.5, 1.0, 0.2) == ""


def test_target_gate_never_commands_reverse_after_overshoot():
    assert target_position_state(0.02, 0.012, 0.02) == "approach"
    assert target_position_state(0.005, 0.012, 0.02) == "target"
    assert target_position_state(-0.013, 0.012, 0.02) == "overshoot"
    assert target_position_state(-0.021, 0.012, 0.02) == "overshoot"


def test_staging_pose_is_35cm_behind_big_hand_goal():
    x, y, yaw = staging_pose(0.3195, -3.2703, -1.5596, 0.35)
    assert math.isclose(x, 0.31558, abs_tol=1e-4)
    assert math.isclose(y, -2.92032, abs_tol=1e-4)
    assert yaw == -1.5596


def test_calibration_round_trip():
    detections = [
        {"center_y_ratio": 0.80, "angle_deg": -0.5},
        {"center_y_ratio": 0.82, "angle_deg": 0.0},
        {"center_y_ratio": 0.84, "angle_deg": 0.5},
    ]
    path = os.path.join(os.path.dirname(__file__), ".calibration-test.yaml")
    try:
        saved = save_calibration(path, detections, 0.06, (480, 640, 3))
        loaded = load_calibration(path)
        assert saved["target_y_ratio"] == 0.82
        assert loaded["target_y_ratio"] == 0.82
        assert loaded["target_angle_deg"] == 0.0
        assert loaded["target_front_gap_m"] == 0.06
        assert loaded["image_width"] == 640
        assert loaded["image_height"] == 480
    finally:
        if os.path.exists(path):
            os.remove(path)
