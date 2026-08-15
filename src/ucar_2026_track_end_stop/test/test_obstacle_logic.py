#!/usr/bin/env python3

import os
import sys


PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from ucar_2026_track_end_stop.obstacle_logic import ObstacleAvoidanceController


def test_board_is_avoided_and_track_is_rejoined():
    controller = ObstacleAvoidanceController(
        confirm_scans=2, enable_delay=0.0, stop_hold=0.0,
        min_shift=0.20, max_shift=0.80, pass_distance=0.50,
        reacquire_duration=0.1)
    clear = {"front": 0.50, "left": 1.2, "right": 0.5}
    assert controller.update(0.0, (0, 0, 0), clear, (0.2, 0, 0)) == (0.2, 0.0, 0.0)
    assert controller.update(0.1, (0, 0, 0), clear, (0.2, 0, 0)) == (0.0, 0.0, 0.0)
    controller.update(0.2, (0, 0, 0), clear, (0.2, 0, 0))
    assert controller.state == "SHIFT_OUT"
    clear["front"] = 1.0
    controller.update(0.3, (0, 0.25, 0), clear, (0.2, 0, 0))
    assert controller.state == "PASS"
    controller.update(0.4, (0.55, 0.25, 0), clear, (0.2, 0, 0))
    assert controller.state == "SHIFT_BACK"
    controller.update(0.5, (0.55, 0.0, 0), clear, (0.2, 0, 0))
    assert controller.state == "REACQUIRE"
    controller.update(0.7, (0.60, 0.0, 0), clear, (0.2, 0, 0))
    assert controller.state == "COMPLETE"


def test_no_clear_side_faults_without_motion():
    controller = ObstacleAvoidanceController(confirm_scans=1, enable_delay=0.0)
    command = controller.update(
        0.0, (0, 0, 0),
        {"front": 0.4, "left": 0.3, "right": 0.3},
        (0.2, 0, 0))
    assert command == (0.0, 0.0, 0.0)
    assert controller.state == "FAULT"
