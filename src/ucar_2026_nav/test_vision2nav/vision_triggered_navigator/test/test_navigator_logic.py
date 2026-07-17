#!/usr/bin/env python3

import math
import os
import sys

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from navigator_logic import (
    build_observation_candidates,
    center_angular_command,
    footprint_max_cost,
    normalize_angle,
    scan_dwell_deadline,
    split_scan_angle,
)


def test_candidates_stay_inside_wall_clearance_and_keep_primary():
    candidates = build_observation_candidates(
        -1.65, -1.77,
        [[0, 0], [0, 0.28], [0, -0.28], [0.28, 0], [-0.28, 0]],
        (-2.23, 2.80, -3.27, -1.19),
        0.36,
    )
    assert candidates[0] == (-1.65, -1.77)
    assert all(-1.87 <= x <= 2.44 for x, _y in candidates)
    assert all(-2.91 <= y <= -1.55 for _x, y in candidates)


def test_center_command_stops_in_tolerance_and_has_configurable_sign():
    assert center_angular_command(0.04, 0.08, 0.08, 0.18, -1) == 0.0
    assert center_angular_command(0.5, 0.08, 0.08, 0.18, -1) < 0.0
    assert center_angular_command(0.5, 0.08, 0.08, 0.18, 1) > 0.0


def test_footprint_allows_inflation_but_rejects_lethal_cells():
    width = height = 21
    data = [0] * (width * height)
    data[10 * width + 10] = 99
    known, max_cost, blocked = footprint_max_cost(
        data, width, height, 0.1, 0.0, 0.0, 1.05, 1.05, 0.21, 253)
    assert known and max_cost == 99 and not blocked
    data[10 * width + 10] = 253
    known, max_cost, blocked = footprint_max_cost(
        data, width, height, 0.1, 0.0, 0.0, 1.05, 1.05, 0.21, 253)
    assert known and max_cost == 253 and blocked


def test_angle_wrap():
    assert math.isclose(normalize_angle(3.0 * math.pi), -math.pi)


def test_scan_angle_is_split_without_changing_total_sweep():
    steps = split_scan_angle(math.radians(129.0), math.radians(20.0))
    assert len(steps) == 7
    assert all(step <= math.radians(20.0) + 1e-9 for step in steps)
    assert math.isclose(sum(steps), math.radians(129.0), abs_tol=1e-9)
    assert math.isclose(steps[-1], math.radians(9.0), abs_tol=1e-9)


def test_scan_dwell_extends_for_candidate_but_is_bounded():
    assert math.isclose(scan_dwell_deadline(10.0, 0.65, 0.0, 1.2, 2.0), 10.65)
    assert math.isclose(scan_dwell_deadline(10.0, 0.65, 10.2, 1.2, 2.0), 11.4)
    assert math.isclose(scan_dwell_deadline(10.0, 0.65, 11.8, 1.2, 2.0), 12.0)
