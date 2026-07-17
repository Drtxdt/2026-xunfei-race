#!/usr/bin/env python3

import math
import os
import sys

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from navigator_logic import (
    build_observation_candidates,
    build_quadrilateral_walls,
    center_angular_command,
    center_step_angle,
    footprint_max_cost,
    latch_trigger,
    normalize_angle,
    parking_footprint_margins,
    parking_footprint_inside,
    parking_goal_from_wall,
    ray_segment_intersection,
    scan_dwell_deadline,
    split_scan_angle,
)


MEASURED_CORNERS = [
    [-2.2311, -1.2505],
    [2.8000, -1.1940],
    [-2.2197, -3.2746],
    [2.7739, -3.2186],
]


def test_one_shot_trigger_is_idempotent():
    latched, accepted = latch_trigger(False)
    assert latched and accepted
    latched, accepted = latch_trigger(latched)
    assert latched and not accepted


def test_measured_quadrilateral_wall_normals_point_inward():
    walls = build_quadrilateral_walls(MEASURED_CORNERS)
    centroid = (
        sum(point[0] for point in MEASURED_CORNERS) / 4.0,
        sum(point[1] for point in MEASURED_CORNERS) / 4.0,
    )
    assert [wall[0] for wall in walls] == ["left", "right", "bottom", "top"]
    for _name, start, end, normal in walls:
        midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        toward_center = (centroid[0] - midpoint[0], centroid[1] - midpoint[1])
        assert math.isclose(math.hypot(*normal), 1.0, abs_tol=1e-9)
        assert normal[0] * toward_center[0] + normal[1] * toward_center[1] > 0.0


def test_rays_hit_each_skewed_wall_and_make_continuous_goals():
    walls = build_quadrilateral_walls(MEASURED_CORNERS)
    centroid = (
        sum(point[0] for point in MEASURED_CORNERS) / 4.0,
        sum(point[1] for point in MEASURED_CORNERS) / 4.0,
    )
    for _name, start, end, normal in walls:
        midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        direction = (midpoint[0] - centroid[0], midpoint[1] - centroid[1])
        ray_t = ray_segment_intersection(centroid, direction, start, end)
        assert ray_t is not None
        intersection = (
            centroid[0] + ray_t * direction[0],
            centroid[1] + ray_t * direction[1],
        )
        assert math.isclose(intersection[0], midpoint[0], abs_tol=1e-8)
        assert math.isclose(intersection[1], midpoint[1], abs_tol=1e-8)
        gx, gy, yaw = parking_goal_from_wall(intersection, normal, 0.25)
        normal_distance = (gx - intersection[0]) * normal[0] + (gy - intersection[1]) * normal[1]
        assert math.isclose(normal_distance, 0.25, abs_tol=1e-9)
        assert math.isclose(math.cos(yaw), -normal[0], abs_tol=1e-9)
        assert math.isclose(math.sin(yaw), -normal[1], abs_tol=1e-9)


def test_parking_goal_supports_independent_normal_and_tangent_calibration():
    gx, gy, yaw = parking_goal_from_wall(
        (0.0, 0.0), (1.0, 0.0), 0.25,
        normal_offset=0.02, tangent_offset=0.03,
    )
    assert math.isclose(gx, 0.27)
    assert math.isclose(gy, 0.03)
    assert math.isclose(abs(yaw), math.pi)


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


def test_center_step_uses_coarse_fine_and_stop_bands():
    assert center_step_angle(0.40, 0.08, 0.20,
                             math.radians(4), math.radians(2)) == math.radians(4)
    assert center_step_angle(0.12, 0.08, 0.20,
                             math.radians(4), math.radians(2)) == math.radians(2)
    assert center_step_angle(0.04, 0.08, 0.20,
                             math.radians(4), math.radians(2)) == 0.0


def test_full_footprint_must_fit_inside_50cm_wall_box():
    valid_poses = [
        ((0.25, 0.0, math.pi), (1.0, 0.0)),
        ((-0.25, 0.0, 0.0), (-1.0, 0.0)),
        ((0.0, 0.25, -math.pi / 2.0), (0.0, 1.0)),
        ((0.0, -0.25, math.pi / 2.0), (0.0, -1.0)),
    ]
    for pose, inward in valid_poses:
        assert parking_footprint_inside(
            pose, (0.0, 0.0), inward,
            0.50, 0.50, 0.171, 0.128, 0.01)
    assert not parking_footprint_inside(
        (0.40, 0.0, math.pi), (0.0, 0.0), (1.0, 0.0),
        0.50, 0.50, 0.171, 0.128, 0.01)
    assert not parking_footprint_inside(
        (0.25, 0.20, math.pi), (0.0, 0.0), (1.0, 0.0),
        0.50, 0.50, 0.171, 0.128, 0.01)

    diagnostics = parking_footprint_margins(
        (0.25, 0.0, math.pi), (0.0, 0.0), (1.0, 0.0),
        0.50, 0.50, 0.171, 0.128, 0.01)
    assert diagnostics["inside"]
    assert diagnostics["near_margin"] > 0.0
    assert diagnostics["far_margin"] > 0.0
    assert diagnostics["side_margin"] > 0.0
    assert math.isclose(diagnostics["normal_error"], 0.0, abs_tol=1e-9)
    assert math.isclose(diagnostics["tangent_error"], 0.0, abs_tol=1e-9)
    assert len(diagnostics["corners"]) == 4
    assert all("side_margin" in corner for corner in diagnostics["corners"])


def test_tight_goal_tolerance_keeps_rotated_footprint_in_box():
    assert parking_footprint_inside(
        (0.29, 0.04, math.pi + 0.06), (0.0, 0.0), (1.0, 0.0),
        0.50, 0.50, 0.171, 0.128, 0.01)


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
