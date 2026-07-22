#!/usr/bin/env python3

import ast
import math
import os
import sys

import yaml
import pytest

SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from navigator_logic import (
    build_quadrilateral_walls,
    center_angular_command,
    center_step_angle,
    coverage_motion_is_rotation_stall,
    coverage_position_needs_yaw_alignment,
    coverage_timeout_decision,
    cyclic_coverage_order,
    costmap_value_at,
    docking_command,
    docking_pose_errors,
    docking_within_tolerance,
    exact_observation_target,
    fit_wall_line,
    footprint_max_cost,
    latch_trigger,
    lidar_base_wall_distance,
    lidar_requires_stop,
    normalize_angle,
    parking_footprint_margins,
    parking_footprint_inside,
    parking_goal_from_wall,
    ray_segment_intersection,
    rotation_clearance_is_safe,
    scan_dwell_deadline,
    sensor_is_fresh,
    staging_pose_reached,
    staging_motion_is_rotation_stall,
    target_sample_is_fresh,
    should_skip_coverage_anchor,
    split_scan_angle,
    wall_normal_distance,
    wall_fit_matches_expected,
    wall_fit_is_continuous,
    wall_frame_docking_command,
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


def test_in_place_rotation_requires_fresh_all_around_clearance():
    assert rotation_clearance_is_safe(0.31, 0.1, 0.30)
    assert not rotation_clearance_is_safe(0.29, 0.1, 0.30)
    assert not rotation_clearance_is_safe(None, 0.1, 0.30)
    assert not rotation_clearance_is_safe(1.0, 0.6, 0.30, max_scan_age=0.5)
    latched, accepted = latch_trigger(latched)
    assert latched and not accepted


def test_clearance_skip_does_not_report_scan_success():
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "vision_triggered_navigator.py"))
    with open(script_path, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=script_path)

    navigator = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VisionTriggeredNavigator"
    )
    step_scan = next(
        node for node in navigator.body
        if isinstance(node, ast.FunctionDef) and node.name == "_step_scan"
    )
    clearance_if = next(
        node for node in ast.walk(step_scan)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Call)
        and isinstance(node.test.operand.func, ast.Attribute)
        and node.test.operand.func.attr == "_rotation_clearance_is_safe"
    )
    returns = [node for node in clearance_if.body if isinstance(node, ast.Return)]
    assert len(returns) == 1
    assert isinstance(returns[0].value, ast.Constant)
    assert returns[0].value.value is False


def test_second_search_starts_nearest_and_preserves_cyclic_route():
    points = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.0},
        {"x": 2.0, "y": 0.0},
        {"x": 3.0, "y": 0.0},
    ]
    assert cyclic_coverage_order(points, 2.1, 0.0) == [2, 3, 0, 1]


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


def test_calibrated_nine_anchor_order_is_preserved_without_offsets():
    calibrated = [
        (-1.6499, -1.7735, 1.0417),
        (-1.6613, -2.2796, -3.1404),
        (-1.6846, -2.7856, -3.1176),
        (-0.6965, -2.8239, -1.5594),
        (1.2660, -2.8863, -1.5443),
        (2.3356, -2.7417, -2.1786),
        (2.3471, -1.6090, -0.8607),
        (1.2641, -1.6452, 0.8530),
        (0.3011, -1.6319, 0.8913),
    ]
    points = [
        {"x": x, "y": y, "yaw": yaw, "rotations": []}
        for x, y, yaw in calibrated
    ]
    assert [exact_observation_target(point) for point in points] == calibrated

    config_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "config",
        "vision_triggered_navigator.yaml"))
    with open(config_path, "r", encoding="utf-8") as stream:
        configured = yaml.safe_load(stream)["patrol_points"]
    assert [exact_observation_target(point) for point in configured] == calibrated
    assert [[(rotation["direction"], rotation["duration"])
             for rotation in point["rotations"]] for point in configured] == [
        [("left", 4.5)],
        [("right", 3.0), ("left", 3.5)],
        [("left", 4.5)],
        [("right", 3.0), ("left", 3.0)],
        [("right", 3.0), ("left", 3.0)],
        [("left", 4.5)],
        [("left", 4.5)],
        [("left", 4.0)],
        [("left", 4.0)],
    ]


def test_only_known_lethal_cost_skips_a_coverage_anchor():
    assert not should_skip_coverage_anchor(False, -1, 253)
    assert not should_skip_coverage_anchor(True, 252, 253)
    assert should_skip_coverage_anchor(True, 253, 253)
    assert should_skip_coverage_anchor(True, 254, 253)


def test_coverage_goal_soft_timeout_extends_only_with_recent_progress():
    assert coverage_timeout_decision(24.9, 0.0, 25.0, 40.0, 0.03) == "continue"
    assert coverage_timeout_decision(25.0, 0.029, 25.0, 40.0, 0.03) == "soft_timeout"
    assert coverage_timeout_decision(25.0, 0.03, 25.0, 40.0, 0.03) == "extend"
    assert coverage_timeout_decision(39.9, 0.10, 25.0, 40.0, 0.03) == "extend"
    assert coverage_timeout_decision(40.0, 1.0, 25.0, 40.0, 0.03) == "hard_timeout"


def test_coverage_rotation_stall_and_local_yaw_handoff():
    assert coverage_motion_is_rotation_stall(
        0.029, math.radians(90.1), 0.03, math.radians(90.0))
    assert not coverage_motion_is_rotation_stall(
        0.031, math.radians(180.0), 0.03, math.radians(90.0))
    assert coverage_position_needs_yaw_alignment(0.15, math.pi, 0.15, 0.06)
    assert not coverage_position_needs_yaw_alignment(0.151, math.pi, 0.15, 0.06)
    assert not coverage_position_needs_yaw_alignment(0.10, 0.05, 0.15, 0.06)


def test_recenter_requires_a_fresh_target_sample_before_motion():
    assert target_sample_is_fresh(-0.02, 9.6, 10.0, 0.8)
    assert not target_sample_is_fresh(None, 9.9, 10.0, 0.8)
    assert not target_sample_is_fresh(-0.02, 9.0, 10.0, 0.8)


def test_cost_query_uses_coordinates_already_transformed_to_costmap_frame():
    data = [0] * 16
    data[2 * 4 + 1] = 99
    assert costmap_value_at(data, 4, 4, 1.0, 0.0, 0.0, 1.2, 2.4) == 99
    assert costmap_value_at(data, 4, 4, 1.0, 0.0, 0.0, 100.0, 100.0) == -1
    data[2 * 4 + 1] = -1
    assert costmap_value_at(data, 4, 4, 1.0, 0.0, 0.0, 1.2, 2.4) == -1


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
    for value, expected_blocked in [(252, False), (253, True), (254, True)]:
        data[10 * width + 10] = value
        known, max_cost, blocked = footprint_max_cost(
            data, width, height, 0.1, 0.0, 0.0, 1.05, 1.05, 0.21, 253)
        assert known and max_cost == value and blocked is expected_blocked

    unknown = [-1] * (width * height)
    known, max_cost, blocked = footprint_max_cost(
        unknown, width, height, 0.1, 0.0, 0.0, 1.05, 1.05, 0.21, 253)
    assert not known and max_cost == -1 and not blocked


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


def test_docking_errors_are_expressed_in_robot_body_frame():
    forward, lateral, yaw_error = docking_pose_errors(
        (1.0, 2.0, math.pi / 2.0), (0.8, 2.3, math.pi / 2.0 + 0.04))
    assert math.isclose(forward, 0.3, abs_tol=1e-9)
    assert math.isclose(lateral, 0.2, abs_tol=1e-9)
    assert math.isclose(yaw_error, 0.04, abs_tol=1e-9)


def test_docking_aligns_before_forward_motion_and_limits_every_axis():
    command = docking_command(
        (0.30, 0.10, 0.10), 0.015, 0.02, 0.035,
        0.10, 0.06, 0.15)
    assert command == (0.0, 0.06, 0.15)

    command = docking_command(
        (0.30, 0.01, 0.01), 0.015, 0.02, 0.035,
        0.10, 0.06, 0.15)
    assert command == (0.10, 0.0, 0.0)
    assert docking_within_tolerance(
        (0.015, -0.02, 0.035), 0.015, 0.02, 0.035)
    assert not docking_within_tolerance(
        (0.016, 0.0, 0.0), 0.015, 0.02, 0.035)


def test_staging_watchdog_detects_rotation_without_translation():
    assert staging_motion_is_rotation_stall(
        0.029, math.radians(45.1), 0.03, math.radians(45.0))
    assert not staging_motion_is_rotation_stall(
        0.030, math.radians(90.0), 0.03, math.radians(45.0))
    assert not staging_motion_is_rotation_stall(
        0.0, math.radians(45.0), 0.03, math.radians(45.0))


def test_sensor_freshness_and_lidar_extrinsic_safety_distance():
    assert sensor_is_fresh(9.6, 10.0, 0.5)
    assert not sensor_is_fresh(9.4, 10.0, 0.5)
    assert not sensor_is_fresh(0.0, 10.0, 0.5)
    assert math.isclose(lidar_base_wall_distance(0.14, 0.08), 0.22)
    # 0.14m raw is the expected wall at a safe 0.22m base distance.
    assert not lidar_requires_stop(0.14, 0.22, 0.22, 0.15)
    # The same raw return is an unexpected obstacle while geometry says 0.35m.
    assert lidar_requires_stop(0.14, 0.22, 0.35, 0.15)
    assert lidar_requires_stop(0.06, 0.14, 0.22, 0.15)


def test_26cm_final_goal_has_more_than_2cm_footprint_margin():
    diagnostics = parking_footprint_margins(
        (0.26, 0.0, math.pi), (0.0, 0.0), (1.0, 0.0),
        0.50, 0.50, 0.171, 0.128, 0.0)
    assert diagnostics["inside"]
    assert min(diagnostics["near_margin"], diagnostics["far_margin"],
               diagnostics["side_margin"]) >= 0.02
    assert math.isclose(
        wall_normal_distance((0.26, 0.0, math.pi), (0.0, 0.0), (1.0, 0.0)),
        0.26)


def test_staging_and_final_goals_share_wall_tangent_and_yaw():
    wall_point = (1.2, -0.7)
    normal = (-0.8, 0.6)
    staging = parking_goal_from_wall(wall_point, normal, 0.55)
    final = parking_goal_from_wall(wall_point, normal, 0.26)
    assert math.isclose(staging[2], final[2], abs_tol=1e-12)
    assert math.isclose(math.hypot(staging[0] - final[0],
                                   staging[1] - final[1]), 0.29, abs_tol=1e-12)


@pytest.mark.parametrize("wall_point,normal", [
    ((2.7767, -2.9992), (-0.9999, 0.0129)),   # task2_food.log
    ((-2.2238, -2.5445), (1.0000, 0.0056)),  # task2_daily.log
    ((0.6399, -1.2183), (0.0112, -0.9999)),  # task2_electronics.log
])
def test_logged_wall_solutions_make_safe_staging_and_26cm_final_goal(
        wall_point, normal):
    staging = parking_goal_from_wall(wall_point, normal, 0.55)
    final = parking_goal_from_wall(wall_point, normal, 0.26)
    assert math.isclose(wall_normal_distance(staging, wall_point, normal),
                        0.55, abs_tol=1e-9)
    assert math.isclose(wall_normal_distance(final, wall_point, normal),
                        0.26, abs_tol=1e-9)
    diagnostics = parking_footprint_margins(
        final, wall_point, normal, 0.50, 0.50, 0.171, 0.128, 0.0)
    assert diagnostics["inside"]
    assert min(diagnostics["near_margin"], diagnostics["far_margin"],
               diagnostics["side_margin"]) >= 0.02


def test_staging_requires_position_and_heading_together():
    goal = (1.0, 2.0, 0.5)
    assert staging_pose_reached((1.08, 2.0, 0.58), goal, 0.10, 0.10)
    assert not staging_pose_reached((1.11, 2.0, 0.5), goal, 0.10, 0.10)
    assert not staging_pose_reached((1.0, 2.0, 0.61), goal, 0.10, 0.10)


def test_wall_frame_controller_honors_deadband_and_phase_order():
    assert wall_frame_docking_command(
        0.20, 0.10, 0.055, 0.015, 0.02, 0.035,
        0.10, 0.06, 0.15, 0.15) == (0.0, 0.0, 0.15)
    assert wall_frame_docking_command(
        0.20, -0.10, 0.02, 0.015, 0.02, 0.035,
        0.10, 0.06, 0.15, 0.15) == (0.0, -0.06, 0.0)
    assert wall_frame_docking_command(
        0.20, 0.01, 0.02, 0.015, 0.02, 0.035,
        0.10, 0.06, 0.15, 0.15) == (0.10, 0.0, 0.0)


def test_wall_fit_uses_long_wall_and_rejects_short_cone_cluster():
    normal_angle = 0.05
    nx, ny = math.cos(normal_angle), math.sin(normal_angle)
    tx, ty = -ny, nx
    points = []
    for index in range(41):
        along = -0.30 + index * 0.015
        noise = ((index % 3) - 1) * 0.002
        points.append((nx * (0.42 + noise) + tx * along,
                       ny * (0.42 + noise) + ty * along))
    # Dense but physically short clutter must not win as a wall.
    points.extend((0.24, -0.02 + index * 0.004) for index in range(10))
    fit = fit_wall_line(points, 12, 0.25, 0.015)
    assert fit is not None
    assert fit["inliers"] >= 35
    assert math.isclose(fit["distance"], 0.42, abs_tol=0.005)
    assert abs(normalize_angle(fit["normal_angle"] - normal_angle)) < 0.02
    assert wall_fit_matches_expected(fit, 0.0, math.radians(20))
    assert not wall_fit_matches_expected(fit, math.pi / 2.0, math.radians(20))


def test_near_wall_fit_may_shrink_only_when_continuous():
    previous = {
        "distance": 0.272, "normal_angle": 0.011,
        "span": 0.252, "residual": 0.0014, "inliers": 80,
    }
    current = {
        "distance": 0.260, "normal_angle": 0.009,
        "span": 0.19, "residual": 0.0015, "inliers": 60,
    }
    assert wall_fit_is_continuous(
        current, previous, 0.05, math.radians(8.0))
    jumped = dict(current, distance=0.19)
    assert not wall_fit_is_continuous(
        jumped, previous, 0.05, math.radians(8.0))


@pytest.mark.parametrize("logged_yaw_error", [-1.690, -1.283, -0.341])
def test_logged_bad_handoffs_are_rejected_before_direct_docking(logged_yaw_error):
    assert not staging_pose_reached(
        (0.0, 0.0, logged_yaw_error), (0.0, 0.0, 0.0), 0.10, 0.10)


@pytest.mark.parametrize("label", ["food", "daily", "electronics"])
def test_corrected_staging_pose_finishes_three_phase_docking_within_15s(label):
    del label
    normal_error, tangent_error, yaw_error = 0.33, 0.10, 0.10
    dt = 0.05
    elapsed = 0.0
    while elapsed < 15.0 and not docking_within_tolerance(
            (normal_error, tangent_error, yaw_error), 0.015, 0.02, 0.035):
        command = wall_frame_docking_command(
            normal_error, tangent_error, yaw_error,
            0.015, 0.02, 0.035, 0.10, 0.06, 0.15, 0.15)
        normal_error -= command[0] * dt
        tangent_error -= command[1] * dt
        yaw_error -= command[2] * dt
        elapsed += dt
    assert elapsed < 15.0
    assert docking_within_tolerance(
        (normal_error, tangent_error, yaw_error), 0.015, 0.02, 0.035)
