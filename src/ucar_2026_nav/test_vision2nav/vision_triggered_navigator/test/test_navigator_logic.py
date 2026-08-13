#!/usr/bin/env python3

import ast
import math
import os
import sys
import xml.etree.ElementTree as ET

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
    coverage_anchor_order,
    coverage_near_anchor_action,
    coverage_non_target_early_exit_ready,
    coverage_non_target_observation_matches,
    coverage_position_needs_yaw_alignment,
    coverage_speed_profile,
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
    nearest_wall_hit,
    normalize_angle,
    obstacle_clearance_requires_stop,
    parking_footprint_margins,
    parking_footprint_inside,
    parking_goal_from_wall,
    parking_obstacle_action,
    parking_recenter_required,
    parking_rotation_obstacle_clearance,
    polar_sector_min,
    ray_segment_intersection,
    recovery_rear_distance,
    rotation_clearance_allows_near_wall,
    rotation_clearance_consensus,
    rotation_clearance_is_safe,
    scan_dwell_deadline,
    scan_step_timeout_extension,
    sensor_is_fresh,
    should_retry_coverage_goal,
    staging_pose_reached,
    staging_motion_is_rotation_stall,
    target_sample_is_fresh,
    should_skip_coverage_anchor,
    split_scan_angle,
    staging_handoff_accepted,
    staging_wall_frame_accepted,
    swept_footprint_obstacle,
    wall_normal_distance,
    wall_fit_matches_expected,
    wall_fit_is_continuous,
    wall_frame_docking_command,
    wall_frame_pose_errors,
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


def test_in_place_rotation_requires_fresh_all_around_clearance():
    assert rotation_clearance_is_safe(0.31, 0.1, 0.30)
    assert not rotation_clearance_is_safe(0.29, 0.1, 0.30)
    assert not rotation_clearance_is_safe(None, 0.1, 0.30)
    assert not rotation_clearance_is_safe(1.0, 0.6, 0.30, max_scan_age=0.5)


def test_rotation_clearance_consensus_absorbs_only_small_lidar_jitter():
    samples = [(9.80, 0.279), (9.90, 0.280), (10.00, 0.279)]
    safe, median, count = rotation_clearance_consensus(
        samples, now=10.0, min_clearance=0.28, tolerance=0.005)
    assert safe
    assert math.isclose(median, 0.279)
    assert count == 3

    unsafe, median, count = rotation_clearance_consensus(
        [(9.80, 0.274), (9.90, 0.274), (10.00, 0.274)],
        now=10.0, min_clearance=0.28, tolerance=0.005)
    assert not unsafe
    assert math.isclose(median, 0.274)
    assert count == 3


def test_parking_clearance_tolerance_absorbs_only_five_mm_jitter():
    assert not obstacle_clearance_requires_stop(0.279, 0.28, 0.005)
    assert not obstacle_clearance_requires_stop(0.278, 0.28, 0.005)
    assert not obstacle_clearance_requires_stop(0.275, 0.28, 0.005)
    assert obstacle_clearance_requires_stop(0.274, 0.28, 0.005)
    assert obstacle_clearance_requires_stop(None, 0.28, 0.005)


def test_coverage_speed_profile_accelerates_only_with_open_clearance():
    thresholds = (0.45, 0.55, 0.75, 0.90)
    assert coverage_speed_profile(None, "fast", *thresholds) == "cruise"
    assert coverage_speed_profile(0.91, "cruise", *thresholds) == "fast"
    assert coverage_speed_profile(0.80, "fast", *thresholds) == "fast"
    assert coverage_speed_profile(0.74, "fast", *thresholds) == "cruise"
    assert coverage_speed_profile(0.44, "cruise", *thresholds) == "caution"
    assert coverage_speed_profile(0.50, "caution", *thresholds) == "caution"
    assert coverage_speed_profile(0.56, "caution", *thresholds) == "cruise"


def test_coverage_speed_profile_rejects_overlapping_thresholds():
    with pytest.raises(ValueError):
        coverage_speed_profile(1.0, "cruise", 0.55, 0.45, 0.75, 0.90)


def test_non_target_scan_exit_matches_only_the_active_anchor():
    assert coverage_non_target_observation_matches(2, 2, "daily", True)
    assert not coverage_non_target_observation_matches(
        2, 3, "daily", True)
    assert not coverage_non_target_observation_matches(2, 2, "", True)
    assert not coverage_non_target_observation_matches(
        2, 2, "daily", False)


def test_non_target_scan_exit_waits_for_minimum_deliberate_steps():
    assert not coverage_non_target_early_exit_ready(0, 2)
    assert not coverage_non_target_early_exit_ready(1, 2)
    assert coverage_non_target_early_exit_ready(2, 2)
    assert coverage_non_target_early_exit_ready(3, 2)
    assert coverage_non_target_early_exit_ready(0, 0)
    assert not coverage_non_target_early_exit_ready("invalid", 2)


def test_coverage_speed_profile_is_wired_through_launch():
    package_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    config_path = os.path.join(
        package_dir, "config", "vision_triggered_navigator.yaml")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert math.isclose(float(config["coverage_max_vel_x"]), 0.72)
    assert math.isclose(float(config["coverage_max_vel_theta"]), 1.45)
    assert math.isclose(float(config["coverage_cruise_vel_x"]), 0.70)
    assert math.isclose(float(config["coverage_cruise_vel_theta"]), 1.30)
    assert math.isclose(float(config["coverage_caution_vel_x"]), 0.53)
    assert math.isclose(float(config["coverage_caution_vel_theta"]), 1.12)
    assert math.isclose(
        float(config["coverage_fast_enter_clearance"]), 0.90)

    launch_path = os.path.join(
        package_dir, "launch", "vision_triggered_navigator.launch")
    root = ET.parse(launch_path).getroot()
    launch_args = {
        item.attrib["name"]: item.attrib.get("default")
        for item in root.findall("arg")
    }
    node_params = {
        item.attrib["name"]: item.attrib.get("value")
        for item in root.find("node").findall("param")
    }
    for name in (
            "coverage_cruise_vel_x",
            "coverage_caution_vel_x",
            "coverage_caution_enter_clearance",
            "coverage_caution_exit_clearance",
            "coverage_fast_exit_clearance",
            "coverage_fast_enter_clearance"):
        assert name in launch_args
        assert node_params[name] == "$(arg {})".format(name)


def test_non_target_early_exit_is_wired_through_launch():
    package_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    config_path = os.path.join(
        package_dir, "config", "vision_triggered_navigator.yaml")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["coverage_non_target_early_exit"] is False
    assert int(config["coverage_non_target_min_scan_steps"]) == 2
    assert config["non_target_topic"] == "/vision/non_target_observation"

    launch_path = os.path.join(
        package_dir, "launch", "vision_triggered_navigator.launch")
    root = ET.parse(launch_path).getroot()
    launch_args = {
        item.attrib["name"]: item.attrib.get("default")
        for item in root.findall("arg")
    }
    node_params = {
        item.attrib["name"]: item.attrib.get("value")
        for item in root.find("node").findall("param")
    }
    for name in (
            "non_target_topic",
            "coverage_non_target_early_exit",
            "coverage_non_target_min_scan_steps"):
        assert name in launch_args
        assert node_params[name] == "$(arg {})".format(name)

    script_path = os.path.join(
        package_dir, "scripts", "vision_triggered_navigator.py")
    with open(script_path, "r", encoding="utf-8") as stream:
        source = stream.read()
    assert "early exit is deferred until deliberate" in source
    assert "remaining angles at this anchor are redundant" in source
    assert "剩余扫描已去重" in source


def test_visual_parking_restores_cruise_speed_profile():
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "vision_triggered_navigator.py"))
    with open(script_path, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=script_path)

    navigator = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and
        node.name == "VisionTriggeredNavigator")
    run_method = next(
        node for node in navigator.body
        if isinstance(node, ast.FunctionDef) and node.name == "run")
    vision_branch = next(
        node for node in ast.walk(run_method)
        if isinstance(node, ast.If) and
        any(isinstance(item, ast.Constant) and item.value == "VISION"
            for item in ast.walk(node.test)))
    calls = [
        item
        for statement in vision_branch.body
        for item in ast.walk(statement)
        if isinstance(item, ast.Call) and
        isinstance(item.func, ast.Attribute) and
        item.func.attr == "_set_coverage_speed_profile"
    ]
    assert any(
        call.args and isinstance(call.args[0], ast.Constant) and
        call.args[0].value == "cruise" and
        any(keyword.arg == "force" and
            isinstance(keyword.value, ast.Constant) and
            keyword.value.value is True
            for keyword in call.keywords)
        for call in calls)


def test_parking_clearance_tolerance_is_wired_through_launch():
    package_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    config_path = os.path.join(
        package_dir, "config", "vision_triggered_navigator.yaml")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert math.isclose(
        float(config["parking_obstacle_clearance_tolerance"]), 0.005)

    launch_path = os.path.join(
        package_dir, "launch", "vision_triggered_navigator.launch")
    root = ET.parse(launch_path).getroot()
    launch_args = {
        item.attrib["name"]: item.attrib.get("default")
        for item in root.findall("arg")
    }
    node_params = {
        item.attrib["name"]: item.attrib.get("value")
        for item in root.find("node").findall("param")
    }
    assert launch_args["parking_obstacle_clearance_tolerance"] == "0.005"
    assert node_params["parking_obstacle_clearance_tolerance"] == (
        "$(arg parking_obstacle_clearance_tolerance)")


def test_swept_footprint_parameters_are_wired_through_launch():
    package_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    with open(os.path.join(
            package_dir, "config", "vision_triggered_navigator.yaml"),
            "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["parking_obstacle_required_scans"] == 2
    assert config["parking_recovery_retry_count"] == 1
    assert config["parking_diagnostics_topic"] == (
        "/vision_triggered_navigator/parking_diagnostics")
    assert math.isclose(config["camera_boresight_yaw_offset"], 0.292)
    assert math.isclose(config["parking_endpoint_min_clearance"], 0.16)
    assert math.isclose(config["parking_staging_offset"], 0.55)

    root = ET.parse(os.path.join(
        package_dir, "launch", "vision_triggered_navigator.launch")).getroot()
    launch_args = {item.attrib["name"] for item in root.findall("arg")}
    node_params = {
        item.attrib["name"]: item.attrib.get("value")
        for item in root.find("node").findall("param")
    }
    for name in (
            "parking_staging_offset",
            "parking_endpoint_min_clearance",
            "parking_obstacle_required_scans",
            "parking_recovery_retry_count",
            "parking_diagnostics_topic"):
        assert name in launch_args
        assert node_params[name] == "$(arg {})".format(name)


def test_adjacent_wall_near_points_outside_sweep_are_allowed():
    half_length, half_width, margin = 0.171, 0.128, 0.02
    rotation = swept_footprint_obstacle(
        [(0.08, -0.259)], (0.0, 0.0, 0.30), (0.0, 0.0, 0.10),
        half_length, half_width, margin)
    assert math.isclose(math.hypot(0.08, 0.259), 0.271, abs_tol=0.001)
    assert not rotation["blocked"]

    lateral = swept_footprint_obstacle(
        [(0.0, -0.243)], (0.0, -0.06, 0.0), (0.0, -0.06, 0.0),
        half_length, half_width, margin)
    assert not lateral["blocked"]


def test_logged_boresight_calibration_resolves_right_wall_box_center():
    walls = build_quadrilateral_walls(MEASURED_CORNERS)
    observations = (
        ((2.3344, -2.7581), -0.7420),
        ((2.0986, -2.7616), -0.5815),
    )
    corrected = []
    for origin, base_yaw in observations:
        zero = nearest_wall_hit(walls, origin, base_yaw)
        assert zero["endpoint_clearance"] < 0.06
        hit = nearest_wall_hit(walls, origin, base_yaw + 0.292)
        assert hit["wall"] == "right"
        assert math.isclose(hit["point"][1], -2.97, abs_tol=0.01)
        assert 0.24 < hit["endpoint_clearance"] < 0.26
        corrected.append(hit["point"])
    assert math.hypot(
        corrected[0][0] - corrected[1][0],
        corrected[0][1] - corrected[1][1]) < 0.02


def test_corner_endpoint_guard_stops_without_reobservation_motion():
    package_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    source_path = os.path.join(
        package_dir, "scripts", "vision_triggered_navigator.py")
    with open(source_path, "r", encoding="utf-8") as stream:
        source = stream.read()
    assert "parking_geometry_invalid" in source
    assert "parking_endpoint_min_clearance" in source
    assert "corner_observation_pose" not in source
    assert "parking_corner_" not in source
    assert "_navigate_to_corner_observation" not in source


def test_cone_inside_rotation_lateral_or_reverse_sweep_is_blocked():
    dimensions = (0.171, 0.128, 0.02)
    cases = (
        ([(0.22, 0.0)], (0.0, 0.0, 0.30), (0.0, 0.0, 0.10)),
        ([(0.0, -0.19)], (0.0, -0.06, 0.0), (0.0, -0.12, 0.0)),
        ([(-0.30, 0.0)], (-0.06, 0.0, 0.0), (-0.35, 0.0, 0.0)),
    )
    for points, command, errors in cases:
        assert swept_footprint_obstacle(
            points, command, errors, *dimensions)["blocked"]


def test_staging_rejects_small_euclidean_error_with_large_tangent_error():
    assert math.hypot(0.02, 0.06) < 0.16
    assert not staging_wall_frame_accepted(0.02, 0.06, 0.01)
    assert staging_wall_frame_accepted(0.02, 0.03, 0.07)


def test_staging_errors_are_measured_in_fixed_wall_frame():
    errors = wall_frame_pose_errors(
        (0.0, 0.0, math.radians(20.0)),
        (0.02, 0.06, 0.0),
    )
    assert math.isclose(errors[0], 0.02)
    assert math.isclose(errors[1], 0.06)


def test_parking_obstacle_recovery_runs_once_then_fails_safe():
    assert parking_obstacle_action(1, 0, 2, 1) == "hold"
    assert parking_obstacle_action(2, 0, 2, 1) == "recover"
    assert parking_obstacle_action(2, 1, 2, 1) == "fail"


def test_recovery_sweep_accounts_for_unaligned_wall_projection():
    perpendicular_error = -0.29
    rear = recovery_rear_distance(
        perpendicular_error, math.radians(42.0))
    assert rear < perpendicular_error
    assert math.isclose(
        rear, perpendicular_error / math.cos(math.radians(42.0)))
    assert recovery_rear_distance(
        perpendicular_error, math.radians(80.0)) is None


def test_normal_docking_path_uses_sweep_not_legacy_fixed_clearance():
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "vision_triggered_navigator.py"))
    with open(script_path, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=script_path)
    navigator = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and
        node.name == "VisionTriggeredNavigator")
    docking = next(
        node for node in navigator.body
        if isinstance(node, ast.FunctionDef) and
        node.name == "_run_parking_docking")
    calls = {
        node.func.attr for node in ast.walk(docking)
        if isinstance(node, ast.Call) and
        isinstance(node.func, ast.Attribute)
    }
    assert "_parking_sweep_diagnostics" in calls
    assert "_parking_command_clearance" not in calls


def test_normal_parking_has_no_staging_navigation_and_rejects_reverse():
    package_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    script_path = os.path.join(
        package_dir, "scripts", "vision_triggered_navigator.py")
    with open(script_path, "r", encoding="utf-8") as stream:
        source = stream.read()
    assert "def _navigate_to_parking_staging" not in source
    assert "compute_staging_goal" not in source
    assert "parking_wall_coarse_aligning" in source
    assert "parking_reverse_rejected" in source

    run_start = source.index("    def run(self):")
    run_source = source[run_start:]
    assert run_source.index("_align_to_parking_wall(") < run_source.index(
        "_run_parking_docking(")

    docking_start = source.index("    def _run_parking_docking")
    docking_source = source[docking_start:run_start]
    assert "if command[0] < -1e-9:" in docking_source
    assert docking_source.index("parking_reverse_rejected") < (
        docking_source.index("twist.linear.x, twist.linear.y"))

    recovery_start = source.index("    def _recover_parking_to_staging")
    recovery_source = source[recovery_start:docking_start]
    assert "command = (-self.parking_recovery_reverse_speed" in recovery_source
    assert "twist.linear.x = command[0]" in recovery_source


def test_boresight_and_endpoint_guard_are_wired_through_launch():
    package_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    with open(os.path.join(
            package_dir, "config", "vision_triggered_navigator.yaml"),
            "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert math.isclose(config["camera_boresight_yaw_offset"], 0.292)
    assert math.isclose(config["parking_endpoint_min_clearance"], 0.16)

    root = ET.parse(os.path.join(
        package_dir, "launch", "vision_triggered_navigator.launch")).getroot()
    launch_args = {
        item.attrib["name"]: item.attrib.get("default")
        for item in root.findall("arg")
    }
    node_params = {
        item.attrib["name"]: item.attrib.get("value")
        for item in root.find("node").findall("param")
    }
    assert launch_args["camera_boresight_yaw_offset"] == "0.292"
    assert launch_args["parking_endpoint_min_clearance"] == "0.16"
    assert node_params["parking_endpoint_min_clearance"] == (
        "$(arg parking_endpoint_min_clearance)")
    for legacy in (
            "parking_corner_min_tangent_clearance",
            "parking_corner_observation_offset",
            "parking_corner_parallax_offset",
            "parking_corner_observation_timeout_sec"):
        assert legacy not in launch_args


def test_parking_rotation_filters_logged_wall_return_beyond_front_sector():
    normal_angle = -0.047
    wall_fit = {
        "normal": (math.cos(normal_angle), math.sin(normal_angle)),
        "distance": 0.306,
    }
    samples = [
        (math.radians(36.0), 0.267),
        (math.radians(-40.0), 0.310),
        (math.radians(90.0), 0.80),
    ]
    clearance = parking_rotation_obstacle_clearance(
        samples, wall_fit, 0.08, math.radians(35.0), 0.0225, 0.235)
    assert math.isclose(clearance, 0.80)
    assert not obstacle_clearance_requires_stop(
        clearance, 0.28, 0.005)
    wall_only_clearance = parking_rotation_obstacle_clearance(
        samples[:2], wall_fit, 0.08, math.radians(35.0), 0.0225, 0.235)
    assert math.isinf(wall_only_clearance)
    assert not obstacle_clearance_requires_stop(
        wall_only_clearance, 0.28, 0.005)


def test_parking_rotation_keeps_cone_that_protrudes_from_fitted_wall():
    normal_angle = -0.047
    wall_fit = {
        "normal": (math.cos(normal_angle), math.sin(normal_angle)),
        "distance": 0.306,
    }
    samples = [
        (math.radians(36.0), 0.267),
        (math.radians(60.0), 0.267),
        (math.radians(90.0), 0.80),
    ]
    clearance = parking_rotation_obstacle_clearance(
        samples, wall_fit, 0.08, math.radians(35.0), 0.0225, 0.235)
    assert math.isclose(clearance, 0.267)
    assert obstacle_clearance_requires_stop(
        clearance, 0.28, 0.005)


def test_parking_rotation_never_filters_wall_like_return_inside_footprint():
    wall_fit = {
        "normal": (1.0, 0.0),
        "distance": 0.20,
    }
    samples = [(math.radians(36.0), 0.16)]
    clearance = parking_rotation_obstacle_clearance(
        samples, wall_fit, 0.08, math.radians(35.0), 0.0225, 0.235)
    assert math.isclose(clearance, 0.16)
    assert obstacle_clearance_requires_stop(
        clearance, 0.28, 0.005)


def test_parking_rotation_requires_evidence_without_a_trusted_wall():
    samples = [(math.radians(36.0), 0.267)]
    assert parking_rotation_obstacle_clearance(
        samples, None, 0.08, math.radians(35.0), 0.0225) is None


def test_rotation_clearance_consensus_requires_enough_fresh_samples():
    safe, median, count = rotation_clearance_consensus(
        [(9.00, 0.50), (9.90, 0.279)],
        now=10.0, min_clearance=0.28, tolerance=0.005,
        max_sample_age=0.35, min_samples=3)
    assert not safe
    assert median is None
    assert count == 1


def test_close_continuous_wall_can_use_rotation_clearance_exception():
    distance = 0.264
    samples = []
    for degrees in range(-50, 51, 2):
        angle = math.radians(degrees)
        samples.append((angle, distance / math.cos(angle)))
    assert rotation_clearance_allows_near_wall(
        samples,
        scan_age=0.1,
        min_clearance=0.28,
        lidar_forward_offset=0.08,
        footprint_radius=0.215,
    )


def test_compact_cone_cannot_use_rotation_clearance_exception():
    samples = [
        (math.radians(degrees), 0.264 + 0.002 * abs(degrees - 90))
        for degrees in range(84, 97, 2)
    ]
    assert not rotation_clearance_allows_near_wall(
        samples,
        scan_age=0.1,
        min_clearance=0.28,
        lidar_forward_offset=0.08,
        footprint_radius=0.215,
    )


def test_stale_scan_cannot_use_rotation_clearance_exception():
    samples = [(0.0, 0.264)] * 20
    assert not rotation_clearance_allows_near_wall(
        samples,
        scan_age=0.6,
        min_clearance=0.28,
        max_scan_age=0.5,
    )


def test_polar_sector_min_wraps_and_ignores_other_directions():
    samples = [
        (math.radians(179.0), 0.42),
        (math.radians(-179.0), 0.31),
        (0.0, 0.10),
        (float("nan"), 0.01),
    ]
    assert polar_sector_min(
        samples, math.pi, math.radians(10.0)) == pytest.approx(0.31)
    assert polar_sector_min(
        samples, 0.5 * math.pi, math.radians(10.0)) is None


def test_navigation_node_imports_rotation_clearance_helper():
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "vision_triggered_navigator.py"))
    with open(script_path, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=script_path)

    imported_names = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "navigator_logic"
        for alias in node.names
    }
    assert "rotation_clearance_allows_near_wall" in imported_names
    assert "rotation_clearance_is_safe" in imported_names


def test_coverage_navigation_reports_transit_and_observation_anchor_ids():
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "vision_triggered_navigator.py"))
    with open(script_path, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=script_path)

    constants = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "coverage_anchor_transit:{}" in constants
    assert "coverage_anchor_observing:{}" in constants


def test_remembered_target_uses_saved_heading_and_short_scan_only():
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "vision_triggered_navigator.py"))
    with open(script_path, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=script_path)

    navigator = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VisionTriggeredNavigator"
    )
    method_names = {
        node.name for node in navigator.body if isinstance(node, ast.FunctionDef)
    }
    assert "_align_remembered_odom_yaw" in method_names
    assert "_scan_remembered_heading_window" in method_names

    constants = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "coverage_remembered_heading_observing:{}" in constants
    assert (
        "[vision_triggered_navigator] Remembered target was not "
        "confirmed at anchor %d within the short scan; continue to "
        "the remaining anchors without repeating this anchor's full "
        "rotation plan."
    ) in constants


def test_clearance_block_preserves_stationary_scan():
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
        and node.test.operand.func.attr == "_rotation_clearance_safe"
    )
    returns = [node for node in clearance_if.body if isinstance(node, ast.Return)]
    assert len(returns) == 1
    assert isinstance(returns[0].value, ast.Constant)
    assert returns[0].value.value is True


def test_anchor_yaw_clearance_block_holds_and_continues_coverage():
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "vision_triggered_navigator.py"))
    with open(script_path, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=script_path)

    navigator = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VisionTriggeredNavigator"
    )
    visit = next(
        node for node in navigator.body
        if isinstance(node, ast.FunctionDef) and node.name == "_visit_coverage_point"
    )
    clearance_if = next(
        node for node in ast.walk(visit)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "rotation_clearance_blocked"
    )
    called = {
        node.func.attr
        for node in ast.walk(clearance_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    returned = {
        node.value.value
        for node in ast.walk(clearance_if)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
    }
    assert "_hold_scan_step" in called
    assert "covered" in returned
    assert "failed" not in returned


def test_centering_loss_rearms_coverage_search_instead_of_stopping():
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "vision_triggered_navigator.py"))
    with open(script_path, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=script_path)

    constants = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "centering_retry_pending" in constants
    assert "centering_recovering" in constants

    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "triggered"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
    ]
    assert assignments


def test_coverage_goal_retries_aborted_timeout_and_rotation_stall_once():
    assert should_retry_coverage_goal(4, False, False, 0, 1)
    assert should_retry_coverage_goal(2, False, True, 0, 1)
    assert should_retry_coverage_goal(2, True, False, 0, 1)
    assert not should_retry_coverage_goal(4, False, False, 1, 1)
    assert not should_retry_coverage_goal(3, False, False, 0, 1)


def test_exhausted_coverage_navigation_is_skipped_without_requeue():
    package_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    script_path = os.path.join(
        package_dir, "scripts", "vision_triggered_navigator.py")
    with open(script_path, "r", encoding="utf-8") as stream:
        source = stream.read()
    assert 'return "navigation_failed"' in source
    assert '"coverage_anchor_skipped:{}"' in source
    assert "requeue_failed_coverage_anchor" not in source
    assert "coverage_anchor_deferred" not in source

    config_path = os.path.join(
        package_dir, "config", "vision_triggered_navigator.yaml")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert int(config["coverage_goal_retry_count"]) == 1
    assert "coverage_failed_revisit_limit" not in config

    launch_path = os.path.join(
        package_dir, "launch", "vision_triggered_navigator.launch")
    root = ET.parse(launch_path).getroot()
    launch_args = {
        item.attrib["name"]: item.attrib.get("default")
        for item in root.findall("arg")
    }
    node_params = {
        item.attrib["name"]: item.attrib.get("value")
        for item in root.find("node").findall("param")
    }
    assert launch_args["coverage_goal_retry_count"] == "1"
    assert node_params["coverage_goal_retry_count"] == (
        "$(arg coverage_goal_retry_count)")
    assert "coverage_failed_revisit_limit" not in launch_args


def test_second_search_starts_nearest_and_preserves_cyclic_route():
    points = [
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.0},
        {"x": 2.0, "y": 0.0},
        {"x": 3.0, "y": 0.0},
    ]
    assert cyclic_coverage_order(points, 2.1, 0.0) == [2, 3, 0, 1]


def test_coverage_anchor_order_resumes_at_explicit_anchor():
    assert coverage_anchor_order(5, preferred_anchor=3) == [2, 3, 4, 0, 1]


def test_coverage_anchor_order_skips_confirmed_irrelevant_anchors():
    assert coverage_anchor_order(
        5,
        skipped_anchors=(2, 4),
        nearest_order=[3, 4, 0, 1, 2],
    ) == [4, 0, 2]


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
        (-1.7000, -1.8735, 1.0417),
        (-1.6613, -2.2796, -3.1404),
        (-1.6846, -2.7856, -3.1176),
        (-0.6965, -2.9239, -1.5594),
        (1.3000, -2.8863, -1.5443),
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
        [("right", 2.5), ("left", 3.0)],
        [("right", 1.5), ("left", 3.0)],
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


def test_near_blocked_anchor_becomes_stationary_observation():
    assert coverage_near_anchor_action(0.46, None, 0.0) == "outside"
    assert coverage_near_anchor_action(0.44, None, 0.0) == "start"
    assert coverage_near_anchor_action(0.42, 0.44, 3.1) == "observe"
    assert coverage_near_anchor_action(0.40, 0.44, 2.9) == "reset"
    assert coverage_near_anchor_action(0.40, 0.40, 3.0) == "observe"


def test_successful_staging_goal_uses_bounded_handoff_envelope():
    current = (0.122, 0.0, 0.080)
    goal = (0.0, 0.0, 0.0)
    assert not staging_pose_reached(current, goal, 0.10, 0.10)
    assert staging_handoff_accepted(
        current, goal, True, 0.10, 0.10, 0.15, 0.12)
    assert not staging_handoff_accepted(
        current, goal, False, 0.10, 0.10, 0.15, 0.12)
    assert not staging_handoff_accepted(
        (0.151, 0.0, 0.080), goal, True,
        0.10, 0.10, 0.15, 0.12)


def test_recenter_requires_a_fresh_target_sample_before_motion():
    assert target_sample_is_fresh(-0.02, 9.6, 10.0, 0.8)
    assert not target_sample_is_fresh(None, 9.9, 10.0, 0.8)
    assert not target_sample_is_fresh(-0.02, 9.0, 10.0, 0.8)


def test_recenter_preserves_an_initial_alignment_already_in_tolerance():
    assert not parking_recenter_required(0.061, 0.08)
    assert not parking_recenter_required(-0.061, 0.08)
    assert parking_recenter_required(0.081, 0.08)
    assert parking_recenter_required(None, 0.08)


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


def test_slow_scan_step_gets_bounded_time_to_finish_remaining_angle():
    extra = scan_step_timeout_extension(
        math.radians(13.0),
        math.radians(20.0),
        elapsed=2.7,
        progress_age=0.2,
        commanded_speed=0.50,
        max_extra_sec=3.0,
    )
    assert 1.8 < extra < 2.2

    capped = scan_step_timeout_extension(
        math.radians(1.0),
        math.radians(20.0),
        elapsed=2.7,
        progress_age=0.1,
        commanded_speed=0.50,
        max_extra_sec=3.0,
    )
    assert math.isclose(capped, 3.0)


def test_stalled_scan_step_does_not_extend_indefinitely():
    assert scan_step_timeout_extension(
        math.radians(13.0),
        math.radians(20.0),
        elapsed=2.7,
        progress_age=1.0,
        commanded_speed=0.50,
        max_extra_sec=3.0,
        progress_fresh_sec=0.8,
    ) == 0.0
    assert scan_step_timeout_extension(
        0.0,
        math.radians(20.0),
        elapsed=2.7,
        progress_age=0.0,
        commanded_speed=0.50,
        max_extra_sec=3.0,
    ) == 0.0


def test_scan_step_recovery_is_configured_and_wired_through_launch():
    package_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    config_path = os.path.join(
        package_dir, "config", "vision_triggered_navigator.yaml")
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert math.isclose(float(config["coverage_scan_step_max_extra_sec"]), 3.0)
    assert int(config["coverage_scan_step_retry_count"]) == 1
    assert math.isclose(
        float(config["coverage_scan_step_retry_settle_sec"]), 0.20)
    assert math.isclose(float(config["coverage_scan_progress_fresh_sec"]), 0.8)
    assert math.isclose(
        float(config["coverage_scan_progress_epsilon_deg"]), 0.5)

    launch_path = os.path.join(
        package_dir, "launch", "vision_triggered_navigator.launch")
    root = ET.parse(launch_path).getroot()
    launch_args = {
        item.attrib["name"]: item.attrib.get("default")
        for item in root.findall("arg")
    }
    node_params = {
        item.attrib["name"]: item.attrib.get("value")
        for item in root.find("node").findall("param")
    }
    for name in (
            "coverage_scan_step_max_extra_sec",
            "coverage_scan_step_retry_count",
            "coverage_scan_step_retry_settle_sec",
            "coverage_scan_progress_fresh_sec",
            "coverage_scan_progress_epsilon_deg"):
        assert name in launch_args
        assert node_params[name] == "$(arg {})".format(name)

    script_path = os.path.join(
        package_dir, "scripts", "vision_triggered_navigator.py")
    with open(script_path, "r", encoding="utf-8") as stream:
        source = stream.read()
    assert "scan_step_timeout_extension(" in source
    assert "retry_index < self.coverage_scan_step_retry_count" in source
    assert "attempt_angle = remaining_angle" in source


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


def test_all_direct_task2_rotations_have_lidar_clearance_guards():
    script_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "scripts",
        "vision_triggered_navigator.py"))
    with open(script_path, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=script_path)
    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VisionTriggeredNavigator")
    methods = {
        node.name: node for node in controller.body
        if isinstance(node, ast.FunctionDef)
    }
    for method_name in ("_step_scan", "rotate", "_rotate_center_step"):
        called = {
            node.func.attr
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        assert "_rotation_clearance_safe" in called
