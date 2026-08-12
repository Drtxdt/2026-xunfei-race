#!/usr/bin/env python3

import json
import pathlib
import sys
import unittest


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from ucar_2026_strict_mission.logic import (  # noqa: E402
    ApproachPolicy,
    ConsecutiveBandFilter,
    DistanceCalibration,
    StableLineDistanceFilter,
    forward_progress,
    heading_alignment_command,
    lateral_displacement,
    line_alignment_command,
    lowest_horizontal_band,
    select_final_advance,
    track_launch_for_decision,
    traffic_decision_from_payload,
    valid_stop_line_geometry,
)


class DistanceCalibrationTests(unittest.TestCase):
    def test_interpolates_distance_between_calibration_rows(self):
        calibration = DistanceCalibration([[0.50, 0.40], [0.90, 0.08]])
        self.assertAlmostEqual(calibration.distance_for_ratio(0.70), 0.24)

    def test_rejects_non_monotonic_rows(self):
        with self.assertRaises(ValueError):
            DistanceCalibration([[0.50, 0.20], [0.80, 0.30]])

    def test_returns_none_outside_calibrated_image_range(self):
        calibration = DistanceCalibration([[0.50, 0.40], [0.90, 0.08]])
        self.assertIsNone(calibration.distance_for_ratio(0.45))
        self.assertIsNone(calibration.distance_for_ratio(0.95))


class OdometryProgressTests(unittest.TestCase):
    def test_projects_displacement_along_starting_heading(self):
        self.assertAlmostEqual(
            forward_progress((1.0, 2.0, 0.0), (1.12, 2.04, 0.0)),
            0.12,
        )
        self.assertAlmostEqual(
            forward_progress((1.0, 2.0, 1.57079632679),
                             (1.03, 2.12, 1.57079632679)),
            0.12,
            places=6,
        )

    def test_reverse_motion_does_not_count_as_progress(self):
        self.assertLess(
            forward_progress((0.0, 0.0, 0.0), (-0.03, 0.0, 0.0)),
            0.0,
        )

    def test_competition_requires_completed_planned_final_advance(self):
        competition_root = PACKAGE_ROOT.parent / "ucar_2026_competition"
        config = (
            competition_root / "config" / "competition.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("task4_final_progress_tolerance_m: 0.008", config)
        self.assertIn("task4_final_stop_min_m: 0.03", config)
        self.assertIn("task4_final_stop_max_m: 0.05", config)

        flow = (
            competition_root / "scripts" / "competition_flow.py"
        ).read_text(encoding="utf-8")
        self.assertIn('status.get("final_progress_m")', flow)
        self.assertIn("final_advance_completed", flow)
        self.assertIn("task4 final advance incomplete", flow)
        self.assertIn('status.get("final_visual_verified"', flow)
        self.assertIn('status.get("final_stop_line_color")', flow)
        self.assertIn("task4 final stop not accepted", flow)
        self.assertIn('final_stop_source == "hard_advance_timeout"', flow)

    def test_calibrated_advance_uses_verified_hard_stop_distance(self):
        config = json.loads(
            (PACKAGE_ROOT / "config" / "strict_mission.yaml").read_text(
                encoding="utf-8"))
        self.assertEqual(
            config["calibrated_final_advance_fallback_sec"], 3.0)
        self.assertEqual(config["final_advance_m"], 0.20)
        self.assertEqual(
            config["final_advance_target_clearance_m"], 0.05)
        self.assertEqual(config["final_advance_no_vision_m"], 0.155)
        self.assertEqual(
            config["final_advance_visual_max_age_sec"], 0.75)
        self.assertEqual(config["final_advance_min_command_m"], 0.015)
        self.assertEqual(config["final_advance_visual_bias_m"], 0.03)
        self.assertEqual(config["final_visual_confirm_frames"], 3)
        self.assertEqual(config["final_visual_max_spread_m"], 0.02)
        self.assertEqual(
            config["distance_calibration_reference"],
            "front_wheel_to_yellow_line",
        )
        self.assertEqual(config["final_visual_approach_timeout_sec"], 3.0)
        self.assertEqual(
            config["final_visual_line_missing_fault_sec"], 5.0)
        self.assertGreaterEqual(config["final_advance_speed_mps"], 0.045)
        self.assertGreaterEqual(
            config["final_advance_creep_speed_mps"], 0.030)
        self.assertLessEqual(
            config["final_advance_creep_speed_mps"],
            config["final_advance_speed_mps"],
        )
        self.assertGreater(
            config["calibrated_final_advance_fallback_sec"],
            config["line_search_delay_sec"],
        )

        node_source = (
            PACKAGE_ROOT / "scripts" / "strict_mission_node.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "using calibrated guarded final advance",
            node_source,
        )
        self.assertIn('"~final_advance_no_vision_m", 0.155)', node_source)
        self.assertIn(
            '"~calibrated_final_advance_fallback_sec", 3.0)',
            node_source,
        )
        self.assertIn("TASK4_FINAL_ADVANCE planned=", node_source)
        self.assertIn("confirmed_color=%s", node_source)
        self.assertIn("candidate_color=%s", node_source)
        self.assertIn('self.state = "FINAL_VISUAL_APPROACH"', node_source)
        self.assertIn("self.final_parked_event", node_source)
        self.assertIn("final yellow stop-line clearance confirmed", node_source)
        self.assertIn('self.final_stop_source = "hard_advance_timeout"', node_source)
        self.assertIn(
            "accepting the completed guarded hard advance", node_source)
        self.assertNotIn('color_mode="white"', node_source)
        self.assertNotIn('~white_v_min', node_source)

    def test_unconfirmed_visual_candidate_is_diagnostic_only(self):
        node_source = (
            PACKAGE_ROOT / "scripts" / "strict_mission_node.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "measured_distance = self.visual_stop_distance_m",
            node_source,
        )
        self.assertIn(
            "measured_at = self.visual_stop_distance_at",
            node_source,
        )
        self.assertIn(
            "candidate_distance = self.last_distance_m",
            node_source,
        )
        self.assertNotIn(
            "candidate_distance_m=candidate_distance",
            node_source,
        )
        distance, source = select_final_advance(
            None,
            None,
            0.05,
            0.20,
            0.155,
            0.75,
            0.015,
            0.03,
        )
        self.assertEqual(distance, 0.155)
        self.assertEqual(source, "no_vision_fallback")

    def test_fresh_visual_distance_selects_only_required_clearance(self):
        distance, source = select_final_advance(
            0.0983333333,
            0.14,
            0.05,
            0.20,
            0.18,
            0.75,
            0.015,
        )
        self.assertAlmostEqual(distance, 0.0483333333)
        self.assertEqual(source, "visual_distance")

    def test_visual_distance_bias_compensates_real_car_far_range_error(self):
        distance, source = select_final_advance(
            0.198,
            0.10,
            0.05,
            0.20,
            0.18,
            0.75,
            0.015,
            0.03,
        )
        self.assertAlmostEqual(distance, 0.178)
        self.assertEqual(source, "visual_distance")

        distance, source = select_final_advance(
            0.058,
            0.10,
            0.05,
            0.20,
            0.18,
            0.75,
            0.015,
            0.03,
        )
        self.assertEqual(distance, 0.0)
        self.assertEqual(source, "visual_hold")

    def test_stable_line_distance_requires_yellow_consensus(self):
        distance_filter = StableLineDistanceFilter(
            required=3, max_spread_m=0.02)
        self.assertIsNone(distance_filter.push(0.198, "yellow"))
        self.assertIsNone(distance_filter.push(0.202, "yellow"))
        self.assertAlmostEqual(
            distance_filter.push(0.196, "yellow"), 0.198)
        self.assertEqual(distance_filter.hits, 3)

        self.assertIsNone(distance_filter.push(0.195, "white"))
        self.assertEqual(distance_filter.hits, 0)
        self.assertIsNone(distance_filter.push(0.196, "white"))
        self.assertIsNone(distance_filter.push(0.194, "white"))
        self.assertEqual(distance_filter.hits, 0)

    def test_stable_line_distance_rejects_spread_and_misalignment(self):
        distance_filter = StableLineDistanceFilter(
            required=3, max_spread_m=0.01)
        self.assertIsNone(distance_filter.push(0.18, "yellow"))
        self.assertIsNone(distance_filter.push(0.19, "yellow"))
        self.assertIsNone(distance_filter.push(0.22, "yellow"))
        self.assertEqual(distance_filter.hits, 1)
        self.assertIsNone(
            distance_filter.push(0.20, "yellow", aligned=False))
        self.assertEqual(distance_filter.hits, 0)

    def test_visual_distance_caps_long_advance_and_holds_near_line(self):
        distance, source = select_final_advance(
            0.2525, 0.10, 0.05, 0.20, 0.18, 0.75, 0.015)
        self.assertEqual(distance, 0.20)
        self.assertEqual(source, "visual_distance")

        distance, source = select_final_advance(
            0.058, 0.10, 0.05, 0.20, 0.18, 0.75, 0.015)
        self.assertEqual(distance, 0.0)
        self.assertEqual(source, "visual_hold")

    def test_stale_or_missing_visual_distance_uses_safe_fallback(self):
        for measured, age in ((None, None), (0.098, 0.80)):
            distance, source = select_final_advance(
                measured, age, 0.05, 0.20, 0.155, 0.75, 0.015)
            self.assertEqual(distance, 0.155)
            self.assertEqual(source, "no_vision_fallback")

    def test_task4_tightens_and_restores_teb_goal_tolerances(self):
        config = json.loads(
            (PACKAGE_ROOT / "config" / "strict_mission.yaml").read_text(
                encoding="utf-8"))
        self.assertTrue(config["tighten_staging_goal_tolerance"])
        self.assertEqual(config["staging_xy_goal_tolerance"], 0.04)
        self.assertEqual(config["staging_yaw_goal_tolerance"], 0.08)

        node_source = (
            PACKAGE_ROOT / "scripts" / "strict_mission_node.py"
        ).read_text(encoding="utf-8")
        self.assertIn("TASK4_TOLERANCE_GUARD applied", node_source)
        self.assertIn("restore_staging_tolerances", node_source)

        package_xml = (
            PACKAGE_ROOT / "package.xml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "<depend>dynamic_reconfigure</depend>",
            package_xml,
        )

    def test_reports_drift_across_starting_heading(self):
        self.assertAlmostEqual(
            lateral_displacement(
                (1.0, 2.0, 0.0),
                (1.10, 2.03, 0.0),
            ),
            0.03,
        )
        self.assertAlmostEqual(
            lateral_displacement(
                (1.0, 2.0, 1.57079632679),
                (0.97, 2.10, 1.57079632679),
            ),
            0.03,
            places=6,
        )


class HeadingAlignmentTests(unittest.TestCase):
    def test_staging_heading_config_clears_drive_deadband(self):
        config = json.loads(
            (PACKAGE_ROOT / "config" / "strict_mission.yaml").read_text(
                encoding="utf-8"))
        self.assertGreaterEqual(config["staging_heading_tolerance_deg"], 5.8)
        self.assertGreaterEqual(config["staging_heading_min_speed"], 0.20)
        self.assertGreaterEqual(
            config["staging_heading_max_speed"],
            config["staging_heading_min_speed"],
        )
        self.assertTrue(config["staging_heading_fallback_to_vision"])

        node_source = (
            PACKAGE_ROOT / "scripts" / "strict_mission_node.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "continuing with visual stop-line alignment",
            node_source,
        )

    def test_stops_inside_heading_tolerance(self):
        self.assertEqual(
            heading_alignment_command(0.02, 0.04, 0.9, 0.06, 0.16),
            0.0,
        )

    def test_preserves_turn_direction_and_speed_bounds(self):
        self.assertAlmostEqual(
            heading_alignment_command(-0.20, 0.04, 0.9, 0.06, 0.16),
            -0.16,
        )
        self.assertAlmostEqual(
            heading_alignment_command(0.05, 0.04, 0.9, 0.06, 0.16),
            0.06,
        )

    def test_rejects_invalid_heading_parameters(self):
        with self.assertRaises(ValueError):
            heading_alignment_command(0.2, 0.0, 0.9, 0.06, 0.16)
        with self.assertRaises(ValueError):
            heading_alignment_command(0.2, 0.04, 0.9, 0.20, 0.16)


class ApproachPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ApproachPolicy(
            target_min_m=0.05,
            target_max_m=0.07,
            absolute_max_m=0.10,
            calibration_error_m=0.03,
        )

    def test_rejects_target_whose_error_can_exceed_ten_centimetres(self):
        with self.assertRaises(ValueError):
            ApproachPolicy(0.05, 0.08, 0.10, 0.03)

    def test_uses_staged_conservative_speeds(self):
        self.assertEqual(self.policy.command_for_distance(0.50), 0.10)
        self.assertEqual(self.policy.command_for_distance(0.25), 0.06)
        self.assertEqual(self.policy.command_for_distance(0.14), 0.05)
        self.assertEqual(self.policy.command_for_distance(0.09), 0.045)

    def test_stops_inside_target_band_and_when_too_close(self):
        self.assertEqual(self.policy.command_for_distance(0.06), 0.0)
        self.assertEqual(self.policy.command_for_distance(0.04), 0.0)
        self.assertTrue(self.policy.in_target_band(0.06))
        self.assertFalse(self.policy.in_target_band(0.08))

    def test_unknown_distance_is_fail_safe_stop(self):
        self.assertEqual(self.policy.command_for_distance(None), 0.0)


class ConfirmationTests(unittest.TestCase):
    def test_requires_consecutive_in_band_frames(self):
        filt = ConsecutiveBandFilter(3, 0.05, 0.07)
        self.assertFalse(filt.push(0.06))
        self.assertFalse(filt.push(0.06))
        self.assertFalse(filt.push(0.09))
        self.assertFalse(filt.push(0.06))
        self.assertFalse(filt.push(0.06))
        self.assertTrue(filt.push(0.06))


class TrafficTests(unittest.TestCase):
    def test_parses_only_active_consensus(self):
        active = {"consensus": {"active": True, "class_name": "green_left"}}
        inactive = {"consensus": {"active": False, "class_name": "green_left"}}
        self.assertEqual(traffic_decision_from_payload(active), "left")
        self.assertIsNone(traffic_decision_from_payload(inactive))

    def test_maps_red_to_stop_and_routes_to_existing_launch_files(self):
        red = json.loads('{"consensus":{"active":true,"class_name":"red_light"}}')
        self.assertEqual(traffic_decision_from_payload(red), "stop")
        self.assertEqual(
            track_launch_for_decision("straight"),
            ("stable_right_track_end_stop.launch",
             "/stable_right_track_end_stop/status", "stable_right_finish"),
        )
        with self.assertRaises(ValueError):
            track_launch_for_decision("stop")


class StopLineDetectionTests(unittest.TestCase):
    def test_accepts_wide_horizontal_filled_candidate(self):
        self.assertTrue(valid_stop_line_geometry(
            width_ratio=0.72, height_ratio=0.05, fill_ratio=0.85,
            bottom_ratio=0.78,
        ))

    def test_rejects_vertical_short_sparse_or_high_candidates(self):
        self.assertFalse(valid_stop_line_geometry(0.10, 0.45, 0.90, 0.90))
        self.assertFalse(valid_stop_line_geometry(0.30, 0.04, 0.90, 0.90))
        self.assertFalse(valid_stop_line_geometry(0.70, 0.04, 0.20, 0.90))
        self.assertFalse(valid_stop_line_geometry(0.70, 0.04, 0.90, 0.40))

    def test_selects_lowest_wide_horizontal_band(self):
        occupancies = [0.0] * 100
        occupancies[20:25] = [0.8] * 5
        occupancies[70:76] = [0.7] * 6
        self.assertEqual(
            lowest_horizontal_band(occupancies, 0.45, 12),
            (70, 75),
        )

    def test_rejects_single_row_noise_and_tall_white_region(self):
        occupancies = [0.0] * 100
        occupancies[50] = 0.9
        occupancies[70:90] = [0.9] * 20
        self.assertIsNone(lowest_horizontal_band(occupancies, 0.45, 12))


class StopLineAlignmentTests(unittest.TestCase):
    def command(self, angle, center):
        return line_alignment_command(
            angle, center, 0.05, 0.06,
            0.8, 0.16, -1.0,
            0.10, 0.045, -1.0,
        )

    def test_corrects_yaw_before_lateral_or_forward_motion(self):
        mode, lateral, yaw, aligned = self.command(0.20, 0.30)
        self.assertEqual(mode, "yaw")
        self.assertEqual(lateral, 0.0)
        self.assertLess(yaw, 0.0)
        self.assertFalse(aligned)

    def test_corrects_lateral_error_only_after_yaw_is_aligned(self):
        mode, lateral, yaw, aligned = self.command(0.01, 0.30)
        self.assertEqual(mode, "lateral")
        self.assertLess(lateral, 0.0)
        self.assertEqual(yaw, 0.0)
        self.assertFalse(aligned)

    def test_allows_forward_motion_only_when_fully_aligned(self):
        self.assertEqual(
            self.command(0.01, 0.02),
            ("forward", 0.0, 0.0, True),
        )

    def test_applies_minimum_yaw_speed_for_small_actionable_error(self):
        mode, lateral, yaw, aligned = line_alignment_command(
            0.051, 0.0, 0.05, 0.06,
            0.1, 0.16, -1.0,
            0.10, 0.045, -1.0,
            yaw_min=0.04,
        )
        self.assertEqual(mode, "yaw")
        self.assertEqual(lateral, 0.0)
        self.assertAlmostEqual(yaw, -0.04)
        self.assertFalse(aligned)

    def test_applies_minimum_lateral_speed_near_image_center(self):
        mode, lateral, yaw, aligned = line_alignment_command(
            0.0, 0.061, 0.05, 0.06,
            0.8, 0.16, -1.0,
            0.02, 0.045, -1.0,
            lateral_min=0.015,
        )
        self.assertEqual(mode, "lateral")
        self.assertAlmostEqual(lateral, -0.015)
        self.assertEqual(yaw, 0.0)
        self.assertFalse(aligned)

    def test_rejects_minimum_speed_above_limit(self):
        with self.assertRaises(ValueError):
            line_alignment_command(
                0.0, 0.0, 0.05, 0.06,
                0.8, 0.16, -1.0,
                0.10, 0.045, -1.0,
                lateral_min=0.05,
            )


if __name__ == "__main__":
    unittest.main()
