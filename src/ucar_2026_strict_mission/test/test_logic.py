#!/usr/bin/env python3

import json
import pathlib
import sys
import unittest
import xml.etree.ElementTree as ET


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from ucar_2026_strict_mission.logic import (  # noqa: E402
    ApproachPolicy,
    ConsecutiveBandFilter,
    DistanceCalibration,
    classify_line_observation,
    far_line_has_progress,
    forward_progress,
    lowest_horizontal_band,
    track_launch_for_decision,
    traffic_decision_from_payload,
    valid_stop_line_geometry,
)


class TrafficStagingPoseConfigTests(unittest.TestCase):
    def test_all_competition_entrypoints_share_calibrated_staging_pose(self):
        expected = ("0.2495", "-3.0", "-1.5708")
        competition_launch = PACKAGE_ROOT.parent / "ucar_2026_competition" / "launch"

        for launch_name in ("task3_task4.launch", "full_competition.launch"):
            root = ET.parse(str(competition_launch / launch_name)).getroot()
            args = {
                item.attrib["name"]: item.attrib.get("default")
                for item in root.findall("arg")
            }
            self.assertEqual(
                (args["traffic_x"], args["traffic_y"], args["traffic_yaw"]),
                expected,
            )

        strict_root = ET.parse(str(PACKAGE_ROOT / "launch" / "strict_mission.launch")).getroot()
        strict_args = {
            item.attrib["name"]: item.attrib.get("default")
            for item in strict_root.findall("arg")
        }
        self.assertEqual(
            (
                strict_args["traffic_staging_x"],
                strict_args["traffic_staging_y"],
                strict_args["traffic_staging_yaw"],
            ),
            expected,
        )

        with (PACKAGE_ROOT / "config" / "strict_mission.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            config = json.load(stream)
        self.assertEqual(
            (
                str(config["traffic_staging_x"]),
                str(config["traffic_staging_y"]),
                str(config["traffic_staging_yaw"]),
            ),
            expected,
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

    def test_exposes_calibration_bounds_for_observation_classification(self):
        calibration = DistanceCalibration([[0.50, 0.40], [0.90, 0.08]])
        self.assertEqual(calibration.min_row_ratio, 0.50)
        self.assertEqual(calibration.max_row_ratio, 0.90)


class LineObservationTests(unittest.TestCase):
    def test_classifies_all_four_stop_line_observations(self):
        self.assertEqual(classify_line_observation(None, 0.55, 0.94), "missing")
        self.assertEqual(classify_line_observation(0.50, 0.55, 0.94), "far_visible")
        self.assertEqual(classify_line_observation(0.75, 0.55, 0.94), "calibrated")
        self.assertEqual(classify_line_observation(0.96, 0.55, 0.94), "too_close")

    def test_far_progress_accepts_image_or_odometry_and_rejects_stall(self):
        self.assertTrue(far_line_has_progress(0.48, 0.50, 0.0, 0.01, 0.03))
        self.assertTrue(far_line_has_progress(0.48, 0.481, 0.04, 0.01, 0.03))
        self.assertFalse(far_line_has_progress(0.48, 0.485, 0.02, 0.01, 0.03))

    def test_launch_and_config_expose_far_line_safety_parameters(self):
        launch_root = ET.parse(
            str(PACKAGE_ROOT / "launch" / "strict_mission.launch")
        ).getroot()
        launch_args = {
            item.attrib["name"]: item.attrib.get("default")
            for item in launch_root.findall("arg")
        }
        self.assertEqual(launch_args["line_far_approach_speed"], "0.06")
        self.assertIn("line_far_progress_timeout_sec", launch_args)
        self.assertIn("line_far_min_ratio_progress", launch_args)
        self.assertIn("line_far_min_odom_progress", launch_args)
        with (PACKAGE_ROOT / "config" / "strict_mission.yaml").open(
                "r", encoding="utf-8") as stream:
            config = json.load(stream)
        self.assertEqual(config["line_far_approach_speed"], 0.06)
        self.assertEqual(config["final_advance_m"], 0.12)


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


if __name__ == "__main__":
    unittest.main()
