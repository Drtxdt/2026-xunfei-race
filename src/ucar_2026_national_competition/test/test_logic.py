# -*- coding: utf-8 -*-
"""Unit tests for ucar_2026_national_competition pure logic."""

from __future__ import annotations

import math
import unittest

from ucar_2026_national_competition.logic import (
    build_roslaunch_command,
    flow_launch_args,
    heading_alignment_command,
    handover_chain,
    items_equal_allowed,
    min_valid_range,
    provincial_flow_paused,
    provincial_flow_terminal,
    rotation_clearance_ok,
    shortest_angular_error,
    stage_sequence,
    status_state,
    task1_categories_match,
    validate_pose,
)


TASK1 = {
    "pickup_item": "香蕉",
    "pickup_major": "food",
    "pickup_workshop": "食品加工车间",
    "sim_item": "毛巾",
    "sim_major": "daily",
    "sim_workshop": "日用品加工车间",
}


class StageSequenceTests(unittest.TestCase):
    def test_full_with_ramp(self):
        stages = stage_sequence("full", True, True)
        self.assertEqual(stages[0], "voice_handshake")
        self.assertIn("traverse_ramp", stages)
        self.assertEqual(stages[-1], "handover")

    def test_full_without_ramp_drops_ramp_stages(self):
        stages = stage_sequence("full", True, False)
        self.assertNotIn("traverse_ramp", stages)
        self.assertNotIn("post_ramp_recovery", stages)
        self.assertEqual(stages[-1], "handover")

    def test_task1_mode_has_no_handover(self):
        stages = stage_sequence("task1", True, True)
        self.assertNotIn("handover", stages)
        self.assertEqual(stages[-1], "reason_and_announce")

    def test_ramp_mode(self):
        self.assertEqual(
            stage_sequence("ramp", True, True),
            ("navigate_ramp_staging", "traverse_ramp", "post_ramp_recovery"))

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            stage_sequence("task9", True, True)


class HandoverChainTests(unittest.TestCase):
    def test_chain_with_simulation(self):
        self.assertEqual(
            handover_chain(True), ("task2", "task3_task4", "task4_task5"))

    def test_chain_without_simulation(self):
        self.assertEqual(
            handover_chain(False), ("task2", "task4", "task5"))


class FlowArgsTests(unittest.TestCase):
    def test_args_carry_task1_result(self):
        args = flow_launch_args("task2", TASK1, True)
        self.assertEqual(args["start_stage"], "task2")
        self.assertEqual(args["target_category"], "food")
        self.assertEqual(args["target_item"], "香蕉")
        self.assertEqual(args["target_workshop"], "食品加工车间")
        self.assertEqual(args["sim_target_category"], "daily")
        self.assertTrue(args["enable_simulation"])

    def test_traffic_pose_passthrough(self):
        args = flow_launch_args(
            "task3_task4", TASK1, True,
            traffic_pose=(0.24, -3.10, -1.56, True))
        self.assertTrue(args["traffic_pose_configured"])
        self.assertAlmostEqual(args["traffic_x"], 0.24)

    def test_track_package_passthrough(self):
        args = flow_launch_args("task4_task5", TASK1, True)
        self.assertEqual(args["track_package"], "ucar_2026_track_end_stop")
        args = flow_launch_args(
            "task4_task5", TASK1, True,
            track_package="ucar_2026_track_end_stop_provincial")
        self.assertEqual(
            args["track_package"], "ucar_2026_track_end_stop_provincial")

    def test_missing_pickup_category_rejected(self):
        broken = dict(TASK1, pickup_major="")
        with self.assertRaises(ValueError):
            flow_launch_args("task2", broken, True)
        with self.assertRaises(ValueError):
            flow_launch_args("", TASK1, True)


class CommandTests(unittest.TestCase):
    def test_bools_render_lowercase(self):
        command = build_roslaunch_command(
            "ucar_2026_competition", "flow_node.launch",
            {"start_stage": "task2", "enable_simulation": True})
        self.assertEqual(
            command,
            ["roslaunch", "ucar_2026_competition", "flow_node.launch",
             "start_stage:=task2", "enable_simulation:=true"])


class StatusTests(unittest.TestCase):
    def test_pause_and_terminal_detection(self):
        self.assertTrue(provincial_flow_paused({"state": "paused"}))
        self.assertFalse(provincial_flow_paused({"state": "running"}))
        self.assertTrue(provincial_flow_terminal({"state": "completed"}))
        self.assertEqual(status_state({"state": "Running"}), "running")
        self.assertEqual(status_state(None), "")


class SafetyTests(unittest.TestCase):
    def test_min_valid_range(self):
        ranges = [1.0, float("inf"), 0.5, float("nan"), 9.0]
        self.assertAlmostEqual(min_valid_range(ranges, 0.1, 5.0), 0.5)
        self.assertIsNone(min_valid_range([float("inf")]))

    def test_rotation_clearance(self):
        self.assertTrue(rotation_clearance_ok(None, 0.15))
        self.assertTrue(rotation_clearance_ok(0.30, 0.15))
        self.assertFalse(rotation_clearance_ok(0.10, 0.15))

    def test_pose_validation(self):
        self.assertEqual(
            validate_pose(1.0, 2.0, 0.5, "test"), (1.0, 2.0, 0.5))
        with self.assertRaises(ValueError):
            validate_pose(float("nan"), 0.0, 0.0, "test")

    def test_category_match(self):
        self.assertTrue(task1_categories_match(TASK1, "food", "daily"))
        self.assertFalse(task1_categories_match(TASK1, "daily", "daily"))
        self.assertTrue(task1_categories_match(
            dict(TASK1, sim_major=""), "food", ""))

    def test_duplicate_item_rule(self):
        self.assertTrue(items_equal_allowed("food", "food"))
        self.assertFalse(items_equal_allowed("food", "daily"))


class HeadingAlignmentTests(unittest.TestCase):
    def test_shortest_error_wraps_across_pi(self):
        error = shortest_angular_error(
            math.radians(-179.0), math.radians(179.0))
        self.assertAlmostEqual(math.degrees(error), 2.0, places=6)

    def test_command_stops_inside_tolerance(self):
        self.assertEqual(
            heading_alignment_command(
                math.radians(1.0), math.radians(2.0), 1.5, 0.20, 0.25),
            0.0,
        )

    def test_command_uses_minimum_and_direction(self):
        command = heading_alignment_command(
            math.radians(-3.0), math.radians(2.0), 1.5, 0.20, 0.25)
        self.assertAlmostEqual(command, -0.20)

    def test_command_is_capped(self):
        command = heading_alignment_command(
            math.radians(30.0), math.radians(2.0), 1.5, 0.20, 0.25)
        self.assertAlmostEqual(command, 0.25)

    def test_invalid_speed_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            heading_alignment_command(0.2, 0.02, 1.5, 0.30, 0.25)


if __name__ == "__main__":
    unittest.main()
