#!/usr/bin/env python3
import json
import math
import os
import sys
import unittest
import xml.etree.ElementTree as ET

PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from ucar_2026_competition.logic import (
    CATEGORY_LABELS,
    base_is_stopped,
    build_task1_instruction,
    ConsecutiveTargetFilter,
    DirectedYawAccumulator,
    JsonLineBuffer,
    TRACK_CONFIG,
    TemporalTargetFilter,
    normalize_category,
    normalize_angle,
    parse_category,
    parse_task1_categories,
    qr_values_from_payload,
    scan_sector_min,
    stage_sequence,
    task2_delivery_targets,
    task4_handoff_required,
    task4_start_action,
    traffic_decision_from_payload,
    task2_announcement_required,
    trigger_delivery_state,
)


class CompetitionLogicTest(unittest.TestCase):
    def test_competition_config_uses_faster_safe_qr_scan(self):
        config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "config", "competition.yaml"))
        with open(config_path, "r", encoding="utf-8") as stream:
            config = {}
            for line in stream:
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                config[key.strip()] = value.strip()
        self.assertEqual(float(config["qr_scan_angular_speed"]), 0.35)
        self.assertAlmostEqual(
            float(config["qr_scan_step_angle_rad"]), math.radians(30.0))
        self.assertEqual(float(config["qr_scan_settle_sec"]), 0.3)
        self.assertEqual(float(config["qr_decoder_warmup_sec"]), 1.2)
        self.assertGreaterEqual(float(config["qr_scan_result_grace_sec"]), 3.0)
        self.assertAlmostEqual(
            float(config["qr_scan_extra_sweep_angle_rad"]), math.radians(120.0))

    def test_normalize_angle(self):
        self.assertAlmostEqual(normalize_angle(3.0 * 3.141592653589793), -3.141592653589793)
        self.assertAlmostEqual(normalize_angle(-0.25), -0.25)

    def test_yaw_accumulator_crosses_pi_wrap(self):
        tracker = DirectedYawAccumulator(direction=1.0)
        tracker.reset(3.10)
        self.assertAlmostEqual(tracker.update(-3.10), 0.08318530717958605)
        self.assertGreater(tracker.update(-2.90), 0.28)

    def test_yaw_accumulator_ignores_reverse_jitter(self):
        tracker = DirectedYawAccumulator(direction=1.0)
        tracker.reset(0.0)
        self.assertAlmostEqual(tracker.update(0.2), 0.2)
        self.assertAlmostEqual(tracker.update(0.19), 0.2)
        self.assertAlmostEqual(tracker.update(0.21), 0.21)

    def test_voice_category(self):
        self.assertEqual(parse_category("小飞小飞，请取得食品类"), "food")
        self.assertEqual(parse_category("取得食品"), "food")
        self.assertEqual(parse_category("请取得日用品类"), "daily")
        self.assertEqual(parse_category("取得日用品"), "daily")
        self.assertEqual(parse_category("请取得电子产品类"), "electronics")
        self.assertEqual(parse_category("取得电子产品"), "electronics")

    def test_official_task1_command_keeps_physical_and_simulation_targets(self):
        command = (
            "小飞小飞，前往物品领取区，取得日用品类，放置在对应仓库，"
            "并领取仿真环境中需要的食品类放置在对应仓库"
        )
        self.assertEqual(parse_task1_categories(command), ("daily", "food"))
        self.assertEqual(
            build_task1_instruction("daily", "food"), command)

    def test_task2_visits_physical_then_distinct_simulation_workshop(self):
        self.assertEqual(
            task2_delivery_targets(
                ("daily", "牙膏", "日用品加工车间"),
                ("food", "香蕉", "食品加工车间"),
            ),
            (
                ("physical", "daily", "牙膏", "日用品加工车间"),
                ("simulation", "food", "香蕉", "食品加工车间"),
            ),
        )
        self.assertEqual(
            task2_delivery_targets(
                ("daily", "牙膏", "日用品加工车间"),
                ("daily", "牙膏", "日用品加工车间"),
            ),
            (("physical", "daily", "牙膏", "日用品加工车间"),),
        )

    def test_rear_lidar_sector_ignores_front_and_invalid_samples(self):
        ranges = [float("inf")] * 8
        ranges[0] = 0.10
        ranges[4] = 0.42
        ranges[5] = float("nan")
        self.assertAlmostEqual(
            scan_sector_min(
                ranges, 0.0, math.pi / 4.0, math.pi, math.pi / 4.0,
                0.05, 8.0),
            0.42,
        )

    def test_task2_config_requires_lidar_guarded_inter_visit_exit(self):
        config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "config", "competition.yaml"))
        with open(config_path, "r", encoding="utf-8") as stream:
            content = stream.read()
        self.assertIn("task2_inter_visit_reverse_distance_m: 0.32", content)
        self.assertIn("task2_inter_visit_rear_clearance_m: 0.28", content)

    def test_ocr_alias(self):
        self.assertEqual(normalize_category("electronic"), "electronics")
        self.assertEqual(normalize_category("食品加工车间"), "food")

    def test_all_three_task2_categories_are_data_driven(self):
        expected = {
            "food": ("食品", "食品加工车间"),
            "daily": ("日用品", "日用品加工车间"),
            "electronics": ("电子产品", "电子产品生产车间"),
        }
        self.assertEqual(CATEGORY_LABELS, expected)
        for category in expected:
            self.assertEqual(normalize_category(category), category)

    def test_ocr_consecutive_filter(self):
        target_filter = ConsecutiveTargetFilter(3)
        self.assertFalse(target_filter.push("food", "food"))
        self.assertFalse(target_filter.push("food", "daily"))
        self.assertFalse(target_filter.push("food", "food"))
        self.assertFalse(target_filter.push("food", "food"))
        self.assertTrue(target_filter.push("food", "food"))

    def test_ocr_temporal_filter_keeps_blank_frame(self):
        target_filter = TemporalTargetFilter(2, 1.5)
        self.assertFalse(target_filter.push("food", "food", now=10.0))
        self.assertFalse(target_filter.push("food", None, now=10.4))
        self.assertTrue(target_filter.push("food", "food", now=11.0))

    def test_ocr_temporal_filter_expires_and_resets_on_competitor(self):
        target_filter = TemporalTargetFilter(2, 1.5)
        self.assertFalse(target_filter.push("food", "food", now=10.0))
        self.assertFalse(target_filter.push("food", "food", now=12.0))
        self.assertFalse(target_filter.push("food", "daily", now=12.2))
        self.assertFalse(target_filter.push("food", "food", now=12.3))

    def test_reliable_trigger_requires_service_and_status_ack(self):
        self.assertEqual(trigger_delivery_state(False, "", 0.5, 2.0), "pending")
        self.assertEqual(trigger_delivery_state(True, "searching", 1.9, 2.0), "pending")
        self.assertEqual(
            trigger_delivery_state(True, "target_locked", 1.9, 2.0),
            "acknowledged",
        )

    def test_reliable_trigger_times_out_safely(self):
        self.assertEqual(trigger_delivery_state(False, "", 2.0, 2.0), "failed")
        self.assertEqual(trigger_delivery_state(True, "searching", 2.1, 2.0), "failed")

    def test_qr_values(self):
        payload = {"items": [
            {"raw": "u1", "result": "苹果", "ok": True},
            {"raw": "毛巾", "result": None, "ok": False},
            {"raw": "https://bad", "result": None, "ok": False},
        ]}
        self.assertEqual(qr_values_from_payload(payload), [("u1", "苹果"), ("毛巾", "毛巾")])

    def test_traffic_and_track_mapping(self):
        payload = {"consensus": {"active": True, "class_name": "green_straight"}}
        self.assertEqual(traffic_decision_from_payload(payload), "straight")
        self.assertEqual(TRACK_CONFIG["straight"][0], "stable_right_track_end_stop.launch")

    def test_fragmented_json_lines(self):
        decoder = JsonLineBuffer()
        self.assertEqual(decoder.feed(b'{"type":"sta'), [])
        events = decoder.feed(b'te","data":"OK"}\n' + json.dumps({"type": "done", "data": True}).encode() + b"\n")
        self.assertEqual(events[0]["data"], "OK")
        self.assertTrue(events[1]["data"])

    def test_state_machine_sequence(self):
        self.assertEqual(stage_sequence("full"), ("task1", "task2", "task4", "task5"))
        self.assertEqual(
            stage_sequence("full", enable_simulation=True),
            ("task1", "task2", "task3", "task4", "task5"),
        )
        self.assertEqual(stage_sequence("task1_task2"), ("task1", "task2"))
        self.assertEqual(stage_sequence("task3_task4"), ("task3", "task4"))
        self.assertEqual(
            stage_sequence("task3_task4", enable_simulation=False),
            ("task3", "task4"),
        )
        self.assertEqual(stage_sequence("task4_task5"), ("task4", "task5"))
        self.assertEqual(stage_sequence("task4"), ("task4",))

    def test_task4_handoff_covers_simulation_enabled_and_disabled_flows(self):
        self.assertTrue(task4_handoff_required("task2", "task4"))
        self.assertTrue(task4_handoff_required("task3", "task4"))
        self.assertFalse(task4_handoff_required("task2", "task3"))
        self.assertFalse(task4_handoff_required(None, "task4"))

    def test_full_launch_exposes_optional_simulation_parameter(self):
        launch_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "launch", "full_competition.launch"))
        root = ET.parse(launch_path).getroot()
        launch_args = {item.attrib["name"]: item.attrib.get("default")
                       for item in root.findall("arg")}
        self.assertEqual(launch_args["enable_simulation"], "true")

        flow_include = next(
            item for item in root.findall("include")
            if "flow_node.launch" in item.attrib.get("file", ""))
        flow_args = {item.attrib["name"]: item.attrib.get("value")
                     for item in flow_include.findall("arg")}
        self.assertEqual(flow_args["enable_simulation"], "$(arg enable_simulation)")

        flow_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "launch", "flow_node.launch"))
        flow_root = ET.parse(flow_path).getroot()
        flow_launch_args = {item.attrib["name"]: item.attrib.get("default")
                            for item in flow_root.findall("arg")}
        self.assertEqual(flow_launch_args["enable_simulation"], "true")
        flow_params = {item.attrib["name"]: item.attrib.get("value")
                       for item in flow_root.find("node").findall("param")}
        self.assertEqual(
            flow_params["enable_simulation"], "$(arg enable_simulation)")
        self.assertEqual(
            flow_params["sim_target_category"], "$(arg sim_target_category)")
        self.assertEqual(
            flow_params["coverage_rotation_min_clearance"],
            "$(arg coverage_rotation_min_clearance)")

    def test_task4_start_action_supports_manual_stop_line_start(self):
        self.assertEqual(task4_start_action(True, False), "detect")
        self.assertEqual(task4_start_action(False, True), "approach")
        with self.assertRaises(ValueError):
            task4_start_action(False, False)

    def test_task4_task5_launch_starts_at_stop_line_and_keeps_one_flow(self):
        launch_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "launch", "task4_task5.launch"))
        self.assertTrue(os.path.exists(launch_path), launch_path)
        root = ET.parse(launch_path).getroot()

        launch_args = {item.attrib["name"]: item.attrib.get("default")
                       for item in root.findall("arg")}
        self.assertEqual(launch_args["start_external_voice"], "true")

        core_include = next(
            item for item in root.findall("include")
            if "common_core.launch" in item.attrib.get("file", ""))
        core_args = {item.attrib["name"]: item.attrib.get("value")
                     for item in core_include.findall("arg")}
        self.assertEqual(core_args["start_nav"], "true")
        self.assertEqual(core_args["start_camera"], "true")
        self.assertEqual(core_args["start_speech"], "true")

        flow_include = next(
            item for item in root.findall("include")
            if "flow_node.launch" in item.attrib.get("file", ""))
        flow_args = {item.attrib["name"]: item.attrib.get("value")
                     for item in flow_include.findall("arg")}
        self.assertEqual(flow_args["start_stage"], "task4_task5")
        self.assertEqual(flow_args["skip_task4_stop_line_approach"], "true")

        flow_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "launch", "flow_node.launch"))
        flow_root = ET.parse(flow_path).getroot()
        flow_launch_args = {item.attrib["name"]: item.attrib.get("default")
                            for item in flow_root.findall("arg")}
        self.assertEqual(
            flow_launch_args["skip_task4_stop_line_approach"], "false")
        flow_node = flow_root.find("node")
        flow_params = {item.attrib["name"]: item.attrib.get("value")
                       for item in flow_node.findall("param")}
        self.assertEqual(
            flow_params["skip_task4_stop_line_approach"],
            "$(arg skip_task4_stop_line_approach)",
        )

    def test_task3_task4_launch_preserves_localization(self):
        launch_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "launch", "task3_task4.launch"))
        root = ET.parse(launch_path).getroot()
        core_include = next(
            item for item in root.findall("include")
            if "common_core.launch" in item.attrib.get("file", ""))
        core_args = {item.attrib["name"]: item.attrib.get("value")
                     for item in core_include.findall("arg")}
        self.assertEqual(core_args["start_nav"], "true")
        flow_include = next(
            item for item in root.findall("include")
            if "flow_node.launch" in item.attrib.get("file", ""))
        flow_args = {item.attrib["name"]: item.attrib.get("value")
                     for item in flow_include.findall("arg")}
        self.assertEqual(flow_args["start_stage"], "task3_task4")
        self.assertEqual(flow_args["navigator_publish_initial_pose"], "false")

    def test_task_handoff_stationary_gate(self):
        self.assertTrue(base_is_stopped(0.005, -0.005, 0.01))
        self.assertFalse(base_is_stopped(0.02, 0.0, 0.0))
        self.assertFalse(base_is_stopped(0.0, 0.0, 0.03))

    def test_task2_announcement_is_allowed_once_and_only_after_arrival(self):
        self.assertFalse(task2_announcement_required("parking_verifying", False))
        self.assertTrue(task2_announcement_required("arrived", False))
        self.assertFalse(task2_announcement_required("arrived", True))

    def test_task1_task2_launch_preserves_localization(self):
        launch_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "launch", "task1_task2.launch"))
        root = ET.parse(launch_path).getroot()
        docking_timeout_arg = root.find("arg[@name='parking_docking_timeout_sec']")
        self.assertIsNotNone(docking_timeout_arg)
        self.assertEqual(docking_timeout_arg.attrib.get("default"), "25.0")
        flow_include = next(
            item for item in root.findall("include")
            if "flow_node.launch" in item.attrib.get("file", ""))
        args = {item.attrib["name"]: item.attrib.get("value")
                for item in flow_include.findall("arg")}
        self.assertEqual(args["start_stage"], "task1_task2")
        self.assertEqual(args["navigator_publish_initial_pose"], "false")
        self.assertEqual(args["parking_recenter_initial_wait_sec"],
                         "$(arg parking_recenter_initial_wait_sec)")
        self.assertEqual(args["coverage_goal_soft_timeout_sec"],
                         "$(arg coverage_goal_soft_timeout_sec)")
        self.assertEqual(args["coverage_goal_hard_timeout_sec"],
                         "$(arg coverage_goal_hard_timeout_sec)")


if __name__ == "__main__":
    unittest.main()
