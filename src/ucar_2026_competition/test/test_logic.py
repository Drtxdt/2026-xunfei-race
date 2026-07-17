#!/usr/bin/env python3
import json
import os
import sys
import unittest

PACKAGE_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if PACKAGE_SRC not in sys.path:
    sys.path.insert(0, PACKAGE_SRC)

from ucar_2026_competition.logic import (
    ConsecutiveTargetFilter,
    DirectedYawAccumulator,
    JsonLineBuffer,
    TRACK_CONFIG,
    TemporalTargetFilter,
    normalize_category,
    normalize_angle,
    parse_category,
    qr_values_from_payload,
    stage_sequence,
    traffic_decision_from_payload,
)


class CompetitionLogicTest(unittest.TestCase):
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

    def test_ocr_alias(self):
        self.assertEqual(normalize_category("electronic"), "electronics")
        self.assertEqual(normalize_category("食品加工车间"), "food")

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
        self.assertEqual(stage_sequence("full"), ("task1", "task2", "task3", "task4", "task5"))
        self.assertEqual(stage_sequence("task4"), ("task4",))


if __name__ == "__main__":
    unittest.main()
