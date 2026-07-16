#!/usr/bin/env python3
import json
import unittest

from ucar_2026_competition.logic import (
    ConsecutiveTargetFilter,
    JsonLineBuffer,
    TRACK_CONFIG,
    normalize_category,
    parse_category,
    qr_values_from_payload,
    stage_sequence,
    traffic_decision_from_payload,
)


class CompetitionLogicTest(unittest.TestCase):
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
