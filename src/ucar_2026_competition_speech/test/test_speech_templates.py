#!/usr/bin/env python3

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ucar_2026_competition_speech.speech_templates import (
    build_announcement,
    estimate_duration,
)


class SpeechTemplateTest(unittest.TestCase):
    def test_all_official_templates(self):
        self.assertEqual(build_announcement("task1", text="推理结果。")[1], "推理结果。")
        self.assertEqual(
            build_announcement("task2", item="香蕉", workshop="食品加工车间")[1],
            "已将香蕉放入食品加工车间",
        )
        self.assertEqual(
            build_announcement("task3", item="毛巾", workshop="日用品加工车间")[1],
            "仿真任务已完成，已将毛巾放入日用品加工车间",
        )
        self.assertEqual(build_announcement("task4", decision="left")[1], "左转")
        self.assertEqual(build_announcement("task4", decision="红灯")[1], "停止")
        self.assertEqual(build_announcement("task5")[1], "任务完成")

    def test_required_fields(self):
        with self.assertRaises(ValueError):
            build_announcement("task2", item="香蕉")
        with self.assertRaises(ValueError):
            build_announcement("task4", decision="unknown")
        with self.assertRaises(ValueError):
            build_announcement("task1")

    def test_duration_is_conservative(self):
        self.assertGreater(estimate_duration("任务完成"), 2.0)
        self.assertGreater(
            estimate_duration("仿真任务已完成，已将毛巾放入日用品加工车间"),
            estimate_duration("任务完成"),
        )


if __name__ == "__main__":
    unittest.main()
