#!/usr/bin/env python3

import pathlib
import unittest
import xml.etree.ElementTree as ET


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "stable_right_track_end_stop.launch"
CONFIG_PATH = PACKAGE_ROOT / "config" / "stable_right_track_end_stop.yaml"
SOURCE_PATH = PACKAGE_ROOT / "src" / "stable_right_track_end_stop_node.cpp"


class StraightTrackContractTest(unittest.TestCase):
    def test_launch_uses_dedicated_straight_config_without_inline_tuning(self):
        root = ET.parse(str(LAUNCH_PATH)).getroot()
        launch_args = {item.attrib["name"]: item.attrib.get("default", "")
                       for item in root.findall("arg")}
        self.assertIn(
            "/config/stable_right_track_end_stop.yaml",
            launch_args["config_file"],
        )

        controller = next(
            node for node in root.findall("node")
            if node.attrib.get("type") == "stable_right_track_end_stop_node")
        inline_params = {item.attrib["name"] for item in controller.findall("param")}
        tuning_params = {
            "target_right_x", "base_speed", "curve_speed", "search_speed",
            "search_angular_speed", "lost_linear_speed", "lost_angular_speed",
            "kp", "kd", "error_alpha", "max_angular_speed",
        }
        self.assertTrue(tuning_params.isdisjoint(inline_params))

    def test_straight_config_is_fail_safe_and_tracks_right_boundary(self):
        self.assertTrue(CONFIG_PATH.exists(), str(CONFIG_PATH))
        config = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("right_line_offset_px: 200", config)
        self.assertIn("lost_timeout: 0.80", config)
        self.assertIn("lost_linear_speed: 0.06", config)
        self.assertIn("lost_angular_speed: 0.00", config)
        self.assertIn("stop_on_lost: true", config)

    def test_node_selects_only_rightmost_segment_and_stops_after_line_loss(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.assertIn("const Segment& rightmost = segments.back();", source)
        self.assertNotIn("heading_angular", source)
        self.assertIn('private_nh_.param("lost_timeout"', source)
        self.assertIn('private_nh_.param("stop_on_lost"', source)
        self.assertIn('setStatus("stable_right_lost_stop")', source)
        self.assertIn(
            "end_forward_distance_m_ = std::max(0.0, end_forward_distance_m_ - 0.05);",
            source,
        )


if __name__ == "__main__":
    unittest.main()
