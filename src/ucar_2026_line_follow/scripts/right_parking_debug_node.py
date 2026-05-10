#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from right_line_follow_node import RightLineFollowNode


class RightParkingDebugNode(RightLineFollowNode):
    def __init__(self):
        if not rospy.has_param("~finish_auto_stop"):
            rospy.set_param("~finish_auto_stop", False)
        self.debug_output_path = rospy.get_param("~debug_output_path", "")
        self.debug_stop_publish_count = int(rospy.get_param("~debug_stop_publish_count", 12))
        self.debug_stop_publish_interval = float(rospy.get_param("~debug_stop_publish_interval", 0.03))
        super().__init__()
        rospy.on_shutdown(self.save_debug_snapshot)
        rospy.loginfo("right_parking_debug_node ready. Press Ctrl+C at the desired right-lane stop position.")

    def save_debug_snapshot(self):
        self.hard_stop_robot()
        for _ in range(max(0, self.debug_stop_publish_count)):
            self.cmd_pub.publish(Twist())
            time.sleep(max(0.0, self.debug_stop_publish_interval))

        snapshot = self.last_debug_snapshot or {}
        output = {
            "note": "Snapshot saved when right_parking_debug_node shut down. Use this to tune visual parking stop thresholds for right-lane follow.",
            "shutdown_time": datetime.now().isoformat(timespec="seconds"),
            "snapshot_available": bool(snapshot),
            "node": {
                "status": self.status,
                "startup_phase": self.startup_phase,
                "finish_detection_enabled": self.finish_detection_enabled,
                "finish_auto_stop": self.finish_auto_stop,
                "finish_frames": self.finish_frames,
                "finish_confirm_frames": self.finish_confirm_frames,
                "last_target_center": self.last_target_center,
                "last_error_px": self.last_error_px,
                "last_cmd_linear": self.last_cmd_linear,
                "last_cmd_angular": self.last_cmd_angular,
                "last_parking_detected": self.last_parking_result.detected,
                "last_parking_full_box_detected": self.last_parking_result.full_box_detected,
                "last_parking_stop_pose_detected": self.last_parking_result.stop_pose_detected,
                "last_parking_closed_shape_detected": self.last_parking_result.closed_shape_detected,
                "last_parking_box": self.last_parking_result.box,
                "last_parking_closed_shape_box": self.last_parking_result.closed_shape_box,
                "last_parking_closed_shape_score": self.last_parking_result.closed_shape_score,
                "last_parking_closed_top_ratio": self.last_parking_result.closed_top_ratio,
                "last_parking_closed_bottom_ratio": self.last_parking_result.closed_bottom_ratio,
                "last_parking_closed_left_ratio": self.last_parking_result.closed_left_ratio,
                "last_parking_closed_right_ratio": self.last_parking_result.closed_right_ratio,
                "last_parking_horizontal_rows": self.last_parking_result.horizontal_rows,
                "last_parking_horizontal_left_x_ratio": self.last_parking_result.horizontal_left_x_ratio,
                "last_parking_horizontal_right_x_ratio": self.last_parking_result.horizontal_right_x_ratio,
                "last_parking_bottom_y_ratio": self.last_parking_result.bottom_y_ratio,
                "last_parking_horizontal_width_ratio": self.last_parking_result.horizontal_width_ratio,
                "last_parking_vertical_left_height_ratio": self.last_parking_result.vertical_left_height_ratio,
                "last_parking_vertical_right_height_ratio": self.last_parking_result.vertical_right_height_ratio,
            },
            "snapshot": snapshot,
            "recommended_fields_to_send": [
                "snapshot.finish_candidate_box",
                "snapshot.finish_metrics",
                "snapshot.lane_center_px",
                "snapshot.target_center_px",
                "snapshot.image_width",
                "snapshot.image_height",
                "snapshot.roi_origin_y",
                "snapshot.status",
                "snapshot.startup_phase",
                "snapshot.finish_detection_enabled",
                "snapshot.finish_frames",
                "snapshot.observations",
            ],
        }

        path = self.resolve_output_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.json_ready(output), handle, indent=2, ensure_ascii=False)
            rospy.loginfo("right parking debug snapshot saved: %s", path)
            self.log_summary(snapshot)
        except OSError as exc:
            rospy.logerr("failed to save right parking debug snapshot to %s: %s", path, exc)

    def resolve_output_path(self) -> str:
        if self.debug_output_path:
            return os.path.abspath(os.path.expanduser(self.debug_output_path))
        ros_home = os.environ.get("ROS_HOME", os.path.expanduser("~/.ros"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(ros_home, "ucar_2026_line_follow", "right_parking_debug_%s.json" % stamp)

    def log_summary(self, snapshot):
        if not snapshot:
            rospy.logwarn("no right parking image snapshot was available before shutdown")
            return

        rospy.loginfo(
            "right parking debug summary: status=%s phase=%s finish_frames=%s box=%s metrics=%s lane_center=%s target=%s",
            snapshot.get("status"),
            snapshot.get("startup_phase"),
            snapshot.get("finish_frames"),
            snapshot.get("finish_candidate_box"),
            snapshot.get("finish_metrics"),
            snapshot.get("lane_center_px"),
            snapshot.get("target_center_px"),
        )

    def json_ready(self, value: Any):
        if isinstance(value, dict):
            return {str(key): self.json_ready(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.json_ready(item) for item in value]
        if isinstance(value, tuple):
            return [self.json_ready(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value


def main():
    rospy.init_node("right_parking_debug_node")
    RightParkingDebugNode()
    rospy.spin()


if __name__ == "__main__":
    main()
