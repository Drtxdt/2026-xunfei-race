#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from line_follow_node import LineFollowNode


class ParkingDebugNode(LineFollowNode):
    def __init__(self):
        if not rospy.has_param("~finish_auto_stop"):
            rospy.set_param("~finish_auto_stop", False)
        self.debug_output_path = rospy.get_param("~debug_output_path", "")
        self.debug_stop_publish_count = int(rospy.get_param("~debug_stop_publish_count", 10))
        self.debug_stop_publish_interval = float(rospy.get_param("~debug_stop_publish_interval", 0.03))
        super().__init__()
        rospy.on_shutdown(self.save_debug_snapshot)
        rospy.loginfo("parking_debug_node ready. Press Ctrl+C at the desired stop position.")

    def save_debug_snapshot(self):
        self.hard_stop_robot()
        for _ in range(max(0, self.debug_stop_publish_count)):
            self.cmd_pub.publish(Twist())
            time.sleep(max(0.0, self.debug_stop_publish_interval))

        snapshot = self.last_debug_snapshot or {}
        output = {
            "note": "Snapshot saved when parking_debug_node shut down. Use this to tune visual parking stop thresholds.",
            "shutdown_time": datetime.now().isoformat(timespec="seconds"),
            "snapshot_available": bool(self.last_debug_snapshot),
            "node": {
                "status": self.status,
                "finish_phase": self.finish_phase,
                "finish_detection_enabled": self.finish_detection_enabled,
                "finish_frames": self.finish_frames,
                "finish_confirm_frames": self.finish_confirm_frames,
                "finish_parking_candidate_frames": self.finish_parking_candidate_frames,
                "finish_parking_reached_frames": self.finish_parking_reached_frames,
                "finish_parking_bottom_y_ratio": self.finish_parking_bottom_y_ratio,
                "finish_odom_active": self.finish_odom_active,
                "finish_odom_start_xy": self.finish_odom_start_xy,
                "finish_odom_current_xy": self.current_odom_xy,
                "finish_odom_distance_m": self.finish_odom_distance_m,
            },
            "snapshot": snapshot,
            "recommended_fields_to_send": [
                "snapshot.finish_candidate_box",
                "snapshot.finish_metrics",
                "snapshot.lane_center_px",
                "snapshot.image_width",
                "snapshot.image_height",
                "snapshot.roi_origin_y",
                "snapshot.status",
                "snapshot.finish_phase",
                "snapshot.finish_frames",
                "snapshot.finish_parking_bottom_y_ratio",
                "snapshot.finish_odom_start_xy",
                "snapshot.finish_odom_current_xy",
                "snapshot.finish_odom_distance_m",
            ],
        }

        path = self.resolve_output_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.json_ready(output), handle, indent=2, ensure_ascii=False)
            rospy.loginfo("parking debug snapshot saved: %s", path)
            self.log_summary(snapshot)
        except OSError as exc:
            rospy.logerr("failed to save parking debug snapshot to %s: %s", path, exc)

    def resolve_output_path(self) -> str:
        if self.debug_output_path:
            return os.path.abspath(os.path.expanduser(self.debug_output_path))
        ros_home = os.environ.get("ROS_HOME", os.path.expanduser("~/.ros"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(ros_home, "ucar_2026_line_follow", "parking_debug_%s.json" % stamp)

    def log_summary(self, snapshot):
        if not snapshot:
            rospy.logwarn("no image snapshot was available before shutdown")
            return

        box = snapshot.get("finish_candidate_box")
        metrics = snapshot.get("finish_metrics", {})
        rospy.loginfo(
            "parking debug summary: status=%s phase=%s finish_frames=%s odom_distance=%s box=%s metrics=%s",
            snapshot.get("status"),
            snapshot.get("finish_phase"),
            snapshot.get("finish_frames"),
            snapshot.get("finish_odom_distance_m"),
            box,
            metrics,
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
    rospy.init_node("parking_debug_node")
    ParkingDebugNode()
    rospy.spin()


if __name__ == "__main__":
    main()
