#!/usr/bin/env python3
"""
直行+右转路径的停车调试节点
运行时 finish_auto_stop 自动设为 false，在期望停车点手动 Ctrl+C 保存快照
"""
import json
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from line_follow_straight_right import LineFollowStraightRightNode


class ParkingDebugStraightRightNode(LineFollowStraightRightNode):
    def __init__(self):
        if not rospy.has_param("~finish_auto_stop"):
            rospy.set_param("~finish_auto_stop", False)
        self.debug_output_path = rospy.get_param("~debug_output_path", "")
        self.debug_stop_publish_count = int(rospy.get_param("~debug_stop_publish_count", 10))
        self.debug_stop_publish_interval = float(rospy.get_param("~debug_stop_publish_interval", 0.03))
        super().__init__()
        rospy.on_shutdown(self.save_debug_snapshot)
        rospy.loginfo("Parking debug (straight+right) node ready. Ctrl+C to save snapshot.")

    def save_debug_snapshot(self):
        self.hard_stop_robot()
        for _ in range(max(0, self.debug_stop_publish_count)):
            self.cmd_pub.publish(Twist())
            time.sleep(max(0.0, self.debug_stop_publish_interval))

        snapshot = self.last_debug_snapshot or {}
        output = {
            "note": "Snapshot for straight+right parking tuning",
            "shutdown_time": datetime.now().isoformat(timespec="seconds"),
            "snapshot_available": bool(self.last_debug_snapshot),
            "node": {
                "status": self.status,
                "finish_phase": self.finish_phase,
                "finish_frames": self.finish_frames,
                "finish_parking_candidate_frames": self.finish_parking_candidate_frames,
                "finish_parking_reached_frames": self.finish_parking_reached_frames,
                "finish_parking_bottom_y_ratio": self.finish_parking_bottom_y_ratio,
                "finish_odom_active": self.finish_odom_active,
                "finish_odom_distance_m": self.finish_odom_distance_m,
                "fork_handled_count": self.fork_handled_count,
            },
            "snapshot": snapshot,
            "recommended_fields_to_send": [
                "snapshot.finish_candidate_box",
                "snapshot.finish_metrics",
                "snapshot.lane_center_px",
                "snapshot.image_width",
                "snapshot.image_height",
                "snapshot.finish_parking_bottom_y_ratio",
            ],
        }

        path = self.resolve_output_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.json_ready(output), f, indent=2, ensure_ascii=False)
            rospy.loginfo("Parking debug snapshot saved: %s", path)
            self.log_summary(snapshot)
        except OSError as exc:
            rospy.logerr("Failed to save snapshot: %s", exc)

    def resolve_output_path(self) -> str:
        if self.debug_output_path:
            return os.path.abspath(os.path.expanduser(self.debug_output_path))
        ros_home = os.environ.get("ROS_HOME", os.path.expanduser("~/.ros"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(ros_home, "ucar_2026_straight_right", "parking_debug_%s.json" % stamp)

    def log_summary(self, snapshot):
        if not snapshot:
            rospy.logwarn("No image snapshot available")
            return
        box = snapshot.get("finish_candidate_box")
        metrics = snapshot.get("finish_metrics", {})
        rospy.loginfo(
            "Parking debug summary: status=%s phase=%s finish_frames=%s odom_dist=%.3f box=%s metrics=%s",
            snapshot.get("status"),
            snapshot.get("finish_phase"),
            snapshot.get("finish_frames"),
            snapshot.get("finish_odom_distance_m"),
            box,
            metrics,
        )

    def json_ready(self, value: Any):
        if isinstance(value, dict):
            return {str(k): self.json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.json_ready(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value


def main():
    rospy.init_node("parking_debug_straight_right")
    ParkingDebugStraightRightNode()
    rospy.spin()


if __name__ == "__main__":
    main()