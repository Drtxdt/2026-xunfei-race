#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Speak stable YOLO traffic-light consensus changes."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import rospy
from std_msgs.msg import String


DEFAULT_SPEECH_TEXT = {
    "red_light": "识别到红灯，请停车等待。",
    "green_straight": "识别到绿灯直行，请直行。",
    "green_left": "识别到绿灯左转，请左转。",
    "green_right": "识别到绿灯右转，请右转。",
}


class TrafficLightSpeakTestNode:
    def __init__(self) -> None:
        self.detections_topic = rospy.get_param("~detections_topic", "/traffic_light/detections")
        self.speak_topic = rospy.get_param("~speak_topic", "/speak")
        self.status_topic = rospy.get_param("~status_topic", "/traffic_light_speak_test/status")
        self.min_interval_sec = float(rospy.get_param("~min_interval_sec", 1.5))
        self.initial_publish_delay_sec = float(rospy.get_param("~initial_publish_delay_sec", 0.3))
        self.publish_retries = int(rospy.get_param("~publish_retries", 1))
        self.retry_interval_sec = float(rospy.get_param("~retry_interval_sec", 0.25))
        self.repeat_same = bool(rospy.get_param("~repeat_same", False))
        self.speak_on_inactive = bool(rospy.get_param("~speak_on_inactive", False))
        self.inactive_text = rospy.get_param("~inactive_text", "")
        self.speak_prefix = rospy.get_param("~speak_prefix", "")
        self.speak_suffix = rospy.get_param("~speak_suffix", "")

        self.speech_text = {
            class_name: rospy.get_param("~text_%s" % class_name, text)
            for class_name, text in DEFAULT_SPEECH_TEXT.items()
        }

        self.last_class: Optional[str] = None
        self.last_spoken_at = 0.0

        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(self.detections_topic, String, self.detections_cb, queue_size=10)

        rospy.loginfo(
            "traffic_light_speak_test: detections=%s speak=%s status=%s",
            self.detections_topic,
            self.speak_topic,
            self.status_topic,
        )
        self.publish_status("waiting")

    def detections_cb(self, msg: String) -> None:
        payload = self.parse_json(msg.data)
        if not isinstance(payload, dict):
            self.publish_status("ignored:bad_json")
            return

        consensus = payload.get("consensus")
        if not isinstance(consensus, dict):
            self.publish_status("ignored:no_consensus")
            return

        active = bool(consensus.get("active", False))
        class_name = consensus.get("class_name")

        if not active:
            if self.speak_on_inactive and self.inactive_text and self.last_class is not None:
                self.last_class = None
                self.publish_speech(self.inactive_text, "inactive")
            else:
                self.publish_status("ignored:inactive")
            return

        if not isinstance(class_name, str) or not class_name:
            self.publish_status("ignored:empty_class")
            return

        if class_name not in self.speech_text:
            self.publish_status("ignored:unknown_class:%s" % class_name)
            return

        if not self.repeat_same and class_name == self.last_class:
            self.publish_status("ignored:same_class:%s" % class_name)
            return

        self.last_class = class_name
        self.publish_speech(self.speech_text[class_name], class_name)

    def publish_speech(self, text: str, class_name: str) -> None:
        now = time.time()
        if now - self.last_spoken_at < self.min_interval_sec:
            self.publish_status("ignored:interval:%s" % class_name)
            return

        speech_text = "%s%s%s" % (self.speak_prefix, text, self.speak_suffix)
        self.last_spoken_at = now
        self.publish_status("speaking:%s" % class_name)
        rospy.loginfo("traffic_light_speak_test publish to %s: %s", self.speak_topic, speech_text)

        if self.initial_publish_delay_sec > 0.0:
            rospy.sleep(self.initial_publish_delay_sec)
        for idx in range(max(1, self.publish_retries)):
            self.speak_pub.publish(String(data=speech_text))
            if idx + 1 < self.publish_retries:
                rospy.sleep(self.retry_interval_sec)

    def publish_status(self, status: str) -> None:
        self.status_pub.publish(String(data=status))

    @staticmethod
    def parse_json(text: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(text, str):
            return None
        try:
            return json.loads(text)
        except Exception:
            return None


def main() -> None:
    rospy.init_node("traffic_light_speak_test_node")
    TrafficLightSpeakTestNode()
    rospy.spin()


if __name__ == "__main__":
    main()
