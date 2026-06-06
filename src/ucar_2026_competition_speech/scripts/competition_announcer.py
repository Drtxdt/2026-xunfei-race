#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serialize all competition announcements through the robot's legacy /speak TTS."""

from __future__ import annotations

import json
import threading
import time
from typing import Dict

import rospy
from std_msgs.msg import String
from ucar_2026_competition_speech.srv import Announce, AnnounceResponse

from ucar_2026_competition_speech.speech_templates import (
    build_announcement,
    estimate_duration,
)


class CompetitionAnnouncer:
    def __init__(self) -> None:
        self.speak_topic = rospy.get_param("~speak_topic", "/speak")
        self.request_topic = rospy.get_param(
            "~request_topic", "/competition_speech/request"
        )
        self.status_topic = rospy.get_param(
            "~status_topic", "/competition_speech/status"
        )
        self.completed_topic = rospy.get_param(
            "~completed_topic", "/competition_speech/completed"
        )
        self.service_name = rospy.get_param(
            "~service_name", "/competition_speech/announce"
        )
        self.chars_per_second = float(rospy.get_param("~chars_per_second", 3.0))
        self.startup_sec = float(rospy.get_param("~startup_sec", 1.0))
        self.tail_sec = float(rospy.get_param("~tail_sec", 1.0))
        self.subscriber_wait_sec = float(rospy.get_param("~subscriber_wait_sec", 3.0))
        self.duplicate_window_sec = float(rospy.get_param("~duplicate_window_sec", 5.0))
        self.finish_status_topic = rospy.get_param("~finish_status_topic", "").strip()
        self.finish_status_value = rospy.get_param("~finish_status_value", "finish").strip()

        self.lock = threading.Lock()
        self.last_text = ""
        self.last_started_at = 0.0
        self.finish_announced = False

        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=10)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10)
        self.completed_pub = rospy.Publisher(self.completed_topic, String, queue_size=10)
        rospy.Subscriber(self.request_topic, String, self.request_cb, queue_size=10)
        if self.finish_status_topic:
            rospy.Subscriber(
                self.finish_status_topic, String, self.finish_status_cb, queue_size=10
            )
        rospy.Service(self.service_name, Announce, self.service_cb)

        rospy.loginfo(
            "competition_announcer ready: service=%s request=%s speak=%s",
            self.service_name,
            self.request_topic,
            self.speak_topic,
        )

    def publish_status(self, state: str, event: str, text: str, duration: float) -> None:
        payload: Dict[str, object] = {
            "stamp": time.time(),
            "state": state,
            "event": event,
            "text": text,
            "estimated_duration": duration,
        }
        self.status_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        )

    def announce(
        self,
        event: str,
        item: str = "",
        workshop: str = "",
        decision: str = "",
        text: str = "",
        wait: bool = True,
    ):
        try:
            event, speech_text = build_announcement(
                event, item=item, workshop=workshop, decision=decision, text=text
            )
        except ValueError as exc:
            return False, "", 0.0, str(exc)

        duration = estimate_duration(
            speech_text,
            chars_per_second=self.chars_per_second,
            startup_sec=self.startup_sec,
            tail_sec=self.tail_sec,
        )

        with self.lock:
            now = time.time()
            if (
                speech_text == self.last_text
                and now - self.last_started_at < self.duplicate_window_sec
            ):
                return True, speech_text, duration, "duplicate announcement ignored"

            deadline = now + self.subscriber_wait_sec
            while self.speak_pub.get_num_connections() == 0 and time.time() < deadline:
                if rospy.is_shutdown():
                    return False, speech_text, duration, "ROS shutdown"
                rospy.sleep(0.05)
            if self.speak_pub.get_num_connections() == 0:
                rospy.logwarn("No subscriber on %s; publishing for logs anyway", self.speak_topic)

            self.last_text = speech_text
            self.last_started_at = time.time()
            self.publish_status("speaking", event, speech_text, duration)
            rospy.loginfo("Competition announcement [%s]: %s", event, speech_text)
            self.speak_pub.publish(String(data=speech_text))

            if wait:
                rospy.sleep(duration)
                self.publish_status("completed", event, speech_text, duration)
                self.completed_pub.publish(String(data=event))
                rospy.loginfo("Competition announcement completed [%s]", event)

        return True, speech_text, duration, "completed" if wait else "published"

    def service_cb(self, req) -> AnnounceResponse:
        success, text, duration, message = self.announce(
            req.event,
            item=req.item,
            workshop=req.workshop,
            decision=req.decision,
            text=req.text,
            wait=req.wait,
        )
        return AnnounceResponse(success, text, duration, message)

    def request_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.announce(
                str(data.get("event", "")),
                item=str(data.get("item", "")),
                workshop=str(data.get("workshop", "")),
                decision=str(data.get("decision", "")),
                text=str(data.get("text", "")),
                wait=bool(data.get("wait", True)),
            )
        except Exception as exc:
            rospy.logerr("Invalid competition speech request: %s", exc)

    def finish_status_cb(self, msg: String) -> None:
        if self.finish_announced or msg.data.strip() != self.finish_status_value:
            return
        self.finish_announced = True
        threading.Thread(
            target=self.announce, args=("task5",), kwargs={"wait": True}, daemon=True
        ).start()


def main() -> None:
    rospy.init_node("competition_announcer")
    CompetitionAnnouncer()
    rospy.spin()


if __name__ == "__main__":
    main()
