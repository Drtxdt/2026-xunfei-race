#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run subtask 1 once: wakeup -> QR area -> QR items -> LLM -> TTS."""

from __future__ import annotations

import json
import time
from collections import OrderedDict

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import String
from ucar_2026_competition_speech.srv import Announce
from ucar_2026_smart_factory_llm.srv import ReasonPickupOrder


def get_bool_param(name: str, default: bool) -> bool:
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class Task1FullOnce:
    def __init__(self) -> None:
        self.wait_for_wakeup = get_bool_param("~wait_for_wakeup", True)
        self.navigate_to_qr = get_bool_param("~navigate_to_qr", True)
        self.wakeup_topic = rospy.get_param("~wakeup_topic", "/wakeup")
        self.qr_topic = rospy.get_param("~qr_topic", "/qr_code_data")
        self.service_name = rospy.get_param(
            "~service_name", "/smart_factory_llm/reason_pickup_order"
        )
        self.speak_topic = rospy.get_param("~speak_topic", "/speak")
        self.announce_service = rospy.get_param(
            "~announce_service", "/competition_speech/announce"
        )
        self.announce_service_timeout_sec = float(
            rospy.get_param("~announce_service_timeout_sec", 2.0)
        )
        self.voice_instruction = rospy.get_param("~voice_instruction", "").strip()
        self.expected_count = int(rospy.get_param("~expected_count", 3))
        self.timeout_sec = float(rospy.get_param("~timeout_sec", 45.0))
        self.move_base_timeout_sec = float(rospy.get_param("~move_base_timeout_sec", 60.0))
        self.wait_per_char_sec = float(rospy.get_param("~wait_per_char_sec", 1.0 / 3.0))
        self.min_wait_sec = float(rospy.get_param("~min_wait_sec", 2.0))

        self.items_by_key: OrderedDict[str, str] = OrderedDict()
        self.started_at = None
        self.done = False

        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=1)

    def run(self) -> None:
        if not self.voice_instruction:
            rospy.logerr("Missing required private param: voice_instruction")
            return

        if self.wait_for_wakeup:
            rospy.loginfo("Waiting for wakeup topic: %s", self.wakeup_topic)
            rospy.wait_for_message(self.wakeup_topic, String)
            rospy.loginfo("Wakeup received")

        if self.navigate_to_qr and not self.goto_qr_area():
            return

        rospy.loginfo("Collecting %d QR item(s) from %s", self.expected_count, self.qr_topic)
        self.started_at = time.time()
        rospy.Subscriber(self.qr_topic, String, self.qr_cb, queue_size=10)
        rospy.Timer(rospy.Duration(0.5), self.timer_cb)
        rospy.spin()

    def goto_qr_area(self) -> bool:
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = rospy.get_param("~goal_frame", "map")
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = float(rospy.get_param("~qr_x", 1.00))
        goal.target_pose.pose.position.y = float(rospy.get_param("~qr_y", 0.51))
        goal.target_pose.pose.position.z = float(rospy.get_param("~qr_z", 0.0))
        goal.target_pose.pose.orientation.x = float(rospy.get_param("~qr_qx", 0.0))
        goal.target_pose.pose.orientation.y = float(rospy.get_param("~qr_qy", 0.0))
        goal.target_pose.pose.orientation.z = float(rospy.get_param("~qr_qz", 1.0))
        goal.target_pose.pose.orientation.w = float(rospy.get_param("~qr_qw", 0.0))

        client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        if not client.wait_for_server(rospy.Duration(self.move_base_timeout_sec)):
            rospy.logerr("move_base action server is not available")
            return False

        rospy.loginfo(
            "Sending QR area goal: x=%.2f y=%.2f qz=%.2f qw=%.2f",
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
            goal.target_pose.pose.orientation.z,
            goal.target_pose.pose.orientation.w,
        )
        client.send_goal(goal)
        if not client.wait_for_result(rospy.Duration(self.move_base_timeout_sec)):
            client.cancel_goal()
            rospy.logerr("Timed out navigating to QR area")
            return False

        state = client.get_state()
        if state != GoalStatus.SUCCEEDED:
            rospy.logerr("Failed to reach QR area, move_base state=%s", state)
            return False

        rospy.loginfo("Reached QR area")
        return True

    def qr_cb(self, msg: String) -> None:
        if self.done:
            return

        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            rospy.logwarn("Ignore invalid QR JSON: %s", exc)
            return

        for item in payload.get("items", []):
            raw = str(item.get("raw") or "").strip()
            result = str(item.get("result") or "").strip()
            ok = bool(item.get("ok"))
            if not result and raw and not raw.startswith(("http://", "https://")):
                result = raw
                ok = True

            key = raw or result
            if not key or not result or not ok:
                continue
            if key in self.items_by_key:
                continue

            self.items_by_key[key] = result
            rospy.loginfo(
                "Accepted QR item %d/%d: %s",
                len(self.items_by_key),
                self.expected_count,
                result,
            )

            if len(self.items_by_key) >= self.expected_count:
                self.call_llm_and_speak()
                return

    def timer_cb(self, _event) -> None:
        if self.done or self.started_at is None:
            return
        if time.time() - self.started_at < self.timeout_sec:
            return

        rospy.logerr(
            "Timed out waiting for QR items: got %d/%d: %s",
            len(self.items_by_key),
            self.expected_count,
            list(self.items_by_key.values()),
        )
        self.done = True
        rospy.signal_shutdown("QR item collection timeout")

    def call_llm_and_speak(self) -> None:
        self.done = True
        items = list(self.items_by_key.values())
        item_a, item_b, item_c = items[:3]

        rospy.loginfo("Waiting for LLM service: %s", self.service_name)
        rospy.wait_for_service(self.service_name)
        reason_pickup = rospy.ServiceProxy(self.service_name, ReasonPickupOrder)

        rospy.loginfo("Calling LLM service with items: %s, %s, %s", item_a, item_b, item_c)
        res = reason_pickup(item_a, item_b, item_c, self.voice_instruction)
        if not res.success:
            rospy.logerr("LLM reasoning failed: %s", res.error_message)
            rospy.signal_shutdown("LLM reasoning failed")
            return

        rospy.loginfo(
            "Task1 result: pickup=%s/%s -> %s, simulation=%s/%s -> %s",
            res.pickup_item,
            res.pickup_major,
            res.pickup_workshop,
            res.sim_item,
            res.sim_major,
            res.sim_workshop,
        )

        speech_text = res.announcement_full.strip()
        if not speech_text:
            rospy.logerr("LLM result has empty announcement_full")
            rospy.signal_shutdown("empty speech text")
            return

        if not self.announce_task1(speech_text):
            rospy.sleep(1.0)
            rospy.logwarn(
                "Competition speech service unavailable; publishing directly to %s",
                self.speak_topic,
            )
            self.speak_pub.publish(String(data=speech_text))
            wait_sec = max(self.min_wait_sec, len(speech_text) * self.wait_per_char_sec)
            rospy.sleep(wait_sec)
        rospy.signal_shutdown("task1 completed")

    def announce_task1(self, speech_text: str) -> bool:
        try:
            rospy.wait_for_service(
                self.announce_service, timeout=self.announce_service_timeout_sec
            )
            announce = rospy.ServiceProxy(self.announce_service, Announce)
            response = announce("task1", "", "", "", speech_text, True)
            if not response.success:
                rospy.logerr("Competition announcement failed: %s", response.message)
                return False
            rospy.loginfo("Competition announcement completed: %s", response.speech_text)
            return True
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("Competition announcement service error: %s", exc)
            return False


def main() -> None:
    rospy.init_node("task1_full_once")
    Task1FullOnce().run()


if __name__ == "__main__":
    main()
