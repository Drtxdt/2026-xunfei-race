#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect three QR item names, call the LLM service, then publish TTS text."""

from __future__ import annotations

import json
import time
from collections import OrderedDict

import rospy
from std_msgs.msg import String
from ucar_2026_competition_speech.srv import Announce
from ucar_2026_smart_factory_llm.srv import ReasonPickupOrder


class QRToLLMSpeakOnce:
    def __init__(self) -> None:
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
        self.timeout_sec = float(rospy.get_param("~timeout_sec", 30.0))
        self.wait_per_char_sec = float(rospy.get_param("~wait_per_char_sec", 1.0 / 3.0))
        self.min_wait_sec = float(rospy.get_param("~min_wait_sec", 2.0))

        self.items_by_key: OrderedDict[str, str] = OrderedDict()
        self.started_at = time.time()
        self.done = False

        if not self.voice_instruction:
            rospy.logerr("Missing required private param: voice_instruction")
            self.done = True
            return

        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=1)
        self.sub = rospy.Subscriber(self.qr_topic, String, self.qr_cb, queue_size=10)
        rospy.Timer(rospy.Duration(0.5), self.timer_cb)

        rospy.loginfo("QR topic: %s", self.qr_topic)
        rospy.loginfo("LLM service: %s", self.service_name)
        rospy.loginfo("Speak topic: %s", self.speak_topic)
        rospy.loginfo("Waiting for %d stable QR items...", self.expected_count)

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
        if self.done:
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

        speech_text = res.announcement_full.strip()
        if not speech_text:
            rospy.logerr("LLM result has empty announcement_full")
            rospy.signal_shutdown("empty speech text")
            return

        if not self.announce_task1(speech_text):
            rospy.sleep(1.0)
            rospy.logwarn("Publishing directly to fallback TTS topic %s", self.speak_topic)
            self.speak_pub.publish(String(data=speech_text))
            wait_sec = max(self.min_wait_sec, len(speech_text) * self.wait_per_char_sec)
            rospy.sleep(wait_sec)
        rospy.signal_shutdown("QR to LLM speak completed")

    def announce_task1(self, speech_text: str) -> bool:
        try:
            rospy.wait_for_service(
                self.announce_service, timeout=self.announce_service_timeout_sec
            )
            response = rospy.ServiceProxy(self.announce_service, Announce)(
                "task1", "", "", "", speech_text, True
            )
            if response.success:
                rospy.loginfo("Competition announcement completed: %s", response.speech_text)
                return True
            rospy.logerr("Competition announcement failed: %s", response.message)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("Competition announcement service error: %s", exc)
        return False


def main() -> None:
    rospy.init_node("qr_to_llm_speak_once")
    node = QRToLLMSpeakOnce()
    if not node.done:
        rospy.spin()


if __name__ == "__main__":
    main()
