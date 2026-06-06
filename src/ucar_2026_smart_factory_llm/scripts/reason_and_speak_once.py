#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Call the task-1 LLM service once, then publish the result to /speak."""

from __future__ import annotations

import rospy
from std_msgs.msg import String
from ucar_2026_competition_speech.srv import Announce
from ucar_2026_smart_factory_llm.srv import ReasonPickupOrder


def main() -> None:
    rospy.init_node("smart_factory_llm_reason_and_speak_once")

    service_name = rospy.get_param("~service_name", "/smart_factory_llm/reason_pickup_order")
    speak_topic = rospy.get_param("~speak_topic", "/speak")
    announce_service = rospy.get_param("~announce_service", "/competition_speech/announce")
    announce_service_timeout_sec = float(rospy.get_param("~announce_service_timeout_sec", 2.0))
    item_a = rospy.get_param("~item_a", "").strip()
    item_b = rospy.get_param("~item_b", "").strip()
    item_c = rospy.get_param("~item_c", "").strip()
    voice_instruction = rospy.get_param("~voice_instruction", "").strip()
    wait_per_char_sec = float(rospy.get_param("~wait_per_char_sec", 1.0 / 3.0))
    min_wait_sec = float(rospy.get_param("~min_wait_sec", 2.0))

    missing = [
        name
        for name, value in (
            ("item_a", item_a),
            ("item_b", item_b),
            ("item_c", item_c),
            ("voice_instruction", voice_instruction),
        )
        if not value
    ]
    if missing:
        rospy.logerr("Missing required private params: %s", ", ".join(missing))
        return

    rospy.loginfo("Waiting for LLM service: %s", service_name)
    rospy.wait_for_service(service_name)
    reason_pickup = rospy.ServiceProxy(service_name, ReasonPickupOrder)

    rospy.loginfo("Calling LLM service with items: %s, %s, %s", item_a, item_b, item_c)
    res = reason_pickup(item_a, item_b, item_c, voice_instruction)
    if not res.success:
        rospy.logerr("LLM reasoning failed: %s", res.error_message)
        return

    speech_text = res.announcement_full.strip()
    if not speech_text:
        rospy.logerr("LLM result has empty announcement_full")
        return

    try:
        rospy.wait_for_service(announce_service, timeout=announce_service_timeout_sec)
        response = rospy.ServiceProxy(announce_service, Announce)(
            "task1", "", "", "", speech_text, True
        )
        if response.success:
            rospy.loginfo("Competition announcement completed: %s", response.speech_text)
            return
        rospy.logerr("Competition announcement failed: %s", response.message)
    except (rospy.ROSException, rospy.ServiceException) as exc:
        rospy.logwarn("Competition announcement service error: %s", exc)

    pub = rospy.Publisher(speak_topic, String, queue_size=1)
    rospy.sleep(1.0)
    rospy.logwarn("Publishing directly to fallback TTS topic %s", speak_topic)
    pub.publish(String(data=speech_text))
    rospy.sleep(max(min_wait_sec, len(speech_text) * wait_per_char_sec))


if __name__ == "__main__":
    main()
