#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean QR JSON results, call Spark LLM for reasoning, and publish to /speak."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any, Iterable, Optional, Tuple

import rospy
from std_msgs.msg import String
from ucar_2026_smart_factory_llm.srv import ReasonPickupOrder


class QRSpeakTestNode:
    def __init__(self) -> None:
        self.qr_topic = rospy.get_param("~qr_topic", "/qr_code_data")
        self.speak_topic = rospy.get_param("~speak_topic", "/speak")
        self.status_topic = rospy.get_param("~status_topic", "/qr_speak_test/status")
        self.speak_prefix = rospy.get_param("~speak_prefix", "")
        self.speak_suffix = rospy.get_param("~speak_suffix", "")
        self.min_interval_sec = float(rospy.get_param("~min_interval_sec", 1.5))
        self.publish_retries = int(rospy.get_param("~publish_retries", 1))
        self.retry_interval_sec = float(rospy.get_param("~retry_interval_sec", 0.25))
        self.initial_publish_delay_sec = float(rospy.get_param("~initial_publish_delay_sec", 0.8))
        self.expected_count = int(rospy.get_param("~expected_count", 3))
        self.slow_speech = bool(rospy.get_param("~slow_speech", True))
        self.voice_instruction = rospy.get_param("~voice_instruction", "").strip()
        self.llm_service_name = rospy.get_param(
            "~llm_service_name", "/smart_factory_llm/reason_pickup_order"
        )
        self.llm_service_timeout = float(rospy.get_param("~llm_service_timeout", 30.0))

        self.items_by_key: "OrderedDict[str, str]" = OrderedDict()
        self.last_spoken_at = 0.0
        self.processed = False

        if not self.voice_instruction:
            rospy.logerr("qr_speak_test: voice_instruction param is empty, LLM call will be skipped.")

        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(self.qr_topic, String, self.qr_cb, queue_size=10)

        rospy.loginfo(
            "qr_speak_test: qr_topic=%s speak_topic=%s llm_service=%s slow=%s expected=%d",
            self.qr_topic,
            self.speak_topic,
            self.llm_service_name,
            self.slow_speech,
            self.expected_count,
        )
        self.publish_status("waiting_for_qr")

    def qr_cb(self, msg: String) -> None:
        if self.processed:
            if time.time() - self.last_spoken_at < self.min_interval_sec:
                return
            self.processed = False

        if not self.voice_instruction:
            return

        for key, text in self.extract_speakable_items(msg.data):
            if not text:
                continue
            if key in self.items_by_key:
                continue

            self.items_by_key[key] = text
            rospy.loginfo(
                "Accepted QR item %d/%d: %s",
                len(self.items_by_key),
                self.expected_count,
                text,
            )

            if len(self.items_by_key) >= self.expected_count:
                self.call_llm_and_speak()
                return

    def call_llm_and_speak(self) -> None:
        self.processed = True
        items = list(self.items_by_key.values())
        item_a, item_b, item_c = items[:3]

        rospy.loginfo("Waiting for LLM service: %s", self.llm_service_name)
        try:
            rospy.wait_for_service(self.llm_service_name, timeout=self.llm_service_timeout)
        except rospy.ROSException as e:
            rospy.logerr("LLM service not available within %.1fs: %s", self.llm_service_timeout, e)
            self.publish_status("llm_service_unavailable")
            self.items_by_key.clear()
            self.processed = False
            return

        reason_pickup = rospy.ServiceProxy(self.llm_service_name, ReasonPickupOrder)

        rospy.loginfo("Calling LLM with items: %s, %s, %s", item_a, item_b, item_c)
        try:
            res = reason_pickup(item_a, item_b, item_c, self.voice_instruction)
        except rospy.ServiceException as e:
            rospy.logerr("LLM service call failed: %s", e)
            self.publish_status("llm_call_failed:%s" % str(e))
            self.items_by_key.clear()
            self.processed = False
            return

        if not res.success:
            rospy.logerr("LLM reasoning failed: %s", res.error_message)
            self.publish_status("llm_reasoning_failed:%s" % res.error_message)
            self.items_by_key.clear()
            self.processed = False
            return

        speech_text = res.announcement_full.strip()
        if not speech_text:
            rospy.logerr("LLM returned empty announcement_full")
            self.publish_status("empty_announcement")
            self.items_by_key.clear()
            self.processed = False
            return

        full_text = "%s%s%s" % (self.speak_prefix, speech_text, self.speak_suffix)
        self.publish_speech(full_text, force=True)
        self.items_by_key.clear()

    def extract_speakable_items(self, raw_text: str) -> Iterable[Tuple[str, str]]:
        payload = self.parse_json(raw_text)
        if isinstance(payload, dict):
            items = payload.get("items")
            if isinstance(items, list):
                for item in items:
                    key, text = self.clean_item(item)
                    if key and text:
                        yield key, text
                return

            key, text = self.clean_item(payload)
            if key and text:
                yield key, text
                return

        text = self.clean_plain_text(raw_text)
        if text:
            yield text, text

    def clean_item(self, item: Any) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(item, dict):
            text = self.clean_plain_text(str(item))
            return (text, text) if text else (None, None)

        raw = self.clean_plain_text(item.get("raw"))
        result = self.clean_plain_text(item.get("result"))
        api = item.get("api")
        if not result and isinstance(api, dict):
            result = self.clean_plain_text(api.get("result"))

        if not result and raw:
            raw_json = self.parse_json(raw)
            if isinstance(raw_json, dict):
                result = self.clean_plain_text(raw_json.get("result"))
            elif not self.is_url(raw):
                result = raw

        ok = item.get("ok")
        if ok is False and not result:
            return None, None

        key = raw or result
        return key, result

    def slow_text(self, text: str) -> str:
        if not self.slow_speech:
            return text
        if "，" not in text and "。" not in text:
            text = text.replace("属于", "，属于，")
            text = text.replace("应放置在", "，应放置在，")
            text = text.replace("仿真环境中", "仿真环境中，")
        if not text.endswith(("。", "！", "？")):
            text += "。"
        return text

    def clean_plain_text(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"none", "null", "true", "false"}:
            return None
        if self.is_url(text):
            return None
        return text

    def parse_json(self, text: Any) -> Any:
        if not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return None

    def is_url(self, text: str) -> bool:
        return text.startswith("http://") or text.startswith("https://")

    def publish_speech(self, speech_text: str, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_spoken_at < self.min_interval_sec:
            return
        self.last_spoken_at = now
        speech_text = self.slow_text(speech_text)
        rospy.loginfo("qr_speak_test publish to %s: %s", self.speak_topic, speech_text)
        self.publish_status("speaking:%s" % speech_text)
        if self.initial_publish_delay_sec > 0.0:
            rospy.sleep(self.initial_publish_delay_sec)
        for idx in range(max(1, self.publish_retries)):
            self.speak_pub.publish(String(data=speech_text))
            if idx + 1 < self.publish_retries:
                rospy.sleep(self.retry_interval_sec)

    def publish_status(self, status: str) -> None:
        self.status_pub.publish(String(data=status))


def main() -> None:
    rospy.init_node("qr_speak_test_node")
    QRSpeakTestNode()
    rospy.spin()


if __name__ == "__main__":
    main()
