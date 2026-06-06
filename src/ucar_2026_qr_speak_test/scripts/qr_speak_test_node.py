#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean QR JSON results and publish speakable text to /speak."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import rospy
from std_msgs.msg import String


class QRSpeakTestNode:
    def __init__(self) -> None:
        self.qr_topic = rospy.get_param("~qr_topic", "/qr_code_data")
        self.speak_topic = rospy.get_param("~speak_topic", "/speak")
        self.status_topic = rospy.get_param("~status_topic", "/qr_speak_test/status")
        self.speak_prefix = rospy.get_param("~speak_prefix", "")
        self.speak_suffix = rospy.get_param("~speak_suffix", "")
        self.repeat_same = bool(rospy.get_param("~repeat_same", False))
        self.min_interval_sec = float(rospy.get_param("~min_interval_sec", 1.5))
        self.publish_retries = int(rospy.get_param("~publish_retries", 2))
        self.retry_interval_sec = float(rospy.get_param("~retry_interval_sec", 0.25))

        self.spoken_keys = set()
        self.last_spoken_at = 0.0
        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(self.qr_topic, String, self.qr_cb, queue_size=10)

        rospy.loginfo("qr_speak_test: qr_topic=%s speak_topic=%s", self.qr_topic, self.speak_topic)
        self.publish_status("waiting_for_qr")

    def qr_cb(self, msg: String) -> None:
        for key, text in self.extract_speakable_items(msg.data):
            if not text:
                continue
            if not self.repeat_same and key in self.spoken_keys:
                continue
            now = time.time()
            if now - self.last_spoken_at < self.min_interval_sec:
                continue

            speech_text = "%s%s%s" % (self.speak_prefix, text, self.speak_suffix)
            self.spoken_keys.add(key)
            self.last_spoken_at = now
            self.publish_speech(speech_text)

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

    def publish_speech(self, speech_text: str) -> None:
        rospy.loginfo("qr_speak_test publish to %s: %s", self.speak_topic, speech_text)
        self.publish_status("speaking:%s" % speech_text)
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
