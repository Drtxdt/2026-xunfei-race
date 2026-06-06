#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean QR JSON results, infer task output, and publish to /speak."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any, Iterable, Optional, Tuple

import rospy
from std_msgs.msg import String


MAJOR_TO_WORKSHOP = {
    "食品大类": "食品加工车间",
    "日用品大类": "日用品加工车间",
    "电子产品大类": "电子产品生产车间",
}

ITEM_TO_MAJOR = {
    "苹果": "食品大类",
    "猪肉": "食品大类",
    "草莓": "食品大类",
    "香蕉": "食品大类",
    "饺子": "食品大类",
    "面条": "食品大类",
    "薯片": "食品大类",
    "馒头": "食品大类",
    "纸巾": "日用品大类",
    "毛巾": "日用品大类",
    "牙刷": "日用品大类",
    "洗衣液": "日用品大类",
    "T恤衫": "日用品大类",
    "手机": "电子产品大类",
    "耳机": "电子产品大类",
    "充电器": "电子产品大类",
    "鼠标": "电子产品大类",
    "数据线": "电子产品大类",
}

CATEGORY_TO_MAJOR = {
    "food": "食品大类",
    "daily": "日用品大类",
    "electronic": "电子产品大类",
    "electronics": "电子产品大类",
}


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
        self.initial_publish_delay_sec = float(rospy.get_param("~initial_publish_delay_sec", 0.8))
        self.expected_count = int(rospy.get_param("~expected_count", 3))
        self.output_mode = rospy.get_param("~output_mode", "task_result")
        self.pickup_target_major = self.normalize_major(
            rospy.get_param("~pickup_target_major", "食品大类")
        )
        self.sim_target_major = self.normalize_major(
            rospy.get_param("~sim_target_major", "日用品大类")
        )
        self.slow_speech = bool(rospy.get_param("~slow_speech", True))
        self.speak_each_item = bool(rospy.get_param("~speak_each_item", True))
        self.allow_fallback_task_result = bool(rospy.get_param("~allow_fallback_task_result", True))

        self.spoken_keys = set()
        self.confirmed_keys = set()
        self.items_by_key: "OrderedDict[str, Tuple[str, Optional[str]]]" = OrderedDict()
        self.task_result_spoken = False
        self.last_spoken_at = 0.0
        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(self.qr_topic, String, self.qr_cb, queue_size=10)

        rospy.loginfo(
            "qr_speak_test: qr_topic=%s speak_topic=%s pickup=%s sim=%s mode=%s",
            self.qr_topic,
            self.speak_topic,
            self.pickup_target_major,
            self.sim_target_major,
            self.output_mode,
        )
        self.publish_status("waiting_for_qr")

    def qr_cb(self, msg: String) -> None:
        for key, text, major_hint in self.extract_speakable_items(msg.data):
            if not text:
                continue
            if key in self.items_by_key:
                continue

            major = major_hint or self.major_for_item(text)
            self.items_by_key[key] = (text, major)
            rospy.loginfo(
                "Accepted QR item %d/%d: %s major=%s",
                len(self.items_by_key),
                self.expected_count,
                text,
                major or "unknown",
            )

            if self.output_mode == "item":
                if not self.repeat_same and key in self.spoken_keys:
                    continue
                self.spoken_keys.add(key)
                self.publish_speech("%s%s%s" % (self.speak_prefix, text, self.speak_suffix))
            elif self.speak_each_item and key not in self.confirmed_keys:
                self.confirmed_keys.add(key)
                self.publish_speech("已识别%s" % text, force=True)

        if self.output_mode == "task_result":
            self.try_publish_task_result()

    def try_publish_task_result(self) -> None:
        if self.task_result_spoken and not self.repeat_same:
            return
        if len(self.items_by_key) < self.expected_count:
            self.publish_status("waiting_for_qr:%d/%d" % (len(self.items_by_key), self.expected_count))
            return

        pickup_item = self.find_item_by_major(self.pickup_target_major)
        sim_item = self.find_item_by_major(self.sim_target_major)
        if not pickup_item or not sim_item:
            if self.allow_fallback_task_result:
                fallback_items = list(self.items_by_key.values())
                if not pickup_item and fallback_items:
                    pickup_item = fallback_items[0][0]
                if not sim_item and len(fallback_items) > 1:
                    sim_item = fallback_items[1][0]
                elif not sim_item and fallback_items:
                    sim_item = fallback_items[0][0]
                if pickup_item and sim_item:
                    rospy.logwarn("Using fallback task result items: pickup=%s sim=%s", pickup_item, sim_item)
                    text = self.build_task_result_text(
                        pickup_item,
                        self.pickup_target_major,
                        sim_item,
                        self.sim_target_major,
                    )
                    self.task_result_spoken = True
                    self.publish_speech(text, force=True)
                    return

            rospy.logwarn(
                "Cannot build task result yet. pickup=%s item=%s sim=%s item=%s items=%s",
                self.pickup_target_major,
                pickup_item,
                self.sim_target_major,
                sim_item,
                list(self.items_by_key.values()),
            )
            self.publish_status("task_result_missing_target")
            return

        text = self.build_task_result_text(pickup_item, self.pickup_target_major, sim_item, self.sim_target_major)
        self.task_result_spoken = True
        self.publish_speech(text, force=True)

    def find_item_by_major(self, major: str) -> Optional[str]:
        for item, item_major in self.items_by_key.values():
            if item_major == major:
                return item
        return None

    def build_task_result_text(self, pickup_item: str, pickup_major: str, sim_item: str, sim_major: str) -> str:
        return "取得%s属于%s应放置在%s，仿真环境中取得%s属于%s应放置在%s" % (
            pickup_item,
            pickup_major,
            MAJOR_TO_WORKSHOP[pickup_major],
            sim_item,
            sim_major,
            MAJOR_TO_WORKSHOP[sim_major],
        )

    def slow_text(self, text: str) -> str:
        if not self.slow_speech:
            return text
        text = text.replace("属于", "，属于")
        text = text.replace("应放置在", "，应放置在")
        text = text.replace("，仿真环境中", "。仿真环境中")
        if not text.endswith("。"):
            text += "。"
        return text

    def normalize_major(self, value: Any) -> str:
        text = str(value or "").strip()
        lowered = text.lower()
        if lowered in CATEGORY_TO_MAJOR:
            return CATEGORY_TO_MAJOR[lowered]
        if "食品" in text:
            return "食品大类"
        if "日用" in text or "日化" in text:
            return "日用品大类"
        if "电子" in text or "数码" in text or "电器" in text:
            return "电子产品大类"
        if text in MAJOR_TO_WORKSHOP:
            return text
        rospy.logwarn("Unknown target major '%s', fallback to 食品大类", text)
        return "食品大类"

    def major_for_item(self, item: str) -> Optional[str]:
        return ITEM_TO_MAJOR.get(item)

    def extract_speakable_items(self, raw_text: str) -> Iterable[Tuple[str, str, Optional[str]]]:
        payload = self.parse_json(raw_text)
        if isinstance(payload, dict):
            items = payload.get("items")
            if isinstance(items, list):
                for item in items:
                    key, text, major = self.clean_item(item)
                    if key and text:
                        yield key, text, major
                return

            key, text, major = self.clean_item(payload)
            if key and text:
                yield key, text, major
                return

        text = self.clean_plain_text(raw_text)
        if text:
            yield text, text, self.major_for_item(text)

    def clean_item(self, item: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not isinstance(item, dict):
            text = self.clean_plain_text(str(item))
            return (text, text, self.major_for_item(text)) if text else (None, None, None)

        raw = self.clean_plain_text(item.get("raw"))
        result = self.clean_plain_text(item.get("result"))
        api = item.get("api")
        major_hint = None
        if not result and isinstance(api, dict):
            result = self.clean_plain_text(api.get("result"))
        if isinstance(api, dict):
            category = self.clean_plain_text(api.get("category"))
            if category:
                major_hint = CATEGORY_TO_MAJOR.get(category.lower())

        if not result and raw:
            raw_json = self.parse_json(raw)
            if isinstance(raw_json, dict):
                result = self.clean_plain_text(raw_json.get("result"))
                category = self.clean_plain_text(raw_json.get("category"))
                if category:
                    major_hint = CATEGORY_TO_MAJOR.get(category.lower())
            elif not self.is_url(raw):
                result = raw

        ok = item.get("ok")
        if ok is False and not result:
            return None, None, None

        key = raw or result
        return key, result, major_hint or self.major_for_item(result or "")

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
