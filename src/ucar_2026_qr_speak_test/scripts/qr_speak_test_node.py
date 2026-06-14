#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean QR JSON, call Spark LLM directly, and publish slow speech to /speak."""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import rospy
from std_msgs.msg import String

_LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是第21届全国大学生智能汽车竞赛「讯飞智慧工厂」赛项的调度推理模块。
你必须严格根据用户给出的三个货品名称（来自现场二维码返回的 JSON 字段 result）和用户语音指令，完成语义推理。

大类与目标车间对应关系（播报时必须使用下列车间全名）：
- 食品、食品加工类、生鲜、食材等相关大类 → 车间名：食品加工车间；对外表述可用「食品大类」。
- 日用品、日化、纺织、清洁用品等相关大类 → 车间名：日用品加工车间；对外表述可用「日用品大类」。
- 电子产品、数码、电器等相关大类 → 车间名：电子产品生产车间；对外表述可用「电子产品大类」。

推理要求：
1. 从语音中解析两个目标：①物品领取区要取的「目标大类」；②仿真环境中要取的「目标大类」。若语音未明确写出，依据指令里出现的「取得…」「放置在…」等语义尽力推断；仍无法确定时以 null 表示并在 err_hint 说明。
2. 判断三个货品各自属于哪一大类（食品/日用品/电子产品）。
3. 对①②各自在三个货品中选出唯一最匹配的一项；若多个候选，选与语音关键词最贴近的一项。

只输出一个 JSON 对象，不要 Markdown，不要代码围栏。键必须齐全，格式如下：
{
  "pickup_item": "字符串或null",
  "pickup_major": "食品大类|日用品大类|电子产品大类之一或null",
  "pickup_workshop": "食品加工车间|日用品加工车间|电子产品生产车间之一或null",
  "sim_item": "字符串或null",
  "sim_major": "同上或null",
  "sim_workshop": "同上或null",
  "announcement_physical": "取得X属于Y应放置在Z",
  "announcement_simulation": "仿真环境中取得X属于Y应放置在Z",
  "err_hint": "无问题时为空字符串"
}

announcement 句式必须与赛题一致：
- announcement_physical 必须以「取得」开头，包含「属于」「应放置在」，车间为上述三个全名之一。
- announcement_simulation 必须以「仿真环境中取得」开头（中间不要逗号），同样包含「属于」「应放置在」。
"""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _parse_llm_json(content: str) -> Dict[str, Any]:
    raw = _strip_code_fence(content)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("model output contains no JSON object")
    return json.loads(raw[start : end + 1])


def _build_user_prompt(items: List[str], voice: str) -> str:
    lines = [
        "三个货品名称（与车载视觉依次读取二维码的结果一致，可能对应食品/日用品/电子产品母类链接）：",
        "1) {}".format(items[0]),
        "2) {}".format(items[1]),
        "3) {}".format(items[2]),
        "",
        "用户语音指令全文：",
        voice.strip(),
    ]
    return "\n".join(lines)


def _fill_announcements(data: Dict[str, Any]) -> Tuple[str, str, str]:
    phy = (data.get("announcement_physical") or "").strip()
    sim = (data.get("announcement_simulation") or "").strip()

    def _line(prefix: str, item: str, major: str, workshop: str) -> str:
        if prefix == "sim":
            return "仿真环境中取得{}属于{}应放置在{}".format(item, major, workshop)
        return "取得{}属于{}应放置在{}".format(item, major, workshop)

    if not phy and data.get("pickup_item"):
        phy = _line("phy", str(data.get("pickup_item") or ""),
                     str(data.get("pickup_major") or ""),
                     str(data.get("pickup_workshop") or ""))
    if not sim and data.get("sim_item"):
        sim = _line("sim", str(data.get("sim_item") or ""),
                     str(data.get("sim_major") or ""),
                     str(data.get("sim_workshop") or ""))
    full = ""
    if phy and sim:
        full = "{}，{}".format(phy, sim)
    elif phy:
        full = phy
    else:
        full = sim
    return phy, sim, full


class SparkX2Client:
    """Spark X2 HTTP Chat Completions (Bearer APIPassword)."""

    def __init__(self, api_password: str, base_url: str, model: str, timeout_sec: float) -> None:
        self._api_password = api_password.strip()
        self._url = base_url.strip()
        self._model = model.strip()
        self._timeout = timeout_sec
        if not self._api_password:
            raise ValueError("api_password is empty, set XF_SPARK_API_PASSWORD or ~api_password")

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={
                "Authorization": "Bearer {}".format(self._api_password),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout, context=ctx) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError("Spark HTTP {}: {}".format(e.code, detail)) from e
        except urllib.error.URLError as e:
            raise RuntimeError("Spark network error: {}".format(e)) from e

        decoded = json.loads(body)
        choices = decoded.get("choices") or []
        if not choices:
            raise RuntimeError("Spark response has no choices: {}".format(body[:800]))
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not content:
            raise RuntimeError("Spark response has no content: {}".format(body[:800]))
        return str(content)


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
        self.debounce_sec = float(rospy.get_param("~debounce_sec", 2.0))
        self.min_item_count = int(rospy.get_param("~min_item_count", 1))
        self.slow_speech = bool(rospy.get_param("~slow_speech", True))
        self.voice_instruction = rospy.get_param("~voice_instruction", "").strip()

        self.api_password = rospy.get_param(
            "~api_password", os.environ.get("XF_SPARK_API_PASSWORD", "")
        ).strip()
        self.spark_base_url = rospy.get_param(
            "~spark_base_url", "https://spark-api-open.xf-yun.com/x2/chat/completions"
        )
        self.spark_model = rospy.get_param("~spark_model", "spark-x")
        self.request_timeout_sec = float(rospy.get_param("~request_timeout_sec", 90.0))

        self.items_by_key: "OrderedDict[str, str]" = OrderedDict()
        self.last_spoken_at = 0.0
        self.processed = False
        self._debounce_timer: Optional[rospy.Timer] = None

        if not self.voice_instruction:
            rospy.logerr("qr_speak_test: voice_instruction param is empty, LLM call will be skipped.")
        if not self.api_password:
            rospy.logerr(
                "qr_speak_test: api_password is empty. "
                "Set XF_SPARK_API_PASSWORD env var or ~api_password ROS param."
            )

        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(self.qr_topic, String, self.qr_cb, queue_size=10)

        rospy.loginfo(
            "qr_speak_test: qr_topic=%s speak_topic=%s spark_model=%s "
            "slow=%s min_items=%d debounce=%.1fs timeout=%.0fs",
            self.qr_topic, self.speak_topic, self.spark_model,
            self.slow_speech, self.min_item_count,
            self.debounce_sec, self.request_timeout_sec,
        )
        self.publish_status("waiting_for_qr")

    def _reset_debounce(self) -> None:
        if self._debounce_timer is not None:
            self._debounce_timer.shutdown()
        self._debounce_timer = rospy.Timer(
            rospy.Duration(self.debounce_sec),
            self._on_debounce,
            oneshot=True,
        )

    def qr_cb(self, msg: String) -> None:
        if self.processed:
            if time.time() - self.last_spoken_at < self.min_interval_sec:
                return
            self.processed = False

        if not self.voice_instruction or not self.api_password:
            return

        new_item = False
        for key, text in self.extract_speakable_items(msg.data):
            if not text:
                continue
            if key in self.items_by_key:
                continue

            self.items_by_key[key] = text
            new_item = True
            rospy.loginfo(
                "Accepted QR item %d: %s (need at least %d)",
                len(self.items_by_key), text, self.min_item_count,
            )

        if new_item:
            self._reset_debounce()

    def _on_debounce(self, _event: rospy.timer.TimerEvent) -> None:
        if self.processed and time.time() - self.last_spoken_at < self.min_interval_sec:
            return
        if len(self.items_by_key) < self.min_item_count:
            self.publish_status(
                "debounce_insufficient:%d/%d" % (len(self.items_by_key), self.min_item_count)
            )
            return
        self.call_llm_and_speak()

    def call_llm_and_speak(self) -> None:
        self.processed = True
        if self._debounce_timer is not None:
            self._debounce_timer.shutdown()
            self._debounce_timer = None

        items = list(self.items_by_key.values())
        padded = list(items)
        while len(padded) < 3:
            padded.append(padded[-1])
        item_a, item_b, item_c = padded[:3]

        try:
            client = SparkX2Client(
                self.api_password, self.spark_base_url,
                self.spark_model, self.request_timeout_sec,
            )
        except ValueError as e:
            rospy.logerr("Spark client init failed: %s", e)
            self.publish_status("spark_init_failed:%s" % str(e))
            self.items_by_key.clear()
            self.processed = False
            return

        user_prompt = _build_user_prompt([item_a, item_b, item_c], self.voice_instruction)
        rospy.loginfo("Calling Spark LLM with items: %s, %s, %s", item_a, item_b, item_c)

        try:
            content = client.chat(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            rospy.logerr("Spark LLM call failed: %s", e)
            self.publish_status("spark_call_failed:%s" % str(e))
            self.items_by_key.clear()
            self.processed = False
            return

        rospy.loginfo("Spark LLM raw reply: %s", content[:300])

        try:
            data = _parse_llm_json(content)
        except Exception as e:
            rospy.logerr("Failed to parse LLM JSON: %s", e)
            self.publish_status("llm_parse_failed:%s" % str(e))
            self.items_by_key.clear()
            self.processed = False
            return

        hint = (data.get("err_hint") or "").strip()
        if hint:
            rospy.logwarn("LLM hint: %s", hint)

        _, _, full = _fill_announcements(data)
        speech_text = full.strip()
        if not speech_text:
            rospy.logerr("LLM returned empty announcement")
            self.publish_status("empty_announcement")
            self.items_by_key.clear()
            self.processed = False
            return

        rospy.loginfo("LLM announcement: %s", speech_text)
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
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    QRSpeakTestNode()
    rospy.spin()


if __name__ == "__main__":
    main()
