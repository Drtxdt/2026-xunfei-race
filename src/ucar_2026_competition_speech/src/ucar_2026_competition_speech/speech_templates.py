# -*- coding: utf-8 -*-
"""Pure helpers for generating the exact competition announcement texts."""

from __future__ import annotations

import re
from typing import Tuple


EVENT_ALIASES = {
    "1": "task1",
    "task1": "task1",
    "reasoning": "task1",
    "2": "task2",
    "task2": "task2",
    "physical": "task2",
    "3": "task3",
    "task3": "task3",
    "simulation": "task3",
    "4": "task4",
    "task4": "task4",
    "traffic": "task4",
    "5": "task5",
    "task5": "task5",
    "finish": "task5",
    "custom": "custom",
}

DECISION_ALIASES = {
    "left": "左转",
    "turn_left": "左转",
    "左": "左转",
    "左转": "左转",
    "right": "右转",
    "turn_right": "右转",
    "右": "右转",
    "右转": "右转",
    "straight": "直行",
    "forward": "直行",
    "直": "直行",
    "直行": "直行",
    "stop": "停止",
    "red": "停止",
    "红灯": "停止",
    "停止": "停止",
}


def clean_text(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").strip())


def normalize_event(event: str) -> str:
    key = (event or "").strip().lower()
    normalized = EVENT_ALIASES.get(key)
    if not normalized:
        raise ValueError("unsupported event: {}".format(event))
    return normalized


def normalize_decision(decision: str) -> str:
    key = clean_text(decision).lower()
    normalized = DECISION_ALIASES.get(key)
    if not normalized:
        raise ValueError("unsupported traffic decision: {}".format(decision))
    return normalized


def build_announcement(
    event: str,
    item: str = "",
    workshop: str = "",
    decision: str = "",
    text: str = "",
) -> Tuple[str, str]:
    normalized_event = normalize_event(event)
    item = clean_text(item)
    workshop = clean_text(workshop)
    text = (text or "").strip()

    if normalized_event in ("task1", "custom"):
        if not text:
            raise ValueError("{} requires text".format(normalized_event))
        return normalized_event, text

    if normalized_event == "task2":
        if not item or not workshop:
            raise ValueError("task2 requires item and workshop")
        return normalized_event, "已将{}放入{}".format(item, workshop)

    if normalized_event == "task3":
        if not item or not workshop:
            raise ValueError("task3 requires item and workshop")
        return normalized_event, "仿真任务已完成，已将{}放入{}".format(item, workshop)

    if normalized_event == "task4":
        return normalized_event, normalize_decision(decision)

    return normalized_event, "任务完成"


def estimate_duration(
    text: str,
    chars_per_second: float = 3.0,
    startup_sec: float = 1.0,
    tail_sec: float = 1.0,
) -> float:
    visible_chars = len(re.sub(r"[\s，。！？、；：,.!?;:]", "", text or ""))
    punctuation_count = len(re.findall(r"[，。！？、；：,.!?;:]", text or ""))
    speech_sec = visible_chars / max(chars_per_second, 0.1)
    pause_sec = punctuation_count * 0.18
    return max(2.0, startup_sec + speech_sec + pause_sec + tail_sec)
