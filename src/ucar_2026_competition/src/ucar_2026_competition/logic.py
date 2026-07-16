"""ROS-independent parsing and protocol helpers."""

import json


CATEGORY_LABELS = {
    "food": ("食品", "食品加工车间"),
    "daily": ("日用品", "日用品加工车间"),
    "electronics": ("电子产品", "电子产品生产车间"),
}

OCR_CATEGORY_ALIASES = {
    "food": "food",
    "食品": "food",
    "daily": "daily",
    "日用品": "daily",
    "electronic": "electronics",
    "electronics": "electronics",
    "电子": "electronics",
    "电子产品": "electronics",
}

TRAFFIC_CLASS_TO_DECISION = {
    "green_left": "left",
    "green_right": "right",
    "green_straight": "straight",
    "red_light": "stop",
}

TRACK_CONFIG = {
    "left": ("track_end_stop.launch", "/track_end_stop/status", "finish"),
    "right": ("right_track_end_stop.launch", "/right_track_end_stop/status", "right_finish"),
    "straight": (
        "stable_right_track_end_stop.launch",
        "/stable_right_track_end_stop/status",
        "stable_right_finish",
    ),
}


def stage_sequence(mode):
    normalized = str(mode or "").strip().lower()
    if normalized == "full":
        return ("task1", "task2", "task3", "task4", "task5")
    if normalized in ("task1", "task2", "task3", "task4", "task5"):
        return (normalized,)
    raise ValueError("unsupported start_stage: {}".format(mode))


class ConsecutiveTargetFilter:
    def __init__(self, required=3):
        self.required = max(1, int(required))
        self.hits = 0

    def reset(self):
        self.hits = 0

    def push(self, target, observed):
        if target and observed == target:
            self.hits += 1
        else:
            self.hits = 0
        return self.hits >= self.required


def parse_category(text):
    compact = "".join(str(text or "").split()).lower()
    if "日用品" in compact or "daily" in compact:
        return "daily"
    if "电子产品" in compact or "electronics" in compact or "electronic" in compact:
        return "electronics"
    if "食品" in compact or "food" in compact:
        return "food"
    return None


def normalize_category(value):
    text = str(value or "").strip().lower()
    return OCR_CATEGORY_ALIASES.get(text) or parse_category(text)


def qr_values_from_payload(payload):
    values = []
    for entry in payload.get("items", []):
        raw = str(entry.get("raw") or "").strip()
        result = str(entry.get("result") or "").strip()
        ok = bool(entry.get("ok"))
        if not result and raw and not raw.startswith(("http://", "https://")):
            result, ok = raw, True
        key = raw or result
        if key and result and ok:
            values.append((key, result))
    return values


def traffic_decision_from_payload(payload):
    consensus = payload.get("consensus", {})
    if not consensus.get("active"):
        return None
    return TRAFFIC_CLASS_TO_DECISION.get(consensus.get("class_name"))


class JsonLineBuffer:
    def __init__(self):
        self.buffer = b""

    def feed(self, chunk):
        self.buffer += chunk
        events = []
        while b"\n" in self.buffer:
            line, self.buffer = self.buffer.split(b"\n", 1)
            try:
                events.append(json.loads(line.decode("utf-8")))
            except (ValueError, UnicodeDecodeError):
                continue
        return events
