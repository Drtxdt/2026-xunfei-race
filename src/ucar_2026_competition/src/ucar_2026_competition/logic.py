"""ROS-independent parsing, protocol, and motion helpers."""

import json
import math


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


def normalize_angle(angle):
    """Normalize an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class DirectedYawAccumulator:
    """Accumulate odometry yaw progress while handling the +/-pi wrap."""

    def __init__(self, direction=1.0):
        self.direction = 1.0 if float(direction) >= 0.0 else -1.0
        self.start_yaw = None
        self.last_yaw = None
        self.progress = 0.0

    def reset(self, yaw):
        self.start_yaw = float(yaw)
        self.last_yaw = float(yaw)
        self.progress = 0.0

    def update(self, yaw):
        yaw = float(yaw)
        if self.last_yaw is None:
            self.reset(yaw)
            return self.progress
        self.last_yaw = yaw
        # Measure net rotation from this step's starting yaw. Keeping the
        # maximum suppresses small reverse jitter without counting the same
        # forward arc twice after the chassis recovers from that jitter.
        net_progress = normalize_angle(yaw - self.start_yaw) * self.direction
        if net_progress > self.progress:
            self.progress = net_progress
        return self.progress


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
