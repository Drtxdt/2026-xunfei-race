"""ROS-independent parsing, protocol, and motion helpers."""

import json
import math
import time


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


def stage_sequence(mode, enable_simulation=False):
    normalized = str(mode or "").strip().lower()
    if normalized == "full":
        if bool(enable_simulation):
            return ("task1", "task2", "task3", "task4", "task5")
        return ("task1", "task2", "task4", "task5")
    if normalized == "task1_task2":
        return ("task1", "task2")
    if normalized == "task3_task4":
        return ("task3", "task4")
    if normalized == "task4_task5":
        return ("task4", "task5")
    if normalized in ("task1", "task2", "task3", "task4", "task5"):
        return (normalized,)
    raise ValueError("unsupported start_stage: {}".format(mode))


def task4_handoff_required(previous_stage, current_stage):
    """Require a stationary localization handoff from production to task4."""
    return (str(current_stage or "").strip().lower() == "task4" and
            str(previous_stage or "").strip().lower() in ("task2", "task3"))


def task4_start_action(skip_stop_line_approach, traffic_pose_configured):
    """Select whether task4 starts at the line or navigates to it."""
    if bool(skip_stop_line_approach):
        return "detect"
    if bool(traffic_pose_configured):
        return "approach"
    raise ValueError(
        "traffic pose is not configured; set traffic coordinates or start at stop line")


def base_is_stopped(linear_x, linear_y, angular_z,
                    linear_tolerance=0.01, angular_tolerance=0.02):
    return (math.hypot(float(linear_x), float(linear_y)) <=
            abs(float(linear_tolerance)) and
            abs(float(angular_z)) <= abs(float(angular_tolerance)))


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


class TemporalTargetFilter:
    """Confirm repeated target evidence within a time window; blanks do not erase it."""

    def __init__(self, required=2, window_sec=1.5):
        self.required = max(1, int(required))
        self.window_sec = max(0.05, float(window_sec))
        self.hit_times = []

    @property
    def hit_count(self):
        return len(self.hit_times)

    def reset(self):
        self.hit_times = []

    def push(self, target, observed, now=None):
        now = time.monotonic() if now is None else float(now)
        self.hit_times = [
            stamp for stamp in self.hit_times
            if now - stamp <= self.window_sec
        ]
        if observed is None or observed == "":
            return len(self.hit_times) >= self.required
        if not target or observed != target:
            self.reset()
            return False
        self.hit_times.append(now)
        return len(self.hit_times) >= self.required


TRIGGER_ACK_STATES = frozenset((
    "triggered",
    "target_locked",
    "target_centering",
    "centered",
    "parking_approaching",
    "parking_verifying",
    "arrived",
))


def trigger_delivery_state(service_accepted, navigator_status, elapsed, timeout_sec):
    """Classify a reliable target-trigger handshake without depending on ROS."""
    if bool(service_accepted) and str(navigator_status or "").strip().lower() in TRIGGER_ACK_STATES:
        return "acknowledged"
    if float(elapsed) >= max(0.0, float(timeout_sec)):
        return "failed"
    return "pending"


def task2_announcement_required(navigator_status, already_completed):
    """Allow the official task2 speech exactly once, and only after arrival."""
    return (str(navigator_status or "").strip().lower() == "arrived" and
            not bool(already_completed))


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
