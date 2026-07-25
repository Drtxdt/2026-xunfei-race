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

FINISH_EXTRA_FORWARD_DISTANCE_M = 0.10


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


TASK2_RECOVERABLE_NAVIGATION_FAILURES = frozenset((
    "failed",
    "centering_failed",
    "parking_staging_failed",
    "parking_recenter_failed",
    "parking_wall_fit_failed",
    "parking_docking_failed",
    "parking_validation_failed",
    "coverage_recovery_disable_failed",
))


def task2_navigation_outcome(status, timed_out=False):
    """Classify factory navigation without turning parking failure into a pause."""
    normalized = str(status or "").strip().lower()
    if normalized == "arrived":
        return "arrived"
    if normalized == "centered":
        return "centered"
    if bool(timed_out) or normalized in TASK2_RECOVERABLE_NAVIGATION_FAILURES:
        return "continue"
    return "waiting"


def task2_remembered_anchor_ready(required_anchor, observed_anchor):
    """Gate a remembered target until the robot is observing that exact anchor."""
    try:
        required = int(required_anchor or 0)
    except (TypeError, ValueError):
        required = 0
    try:
        observed = int(observed_anchor or 0)
    except (TypeError, ValueError):
        observed = 0
    return required <= 0 or observed == required


def remembered_factory_poses(
        robot_pose, target_center_x, image_width, horizontal_fov,
        corners, parking_offset, staging_offset,
        boresight_offset=0.0, steering_sign=-1.0,
        normal_offset=0.0, tangent_offset=0.0):
    """Project an OCR sign ray onto the factory wall and return true goal poses."""
    px, py, robot_yaw = [float(value) for value in robot_pose]
    width = float(image_width)
    if width <= 1.0 or len(corners) != 4:
        return None
    error = (
        float(target_center_x) - width * 0.5) / (width * 0.5)
    ray_yaw = normalize_angle(
        robot_yaw + float(boresight_offset)
        + error * abs(float(horizontal_fov)) * float(steering_sign))
    direction = (math.cos(ray_yaw), math.sin(ray_yaw))

    points = [(float(point[0]), float(point[1])) for point in corners]
    centroid = (
        sum(point[0] for point in points) / 4.0,
        sum(point[1] for point in points) / 4.0,
    )
    segments = (
        (points[0], points[2]),
        (points[1], points[3]),
        (points[2], points[3]),
        (points[0], points[1]),
    )
    best = None
    for start, end in segments:
        vx, vy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(vx, vy)
        if length <= 1e-9:
            continue
        normal = (-vy / length, vx / length)
        midpoint = (
            (start[0] + end[0]) * 0.5,
            (start[1] + end[1]) * 0.5,
        )
        if (
            normal[0] * (centroid[0] - midpoint[0])
            + normal[1] * (centroid[1] - midpoint[1])
        ) < 0.0:
            normal = (-normal[0], -normal[1])
        denominator = -vx * direction[1] + vy * direction[0]
        if abs(denominator) <= 1e-9:
            continue
        wx, wy = px - start[0], py - start[1]
        ray_t = (vx * wy - vy * wx) / denominator
        segment_u = (
            -wx * direction[1] + wy * direction[0]) / denominator
        if (
            ray_t <= 1e-9
            or segment_u < -1e-6
            or segment_u > 1.0 + 1e-6
        ):
            continue
        if best is None or ray_t < best[0]:
            best = (
                ray_t,
                (px + ray_t * direction[0],
                 py + ray_t * direction[1]),
                normal,
            )
    if best is None:
        return None

    wall_point, normal = best[1], best[2]
    tangent = (-normal[1], normal[0])

    def goal(offset):
        distance = float(offset) + float(normal_offset)
        x = (
            wall_point[0] + normal[0] * distance
            + tangent[0] * float(tangent_offset))
        y = (
            wall_point[1] + normal[1] * distance
            + tangent[1] * float(tangent_offset))
        yaw = math.atan2(-normal[1], -normal[0])
        return (x, y, yaw)

    return {
        "wall_point": wall_point,
        "parking_pose": goal(parking_offset),
        "staging_pose": goal(staging_offset),
        "ray_yaw": ray_yaw,
    }


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


def parse_task1_categories(text):
    """Parse the physical and simulation categories from the official command."""
    compact = "".join(str(text or "").split()).lower()
    marker_positions = [
        compact.find(marker)
        for marker in (
            "仿真环境", "仿真", "模拟环境", "虚拟环境", "simulation", "sim"
        )
        if compact.find(marker) >= 0
    ]
    if not marker_positions:
        return parse_category(compact), None
    split_at = min(marker_positions)
    return parse_category(compact[:split_at]), parse_category(compact[split_at:])


def parse_task_categories(text):
    """Backward-compatible name used by the earlier dual-category branch."""
    return parse_task1_categories(text)


def build_task1_instruction(pickup_category, sim_category):
    pickup = CATEGORY_LABELS.get(pickup_category)
    simulation = CATEGORY_LABELS.get(sim_category)
    if not pickup or not simulation:
        raise ValueError("both task1 categories are required")
    return (
        "小飞小飞，前往物品领取区，取得{}类，放置在对应仓库，"
        "并领取仿真环境中需要的{}类放置在对应仓库"
    ).format(pickup[0], simulation[0])


def task2_delivery_targets(pickup, simulation):
    """Return ordered workshop visits, omitting a duplicate second workshop."""
    pickup = tuple(pickup)
    simulation = tuple(simulation)
    if len(pickup) != 3 or len(simulation) != 3:
        raise ValueError("task2 delivery targets require category, item and workshop")
    visits = [("physical",) + pickup]
    if pickup[0] != simulation[0]:
        visits.append(("simulation",) + simulation)
    return tuple(visits)


def task2_semantic_coverage_hint(memory, target_category):
    """Return the remembered target anchor and confirmed irrelevant anchors."""
    target_category = normalize_category(target_category)
    preferred = 0
    skipped = set()
    for category, observation in (memory or {}).items():
        normalized = normalize_category(category)
        anchor = observation.get("anchor", 0) if isinstance(observation, dict) else observation
        try:
            anchor = int(anchor)
        except (TypeError, ValueError):
            continue
        if anchor <= 0:
            continue
        if normalized == target_category:
            preferred = anchor
        else:
            skipped.add(anchor)
    skipped.discard(preferred)
    return preferred, tuple(sorted(skipped))


def normalize_category(value):
    text = str(value or "").strip().lower()
    return OCR_CATEGORY_ALIASES.get(text) or parse_category(text)


def scan_sector_min(ranges, angle_min, angle_increment, center_angle,
                    half_angle, range_min=0.0, range_max=float("inf")):
    """Return the nearest finite lidar sample in a wrapped angular sector."""
    if not ranges or abs(float(angle_increment)) <= 1e-12:
        return None
    center_angle = float(center_angle)
    half_angle = abs(float(half_angle))
    lower = max(0.0, float(range_min))
    upper = float(range_max)
    nearest = None
    for index, value in enumerate(ranges):
        value = float(value)
        if not math.isfinite(value) or value < lower or value > upper:
            continue
        angle = float(angle_min) + index * float(angle_increment)
        if abs(normalize_angle(angle - center_angle)) > half_angle:
            continue
        if nearest is None or value < nearest:
            nearest = value
    return nearest


def split_rotation_steps(total_angle, step_angle):
    """Split a positive rotation into exact bounded steps without overshooting."""
    remaining = max(0.0, float(total_angle))
    step_angle = float(step_angle)
    if step_angle <= 0.0:
        raise ValueError("step_angle must be positive")
    steps = []
    while remaining > 1e-9:
        current = min(step_angle, remaining)
        steps.append(current)
        remaining -= current
    return tuple(steps)


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
