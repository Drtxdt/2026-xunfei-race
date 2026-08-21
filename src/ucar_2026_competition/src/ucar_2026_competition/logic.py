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

TASK4_STAGING_CALIBRATED = (0.2395, -3.10, -1.5596)
TASK4_STAGING_RETIRED = (0.3195, -3.00, -1.5596)


def normalize_angle(angle):
    """Normalize an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_task4_staging_pose(x, y, yaw):
    """Migrate the retired task4 staging calibration while preserving custom poses."""
    pose = (float(x), float(y), float(yaw))
    retired = (
        abs(pose[0] - TASK4_STAGING_RETIRED[0]) <= 0.002
        and abs(pose[1] - TASK4_STAGING_RETIRED[1]) <= 0.002
        and abs(normalize_angle(
            pose[2] - TASK4_STAGING_RETIRED[2])) <= 0.02
    )
    if retired:
        return TASK4_STAGING_CALIBRATED, True
    return pose, False


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


def final_advance_completed(planned_m, progress_m, tolerance_m=0.008):
    """Accept a completed variable-length final advance within odometry tolerance."""
    planned = float(planned_m)
    progress = float(progress_m)
    tolerance = float(tolerance_m)
    if planned < 0.0 or progress < 0.0 or tolerance < 0.0:
        return False
    return progress + tolerance >= planned


def target_bbox_ratios(bbox, image_width, image_height):
    """Return normalized width, height, and area for a polygonal target box."""
    try:
        image_width = float(image_width)
        image_height = float(image_height)
        points = [
            (float(point[0]), float(point[1]))
            for point in bbox
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
    except (TypeError, ValueError):
        return None
    if image_width <= 1.0 or image_height <= 1.0 or len(points) < 2:
        return None
    width = max(point[0] for point in points) - min(point[0] for point in points)
    height = max(point[1] for point in points) - min(point[1] for point in points)
    if width <= 0.0 or height <= 0.0:
        return None
    return (
        width / image_width,
        height / image_height,
        width * height / (image_width * image_height),
    )


def target_bbox_is_close_enough(
        bbox, image_width, image_height,
        min_width_ratio=0.11, min_height_ratio=0.06, min_area_ratio=0.006):
    """Reject distant OCR signs that are too small for reliable centering."""
    ratios = target_bbox_ratios(bbox, image_width, image_height)
    if ratios is None:
        return False
    width_ratio, height_ratio, area_ratio = ratios
    return (
        width_ratio >= float(min_width_ratio)
        and height_ratio >= float(min_height_ratio)
        and area_ratio >= float(min_area_ratio)
    )


def task2_target_trigger_is_eligible(bbox_is_close, active_anchor):
    """Only lock a close OCR target while stopped at a coverage anchor.

    The navigator deliberately clears its active anchor during transit.  This
    prevents an oblique sign glimpse between cones from interrupting move_base
    and turning an arbitrary transit pose into the parking approach origin.
    """
    if not bool(bbox_is_close):
        return False
    try:
        return int(active_anchor) > 0
    except (TypeError, ValueError):
        return False


def task2_prewarm_reusable(enabled, phase, expected_category,
                           prewarmed_category, ocr_alive, navigator_alive):
    """Reuse only a complete physical-search prewarm with matching OCR class."""
    return bool(
        enabled and phase == "physical" and expected_category and
        expected_category == prewarmed_category and
        ocr_alive and navigator_alive)


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


_TASK_CATEGORY_ALIASES = (
    (
        "electronics",
        (
            "电子产品", "电子设备", "数码产品", "电子类", "数码类", "电器类",
            "电子", "数码", "电器", "electronics", "electronic",
        ),
    ),
    (
        "daily",
        (
            "日用品", "生活用品", "日常用品", "日用百货", "清洁用品", "日用类",
            "日化类", "日化", "纺织", "daily",
        ),
    ),
    (
        "food",
        ("食品", "食物", "食材", "生鲜类", "生鲜", "food"),
    ),
)

_TASK_SIMULATION_MARKERS = (
    "仿真环境", "仿真系统", "仿真任务", "仿真场景", "仿真",
    "模拟环境", "模拟系统", "模拟任务", "模拟场景",
    "虚拟环境", "虚拟任务", "虚拟场景",
    "仿真端", "模拟端", "虚拟端", "模拟", "虚拟",
    "线上环境", "线上任务", "线上", "simulation", "simulated", "virtual", "sim",
)

_TASK_PHYSICAL_MARKERS = (
    "物品领取区", "领取区", "实体环境", "实体场地", "实体任务", "实体部分",
    "实体车", "实体", "实车", "物理环境", "物理任务",
    "现实环境", "现实场地", "现实任务", "真实环境", "真实场地",
    "实体端", "现实", "现场", "实物", "实际环境", "实际场地",
    "线下环境", "线下任务", "线下",
    "physical", "realworld", "real-world",
)

_TASK_SHARED_TARGET_MARKERS = (
    "都要", "都需要", "都是", "均为", "均需", "同为", "相同", "同样", "同一",
    "一致", "共同", "both",
)

_TASK_DISTINCT_TARGET_MARKERS = (
    "不相同", "不同", "不一样", "分别", "distinct", "different",
)


def _semantic_text(text):
    return "".join(str(text or "").split()).lower()


def _find_semantic_mentions(text, alias_groups):
    """Return non-overlapping (value, start, end) aliases, preferring longest."""
    candidates = []
    for value, aliases in alias_groups:
        for alias in aliases:
            offset = 0
            while True:
                start = text.find(alias, offset)
                if start < 0:
                    break
                end = start + len(alias)
                candidates.append((value, start, end))
                offset = start + 1

    candidates.sort(key=lambda mention: (
        mention[1], -(mention[2] - mention[1]), mention[0]))
    mentions = []
    for candidate in candidates:
        if any(
                candidate[1] < selected[2] and candidate[2] > selected[1]
                for selected in mentions):
            continue
        mentions.append(candidate)
    mentions.sort(key=lambda mention: (mention[1], mention[2]))
    return mentions


def _category_association_score(text, marker, category, categories):
    if category[2] <= marker[1]:
        between_start, between_end = category[2], marker[1]
        gap = marker[1] - category[2]
        direction_penalty = 1
    elif marker[2] <= category[1]:
        between_start, between_end = marker[2], category[1]
        gap = category[1] - marker[2]
        direction_penalty = 0
    else:
        between_start = between_end = category[1]
        gap = 0
        direction_penalty = 0
    intervening = sum(
        1
        for other in categories
        if other is not category
        and other[1] >= between_start
        and other[2] <= between_end
    )
    clause_boundaries = sum(
        text.count(separator, between_start, between_end)
        for separator in ("，", ",", "。", ".", "；", ";", "！", "!", "？", "?", "：", ":")
    )
    return intervening, clause_boundaries, gap, direction_penalty, category[1]


def _closest_category_index(text, markers, categories, excluded=()):
    excluded = set(excluded)
    candidates = []
    for marker in markers:
        for index, category in enumerate(categories):
            if index in excluded:
                continue
            candidates.append((
                _category_association_score(text, marker, category, categories),
                index,
            ))
    if not candidates:
        return None
    return min(candidates)[1]


def _implicit_physical_index(categories, simulation_index):
    available = [
        index for index in range(len(categories)) if index != simulation_index
    ]
    if not available:
        return None
    if len(available) == 1:
        return available[0]

    simulation_category = categories[simulation_index][0]
    other_categories = {
        categories[index][0]
        for index in available
        if categories[index][0] != simulation_category
    }
    if len(other_categories) == 1:
        expected = next(iter(other_categories))
        return next(
            index for index in available if categories[index][0] == expected)

    available_categories = {categories[index][0] for index in available}
    if len(available_categories) == 1:
        return available[0]
    return None


def parse_category(text):
    compact = _semantic_text(text)
    mentions = _find_semantic_mentions(compact, _TASK_CATEGORY_ALIASES)
    return mentions[0][0] if mentions else None


def parse_task1_categories(text):
    """Extract physical and simulation targets from free-form task speech."""
    compact = _semantic_text(text)
    categories = _find_semantic_mentions(compact, _TASK_CATEGORY_ALIASES)
    simulation_markers = _find_semantic_mentions(
        compact, (("simulation", _TASK_SIMULATION_MARKERS),))
    if not simulation_markers:
        return parse_category(compact), None
    if not categories:
        return None, None

    physical_markers = _find_semantic_mentions(
        compact, (("physical", _TASK_PHYSICAL_MARKERS),))

    # "实体和仿真都要食品" states two targets with one category occurrence.
    if len(categories) == 1 and physical_markers:
        category = categories[0]
        shared_wording = any(
            marker in compact for marker in _TASK_SHARED_TARGET_MARKERS)
        distinct_wording = any(
            marker in compact for marker in _TASK_DISTINCT_TARGET_MARKERS)
        if shared_wording and not distinct_wording:
            return category[0], category[0]

    simulation_index = _closest_category_index(
        compact, simulation_markers, categories)
    if simulation_index is None:
        return None, None

    if physical_markers:
        physical_index = _closest_category_index(
            compact, physical_markers, categories, excluded=(simulation_index,))
    else:
        physical_index = _implicit_physical_index(categories, simulation_index)

    simulation_category = categories[simulation_index][0]
    if physical_index is None:
        return None, simulation_category
    return categories[physical_index][0], simulation_category


def task1_transcript_is_complete(pickup_category, sim_category, is_final):
    """Accept a task command only after IAT has finalized both target roles."""
    return bool(is_final and pickup_category and sim_category)


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


def task2_resumed_coverage_hint(memory, target_category, last_anchor,
                                anchor_count=9):
    """Prefer remembered target, otherwise continue through unvisited anchors."""
    preferred, skipped = task2_semantic_coverage_hint(memory, target_category)
    if preferred:
        return preferred, skipped
    try:
        last_anchor = int(last_anchor)
    except (TypeError, ValueError):
        last_anchor = 0
    anchor_count = max(0, int(anchor_count))
    if anchor_count and 1 <= last_anchor <= anchor_count:
        if last_anchor < anchor_count:
            preferred = last_anchor + 1
            skipped = tuple(sorted(
                set(skipped).union(range(1, last_anchor + 1))))
        else:
            # A complete first pass can only avoid rescanning when the target
            # was remembered. Fall back to a full pass if memory was empty.
            preferred = 0
    return preferred, skipped


def normalize_coverage_anchor_ids(value, anchor_count=9):
    """Return sorted, unique one-based anchor IDs from ROS list/string input."""
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = (value,)
    anchor_count = max(0, int(anchor_count))
    anchors = set()
    for item in values:
        try:
            anchor = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if 1 <= anchor <= anchor_count:
            anchors.add(anchor)
    return tuple(sorted(anchors))


def normalize_category(value):
    text = str(value or "").strip().lower()
    return OCR_CATEGORY_ALIASES.get(text) or parse_category(text)


def non_target_observation_is_actionable(
        target_category, observed_category, memory_confirmed,
        bbox_eligible, score, minimum_score, anchor):
    """Accept only a close, repeated non-target sign at a known anchor."""
    target = normalize_category(target_category)
    observed = normalize_category(observed_category)
    try:
        anchor_id = int(anchor)
        confidence = float(score)
        threshold = float(minimum_score)
    except (TypeError, ValueError):
        return False
    return bool(
        target and observed and observed != target and
        bool(memory_confirmed) and bool(bbox_eligible) and
        anchor_id > 0 and math.isfinite(confidence) and
        confidence >= threshold
    )


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
