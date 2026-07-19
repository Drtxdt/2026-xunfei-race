"""ROS-independent safety logic for the strict post-warehouse mission."""

from __future__ import annotations

from bisect import bisect_right
import math


TRAFFIC_CLASS_TO_DECISION = {
    "green_left": "left",
    "green_right": "right",
    "green_straight": "straight",
    "red_light": "stop",
}

TRACK_CONFIG = {
    "left": ("track_end_stop.launch", "/track_end_stop/status", "finish"),
    "right": (
        "right_track_end_stop.launch",
        "/right_track_end_stop/status",
        "right_finish",
    ),
    "straight": (
        "stable_right_track_end_stop.launch",
        "/stable_right_track_end_stop/status",
        "stable_right_finish",
    ),
}


def forward_progress(start_pose, current_pose):
    """Return odometry displacement along the starting heading."""
    start_x, start_y, start_yaw = (float(value) for value in start_pose)
    current_x, current_y = (float(value) for value in current_pose[:2])
    delta_x = current_x - start_x
    delta_y = current_y - start_y
    return delta_x * math.cos(start_yaw) + delta_y * math.sin(start_yaw)


def lowest_horizontal_band(row_occupancies, min_occupancy, max_band_rows,
                           min_band_rows=2):
    """Return the lowest credible wide horizontal band as (start, end)."""
    threshold = float(min_occupancy)
    max_rows = max(1, int(max_band_rows))
    min_rows = max(1, int(min_band_rows))
    runs = []
    start = None
    for index, occupancy in enumerate(row_occupancies):
        if float(occupancy) >= threshold:
            if start is None:
                start = index
            continue
        if start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(row_occupancies) - 1))
    credible = [
        run for run in runs
        if min_rows <= run[1] - run[0] + 1 <= max_rows
    ]
    return max(credible, key=lambda run: run[1]) if credible else None


class DistanceCalibration:
    """Monotonic image-row to front-bumper distance calibration."""

    def __init__(self, points):
        parsed = [(float(row), float(distance)) for row, distance in points]
        if len(parsed) < 2:
            raise ValueError("distance_calibration requires at least two points")
        rows = [item[0] for item in parsed]
        distances = [item[1] for item in parsed]
        if any(not 0.0 <= row <= 1.0 for row in rows):
            raise ValueError("calibration row ratios must be within [0, 1]")
        if any(distance < 0.0 for distance in distances):
            raise ValueError("calibration distances must be non-negative")
        if any(rows[index] >= rows[index + 1] for index in range(len(rows) - 1)):
            raise ValueError("calibration row ratios must increase strictly")
        if any(
            distances[index] <= distances[index + 1]
            for index in range(len(distances) - 1)
        ):
            raise ValueError("distance must decrease as the line moves down the image")
        self._points = parsed
        self._rows = rows

    def distance_for_ratio(self, row_ratio):
        row = float(row_ratio)
        if row < self._rows[0] or row > self._rows[-1]:
            return None
        if row == self._rows[-1]:
            return self._points[-1][1]
        upper = bisect_right(self._rows, row)
        lower = max(0, upper - 1)
        row0, distance0 = self._points[lower]
        row1, distance1 = self._points[upper]
        fraction = (row - row0) / (row1 - row0)
        return distance0 + fraction * (distance1 - distance0)


class ApproachPolicy:
    """Conservative longitudinal command selected from measured clearance."""

    def __init__(
        self,
        target_min_m,
        target_max_m,
        absolute_max_m,
        calibration_error_m,
        speed_far=0.10,
        speed_medium=0.06,
        speed_near=0.05,
        speed_creep=0.045,
    ):
        self.target_min_m = float(target_min_m)
        self.target_max_m = float(target_max_m)
        self.absolute_max_m = float(absolute_max_m)
        self.calibration_error_m = float(calibration_error_m)
        if not 0.0 < self.target_min_m <= self.target_max_m:
            raise ValueError("target distance band is invalid")
        if self.target_max_m + self.calibration_error_m > self.absolute_max_m + 1e-9:
            raise ValueError("target plus calibration error exceeds absolute maximum")
        self.speed_far = float(speed_far)
        self.speed_medium = float(speed_medium)
        self.speed_near = float(speed_near)
        self.speed_creep = float(speed_creep)

    def in_target_band(self, distance_m):
        if distance_m is None:
            return False
        distance = float(distance_m)
        return self.target_min_m <= distance <= self.target_max_m

    def command_for_distance(self, distance_m):
        if distance_m is None:
            return 0.0
        distance = float(distance_m)
        if distance <= self.target_max_m:
            return 0.0
        if distance > 0.35:
            return self.speed_far
        if distance > 0.18:
            return self.speed_medium
        if distance > 0.11:
            return self.speed_near
        return self.speed_creep


class ConsecutiveBandFilter:
    def __init__(self, required_frames, lower, upper):
        self.required_frames = max(1, int(required_frames))
        self.lower = float(lower)
        self.upper = float(upper)
        self.hits = 0

    def reset(self):
        self.hits = 0

    def push(self, value):
        if value is not None and self.lower <= float(value) <= self.upper:
            self.hits += 1
        else:
            self.hits = 0
        return self.hits >= self.required_frames


def traffic_decision_from_payload(payload):
    consensus = payload.get("consensus") or {}
    if not consensus.get("active"):
        return None
    return TRAFFIC_CLASS_TO_DECISION.get(consensus.get("class_name"))


def track_launch_for_decision(decision):
    normalized = str(decision or "").strip().lower()
    if normalized not in TRACK_CONFIG:
        raise ValueError("traffic decision must be left, right, or straight")
    return TRACK_CONFIG[normalized]


def valid_stop_line_geometry(
    width_ratio,
    height_ratio,
    fill_ratio,
    bottom_ratio,
    min_width_ratio=0.45,
    max_height_ratio=0.12,
    min_fill_ratio=0.55,
    min_bottom_ratio=0.55,
):
    """Reject objects that cannot be a wide horizontal ground stop line."""
    width = float(width_ratio)
    height = float(height_ratio)
    fill = float(fill_ratio)
    bottom = float(bottom_ratio)
    return (
        width >= float(min_width_ratio)
        and 0.0 < height <= float(max_height_ratio)
        and width / height >= 5.0
        and fill >= float(min_fill_ratio)
        and bottom >= float(min_bottom_ratio)
    )
