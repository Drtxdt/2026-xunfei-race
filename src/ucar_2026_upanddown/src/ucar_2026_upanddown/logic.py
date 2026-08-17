# -*- coding: utf-8 -*-
"""ROS-independent logic for the national ramp (up-flat-down) traversal.

The national competition inserts a 1.5 m ramp (22 deg up, flat, 25 deg down)
along the task-1 navigation route.  A 2D lidar cannot see anything useful on
the ramp, so the traverse is driven open-loop with:

* IMU pitch -> segment the ramp (uphill / plateau / downhill / complete)
* IMU or odometry yaw -> heading hold so the robot stays on the ramp axis
* odometry path length -> exit margin and distance watchdog
"""

from __future__ import annotations

import math


RAMP_LEVEL = "level"
RAMP_UP = "up"
RAMP_PLATEAU = "plateau"
RAMP_DOWN = "down"
RAMP_COMPLETE = "complete"

SEGMENT_ORDER = (RAMP_LEVEL, RAMP_UP, RAMP_PLATEAU, RAMP_DOWN, RAMP_COMPLETE)


def normalize_angle(angle):
    """Normalize an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def rpy_from_quaternion(x, y, z, w):
    """Return (roll, pitch, yaw) in radians for a unit quaternion (ZYX)."""
    x = float(x)
    y = float(y)
    z = float(z)
    w = float(w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9 or not math.isfinite(norm):
        return float("nan"), float("nan"), float("nan")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class PitchFilter:
    """Median filter over a small window of pitch samples in degrees."""

    def __init__(self, window=5):
        self.window = max(1, int(window))
        self.samples = []

    def reset(self):
        self.samples = []

    def push(self, pitch_deg):
        value = float(pitch_deg)
        if not math.isfinite(value):
            return self.value()
        self.samples.append(value)
        if len(self.samples) > self.window:
            self.samples.pop(0)
        return self.value()

    def value(self):
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        count = len(ordered)
        middle = count // 2
        if count % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])


class RampSegmenter:
    """Segment the up-flat-down ramp from filtered IMU pitch.

    Strict sequence: level -> up -> plateau -> down -> complete.  Each
    transition must hold for ``confirm_frames`` consecutive samples before it
    latches, which rejects single-frame noise and pitch oscillation at the
    ramp joints.
    """

    def __init__(
        self,
        up_enter_deg=8.0,
        up_exit_deg=3.0,
        down_enter_deg=-8.0,
        down_exit_deg=-3.0,
        confirm_frames=3,
    ):
        self.up_enter_deg = float(up_enter_deg)
        self.up_exit_deg = float(up_exit_deg)
        self.down_enter_deg = float(down_enter_deg)
        self.down_exit_deg = float(down_exit_deg)
        self.confirm_frames = max(1, int(confirm_frames))
        self._validate()

        self.state = RAMP_LEVEL
        self.max_pitch_deg = 0.0
        self.min_pitch_deg = 0.0
        self.up_seen = False
        self.down_seen = False
        self._pending = None
        self._hits = 0

    def _validate(self):
        if not 0.0 <= self.up_exit_deg < self.up_enter_deg:
            raise ValueError("require 0 <= up_exit_deg < up_enter_deg")
        if not self.down_enter_deg < self.down_exit_deg <= 0.0:
            raise ValueError("require down_enter_deg < down_exit_deg <= 0")

    def reset(self):
        self.state = RAMP_LEVEL
        self.max_pitch_deg = 0.0
        self.min_pitch_deg = 0.0
        self.up_seen = False
        self.down_seen = False
        self._pending = None
        self._hits = 0

    def _offer(self, candidate, satisfied):
        if satisfied:
            if self._pending == candidate:
                self._hits += 1
            else:
                self._pending = candidate
                self._hits = 1
            return self._hits >= self.confirm_frames
        if self._pending == candidate:
            self._pending = None
            self._hits = 0
        return False

    def update(self, pitch_deg):
        """Feed one filtered pitch sample; return the current segment state."""
        pitch = float(pitch_deg)
        if not math.isfinite(pitch):
            return self.state

        if pitch > self.max_pitch_deg:
            self.max_pitch_deg = pitch
        if pitch < self.min_pitch_deg:
            self.min_pitch_deg = pitch
        if pitch >= self.up_enter_deg:
            self.up_seen = True
        if pitch <= self.down_enter_deg:
            self.down_seen = True

        if self.state == RAMP_LEVEL:
            if self._offer(RAMP_UP, pitch >= self.up_enter_deg):
                self.state = RAMP_UP
        elif self.state == RAMP_UP:
            if self._offer(RAMP_PLATEAU, abs(pitch) <= self.up_exit_deg):
                self.state = RAMP_PLATEAU
        elif self.state == RAMP_PLATEAU:
            if self._offer(RAMP_DOWN, pitch <= self.down_enter_deg):
                self.state = RAMP_DOWN
        elif self.state == RAMP_DOWN:
            if self._offer(
                    RAMP_COMPLETE,
                    abs(pitch) <= abs(self.down_exit_deg)):
                self.state = RAMP_COMPLETE
        return self.state

    def pitch_signature_valid(self, min_up_deg=10.0, min_down_deg=-10.0):
        """True when both slopes were actually observed on the traverse."""
        return self.max_pitch_deg >= float(min_up_deg) and \
            self.min_pitch_deg <= float(min_down_deg)


class SoftSpeedProfile:
    """Rate-limited speed command (trapezoid) so the ramp wheels never slip."""

    def __init__(self, accel_limit=0.25, decel_limit=0.40):
        if float(accel_limit) <= 0.0 or float(decel_limit) <= 0.0:
            raise ValueError("accel/decel limits must be positive")
        self.accel_limit = float(accel_limit)
        self.decel_limit = float(decel_limit)
        self.current = 0.0

    def reset(self, speed=0.0):
        speed = max(0.0, float(speed))
        self.current = speed

    def update(self, target, dt):
        target = max(0.0, float(target))
        dt = max(0.0, float(dt))
        if target > self.current:
            self.current = min(target, self.current + self.accel_limit * dt)
        else:
            self.current = max(target, self.current - self.decel_limit * dt)
        return self.current


class HeadingHoldController:
    """Proportional heading hold with a dead band and angular clamp."""

    def __init__(self, kp=1.6, max_angular=0.35, deadband_rad=0.026):
        if float(kp) < 0.0 or float(max_angular) <= 0.0:
            raise ValueError("kp must be non-negative and max_angular positive")
        self.kp = float(kp)
        self.max_angular = float(max_angular)
        self.deadband_rad = abs(float(deadband_rad))

    def command(self, yaw_error):
        error = normalize_angle(yaw_error)
        if abs(error) <= self.deadband_rad:
            return 0.0
        angular = self.kp * error
        return max(-self.max_angular, min(self.max_angular, angular))


def path_length(start_pose, current_pose):
    """Planar odometry path length between two (x, y, yaw) poses."""
    return math.hypot(
        float(current_pose[0]) - float(start_pose[0]),
        float(current_pose[1]) - float(start_pose[1]),
    )


def distance_budget_exceeded(traveled_m, nominal_m, margin_m):
    """Watchdog: the traverse must never exceed nominal length + margin."""
    return float(traveled_m) > float(nominal_m) + float(margin_m)


def rotation_steps(total_rad, step_rad):
    """Split a rotation arc into bounded steps (used by the QR spin scan)."""
    total = abs(float(total_rad))
    step = abs(float(step_rad))
    if step <= 1e-9:
        raise ValueError("step must be positive")
    steps = []
    remaining = total
    while remaining > 1e-6:
        current = min(step, remaining)
        steps.append(current)
        remaining -= current
    return tuple(steps)


def validate_ramp_config(
    up_enter_deg,
    up_exit_deg,
    down_enter_deg,
    down_exit_deg,
    nominal_ramp_length_m,
    exit_extra_m,
):
    """Validate the parameter set early so a bad config cannot start motion."""
    RampSegmenter(
        up_enter_deg, up_exit_deg, down_enter_deg, down_exit_deg, 1)
    nominal = float(nominal_ramp_length_m)
    extra = float(exit_extra_m)
    if nominal <= 0.0:
        raise ValueError("nominal ramp length must be positive")
    if extra < 0.0:
        raise ValueError("exit extra distance must be non-negative")
    return True
