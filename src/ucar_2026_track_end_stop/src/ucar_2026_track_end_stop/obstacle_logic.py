"""ROS-independent obstacle-avoidance state machine."""

from __future__ import annotations

import math


def normalize_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def local_progress(origin, pose):
    ox, oy, oyaw = (float(value) for value in origin)
    px, py = (float(value) for value in pose[:2])
    dx = px - ox
    dy = py - oy
    return (
        dx * math.cos(oyaw) + dy * math.sin(oyaw),
        -dx * math.sin(oyaw) + dy * math.cos(oyaw),
    )


class ObstacleAvoidanceController(object):
    """Sidestep a board, pass it, and return to the original track line."""

    def __init__(
            self, trigger_distance=0.58, clear_distance=0.72,
            confirm_scans=3, min_side_clearance=0.45,
            min_shift=0.24, max_shift=0.82, shift_speed=0.12,
            pass_distance=0.58, pass_speed=0.12,
            return_tolerance=0.025, stop_hold=0.20,
            reacquire_duration=1.2, reacquire_speed=0.07,
            enable_delay=4.0, emergency_distance=0.16):
        self.trigger_distance = float(trigger_distance)
        self.clear_distance = float(clear_distance)
        self.confirm_scans = max(1, int(confirm_scans))
        self.min_side_clearance = float(min_side_clearance)
        self.min_shift = float(min_shift)
        self.max_shift = float(max_shift)
        self.shift_speed = float(shift_speed)
        self.pass_distance = float(pass_distance)
        self.pass_speed = float(pass_speed)
        self.return_tolerance = float(return_tolerance)
        self.stop_hold = float(stop_hold)
        self.reacquire_duration = float(reacquire_duration)
        self.reacquire_speed = float(reacquire_speed)
        self.enable_delay = float(enable_delay)
        self.emergency_distance = float(emergency_distance)
        self.state = "FOLLOW"
        self.started_at = None
        self.state_started_at = None
        self.origin = None
        self.pass_started_forward = 0.0
        self.side_sign = 0.0
        self.obstacle_hits = 0
        self.fault = ""

    def fail(self, reason):
        self.state = "FAULT"
        self.fault = str(reason)
        return 0.0, 0.0, 0.0

    def update(self, now, pose, clearances, raw_command):
        now = float(now)
        if self.started_at is None:
            self.started_at = now
        front = float(clearances.get("front", float("inf")))
        left = float(clearances.get("left", 0.0))
        right = float(clearances.get("right", 0.0))
        raw_x, raw_y, raw_yaw = (float(value) for value in raw_command)
        if self.state == "FAULT":
            return 0.0, 0.0, 0.0
        if self.state == "COMPLETE":
            return raw_x, raw_y, raw_yaw
        if self.state == "FOLLOW":
            eligible = now - self.started_at >= self.enable_delay and raw_x > 0.02
            if eligible and front <= self.trigger_distance:
                self.obstacle_hits += 1
            else:
                self.obstacle_hits = 0
            if self.obstacle_hits < self.confirm_scans:
                return raw_x, raw_y, raw_yaw
            if left < self.min_side_clearance and right < self.min_side_clearance:
                return self.fail("no collision-free side around board")
            self.side_sign = 1.0 if left >= right else -1.0
            self.origin = tuple(float(value) for value in pose)
            self.state = "STOP"
            self.state_started_at = now
            return 0.0, 0.0, 0.0

        if self.origin is None:
            return self.fail("avoidance origin missing")
        forward, lateral = local_progress(self.origin, pose)
        signed_lateral = self.side_sign * lateral
        if self.state == "STOP":
            if now - self.state_started_at >= self.stop_hold:
                self.state = "SHIFT_OUT"
                self.state_started_at = now
            return 0.0, 0.0, 0.0
        if self.state == "SHIFT_OUT":
            if signed_lateral > self.max_shift:
                return self.fail("board did not clear within lateral limit")
            if signed_lateral >= self.min_shift and front >= self.clear_distance:
                self.state = "PASS"
                self.state_started_at = now
                self.pass_started_forward = forward
                return 0.0, 0.0, 0.0
            return 0.0, self.side_sign * self.shift_speed, 0.0
        if self.state == "PASS":
            if front <= self.emergency_distance:
                return self.fail("unexpected obstacle while passing board")
            if forward - self.pass_started_forward >= self.pass_distance:
                self.state = "SHIFT_BACK"
                self.state_started_at = now
                return 0.0, 0.0, 0.0
            return self.pass_speed, 0.0, 0.0
        if self.state == "SHIFT_BACK":
            if signed_lateral <= self.return_tolerance:
                self.state = "REACQUIRE"
                self.state_started_at = now
                return 0.0, 0.0, 0.0
            return 0.0, -self.side_sign * self.shift_speed, 0.0
        if self.state == "REACQUIRE":
            if now - self.state_started_at >= self.reacquire_duration:
                self.state = "COMPLETE"
                return raw_x, raw_y, raw_yaw
            return max(self.reacquire_speed, raw_x * 0.6), raw_y * 0.6, raw_yaw * 0.6
        return self.fail("unknown avoidance state")
