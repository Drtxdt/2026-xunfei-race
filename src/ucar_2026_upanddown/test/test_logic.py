# -*- coding: utf-8 -*-
"""Unit tests for ucar_2026_upanddown pure logic."""

from __future__ import annotations

import math
import unittest

from ucar_2026_upanddown.logic import (
    HeadingHoldController,
    PitchFilter,
    RampSegmenter,
    SoftSpeedProfile,
    distance_budget_exceeded,
    normalize_angle,
    path_length,
    rpy_from_quaternion,
    rotation_steps,
    validate_ramp_config,
)


class QuaternionTests(unittest.TestCase):
    def test_level_quaternion_has_zero_pitch(self):
        yaw = 0.7
        quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        roll, pitch, out_yaw = rpy_from_quaternion(*quat)
        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(out_yaw, yaw, places=6)

    def test_pure_pitch_quaternion(self):
        pitch = math.radians(22.0)
        quat = (0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2))
        _roll, out_pitch, _yaw = rpy_from_quaternion(*quat)
        self.assertAlmostEqual(math.degrees(out_pitch), 22.0, places=4)

    def test_zero_quaternion_is_nan(self):
        roll, pitch, yaw = rpy_from_quaternion(0.0, 0.0, 0.0, 0.0)
        self.assertTrue(math.isnan(roll))
        self.assertTrue(math.isnan(pitch))
        self.assertTrue(math.isnan(yaw))


class SegmenterTests(unittest.TestCase):
    def make(self, **kwargs):
        return RampSegmenter(**kwargs)

    def feed(self, segmenter, values):
        for value in values:
            state = segmenter.update(value)
        return state

    def test_full_ramp_sequence(self):
        seg = self.make(confirm_frames=2)
        self.assertEqual(
            self.feed(seg, [0.0, 0.5, 0.0]), "level")
        self.assertEqual(self.feed(seg, [22.0, 22.0]), "up")
        self.assertEqual(self.feed(seg, [0.5, 0.0]), "plateau")
        self.assertEqual(self.feed(seg, [-25.0, -25.0]), "down")
        self.assertEqual(self.feed(seg, [0.2, -0.2]), "complete")
        self.assertTrue(seg.pitch_signature_valid(12.0, -12.0))
        self.assertAlmostEqual(seg.max_pitch_deg, 22.0)
        self.assertAlmostEqual(seg.min_pitch_deg, -25.0)

    def test_single_frame_spike_does_not_transition(self):
        seg = self.make(confirm_frames=3)
        for value in (0.0, 22.0, 0.0, 0.0):
            seg.update(value)
        self.assertEqual(seg.state, "level")

    def test_plateau_noise_does_not_rewind(self):
        seg = self.make(confirm_frames=2)
        self.feed(seg, [22.0, 22.0, 0.0, 0.0])
        self.assertEqual(seg.state, "plateau")
        # Slight pitch wiggle on the plateau must not matter.
        self.feed(seg, [5.0, 2.0])
        self.assertEqual(seg.state, "plateau")

    def test_level_cannot_jump_to_down(self):
        seg = self.make(confirm_frames=1)
        self.feed(seg, [-25.0] * 5)
        self.assertEqual(seg.state, "level")

    def test_down_cannot_complete_on_uphill_pitch(self):
        seg = self.make(confirm_frames=1)
        self.feed(seg, [22.0])
        self.feed(seg, [0.0])
        self.assertEqual(seg.state, "plateau")
        self.feed(seg, [20.0])
        self.assertEqual(seg.state, "plateau")

    def test_invalid_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            self.make(up_enter_deg=2.0, up_exit_deg=3.0)
        with self.assertRaises(ValueError):
            self.make(down_enter_deg=-2.0, down_exit_deg=-3.0)

    def test_nonfinite_sample_ignored(self):
        seg = self.make(confirm_frames=2)
        seg.update(float("nan"))
        self.assertEqual(seg.state, "level")


class FilterTests(unittest.TestCase):
    def test_median_filter_rejects_spike(self):
        filt = PitchFilter(window=5)
        for value in (1.0, 1.1, 9.0, 1.2, 0.9):
            filt.push(value)
        self.assertLess(filt.value(), 2.0)

    def test_empty_filter_returns_none(self):
        self.assertIsNone(PitchFilter().value())


class SpeedProfileTests(unittest.TestCase):
    def test_accel_limit_respected(self):
        profile = SoftSpeedProfile(accel_limit=0.25, decel_limit=0.40)
        speed = profile.update(0.16, 0.05)
        self.assertAlmostEqual(speed, 0.0125, places=6)

    def test_decel_limit_respected(self):
        profile = SoftSpeedProfile(accel_limit=0.25, decel_limit=0.40)
        profile.reset(0.20)
        speed = profile.update(0.0, 0.05)
        self.assertAlmostEqual(speed, 0.18, places=6)

    def test_target_clamped_at_zero(self):
        profile = SoftSpeedProfile()
        profile.reset(0.1)
        self.assertEqual(profile.update(-1.0, 1.0), 0.0)


class HeadingControllerTests(unittest.TestCase):
    def test_deadband_outputs_zero(self):
        ctrl = HeadingHoldController(deadband_rad=0.03)
        self.assertEqual(ctrl.command(0.02), 0.0)

    def test_clamp_and_sign(self):
        ctrl = HeadingHoldController(kp=10.0, max_angular=0.35)
        self.assertAlmostEqual(ctrl.command(1.0), 0.35, places=6)
        self.assertAlmostEqual(ctrl.command(-1.0), -0.35, places=6)

    def test_wrap_around_error(self):
        ctrl = HeadingHoldController(kp=1.0, max_angular=10.0)
        error = normalize_angle(math.pi - 0.1 - (-math.pi + 0.1))
        self.assertAlmostEqual(error, -0.2, delta=1e-6)


class GeometryTests(unittest.TestCase):
    def test_path_length(self):
        self.assertAlmostEqual(
            path_length((0.0, 0.0, 0.0), (0.3, 0.4, 0.0)), 0.5)

    def test_distance_budget(self):
        self.assertFalse(distance_budget_exceeded(2.0, 1.5, 0.9))
        self.assertTrue(distance_budget_exceeded(2.5, 1.5, 0.9))

    def test_rotation_steps_bounds(self):
        steps = rotation_steps(2 * math.pi, math.radians(20.1))
        self.assertLessEqual(max(steps), math.radians(20.1) + 1e-9)
        self.assertAlmostEqual(sum(steps), 2 * math.pi, places=6)


class ConfigValidationTests(unittest.TestCase):
    def test_valid_config_passes(self):
        self.assertTrue(validate_ramp_config(8.0, 3.0, -8.0, -3.0, 1.5, 0.28))

    def test_bad_length_rejected(self):
        with self.assertRaises(ValueError):
            validate_ramp_config(8.0, 3.0, -8.0, -3.0, 0.0, 0.28)
        with self.assertRaises(ValueError):
            validate_ramp_config(8.0, 3.0, -8.0, -3.0, 1.5, -0.1)


if __name__ == "__main__":
    unittest.main()
