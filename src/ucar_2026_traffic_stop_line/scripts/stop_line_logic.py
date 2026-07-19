#!/usr/bin/env python3
"""ROS-independent stop-line detection and calibration helpers."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import yaml


DEFAULT_DETECTION = {
    "roi_y_start_ratio": 0.45,
    "roi_y_end_ratio": 0.98,
    "white_s_max": 85,
    "white_v_min": 155,
    "gray_white_threshold": 175,
    "morph_kernel_size": 5,
    "min_area": 400.0,
    "max_area_ratio": 0.15,
    "min_fill_ratio": 0.35,
    "min_aspect_ratio": 4.0,
    "min_width_ratio": 0.35,
    "max_height_ratio": 0.18,
    "max_detection_angle_deg": 20.0,
}


def clamp(value, lower, upper):
    return max(float(lower), min(float(upper), float(value)))


def normalize_line_angle(angle_deg):
    angle = float(angle_deg)
    while angle >= 90.0:
        angle -= 180.0
    while angle < -90.0:
        angle += 180.0
    return angle


def _odd_kernel(value):
    size = max(1, int(value))
    return size if size % 2 else size + 1


def extract_white_mask(frame, params=None):
    cfg = dict(DEFAULT_DETECTION)
    cfg.update(params or {})
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(
        hsv,
        np.array((0, 0, int(cfg["white_v_min"])), dtype=np.uint8),
        np.array((180, int(cfg["white_s_max"]), 255), dtype=np.uint8),
    )
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, gray_mask = cv2.threshold(
        gray, int(cfg["gray_white_threshold"]), 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(hsv_mask, gray_mask)
    kernel_size = _odd_kernel(cfg["morph_kernel_size"])
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def detect_stop_line(frame, params=None, target_angle_deg=0.0):
    """Return the best strict horizontal white-line candidate or None."""
    if frame is None or frame.size == 0:
        return None, None
    cfg = dict(DEFAULT_DETECTION)
    cfg.update(params or {})
    height, width = frame.shape[:2]
    mask = extract_white_mask(frame, cfg)
    y0 = int(clamp(cfg["roi_y_start_ratio"], 0.0, 0.95) * height)
    y1 = int(clamp(cfg["roi_y_end_ratio"], 0.05, 1.0) * height)
    if y1 <= y0:
        return None, mask
    roi = mask[y0:y1, :]
    contours = cv2.findContours(
        roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    candidates = []
    image_area = float(width * height)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(cfg["min_area"]):
            continue
        if area / image_area > float(cfg["max_area_ratio"]):
            continue
        x, local_y, box_w, box_h = cv2.boundingRect(contour)
        if box_h <= 0 or box_w <= 0:
            continue
        width_ratio = box_w / float(width)
        height_ratio = box_h / float(height)
        aspect = box_w / float(box_h)
        fill = area / float(box_w * box_h)
        if width_ratio < float(cfg["min_width_ratio"]):
            continue
        if height_ratio > float(cfg["max_height_ratio"]):
            continue
        if aspect < float(cfg["min_aspect_ratio"]):
            continue
        if fill < float(cfg["min_fill_ratio"]):
            continue
        line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(np.asarray(line[0]).reshape(-1)[0])
        vy = float(np.asarray(line[1]).reshape(-1)[0])
        angle = normalize_line_angle(math.degrees(math.atan2(vy, vx)))
        angle_error = normalize_line_angle(angle - float(target_angle_deg))
        max_angle = float(cfg["max_detection_angle_deg"])
        if abs(angle_error) > max_angle:
            continue
        full_y = y0 + local_y
        center_y_ratio = (full_y + box_h * 0.5) / float(height)
        score = clamp(
            0.50 * width_ratio + 0.20 * fill +
            0.30 * (1.0 - abs(angle_error) / max(max_angle, 1e-6)),
            0.0, 1.0)
        candidates.append({
            "angle_deg": angle,
            "angle_error_deg": angle_error,
            "center_y_ratio": center_y_ratio,
            "width_ratio": width_ratio,
            "height_ratio": height_ratio,
            "fill_ratio": fill,
            "area": area,
            "confidence": score,
            "bbox": [int(x), int(full_y), int(box_w), int(box_h)],
        })
    if not candidates:
        return None, mask
    return max(candidates, key=lambda item: item["confidence"]), mask


def confirmed_window(values, required):
    return sum(bool(value) for value in values) >= max(1, int(required))


def approach_speed(y_error, far_speed=0.06, mid_speed=0.035,
                   near_speed=0.02, far_error=0.12, mid_error=0.04):
    error = float(y_error)
    if error > float(far_error):
        return float(far_speed)
    if error > float(mid_error):
        return float(mid_speed)
    return float(near_speed)


def safety_failure(now, image_at, odom_at, scan_at, timeout_sec,
                   front_distance, obstacle_distance):
    timeout = max(0.0, float(timeout_sec))
    for name, stamp in (("image", image_at), ("odom", odom_at), ("scan", scan_at)):
        if float(now) - float(stamp) > timeout:
            return name + "_stale"
    if float(front_distance) < float(obstacle_distance):
        return "front_obstacle_{:.3f}m".format(float(front_distance))
    return ""


def target_position_state(y_error, tolerance, overshoot_margin):
    error = float(y_error)
    # Once the line has passed the acceptable target band, stop immediately.
    # overshoot_margin may only tighten this boundary; it must never permit
    # another forward command after crossing the calibrated target row.
    overshoot_limit = min(
        abs(float(tolerance)), abs(float(overshoot_margin)))
    if error < -overshoot_limit:
        return "overshoot"
    if abs(error) <= abs(float(tolerance)):
        return "target"
    return "approach"


def staging_pose(reference_x, reference_y, reference_yaw, backoff):
    distance = max(0.0, float(backoff))
    yaw = float(reference_yaw)
    return (
        float(reference_x) - distance * math.cos(yaw),
        float(reference_y) - distance * math.sin(yaw),
        yaw,
    )


def load_calibration(path):
    expanded = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(expanded):
        return None
    with open(expanded, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    required = ("target_y_ratio", "target_angle_deg", "target_front_gap_m")
    if any(key not in data for key in required):
        return None
    target_y = float(data["target_y_ratio"])
    if not 0.0 < target_y < 1.0:
        return None
    return {
        "target_y_ratio": target_y,
        "target_angle_deg": float(data["target_angle_deg"]),
        "target_front_gap_m": float(data["target_front_gap_m"]),
        "image_width": int(data.get("image_width", 0)),
        "image_height": int(data.get("image_height", 0)),
        "sample_count": int(data.get("sample_count", 0)),
    }


def save_calibration(path, detections, target_front_gap_m, image_shape):
    valid = [item for item in detections if item is not None]
    if not valid:
        raise ValueError("no valid stop-line detections to calibrate")
    target_y = float(np.median([item["center_y_ratio"] for item in valid]))
    target_angle = float(np.median([item["angle_deg"] for item in valid]))
    height, width = image_shape[:2]
    payload = {
        "target_y_ratio": target_y,
        "target_angle_deg": target_angle,
        "target_front_gap_m": float(target_front_gap_m),
        "image_width": int(width),
        "image_height": int(height),
        "sample_count": len(valid),
    }
    expanded = os.path.abspath(os.path.expanduser(str(path)))
    parent = os.path.dirname(expanded)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(expanded, "w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=True)
    return payload
