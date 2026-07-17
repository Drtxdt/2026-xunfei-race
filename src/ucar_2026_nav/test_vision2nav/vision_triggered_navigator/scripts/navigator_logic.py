#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ROS-independent helpers for coverage-oriented factory search."""

from __future__ import division

import math


def latch_trigger(already_latched):
    """Return ``(latched, accepted_now)`` for an idempotent one-shot trigger."""
    if bool(already_latched):
        return True, False
    return True, True


def normalize_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def build_quadrilateral_walls(corners):
    """Build measured wall segments with inward unit normals.

    Corner order is ``top-left, top-right, bottom-left, bottom-right`` as used
    by the existing YAML.  The arena centroid selects the inward side, so the
    calculation remains correct when the mapped walls are slightly skewed.
    """
    if len(corners) != 4:
        raise ValueError("vision_rect_corners must contain exactly four points")
    points = [(float(point[0]), float(point[1])) for point in corners]
    top_left, top_right, bottom_left, bottom_right = points
    centroid = (
        sum(point[0] for point in points) / 4.0,
        sum(point[1] for point in points) / 4.0,
    )
    segments = [
        ("left", top_left, bottom_left),
        ("right", top_right, bottom_right),
        ("bottom", bottom_left, bottom_right),
        ("top", top_left, top_right),
    ]
    walls = []
    for name, start, end in segments:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            raise ValueError("wall {} has zero length".format(name))
        normal = (-dy / length, dx / length)
        midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        toward_center = (centroid[0] - midpoint[0], centroid[1] - midpoint[1])
        if normal[0] * toward_center[0] + normal[1] * toward_center[1] < 0.0:
            normal = (-normal[0], -normal[1])
        walls.append((name, start, end, normal))
    return walls


def ray_segment_intersection(origin, direction, start, end):
    """Return positive ray parameter ``t`` for a 2-D segment intersection."""
    ox, oy = [float(value) for value in origin]
    dx, dy = [float(value) for value in direction]
    ax, ay = [float(value) for value in start]
    bx, by = [float(value) for value in end]
    vx = bx - ax
    vy = by - ay
    denominator = -vx * dy + vy * dx
    if abs(denominator) < 1e-9:
        return None
    wx = ox - ax
    wy = oy - ay
    ray_t = (vx * wy - vy * wx) / denominator
    segment_u = (-wx * dy + wy * dx) / denominator
    if ray_t <= 1e-9 or segment_u < -1e-6 or segment_u > 1.0 + 1e-6:
        return None
    return ray_t


def parking_goal_from_wall(wall_point, inward_normal, offset,
                           normal_offset=0.0, tangent_offset=0.0):
    """Return a continuous parking pose from a measured wall intersection."""
    ix, iy = [float(value) for value in wall_point]
    nx, ny = [float(value) for value in inward_normal]
    length = math.hypot(nx, ny)
    if length <= 1e-9:
        raise ValueError("inward normal must be non-zero")
    nx /= length
    ny /= length
    tx, ty = -ny, nx
    normal_distance = float(offset) + float(normal_offset)
    gx = ix + nx * normal_distance + tx * float(tangent_offset)
    gy = iy + ny * normal_distance + ty * float(tangent_offset)
    yaw = math.atan2(-ny, -nx)
    return gx, gy, yaw


def split_scan_angle(total_angle, step_angle):
    """Split one configured sweep into fixed steps while preserving its total angle."""
    remaining = max(0.0, float(total_angle))
    step = max(1e-6, float(step_angle))
    result = []
    while remaining > 1e-6:
        current = min(step, remaining)
        result.append(current)
        remaining -= current
    return result


def scan_dwell_deadline(started_at, dwell_sec, candidate_at,
                        candidate_hold_sec, max_dwell_sec):
    """Return a bounded dwell deadline, extending it for a fresh OCR candidate."""
    started_at = float(started_at)
    deadline = started_at + max(0.0, float(dwell_sec))
    candidate_at = float(candidate_at or 0.0)
    if candidate_at >= started_at:
        deadline = max(deadline, candidate_at + max(0.0, float(candidate_hold_sec)))
    return min(deadline, started_at + max(0.0, float(max_dwell_sec)))


def build_observation_candidates(x, y, offsets, bounds, min_wall_clearance):
    """Return unique candidate positions that remain safely inside the arena."""
    x_min, x_max, y_min, y_max = [float(value) for value in bounds]
    clearance = max(0.0, float(min_wall_clearance))
    candidates = []
    for dx, dy in offsets:
        cx = float(x) + float(dx)
        cy = float(y) + float(dy)
        if cx < x_min + clearance or cx > x_max - clearance:
            continue
        if cy < y_min + clearance or cy > y_max - clearance:
            continue
        key = (round(cx, 6), round(cy, 6))
        if key not in [(round(px, 6), round(py, 6)) for px, py in candidates]:
            candidates.append((cx, cy))
    return candidates


def center_angular_command(error, tolerance, min_speed, max_speed, steering_sign=-1.0):
    """Compute a bounded angular command from normalized horizontal image error."""
    error = float(error)
    tolerance = abs(float(tolerance))
    if abs(error) <= tolerance:
        return 0.0
    min_speed = abs(float(min_speed))
    max_speed = max(min_speed, abs(float(max_speed)))
    magnitude = min(max_speed, max(min_speed, abs(error) * max_speed))
    return (1.0 if float(steering_sign) >= 0.0 else -1.0) * math.copysign(magnitude, error)


def center_step_angle(error, tolerance, fine_threshold,
                      coarse_step_angle, fine_step_angle):
    """Return the next closed-loop centering step angle in radians."""
    magnitude = abs(float(error))
    if magnitude <= abs(float(tolerance)):
        return 0.0
    if magnitude <= abs(float(fine_threshold)):
        return abs(float(fine_step_angle))
    return abs(float(coarse_step_angle))


def parking_footprint_margins(pose, wall_point, inward_normal,
                              box_width, box_depth,
                              footprint_half_length, footprint_half_width,
                              margin=0.0):
    """Return wall-frame footprint coordinates and remaining box margins.

    ``wall_point`` is the middle of the box edge touching the wall.  Positive
    normal distance points into the arena; tangent distance is measured along
    the wall.  The check is deliberately based on the full navigation
    footprint, which is more conservative than checking only wheel centres.
    """
    px, py, yaw = [float(value) for value in pose]
    wx, wy = [float(value) for value in wall_point]
    nx, ny = [float(value) for value in inward_normal]
    normal_length = math.hypot(nx, ny)
    if normal_length <= 1e-9:
        return {"inside": False, "error": "zero inward normal", "corners": []}
    nx /= normal_length
    ny /= normal_length
    tx, ty = -ny, nx

    half_length = abs(float(footprint_half_length))
    half_width = abs(float(footprint_half_width))
    width_limit = abs(float(box_width)) * 0.5 - max(0.0, float(margin))
    depth_min = max(0.0, float(margin))
    depth_max = abs(float(box_depth)) - max(0.0, float(margin))
    if width_limit <= 0.0 or depth_max <= depth_min:
        return {"inside": False, "error": "invalid box dimensions", "corners": []}

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    corners = []
    for local_x in (-half_length, half_length):
        for local_y in (-half_width, half_width):
            corner_x = px + local_x * cos_yaw - local_y * sin_yaw
            corner_y = py + local_x * sin_yaw + local_y * cos_yaw
            delta_x = corner_x - wx
            delta_y = corner_y - wy
            normal_distance = delta_x * nx + delta_y * ny
            tangent_distance = delta_x * tx + delta_y * ty
            corners.append({
                "x": corner_x,
                "y": corner_y,
                "normal": normal_distance,
                "tangent": tangent_distance,
                "near_margin": normal_distance - depth_min,
                "far_margin": depth_max - normal_distance,
                "side_margin": width_limit - abs(tangent_distance),
            })
    normal_min = min(item["normal"] for item in corners)
    normal_max = max(item["normal"] for item in corners)
    tangent_min = min(item["tangent"] for item in corners)
    tangent_max = max(item["tangent"] for item in corners)
    tangent_abs_max = max(abs(item["tangent"]) for item in corners)
    near_margin = normal_min - depth_min
    far_margin = depth_max - normal_max
    side_margin = width_limit - tangent_abs_max
    normal_error = (normal_min + normal_max) * 0.5 - abs(float(box_depth)) * 0.5
    tangent_error = (tangent_min + tangent_max) * 0.5
    return {
        "inside": near_margin >= 0.0 and far_margin >= 0.0 and side_margin >= 0.0,
        "error": "",
        "normal_min": normal_min,
        "normal_max": normal_max,
        "tangent_min": tangent_min,
        "tangent_max": tangent_max,
        "tangent_abs_max": tangent_abs_max,
        "normal_error": normal_error,
        "tangent_error": tangent_error,
        "near_margin": near_margin,
        "far_margin": far_margin,
        "side_margin": side_margin,
        "corners": corners,
    }


def parking_footprint_inside(pose, wall_point, inward_normal,
                             box_width, box_depth,
                             footprint_half_length, footprint_half_width,
                             margin=0.0):
    """Check a rectangular base footprint against a wall-aligned parking box."""
    return bool(parking_footprint_margins(
        pose, wall_point, inward_normal, box_width, box_depth,
        footprint_half_length, footprint_half_width, margin).get("inside"))


def footprint_max_cost(data, width, height, resolution, origin_x, origin_y,
                       x, y, radius, lethal_cost=253):
    """Return (known, max_cost, blocked) for a circular footprint in one grid frame."""
    width = int(width)
    height = int(height)
    resolution = float(resolution)
    if width <= 0 or height <= 0 or resolution <= 0.0:
        return False, -1, False

    mx = int(math.floor((float(x) - float(origin_x)) / resolution))
    my = int(math.floor((float(y) - float(origin_y)) / resolution))
    cells = int(math.ceil(max(0.0, float(radius)) / resolution))
    if mx - cells < 0 or my - cells < 0 or mx + cells >= width or my + cells >= height:
        return False, -1, False

    max_cost = 0
    radius_sq = max(0.0, float(radius)) ** 2
    for gy in range(my - cells, my + cells + 1):
        for gx in range(mx - cells, mx + cells + 1):
            wx = float(origin_x) + (gx + 0.5) * resolution
            wy = float(origin_y) + (gy + 0.5) * resolution
            if (wx - float(x)) ** 2 + (wy - float(y)) ** 2 > radius_sq:
                continue
            raw = int(data[gy * width + gx]) & 0xFF
            if raw == 255:
                continue
            max_cost = max(max_cost, raw)
            if raw >= int(lethal_cost):
                return True, max_cost, True
    return True, max_cost, False
