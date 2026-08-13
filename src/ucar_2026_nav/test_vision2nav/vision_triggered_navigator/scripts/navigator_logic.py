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


def rotation_clearance_is_safe(nearest_range, scan_age, min_clearance,
                               max_scan_age=0.5):
    """Require a fresh all-around lidar clearance before an in-place turn."""
    if nearest_range is None:
        return False
    return (0.0 <= float(scan_age) <= abs(float(max_scan_age)) and
            float(nearest_range) >= abs(float(min_clearance)))


def rotation_clearance_consensus(samples, now, min_clearance,
                                 tolerance=0.005, max_sample_age=0.35,
                                 min_samples=3):
    """Return ``(safe, median, count)`` for recent all-around scan minima."""
    values = []
    now = float(now)
    max_sample_age = abs(float(max_sample_age))
    for stamp, distance in samples or ():
        if distance is None:
            continue
        age = now - float(stamp)
        distance = float(distance)
        if (0.0 <= age <= max_sample_age and math.isfinite(distance) and
                distance >= 0.0):
            values.append(distance)
    required = max(1, int(min_samples))
    if len(values) < required:
        return False, None, len(values)
    values.sort()
    middle = len(values) // 2
    if len(values) % 2:
        median = values[middle]
    else:
        median = 0.5 * (values[middle - 1] + values[middle])
    safe = median + abs(float(tolerance)) >= abs(float(min_clearance))
    return safe, median, len(values)


def obstacle_clearance_requires_stop(nearest_range, min_clearance,
                                     tolerance=0.0):
    """Return whether a directional obstacle reading must stop motion."""
    if nearest_range is None:
        return True
    return (float(nearest_range) + abs(float(tolerance)) <
            abs(float(min_clearance)))


def swept_footprint_obstacle(points, command, errors,
                             footprint_half_length,
                             footprint_half_width, margin=0.02):
    """Return diagnostics for scan points inside the commanded swept body.

    Points and commands are in ``base_link``.  Parking is phase ordered, so a
    translation sweep is an axis-aligned rectangle extended through the full
    remaining axis error.  A turn uses the rectangular footprint's
    circumscribed radius and is therefore conservative for every intermediate
    yaw.
    """
    command_x, command_y, command_yaw = [float(value) for value in command]
    normal_error, tangent_error, _yaw_error = [float(value) for value in errors]
    raw_half_length = abs(float(footprint_half_length))
    raw_half_width = abs(float(footprint_half_width))
    safety_margin = max(0.0, float(margin))
    half_length = raw_half_length + safety_margin
    half_width = raw_half_width + safety_margin
    phase = None
    bounds = None
    radius = None
    if abs(command_yaw) > 1e-9:
        phase = "rotation"
        radius = math.hypot(raw_half_length, raw_half_width) + safety_margin
    elif abs(command_y) > 1e-9:
        phase = "left" if command_y > 0.0 else "right"
        displacement = tangent_error
        bounds = (
            -half_length, half_length,
            min(0.0, displacement) - half_width,
            max(0.0, displacement) + half_width,
        )
    elif abs(command_x) > 1e-9:
        phase = "forward" if command_x > 0.0 else "rear"
        displacement = normal_error
        bounds = (
            min(0.0, displacement) - half_length,
            max(0.0, displacement) + half_length,
            -half_width, half_width,
        )
    if phase is None:
        return {"blocked": False, "phase": None, "point": None,
                "clearance": float("inf"), "bounds": None}

    nearest_point = None
    nearest_clearance = float("inf")
    for point_x, point_y in points or ():
        point_x, point_y = float(point_x), float(point_y)
        if not math.isfinite(point_x) or not math.isfinite(point_y):
            continue
        if radius is not None:
            clearance = math.hypot(point_x, point_y) - radius
            inside = clearance <= 0.0
        else:
            min_x, max_x, min_y, max_y = bounds
            dx = max(min_x - point_x, 0.0, point_x - max_x)
            dy = max(min_y - point_y, 0.0, point_y - max_y)
            clearance = math.hypot(dx, dy)
            inside = (min_x <= point_x <= max_x and
                      min_y <= point_y <= max_y)
        if clearance < nearest_clearance:
            nearest_clearance = clearance
            nearest_point = (point_x, point_y)
        if inside:
            return {
                "blocked": True,
                "phase": phase,
                "point": (point_x, point_y),
                "clearance": clearance,
                "bounds": bounds,
                "radius": radius,
            }
    return {
        "blocked": False,
        "phase": phase,
        "point": nearest_point,
        "clearance": nearest_clearance,
        "bounds": bounds,
        "radius": radius,
    }


def coverage_obstacle_confirmation(blocked, previous_count,
                                   fresh_sample=True, required_scans=2):
    """Count consecutive fresh blocking scans and report confirmation."""
    if not bool(blocked):
        return 0, False
    count = max(0, int(previous_count))
    if bool(fresh_sample):
        count += 1
    return count, count >= max(2, int(required_scans))


def rotation_swept_obstacle(points, footprint_radius=0.215, margin=0.02):
    """Return a circular in-place-rotation envelope in ``base_link``."""
    return swept_footprint_obstacle(
        points,
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        abs(float(footprint_radius)),
        0.0,
        margin,
    )


def staging_wall_frame_accepted(normal_error, tangent_error, yaw_error,
                                normal_tolerance=0.08,
                                tangent_tolerance=0.04,
                                yaw_tolerance=0.08):
    """Require axis-specific staging accuracy before close-range docking."""
    return (
        abs(float(normal_error)) <= abs(float(normal_tolerance)) and
        abs(float(tangent_error)) <= abs(float(tangent_tolerance)) and
        abs(float(yaw_error)) <= abs(float(yaw_tolerance))
    )


def parking_obstacle_action(consecutive_blocks, recovery_attempts,
                            required_blocks=2, retry_count=1):
    """Choose hold, one bounded recovery, or a terminal safe stop."""
    if int(consecutive_blocks) < max(1, int(required_blocks)):
        return "hold"
    if int(recovery_attempts) < max(0, int(retry_count)):
        return "recover"
    return "fail"


def recovery_rear_distance(wall_distance_error, wall_normal_angle,
                           minimum_projection=0.30):
    """Convert perpendicular wall error to required base-frame rear travel."""
    projection = math.cos(float(wall_normal_angle))
    if projection <= abs(float(minimum_projection)):
        return None
    return float(wall_distance_error) / projection


def coverage_speed_profile(clearance, current_profile,
                           caution_enter_clearance,
                           caution_exit_clearance,
                           fast_exit_clearance,
                           fast_enter_clearance):
    """Select a hysteretic coverage speed profile from lidar clearance."""
    caution_enter = float(caution_enter_clearance)
    caution_exit = float(caution_exit_clearance)
    fast_exit = float(fast_exit_clearance)
    fast_enter = float(fast_enter_clearance)
    if not (0.0 <= caution_enter <= caution_exit <
            fast_exit <= fast_enter):
        raise ValueError("coverage speed clearance thresholds are invalid")

    profile = str(current_profile or "cruise").strip().lower()
    if profile not in ("caution", "cruise", "fast"):
        profile = "cruise"
    if clearance is None:
        return "cruise"
    value = float(clearance)
    if not math.isfinite(value) or value < 0.0:
        return "cruise"

    if profile == "fast" and value >= fast_exit:
        return "fast"
    if profile == "caution" and value <= caution_exit:
        return "caution"
    if value <= caution_enter:
        return "caution"
    if value >= fast_enter:
        return "fast"
    return "cruise"


def coverage_non_target_observation_matches(
        active_anchor, observed_anchor, category, enabled=True):
    """Return whether a confirmed non-target observation belongs here."""
    if not enabled or not str(category or "").strip():
        return False
    try:
        active = int(active_anchor)
        observed = int(observed_anchor)
    except (TypeError, ValueError):
        return False
    return active > 0 and active == observed


def coverage_non_target_early_exit_ready(
        completed_scan_steps, minimum_scan_steps):
    """Return whether deliberate scan coverage permits a non-target exit."""
    try:
        completed = int(completed_scan_steps)
        minimum = int(minimum_scan_steps)
    except (TypeError, ValueError):
        return False
    return completed >= max(0, minimum)


def parking_rotation_obstacle_clearance(
        samples, wall_fit, lidar_forward_offset=0.0,
        front_half_angle=math.radians(35.0),
        wall_residual_tolerance=0.02,
        minimum_base_clearance=0.0):
    """Return nearest rotation obstacle after removing a trusted front wall.

    Parking already validates ``wall_fit`` for span, orientation, residual,
    and temporal continuity.  At close range that same wall extends beyond
    the fixed front angular sector, so an angle-only exclusion can mistake it
    for a side obstacle.  Returns infinity when every non-front return belongs
    to the fitted wall, and ``None`` when no usable non-front evidence exists.
    """
    if not wall_fit:
        return None
    try:
        nx, ny = [float(value) for value in wall_fit["normal"]]
        wall_distance = float(wall_fit["distance"])
    except (KeyError, TypeError, ValueError):
        return None
    normal_length = math.hypot(nx, ny)
    if (normal_length <= 1e-9 or not math.isfinite(wall_distance)):
        return None
    nx /= normal_length
    ny /= normal_length
    wall_distance /= normal_length

    offset = float(lidar_forward_offset)
    half_angle = abs(float(front_half_angle))
    residual_limit = abs(float(wall_residual_tolerance))
    base_clearance = max(0.0, float(minimum_base_clearance))
    nearest = None
    wall_return_count = 0
    for angle, distance in samples or ():
        angle = normalize_angle(float(angle))
        distance = float(distance)
        if (not math.isfinite(angle) or not math.isfinite(distance) or
                distance <= 0.0 or abs(angle) <= half_angle):
            continue
        point_x = offset + distance * math.cos(angle)
        point_y = distance * math.sin(angle)
        wall_residual = abs(
            point_x * nx + point_y * ny - wall_distance)
        if (wall_residual <= residual_limit and
                math.hypot(point_x, point_y) >= base_clearance):
            wall_return_count += 1
            continue
        nearest = distance if nearest is None else min(nearest, distance)
    if nearest is not None:
        return nearest
    if wall_return_count:
        return float("inf")
    return None


def polar_sector_min(samples, center_angle, half_angle):
    """Return the nearest valid polar sample in a wrapped angular sector."""
    nearest = None
    center_angle = float(center_angle)
    half_angle = abs(float(half_angle))
    for angle, distance in samples or ():
        angle = float(angle)
        distance = float(distance)
        if not math.isfinite(angle) or not math.isfinite(distance):
            continue
        if abs(normalize_angle(angle - center_angle)) > half_angle:
            continue
        nearest = distance if nearest is None else min(nearest, distance)
    return nearest


def cyclic_coverage_order(points, robot_x, robot_y):
    """Start at the nearest anchor while preserving the calibrated cycle."""
    if not points:
        return []
    nearest = min(
        range(len(points)),
        key=lambda index: (
            (float(points[index]["x"]) - float(robot_x)) ** 2 +
            (float(points[index]["y"]) - float(robot_y)) ** 2,
            index,
        ),
    )
    return list(range(nearest, len(points))) + list(range(0, nearest))


def coverage_anchor_order(count, preferred_anchor=0, skipped_anchors=(),
                          nearest_order=None):
    """Build a one-pass zero-based coverage order.

    Public anchor parameters are one-based.  An explicit remembered anchor
    wins; otherwise the caller may provide an order beginning at the nearest
    current anchor.  Only explicitly supplied (manually calibrated) anchors
    are omitted.
    """
    count = max(0, int(count))
    if count == 0:
        return []
    preferred = int(preferred_anchor or 0)
    if 1 <= preferred <= count:
        start = preferred - 1
        order = list(range(start, count)) + list(range(0, start))
    elif nearest_order is not None:
        order = [int(index) for index in nearest_order
                 if 0 <= int(index) < count]
    else:
        order = list(range(count))
    skipped = set()
    for anchor in skipped_anchors or ():
        try:
            anchor = int(anchor)
        except (TypeError, ValueError):
            continue
        if 1 <= anchor <= count:
            skipped.add(anchor - 1)
    return [index for index in order if index not in skipped]


def should_retry_coverage_goal(result, rotation_stall, timed_out,
                               attempt, retry_count, aborted_status=4):
    """Retry an exact anchor after a recoverable move_base failure."""
    if int(attempt) >= max(0, int(retry_count)):
        return False
    try:
        aborted = int(result) == int(aborted_status)
    except (TypeError, ValueError):
        aborted = False
    return bool(rotation_stall) or bool(timed_out) or aborted


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


def nearest_wall_hit(walls, origin, ray_yaw):
    """Return the nearest measured wall hit and endpoint clearance."""
    ox, oy = [float(value) for value in origin]
    direction = (math.cos(float(ray_yaw)), math.sin(float(ray_yaw)))
    best = None
    for wall_name, start, end, normal in walls:
        distance = ray_segment_intersection(
            (ox, oy), direction, start, end)
        if distance is None:
            continue
        if best is not None and distance >= best["distance"]:
            continue
        point = (
            ox + distance * direction[0],
            oy + distance * direction[1],
        )
        best = {
            "wall": wall_name,
            "point": point,
            "normal": normal,
            "distance": distance,
            "endpoint_clearance": min(
                math.hypot(point[0] - endpoint[0],
                           point[1] - endpoint[1])
                for endpoint in (start, end)),
        }
    return best


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


def docking_pose_errors(current_pose, target_pose):
    """Return target errors in the current robot body frame plus yaw error."""
    x, y, yaw = [float(value) for value in current_pose]
    target_x, target_y, target_yaw = [float(value) for value in target_pose]
    dx = target_x - x
    dy = target_y - y
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    forward = cos_yaw * dx + sin_yaw * dy
    lateral = -sin_yaw * dx + cos_yaw * dy
    return forward, lateral, normalize_angle(target_yaw - yaw)


def wall_frame_pose_errors(current_pose, target_pose):
    """Return errors on the target wall-normal and wall-tangent axes."""
    x, y, yaw = [float(value) for value in current_pose]
    target_x, target_y, target_yaw = [float(value) for value in target_pose]
    dx = target_x - x
    dy = target_y - y
    cos_target = math.cos(target_yaw)
    sin_target = math.sin(target_yaw)
    normal = cos_target * dx + sin_target * dy
    tangent = -sin_target * dx + cos_target * dy
    return normal, tangent, normalize_angle(target_yaw - yaw)


def bounded_axis_command(error, tolerance, gain, maximum, minimum=0.0):
    """Return a signed proportional command with deadband and bounds."""
    error = float(error)
    if abs(error) <= abs(float(tolerance)):
        return 0.0
    magnitude = min(abs(float(maximum)), abs(float(gain)) * abs(error))
    magnitude = max(min(abs(float(minimum)), abs(float(maximum))), magnitude)
    return math.copysign(magnitude, error)


def docking_command(errors, normal_tolerance, tangent_tolerance, yaw_tolerance,
                    max_x, max_y, max_yaw,
                    gain_x=0.8, gain_y=1.0, gain_yaw=1.5,
                    min_x=0.03, min_y=0.025, min_yaw=0.05):
    """Compute a conservative two-phase holonomic docking command.

    Tangent and yaw alignment happen before forward motion.  Once aligned,
    small lateral/yaw corrections remain active during the straight approach.
    """
    forward, lateral, yaw_error = [float(value) for value in errors]
    aligned = (abs(lateral) <= abs(float(tangent_tolerance)) and
               abs(yaw_error) <= abs(float(yaw_tolerance)))
    command_x = 0.0
    if aligned:
        command_x = bounded_axis_command(
            forward, normal_tolerance, gain_x, max_x, min_x)
    command_y = bounded_axis_command(
        lateral, tangent_tolerance, gain_y, max_y, min_y)
    command_yaw = bounded_axis_command(
        yaw_error, yaw_tolerance, gain_yaw, max_yaw, min_yaw)
    return command_x, command_y, command_yaw


def wall_frame_docking_command(normal_error, tangent_error, yaw_error,
                               normal_tolerance, tangent_tolerance,
                               yaw_tolerance, max_x, max_y, max_yaw,
                               min_yaw=0.15):
    """Three-phase command for a mecanum base in a measured wall frame.

    Rotation has priority, followed by tangent translation.  Forward motion
    is permitted only after both are in tolerance, preventing diagonal entry.
    """
    normal_error = float(normal_error)
    tangent_error = float(tangent_error)
    yaw_error = float(yaw_error)
    if abs(yaw_error) > abs(float(yaw_tolerance)):
        return (0.0, 0.0, bounded_axis_command(
            yaw_error, yaw_tolerance, 1.5, max_yaw, min_yaw))
    if abs(tangent_error) > abs(float(tangent_tolerance)):
        return (0.0, bounded_axis_command(
            tangent_error, tangent_tolerance, 1.0, max_y, 0.025), 0.0)
    return (bounded_axis_command(
        normal_error, normal_tolerance, 0.8, max_x, 0.03), 0.0, 0.0)


def staging_pose_reached(current_pose, goal_pose,
                         position_tolerance=0.10, yaw_tolerance=0.10):
    """Require both translation and heading before move_base handoff."""
    distance = math.hypot(
        float(goal_pose[0]) - float(current_pose[0]),
        float(goal_pose[1]) - float(current_pose[1]))
    yaw_error = abs(normalize_angle(
        float(goal_pose[2]) - float(current_pose[2])))
    return (distance <= abs(float(position_tolerance)) and
            yaw_error <= abs(float(yaw_tolerance)))


def staging_handoff_accepted(current_pose, goal_pose, action_succeeded,
                             position_tolerance=0.10,
                             yaw_tolerance=0.10,
                             success_position_tolerance=0.15,
                             success_yaw_tolerance=0.12):
    """Allow a bounded handoff after move_base reports success.

    The normal tolerances remain active while move_base is driving.  A small
    extra envelope is allowed only after SUCCEEDED because the following
    docking controller performs the precise wall-relative approach.
    """
    if staging_pose_reached(
            current_pose, goal_pose, position_tolerance, yaw_tolerance):
        return True
    if not bool(action_succeeded):
        return False
    return staging_pose_reached(
        current_pose, goal_pose,
        max(abs(float(position_tolerance)),
            abs(float(success_position_tolerance))),
        max(abs(float(yaw_tolerance)), abs(float(success_yaw_tolerance))),
    )


def fit_wall_line(points, min_points=12, min_span=0.25,
                  max_residual=0.015):
    """Robustly fit a front wall in base coordinates without numpy.

    Pair hypotheses select the largest line-like support; PCA then refines the
    winning inliers.  The returned normal points from the base toward the wall.
    """
    pts = [(float(x), float(y)) for x, y in points
           if math.isfinite(float(x)) and math.isfinite(float(y))]
    min_points = max(2, int(min_points))
    if len(pts) < min_points:
        return None
    # Deterministic downsampling bounds pair hypotheses on the ARM computer.
    if len(pts) > 80:
        step = float(len(pts) - 1) / 79.0
        pts = [pts[int(round(i * step))] for i in range(80)]
    hypothesis_step = max(1, int(math.ceil(len(pts) / 24.0)))
    hypothesis_indices = list(range(0, len(pts), hypothesis_step))
    if hypothesis_indices[-1] != len(pts) - 1:
        hypothesis_indices.append(len(pts) - 1)
    threshold = max(1e-4, float(max_residual) * 1.5)
    best = []
    for index_i in range(len(hypothesis_indices) - 1):
        i = hypothesis_indices[index_i]
        ax, ay = pts[i]
        for j in hypothesis_indices[index_i + 1:]:
            bx, by = pts[j]
            dx, dy = bx - ax, by - ay
            length = math.hypot(dx, dy)
            if length < float(min_span) * 0.5:
                continue
            nx, ny = -dy / length, dx / length
            support = [p for p in pts
                       if abs((p[0] - ax) * nx + (p[1] - ay) * ny) <= threshold]
            support_projection = [
                (p[0] - ax) * dx / length + (p[1] - ay) * dy / length
                for p in support]
            if (len(support) < min_points or
                    max(support_projection) - min(support_projection) <
                    abs(float(min_span))):
                continue
            if len(support) > len(best):
                best = support
    if len(best) < min_points:
        return None

    cx = sum(p[0] for p in best) / len(best)
    cy = sum(p[1] for p in best) / len(best)
    sxx = sum((p[0] - cx) ** 2 for p in best)
    syy = sum((p[1] - cy) ** 2 for p in best)
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in best)
    tangent_angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    tx, ty = math.cos(tangent_angle), math.sin(tangent_angle)
    nx, ny = -ty, tx
    if nx * cx + ny * cy < 0.0:
        nx, ny = -nx, -ny
    residuals = [abs((p[0] - cx) * nx + (p[1] - cy) * ny)
                 for p in best]
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    projections = [(p[0] - cx) * tx + (p[1] - cy) * ty for p in best]
    span = max(projections) - min(projections)
    distance = cx * nx + cy * ny
    if (span < abs(float(min_span)) or
            rms > abs(float(max_residual)) or distance <= 0.0):
        return None
    return {
        "distance": distance,
        "normal_angle": math.atan2(ny, nx),
        "normal": (nx, ny),
        "span": span,
        "residual": rms,
        "inliers": len(best),
    }


def rotation_clearance_allows_near_wall(
        samples, scan_age, min_clearance, max_scan_age=0.5,
        lidar_forward_offset=0.0, footprint_radius=0.215,
        footprint_margin=0.02, wall_range_band=0.16,
        wall_min_points=10, wall_min_span=0.25,
        wall_max_residual=0.02):
    """Allow a conservative close-range turn only beside a continuous wall.

    The ordinary all-around threshold remains the first line of defence.  This
    exception is for calibrated observation poses where the lidar can be less
    than that threshold from a wall although the wall remains outside the
    chassis rotation envelope.  A compact return such as a cone cannot satisfy
    the required line span and nearest-point agreement.
    """
    if not (0.0 <= float(scan_age) <= abs(float(max_scan_age))):
        return False
    valid = [
        (normalize_angle(float(angle)), float(distance))
        for angle, distance in samples or ()
        if (math.isfinite(float(angle)) and
            math.isfinite(float(distance)) and float(distance) > 0.0)
    ]
    if not valid:
        return False

    nearest_angle, nearest_range = min(valid, key=lambda item: item[1])
    if nearest_range >= abs(float(min_clearance)):
        return True

    offset = float(lidar_forward_offset)
    base_points = [
        (offset + distance * math.cos(angle),
         distance * math.sin(angle))
        for angle, distance in valid
    ]
    nearest_base_range = min(math.hypot(x, y) for x, y in base_points)
    required_base_range = (
        abs(float(footprint_radius)) + abs(float(footprint_margin)))
    if nearest_base_range < required_base_range:
        return False

    maximum_wall_range = (
        abs(float(min_clearance)) + abs(float(wall_range_band)))
    wall_half_angle = math.radians(60.0)
    wall_points = [
        (offset + distance * math.cos(angle),
         distance * math.sin(angle))
        for angle, distance in valid
        if (distance <= maximum_wall_range and
            abs(normalize_angle(angle - nearest_angle)) <= wall_half_angle)
    ]
    fit = fit_wall_line(
        wall_points,
        min_points=wall_min_points,
        min_span=wall_min_span,
        max_residual=wall_max_residual,
    )
    if fit is None:
        return False

    nearest_point = (
        offset + nearest_range * math.cos(nearest_angle),
        nearest_range * math.sin(nearest_angle),
    )
    nx, ny = fit["normal"]
    nearest_residual = abs(
        nearest_point[0] * nx + nearest_point[1] * ny - fit["distance"])
    return nearest_residual <= max(
        0.01, abs(float(wall_max_residual)) * 1.5)


def wall_fit_matches_expected(fit, expected_normal_angle,
                              maximum_error=math.radians(20.0)):
    if not fit:
        return False
    return abs(normalize_angle(
        float(fit["normal_angle"]) - float(expected_normal_angle))) <= abs(
            float(maximum_error))


def wall_fit_is_continuous(current, previous, maximum_distance_jump=0.05,
                           maximum_normal_jump=math.radians(8.0)):
    """Accept a near-field fit only when it continues an acquired wall."""
    if not current or not previous:
        return False
    return (abs(float(current["distance"]) - float(previous["distance"])) <=
            abs(float(maximum_distance_jump)) and
            abs(normalize_angle(float(current["normal_angle"]) -
                                float(previous["normal_angle"]))) <=
            abs(float(maximum_normal_jump)))


def docking_within_tolerance(errors, normal_tolerance,
                             tangent_tolerance, yaw_tolerance):
    forward, lateral, yaw_error = [abs(float(value)) for value in errors]
    return (forward <= abs(float(normal_tolerance)) and
            lateral <= abs(float(tangent_tolerance)) and
            yaw_error <= abs(float(yaw_tolerance)))


def staging_motion_is_rotation_stall(distance_moved, yaw_accumulated,
                                     minimum_distance=0.03,
                                     maximum_yaw=math.radians(45.0)):
    """Detect move_base rotating without useful translation in one window."""
    return (float(distance_moved) < abs(float(minimum_distance)) and
            abs(float(yaw_accumulated)) > abs(float(maximum_yaw)))


def coverage_motion_is_rotation_stall(distance_moved, yaw_accumulated,
                                      minimum_distance=0.03,
                                      maximum_yaw=math.radians(90.0)):
    """Detect a coverage goal about to enter move_base rotation recovery."""
    return staging_motion_is_rotation_stall(
        distance_moved, yaw_accumulated, minimum_distance, maximum_yaw)


def coverage_position_needs_yaw_alignment(distance, yaw_error,
                                          position_tolerance=0.15,
                                          yaw_tolerance=0.06):
    """Hand a reached position's remaining heading correction to odometry."""
    return (float(distance) <= abs(float(position_tolerance)) and
            abs(float(yaw_error)) > abs(float(yaw_tolerance)))


def coverage_near_anchor_action(distance, baseline_distance, elapsed,
                                observation_radius=0.45,
                                stall_timeout=0.8,
                                minimum_progress=0.03):
    """Decide whether a blocked near-anchor goal should become an observation.

    Exact observation coordinates can be occupied by a cone or inflated
    costmap cell.  Once the robot is close enough to see the sign, measurable
    progress restarts the short watchdog; otherwise a stalled goal is handed
    over to the stationary scan instead of waiting for the global timeout.
    """
    distance = float(distance)
    if distance > abs(float(observation_radius)):
        return "outside"
    if baseline_distance is None:
        return "start"
    if float(baseline_distance) - distance >= abs(float(minimum_progress)):
        return "reset"
    if max(0.0, float(elapsed)) >= max(0.0, float(stall_timeout)):
        return "observe"
    return "continue"


def coverage_timeout_decision(elapsed, window_progress,
                              soft_timeout=25.0, hard_timeout=40.0,
                              minimum_progress=0.03):
    """Decide whether an exact coverage goal should continue or stop.

    The soft deadline may be crossed only while the base is still making
    measurable progress.  Once extended, the hard deadline remains absolute.
    """
    elapsed = max(0.0, float(elapsed))
    soft_timeout = max(0.0, float(soft_timeout))
    hard_timeout = max(soft_timeout, float(hard_timeout))
    if elapsed >= hard_timeout:
        return "hard_timeout"
    if elapsed < soft_timeout:
        return "continue"
    if float(window_progress) >= abs(float(minimum_progress)):
        return "extend"
    return "soft_timeout"


def target_sample_is_fresh(target_error, received_at, now, timeout):
    """Return whether an OCR target box can safely start recentering."""
    return (target_error is not None and
            sensor_is_fresh(received_at, now, timeout))


def parking_recenter_required(initial_center_error, tolerance):
    """Return whether staging should run another visual centering pass."""
    if initial_center_error is None:
        return True
    return abs(float(initial_center_error)) > abs(float(tolerance))


def wall_normal_distance(pose, wall_point, inward_normal):
    """Return base-centre distance from a wall along its inward normal."""
    x, y = float(pose[0]), float(pose[1])
    wall_x, wall_y = float(wall_point[0]), float(wall_point[1])
    nx, ny = float(inward_normal[0]), float(inward_normal[1])
    length = math.hypot(nx, ny)
    if length <= 1e-9:
        raise ValueError("inward normal must be non-zero")
    return ((x - wall_x) * nx + (y - wall_y) * ny) / length


def sensor_is_fresh(received_at, now, timeout):
    """Return whether a sensor sample is present and within its age budget."""
    received_at = float(received_at or 0.0)
    return (received_at > 0.0 and
            float(now) - received_at <= max(0.0, float(timeout)))


def lidar_base_wall_distance(raw_distance, forward_offset):
    """Convert a forward laser range to an equivalent base-centre wall range.

    The UCAR laser origin is mounted ahead of ``base_link``.  Applying this
    extrinsic prevents a safe 0.22 m base target from being rejected as a
    0.14 m raw laser reading.
    """
    return float(raw_distance) + float(forward_offset)


def lidar_requires_stop(raw_distance, base_equivalent_distance,
                        geometric_wall_distance, stop_distance,
                        mismatch_tolerance=0.03):
    """Reject a hard-close return or a return inconsistent with the wall.

    A raw wall return can legitimately be below ``stop_distance`` because the
    laser sits ahead of base_link.  It is safe only when its base-equivalent
    range agrees with the independently computed wall geometry.
    """
    raw_distance = float(raw_distance)
    base_equivalent_distance = float(base_equivalent_distance)
    geometric_wall_distance = float(geometric_wall_distance)
    stop_distance = abs(float(stop_distance))
    mismatch_tolerance = abs(float(mismatch_tolerance))
    if base_equivalent_distance < stop_distance:
        return True
    return (raw_distance < stop_distance and
            base_equivalent_distance <
            geometric_wall_distance - mismatch_tolerance)


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


def scan_step_timeout_extension(progress, target, elapsed, progress_age,
                                commanded_speed, max_extra_sec,
                                progress_fresh_sec=0.8,
                                min_progress=math.radians(0.5),
                                reserve_sec=0.35):
    """Return bounded extra time while a scan step is still making progress."""
    progress = max(0.0, float(progress))
    target = max(0.0, float(target))
    remaining = max(0.0, target - progress)
    max_extra_sec = max(0.0, float(max_extra_sec))
    if remaining <= 1e-6 or max_extra_sec <= 0.0:
        return 0.0
    if progress < max(0.0, float(min_progress)):
        return 0.0
    if float(progress_age) > max(0.0, float(progress_fresh_sec)):
        return 0.0

    commanded_speed = abs(float(commanded_speed))
    if commanded_speed <= 1e-6:
        return 0.0
    measured_speed = progress / max(0.05, float(elapsed))
    conservative_speed = max(
        0.03,
        commanded_speed * 0.15,
        min(commanded_speed, measured_speed * 0.75),
    )
    estimate = remaining / conservative_speed + max(0.0, float(reserve_sec))
    return min(max_extra_sec, estimate)


def scan_dwell_deadline(started_at, dwell_sec, candidate_at,
                        candidate_hold_sec, max_dwell_sec):
    """Return a bounded dwell deadline, extending it for a fresh OCR candidate."""
    started_at = float(started_at)
    deadline = started_at + max(0.0, float(dwell_sec))
    candidate_at = float(candidate_at or 0.0)
    if candidate_at >= started_at:
        deadline = max(deadline, candidate_at + max(0.0, float(candidate_hold_sec)))
    return min(deadline, started_at + max(0.0, float(max_dwell_sec)))


def exact_observation_target(point):
    """Return the calibrated observation pose without generating offsets."""
    return float(point["x"]), float(point["y"]), float(point["yaw"])


def should_skip_coverage_anchor(cost_known, max_cost, lethal_cost=253):
    """Only a known lethal/inscribed footprint cost may skip an anchor."""
    return bool(cost_known) and int(max_cost) >= int(lethal_cost)


def costmap_value_at(data, width, height, resolution,
                     origin_x, origin_y, x, y):
    """Read one already-transformed costmap point; return -1 when unknown."""
    resolution = float(resolution)
    if resolution <= 0.0:
        return -1
    mx = int(math.floor((float(x) - float(origin_x)) / resolution))
    my = int(math.floor((float(y) - float(origin_y)) / resolution))
    width = int(width)
    height = int(height)
    if mx < 0 or mx >= width or my < 0 or my >= height:
        return -1
    raw = int(data[my * width + mx]) & 0xFF
    return -1 if raw == 255 else raw


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
    saw_known = False
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
            saw_known = True
            max_cost = max(max_cost, raw)
            if raw >= int(lethal_cost):
                return True, max_cost, True
    if not saw_known:
        return False, -1, False
    return True, max_cost, False
