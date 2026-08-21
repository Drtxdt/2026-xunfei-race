# -*- coding: utf-8 -*-
"""ROS-independent logic for the national competition flow.

The national run keeps the provincial sub-tasks but inserts a ramp
(up-flat-down, ~1.5 m) into the task-1 navigation phase:

    voice -> navigate to ramp staging -> traverse ramp (ucar_2026_upanddown)
    -> recover localization -> navigate to QR area -> spin-scan QR codes
    -> Spark LLM reasoning -> task-1 announcement
    -> hand over to the untouched provincial flow for task2..task5

The hand-over launches the provincial ``flow_node.launch`` sequentially with
the stage combinations it already supports, passing the task-1 result as
launch arguments.  The only addition on the provincial side is an optional
``track_package`` launch argument (defaulting to the provincial track
package); the national run sets it to the national track package so task5
line following uses the gray-barrier obstacle-avoidance variant.
"""

from __future__ import annotations

import math

from ucar_2026_competition.logic import normalize_category


NATIONAL_STAGES = (
    "voice_handshake",
    "navigate_ramp_staging",
    "traverse_ramp",
    "post_ramp_recovery",
    "relocalize_after_ramp",
    "navigate_qr_area",
    "scan_qr",
    "reason_and_announce",
    "handover",
)

# Provincial flow_node.launch start_stage values, in execution order.
HANDOVER_CHAIN_SIM = ("task2", "task3", "task4_task5")
HANDOVER_CHAIN_NO_SIM = ("task2", "task4", "task5")


def handover_chain(enable_simulation):
    """Return the provincial start_stage sequence for the hand-over."""
    return HANDOVER_CHAIN_SIM if bool(enable_simulation) else HANDOVER_CHAIN_NO_SIM


def stage_sequence(mode="full", enable_simulation=True, ramp_enabled=True):
    """Return the national stage tuple for the requested run mode."""
    normalized = str(mode or "").strip().lower()
    if normalized == "full":
        stages = list(NATIONAL_STAGES)
        if not ramp_enabled:
            stages.remove("traverse_ramp")
            stages.remove("post_ramp_recovery")
            stages.remove("relocalize_after_ramp")
        return tuple(stages)
    if normalized == "task1":
        return tuple(
            stage for stage in NATIONAL_STAGES if stage != "handover")
    if normalized == "ramp":
        return ("navigate_ramp_staging", "traverse_ramp",
                "post_ramp_recovery", "relocalize_after_ramp")
    raise ValueError("unsupported national start_stage: {}".format(mode))


def validate_pose(x, y, yaw, name):
    """Reject NaN / infinite waypoint coordinates before move_base sees them."""
    values = (float(x), float(y), float(yaw))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("{} pose must be finite: {}".format(name, values))
    return values


def flow_launch_args(
    start_stage,
    task1_result,
    enable_simulation,
    traffic_pose=None,
    skip_task4_stop_line_approach=False,
    track_package="ucar_2026_track_end_stop",
    sim_bridge_host="192.168.1.28",
):
    """Build the roslaunch arguments for one provincial hand-over launch.

    ``task1_result`` is a mapping with pickup/sim category, item and workshop
    fields (the ReasonPickupOrder response).  ``track_package`` selects the
    line-following package the provincial task5 stage will roslaunch.
    """
    start_stage = str(start_stage or "").strip()
    if not start_stage:
        raise ValueError("hand-over start_stage is required")

    pickup_category = str(task1_result.get("pickup_major") or "").strip()
    sim_category = str(task1_result.get("sim_major") or "").strip()
    if not pickup_category:
        raise ValueError("task-1 result is missing pickup_major")

    args = {
        "start_stage": start_stage,
        "target_category": pickup_category,
        "target_item": str(task1_result.get("pickup_item") or ""),
        "target_workshop": str(task1_result.get("pickup_workshop") or ""),
        "sim_target_category": sim_category,
        "sim_item": str(task1_result.get("sim_item") or ""),
        "sim_workshop": str(task1_result.get("sim_workshop") or ""),
        "enable_simulation": bool(enable_simulation),
        "skip_task4_stop_line_approach": bool(skip_task4_stop_line_approach),
        "track_package": str(track_package),
        "sim_bridge_host": str(sim_bridge_host),
    }
    if traffic_pose is not None:
        x, y, yaw, configured = traffic_pose
        args["traffic_pose_configured"] = bool(configured)
        args["traffic_x"] = float(x)
        args["traffic_y"] = float(y)
        args["traffic_yaw"] = float(yaw)
    return args


def build_roslaunch_command(package, launch_file, args):
    """Render ``roslaunch pkg file key:=value ...`` exactly like the flow does."""
    command = ["roslaunch", str(package), str(launch_file)]
    for name, value in dict(args or {}).items():
        if isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        command.append("{}:={}".format(name, rendered))
    return command


def status_state(payload):
    """Extract the state field from a /competition/status JSON payload."""
    try:
        return str(payload.get("state") or "").strip().lower()
    except AttributeError:
        return ""


def provincial_flow_paused(payload):
    """True when the provincial child is blocked waiting for /competition/resume."""
    return status_state(payload) == "paused"


def provincial_flow_terminal(payload):
    """True when the provincial child reported a terminal stage state."""
    return status_state(payload) in ("completed", "aborted")


def min_valid_range(ranges, range_min=0.0, range_max=float("inf")):
    """Nearest finite lidar sample, or None when everything is invalid."""
    nearest = None
    lower = max(0.0, float(range_min))
    upper = float(range_max)
    for value in ranges or ():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value < lower or value > upper:
            continue
        if nearest is None or value < nearest:
            nearest = value
    return nearest


def rotation_clearance_ok(nearest_range, clearance_m):
    """None (no data) passes; a real reading must clear the margin."""
    if nearest_range is None:
        return True
    return float(nearest_range) >= float(clearance_m)


def task1_categories_match(result, pickup_category, sim_category):
    """The LLM result must agree with the voice-parsed target categories.

    The Spark X2 response may return Chinese category names
    (e.g. '日用品大类', '食品大类') while the voice parser returns the
    canonical English keys ('daily', 'food').  Normalize both sides before
    comparing.
    """
    result_pickup = normalize_category(result.get("pickup_major"))
    result_sim = normalize_category(result.get("sim_major"))
    expected_pickup = normalize_category(pickup_category)
    expected_sim = normalize_category(sim_category)

    if result_pickup != expected_pickup:
        return False
    if not expected_sim:
        return True
    return result_sim == expected_sim


def items_equal_allowed(pickup_category, sim_category):
    """Duplicate items are only legal when both targets share a category."""
    return str(pickup_category or "") == str(sim_category or "")
