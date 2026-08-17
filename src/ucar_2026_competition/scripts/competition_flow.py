#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Competition state machine for the five smart-factory subtasks."""

from __future__ import annotations

import json
import math
import os
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections import OrderedDict

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from move_base_msgs.msg import (
    MoveBaseAction,
    MoveBaseActionGoal,
    MoveBaseActionResult,
    MoveBaseGoal,
)
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger, TriggerResponse
from ucar_2026_competition_speech.srv import Announce
from ucar_2026_smart_factory_llm.srv import ReasonPickupOrder
from ucar_2026_competition.logic import (
    CATEGORY_LABELS,
    base_is_stopped,
    build_task1_instruction,
    final_advance_completed,
    TemporalTargetFilter,
    DirectedYawAccumulator,
    JsonLineBuffer,
    TRACK_CONFIG,
    normalize_angle,
    normalize_category,
    normalize_coverage_anchor_ids,
    normalize_task4_staging_pose,
    non_target_observation_is_actionable,
    parse_task1_categories,
    qr_values_from_payload,
    scan_sector_min,
    split_rotation_steps,
    stage_sequence,
    task2_delivery_targets,
    task2_resumed_coverage_hint,
    task2_prewarm_reusable,
    task2_target_trigger_is_eligible,
    task4_handoff_required,
    task4_start_action,
    traffic_decision_from_payload,
    task2_announcement_required,
    target_bbox_is_close_enough,
    target_bbox_ratios,
    trigger_delivery_state,
)


class StageError(RuntimeError):
    pass


class Aborted(RuntimeError):
    pass


def bool_param(name, default=False):
    value = rospy.get_param(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class CompetitionFlow:
    def __init__(self):
        self.mode = rospy.get_param("~start_stage", "full").strip().lower()
        self.enable_simulation = bool_param("~enable_simulation", False)
        self.debug = bool_param("~debug", False)
        self.aborted = threading.Event()
        self.resume_event = threading.Event()
        self.children = {}
        self.lock = threading.RLock()
        self.voice_transition_lock = threading.Lock()
        self.transition_announcement = None
        self.next_stage = None

        self.status_pub = rospy.Publisher(
            rospy.get_param("~status_topic", "/competition/status"),
            String,
            queue_size=20,
            latch=True,
        )
        self.result_pub = rospy.Publisher(
            rospy.get_param("~task1_result_topic", "/competition/task1_result"),
            String,
            queue_size=5,
            latch=True,
        )
        self.traffic_pub = rospy.Publisher(
            "/competition/traffic_decision", String, queue_size=5, latch=True
        )
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=2)
        self.vision_target_pub = rospy.Publisher(
            "/vision/target", String, queue_size=10)
        self.vision_non_target_topic = rospy.get_param(
            "~task2_non_target_topic", "/vision/non_target_observation")
        self.vision_non_target_pub = rospy.Publisher(
            self.vision_non_target_topic, String, queue_size=10)

        self.wakeup_received = False
        self.voice_prompt_started = False
        self.voice_listening = False
        self.voice_command_acknowledged = False
        self.voice_command_ack_in_progress = False
        self.voice_handshake_error = ""
        self.voice_wakeup_generation = 0
        self.question = ""
        self.category = normalize_category(rospy.get_param("~target_category", ""))
        self.sim_category = normalize_category(
            rospy.get_param("~sim_target_category", "")
        )
        if self.mode in ("task2", "task3", "task3_task4") and not self.sim_category:
            self.sim_category = self.category
        self.task1_result = {
            "pickup_item": rospy.get_param("~target_item", "").strip(),
            "pickup_workshop": rospy.get_param("~target_workshop", "").strip(),
            "sim_item": rospy.get_param("~sim_item", "").strip(),
            "sim_workshop": rospy.get_param("~sim_workshop", "").strip(),
        }

        self.qr_items = OrderedDict()
        self.qr_collecting = False
        self.task1_instruction = ""
        self.task1_reasoning_started = False
        self.task1_reasoning_result = None
        self.task1_reasoning_error = ""
        self.task1_reasoning_done = threading.Event()
        self.qr_navigation_watching = False
        self.qr_navigation_goal_id = ""
        self.qr_navigation_result = None
        self.qr_odom_yaw = None
        self.qr_odom_received_at = 0.0
        self.qr_decoder_ready = False
        self.qr_decoder_pending_count = 0
        self.qr_decoder_status_at = 0.0
        self.task1_instruction = ""
        self.task1_llm_generation = 0
        self.task1_llm_thread = None
        self.task1_llm_done = threading.Event()
        self.task1_llm_result = None
        self.task1_llm_error = ""
        self.task1_llm_items = []
        self.base_twist = None
        self.base_pose = None
        self.handoff_scan_received_at = 0.0
        self.rear_scan_min = None
        self.handoff_costmap_received_at = 0.0
        self.task2_inter_visit_rear_half_angle = math.radians(max(
            1.0, float(rospy.get_param(
                "~task2_inter_visit_rear_half_angle_deg", 30.0))))
        self.ocr_target = None
        self.ocr_last_message_at = 0.0
        self.ocr_filter = TemporalTargetFilter(
            rospy.get_param("~ocr_required_hits", 2),
            rospy.get_param("~ocr_evidence_window_sec", 1.5),
        )
        self.ocr_memory_filters = {
            category: TemporalTargetFilter(
                rospy.get_param("~ocr_memory_required_hits", 2),
                rospy.get_param("~ocr_evidence_window_sec", 1.5),
            )
            for category in CATEGORY_LABELS
        }
        self.ocr_memory_min_score = float(rospy.get_param(
            "~ocr_memory_min_score", 0.55))
        self.task2_non_target_early_exit = bool_param(
            "~task2_non_target_early_exit", True)
        self.ocr_non_target_min_score = float(rospy.get_param(
            "~task2_non_target_early_exit_min_score", 0.62))
        self.ocr_trigger_min_bbox_width_ratio = float(rospy.get_param(
            "~task2_trigger_min_bbox_width_ratio", 0.09))
        self.ocr_trigger_min_bbox_height_ratio = float(rospy.get_param(
            "~task2_trigger_min_bbox_height_ratio", 0.06))
        self.ocr_trigger_min_bbox_area_ratio = float(rospy.get_param(
            "~task2_trigger_min_bbox_area_ratio", 0.006))
        self.task2_warehouse_memory = {}
        self.task2_non_target_announced = set()
        self.current_coverage_anchor = None
        self.last_coverage_anchor = None
        self.vision_trigger_latched = False
        self.trigger_request_pending = False
        self.trigger_request_started_at = 0.0
        self.trigger_service_accepted = False
        self.trigger_acknowledged = False
        self.trigger_service_name = rospy.get_param(
            "~target_trigger_service", "/vision_triggered_navigator/trigger_target")
        self.factory_navigation_start_service = rospy.get_param(
            "~factory_navigation_start_service",
            "/vision_triggered_navigator/start_navigation")
        self.task2_prewarm_enabled = bool_param(
            "~task2_prewarm_enabled", True)
        self.task2_prewarm_category = None
        self.trigger_ack_timeout = float(rospy.get_param("~trigger_ack_timeout_sec", 2.0))
        self.navigator_status = ""
        self.task2_announcement_completed = False
        self.traffic_decision = rospy.get_param("~traffic_decision", "").strip().lower()
        self.red_announced = False
        self.strict_mission_status = {}
        self.track_status = {}
        self.rotation_scan_min = None

        rospy.Subscriber("/wakeup", String, self._wakeup_cb, queue_size=5)
        rospy.Subscriber("/question", String, self._question_cb, queue_size=5)
        rospy.Subscriber("/qr_code_data", String, self._qr_cb, queue_size=20)
        rospy.Subscriber(
            "/qr_decoder/status", String,
            self._qr_decoder_status_cb, queue_size=20)
        rospy.Subscriber(
            rospy.get_param("~qr_odom_topic", "/odom"),
            Odometry,
            self._qr_odom_cb,
            queue_size=10,
        )
        rospy.Subscriber(
            "/scan", LaserScan, self._handoff_scan_cb, queue_size=1)
        rospy.Subscriber(
            "/move_base/local_costmap/costmap", OccupancyGrid,
            self._handoff_costmap_cb, queue_size=1)
        rospy.Subscriber(
            "/move_base/goal", MoveBaseActionGoal, self._qr_move_base_goal_cb, queue_size=5
        )
        rospy.Subscriber(
            "/move_base/result",
            MoveBaseActionResult,
            self._qr_move_base_result_cb,
            queue_size=5,
        )
        rospy.Subscriber(
            "/factory_sign_ppocr_rknn_test/result", String, self._ocr_cb, queue_size=20
        )
        rospy.Subscriber(
            "/vision_triggered_navigator/status", String, self._navigator_cb, queue_size=20
        )
        rospy.Subscriber(
            "/traffic_light_rknn_test/detections", String, self._traffic_cb, queue_size=20
        )
        rospy.Subscriber(
            "/strict_mission/status", String, self._strict_mission_cb, queue_size=20
        )
        for _, topic, _ in TRACK_CONFIG.values():
            rospy.Subscriber(topic, String, self._track_cb, callback_args=topic, queue_size=10)

        rospy.Service("/competition/resume", Trigger, self._resume_cb)
        rospy.Service("/competition/abort", Trigger, self._abort_cb)
        rospy.on_shutdown(self.shutdown)

        self.move_base = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        self.publish_status("startup", "ready", "competition controller ready")

    # ------------------------------ callbacks ------------------------------
    def _wakeup_cb(self, _msg):
        with self.lock:
            if self.voice_command_acknowledged or self.voice_command_ack_in_progress:
                return
            self.wakeup_received = True
            self.voice_prompt_started = True
            self.voice_wakeup_generation += 1
            generation = self.voice_wakeup_generation
        threading.Thread(
            target=self._prompt_and_start_listening,
            args=(generation,),
            name="voice-wakeup-handshake",
            daemon=True,
        ).start()

    def _question_cb(self, msg):
        question = msg.data.strip()
        pickup_category, sim_category = parse_task1_categories(question)
        with self.lock:
            if not self.voice_listening or self.voice_command_ack_in_progress:
                rospy.logwarn("ignoring /question outside active voice window: %s", question)
                return
            if not pickup_category or not sim_category:
                rospy.logwarn("ignoring voice text without two target categories: %s", question)
                self.publish_status(
                    "task1", "listening_command",
                    "command must contain physical and simulation categories: {}".format(
                        question
                    ),
                )
                return
            self.question = question
            self.voice_listening = False
            self.voice_command_ack_in_progress = True
        threading.Thread(
            target=self._finish_voice_command,
            args=(pickup_category, sim_category, question),
            name="voice-command-handshake",
            daemon=True,
        ).start()

    def _voice_control(self, param_name, default_service):
        service = rospy.get_param(param_name, default_service)
        try:
            rospy.wait_for_service(service, timeout=5.0)
            response = rospy.ServiceProxy(service, Trigger)()
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError("voice control service failed: {}".format(exc))
        if not response.success:
            raise StageError("voice control rejected: {}".format(response.message))

    def _start_voice_listening(self):
        self._voice_control(
            "~voice_start_listening_service",
            "/speech_command_node/start_listening",
        )

    def _stop_voice_listening(self):
        self._voice_control(
            "~voice_stop_listening_service",
            "/speech_command_node/stop_listening",
        )

    def _set_voice_handshake_error(self, exc):
        with self.lock:
            self.voice_handshake_error = str(exc)
            self.voice_listening = False
            self.voice_command_ack_in_progress = False
        rospy.logerr("voice handshake failed: %s", exc)

    def _prompt_and_start_listening(self, generation):
        try:
            with self.voice_transition_lock:
                with self.lock:
                    if generation != self.voice_wakeup_generation or self.voice_command_acknowledged:
                        return
                    was_listening = self.voice_listening
                    self.voice_listening = False
                if was_listening:
                    self._stop_voice_listening()

                reply = rospy.get_param("~voice_wakeup_reply", "我在").strip() or "我在"
                self.publish_status("task1", "wakeup_ack", "replying and preparing ASR")
                self.announce("custom", text=reply)

                with self.lock:
                    if generation != self.voice_wakeup_generation or self.voice_command_acknowledged:
                        return
                self._start_voice_listening()
                with self.lock:
                    self.voice_listening = True
                self.publish_status(
                    "task1", "listening_command",
                    "waiting for physical and simulation target categories",
                )
        except Exception as exc:
            self._set_voice_handshake_error(exc)

    def _finish_voice_command(self, pickup_category, sim_category, question):
        try:
            with self.voice_transition_lock:
                self._stop_voice_listening()
                reply = rospy.get_param("~voice_command_reply", "好的").strip() or "好的"
                self.publish_status(
                    "task1", "command_ack",
                    "pickup_category={} sim_category={} reply={}".format(
                        pickup_category, sim_category, reply
                    ),
                )
                self.announce("custom", text=reply)
                with self.lock:
                    self.category = pickup_category
                    self.sim_category = sim_category
                    self.voice_command_acknowledged = True
                    self.voice_command_ack_in_progress = False
                self.publish_status(
                    "task1", "voice_ready", "voice command accepted; navigation may start"
                )
        except Exception as exc:
            self._set_voice_handshake_error(exc)

    def _voice_command_ready(self):
        with self.lock:
            if self.voice_handshake_error:
                raise StageError(self.voice_handshake_error)
            return (
                self.wakeup_received
                and self.voice_command_acknowledged
                and self.category
                and self.sim_category
            )

    def _qr_cb(self, msg):
        if not self.qr_collecting:
            return
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        accepted = False
        for key, result in qr_values_from_payload(payload):
            with self.lock:
                if key not in self.qr_items:
                    self.qr_items[key] = result
                    accepted = True
                    rospy.loginfo("QR accepted %d/3: %s", len(self.qr_items), result)
        if accepted:
            self._start_task1_reasoning_if_ready()

    def _qr_decoder_status_cb(self, msg):
        try:
            payload = json.loads(msg.data)
            state = str(payload.get("state") or "")
            pending_count = max(0, int(payload.get("pending_count", 0)))
        except (TypeError, ValueError):
            return
        with self.lock:
            self.qr_decoder_ready = state in ("ready", "idle", "fetching")
            self.qr_decoder_pending_count = pending_count
            self.qr_decoder_status_at = time.monotonic()

    def _start_task1_reasoning_if_ready(self):
        expected_count = int(rospy.get_param("~qr_expected_count", 3))
        with self.lock:
            if (self.task1_reasoning_started or
                    len(self.qr_items) < expected_count or
                    not self.task1_instruction):
                return False
            items = list(self.qr_items.values())[:expected_count]
            instruction = self.task1_instruction
            self.task1_reasoning_started = True
        rospy.loginfo(
            "Task1 pipeline: starting Spark while QR scan teardown continues: %s",
            ", ".join(items),
        )
        threading.Thread(
            target=self._task1_reasoning_worker,
            args=(items, instruction),
            name="task1-spark-reasoning",
            daemon=True,
        ).start()
        self._prewarm_task2()
        return True

    def _task1_reasoning_worker(self, items, instruction):
        started_at = time.monotonic()
        service = rospy.get_param(
            "~llm_service", "/smart_factory_llm/reason_pickup_order")
        result = None
        error = ""
        try:
            rospy.wait_for_service(service, timeout=15.0)
            result = rospy.ServiceProxy(service, ReasonPickupOrder)(
                items[0], items[1], items[2], instruction
            )
        except (rospy.ROSException, rospy.ServiceException, IndexError) as exc:
            error = "LLM service failed: {}".format(exc)
        with self.lock:
            self.task1_reasoning_result = result
            self.task1_reasoning_error = error
        rospy.loginfo(
            "Task1 pipeline: Spark finished in %.3fs success=%s",
            time.monotonic() - started_at,
            bool(result and result.success and not error),
        )
        self.task1_reasoning_done.set()

    def _wait_for_task1_reasoning(self):
        timeout = max(1.0, float(rospy.get_param(
            "~task1_reasoning_timeout_sec", 200.0)))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            if self.task1_reasoning_done.wait(0.05):
                with self.lock:
                    error = self.task1_reasoning_error
                    result = self.task1_reasoning_result
                if error:
                    raise StageError(error)
                if result is None:
                    raise StageError("LLM reasoning completed without a response")
                return result
        raise StageError("LLM reasoning timed out after {:.1f}s".format(timeout))

    def _qr_odom_cb(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        with self.lock:
            self.qr_odom_yaw = yaw
            self.qr_odom_received_at = time.monotonic()
            self.base_twist = (
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.linear.y),
                float(msg.twist.twist.angular.z),
            )
            self.base_pose = (
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                yaw,
            )

    def _handoff_scan_cb(self, msg):
        nearest = scan_sector_min(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            math.pi,
            self.task2_inter_visit_rear_half_angle,
            msg.range_min,
            msg.range_max,
        )
        rotation_nearest = scan_sector_min(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            0.0,
            math.pi,
            msg.range_min,
            msg.range_max,
        )
        with self.lock:
            self.handoff_scan_received_at = time.monotonic()
            self.rear_scan_min = nearest
            self.rotation_scan_min = rotation_nearest

    def _handoff_costmap_cb(self, _msg):
        with self.lock:
            self.handoff_costmap_received_at = time.monotonic()

    def _qr_move_base_goal_cb(self, msg):
        with self.lock:
            if self.qr_navigation_watching:
                self.qr_navigation_goal_id = msg.goal_id.id

    def _qr_move_base_result_cb(self, msg):
        with self.lock:
            if not self.qr_navigation_watching or not self.qr_navigation_goal_id:
                return
            if msg.status.goal_id.id != self.qr_navigation_goal_id:
                return
            self.qr_navigation_result = msg.status.status

    def _ocr_cb(self, msg):
        if not self.ocr_target:
            return
        try:
            payload = json.loads(msg.data)
            category = normalize_category(payload.get("category"))
        except Exception:
            return
        now = time.monotonic()
        self.ocr_last_message_at = now
        memory_confirmed = False
        for candidate, candidate_filter in self.ocr_memory_filters.items():
            if candidate_filter.push(candidate, category, now):
                memory_confirmed = candidate == category
        score = float(payload.get("category_score", 0.0) or 0.0)
        bbox = payload.get("target_bbox")
        bbox_ratios = target_bbox_ratios(
            bbox, payload.get("image_width"), payload.get("image_height"))
        trigger_eligible = target_bbox_is_close_enough(
            bbox,
            payload.get("image_width"),
            payload.get("image_height"),
            self.ocr_trigger_min_bbox_width_ratio,
            self.ocr_trigger_min_bbox_height_ratio,
            self.ocr_trigger_min_bbox_area_ratio,
        )
        with self.lock:
            active_anchor = self.current_coverage_anchor
        target_trigger_eligible = task2_target_trigger_is_eligible(
            trigger_eligible, active_anchor)
        non_target_event = None
        if (memory_confirmed and payload.get("target_bbox") and
                score >= self.ocr_memory_min_score):
            with self.lock:
                anchor = self.current_coverage_anchor
                previous = self.task2_warehouse_memory.get(category)
                pose = self.base_pose
                yaw = pose[2] if pose is not None else None
                area_ratio = bbox_ratios[2] if bbox_ratios is not None else 0.0
                previous_quality = (
                    float(previous.get("area_ratio", 0.0)),
                    float(previous.get("score", 0.0)),
                ) if previous is not None else (-1.0, -1.0)
                quality = (area_ratio, score)
                if (anchor and yaw is not None and math.isfinite(yaw) and
                        (previous is None or quality > previous_quality)):
                    self.task2_warehouse_memory[category] = {
                        "anchor": int(anchor),
                        "score": score,
                        "area_ratio": area_ratio,
                        "odom_yaw": float(yaw),
                        "stamp": time.time(),
                    }
                    # rospy.loginfo(
                    #     "task2 warehouse memory: category=%s anchor=%d "
                    #     "score=%.3f area=%.4f odom_yaw=%.3f",
                    #     category, anchor, score, area_ratio, yaw)
                if non_target_observation_is_actionable(
                        self.ocr_target,
                        category,
                        memory_confirmed,
                        trigger_eligible,
                        score,
                        self.ocr_non_target_min_score,
                        anchor):
                    event_key = (int(anchor), category)
                    if event_key not in self.task2_non_target_announced:
                        self.task2_non_target_announced.add(event_key)
                        non_target_event = {
                            "category": category,
                            "anchor": int(anchor),
                            "score": score,
                            "area_ratio": area_ratio,
                            "odom_yaw": yaw,
                            "confirmed": True,
                        }
        if non_target_event is not None:
            self.vision_non_target_pub.publish(String(
                data=json.dumps(non_target_event, ensure_ascii=False)))
            # rospy.loginfo(
            #     "task2 non-target early-exit notice: category=%s anchor=%d "
            #     "score=%.3f area=%.4f",
            #     non_target_event["category"],
            #     non_target_event["anchor"],
            #     non_target_event["score"],
            #     non_target_event["area_ratio"],
            # )
        if category == self.ocr_target and target_trigger_eligible:
            self.vision_target_pub.publish(msg)
        confirmed = self.ocr_filter.push(
            self.ocr_target,
            category if target_trigger_eligible else None,
            time.monotonic(),
        )
        # rospy.loginfo_throttle(
        #     0.5,
        #     "task2 OCR filter: target=%s observed=%s hits=%d/%d "
        #     "bbox=%s close=%s anchor=%s trigger_eligible=%s bbox_ratio=%s",
        #     self.ocr_target,
        #     category or "none",
        #     self.ocr_filter.hit_count,
        #     self.ocr_filter.required,
        #     bool(bbox),
        #     trigger_eligible,
        #     active_anchor or "transit",
        #     target_trigger_eligible,
        #     ("%.3f/%.3f/%.4f" % bbox_ratios
        #      if bbox_ratios is not None else "invalid"),
        # )
        if (confirmed and category == self.ocr_target and
                target_trigger_eligible and
                not self.vision_trigger_latched):
            self.vision_trigger_latched = True
            self.trigger_request_pending = True
            self.trigger_request_started_at = time.monotonic()
            self.trigger_service_accepted = False
            self.trigger_acknowledged = False
            self.publish_status(
                "task2", "trigger_pending",
                "OCR target confirmed; requesting navigator acknowledgement")
            # rospy.loginfo(
            #     "task2 OCR target confirmed: target=%s hits=%d/%d; "
            #     "reliable trigger pending (will not retrigger)",
            #     self.ocr_target,
            #     self.ocr_filter.hit_count,
            #     self.ocr_filter.required,
            # )

    def _navigator_cb(self, msg):
        status = msg.data.strip().lower()
        if status == "centering_recovering":
            with self.lock:
                self.vision_trigger_latched = False
                self.trigger_request_pending = False
                self.trigger_service_accepted = False
                self.trigger_acknowledged = False
                self.ocr_filter.reset()
                self.current_coverage_anchor = None
            rospy.logwarn(
                "task2 target centering lost; rearmed OCR for a fresh "
                "stationary anchor observation")
        if status == "parking_staging_recovering":
            with self.lock:
                self.vision_trigger_latched = False
                self.trigger_request_pending = False
                self.trigger_service_accepted = False
                self.trigger_acknowledged = False
                self.ocr_filter.reset()
                self.current_coverage_anchor = None
            rospy.logwarn(
                "task2 parking staging failed; rearmed OCR for a fresh "
                "stationary anchor observation with adjusted parking geometry")
        observing_prefixes = (
            "coverage_anchor_observing:",
            "coverage_remembered_heading_observing:",
        )
        if status.startswith(observing_prefixes):
            try:
                anchor = int(status.rsplit(":", 1)[1])
            except (TypeError, ValueError):
                anchor = None
            with self.lock:
                self.current_coverage_anchor = anchor
                if anchor is not None:
                    self.last_coverage_anchor = anchor
            rospy.loginfo("task2 observing coverage anchor: %s", anchor)
            return
        if status.startswith("coverage_anchor_transit:"):
            with self.lock:
                self.current_coverage_anchor = None
            return
        if status != self.navigator_status:
            rospy.loginfo("task2 navigator status: %s", status)
        self.navigator_status = status

    def _deliver_target_trigger(self):
        """Deliver one OCR lock through a synchronous service and wait for status ACK."""
        if not self.trigger_request_pending or self.trigger_acknowledged:
            return
        elapsed = time.monotonic() - self.trigger_request_started_at
        delivery_state = trigger_delivery_state(
            self.trigger_service_accepted,
            self.navigator_status,
            elapsed,
            self.trigger_ack_timeout,
        )
        if delivery_state == "acknowledged":
            self.trigger_acknowledged = True
            self.trigger_request_pending = False
            self.publish_status(
                "task2", "trigger_acknowledged",
                "navigator accepted and acknowledged the OCR target")
            rospy.loginfo(
                "task2 trigger acknowledged by navigator status=%s",
                self.navigator_status)
            return

        if delivery_state == "failed":
            self.publish_status(
                "task2", "trigger_delivery_failed", "",
                "navigator did not acknowledge target within {:.1f}s".format(
                    self.trigger_ack_timeout))
            raise StageError(
                "trigger_delivery_failed: no navigator acknowledgement within {:.1f}s".format(
                    self.trigger_ack_timeout))

        if self.trigger_service_accepted:
            return
        try:
            rospy.wait_for_service(self.trigger_service_name, timeout=0.15)
            response = rospy.ServiceProxy(self.trigger_service_name, Trigger)()
            if not response.success:
                rospy.logwarn_throttle(
                    0.5, "target trigger service rejected request: %s", response.message)
                return
            self.trigger_service_accepted = True
            rospy.loginfo("task2 target trigger service accepted: %s", response.message)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn_throttle(
                0.5, "waiting for reliable target trigger service %s: %s",
                self.trigger_service_name, str(exc))

    def _traffic_cb(self, msg):
        try:
            decision = traffic_decision_from_payload(json.loads(msg.data))
            if decision:
                self.traffic_decision = decision
        except Exception:
            return

    def _strict_mission_cb(self, msg):
        try:
            payload = json.loads(msg.data)
            if isinstance(payload, dict):
                self.strict_mission_status = payload
        except Exception:
            return

    def _track_cb(self, msg, topic):
        self.track_status[topic] = msg.data.strip().lower()

    def _resume_cb(self, _req):
        if self.aborted.is_set():
            return TriggerResponse(False, "competition already aborted")
        self.resume_event.set()
        return TriggerResponse(True, "competition resume requested")

    def _abort_cb(self, _req):
        self.aborted.set()
        self.resume_event.set()
        self.safe_stop(cancel_navigation=True)
        self.stop_all_children()
        return TriggerResponse(True, "competition aborted and vehicle stopped")

    # ------------------------------ infrastructure ------------------------------
    def publish_status(self, stage, state, message="", error=""):
        payload = {
            "stage": stage,
            "state": state,
            "message": message,
            "error": error,
            "stamp": time.time(),
        }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        rospy.loginfo("competition %s/%s: %s", stage, state, message or error)

    def check_abort(self):
        if self.aborted.is_set() or rospy.is_shutdown():
            raise Aborted("competition aborted")

    def pause_and_retry(self, stage, error):
        self.safe_stop(cancel_navigation=True)
        self.publish_status(stage, "paused", "call /competition/resume after fixing it", str(error))
        self.resume_event.clear()
        while not rospy.is_shutdown() and not self.resume_event.wait(0.2):
            self.check_abort()
        self.check_abort()
        self.publish_status(stage, "resuming", "retrying current stage")

    def run_stage(self, stage, function):
        while not rospy.is_shutdown():
            self.check_abort()
            try:
                return function()
            except StageError as exc:
                rospy.logerr("stage %s failed: %s", stage, exc)
                self.stop_all_children()
                self.pause_and_retry(stage, exc)

    def safe_stop(self, cancel_navigation=False):
        if cancel_navigation:
            try:
                self.move_base.cancel_all_goals()
            except Exception:
                pass
        for _ in range(3):
            self.cmd_pub.publish(Twist())
            rospy.sleep(0.03)

    def task1_task2_handoff(self):
        """Keep localization alive while proving all motion authority is idle."""
        self.publish_status(
            "task1", "task2_handoff",
            "cancelling navigation and waiting for a stationary base")
        self.safe_stop(cancel_navigation=True)
        timeout = float(rospy.get_param("~task1_task2_handoff_timeout_sec", 5.0))
        stable_required = float(rospy.get_param(
            "~task1_task2_handoff_stable_sec", 0.5))
        deadline = time.monotonic() + timeout
        stable_since = None
        stationary_ready = False
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            self.safe_stop(cancel_navigation=True)
            state = self.move_base.get_state()
            with self.lock:
                twist = self.base_twist
                odom_age = time.monotonic() - self.qr_odom_received_at
            idle = state not in (GoalStatus.PENDING, GoalStatus.ACTIVE)
            stopped = (twist is not None and odom_age <= 0.5 and
                       base_is_stopped(*twist))
            if idle and stopped:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= stable_required:
                    stationary_ready = True
                    break
            else:
                stable_since = None
            rospy.sleep(0.05)
        if not stationary_ready:
            raise StageError(
                "task1->task2 handoff did not reach {:.1f}s stationary idle state".format(
                    stable_required))

        self.publish_status(
            "task1", "task2_costmap_refreshing",
            "clearing QR-scan obstacle history before task2 coverage navigation")
        try:
            rospy.wait_for_service("/move_base/clear_costmaps", timeout=2.0)
            rospy.ServiceProxy("/move_base/clear_costmaps", Empty)()
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError(
                "task1->task2 costmap refresh failed: {}".format(exc))
        cleared_at = time.monotonic()
        refresh_deadline = cleared_at + 2.0
        while time.monotonic() < refresh_deadline and not rospy.is_shutdown():
            self.check_abort()
            with self.lock:
                scan_fresh = self.handoff_scan_received_at > cleared_at
                costmap_fresh = self.handoff_costmap_received_at > cleared_at
            if scan_fresh and costmap_fresh:
                break
            rospy.sleep(0.05)
        else:
            raise StageError(
                "task1->task2 costmap refresh produced no fresh scan/costmap snapshot")
        self._wait_transition_announcement("task1")
        self.safe_stop(cancel_navigation=True)
        self.publish_status(
            "task1", "completed",
            "voice, QR and reasoning completed; task2 may start immediately")
        self.publish_status(
            "task1", "task2_handoff_ready",
            "move_base idle; fresh costmap; preserving AMCL state")

    def _back_out_of_factory_bay(self, stage, state, detail, parameter_prefix):
        """Leave a wall-facing bay with rear lidar and odometry guards."""
        distance = max(0.0, float(rospy.get_param(
            "~{}_reverse_distance_m".format(parameter_prefix), 0.32)))
        speed = abs(float(rospy.get_param(
            "~{}_reverse_speed_mps".format(parameter_prefix), 0.08)))
        min_clearance = max(0.0, float(rospy.get_param(
            "~{}_rear_clearance_m".format(parameter_prefix), 0.28)))
        stale_sec = max(0.1, float(rospy.get_param(
            "~{}_sensor_stale_sec".format(parameter_prefix), 0.5)))
        timeout = max(1.0, float(rospy.get_param(
            "~{}_timeout_sec".format(parameter_prefix), 7.0)))
        if distance <= 0.0 or speed <= 0.0:
            raise StageError(
                "{} reverse parameters must be positive".format(
                    parameter_prefix))

        self.publish_status(stage, state, detail)
        self.safe_stop(cancel_navigation=True)
        ready_deadline = time.monotonic() + 2.0
        start_pose = None
        while time.monotonic() < ready_deadline and not rospy.is_shutdown():
            self.check_abort()
            with self.lock:
                odom_age = time.monotonic() - self.qr_odom_received_at
                scan_age = time.monotonic() - self.handoff_scan_received_at
                pose = self.base_pose
                rear_min = self.rear_scan_min
            if (pose is not None and rear_min is not None and
                    odom_age <= stale_sec and scan_age <= stale_sec):
                start_pose = pose
                break
            rospy.sleep(0.05)
        if start_pose is None:
            raise StageError(
                "{} has no fresh odom/rear lidar".format(parameter_prefix))

        deadline = time.monotonic() + timeout
        moved = 0.0
        command = Twist()
        command.linear.x = -speed
        try:
            rate = rospy.Rate(20)
            while time.monotonic() < deadline and not rospy.is_shutdown():
                self.check_abort()
                with self.lock:
                    odom_age = time.monotonic() - self.qr_odom_received_at
                    scan_age = time.monotonic() - self.handoff_scan_received_at
                    pose = self.base_pose
                    rear_min = self.rear_scan_min
                if (pose is None or rear_min is None or
                        odom_age > stale_sec or scan_age > stale_sec):
                    raise StageError(
                        "{} lost fresh odom/rear lidar".format(
                            parameter_prefix))
                moved = math.hypot(
                    pose[0] - start_pose[0], pose[1] - start_pose[1])
                if moved >= distance:
                    break
                if rear_min <= min_clearance:
                    raise StageError(
                        "{} rear path blocked at {:.3f}m".format(
                            parameter_prefix, rear_min))
                self.cmd_pub.publish(command)
                rate.sleep()
            else:
                raise StageError(
                    "{} timed out after moving {:.3f}m".format(
                        parameter_prefix, moved))
        finally:
            self.safe_stop(cancel_navigation=True)
        return moved

    def task2_inter_visit_handoff(self):
        """Back out of the first wall-facing bay before searching again."""
        moved = self._back_out_of_factory_bay(
            "task2",
            "leaving_physical_factory",
            "backing out of the first parking bay before the second search",
            "task2_inter_visit",
        )

        self.publish_status(
            "task2", "refreshing_second_search",
            "left first bay by {:.3f}m; refreshing navigation costmaps".format(
                moved))
        try:
            rospy.wait_for_service("/move_base/clear_costmaps", timeout=2.0)
            rospy.ServiceProxy("/move_base/clear_costmaps", Empty)()
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError(
                "task2 second-search costmap refresh failed: {}".format(exc))
        cleared_at = time.monotonic()
        refresh_deadline = cleared_at + 2.0
        while time.monotonic() < refresh_deadline and not rospy.is_shutdown():
            self.check_abort()
            with self.lock:
                scan_fresh = self.handoff_scan_received_at > cleared_at
                costmap_fresh = self.handoff_costmap_received_at > cleared_at
            if scan_fresh and costmap_fresh:
                self.publish_status(
                    "task2", "second_search_ready",
                    "vehicle clear of first bay; fresh costmap available")
                return
            rospy.sleep(0.05)
        raise StageError(
            "task2 second-search refresh produced no fresh scan/costmap snapshot")

    def production_task4_handoff(self, source_stage):
        """Resume physical navigation without resetting the current factory pose."""
        source_stage = str(source_stage or "task3").strip().lower()
        self.publish_status(
            source_stage, "task4_handoff",
            "preserving AMCL pose and preparing physical navigation")
        self.safe_stop(cancel_navigation=True)
        # Never allow a future transition announcement to overlap any motion.
        self._wait_transition_announcement(source_stage)
        if bool_param("~task4_factory_egress_enabled", True):
            moved = self._back_out_of_factory_bay(
                source_stage,
                "leaving_final_factory",
                "backing out of the final parking bay before task4 navigation",
                "task4_factory_egress",
            )
            self.publish_status(
                source_stage,
                "final_factory_exit_ready",
                "left final parking bay by {:.3f}m before task4".format(
                    moved),
            )
        timeout = float(rospy.get_param(
            "~task3_task4_handoff_timeout_sec", 5.0))
        stable_required = float(rospy.get_param(
            "~task3_task4_handoff_stable_sec", 0.5))
        deadline = time.monotonic() + timeout
        stable_since = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            self.safe_stop(cancel_navigation=True)
            state = self.move_base.get_state()
            with self.lock:
                twist = self.base_twist
                odom_age = time.monotonic() - self.qr_odom_received_at
            idle = state not in (GoalStatus.PENDING, GoalStatus.ACTIVE)
            stopped = (twist is not None and odom_age <= 0.5 and
                       base_is_stopped(*twist))
            if idle and stopped:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= stable_required:
                    break
            else:
                stable_since = None
            rospy.sleep(0.05)
        else:
            raise StageError(
                "{}->task4 handoff did not reach {:.1f}s stationary idle state".format(
                    source_stage, stable_required))

        try:
            rospy.wait_for_service("/move_base/clear_costmaps", timeout=2.0)
            rospy.ServiceProxy("/move_base/clear_costmaps", Empty)()
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError(
                "{}->task4 costmap refresh failed: {}".format(source_stage, exc))
        cleared_at = time.monotonic()
        refresh_deadline = cleared_at + 2.0
        while time.monotonic() < refresh_deadline and not rospy.is_shutdown():
            self.check_abort()
            with self.lock:
                scan_fresh = self.handoff_scan_received_at > cleared_at
                costmap_fresh = self.handoff_costmap_received_at > cleared_at
            if scan_fresh and costmap_fresh:
                break
            rospy.sleep(0.05)
        else:
            raise StageError(
                "{}->task4 costmap refresh produced no fresh scan/costmap snapshot".format(
                    source_stage))
        if bool_param("~task4_internal_waypoint_enabled", False):
            waypoint_x = float(rospy.get_param(
                "~task4_internal_waypoint_x", 1.2660))
            waypoint_y = float(rospy.get_param(
                "~task4_internal_waypoint_y", -2.8863))
            waypoint_yaw = float(rospy.get_param(
                "~task4_internal_waypoint_yaw", -1.5443))
            waypoint_timeout = max(5.0, float(rospy.get_param(
                "~task4_internal_waypoint_timeout_sec", 45.0)))
            self.navigate(
                waypoint_x,
                waypoint_y,
                waypoint_yaw,
                source_stage,
                timeout_sec=waypoint_timeout,
                status_state="routing_inside_production",
            )
            self.safe_stop(cancel_navigation=True)
            self.publish_status(
                source_stage,
                "internal_route_ready",
                "reached calibrated internal waypoint before stop-line navigation",
            )
        self.safe_stop(cancel_navigation=True)
        self.publish_status(
            source_stage, "task4_handoff_ready",
            "announcement and costmap refresh complete; task4 may navigate immediately")

    def start_child(self, key, package, launch_file, args=None):
        self.stop_child(key)
        command = ["roslaunch", package, launch_file]
        for name, value in (args or {}).items():
            command.append("{}:={}".format(name, str(value).lower() if isinstance(value, bool) else value))
        rospy.loginfo("starting child: %s", " ".join(command))
        self.children[key] = subprocess.Popen(command, start_new_session=True)
        return self.children[key]

    def stop_child(self, key):
        proc = self.children.pop(key, None)
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                proc.kill()

    def stop_all_children(self):
        for key in list(self.children):
            self.stop_child(key)

    def wait_loop(self, timeout, predicate, child_key=None):
        deadline = time.time() + timeout if timeout > 0 else None
        while not rospy.is_shutdown():
            self.check_abort()
            result = predicate()
            if result:
                return result
            if child_key and child_key in self.children:
                code = self.children[child_key].poll()
                if code is not None:
                    raise StageError("{} exited unexpectedly with code {}".format(child_key, code))
            if deadline and time.time() >= deadline:
                raise StageError("stage timed out after {:.1f}s".format(timeout))
            rospy.sleep(0.1)

    def navigate(self, x, y, yaw, stage, timeout_sec=None, status_state="navigating"):
        timeout = float(
            timeout_sec
            if timeout_sec is not None
            else rospy.get_param("~move_base_timeout_sec", 90.0)
        )
        if not self.move_base.wait_for_server(rospy.Duration(min(timeout, 30.0))):
            raise StageError("move_base action server unavailable")
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = float(x)
        goal.target_pose.pose.position.y = float(y)
        goal.target_pose.pose.orientation.z = math.sin(float(yaw) / 2.0)
        goal.target_pose.pose.orientation.w = math.cos(float(yaw) / 2.0)
        self.publish_status(
            stage,
            status_state,
            "goal x={:.3f} y={:.3f} yaw={:.3f}".format(x, y, yaw),
        )
        self.move_base.send_goal(goal)
        deadline = time.time() + timeout
        while time.time() < deadline and not rospy.is_shutdown():
            self.check_abort()
            state = self.move_base.get_state()
            if state not in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                if state == GoalStatus.SUCCEEDED:
                    return
                raise StageError("move_base failed with state {}".format(state))
            rospy.sleep(0.1)
        self.move_base.cancel_goal()
        raise StageError("move_base goal timed out")

    def announce(self, event, item="", workshop="", decision="", text=""):
        service = rospy.get_param("~announce_service", "/competition_speech/announce")
        try:
            rospy.wait_for_service(service, timeout=5.0)
            response = rospy.ServiceProxy(service, Announce)(
                event, item, workshop, decision, text, True
            )
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError("speech service failed: {}".format(exc))
        if not response.success:
            raise StageError("speech rejected: {}".format(response.message))

    def _start_announcement(self, event, item="", workshop="", decision="", text=""):
        task = {
            "done": threading.Event(),
            "error": None,
            "event": str(event),
        }

        def worker():
            try:
                self.announce(
                    event,
                    item=item,
                    workshop=workshop,
                    decision=decision,
                    text=text,
                )
            except Exception as exc:
                task["error"] = exc
            finally:
                task["done"].set()

        threading.Thread(
            target=worker,
            name="announcement-{}".format(event),
            daemon=True,
        ).start()
        return task

    def _wait_announcement(self, task):
        if task is None:
            return
        while not task["done"].wait(0.05):
            self.check_abort()
        if task["error"] is not None:
            error = task["error"]
            if isinstance(error, StageError):
                raise error
            raise StageError("speech failed: {}".format(error))

    def _start_transition_announcement(
            self, event, item="", workshop="", decision="", text=""):
        if self.transition_announcement is not None:
            raise StageError("previous transition announcement is still active")
        self.transition_announcement = self._start_announcement(
            event,
            item=item,
            workshop=workshop,
            decision=decision,
            text=text,
        )

    def _wait_transition_announcement(self, expected_event):
        task = self.transition_announcement
        if task is None:
            return
        if task["event"] != str(expected_event):
            raise StageError(
                "pending {} announcement cannot complete {} transition".format(
                    task["event"], expected_event))
        try:
            self._wait_announcement(task)
        finally:
            self.transition_announcement = None

    def _announce_while_stationary(
            self, event, item="", workshop="", decision="", text=""):
        """Hold exclusive zero-velocity authority until speech is complete."""
        self.safe_stop(cancel_navigation=True)
        stable_required = max(0.1, float(rospy.get_param(
            "~stationary_announcement_stable_sec", 0.5)))
        odom_stale_sec = max(0.1, float(rospy.get_param(
            "~stationary_announcement_odom_stale_sec", 0.5)))
        ready_timeout = max(stable_required, float(rospy.get_param(
            "~stationary_announcement_ready_timeout_sec", 3.0)))
        deadline = time.monotonic() + ready_timeout
        stable_since = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            self.cmd_pub.publish(Twist())
            with self.lock:
                twist = self.base_twist
                odom_age = time.monotonic() - self.qr_odom_received_at
            stopped = (
                twist is not None
                and odom_age <= odom_stale_sec
                and base_is_stopped(*twist)
            )
            if stopped:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= stable_required:
                    break
            else:
                stable_since = None
            rospy.sleep(0.05)
        else:
            raise StageError(
                "vehicle did not become stationary before {} announcement".format(
                    event))

        task = self._start_announcement(
            event,
            item=item,
            workshop=workshop,
            decision=decision,
            text=text,
        )
        motion_observed = False
        while not task["done"].wait(0.05):
            self.check_abort()
            self.cmd_pub.publish(Twist())
            with self.lock:
                twist = self.base_twist
                odom_age = time.monotonic() - self.qr_odom_received_at
            if (twist is None or odom_age > odom_stale_sec or
                    not base_is_stopped(*twist)):
                motion_observed = True
        self._wait_announcement(task)
        self.safe_stop(cancel_navigation=True)
        if motion_observed:
            raise StageError(
                "vehicle motion or stale odometry observed during {} announcement".format(
                    event))

    def navigate_to_qr_area(self):
        """Run the untouched simple_navigator and observe its move_base result."""
        timeout = float(rospy.get_param("~qr_navigation_timeout_sec", 120.0))
        self.safe_stop(cancel_navigation=True)
        with self.lock:
            self.qr_navigation_watching = True
            self.qr_navigation_goal_id = ""
            self.qr_navigation_result = None
        self.publish_status(
            "task1",
            "navigating",
            "running roslaunch simple_navigator navigate.launch",
        )
        try:
            self.start_child("qr_navigator", "simple_navigator", "navigate.launch")
            deadline = time.time() + timeout
            child_exited_at = None
            while not rospy.is_shutdown():
                self.check_abort()
                with self.lock:
                    result = self.qr_navigation_result
                if result is not None:
                    if result == GoalStatus.SUCCEEDED:
                        break
                    raise StageError(
                        "simple_navigator move_base result state={}".format(result)
                    )

                proc = self.children.get("qr_navigator")
                if proc and proc.poll() is not None:
                    if child_exited_at is None:
                        child_exited_at = time.time()
                    elif time.time() - child_exited_at >= 1.0:
                        raise StageError(
                            "simple_navigator exited without a move_base result (code {})".format(
                                proc.returncode
                            )
                        )
                if time.time() >= deadline:
                    raise StageError(
                        "simple_navigator timed out after {:.1f}s".format(timeout)
                    )
                rospy.sleep(0.1)
        finally:
            with self.lock:
                self.qr_navigation_watching = False
            self.stop_child("qr_navigator")
            self.safe_stop(cancel_navigation=True)

        self.publish_status(
            "task1",
            "qr_area_arrived",
            "simple_navigator reached the configured QR-area waypoint",
        )

    def _qr_count(self):
        with self.lock:
            return len(self.qr_items)

    def _check_qr_decoder(self):
        proc = self.children.get("qr_decoder")
        if proc and proc.poll() is not None:
            raise StageError(
                "QR decoder exited unexpectedly with code {}".format(proc.returncode)
            )

    def _fresh_qr_odom_yaw(self, stale_sec):
        with self.lock:
            yaw = self.qr_odom_yaw
            received_at = self.qr_odom_received_at
        if yaw is None or time.monotonic() - received_at > stale_sec:
            raise StageError(
                "QR scan odometry is missing or stale for more than {:.2f}s".format(
                    stale_sec
                )
            )
        return yaw

    def _require_rotation_clearance(self, stale_sec, minimum, label):
        now = time.monotonic()
        with self.lock:
            scan_age = now - self.handoff_scan_received_at
            nearest = self.rotation_scan_min
        if nearest is None or scan_age > stale_sec:
            self.safe_stop()
            raise StageError(
                "{} refused because /scan is missing or stale".format(label))
        if nearest < minimum:
            self.safe_stop()
            raise StageError(
                "{} refused: nearest obstacle {:.3f}m is below {:.3f}m".format(
                    label, nearest, minimum))

    def _wait_for_qr_odom(self, wait_sec, stale_sec):
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            self._check_qr_decoder()
            with self.lock:
                received_at = self.qr_odom_received_at
                yaw = self.qr_odom_yaw
            if yaw is not None and time.monotonic() - received_at <= stale_sec:
                return yaw
            rospy.sleep(0.05)
        raise StageError("QR scan did not receive fresh odometry within {:.1f}s".format(wait_sec))

    def _wait_for_qr_decoder_ready(self, timeout_sec):
        deadline = time.monotonic() + max(0.5, float(timeout_sec))
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            self._check_qr_decoder()
            with self.lock:
                ready = self.qr_decoder_ready
            if ready:
                return
            rospy.sleep(0.05)
        raise StageError(
            "QR decoder did not report ready within {:.1f}s".format(
                timeout_sec))

    def _settle_for_qr(
            self, duration, expected_count, scan_deadline, stale_sec,
            label="QR scan"):
        """Hold zero velocity until odometry proves a stable recognition dwell."""
        stable_required = max(0.05, float(rospy.get_param(
            "~qr_scan_stationary_hold_sec", 0.20)))
        stop_timeout = max(stable_required, float(rospy.get_param(
            "~qr_scan_stop_timeout_sec", 1.5)))
        local_deadline = min(
            scan_deadline,
            time.monotonic() + stop_timeout + stable_required + duration,
        )
        stable_since = None
        stop_confirmed = False
        zero = Twist()
        while time.monotonic() < local_deadline and not rospy.is_shutdown():
            self.check_abort()
            self._check_qr_decoder()
            self.cmd_pub.publish(zero)
            now = time.monotonic()
            with self.lock:
                twist = self.base_twist
                odom_age = now - self.qr_odom_received_at
            stopped = (
                twist is not None and odom_age <= stale_sec and
                base_is_stopped(*twist)
            )
            if stopped:
                if stable_since is None:
                    stable_since = now
                stable_duration = now - stable_since
                if not stop_confirmed and stable_duration >= stable_required:
                    stop_confirmed = True
                    rospy.loginfo(
                        "%s stop confirmed after %.2fs; recognition dwell %.2fs",
                        label, stable_duration, duration)
                if (stop_confirmed and
                        stable_duration >= stable_required + duration):
                    rospy.loginfo("%s stationary dwell completed", label)
                    return self._qr_count() >= expected_count
            else:
                if stop_confirmed:
                    rospy.logwarn(
                        "%s moved during recognition dwell; restarting stop confirmation",
                        label)
                stable_since = None
                stop_confirmed = False
            if self._qr_count() >= expected_count:
                return True
            rospy.sleep(0.05)
        raise StageError(
            "{} failed to remain stationary for {:.2f}s recognition dwell".format(
                label, duration))

    def _drain_qr_results(
            self, duration, expected_count, scan_deadline, stale_sec):
        self.safe_stop()
        drain_deadline = min(
            scan_deadline, time.monotonic() + max(0.0, float(duration)))
        idle_required = max(0.1, float(rospy.get_param(
            "~qr_scan_pending_idle_sec", 0.5)))
        idle_since = None
        while time.monotonic() < drain_deadline and not rospy.is_shutdown():
            self.check_abort()
            self._check_qr_decoder()
            self.cmd_pub.publish(Twist())
            self._fresh_qr_odom_yaw(stale_sec)
            if self._qr_count() >= expected_count:
                return True
            with self.lock:
                ready = self.qr_decoder_ready
                pending_count = self.qr_decoder_pending_count
            if ready and pending_count == 0:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since >= idle_required:
                    return False
            else:
                idle_since = None
            rospy.sleep(0.05)
        return self._qr_count() >= expected_count

    def _rotate_qr_step(
        self,
        angle,
        speed,
        direction,
        stale_sec,
        scan_deadline,
        step_margin,
    ):
        tracker = DirectedYawAccumulator(direction=direction)
        tracker.reset(self._fresh_qr_odom_yaw(stale_sec))
        minimum = max(0.0, float(rospy.get_param(
            "~qr_rotation_min_clearance", 0.28)))
        step_deadline = min(
            scan_deadline,
            time.monotonic() + angle / speed + step_margin,
        )
        twist = Twist()
        twist.angular.z = speed * direction
        while tracker.progress < angle and not rospy.is_shutdown():
            self.check_abort()
            self._check_qr_decoder()
            self._require_rotation_clearance(
                stale_sec, minimum, "QR scan rotation")
            if time.monotonic() >= step_deadline:
                raise StageError(
                    "QR scan failed to rotate {:.1f} degrees before step timeout".format(
                        math.degrees(angle)
                    )
                )
            yaw = self._fresh_qr_odom_yaw(stale_sec)
            if tracker.update(yaw) >= angle:
                break
            self.cmd_pub.publish(twist)
            rospy.sleep(0.05)
        self.safe_stop()

    def _return_qr_to_yaw(self, target_yaw, speed, tolerance, stale_sec, timeout):
        deadline = time.monotonic() + timeout
        minimum = max(0.0, float(rospy.get_param(
            "~qr_rotation_min_clearance", 0.28)))
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            self._check_qr_decoder()
            self._require_rotation_clearance(
                stale_sec, minimum, "QR final-yaw rotation")
            current_yaw = self._fresh_qr_odom_yaw(stale_sec)
            error = normalize_angle(target_yaw - current_yaw)
            if abs(error) <= tolerance:
                self.safe_stop()
                return
            twist = Twist()
            twist.angular.z = speed if error > 0.0 else -speed
            self.cmd_pub.publish(twist)
            rospy.sleep(0.05)
        self.safe_stop()
        raise StageError("QR scan failed to return to its original final yaw")

    def scan_qr_at_current_pose(self, status_state, total_angle_override=None,
                                extra_sweep_override=None, deadline_override=None):
        """Scan one revolution, then drain async results and optionally sweep again."""
        expected_count = int(rospy.get_param("~qr_expected_count", 3))
        speed = abs(float(rospy.get_param("~qr_scan_angular_speed", 0.80)))
        step_angle = abs(
            float(rospy.get_param("~qr_scan_step_angle_rad", math.radians(25.0)))
        )
        total_angle_param = (
            total_angle_override if total_angle_override is not None
            else rospy.get_param("~qr_scan_total_angle_rad", 2.0 * math.pi))
        total_angle = max(step_angle, abs(float(total_angle_param)))
        settle_sec = max(0.0, float(rospy.get_param("~qr_scan_settle_sec", 0.6)))
        warmup_sec = max(
            settle_sec,
            float(rospy.get_param("~qr_decoder_warmup_sec", 1.2)),
        )
        result_grace_sec = max(
            0.0, float(rospy.get_param("~qr_scan_result_grace_sec", 3.2))
        )
        decoder_ready_timeout = max(
            0.5, float(rospy.get_param(
                "~qr_decoder_ready_timeout_sec", 6.0))
        )
        extra_sweep_param = (
            extra_sweep_override if extra_sweep_override is not None
            else rospy.get_param(
                "~qr_scan_extra_sweep_angle_rad", math.radians(120.0)
            ))
        extra_sweep_angle = max(0.0, float(extra_sweep_param))
        scan_timeout = float(rospy.get_param("~qr_scan_timeout_sec", 60.0))
        stale_sec = float(rospy.get_param("~qr_odom_stale_sec", 0.5))
        odom_wait_sec = float(rospy.get_param("~qr_odom_wait_sec", 2.0))
        step_margin = float(
            rospy.get_param("~qr_scan_step_timeout_margin_sec", 2.0)
        )
        rotation_minimum = max(0.0, float(rospy.get_param(
            "~qr_rotation_min_clearance", 0.28)))
        if speed <= 0.0 or step_angle <= 0.0 or stale_sec <= 0.0:
            raise StageError("QR scan motion parameters must be positive")

        total_steps = int(math.ceil(total_angle / step_angle))
        scan_deadline = time.monotonic() + scan_timeout
        if deadline_override is not None:
            scan_deadline = min(scan_deadline, float(deadline_override))
        self.publish_status(
            "task1",
            status_state,
            "step scan start: count={}/{} steps={}".format(
                self._qr_count(), expected_count, total_steps
            ),
        )
        self._wait_for_qr_decoder_ready(decoder_ready_timeout)
        self._wait_for_qr_odom(odom_wait_sec, stale_sec)
        if self._settle_for_qr(
                warmup_sec, expected_count, scan_deadline, stale_sec,
                "QR warmup"):
            return True

        twist = Twist()
        twist.angular.z = speed
        tracker = DirectedYawAccumulator(direction=1.0)
        def rotate_steps(step_count, phase):
            for step_index in range(step_count):
                if self._qr_count() >= expected_count:
                    self.safe_stop()
                    return True
                if time.monotonic() >= scan_deadline:
                    self.safe_stop()
                    return False

                step_label = "{} step {}/{}".format(
                    phase, step_index + 1, step_count)
                tracker.reset(self._fresh_qr_odom_yaw(stale_sec))
                rospy.loginfo(
                    "%s rotating %.1fdeg", step_label,
                    math.degrees(step_angle))
                step_deadline = time.monotonic() + step_angle / speed + step_margin
                while tracker.progress < step_angle and not rospy.is_shutdown():
                    self.check_abort()
                    self._check_qr_decoder()
                    self._require_rotation_clearance(
                        stale_sec, rotation_minimum, "QR scan rotation")
                    if self._qr_count() >= expected_count:
                        self.safe_stop()
                        return True
                    if time.monotonic() >= scan_deadline:
                        self.safe_stop()
                        return False
                    if time.monotonic() >= step_deadline:
                        raise StageError(
                            "QR scan failed to rotate {:.1f} degrees before step timeout".format(
                                math.degrees(step_angle)
                            )
                        )
                    yaw = self._fresh_qr_odom_yaw(stale_sec)
                    if tracker.update(yaw) >= step_angle:
                        break
                    self.cmd_pub.publish(twist)
                    rospy.sleep(0.05)

                self.safe_stop()
                rospy.loginfo(
                    "%s angle reached %.1fdeg; confirming full stop",
                    step_label, math.degrees(tracker.progress))
                if self._settle_for_qr(
                    settle_sec, expected_count, scan_deadline, stale_sec,
                    step_label,
                ):
                    return True
            return self._qr_count() >= expected_count

        if rotate_steps(total_steps, "QR primary"):
            return True

        self.safe_stop()
        self.publish_status(
            "task1",
            status_state,
            "primary revolution finished: count={}/{}; draining async QR results".format(
                self._qr_count(), expected_count
            ),
        )
        if self._drain_qr_results(
            result_grace_sec, expected_count, scan_deadline, stale_sec
        ):
            return True

        # A decoder that starts on the first visible marker can miss that marker
        # while OpenCV and the HTTP worker warm up. Revisit only the opening arc
        # instead of spending another complete revolution at the same pose.
        if self._qr_count() < expected_count and extra_sweep_angle > 0.0:
            extra_steps = int(math.ceil(extra_sweep_angle / step_angle))
            self.publish_status(
                "task1",
                status_state,
                "{} QR missing after first revolution; extra sweep steps={}".format(
                    expected_count - self._qr_count(), extra_steps
                ),
            )
            if rotate_steps(extra_steps, "QR extra"):
                return True
            self.safe_stop()
            if self._drain_qr_results(
                result_grace_sec, expected_count, scan_deadline, stale_sec
            ):
                return True

        return self._qr_count() >= expected_count

    def scan_qr_at_fallback_point(self, expected_count, qr_total_deadline):
        """Navigate to the configured fallback point near the wall and rescan.

        到达备用点后持续旋转扫描，直到凑满 expected_count 个码
        或到达二维码扫描全流程总时间阈值 qr_total_deadline。
        """
        if not bool(rospy.get_param("~qr_fallback_enabled", False)):
            return False
        goal = rospy.get_param("~qr_fallback_goal", None)
        try:
            goal_x = float(goal["x"])
            goal_y = float(goal["y"])
            goal_yaw = float(goal["yaw"])
        except (KeyError, TypeError, ValueError) as exc:
            rospy.logwarn(
                "qr_fallback_enabled but ~qr_fallback_goal invalid (%s); "
                "skip fallback scan", exc)
            return False
        remaining = float(qr_total_deadline) - time.monotonic()
        if remaining <= 0.0:
            rospy.logwarn(
                "QR total timeout reached before fallback navigation; "
                "skip fallback scan")
            return False
        nav_timeout = min(
            float(rospy.get_param("~qr_fallback_navigation_timeout_sec", 45.0)),
            remaining)
        # 持续旋转：角度上限按剩余时间换算（事实上只受总时间阈值约束）
        speed = abs(float(rospy.get_param("~qr_scan_angular_speed", 0.80)))
        scan_angle = max(2.0 * math.pi, speed * remaining + 2.0 * math.pi)
        self.publish_status(
            "task1",
            "qr_fallback_navigating",
            "primary scan got {}/{}; moving to fallback point "
            "x={:.3f} y={:.3f} yaw={:.3f}".format(
                self._qr_count(), expected_count, goal_x, goal_y, goal_yaw),
        )
        self.navigate(
            goal_x, goal_y, goal_yaw, "task1",
            timeout_sec=nav_timeout, status_state="qr_fallback_navigating")
        return self.scan_qr_at_current_pose(
            "scanning_qr_fallback",
            total_angle_override=scan_angle,
            extra_sweep_override=0.0,
            deadline_override=qr_total_deadline)

    # ------------------------------ stages ------------------------------
    def task1(self):
        with self.lock:
            if self.voice_handshake_error and not self.voice_command_acknowledged:
                self.wakeup_received = False
                self.voice_prompt_started = False
                self.voice_listening = False
                self.voice_command_ack_in_progress = False
                self.voice_handshake_error = ""
        self.publish_status(
            "task1",
            "waiting_voice",
            "say 小飞小飞, wait for 我在, then give physical/simulation categories",
        )
        self.wait_loop(0, self._voice_command_ready)

        category_name = CATEGORY_LABELS[self.category][0]
        sim_category_name = CATEGORY_LABELS[self.sim_category][0]
        instruction = self.question.strip() or build_task1_instruction(
            self.category, self.sim_category
        )
        self.navigate_to_qr_area()
        # The configured simple_navigator goal has completed, but explicitly
        # revoke all navigation authority before this node can rotate the base.
        self.safe_stop(cancel_navigation=True)

        with self.lock:
            self.qr_items.clear()
            self.task1_instruction = instruction
            self.task1_reasoning_started = False
            self.task1_reasoning_result = None
            self.task1_reasoning_error = ""
            self.task1_reasoning_done.clear()
        try:
            qr_scan_started_at = time.monotonic()
            qr_total_deadline = qr_scan_started_at + float(
                rospy.get_param("~qr_total_timeout_sec", 240.0))
            with self.lock:
                self.qr_decoder_ready = False
                self.qr_decoder_pending_count = 0
                self.qr_decoder_status_at = 0.0
            self.start_child("qr_decoder", "ucar_2026_competition", "qr_decoder.launch")
            self.qr_collecting = True
            completed = self.scan_qr_at_current_pose(
                "scanning_qr_primary", deadline_override=qr_total_deadline)

            expected_count = int(rospy.get_param("~qr_expected_count", 3))
            if not completed or self._qr_count() < expected_count:
                completed = self.scan_qr_at_fallback_point(
                    expected_count, qr_total_deadline)
            if not completed or self._qr_count() < expected_count:
                raise StageError(
                    "single QR scan pass got {}/{} unique item(s)".format(
                        self._qr_count(), expected_count
                    )
                )
            self.publish_status(
                "task1",
                "qr_scan_completed",
                "collected {} unique QR items in {:.3f}s".format(
                    self._qr_count(), time.monotonic() - qr_scan_started_at),
            )
        finally:
            self.qr_collecting = False
            self.stop_child("qr_decoder")
            self.safe_stop(cancel_navigation=True)

        with self.lock:
            items = list(self.qr_items.values())[:3]
        self._start_task1_reasoning_if_ready()
        self.publish_status(
            "task1", "reasoning", "waiting for pipelined Spark X2 result")
        result = self._wait_for_task1_reasoning()
        if not result.success:
            raise StageError("LLM reasoning failed: {}".format(result.error_message))
        if normalize_category(result.pickup_major) != self.category:
            raise StageError("LLM physical category does not match voice category")
        if normalize_category(result.sim_major) != self.sim_category:
            raise StageError("LLM simulation category does not match voice category")
        if normalize_category(result.pickup_major) == normalize_category(result.sim_major):
            raise StageError("LLM returned the same category for physical and simulation")

        self.task1_result = {
            "qr_items": items,
            "category": self.category,
            "category_name": category_name,
            "pickup_category": self.category,
            "pickup_category_name": category_name,
            "sim_category": self.sim_category,
            "sim_category_name": sim_category_name,
            "pickup_item": result.pickup_item,
            "pickup_major": result.pickup_major,
            "pickup_workshop": result.pickup_workshop,
            "sim_item": result.sim_item,
            "sim_major": result.sim_major,
            "sim_workshop": result.sim_workshop,
            "announcement": result.announcement_full,
        }
        self.result_pub.publish(String(data=json.dumps(self.task1_result, ensure_ascii=False)))
        if self.next_stage == "task2":
            self._start_transition_announcement(
                "task1", text=result.announcement_full)
            self.publish_status(
                "task1", "announcement_running",
                "QR result announcement overlaps task2 handoff preparation")
        else:
            self.announce("task1", text=result.announcement_full)
            self.publish_status(
                "task1", "completed", "voice, QR and reasoning completed")

    def _factory_ocr_launch_args(self, category):
        return {
            "start_camera": False,
            "start_competition_speech": False,
            "start_viewer": self.debug,
            "recognition_mode": "ppocr_rknn_system",
            "target_category": category,
            "enable_speech": False,
            "required": True,
        }

    def _factory_search_context(self, category, phase):
        resume_enabled = bool_param("~task2_resume_coverage_enabled", True)
        with self.lock:
            anchor_count = len(rospy.get_param(
                "/vision_triggered_navigator/patrol_points", [])) or 9
            if resume_enabled:
                preferred_anchor, skipped_anchors = task2_resumed_coverage_hint(
                    self.task2_warehouse_memory,
                    category,
                    self.last_coverage_anchor if phase == "simulation" else None,
                    anchor_count,
                )
            else:
                preferred_anchor, skipped_anchors = 0, ()
            remembered = self.task2_warehouse_memory.get(category, {})
            remembered_heading_enabled = bool(
                phase == "simulation" and preferred_anchor and
                int(remembered.get("anchor", 0) or 0) ==
                int(preferred_anchor) and
                remembered.get("odom_yaw") is not None
            )
            remembered_odom_yaw = float(
                remembered.get("odom_yaw", 0.0) or 0.0)
            no_workshop_anchors = normalize_coverage_anchor_ids(
                rospy.get_param("~task2_no_workshop_anchors", []),
                anchor_count,
            )
        if preferred_anchor in no_workshop_anchors:
            preferred_anchor = 0
        skipped_anchors = tuple(sorted(
            set(skipped_anchors).union(no_workshop_anchors)))
        return {
            "resume_enabled": resume_enabled,
            "preferred_anchor": preferred_anchor,
            "skipped_anchors": skipped_anchors,
            "remembered_heading_enabled": remembered_heading_enabled,
            "remembered_odom_yaw": remembered_odom_yaw,
            "no_workshop_anchors": no_workshop_anchors,
        }

    def _factory_navigator_launch_args(
            self, phase, center_only, search_context, start_paused=True):
        preferred_anchor = search_context["preferred_anchor"]
        resume_enabled = search_context["resume_enabled"]
        args = {
            "trigger_mode": "vision",
            "vision_topic": "/vision/detected",
            "target_topic": "/vision/target",
            "non_target_topic": self.vision_non_target_topic,
            "coverage_non_target_early_exit": self.task2_non_target_early_exit,
            "coverage_non_target_min_scan_steps": rospy.get_param(
                "~coverage_non_target_min_scan_steps", 2),
            "trigger_service": self.trigger_service_name,
            "start_paused": bool(start_paused),
            "start_navigation_service": self.factory_navigation_start_service,
            "publish_initial_pose": (
                False if self.mode == "task1_task2" else
                bool_param("~navigator_publish_initial_pose", False)),
            "navigate_to_end_after_trigger": False,
            "coverage_search_mode": True,
            "coverage_start_nearest": (
                resume_enabled and phase == "simulation" and
                not preferred_anchor),
            "coverage_preferred_anchor": (
                preferred_anchor
                if resume_enabled and phase == "simulation" else 0),
            "coverage_preferred_odom_yaw_enabled": search_context[
                "remembered_heading_enabled"],
            "coverage_preferred_odom_yaw": search_context[
                "remembered_odom_yaw"],
            "coverage_preferred_confirm_dwell_sec": rospy.get_param(
                "~task2_remembered_heading_confirm_sec", 1.2),
            "coverage_preferred_scan_half_angle_deg": rospy.get_param(
                "~task2_remembered_heading_scan_half_angle_deg", 18.0),
            "coverage_skip_anchors": ",".join(
                str(value) for value in search_context["skipped_anchors"]),
            "coverage_goal_retry_count": max(0, int(rospy.get_param(
                "~coverage_goal_retry_count", 1))),
            "center_only": center_only,
            "validate_parking_box": not center_only,
            "max_coverage_anchors": int(rospy.get_param(
                "~max_coverage_anchors", 0)),
        }
        forwarded_defaults = {
            "coverage_rotation_min_clearance": 0.28,
            "coverage_translation_min_clearance": 0.00,
            "coverage_translation_sector_half_angle_deg": 35.0,
            "coverage_max_vel_x": 0.72,
            "coverage_max_vel_y": 0.72,
            "coverage_max_vel_theta": 1.45,
            "coverage_cruise_vel_x": 0.70,
            "coverage_cruise_vel_y": 0.70,
            "coverage_cruise_vel_theta": 1.30,
            "coverage_caution_vel_x": 0.53,
            "coverage_caution_vel_y": 0.53,
            "coverage_caution_vel_theta": 1.12,
            "coverage_caution_enter_clearance": 0.45,
            "coverage_caution_exit_clearance": 0.55,
            "coverage_fast_exit_clearance": 0.75,
            "coverage_fast_enter_clearance": 0.90,
            "coverage_speed_update_min_interval_sec": 0.50,
            "target_center_steering_sign": -1.0,
            "camera_boresight_yaw_offset": 0.0,
            "parking_goal_offset": 0.22,
            "parking_staging_offset": 0.45,
            "parking_corner_safe_dist": 0.25,
            "parking_retry_tangent_step": 0.15,
            "parking_staging_max_retries": 2,
            "parking_staging_timeout_sec": 20.0,
            "parking_staging_position_tolerance": 0.10,
            "parking_staging_yaw_tolerance": 0.10,
            "parking_docking_timeout_sec": 30.0,
            "parking_obstacle_min_clearance": 0.24,
            "parking_obstacle_clearance_tolerance": 0.005,
            "parking_obstacle_sector_half_angle_deg": 35.0,
            "parking_dock_max_x": 0.10,
            "parking_dock_max_y": 0.06,
            "parking_dock_max_yaw": 0.30,
            "parking_dock_min_yaw": 0.25,
            "parking_dock_normal_tolerance": 0.03,
            "parking_dock_tangent_tolerance": 0.02,
            "parking_dock_yaw_tolerance": 0.07,
            "parking_min_wall_distance": 0.19,
            "parking_lidar_stop_distance": 0.15,
            "parking_recenter_tolerance": 0.04,
            "parking_recenter_timeout_sec": 8.0,
            "parking_recenter_initial_wait_sec": 1.0,
            "parking_wall_fit_half_angle_deg": 35.0,
            "parking_wall_fit_min_points": 12,
            "parking_wall_fit_min_span": 0.25,
            "parking_wall_fit_near_min_span": 0.18,
            "parking_wall_fit_max_distance_jump": 0.05,
            "parking_wall_fit_max_normal_jump_deg": 8.0,
            "parking_wall_fit_max_residual": 0.015,
            "parking_wall_fit_max_normal_error_deg": 20.0,
            "parking_normal_offset": 0.0,
            "parking_tangent_offset": 0.0,
            "parking_box_width": 0.50,
            "parking_box_depth": 0.50,
            "parking_xy_tolerance": 0.04,
            "parking_yaw_tolerance": 0.06,
            "target_center_coarse_step_deg": 4.0,
            "target_center_fine_step_deg": 2.0,
            "target_center_start_speed": 0.28,
            "target_center_step_max_speed": 0.45,
            "target_center_timeout_sec": 12.0,
            "coverage_scan_step_deg": 20.0,
            "coverage_scan_angular_speed": 0.50,
            "coverage_scan_dwell_sec": 0.45,
            "coverage_candidate_hold_sec": 1.2,
            "coverage_scan_max_dwell_sec": 2.0,
            "coverage_scan_pose_timeout_sec": 0.5,
            "coverage_goal_soft_timeout_sec": 7.0,
            "coverage_goal_hard_timeout_sec": 10.0,
            "coverage_goal_progress_window_sec": 5.0,
            "coverage_goal_min_progress": 0.03,
            "coverage_anchor_observation_radius": 0.35,
            "coverage_near_anchor_stall_timeout_sec": 3.0,
        }
        for name, default in forwarded_defaults.items():
            param_name = "target_center_max_speed" if (
                name == "target_center_step_max_speed") else name
            args[name] = rospy.get_param("~" + param_name, default)
        args["vision_offset"] = rospy.get_param("~task2_vision_offset", 0.4)
        return args

    def _child_is_running(self, key):
        proc = self.children.get(key)
        return proc is not None and proc.poll() is None

    def _task2_prewarm_is_reusable(self, category, phase):
        return task2_prewarm_reusable(
            self.task2_prewarm_enabled,
            phase,
            category,
            self.task2_prewarm_category,
            self._child_is_running("factory_ocr"),
            self._child_is_running("factory_navigator"),
        )

    def _prewarm_task2(self):
        if (not self.task2_prewarm_enabled or self.next_stage != "task2" or
                not self.category):
            return False
        category = self.category
        try:
            context = self._factory_search_context(category, "physical")
            # Keep OCR logically disarmed until task2 sets its target.
            self.ocr_target = None
            self.start_child(
                "factory_ocr",
                "factory_sign_ppocr_rknn_test",
                "factory_sign_ppocr_rknn_test.launch",
                self._factory_ocr_launch_args(category),
            )
            self.start_child(
                "factory_navigator",
                "vision_triggered_navigator",
                "vision_triggered_navigator.launch",
                self._factory_navigator_launch_args(
                    "physical", False, context, start_paused=True),
            )
            self.task2_prewarm_category = category
            self.publish_status(
                "task1", "task2_prewarming",
                "OCR loading; navigator alive but movement remains paused")
            rospy.loginfo(
                "Task2 prewarm started for category=%s; OCR target is disabled "
                "and navigator is start_paused.", category)
            return True
        except Exception as exc:
            self.stop_child("factory_ocr")
            self.stop_child("factory_navigator")
            self.task2_prewarm_category = None
            rospy.logwarn(
                "Task2 prewarm failed; task1 continues and task2 will start "
                "normally: %s", exc)
            return False

    def _start_factory_children(self, category, phase, center_only, context):
        reused = self._task2_prewarm_is_reusable(category, phase)
        if reused:
            rospy.loginfo(
                "Task2 reusing prewarmed OCR and paused navigator for %s.",
                category)
        else:
            self.stop_child("factory_ocr")
            self.stop_child("factory_navigator")
            self.start_child(
                "factory_ocr",
                "factory_sign_ppocr_rknn_test",
                "factory_sign_ppocr_rknn_test.launch",
                self._factory_ocr_launch_args(category),
            )
            self.start_child(
                "factory_navigator",
                "vision_triggered_navigator",
                "vision_triggered_navigator.launch",
                self._factory_navigator_launch_args(
                    phase, center_only, context, start_paused=True),
            )
        self.task2_prewarm_category = None
        return reused

    def _release_factory_navigation(self):
        try:
            rospy.wait_for_service(
                self.factory_navigation_start_service, timeout=5.0)
            response = rospy.ServiceProxy(
                self.factory_navigation_start_service, Trigger)()
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError(
                "factory navigation start service failed: {}".format(exc))
        if not response.success:
            raise StageError(
                "factory navigation start was rejected: {}".format(
                    response.message))
        rospy.loginfo(
            "Task2 navigation released after fresh OCR frame: %s",
            response.message)

    def _navigate_factory_target(self, category, item, workshop, phase, announce):
        if not category or not item or not workshop:
            raise StageError("task2 {} target is incomplete".format(phase))
        center_only = bool_param("~task2_center_only", False)

        self.ocr_target = category
        self.ocr_filter.reset()
        self.ocr_last_message_at = 0.0
        self.vision_trigger_latched = False
        self.trigger_request_pending = False
        self.trigger_request_started_at = 0.0
        self.trigger_service_accepted = False
        self.trigger_acknowledged = False
        self.navigator_status = ""
        with self.lock:
            self.current_coverage_anchor = None
            self.task2_non_target_announced = set()
            for memory_filter in self.ocr_memory_filters.values():
                memory_filter.reset()
        search_context = self._factory_search_context(category, phase)
        resume_coverage_enabled = search_context["resume_enabled"]
        preferred_anchor = search_context["preferred_anchor"]
        skipped_anchors = search_context["skipped_anchors"]
        remembered_heading_enabled = search_context[
            "remembered_heading_enabled"]
        remembered_odom_yaw = search_context["remembered_odom_yaw"]
        no_workshop_anchors = search_context["no_workshop_anchors"]
        if no_workshop_anchors:
            rospy.loginfo(
                "task2 calibrated no-workshop anchors skipped: %s",
                ",".join(str(value) for value in no_workshop_anchors))
        if phase == "simulation" and preferred_anchor:
            rospy.loginfo(
                "task2 coverage resume: starting %s search at anchor %d; "
                "skipping irrelevant anchors=%s remembered_heading=%s "
                "odom_yaw=%.3f",
                category, preferred_anchor,
                ",".join(str(value) for value in skipped_anchors) or "none",
                remembered_heading_enabled, remembered_odom_yaw)
        self.publish_status(
            "task2", "searching_{}".format(phase),
            "searching {} factory sign with existing 9-point navigation".format(category))
        try:
            self._start_factory_children(
                category, phase, center_only, search_context)
            # Ignore any callback racing with a mismatched-process teardown;
            # readiness must come from the adopted/restarted OCR process.
            self.ocr_filter.reset()
            self.ocr_last_message_at = 0.0
            self.publish_status("task2", "waiting_ocr", "waiting for first OCR result before motion")
            ocr_ready_deadline = time.time() + float(
                rospy.get_param("~ocr_ready_timeout_sec", 12.0))
            while time.time() < ocr_ready_deadline and not self.ocr_last_message_at:
                self.check_abort()
                for key in ("factory_ocr", "factory_navigator"):
                    proc = self.children.get(key)
                    if proc is None or proc.poll() is not None:
                        code = None if proc is None else proc.returncode
                        raise StageError(
                            "{} exited before OCR ready with code {}".format(
                                key, code))
                rospy.sleep(0.1)
            if not self.ocr_last_message_at:
                raise StageError("factory OCR produced no result before motion timeout")
            self._release_factory_navigation()
            timeout = float(rospy.get_param("~factory_navigation_timeout_sec", 420.0))
            deadline = time.time() + timeout
            while time.time() < deadline:
                self.check_abort()
                self._deliver_target_trigger()
                if self.navigator_status == "arrived":
                    break
                if center_only and self.navigator_status == "centered":
                    break
                if self.navigator_status == "failed":
                    raise StageError("factory navigation failed")
                if self.navigator_status in (
                        "centering_failed", "parking_staging_failed",
                        "parking_recenter_failed", "parking_wall_fit_failed",
                        "parking_docking_failed", "parking_validation_failed",
                        "coverage_recovery_disable_failed"):
                    raise StageError("factory navigation {}".format(
                        self.navigator_status))
                for key in ("factory_navigator", "factory_ocr"):
                    proc = self.children.get(key)
                    if proc and proc.poll() is not None:
                        raise StageError("{} exited unexpectedly with code {}".format(key, proc.returncode))
                rospy.sleep(0.1)
            else:
                raise StageError("factory navigation timed out after {:.1f}s".format(timeout))
        finally:
            self.ocr_target = None
            self.vision_trigger_latched = False
            self.trigger_request_pending = False
            self.trigger_service_accepted = False
            self.trigger_acknowledged = False
            self.stop_child("factory_ocr")
            self.stop_child("factory_navigator")
            self.safe_stop(cancel_navigation=True)
        if center_only:
            self.publish_status("task2", "center_test_completed", "target centering test completed")
            return
        self.safe_stop(cancel_navigation=True)
        announcement_required = announce and task2_announcement_required(
            self.navigator_status, self.task2_announcement_completed)
        if announce and not self.task2_announcement_completed and not announcement_required:
            raise StageError(
                "refusing task2 announcement before confirmed arrived state")
        if announcement_required:
            self.publish_status(
                "task2", "announcing",
                "announcing completed physical warehouse delivery")
            self.announce("task2", item=item, workshop=workshop)
            self.task2_announcement_completed = True
            self.publish_status(
                "task2", "announcement_completed",
                "task2 announcement service completed")
        self.publish_status(
            "task2", "{}_factory_reached".format(phase),
            "{} target factory reached".format(phase))

    def task2(self):
        if not self.category or not self.sim_category:
            raise StageError("task2 physical/simulation categories are missing")
        pickup_item = self.task1_result.get("pickup_item")
        pickup_workshop = (
            self.task1_result.get("pickup_workshop")
            or CATEGORY_LABELS[self.category][1]
        )
        sim_item = self.task1_result.get("sim_item")
        if not sim_item and self.sim_category == self.category:
            sim_item = pickup_item
        sim_workshop = (
            self.task1_result.get("sim_workshop")
            or CATEGORY_LABELS[self.sim_category][1]
        )
        visits = task2_delivery_targets(
            (self.category, pickup_item, pickup_workshop),
            (self.sim_category, sim_item, sim_workshop),
        )
        with self.lock:
            self.task2_warehouse_memory = {}
            self.current_coverage_anchor = None
            self.last_coverage_anchor = None
            for memory_filter in self.ocr_memory_filters.values():
                memory_filter.reset()
        self.task2_announcement_completed = False
        for visit_index, (phase, category, item, workshop) in enumerate(visits):
            if visit_index > 0:
                self.task2_inter_visit_handoff()
            self._navigate_factory_target(
                category, item, workshop, phase, announce=(phase == "physical")
            )
        if len(visits) == 1:
            self.publish_status(
                "task2", "simulation_factory_already_reached",
                "physical and simulation targets share the same workshop")
        self.publish_status(
            "task2", "completed",
            "physical delivery announced; vehicle parked at simulation workshop")

    def task3(self):
        if not self.sim_category:
            raise StageError("task3 sim_target_category is missing")
        host = rospy.get_param("~sim_bridge_host", "").strip()
        port = int(rospy.get_param("~sim_bridge_port", 26003))
        if not host:
            raise StageError("SIM_BRIDGE_HOST / sim_bridge_host is missing")
        timeout = float(rospy.get_param("~sim_timeout_sec", 900.0))
        self.publish_status("task3", "connecting", "connecting to {}:{}".format(host, port))
        try:
            sock = socket.create_connection((host, port), timeout=10.0)
            sock.settimeout(1.0)
        except OSError as exc:
            raise StageError("simulation bridge connection failed: {}".format(exc))
        request_id = str(uuid.uuid4())
        deadline = time.time() + timeout
        result_text = ""
        done_received = False
        done_received_at = 0.0
        try:
            request = {
                "command": "start",
                "target": self.sim_category,
                "request_id": request_id,
            }
            sock.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
            self.publish_status("task3", "running", "simulation task started")
            decoder = JsonLineBuffer()
            while time.time() < deadline:
                self.check_abort()
                if done_received and time.time() - done_received_at > 3.0:
                    raise StageError("simulation reported done without a success result")
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                except OSError as exc:
                    raise StageError("simulation bridge disconnected: {}".format(exc))
                if not chunk:
                    raise StageError("simulation bridge disconnected")
                for event in decoder.feed(chunk):
                    event_type = event.get("type")
                    value = event.get("data")
                    if event_type == "state":
                        state_text = str(value or "")
                        self.publish_status("task3", "running", state_text)
                        if state_text.startswith("FAILED:"):
                            raise StageError(state_text)
                    elif event_type == "result":
                        result_text = str(value or "")
                        if result_text.startswith("FAILED:"):
                            raise StageError(result_text)
                    elif event_type == "done" and bool(value):
                        done_received = True
                        done_received_at = time.time()
                    elif event_type == "error":
                        raise StageError(str(value or "simulation bridge error"))
                if done_received and result_text.startswith("SUCCESS:"):
                    break
            else:
                raise StageError("simulation task timed out")
        finally:
            try:
                sock.close()
            except Exception:
                pass
        if not result_text.startswith("SUCCESS:"):
            raise StageError("simulation completed without a success result")
        item = self.task1_result.get("sim_item") or self.task1_result.get("pickup_item")
        workshop = (
            self.task1_result.get("sim_workshop")
            or CATEGORY_LABELS[self.sim_category][1]
        )
        if not item:
            raise StageError("task3 sim_item is missing")
        self.publish_status(
            "task3", "announcement_running",
            "{}; vehicle locked until announcement completes".format(result_text))
        self._announce_while_stationary(
            "task3", item=item, workshop=workshop)
        self.publish_status(
            "task3", "completed",
            "{}; announcement completed with vehicle stationary".format(
                result_text))

    def approach_task4_stop_line(self):
        self.strict_mission_status = {}
        staging_pose, migrated = normalize_task4_staging_pose(
            rospy.get_param("~traffic_x"),
            rospy.get_param("~traffic_y"),
            rospy.get_param("~traffic_yaw"),
        )
        staging_x, staging_y, staging_yaw = staging_pose
        if migrated:
            rospy.logwarn(
                "task4 retired staging pose requested; auto-correcting "
                "x=0.3195 y=-3.00 to x=0.2395 y=-3.10")
        rospy.loginfo(
            "task4 staging pose in use: x=%.4f y=%.4f yaw=%.4f migrated=%s",
            staging_x, staging_y, staging_yaw, migrated)
        self.publish_status(
            "task4", "approaching_stop_line",
            "staging x={:.4f} y={:.4f} yaw={:.4f}; then visual approach".format(
                staging_x, staging_y, staging_yaw))
        self.start_child(
            "strict_line",
            "ucar_2026_strict_mission",
            "strict_mission.launch",
            {
                "start_traffic_detector": False,
                "start_viewer": self.debug,
                "traffic_pose_configured": True,
                "traffic_staging_x": staging_x,
                "traffic_staging_y": staging_y,
                "traffic_staging_yaw": staging_yaw,
            },
        )
        try:
            rospy.wait_for_service("/strict_mission/start", timeout=10.0)
            response = rospy.ServiceProxy("/strict_mission/start", Trigger)()
            if not response.success:
                raise StageError(
                    "strict stop-line approach refused start: {}".format(response.message))

            timeout = (
                float(rospy.get_param("~move_base_timeout_sec", 90.0))
                + float(rospy.get_param("~line_approach_timeout_sec", 45.0))
                + 15.0
            )
            deadline = time.time() + timeout
            last_state = None
            while time.time() < deadline:
                self.check_abort()
                status = self.strict_mission_status
                state = str(status.get("state", ""))
                if state and state != last_state:
                    rospy.loginfo(
                        "task4 stop-line state: %s distance_m=%s detail=%s",
                        state, status.get("distance_m"), status.get("detail"))
                    last_state = state
                if state == "WAIT_TRAFFIC":
                    distance = status.get("visual_stop_distance_m")
                    final_distance = status.get("final_stop_distance_m")
                    final_color = str(
                        status.get("final_stop_line_color") or "").strip().lower()
                    final_verified = bool(
                        status.get("final_visual_verified", False))
                    final_stop_source = str(
                        status.get("final_stop_source") or "").strip().lower()
                    planned = float(status.get("final_advance_m") or 0.0)
                    progress = float(status.get("final_progress_m") or 0.0)
                    source = str(
                        status.get("final_advance_source") or "").strip()
                    valid_sources = (
                        "visual_distance",
                        "visual_hold",
                        "no_vision_fallback",
                    )
                    tolerance = float(rospy.get_param(
                        "~task4_final_progress_tolerance_m", 0.008))
                    if source not in valid_sources:
                        raise StageError(
                            "task4 final advance source not verified: {}".format(
                                source or "missing"))
                    if not final_advance_completed(
                            planned, progress, tolerance):
                        raise StageError(
                            "task4 final advance incomplete: "
                            "planned={:.3f}m progress={:.3f}m "
                            "tolerance={:.3f}m".format(
                                planned, progress, tolerance))
                    final_min = float(rospy.get_param(
                        "~task4_final_stop_min_m", 0.03))
                    final_max = float(rospy.get_param(
                        "~task4_final_stop_max_m", 0.05))
                    try:
                        final_distance_value = float(final_distance)
                    except (TypeError, ValueError):
                        final_distance_value = None
                    visual_stop_valid = (
                        final_stop_source == "yellow_visual"
                        and final_verified
                        and final_color == "yellow"
                        and final_distance_value is not None
                        and final_min <= final_distance_value <= final_max
                    )
                    hard_advance_fallback = (
                        final_stop_source == "hard_advance_timeout"
                        and not final_verified
                    )
                    if not (visual_stop_valid or hard_advance_fallback):
                        raise StageError(
                            "task4 final stop not accepted: source={} "
                            "verified={} color={} distance={} "
                            "visual_required=[{:.3f},{:.3f}]m".format(
                                final_stop_source or "missing",
                                final_verified, final_color or "missing",
                                "missing" if final_distance_value is None
                                else "{:.3f}m".format(final_distance_value),
                                final_min, final_max))
                    self.publish_status(
                        "task4", "stop_line_reached",
                        "vehicle held before stop line; visual_distance_m={} "
                        "final_advance_source={} final_progress_m={:.3f} "
                        "final_stop_source={} final_yellow_distance_m={}".format(
                            distance, source, progress, final_stop_source,
                            "missing" if final_distance_value is None
                            else "{:.3f}".format(final_distance_value)))
                    return
                if state == "FAULT":
                    raise StageError(
                        "strict stop-line approach failed: {}".format(
                            status.get("error") or status.get("detail") or "unknown fault"))
                proc = self.children.get("strict_line")
                if proc and proc.poll() is not None:
                    raise StageError("strict stop-line approach exited unexpectedly")
                rospy.sleep(0.1)
            raise StageError(
                "strict stop-line approach timed out after {:.1f}s".format(timeout))
        finally:
            self.stop_child("strict_line")
            self.safe_stop(cancel_navigation=True)

    def task4(self):
        skip_approach = bool_param("~skip_task4_stop_line_approach", False)
        configured = bool_param("~traffic_pose_configured", False)
        try:
            start_action = task4_start_action(skip_approach, configured)
        except ValueError as exc:
            raise StageError(str(exc))
        if start_action == "approach":
            self.approach_task4_stop_line()
        else:
            self.safe_stop(cancel_navigation=True)
            self.publish_status(
                "task4", "stop_line_ready",
                "using manually positioned stop-line start; vehicle held stopped")
        self.traffic_decision = ""
        self.red_announced = False
        self.publish_status("task4", "detecting", "waiting for traffic-light consensus")
        try:
            self.start_child(
                "traffic_light",
                "ucar_2026_traffic_light_rknn_test",
                "traffic_light_rknn_x11_speak_test.launch",
                {
                    "start_camera": False,
                    "start_tts": False,
                    "start_competition_speech": False,
                    "start_viewer": self.debug,
                    "enable_speech": False,
                    "required": True,
                },
            )
            deadline = time.time() + float(rospy.get_param("~traffic_timeout_sec", 180.0))
            while time.time() < deadline:
                self.check_abort()
                if self.traffic_decision == "stop":
                    self.safe_stop()
                    if not self.red_announced:
                        self.announce("task4", decision="stop")
                        self.red_announced = True
                        self.publish_status("task4", "red_wait", "red light: holding stop")
                    self.traffic_decision = ""
                elif self.traffic_decision in ("left", "right", "straight"):
                    decision = self.traffic_decision
                    announcement = self._start_announcement(
                        "task4", decision=decision)
                    self.stop_child("traffic_light")
                    self._wait_announcement(announcement)
                    self.traffic_pub.publish(String(data=decision))
                    self.publish_status(
                        "task4", "completed",
                        "decision={}; detector stopped during speech; task5 may start".format(
                            decision))
                    self.traffic_decision = decision
                    return
                proc = self.children.get("traffic_light")
                if proc and proc.poll() is not None:
                    raise StageError("traffic-light detector exited unexpectedly")
                rospy.sleep(0.1)
            raise StageError("traffic-light recognition timed out")
        finally:
            self.stop_child("traffic_light")
            self.safe_stop(cancel_navigation=True)

    def task5(self):
        decision = self.traffic_decision or rospy.get_param("~traffic_decision", "").strip().lower()
        if decision not in TRACK_CONFIG:
            raise StageError("task5 traffic_decision must be left/right/straight")
        launch_file, status_topic, finish_value = TRACK_CONFIG[decision]
        self.safe_stop(cancel_navigation=True)
        self.track_status[status_topic] = ""
        self.publish_status("task5", "line_following", "launching {}".format(launch_file))
        try:
            self.start_child(
                "line_follow",
                "ucar_2026_track_end_stop",
                launch_file,
                {"start_driver": False, "start_camera": False, "start_viewer": self.debug},
            )
            timeout = float(rospy.get_param("~track_timeout_sec", 420.0))
            self.wait_loop(
                timeout,
                lambda: self.track_status.get(status_topic) == finish_value,
                child_key="line_follow",
            )
        finally:
            self.stop_child("line_follow")
            self.safe_stop(cancel_navigation=True)
        self.announce("task5")
        self.publish_status("task5", "completed", "competition completed")

    def run(self):
        try:
            handlers = {
                "task1": self.task1,
                "task2": self.task2,
                "task3": self.task3,
                "task4": self.task4,
                "task5": self.task5,
            }
            previous_stage = None
            stages = stage_sequence(self.mode, self.enable_simulation)
            for index, stage in enumerate(stages):
                self.next_stage = (
                    stages[index + 1] if index + 1 < len(stages) else None)
                if previous_stage == "task1" and stage == "task2":
                    self.run_stage("task1", self.task1_task2_handoff)
                if task4_handoff_required(previous_stage, stage):
                    source_stage = previous_stage
                    self.run_stage(
                        source_stage,
                        lambda: self.production_task4_handoff(source_stage),
                    )
                self.run_stage(stage, handlers[stage])
                previous_stage = stage
            self.publish_status("competition", "completed", "requested flow completed")
        except Aborted as exc:
            self.publish_status("competition", "aborted", error=str(exc))
        except Exception as exc:
            rospy.logerr("unhandled competition error: %s", exc)
            self.safe_stop(cancel_navigation=True)
            self.publish_status("competition", "failed", error=str(exc))
        finally:
            self.shutdown()

    def shutdown(self):
        self.stop_all_children()
        try:
            self.safe_stop(cancel_navigation=True)
        except Exception:
            pass


def main():
    rospy.init_node("competition_flow")
    CompetitionFlow().run()


if __name__ == "__main__":
    main()
