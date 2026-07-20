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
    TemporalTargetFilter,
    DirectedYawAccumulator,
    JsonLineBuffer,
    TRACK_CONFIG,
    normalize_category,
    parse_category,
    qr_values_from_payload,
    stage_sequence,
    task4_handoff_required,
    task4_start_action,
    traffic_decision_from_payload,
    task2_announcement_required,
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

        self.wakeup_received = False
        self.voice_prompt_started = False
        self.voice_listening = False
        self.voice_command_acknowledged = False
        self.voice_command_ack_in_progress = False
        self.voice_handshake_error = ""
        self.voice_wakeup_generation = 0
        self.question = ""
        self.category = normalize_category(rospy.get_param("~target_category", ""))
        self.task1_result = {
            "pickup_item": rospy.get_param("~target_item", "").strip(),
            "pickup_workshop": rospy.get_param("~target_workshop", "").strip(),
            "sim_item": rospy.get_param("~sim_item", "").strip(),
            "sim_workshop": rospy.get_param("~sim_workshop", "").strip(),
        }

        self.qr_items = OrderedDict()
        self.qr_collecting = False
        self.qr_navigation_watching = False
        self.qr_navigation_goal_id = ""
        self.qr_navigation_result = None
        self.qr_odom_yaw = None
        self.qr_odom_received_at = 0.0
        self.base_twist = None
        self.handoff_scan_received_at = 0.0
        self.handoff_costmap_received_at = 0.0
        self.ocr_target = None
        self.ocr_last_message_at = 0.0
        self.ocr_filter = TemporalTargetFilter(
            rospy.get_param("~ocr_required_hits", 2),
            rospy.get_param("~ocr_evidence_window_sec", 1.5),
        )
        self.vision_trigger_latched = False
        self.trigger_request_pending = False
        self.trigger_request_started_at = 0.0
        self.trigger_service_accepted = False
        self.trigger_acknowledged = False
        self.trigger_service_name = rospy.get_param(
            "~target_trigger_service", "/vision_triggered_navigator/trigger_target")
        self.trigger_ack_timeout = float(rospy.get_param("~trigger_ack_timeout_sec", 2.0))
        self.navigator_status = ""
        self.task2_announcement_completed = False
        self.traffic_decision = rospy.get_param("~traffic_decision", "").strip().lower()
        self.red_announced = False
        self.strict_mission_status = {}
        self.track_status = {}

        rospy.Subscriber("/wakeup", String, self._wakeup_cb, queue_size=5)
        rospy.Subscriber("/question", String, self._question_cb, queue_size=5)
        rospy.Subscriber("/qr_code_data", String, self._qr_cb, queue_size=20)
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
        parsed = parse_category(question)
        with self.lock:
            if not self.voice_listening or self.voice_command_ack_in_progress:
                rospy.logwarn("ignoring /question outside active voice window: %s", question)
                return
            if not parsed:
                rospy.logwarn("ignoring voice text without a target category: %s", question)
                self.publish_status(
                    "task1", "listening_command", "ignored non-command speech: {}".format(question)
                )
                return
            self.question = question
            self.voice_listening = False
            self.voice_command_ack_in_progress = True
        threading.Thread(
            target=self._finish_voice_command,
            args=(parsed, question),
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
                    "task1", "listening_command", "waiting for 取得食品/日用品/电子产品"
                )
        except Exception as exc:
            self._set_voice_handshake_error(exc)

    def _finish_voice_command(self, parsed, question):
        try:
            with self.voice_transition_lock:
                self._stop_voice_listening()
                reply = rospy.get_param("~voice_command_reply", "好的").strip() or "好的"
                self.publish_status(
                    "task1", "command_ack", "category={} reply={}".format(parsed, reply)
                )
                self.announce("custom", text=reply)
                with self.lock:
                    self.category = parsed
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
            return self.wakeup_received and self.voice_command_acknowledged and self.category

    def _qr_cb(self, msg):
        if not self.qr_collecting:
            return
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        for key, result in qr_values_from_payload(payload):
            with self.lock:
                if key not in self.qr_items:
                    self.qr_items[key] = result
                    rospy.loginfo("QR accepted %d/3: %s", len(self.qr_items), result)

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

    def _handoff_scan_cb(self, _msg):
        with self.lock:
            self.handoff_scan_received_at = time.monotonic()

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
        self.ocr_last_message_at = time.monotonic()
        if category == self.ocr_target and payload.get("target_bbox"):
            self.vision_target_pub.publish(msg)
        confirmed = self.ocr_filter.push(
            self.ocr_target, category, time.monotonic())
        rospy.loginfo_throttle(
            0.5,
            "task2 OCR filter: target=%s observed=%s hits=%d/%d bbox=%s",
            self.ocr_target,
            category or "none",
            self.ocr_filter.hit_count,
            self.ocr_filter.required,
            bool(payload.get("target_bbox")),
        )
        if (confirmed and category == self.ocr_target and payload.get("target_bbox") and
                not self.vision_trigger_latched):
            self.vision_trigger_latched = True
            self.trigger_request_pending = True
            self.trigger_request_started_at = time.monotonic()
            self.trigger_service_accepted = False
            self.trigger_acknowledged = False
            self.publish_status(
                "task2", "trigger_pending",
                "OCR target confirmed; requesting navigator acknowledgement")
            rospy.loginfo(
                "task2 OCR target confirmed: target=%s hits=%d/%d; "
                "reliable trigger pending (will not retrigger)",
                self.ocr_target,
                self.ocr_filter.hit_count,
                self.ocr_filter.required,
            )

    def _navigator_cb(self, msg):
        status = msg.data.strip().lower()
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
        self.safe_stop(cancel_navigation=True)
        self.publish_status(
            "task1", "task2_handoff_ready",
            "move_base idle; fresh costmap; preserving AMCL state")

    def production_task4_handoff(self, source_stage):
        """Resume physical navigation without resetting the current factory pose."""
        source_stage = str(source_stage or "task3").strip().lower()
        self.publish_status(
            source_stage, "task4_handoff",
            "preserving AMCL pose and preparing physical navigation")
        self.safe_stop(cancel_navigation=True)
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
        self.safe_stop(cancel_navigation=True)
        self.publish_status(
            source_stage, "task4_handoff_ready",
            "current AMCL pose preserved; fresh costmap; task4 may navigate")

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

    def _settle_for_qr(self, duration, expected_count, scan_deadline, stale_sec):
        self.safe_stop()
        settle_deadline = min(scan_deadline, time.monotonic() + duration)
        while time.monotonic() < settle_deadline and not rospy.is_shutdown():
            self.check_abort()
            self._check_qr_decoder()
            self._fresh_qr_odom_yaw(stale_sec)
            if self._qr_count() >= expected_count:
                return True
            rospy.sleep(0.05)
        return self._qr_count() >= expected_count

    def scan_qr_at_current_pose(self, status_state):
        """Rotate one odometry-closed-loop revolution with stable decode pauses."""
        expected_count = int(rospy.get_param("~qr_expected_count", 3))
        speed = abs(float(rospy.get_param("~qr_scan_angular_speed", 0.20)))
        step_angle = abs(
            float(rospy.get_param("~qr_scan_step_angle_rad", math.radians(20.0)))
        )
        settle_sec = max(0.0, float(rospy.get_param("~qr_scan_settle_sec", 0.6)))
        scan_timeout = float(rospy.get_param("~qr_scan_timeout_sec", 60.0))
        stale_sec = float(rospy.get_param("~qr_odom_stale_sec", 0.5))
        odom_wait_sec = float(rospy.get_param("~qr_odom_wait_sec", 2.0))
        step_margin = float(
            rospy.get_param("~qr_scan_step_timeout_margin_sec", 2.0)
        )
        if speed <= 0.0 or step_angle <= 0.0 or stale_sec <= 0.0:
            raise StageError("QR scan motion parameters must be positive")

        total_steps = int(math.ceil((2.0 * math.pi) / step_angle))
        scan_deadline = time.monotonic() + scan_timeout
        self.publish_status(
            "task1",
            status_state,
            "step scan start: count={}/{} steps={}".format(
                self._qr_count(), expected_count, total_steps
            ),
        )
        self._wait_for_qr_odom(odom_wait_sec, stale_sec)
        if self._settle_for_qr(settle_sec, expected_count, scan_deadline, stale_sec):
            return True

        twist = Twist()
        twist.angular.z = speed
        tracker = DirectedYawAccumulator(direction=1.0)
        for _ in range(total_steps):
            if self._qr_count() >= expected_count:
                self.safe_stop()
                return True
            if time.monotonic() >= scan_deadline:
                self.safe_stop()
                return False

            tracker.reset(self._fresh_qr_odom_yaw(stale_sec))
            step_deadline = time.monotonic() + step_angle / speed + step_margin
            while tracker.progress < step_angle and not rospy.is_shutdown():
                self.check_abort()
                self._check_qr_decoder()
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

            if self._settle_for_qr(
                settle_sec, expected_count, scan_deadline, stale_sec
            ):
                return True

        self.safe_stop()
        return self._qr_count() >= expected_count

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
            "say 小飞小飞, wait for 我在, then say 取得食品/日用品/电子产品",
        )
        self.wait_loop(0, self._voice_command_ready)

        category_name = CATEGORY_LABELS[self.category][0]
        instruction = "请取得{}类产品，仿真环境也统一取得{}类产品".format(category_name, category_name)
        self.navigate_to_qr_area()
        # The configured simple_navigator goal has completed, but explicitly
        # revoke all navigation authority before this node can rotate the base.
        self.safe_stop(cancel_navigation=True)

        with self.lock:
            self.qr_items.clear()
        try:
            self.start_child("qr_decoder", "ucar_2026_competition", "qr_decoder.launch")
            self.qr_collecting = True
            completed = self.scan_qr_at_current_pose("scanning_qr_primary")

            if not completed and bool_param("~qr_fallback_enabled", True):
                self.qr_collecting = False
                self.safe_stop(cancel_navigation=True)
                fallback = rospy.get_param(
                    "~qr_fallback_goal",
                    {"x": -1.4643, "y": -0.1390, "yaw": 1.5834},
                )
                self.navigate(
                    fallback["x"],
                    fallback["y"],
                    fallback.get("yaw", 1.5834),
                    "task1",
                    timeout_sec=float(
                        rospy.get_param("~qr_fallback_navigation_timeout_sec", 45.0)
                    ),
                    status_state="qr_repositioning",
                )
                self.safe_stop(cancel_navigation=True)
                self.qr_collecting = True
                completed = self.scan_qr_at_current_pose("scanning_qr_fallback")

            expected_count = int(rospy.get_param("~qr_expected_count", 3))
            if not completed or self._qr_count() < expected_count:
                raise StageError(
                    "QR scan exhausted two poses: got {}/{} unique item(s)".format(
                        self._qr_count(), expected_count
                    )
                )
            self.publish_status(
                "task1",
                "qr_scan_completed",
                "collected {} unique QR items".format(self._qr_count()),
            )
        finally:
            self.qr_collecting = False
            self.stop_child("qr_decoder")
            self.safe_stop(cancel_navigation=True)

        with self.lock:
            items = list(self.qr_items.values())[:3]
        service = rospy.get_param("~llm_service", "/smart_factory_llm/reason_pickup_order")
        self.publish_status("task1", "reasoning", "calling Spark X2")
        try:
            rospy.wait_for_service(service, timeout=15.0)
            result = rospy.ServiceProxy(service, ReasonPickupOrder)(
                items[0], items[1], items[2], instruction
            )
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError("LLM service failed: {}".format(exc))
        if not result.success:
            raise StageError("LLM reasoning failed: {}".format(result.error_message))
        if normalize_category(result.pickup_major) != self.category:
            raise StageError("LLM physical category does not match voice category")
        if normalize_category(result.sim_major) != self.category:
            raise StageError("LLM simulation category does not match voice category")

        self.task1_result = {
            "qr_items": items,
            "category": self.category,
            "category_name": category_name,
            "pickup_item": result.pickup_item,
            "pickup_major": result.pickup_major,
            "pickup_workshop": result.pickup_workshop,
            "sim_item": result.sim_item,
            "sim_major": result.sim_major,
            "sim_workshop": result.sim_workshop,
            "announcement": result.announcement_full,
        }
        self.result_pub.publish(String(data=json.dumps(self.task1_result, ensure_ascii=False)))
        self.announce("task1", text=result.announcement_full)
        self.publish_status("task1", "completed", "voice, QR and reasoning completed")

    def task2(self):
        if not self.category:
            raise StageError("task2 target_category is missing")
        item = self.task1_result.get("pickup_item")
        workshop = self.task1_result.get("pickup_workshop") or CATEGORY_LABELS[self.category][1]
        if not item:
            raise StageError("task2 target_item is missing")
        center_only = bool_param("~task2_center_only", False)

        self.ocr_target = self.category
        self.ocr_filter.reset()
        self.ocr_last_message_at = 0.0
        self.vision_trigger_latched = False
        self.trigger_request_pending = False
        self.trigger_request_started_at = 0.0
        self.trigger_service_accepted = False
        self.trigger_acknowledged = False
        self.navigator_status = ""
        self.publish_status("task2", "searching", "searching target factory sign with existing 9-point navigation")
        try:
            self.start_child(
                "factory_ocr",
                "factory_sign_ppocr_rknn_test",
                "factory_sign_ppocr_rknn_test.launch",
                {
                    "start_camera": False,
                    "start_competition_speech": False,
                    "start_viewer": self.debug,
                    "recognition_mode": "ppocr_rknn_system",
                    "target_category": self.category,
                    "enable_speech": False,
                    "required": True,
                },
            )
            self.publish_status("task2", "waiting_ocr", "waiting for first OCR result before motion")
            ocr_ready_deadline = time.time() + float(
                rospy.get_param("~ocr_ready_timeout_sec", 12.0))
            while time.time() < ocr_ready_deadline and not self.ocr_last_message_at:
                self.check_abort()
                proc = self.children.get("factory_ocr")
                if proc and proc.poll() is not None:
                    raise StageError(
                        "factory_ocr exited before ready with code {}".format(proc.returncode))
                rospy.sleep(0.1)
            if not self.ocr_last_message_at:
                raise StageError("factory OCR produced no result before motion timeout")
            self.start_child(
                "factory_navigator",
                "vision_triggered_navigator",
                "vision_triggered_navigator.launch",
                {
                    "trigger_mode": "vision",
                    "vision_topic": "/vision/detected",
                    "target_topic": "/vision/target",
                    "trigger_service": self.trigger_service_name,
                    "publish_initial_pose": (
                        False if self.mode == "task1_task2" else
                        bool_param("~navigator_publish_initial_pose", False)),
                    "navigate_to_end_after_trigger": False,
                    "coverage_search_mode": True,
                    "target_center_steering_sign": rospy.get_param(
                        "~target_center_steering_sign", -1.0),
                    "camera_boresight_yaw_offset": rospy.get_param(
                        "~camera_boresight_yaw_offset", 0.0),
                    "center_only": center_only,
                    "validate_parking_box": not center_only,
                    "max_coverage_anchors": int(rospy.get_param(
                        "~max_coverage_anchors", 0)),
                    "vision_offset": rospy.get_param("~task2_vision_offset", 0.4),
                    "parking_goal_offset": rospy.get_param(
                        "~parking_goal_offset", 0.26),
                    "parking_staging_offset": rospy.get_param(
                        "~parking_staging_offset", 0.55),
                    "parking_staging_timeout_sec": rospy.get_param(
                        "~parking_staging_timeout_sec", 20.0),
                    "parking_staging_position_tolerance": rospy.get_param(
                        "~parking_staging_position_tolerance", 0.10),
                    "parking_staging_yaw_tolerance": rospy.get_param(
                        "~parking_staging_yaw_tolerance", 0.10),
                    "parking_docking_timeout_sec": rospy.get_param(
                        "~parking_docking_timeout_sec", 15.0),
                    "parking_dock_max_x": rospy.get_param(
                        "~parking_dock_max_x", 0.10),
                    "parking_dock_max_y": rospy.get_param(
                        "~parking_dock_max_y", 0.06),
                    "parking_dock_max_yaw": rospy.get_param(
                        "~parking_dock_max_yaw", 0.15),
                    "parking_dock_min_yaw": rospy.get_param(
                        "~parking_dock_min_yaw", 0.15),
                    "parking_dock_normal_tolerance": rospy.get_param(
                        "~parking_dock_normal_tolerance", 0.015),
                    "parking_dock_tangent_tolerance": rospy.get_param(
                        "~parking_dock_tangent_tolerance", 0.02),
                    "parking_dock_yaw_tolerance": rospy.get_param(
                        "~parking_dock_yaw_tolerance", 0.035),
                    "parking_min_wall_distance": rospy.get_param(
                        "~parking_min_wall_distance", 0.19),
                    "parking_lidar_stop_distance": rospy.get_param(
                        "~parking_lidar_stop_distance", 0.15),
                    "parking_recenter_tolerance": rospy.get_param(
                        "~parking_recenter_tolerance", 0.04),
                    "parking_recenter_timeout_sec": rospy.get_param(
                        "~parking_recenter_timeout_sec", 8.0),
                    "parking_recenter_initial_wait_sec": rospy.get_param(
                        "~parking_recenter_initial_wait_sec", 1.0),
                    "parking_wall_fit_half_angle_deg": rospy.get_param(
                        "~parking_wall_fit_half_angle_deg", 35.0),
                    "parking_wall_fit_min_points": rospy.get_param(
                        "~parking_wall_fit_min_points", 12),
                    "parking_wall_fit_min_span": rospy.get_param(
                        "~parking_wall_fit_min_span", 0.25),
                    "parking_wall_fit_near_min_span": rospy.get_param(
                        "~parking_wall_fit_near_min_span", 0.18),
                    "parking_wall_fit_max_distance_jump": rospy.get_param(
                        "~parking_wall_fit_max_distance_jump", 0.05),
                    "parking_wall_fit_max_normal_jump_deg": rospy.get_param(
                        "~parking_wall_fit_max_normal_jump_deg", 8.0),
                    "parking_wall_fit_max_residual": rospy.get_param(
                        "~parking_wall_fit_max_residual", 0.015),
                    "parking_wall_fit_max_normal_error_deg": rospy.get_param(
                        "~parking_wall_fit_max_normal_error_deg", 20.0),
                    "parking_normal_offset": rospy.get_param(
                        "~parking_normal_offset", 0.0),
                    "parking_tangent_offset": rospy.get_param(
                        "~parking_tangent_offset", 0.0),
                    "parking_box_width": rospy.get_param("~parking_box_width", 0.50),
                    "parking_box_depth": rospy.get_param("~parking_box_depth", 0.50),
                    "parking_xy_tolerance": rospy.get_param(
                        "~parking_xy_tolerance", 0.04),
                    "parking_yaw_tolerance": rospy.get_param(
                        "~parking_yaw_tolerance", 0.06),
                    "target_center_coarse_step_deg": rospy.get_param(
                        "~target_center_coarse_step_deg", 4.0),
                    "target_center_fine_step_deg": rospy.get_param(
                        "~target_center_fine_step_deg", 2.0),
                    "target_center_start_speed": rospy.get_param(
                        "~target_center_start_speed", 0.20),
                    "target_center_step_max_speed": rospy.get_param(
                        "~target_center_max_speed", 0.35),
                    "target_center_timeout_sec": rospy.get_param(
                        "~target_center_timeout_sec", 12.0),
                    "coverage_scan_step_deg": rospy.get_param(
                        "~coverage_scan_step_deg", 20.0),
                    "coverage_scan_angular_speed": rospy.get_param(
                        "~coverage_scan_angular_speed", 0.35),
                    "coverage_scan_dwell_sec": rospy.get_param(
                        "~coverage_scan_dwell_sec", 0.65),
                    "coverage_candidate_hold_sec": rospy.get_param(
                        "~coverage_candidate_hold_sec", 1.2),
                    "coverage_scan_max_dwell_sec": rospy.get_param(
                        "~coverage_scan_max_dwell_sec", 2.0),
                    "coverage_scan_pose_timeout_sec": rospy.get_param(
                        "~coverage_scan_pose_timeout_sec", 0.5),
                    "coverage_goal_soft_timeout_sec": rospy.get_param(
                        "~coverage_goal_soft_timeout_sec", 25.0),
                    "coverage_goal_hard_timeout_sec": rospy.get_param(
                        "~coverage_goal_hard_timeout_sec", 40.0),
                    "coverage_goal_progress_window_sec": rospy.get_param(
                        "~coverage_goal_progress_window_sec", 5.0),
                    "coverage_goal_min_progress": rospy.get_param(
                        "~coverage_goal_min_progress", 0.03),
                },
            )
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
        announcement_required = task2_announcement_required(
            self.navigator_status, self.task2_announcement_completed)
        if not self.task2_announcement_completed and not announcement_required:
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
        self.publish_status("task2", "completed", "target factory reached")

    def task3(self):
        if not self.category:
            raise StageError("task3 target_category is missing")
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
            request = {"command": "start", "target": self.category, "request_id": request_id}
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
        workshop = self.task1_result.get("sim_workshop") or CATEGORY_LABELS[self.category][1]
        if not item:
            raise StageError("task3 sim_item is missing")
        self.announce("task3", item=item, workshop=workshop)
        self.publish_status("task3", "completed", result_text)

    def approach_task4_stop_line(self):
        self.strict_mission_status = {}
        self.publish_status(
            "task4", "approaching_stop_line",
            "navigating to staging pose, then approaching the stop line visually")
        self.start_child(
            "strict_line",
            "ucar_2026_strict_mission",
            "strict_mission.launch",
            {
                "start_traffic_detector": False,
                "start_viewer": self.debug,
                "traffic_pose_configured": True,
                "traffic_staging_x": float(rospy.get_param("~traffic_x")),
                "traffic_staging_y": float(rospy.get_param("~traffic_y")),
                "traffic_staging_yaw": float(rospy.get_param("~traffic_yaw")),
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
            while time.time() < deadline:
                self.check_abort()
                status = self.strict_mission_status
                state = str(status.get("state", ""))
                if state == "WAIT_TRAFFIC":
                    distance = status.get("distance_m")
                    self.publish_status(
                        "task4", "stop_line_reached",
                        "vehicle held before stop line; distance_m={}".format(distance))
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
                    self.announce("task4", decision=decision)
                    self.traffic_pub.publish(String(data=decision))
                    self.publish_status("task4", "completed", "decision={}".format(decision))
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
            for stage in stage_sequence(self.mode, self.enable_simulation):
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
