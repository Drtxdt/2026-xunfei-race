#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""National competition flow controller (task-1 with ramp + provincial hand-over).

All tunable parameters live in config/national_competition.yaml, which is the
single authoritative source: the launch files never override it.  Stages:

  voice_handshake      - "小飞小飞" wakeup, parse the two target categories
  navigate_ramp_staging- move_base to the calibrated ramp staging pose
  traverse_ramp        - delegate to ucar_2026_upanddown ramp_traverse node
  post_ramp_recovery   - settle, verify gate flow, clear costmaps
  navigate_qr_area     - move_base to the QR pickup area
  scan_qr              - spin-scan three QR codes (provincial decoder)
  reason_and_announce  - Spark X2 reasoning + official task-1 announcement
  handover             - launch the untouched provincial flow for task2..task5

Every stage is wrapped in a pause-and-retry loop: on failure the node stops
all motion, publishes a paused status, and waits for /national/resume.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
import time

import actionlib
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger, TriggerResponse

import rospy
from ucar_2026_competition.logic import (
    DirectedYawAccumulator,
    build_task1_instruction,
    parse_task1_categories,
    qr_values_from_payload,
)
from ucar_2026_competition_speech.srv import Announce
from ucar_2026_smart_factory_llm.srv import ReasonPickupOrder

from ucar_2026_national_competition.logic import (
    build_roslaunch_command,
    flow_launch_args,
    handover_chain,
    items_equal_allowed,
    min_valid_range,
    provincial_flow_paused,
    rotation_clearance_ok,
    stage_sequence,
    task1_categories_match,
    validate_pose,
)
from ucar_2026_upanddown.logic import normalize_angle, rotation_steps


class StageError(Exception):
    pass


def _fmt(value):
    """日志辅助: None -> '?', 浮点 -> 保留两位小数字符串。"""
    if value is None:
        return "?"
    try:
        return "{:.2f}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


class NationalFlowNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.abort_flag = False
        self.resume_event = threading.Event()

        self.stage = "startup"
        self.state = "ready"
        self.children = {}
        self.category = None
        self.sim_category = None
        self.question = ""
        self.voice_listening = False
        self.qr_items = {}
        self.qr_collecting = False
        self.qr_decoder_ready = False
        self.odom_msg = None
        self.odom_at = 0.0
        self.scan_msg = None
        self.ramp_state = ""
        self.ramp_state_at = 0.0
        self.ramp_payload = {}
        self._last_ramp_log_state = ""
        self._last_ramp_progress_log_at = 0.0
        self.provincial_status = {}
        self.task1_result_payload = None

        if not bool(rospy.get_param("~config_loaded", False)):
            rospy.logwarn(
                "national_flow: config_loaded marker is MISSING; running on "
                "built-in defaults because national_competition.yaml was not "
                "loaded. Fix the launch file before competition!")

        # ------------------------------ run mode ----------------------------
        self.start_mode = str(rospy.get_param("~start_stage", "full")).strip()
        self.ramp_enabled = bool(rospy.get_param("~ramp_enabled", True))
        self.enable_simulation = bool(rospy.get_param(
            "~enable_simulation", True))

        # ------------------------------ initial pose -------------------------
        self.publish_initial_pose_at_startup = bool(rospy.get_param(
            "~publish_initial_pose_at_startup", True))
        self.initial_pose = validate_pose(
            rospy.get_param("~initial_pose_x", 0.0),
            rospy.get_param("~initial_pose_y", 0.0),
            rospy.get_param("~initial_pose_yaw", 0.0),
            "initial pose",
        )
        self.initial_pose_xy_sigma = float(rospy.get_param(
            "~initial_pose_xy_sigma_m", 0.3))
        self.initial_pose_yaw_sigma_deg = float(rospy.get_param(
            "~initial_pose_yaw_sigma_deg", 30.0))
        self.initial_pose_subscriber_wait_sec = float(rospy.get_param(
            "~initial_pose_subscriber_wait_sec", 5.0))
        self.initial_pose_settle_sec = float(rospy.get_param(
            "~initial_pose_settle_sec", 2.0))

        # ------------------------------ poses --------------------------------
        self.ramp_staging_pose = validate_pose(
            rospy.get_param("~ramp_staging_x", 0.0),
            rospy.get_param("~ramp_staging_y", 0.0),
            rospy.get_param("~ramp_staging_yaw", 0.0),
            "ramp staging",
        )
        self.ramp_navigation_timeout_sec = float(rospy.get_param(
            "~ramp_navigation_timeout_sec", 90.0))
        self.qr_area_pose = validate_pose(
            rospy.get_param("~qr_area_x", -1.6598),
            rospy.get_param("~qr_area_y", -0.8718),
            rospy.get_param("~qr_area_yaw", 2.2645),
            "QR area",
        )
        self.qr_navigation_timeout_sec = float(rospy.get_param(
            "~qr_navigation_timeout_sec", 120.0))

        # ------------------------------ ramp ---------------------------------
        self.ramp_service = rospy.get_param(
            "~ramp_start_service", "/ramp_traverse/start")
        self.ramp_timeout_sec = float(rospy.get_param(
            "~ramp_timeout_sec", 150.0))
        self.ramp_poll_interval_sec = max(
            0.02, float(rospy.get_param("~ramp_poll_interval_sec", 0.1)))
        self.post_ramp_settle_sec = float(rospy.get_param(
            "~post_ramp_settle_sec", 1.5))
        self.post_ramp_clear_costmap = bool(rospy.get_param(
            "~post_ramp_clear_costmap", True))

        # ------------------------------ relocalization -----------------------
        self.relocalize_enabled = bool(rospy.get_param(
            "~relocalize_enabled", True))
        self.relocalize_forward_m = float(rospy.get_param(
            "~relocalize_forward_m", 0.25))
        self.relocalize_pose = validate_pose(
            rospy.get_param("~relocalize_x", -1.2),
            rospy.get_param("~relocalize_y", -0.9),
            rospy.get_param("~relocalize_yaw", 2.2645),
            "relocalize after ramp",
        )
        self.relocalize_xy_sigma = float(rospy.get_param(
            "~relocalize_xy_sigma_m", 0.1))
        self.relocalize_yaw_sigma_deg = float(rospy.get_param(
            "~relocalize_yaw_sigma_deg", 20.0))
        self.relocalize_stationary_sec = float(rospy.get_param(
            "~relocalize_stationary_sec", 0.5))
        self.relocalize_settle_sec = float(rospy.get_param(
            "~relocalize_settle_sec", 2.0))
        self.relocalize_forward_speed = float(rospy.get_param(
            "~relocalize_forward_speed", 0.30))
        self.relocalize_forward_timeout_sec = float(rospy.get_param(
            "~relocalize_forward_timeout_sec", 10.0))

        # ------------------------------ stationary ---------------------------
        self.stationary_timeout_sec = float(rospy.get_param(
            "~stationary_timeout_sec", 5.0))
        self.stationary_stable_sec = float(rospy.get_param(
            "~stationary_stable_sec", 0.5))
        self.stationary_odom_stale_sec = float(rospy.get_param(
            "~stationary_odom_stale_sec", 0.5))

        # ------------------------------ QR scan ------------------------------
        self.qr_expected_count = int(rospy.get_param("~qr_expected_count", 3))
        self.qr_total_timeout_sec = float(rospy.get_param(
            "~qr_total_timeout_sec", 120.0))
        self.qr_scan_angular_speed = float(rospy.get_param(
            "~qr_scan_angular_speed", 0.60))
        self.qr_scan_step_rad = float(rospy.get_param(
            "~qr_scan_step_rad", 0.3507993877991494))
        self.qr_scan_settle_sec = float(rospy.get_param(
            "~qr_scan_settle_sec", 0.3))
        self.qr_drain_sec = float(rospy.get_param("~qr_drain_sec", 20.0))
        self.qr_rotation_clearance = float(rospy.get_param(
            "~qr_rotation_min_clearance", 0.15))
        self.qr_fallback_enabled = bool(rospy.get_param(
            "~qr_fallback_enabled", True))
        self.qr_fallback_pose = validate_pose(
            rospy.get_param("~qr_fallback_x", -1.6598),
            rospy.get_param("~qr_fallback_y", -0.8718),
            rospy.get_param("~qr_fallback_yaw", 2.2645),
            "QR fallback",
        )
        self.qr_fallback_navigation_timeout_sec = float(rospy.get_param(
            "~qr_fallback_navigation_timeout_sec", 45.0))
        self.qr_decoder_ready_timeout_sec = float(rospy.get_param(
            "~qr_decoder_ready_timeout_sec", 6.0))

        # ------------------------------ voice / llm / speech -----------------
        self.llm_service = rospy.get_param(
            "~llm_service", "/smart_factory_llm/reason_pickup_order")
        self.llm_timeout_sec = float(rospy.get_param(
            "~task1_reasoning_timeout_sec", 200.0))
        self.announce_service = rospy.get_param(
            "~announce_service", "/competition_speech/announce")
        self.voice_start_service = rospy.get_param(
            "~voice_start_listening_service",
            "/speech_command_node/start_listening")
        self.voice_stop_service = rospy.get_param(
            "~voice_stop_listening_service",
            "/speech_command_node/stop_listening")

        # ------------------------------ hand-over ----------------------------
        self.handover_package = str(rospy.get_param(
            "~handover_package", "ucar_2026_competition"))
        self.handover_launch_file = str(rospy.get_param(
            "~handover_launch_file", "flow_node.launch"))
        # Line-following package the provincial task5 stage should roslaunch;
        # the national run uses the national track package (gray-barrier
        # obstacle avoidance), the provincial default stays untouched.
        self.track_package = str(rospy.get_param(
            "~track_package", "ucar_2026_track_end_stop"))
        self.handover_poll_interval_sec = max(
            0.05, float(rospy.get_param("~handover_poll_interval_sec", 0.2)))
        self.traffic_pose = (
            rospy.get_param("~traffic_x", 0.2395),
            rospy.get_param("~traffic_y", -3.10),
            rospy.get_param("~traffic_yaw", -1.5596),
            bool(rospy.get_param("~traffic_pose_configured", False)),
        )
        self.skip_task4_stop_line_approach = bool(rospy.get_param(
            "~skip_task4_stop_line_approach", False))

        # ------------------------------ interfaces ---------------------------
        self.status_topic = rospy.get_param("~status_topic", "/national/status")
        self.task1_result_topic = rospy.get_param(
            "~task1_result_topic", "/competition/task1_result")

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.initialpose_pub = rospy.Publisher(
            "/initialpose", PoseWithCovarianceStamped, queue_size=1)
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=10, latch=True)
        self.task1_result_pub = rospy.Publisher(
            self.task1_result_topic, String, queue_size=5, latch=True)

        rospy.Subscriber("/wakeup", String, self._wakeup_cb)
        rospy.Subscriber("/question", String, self._question_cb)
        rospy.Subscriber("/qr_code_data", String, self._qr_cb)
        rospy.Subscriber(
            "/qr_decoder/status", String, self._qr_decoder_status_cb)
        rospy.Subscriber("/odom", Odometry, self._odom_cb)
        rospy.Subscriber("/scan", LaserScan, self._scan_cb)
        rospy.Subscriber(
            "/ramp_traverse/status", String, self._ramp_status_cb)
        rospy.Subscriber(
            "/competition/status", String, self._competition_status_cb)

        rospy.Service("/national/resume", Trigger, self._resume_cb)
        rospy.Service("/national/abort", Trigger, self._abort_cb)
        rospy.on_shutdown(self.shutdown)

        self.move_base = actionlib.SimpleActionClient(
            "/move_base", MoveBaseAction)

        self.publish_status("national controller ready")
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    # ------------------------------ status / safety -----------------------
    def publish_status(self, message, state=None, stage=None):
        with self.lock:
            self.state = state or self.state
            self.stage = stage or self.stage
            payload = {
                "stage": self.stage,
                "state": self.state,
                "message": str(message),
                "error": message if self.state in ("paused", "fault") else "",
                "stamp": rospy.Time.now().to_sec(),
            }
        self.status_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False, sort_keys=True)))

    def check_abort(self):
        if rospy.is_shutdown():
            raise StageError("ros shutdown")
        if self.abort_flag:
            raise StageError("abort requested")

    def safe_stop(self, cancel_navigation=False):
        if cancel_navigation:
            try:
                self.move_base.cancel_all_goals()
            except Exception:
                pass
        for _ in range(3):
            self.cmd_pub.publish(Twist())
            rospy.sleep(0.03)

    def _resume_cb(self, _request):
        self.resume_event.set()
        return TriggerResponse(success=True, message="resuming")

    def _abort_cb(self, _request):
        with self.lock:
            self.abort_flag = True
        self.forward_provincial_abort()
        return TriggerResponse(success=True, message="abort requested")

    def forward_provincial_abort(self):
        try:
            rospy.wait_for_service("/competition/abort", timeout=2.0)
            rospy.ServiceProxy("/competition/abort", Trigger)()
        except (rospy.ROSException, rospy.ServiceException):
            pass

    def shutdown(self):
        self.safe_stop(cancel_navigation=True)
        self.stop_all_children()

    # ------------------------------ callbacks -----------------------------
    def _wakeup_cb(self, _msg):
        threading.Thread(target=self._handle_wakeup, daemon=True).start()

    def _handle_wakeup(self):
        try:
            self.publish_status(
                "wakeup received; replying", stage="voice_handshake")
            self.announce("custom", text=rospy.get_param(
                "~voice_wakeup_reply", "我在"))
            self._voice_control(self.voice_start_service)
            with self.lock:
                self.voice_listening = True
            self.publish_status("listening for the task command")
        except (StageError, rospy.ServiceException) as exc:
            rospy.logerr("voice wakeup handshake failed: %s", exc)
            self.publish_status(
                "voice wakeup handshake failed: {}".format(exc), state="fault")

    def _question_cb(self, msg):
        question = msg.data.strip()
        pickup_category, sim_category = parse_task1_categories(question)
        with self.lock:
            if not self.voice_listening:
                rospy.logwarn("ignoring /question outside voice window: %s",
                              question)
                return
            if not pickup_category or not sim_category:
                rospy.logwarn("command without two categories: %s", question)
                return
            self.voice_listening = False
            self.category = pickup_category
            self.sim_category = sim_category
            self.question = question
        try:
            self._voice_control(self.voice_stop_service)
            self.announce("custom", text=rospy.get_param(
                "~voice_command_reply", "好的"))
        except (StageError, rospy.ServiceException) as exc:
            rospy.logerr("voice command handshake failed: %s", exc)
        self.publish_status("voice command accepted: {} / {}".format(
            pickup_category, sim_category))

    def _voice_control(self, service):
        try:
            rospy.wait_for_service(service, timeout=5.0)
            response = rospy.ServiceProxy(service, Trigger)()
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError("voice control service failed: {}".format(exc))
        if not response.success:
            raise StageError("voice control rejected: {}".format(
                response.message))

    def _qr_cb(self, msg):
        if not self.qr_collecting:
            return
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        for key, result in qr_values_from_payload(payload):
            with self.lock:
                if key not in self.qr_items:
                    self.qr_items[key] = result
                    rospy.loginfo("QR accepted %d/%d: %s", len(self.qr_items),
                                  self.qr_expected_count, result)

    def _qr_decoder_status_cb(self, msg):
        try:
            payload = json.loads(msg.data)
            state = str(payload.get("state") or "")
        except (TypeError, ValueError):
            return
        with self.lock:
            self.qr_decoder_ready = state in ("ready", "idle", "fetching")

    def _odom_cb(self, msg):
        self.odom_msg = msg
        self.odom_at = rospy.get_time()

    def _scan_cb(self, msg):
        self.scan_msg = msg

    def _ramp_status_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.ramp_state = str(payload.get("state") or "")
            self.ramp_state_at = rospy.get_time()
            self.ramp_payload = payload

    def _competition_status_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        with self.lock:
            self.provincial_status = payload

    # ------------------------------ children ------------------------------
    def start_child(self, key, package, launch_file, args=None):
        self.stop_child(key)
        command = build_roslaunch_command(package, launch_file, args)
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

    # ------------------------------ speech / llm --------------------------
    def announce(self, event, item="", workshop="", decision="", text=""):
        try:
            rospy.wait_for_service(self.announce_service, timeout=5.0)
            response = rospy.ServiceProxy(self.announce_service, Announce)(
                event, item, workshop, decision, text, True)
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError("speech service failed: {}".format(exc))
        if not response.success:
            raise StageError("speech rejected: {}".format(response.message))

    # ------------------------------ navigation ----------------------------
    def send_goal(self, pose, timeout_sec, message):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = float(pose[0])
        goal.target_pose.pose.position.y = float(pose[1])
        goal.target_pose.pose.position.z = 0.0
        yaw = float(pose[2])
        goal.target_pose.pose.orientation.z = math.sin(yaw * 0.5)
        goal.target_pose.pose.orientation.w = math.cos(yaw * 0.5)

        if not self.move_base.wait_for_server(rospy.Duration(10.0)):
            raise StageError("move_base action server unavailable")
        self.move_base.send_goal(goal)
        self.publish_status(message)
        finished = self.move_base.wait_for_result(
            rospy.Duration(max(1.0, timeout_sec)))
        if not finished:
            self.move_base.cancel_goal()
            raise StageError("move_base goal timed out after {:.0f}s".format(
                timeout_sec))
        status = self.move_base.get_state()
        if status != 3:  # GoalStatus.SUCCEEDED
            raise StageError("move_base failed with state {}".format(status))

    def wait_stationary(self, stable_sec=None, timeout_sec=None):
        stable_sec = self.stationary_stable_sec if stable_sec is None else float(stable_sec)
        timeout_sec = self.stationary_timeout_sec if timeout_sec is None else float(timeout_sec)
        rospy.loginfo("【国赛·坡道】等待底盘完全静止 (线速度≤0.01m/s 角速度≤"
                      "0.02rad/s, 持续 %.1fs)...", stable_sec)
        self.publish_status("waiting for a stationary base")
        deadline = time.monotonic() + timeout_sec
        stable_since = None
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            self.safe_stop(cancel_navigation=True)
            msg = self.odom_msg
            fresh = (
                self.odom_at > 0.0 and
                rospy.get_time() - self.odom_at <= self.stationary_odom_stale_sec)
            twist = msg.twist.twist if (msg and fresh) else None
            if twist and math.hypot(twist.linear.x, twist.linear.y) <= 0.01 \
                    and abs(twist.angular.z) <= 0.02:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= stable_sec:
                    rospy.loginfo("【国赛·坡道】底盘已确认静止, 可以交接控制权")
                    return
            else:
                stable_since = None
            rospy.sleep(0.1)
        raise StageError("等待底盘静止超时 ({:.0f}s)".format(timeout_sec))

    def odom_yaw(self):
        msg = self.odom_msg
        if msg is None:
            return None
        quat = msg.pose.pose.orientation
        return normalize_angle(2.0 * math.atan2(quat.z, quat.w))

    def rotation_clearance_satisfied(self):
        msg = self.scan_msg
        if msg is None:
            return True
        nearest = min_valid_range(msg.ranges, msg.range_min, msg.range_max)
        return rotation_clearance_ok(nearest, self.qr_rotation_clearance)

    # ------------------------------ stage runners -------------------------
    def stage_voice_handshake(self):
        while not rospy.is_shutdown():
            self.check_abort()
            with self.lock:
                ready = bool(self.category and self.sim_category)
            if ready:
                return
            rospy.sleep(0.1)

    def stage_navigate_ramp_staging(self):
        rospy.loginfo(
            "【国赛·坡道】开始导航至坡道暂泊点: x=%.3f y=%.3f yaw=%.3f (%.1f°), "
            "超时 %.0fs",
            self.ramp_staging_pose[0], self.ramp_staging_pose[1],
            self.ramp_staging_pose[2],
            math.degrees(self.ramp_staging_pose[2]),
            self.ramp_navigation_timeout_sec)
        self.safe_stop(cancel_navigation=True)
        self.send_goal(
            self.ramp_staging_pose,
            self.ramp_navigation_timeout_sec,
            "导航至坡道暂泊点",
        )
        rospy.loginfo("【国赛·坡道】已到达坡道暂泊点, 准备过坡")

    def stage_traverse_ramp(self):
        self.wait_stationary()
        rospy.loginfo("【国赛·坡道】调用过坡服务 %s ...", self.ramp_service)
        self.publish_status("starting the ramp traverse")
        try:
            rospy.wait_for_service(self.ramp_service, timeout=10.0)
            response = rospy.ServiceProxy(self.ramp_service, Trigger)()
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise StageError("过坡启动服务调用失败: {}".format(exc))
        if not response.success:
            raise StageError("过坡节点拒绝启动: {}".format(response.message))
        rospy.loginfo("【国赛·坡道】过坡已启动, 轮询状态 (超时 %.0fs)",
                      self.ramp_timeout_sec)

        with self.lock:
            self.ramp_state = ""
        self._last_ramp_log_state = ""
        self._last_ramp_progress_log_at = 0.0
        deadline = time.monotonic() + self.ramp_timeout_sec
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            with self.lock:
                state = self.ramp_state
                payload = dict(self.ramp_payload)
            if state == "DONE":
                rospy.loginfo("【国赛·坡道】过坡节点报告完成 (DONE)")
                return
            if state in ("FAULT", "ABORTED"):
                raise StageError(
                    "过坡节点进入 {} 状态: {}".format(
                        state, payload.get("error", "无详情")))
            self._log_ramp_progress(state, payload)
            rospy.sleep(self.ramp_poll_interval_sec)
        raise StageError("等待过坡完成超时 ({:.0f}s), 最后状态={}".format(
            self.ramp_timeout_sec, state))

    def _log_ramp_progress(self, state, payload):
        """坡道状态变化立即打印; 状态不变时按 5s 节流打印进度摘要。"""
        now = rospy.get_time()
        if state != self._last_ramp_log_state:
            self._last_ramp_log_state = state
            self._last_ramp_progress_log_at = now
            rospy.loginfo(
                "【国赛·坡道】过坡状态: %s | 段=%s pitch=%s° 里程=%sm",
                state or "(无)",
                payload.get("segment", "?"),
                _fmt(payload.get("pitch_deg")),
                _fmt(payload.get("distance_m")))
        elif now - self._last_ramp_progress_log_at >= 5.0:
            self._last_ramp_progress_log_at = now
            rospy.loginfo(
                "【国赛·坡道】过坡进行中: 状态=%s 段=%s pitch=%s° 里程=%sm "
                "速度=%sm/s",
                state or "(无)",
                payload.get("segment", "?"),
                _fmt(payload.get("pitch_deg")),
                _fmt(payload.get("distance_m")),
                _fmt(payload.get("speed_cmd")))

    def stage_post_ramp_recovery(self):
        rospy.loginfo("【国赛·坡道】过坡完成, 静置 %.1fs 等待 AMCL 吸收恢复后的"
                      "雷达数据...", self.post_ramp_settle_sec)
        self.publish_status("settling after the ramp")
        rospy.sleep(self.post_ramp_settle_sec)
        if not self.post_ramp_clear_costmap:
            rospy.loginfo("【国赛·坡道】按配置跳过代价地图清除")
            return
        rospy.loginfo("【国赛·坡道】清除代价地图 (移除坡道期间的陈旧标记)...")
        try:
            rospy.wait_for_service("/move_base/clear_costmaps", timeout=5.0)
            rospy.ServiceProxy("/move_base/clear_costmaps", Empty)()
            rospy.loginfo("【国赛·坡道】代价地图已清除, 定位恢复阶段结束")
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("【国赛·坡道】clear_costmaps 服务不可用: %s (继续)", exc)

    def stage_relocalize_after_ramp(self):
        if not self.relocalize_enabled:
            rospy.loginfo("【国赛·重定位】按配置跳过坡道后重定位")
            return

        rospy.loginfo("【国赛·重定位】开始坡道后重定位: 前进 %.2fm 后发布 /initialpose",
                      self.relocalize_forward_m)
        self.publish_status("relocalizing after the ramp")

        self.wait_stationary(stable_sec=self.relocalize_stationary_sec)
        self._open_loop_forward(self.relocalize_forward_m)
        self.wait_stationary(stable_sec=self.relocalize_stationary_sec)

        self._publish_initialpose()
        rospy.loginfo("【国赛·重定位】/initialpose 已发布, 静置 %.1fs 等待 AMCL 收敛",
                      self.relocalize_settle_sec)
        rospy.sleep(self.relocalize_settle_sec)
        rospy.loginfo("【国赛·重定位】重定位完成")

    def _open_loop_forward(self, distance_m):
        """开环直行指定距离，靠 /odom 估算里程。"""
        rospy.loginfo("【国赛·重定位】开环前进 %.2fm (速度 %.2fm/s)",
                      distance_m, self.relocalize_forward_speed)
        start_pos = self._odom_position()
        if start_pos is None:
            raise StageError("重定位前进前无有效里程计")

        deadline = time.monotonic() + self.relocalize_forward_timeout_sec
        rate = rospy.Rate(20)
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            current = self._odom_position()
            if current is None:
                raise StageError("重定位前进中丢失里程计")
            traveled = math.hypot(
                current[0] - start_pos[0], current[1] - start_pos[1])
            if traveled >= distance_m:
                break
            twist = Twist()
            twist.linear.x = self.relocalize_forward_speed
            self.cmd_pub.publish(twist)
            rate.sleep()
        self.safe_stop()
        end_pos = self._odom_position()
        if end_pos is not None:
            rospy.loginfo("【国赛·重定位】开环前进结束, 里程计估算行走 %.3fm",
                          math.hypot(end_pos[0] - start_pos[0],
                                     end_pos[1] - start_pos[1]))

    def _odom_position(self):
        msg = self.odom_msg
        if msg is None:
            return None
        return (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _publish_initialpose(self):
        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.pose.position.x = self.relocalize_pose[0]
        pose.pose.pose.position.y = self.relocalize_pose[1]
        pose.pose.pose.position.z = 0.0
        yaw = self.relocalize_pose[2]
        pose.pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.pose.orientation.w = math.cos(yaw * 0.5)

        xy_var = self.relocalize_xy_sigma ** 2
        yaw_var = math.radians(self.relocalize_yaw_sigma_deg) ** 2
        cov = [0.0] * 36
        cov[0] = xy_var      # x
        cov[7] = xy_var      # y
        cov[35] = yaw_var    # yaw
        pose.pose.covariance = cov

        self.initialpose_pub.publish(pose)
        rospy.loginfo(
            "【国赛·重定位】已发布 /initialpose: x=%.3f y=%.3f yaw=%.2f° "
            "xy_σ=%.2fm yaw_σ=%.1f°",
            self.relocalize_pose[0], self.relocalize_pose[1],
            math.degrees(yaw), self.relocalize_xy_sigma,
            self.relocalize_yaw_sigma_deg)

    def _publish_initial_pose_at_startup(self):
        """流程启动前发布 /initialpose，让 AMCL 在第一次导航前完成定位。"""
        rospy.loginfo("【国赛·初始定位】等待 AMCL 订阅 /initialpose ...")
        deadline = rospy.get_time() + self.initial_pose_subscriber_wait_sec
        while (self.initialpose_pub.get_num_connections() == 0 and
               rospy.get_time() < deadline and not rospy.is_shutdown()):
            rospy.sleep(0.1)

        if self.initialpose_pub.get_num_connections() == 0:
            rospy.logwarn("【国赛·初始定位】%.1fs 内没有 /initialpose 订阅者，"
                          "仍尝试发布", self.initial_pose_subscriber_wait_sec)

        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.pose.position.x = self.initial_pose[0]
        pose.pose.pose.position.y = self.initial_pose[1]
        pose.pose.pose.position.z = 0.0
        yaw = self.initial_pose[2]
        pose.pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.pose.orientation.w = math.cos(yaw * 0.5)

        xy_var = self.initial_pose_xy_sigma ** 2
        yaw_var = math.radians(self.initial_pose_yaw_sigma_deg) ** 2
        cov = [0.0] * 36
        cov[0] = xy_var
        cov[7] = xy_var
        cov[35] = yaw_var
        pose.pose.covariance = cov

        self.initialpose_pub.publish(pose)
        rospy.loginfo(
            "【国赛·初始定位】已发布 /initialpose: x=%.3f y=%.3f yaw=%.2f° "
            "xy_σ=%.2fm yaw_σ=%.1f° | 订阅者=%d, 等待 %.1fs 让 AMCL 处理",
            self.initial_pose[0], self.initial_pose[1], math.degrees(yaw),
            self.initial_pose_xy_sigma, self.initial_pose_yaw_sigma_deg,
            self.initialpose_pub.get_num_connections(),
            self.initial_pose_settle_sec)
        rospy.sleep(self.initial_pose_settle_sec)

    def stage_navigate_qr_area(self):
        self.safe_stop(cancel_navigation=True)
        self.send_goal(
            self.qr_area_pose,
            self.qr_navigation_timeout_sec,
            "navigating to the QR pickup area {}".format(self.qr_area_pose),
        )

    def stage_scan_qr(self):
        deadline = rospy.get_time() + self.qr_total_timeout_sec
        with self.lock:
            self.qr_items = {}
            self.qr_collecting = True
        self.start_child(
            "qr_decoder", "ucar_2026_competition", "qr_decoder.launch")
        try:
            self._wait_qr_decoder_ready()
            self._spin_scan(deadline)
            with self.lock:
                collected = len(self.qr_items)
            if collected < self.qr_expected_count and \
                    self.qr_fallback_enabled and \
                    rospy.get_time() < deadline:
                self.publish_status(
                    "primary scan found {}/{}; moving to the fallback point".format(
                        collected, self.qr_expected_count))
                self.send_goal(
                    self.qr_fallback_pose,
                    self.qr_fallback_navigation_timeout_sec,
                    "navigating to the QR fallback point",
                )
                self._spin_scan(deadline)
            with self.lock:
                collected = len(self.qr_items)
                items = list(self.qr_items.values())
            if collected < self.qr_expected_count:
                raise StageError(
                    "only {}/{} QR codes resolved".format(
                        collected, self.qr_expected_count))
            self.qr_items_ordered = items
        finally:
            with self.lock:
                self.qr_collecting = False
            self.stop_child("qr_decoder")
            self.safe_stop()

    def _wait_qr_decoder_ready(self):
        deadline = rospy.get_time() + self.qr_decoder_ready_timeout_sec
        while rospy.get_time() < deadline and not rospy.is_shutdown():
            with self.lock:
                ready = self.qr_decoder_ready
            if ready:
                return
            rospy.sleep(0.1)
        rospy.logwarn("QR decoder did not report ready; scanning anyway")

    def _spin_scan(self, deadline):
        """Spin full circles with settling pauses until codes or deadline."""
        while rospy.get_time() < deadline and not rospy.is_shutdown():
            self.check_abort()
            with self.lock:
                collected = len(self.qr_items)
            if collected >= self.qr_expected_count:
                return
            self._spin_one_circle(deadline)
            if rospy.get_time() >= deadline:
                return
            # Full circle done: stop and let async URL fetches drain.
            drain_end = rospy.get_time() + self.qr_drain_sec
            while rospy.get_time() < drain_end and not rospy.is_shutdown():
                self.check_abort()
                with self.lock:
                    collected = len(self.qr_items)
                if collected >= self.qr_expected_count:
                    return
                rospy.sleep(0.1)

    def _spin_one_circle(self, deadline):
        """Rotate a full circle in bounded steps with settling pauses."""
        steps = rotation_steps(2.0 * math.pi, self.qr_scan_step_rad)
        for step in steps:
            start_yaw = self.odom_yaw()
            if start_yaw is None:
                raise StageError("odometry lost during QR scan")
            accumulator = DirectedYawAccumulator(direction=1.0)
            accumulator.reset(start_yaw)
            while not rospy.is_shutdown():
                self.check_abort()
                if rospy.get_time() >= deadline:
                    return
                with self.lock:
                    collected = len(self.qr_items)
                if collected >= self.qr_expected_count:
                    return
                current_yaw = self.odom_yaw()
                if current_yaw is None:
                    raise StageError("odometry lost during QR scan")
                if accumulator.update(current_yaw) >= step:
                    break
                if not self.rotation_clearance_satisfied():
                    raise StageError(
                        "rotation clearance violated during QR scan")
                twist = Twist()
                twist.angular.z = self.qr_scan_angular_speed
                self.cmd_pub.publish(twist)
                rospy.sleep(0.05)
            self.safe_stop()
            rospy.sleep(self.qr_scan_settle_sec)

    def stage_reason_and_announce(self):
        items = getattr(self, "qr_items_ordered", None) or \
            list(self.qr_items.values())
        items = items[:3]
        instruction = build_task1_instruction(
            self.category, self.sim_category)
        self.publish_status("querying Spark X2 for {}".format(items))
        try:
            rospy.wait_for_service(self.llm_service, timeout=15.0)
            result = rospy.ServiceProxy(self.llm_service, ReasonPickupOrder)(
                items[0], items[1], items[2], instruction)
        except (rospy.ROSException, rospy.ServiceException, IndexError) as exc:
            raise StageError("LLM service failed: {}".format(exc))
        if not result.success:
            raise StageError("LLM reasoning failed: {}".format(
                result.error_message))

        fields = {
            "pickup_item": result.pickup_item,
            "pickup_major": result.pickup_major,
            "pickup_workshop": result.pickup_workshop,
            "sim_item": result.sim_item,
            "sim_major": result.sim_major,
            "sim_workshop": result.sim_workshop,
        }
        if not task1_categories_match(fields, self.category, self.sim_category):
            raise StageError(
                "LLM categories {} disagree with the voice command {}".format(
                    (fields["pickup_major"], fields["sim_major"]),
                    (self.category, self.sim_category)))
        if fields["pickup_item"] == fields["sim_item"] and \
                not items_equal_allowed(self.category, self.sim_category):
            raise StageError("LLM picked the same item for two categories")

        self.announce("task1", text=result.announcement_full)
        payload = dict(fields)
        payload["announcement_full"] = result.announcement_full
        payload["instruction"] = instruction
        self.task1_result_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False, sort_keys=True)))
        with self.lock:
            self.task1_result_payload = payload
        self.publish_status("task-1 announcement delivered")

    def stage_handover(self):
        with self.lock:
            task1_result = dict(self.task1_result_payload or {})
        if not task1_result:
            raise StageError("task-1 result missing; cannot hand over")

        chain = handover_chain(self.enable_simulation)
        for start_stage in chain:
            args = flow_launch_args(
                start_stage,
                task1_result,
                self.enable_simulation,
                traffic_pose=self.traffic_pose,
                skip_task4_stop_line_approach=(
                    self.skip_task4_stop_line_approach),
                track_package=self.track_package,
            )
            self.publish_status(
                "handing over to the provincial flow: {}".format(start_stage))
            proc = self.start_child(
                "provincial_flow", self.handover_package,
                self.handover_launch_file, args)
            try:
                while proc.poll() is None and not rospy.is_shutdown():
                    self.check_abort()
                    with self.lock:
                        paused = provincial_flow_paused(
                            self.provincial_status)
                    if paused:
                        self.forward_pause()
                    rospy.sleep(self.handover_poll_interval_sec)
            finally:
                self.stop_child("provincial_flow")
            if rospy.is_shutdown() or self.abort_flag:
                raise StageError("hand-over interrupted")
            if proc.returncode not in (0, None):
                raise StageError(
                    "provincial flow {} exited with code {}".format(
                        start_stage, proc.returncode))

    def forward_pause(self):
        """Relay a provincial pause to the operator and resume it on request."""
        self.publish_status(
            "provincial flow paused; /national/resume forwards to "
            "/competition/resume", state="paused")
        self.resume_event.clear()
        while not rospy.is_shutdown():
            self.check_abort()
            if self.resume_event.wait(timeout=0.2):
                break
        try:
            rospy.wait_for_service("/competition/resume", timeout=5.0)
            rospy.ServiceProxy("/competition/resume", Trigger)()
        except (rospy.ROSException, rospy.ServiceException) as exc:
            rospy.logwarn("forwarding resume failed: %s", exc)
        self.publish_status("provincial flow resumed", state="running")

    # ------------------------------ orchestration -------------------------
    def run(self):
        if self.publish_initial_pose_at_startup:
            self._publish_initial_pose_at_startup()

        try:
            stages = stage_sequence(
                self.start_mode, self.enable_simulation, self.ramp_enabled)
        except ValueError as exc:
            self.publish_status(str(exc), state="fault")
            return

        index = 0
        while index < len(stages) and not rospy.is_shutdown():
            stage = stages[index]
            runner = getattr(self, "stage_{}".format(stage), None)
            if runner is None:
                self.publish_status("unknown stage {}".format(stage),
                                    state="fault")
                return
            self.publish_status("stage starting", stage=stage, state="running")
            try:
                runner()
                index += 1
            except StageError as exc:
                rospy.logerr("stage %s failed: %s", stage, exc)
                self.stop_all_children()
                self.safe_stop(cancel_navigation=True)
                self.pause_and_retry(stage, str(exc))
        if index >= len(stages) and not rospy.is_shutdown():
            self.publish_status("national flow completed", state="completed")

    def pause_and_retry(self, stage, reason):
        self.publish_status(
            "stage {} failed: {}; waiting for /national/resume".format(
                stage, reason),
            state="paused", stage=stage)
        self.resume_event.clear()
        while not rospy.is_shutdown():
            if self.abort_flag:
                self.publish_status("aborted", state="aborted")
                raise SystemExit(0)
            if self.resume_event.wait(timeout=0.2):
                break
        self.publish_status("retrying stage", stage=stage, state="running")


if __name__ == "__main__":
    rospy.init_node("national_flow")
    NationalFlowNode()
    rospy.spin()
