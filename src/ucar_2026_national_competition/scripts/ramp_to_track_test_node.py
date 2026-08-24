#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ramp-to-track integration test for the national competition.

This node runs a stripped-down national flow that skips voice, QR scanning,
factory-sign search and parking.  The sequence is:

  1. Publish an /initialpose guess so AMCL can localize before moving.
  2. Navigate to the ramp staging pose.
  3. Start the up-and-down ramp traverse via /ramp_traverse/start.
  4. Wait until the ramp node reports DONE.
  5. Re-publish an /initialpose guess at the calibrated post-ramp pose.
  6. Clear costmaps.
  7. Navigate to the yellow-line stop pose (traffic_x/y/yaw).
  8. Hand over to the provincial task4_task5 flow (traffic-light wait +
     line following) by launching ucar_2026_competition/flow_node.launch.

Speech is disabled for this test.  To keep the provincial task4/task5
announcements from failing, this node advertises a dummy
/competition_speech/announce service that returns success without doing
anything.

On any stage failure the node logs the error and exits immediately (no
pause/resume loop).
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
import rospy
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger

from ucar_2026_competition_speech.srv import Announce, AnnounceResponse
from ucar_2026_national_competition.logic import (
    build_roslaunch_command,
    heading_alignment_command,
    shortest_angular_error,
    validate_pose,
)


class StageError(Exception):
    pass


class RampToTrackTestNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.children = {}
        self.odom_msg = None
        self.odom_at = 0.0
        self.scan_msg = None
        self.ramp_state = ""
        self.ramp_state_at = 0.0
        self.ramp_payload = {}

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

        # ------------------------------ ramp staging -------------------------
        self.ramp_staging_pose = validate_pose(
            rospy.get_param("~ramp_staging_x", 0.0),
            rospy.get_param("~ramp_staging_y", 0.0),
            rospy.get_param("~ramp_staging_yaw", 0.0),
            "ramp staging",
        )
        self.ramp_navigation_timeout_sec = float(rospy.get_param(
            "~ramp_navigation_timeout_sec", 90.0))
        self.ramp_heading_alignment_enabled = bool(rospy.get_param(
            "~ramp_heading_alignment_enabled", True))
        self.ramp_heading_frame = str(rospy.get_param(
            "~ramp_heading_frame", "map")).strip()
        self.ramp_heading_base_frame = str(rospy.get_param(
            "~ramp_heading_base_frame", "base_link")).strip()
        self.ramp_heading_tolerance_rad = math.radians(float(rospy.get_param(
            "~ramp_heading_tolerance_deg", 2.0)))
        self.ramp_heading_stable_sec = float(rospy.get_param(
            "~ramp_heading_stable_sec", 0.4))
        self.ramp_heading_timeout_sec = float(rospy.get_param(
            "~ramp_heading_timeout_sec", 12.0))
        self.ramp_heading_kp = float(rospy.get_param(
            "~ramp_heading_kp", 1.5))
        self.ramp_heading_min_speed = float(rospy.get_param(
            "~ramp_heading_min_angular_speed", 0.20))
        self.ramp_heading_max_speed = float(rospy.get_param(
            "~ramp_heading_max_angular_speed", 0.25))

        # ------------------------------ ramp execution -----------------------
        self.ramp_start_service = rospy.get_param(
            "~ramp_start_service", "/ramp_traverse/start")
        self.ramp_timeout_sec = float(rospy.get_param("~ramp_timeout_sec", 150.0))
        self.ramp_poll_interval_sec = max(
            0.02, float(rospy.get_param("~ramp_poll_interval_sec", 0.1)))
        self.post_ramp_settle_sec = float(rospy.get_param(
            "~post_ramp_settle_sec", 1.5))
        self.post_ramp_clear_costmap = bool(rospy.get_param(
            "~post_ramp_clear_costmap", True))

        # ------------------------------ relocalization -----------------------
        self.relocalize_enabled = bool(rospy.get_param(
            "~relocalize_enabled", True))
        self.relocalize_pose = validate_pose(
            rospy.get_param("~relocalize_x", 0.0),
            rospy.get_param("~relocalize_y", 0.0),
            rospy.get_param("~relocalize_yaw", 0.0),
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

        # ------------------------------ stationary check ---------------------
        self.stationary_timeout_sec = float(rospy.get_param(
            "~stationary_timeout_sec", 5.0))
        self.stationary_stable_sec = float(rospy.get_param(
            "~stationary_stable_sec", 0.5))
        self.stationary_odom_stale_sec = float(rospy.get_param(
            "~stationary_odom_stale_sec", 0.5))

        # ------------------------------ yellow-line stop ----------------------
        self.traffic_pose = validate_pose(
            rospy.get_param("~traffic_x", 0.2395),
            rospy.get_param("~traffic_y", -3.10),
            rospy.get_param("~traffic_yaw", -1.5596),
            "yellow-line stop",
        )
        self.traffic_navigation_timeout_sec = float(rospy.get_param(
            "~traffic_navigation_timeout_sec", 90.0))

        # ------------------------------ hand-over ----------------------------
        self.track_package = str(rospy.get_param(
            "~track_package", "ucar_2026_track_end_stop"))
        self.handover_package = str(rospy.get_param(
            "~handover_package", "ucar_2026_competition"))
        self.handover_launch_file = str(rospy.get_param(
            "~handover_launch_file", "flow_node.launch"))

        # ------------------------------ run mode -----------------------------
        self.skip_ramp = bool(rospy.get_param("~skip_ramp", False))

        # ------------------------------ publishers / subscribers -------------
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.initialpose_pub = rospy.Publisher(
            "/initialpose", PoseWithCovarianceStamped, queue_size=1)

        rospy.Subscriber("/odom", Odometry, self._odom_cb)
        rospy.Subscriber("/scan", LaserScan, self._scan_cb)
        rospy.Subscriber("/ramp_traverse/status", String, self._ramp_status_cb)

        # Dummy speech service: keeps provincial task4/task5 announcements
        # from failing when the real speech stack is not started.
        rospy.Service("/competition_speech/announce", Announce,
                      self._announce_cb)

        self.move_base = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        rospy.on_shutdown(self.shutdown)

        rospy.loginfo("ramp_to_track_test node initialized")
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    # ------------------------------ callbacks -----------------------------
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

    def _announce_cb(self, request):
        rospy.loginfo(
            "[ramp-to-track] speech disabled; ignoring announce event=%s",
            request.event)
        return AnnounceResponse(
            success=True,
            speech_text="",
            estimated_duration=0.0,
            message="speech disabled in ramp-to-track test",
        )

    # ------------------------------ children -----------------------------
    def start_child(self, key, package, launch_file, args=None):
        self.stop_child(key)
        command = build_roslaunch_command(package, launch_file, args)
        rospy.loginfo("[ramp-to-track] starting child: %s", " ".join(command))
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

    def shutdown(self):
        self.safe_stop(cancel_navigation=True)
        self.stop_all_children()

    # ------------------------------ safety --------------------------------
    def check_abort(self):
        if rospy.is_shutdown():
            raise StageError("ros shutdown")

    def safe_stop(self, cancel_navigation=False):
        if cancel_navigation:
            try:
                self.move_base.cancel_all_goals()
            except Exception:
                pass
        for _ in range(3):
            self.cmd_pub.publish(Twist())
            rospy.sleep(0.03)

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
        rospy.loginfo("[ramp-to-track] %s", message)
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
        rospy.loginfo(
            "[ramp-to-track] waiting for base to be stationary "
            "(linear<=0.01 m/s, angular<=0.02 rad/s for %.1fs)", stable_sec)
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
                    rospy.loginfo("[ramp-to-track] base is stationary")
                    return
            else:
                stable_since = None
            rospy.sleep(0.1)
        raise StageError("wait_stationary timed out ({:.0f}s)".format(timeout_sec))

    def align_ramp_staging_heading(self):
        if not self.ramp_heading_alignment_enabled:
            rospy.logwarn("[ramp-to-track] ramp heading alignment disabled")
            return
        if not self.ramp_heading_frame or not self.ramp_heading_base_frame:
            raise StageError("ramp heading alignment frames must not be empty")

        target_yaw = self.ramp_staging_pose[2]
        deadline = time.monotonic() + max(1.0, self.ramp_heading_timeout_sec)
        stable_since = None
        last_error = None
        tf_received = False
        rate = rospy.Rate(30)
        rospy.loginfo(
            "[ramp-to-track] aligning ramp heading: target=%.2fdeg "
            "tolerance=%.2fdeg stable=%.1fs",
            math.degrees(target_yaw),
            math.degrees(self.ramp_heading_tolerance_rad),
            self.ramp_heading_stable_sec,
        )
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.ramp_heading_frame,
                    self.ramp_heading_base_frame,
                    rospy.Time(0),
                    rospy.Duration(0.15),
                )
            except tf2_ros.TransformException:
                self.cmd_pub.publish(Twist())
                stable_since = None
                rate.sleep()
                continue

            tf_received = True
            orientation = transform.transform.rotation
            current_yaw = math.atan2(
                2.0 * (orientation.w * orientation.z
                       + orientation.x * orientation.y),
                1.0 - 2.0 * (orientation.y * orientation.y
                             + orientation.z * orientation.z),
            )
            error = shortest_angular_error(target_yaw, current_yaw)
            last_error = error
            angular = heading_alignment_command(
                error,
                self.ramp_heading_tolerance_rad,
                self.ramp_heading_kp,
                self.ramp_heading_min_speed,
                self.ramp_heading_max_speed,
            )
            if angular == 0.0:
                self.cmd_pub.publish(Twist())
                if stable_since is None:
                    stable_since = time.monotonic()
                elif (time.monotonic() - stable_since >=
                      max(0.1, self.ramp_heading_stable_sec)):
                    self.safe_stop(cancel_navigation=True)
                    rospy.loginfo(
                        "[ramp-to-track] ramp heading aligned: actual=%.2fdeg "
                        "error=%+.2fdeg",
                        math.degrees(current_yaw), math.degrees(error))
                    return
            else:
                stable_since = None
                command = Twist()
                command.angular.z = angular
                self.cmd_pub.publish(command)
                rospy.loginfo_throttle(
                    0.5,
                    "[ramp-to-track] heading actual=%.2fdeg error=%+.2fdeg "
                    "cmd=%+.3f",
                    math.degrees(current_yaw), math.degrees(error), angular)
            rate.sleep()

        self.safe_stop(cancel_navigation=True)
        if not tf_received:
            raise StageError(
                "no TF {} -> {} for ramp heading alignment".format(
                    self.ramp_heading_frame, self.ramp_heading_base_frame))
        raise StageError(
            "ramp heading alignment timed out after {:.1f}s; "
            "error={:+.2f}deg".format(
                self.ramp_heading_timeout_sec,
                math.degrees(last_error)
                if last_error is not None else float("nan")))

    def clear_costmaps(self):
        if not self.post_ramp_clear_costmap:
            rospy.loginfo("[ramp-to-track] skipping costmap clear per config")
            return
        rospy.loginfo("[ramp-to-track] clearing costmaps")
        try:
            rospy.wait_for_service("/move_base/clear_costmaps", timeout=5.0)
            rospy.ServiceProxy("/move_base/clear_costmaps", Empty)()
        except Exception as exc:
            rospy.logwarn("[ramp-to-track] clear_costmaps failed: %s", exc)

    # ------------------------------ pose helpers --------------------------
    def _publish_pose(self, pose, xy_sigma, yaw_sigma_deg, settle_sec, label):
        rospy.loginfo("[ramp-to-track] waiting for /initialpose subscribers ...")
        deadline = rospy.get_time() + self.initial_pose_subscriber_wait_sec
        while (self.initialpose_pub.get_num_connections() == 0 and
               rospy.get_time() < deadline and not rospy.is_shutdown()):
            rospy.sleep(0.1)

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        msg.pose.pose.position.z = 0.0
        yaw = float(pose[2])
        msg.pose.pose.orientation.z = math.sin(yaw * 0.5)
        msg.pose.pose.orientation.w = math.cos(yaw * 0.5)

        xy_var = float(xy_sigma) ** 2
        yaw_var = math.radians(float(yaw_sigma_deg)) ** 2
        cov = [0.0] * 36
        cov[0] = xy_var
        cov[7] = xy_var
        cov[35] = yaw_var
        msg.pose.covariance = cov

        self.initialpose_pub.publish(msg)
        rospy.loginfo(
            "[ramp-to-track] published /initialpose for %s: "
            "x=%.3f y=%.3f yaw=%.2f deg; settling %.1fs",
            label, pose[0], pose[1], math.degrees(yaw), settle_sec)
        rospy.sleep(settle_sec)

    def publish_initial_pose(self):
        self._publish_pose(
            self.initial_pose,
            self.initial_pose_xy_sigma,
            self.initial_pose_yaw_sigma_deg,
            self.initial_pose_settle_sec,
            "startup",
        )

    def publish_relocalize_pose(self):
        self._publish_pose(
            self.relocalize_pose,
            self.relocalize_xy_sigma,
            self.relocalize_yaw_sigma_deg,
            self.relocalize_settle_sec,
            "post-ramp relocalization",
        )

    # ------------------------------ ramp ----------------------------------
    def start_ramp(self):
        self.wait_stationary()
        rospy.loginfo("[ramp-to-track] calling ramp start service %s",
                      self.ramp_start_service)
        try:
            rospy.wait_for_service(self.ramp_start_service, timeout=10.0)
            response = rospy.ServiceProxy(self.ramp_start_service, Trigger)()
        except Exception as exc:
            raise StageError("ramp start service failed: {}".format(exc))
        if not response.success:
            raise StageError("ramp start rejected: {}".format(response.message))

    def wait_ramp_done(self):
        rospy.loginfo("[ramp-to-track] waiting for ramp DONE (timeout %.0fs)",
                      self.ramp_timeout_sec)
        deadline = time.monotonic() + self.ramp_timeout_sec
        while time.monotonic() < deadline and not rospy.is_shutdown():
            self.check_abort()
            with self.lock:
                state = self.ramp_state
                payload = dict(self.ramp_payload)
            if state == "DONE":
                rospy.loginfo("[ramp-to-track] ramp traverse DONE")
                return
            if state in ("FAULT", "ABORTED"):
                raise StageError(
                    "ramp node entered {}: {}".format(
                        state, payload.get("error", "no details")))
            rospy.sleep(self.ramp_poll_interval_sec)
        raise StageError("ramp traverse timed out after {:.0f}s".format(
            self.ramp_timeout_sec))

    # ------------------------------ hand-over -----------------------------
    def run_task4_task5(self):
        args = {
            "start_stage": "task4_task5",
            "enable_simulation": False,
            "traffic_pose_configured": True,
            "traffic_x": self.traffic_pose[0],
            "traffic_y": self.traffic_pose[1],
            "traffic_yaw": self.traffic_pose[2],
            "skip_task4_stop_line_approach": False,
            "navigator_publish_initial_pose": False,
            "track_package": self.track_package,
        }
        rospy.loginfo(
            "[ramp-to-track] handing over to %s/%s with args %s",
            self.handover_package, self.handover_launch_file, args)
        proc = self.start_child(
            "task4_task5", self.handover_package, self.handover_launch_file, args)
        while proc.poll() is None and not rospy.is_shutdown():
            self.check_abort()
            rospy.sleep(0.2)
        rospy.loginfo("[ramp-to-track] task4_task5 child process finished")

    # ------------------------------ orchestration -------------------------
    def run(self):
        try:
            if self.publish_initial_pose_at_startup:
                self.publish_initial_pose()

            if self.skip_ramp:
                rospy.loginfo(
                    "[ramp-to-track] skip_ramp=true: navigating directly to "
                    "yellow-line stop: x=%.3f y=%.3f yaw=%.3f",
                    self.traffic_pose[0], self.traffic_pose[1],
                    self.traffic_pose[2])
                self.safe_stop(cancel_navigation=True)
                self.send_goal(
                    self.traffic_pose,
                    self.traffic_navigation_timeout_sec,
                    "navigating to yellow-line stop",
                )
            else:
                rospy.loginfo(
                    "[ramp-to-track] navigating to ramp staging: x=%.3f y=%.3f yaw=%.3f",
                    self.ramp_staging_pose[0], self.ramp_staging_pose[1],
                    self.ramp_staging_pose[2])
                self.safe_stop(cancel_navigation=True)
                self.send_goal(
                    self.ramp_staging_pose,
                    self.ramp_navigation_timeout_sec,
                    "navigating to ramp staging",
                )
                self.wait_stationary()
                self.align_ramp_staging_heading()

                self.start_ramp()
                self.wait_ramp_done()

                rospy.loginfo(
                    "[ramp-to-track] ramp finished; settling %.1fs",
                    self.post_ramp_settle_sec)
                rospy.sleep(self.post_ramp_settle_sec)

                if self.relocalize_enabled:
                    self.wait_stationary(stable_sec=self.relocalize_stationary_sec)
                    self.publish_relocalize_pose()
                    self.clear_costmaps()

                rospy.loginfo(
                    "[ramp-to-track] navigating to yellow-line stop: x=%.3f y=%.3f yaw=%.3f",
                    self.traffic_pose[0], self.traffic_pose[1],
                    self.traffic_pose[2])
                self.safe_stop(cancel_navigation=True)
                self.send_goal(
                    self.traffic_pose,
                    self.traffic_navigation_timeout_sec,
                    "navigating to yellow-line stop",
                )

            self.run_task4_task5()

            rospy.loginfo("[ramp-to-track] test completed successfully")
        except StageError as exc:
            rospy.logerr("[ramp-to-track] stage failed: %s", exc)
            self.shutdown()
            rospy.signal_shutdown("stage failed: {}".format(exc))
        except Exception as exc:
            rospy.logerr("[ramp-to-track] unexpected error: %s", exc)
            self.shutdown()
            rospy.signal_shutdown("unexpected error: {}".format(exc))


if __name__ == "__main__":
    rospy.init_node("ramp_to_track_test")
    RampToTrackTestNode()
    rospy.spin()
