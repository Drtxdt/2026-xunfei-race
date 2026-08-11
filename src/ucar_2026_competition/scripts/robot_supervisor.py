#!/usr/bin/env python3
"""Launch and supervise the physical competition stack over passwordless SSH."""

import os
import signal
import subprocess
import sys
import threading
import time

import rospy
from std_msgs.msg import String

from ucar_2026_competition.local_sim import TERMINATION_STEPS, validate_control_master_uri
from ucar_2026_competition.remote_robot import (
    RobotDeploymentError,
    build_remote_agent_command,
    remote_preflight_script,
    repository_revision,
    ssh_base_command,
    validate_competition_workspace,
)


class RobotSupervisor:
    def __init__(self):
        validate_control_master_uri(
            os.environ.get("ROS_MASTER_URI", "http://localhost:11311"))
        self.target = str(rospy.get_param("~robot_ssh_target", "")).strip()
        self.local_workspace = validate_competition_workspace(
            rospy.get_param("~competition_workspace", ""))
        self.robot_workspace = str(rospy.get_param(
            "~robot_workspace", "/home/ucar/2026-xunfei-race")).strip()
        self.robot_environment_file = str(rospy.get_param(
            "~robot_environment_file",
            "/home/ucar/.config/ucar_2026/robot_env.sh")).strip()
        self.connect_timeout = float(rospy.get_param("~ssh_connect_timeout_sec", 8.0))
        self.startup_timeout = float(rospy.get_param("~robot_startup_timeout_sec", 30.0))
        if self.connect_timeout <= 0 or self.startup_timeout <= 0:
            raise RobotDeploymentError("SSH and robot startup timeouts must be positive")
        self.launch_arguments = dict(rospy.get_param("~physical_launch_arguments", {}))
        self.process = None
        self.status_pub = rospy.Publisher(
            "/competition/robot_supervisor/status", String, queue_size=1, latch=True)
        rospy.on_shutdown(self.shutdown)

    def publish_status(self, state, detail=""):
        self.status_pub.publish(String(data="{}:{}".format(state, detail)))

    def _preflight(self, revision):
        command = ssh_base_command(self.target, self.connect_timeout)
        command.append(remote_preflight_script(
            self.robot_workspace, self.robot_environment_file, revision))
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.startup_timeout,
            )
        except subprocess.TimeoutExpired:
            raise RobotDeploymentError("robot SSH preflight timed out")
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "SSH failed"
            raise RobotDeploymentError("robot SSH preflight failed: {}".format(detail))
        if "REMOTE_PREFLIGHT_OK {}".format(revision) not in completed.stdout:
            raise RobotDeploymentError("robot SSH preflight returned an invalid response")

    def run(self):
        self.publish_status("preflight")
        revision = repository_revision(self.local_workspace)
        self._preflight(revision)
        # Let the local simulator start only after all cheap host/robot checks
        # have passed. This avoids starting Gazebo just to tear it down when a
        # dirty tree, revision mismatch, or SSH problem is detected.
        self.publish_status("preflight_ok", revision)
        remote_command = build_remote_agent_command(
            self.robot_workspace,
            self.robot_environment_file,
            revision,
            self.launch_arguments,
            self.startup_timeout,
        )
        debug = str(self.launch_arguments.get("debug", "false")).lower() == "true"
        command = ssh_base_command(
            self.target, self.connect_timeout, forward_x11=debug)
        command.append(remote_command)
        rospy.loginfo("Starting physical competition over SSH: target=%s revision=%s",
                      self.target, revision)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        ready = threading.Event()
        recent_output = []

        def relay_output():
            for line in self.process.stdout:
                text = line.rstrip()
                recent_output.append(text)
                del recent_output[:-20]
                print("[robot] {}".format(text), flush=True)
                if text.startswith("ROBOT_COMPETITION_READY "):
                    ready.set()

        relay = threading.Thread(
            target=relay_output, name="robot-ssh-output", daemon=True)
        relay.start()
        deadline = time.monotonic() + self.startup_timeout + 5.0
        while not ready.is_set() and not rospy.is_shutdown():
            returncode = self.process.poll()
            if returncode is not None:
                raise RuntimeError(
                    "remote physical competition failed during startup (code {}): {}".format(
                        returncode, " | ".join(recent_output[-5:])))
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "remote physical competition did not become ready within {:.1f}s".format(
                        self.startup_timeout))
            time.sleep(0.1)
        self.publish_status("running", revision)
        while not rospy.is_shutdown():
            returncode = self.process.poll()
            if returncode is not None:
                raise RuntimeError(
                    "remote physical competition exited with code {}".format(returncode))
            time.sleep(0.25)
        return 0

    @staticmethod
    def _stop_process_group(process):
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process_group = os.getpgid(process.pid)
        except OSError:
            return
        for sig, timeout in TERMINATION_STEPS:
            try:
                os.killpg(process_group, sig)
            except OSError:
                return
            try:
                process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                continue

    def shutdown(self):
        self._stop_process_group(self.process)


def main():
    rospy.init_node("robot_supervisor")
    supervisor = None
    try:
        supervisor = RobotSupervisor()
        return supervisor.run()
    except Exception as exc:
        rospy.logfatal("Robot supervisor failed: %s", exc)
        if supervisor is not None:
            supervisor.publish_status("failed", str(exc))
            supervisor.shutdown()
        return 1


if __name__ == "__main__":
    sys.exit(main())
