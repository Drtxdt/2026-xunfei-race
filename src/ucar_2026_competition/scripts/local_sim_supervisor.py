#!/usr/bin/env python3
"""Supervise a local Gazebo stack on a second, isolated ROS master."""

import json
import os
import signal
import socket
import subprocess
import sys
import time

import rospy
from std_msgs.msg import String

from ucar_2026_competition.local_sim import (
    LocalSimConfigError,
    TERMINATION_STEPS,
    build_isolated_environment,
    build_launch_command,
    ensure_port_available,
    validate_port,
    validate_workspace,
)


class LocalSimSupervisor:
    def __init__(self):
        self.workspace = validate_workspace(rospy.get_param("~sim_workspace", ""))
        self.master_port = validate_port(rospy.get_param("~sim_master_port", 11312), "sim_master_port")
        self.bridge_host = str(rospy.get_param("~sim_bridge_host", "127.0.0.1")).strip()
        self.bridge_port = validate_port(rospy.get_param("~sim_bridge_port", 26003), "sim_bridge_port")
        self.gui = bool(rospy.get_param("~sim_gui", True))
        self.startup_timeout = float(rospy.get_param("~sim_startup_timeout_sec", 120.0))
        if self.startup_timeout <= 0:
            raise LocalSimConfigError("sim_startup_timeout_sec must be positive")
        if self.gui and not os.environ.get("DISPLAY"):
            raise LocalSimConfigError("sim_gui is true but DISPLAY is not set")

        self.status_pub = rospy.Publisher("/competition/local_sim/status", String, queue_size=1, latch=True)
        self.roscore = None
        self.sim_launch = None
        self.environment = None
        rospy.on_shutdown(self.shutdown)

    def publish_status(self, state, detail=""):
        payload = {"state": state, "detail": detail, "timestamp": time.time()}
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False, sort_keys=True)))

    @staticmethod
    def _spawn(command, environment):
        return subprocess.Popen(command, env=environment, start_new_session=True)

    @staticmethod
    def _process_exit(process):
        return None if process is None else process.poll()

    def _assert_children_alive(self):
        roscore_exit = self._process_exit(self.roscore)
        launch_exit = self._process_exit(self.sim_launch)
        if roscore_exit is not None:
            raise RuntimeError("isolated roscore exited with code {}".format(roscore_exit))
        if self.sim_launch is not None and launch_exit is not None:
            raise RuntimeError("simulator roslaunch exited with code {}".format(launch_exit))

    @staticmethod
    def _tcp_ready(host, port):
        try:
            connection = socket.create_connection((host, port), timeout=0.4)
            connection.close()
            return True
        except OSError:
            return False

    def _controller_ready(self, timeout):
        try:
            completed = subprocess.run(
                ["rostopic", "echo", "-n", "1", "/task3/state"],
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0 and b"READY" in completed.stdout

    def _ros_command_succeeds(self, command, timeout=2.0):
        try:
            completed = subprocess.run(
                command,
                env=self.environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    def _wait_for_master(self, deadline):
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.roscore.poll() is not None:
                raise RuntimeError("isolated roscore exited with code {}".format(self.roscore.returncode))
            if self._tcp_ready("127.0.0.1", self.master_port):
                return
            time.sleep(0.2)
        raise RuntimeError("isolated ROS master did not become ready before timeout")

    def _wait_for_simulator(self, deadline):
        bridge_ready = False
        controller_ready = False
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self._assert_children_alive()
            bridge_ready = bridge_ready or self._tcp_ready(self.bridge_host, self.bridge_port)
            remaining = max(0.1, deadline - time.monotonic())
            if not controller_ready:
                controller_ready = self._controller_ready(min(2.0, remaining))
            if bridge_ready and controller_ready:
                return
            time.sleep(0.2)
        missing = []
        if not controller_ready:
            missing.append("/task3/state=READY")
        if not bridge_ready:
            missing.append("TCP bridge")
        raise RuntimeError("simulator startup timed out waiting for {}".format(" and ".join(missing)))

    def run(self):
        self.publish_status("starting")
        ensure_port_available("127.0.0.1", self.master_port, "sim_master_port")
        ensure_port_available(self.bridge_host, self.bridge_port, "sim_bridge_port")
        self.environment = build_isolated_environment(self.workspace, self.master_port)

        deadline = time.monotonic() + self.startup_timeout
        self.roscore = self._spawn(["roscore", "-p", str(self.master_port)], self.environment)
        self._wait_for_master(deadline)
        self.publish_status("master_ready")

        command = build_launch_command(self.gui, self.bridge_host, self.bridge_port)
        rospy.loginfo("Starting isolated simulator: %s", " ".join(command))
        self.sim_launch = self._spawn(command, self.environment)
        self._wait_for_simulator(deadline)
        self.publish_status("ready")
        rospy.loginfo("Local simulator is ready on ROS master 127.0.0.1:%d", self.master_port)

        next_health_check = 0.0
        while not rospy.is_shutdown():
            self._assert_children_alive()
            now = time.monotonic()
            if now >= next_health_check:
                if not self._ros_command_succeeds(
                        ["rosservice", "info", "/gazebo/get_world_properties"]):
                    raise RuntimeError("Gazebo health service disappeared")
                next_health_check = now + 1.0
            time.sleep(0.25)
        return 0

    @staticmethod
    def _stop_process_group(process, label):
        if process is None or process.poll() is not None:
            return
        try:
            process_group = os.getpgid(process.pid)
        except (OSError, ProcessLookupError):
            return
        for sig, timeout in TERMINATION_STEPS:
            try:
                os.killpg(process_group, sig)
            except (OSError, ProcessLookupError):
                return
            try:
                process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                signal_name = getattr(signal.Signals(sig), "name", str(sig))
                rospy.logwarn("%s did not stop after %s", label, signal_name)

    def shutdown(self):
        self._stop_process_group(self.sim_launch, "simulator roslaunch")
        self._stop_process_group(self.roscore, "isolated roscore")
        if hasattr(self, "status_pub"):
            try:
                self.publish_status("stopped")
            except rospy.ROSException:
                pass


def main():
    rospy.init_node("local_sim_supervisor")
    supervisor = None
    try:
        supervisor = LocalSimSupervisor()
        return supervisor.run()
    except Exception as exc:
        rospy.logfatal("Local simulator supervisor failed: %s", exc)
        if supervisor is not None:
            supervisor.publish_status("failed", str(exc))
            supervisor.shutdown()
        return 1


if __name__ == "__main__":
    sys.exit(main())
