#!/usr/bin/env python3
"""Robot-side process-group owner for the physical competition roslaunch."""

import argparse
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import threading
import time

from ucar_2026_competition.local_sim import TERMINATION_STEPS
from ucar_2026_competition.remote_robot import (
    RobotDeploymentError,
    decode_launch_arguments,
    resolve_physical_launch_arguments,
)


def fail(message):
    raise RobotDeploymentError(message)


def validate_runtime(arguments, launch_arguments):
    workspace = os.path.realpath(arguments.workspace)
    environment_file = os.path.realpath(arguments.environment_file)
    if not os.path.isfile(environment_file):
        fail("robot environment file is missing: {}".format(environment_file))
    if not os.environ.get("XF_SPARK_API_PASSWORD", "").strip():
        fail("XF_SPARK_API_PASSWORD is missing from {}".format(environment_file))
    if launch_arguments.get("start_external_voice", "true").lower() == "true":
        credentials = os.path.expanduser(os.environ.get(
            "IAT_CREDENTIALS_FILE",
            "/home/ucar/.config/ucar_2026/iat_credentials.json"))
        if not os.path.isfile(credentials):
            fail("IAT credentials are missing: {}".format(credentials))
        if importlib.util.find_spec("websocket") is None:
            fail("Python websocket-client module is missing on the robot")
        legacy_workspace = os.path.expanduser(
            os.environ.get("ROBOT_WS", "/home/ucar/ucar_ws"))
        legacy_required = (
            os.path.join(legacy_workspace, "src", "speech_command", "package.xml"),
            os.path.join(
                legacy_workspace, "devel", "lib", "speech_command",
                "speech_command_node"),
            os.path.join(
                legacy_workspace, "devel", "lib", "speech_command",
                "voice_speak_node"),
        )
        missing_legacy = [path for path in legacy_required if not os.path.isfile(path)]
        if missing_legacy:
            fail("legacy speech runtime is incomplete: {}".format(
                ", ".join(missing_legacy)))
    revision = subprocess.check_output(
        ["git", "-C", workspace, "rev-parse", "HEAD"], text=True).strip()
    if revision != arguments.expected_revision:
        fail("competition revision changed after preflight")
    status = subprocess.check_output(
        ["git", "-C", workspace, "status", "--porcelain"], text=True).strip()
    if status:
        fail("robot competition repository changed after preflight")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.3)
    try:
        if probe.connect_ex(("127.0.0.1", 11311)) == 0:
            fail("robot ROS master port 11311 is already in use")
    finally:
        probe.close()
    return workspace


def stop_process_group(process):
    if process is None or process.poll() is not None:
        return
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--environment-file", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--launch-arguments", required=True)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    return parser.parse_args()


def main():
    arguments = parse_args()
    process = None
    stopping = threading.Event()

    def request_stop(_signum=None, _frame=None):
        stopping.set()

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)

    def watch_stdin():
        try:
            while not stopping.is_set():
                chunk = sys.stdin.buffer.read(1)
                if not chunk:
                    stopping.set()
                    return
        except OSError:
            stopping.set()

    try:
        launch_arguments = decode_launch_arguments(arguments.launch_arguments)
        _workspace = validate_runtime(arguments, launch_arguments)
        launch_arguments = resolve_physical_launch_arguments(
            launch_arguments, os.environ.get("SSH_CONNECTION", ""))
        command = [
            "roslaunch", "ucar_2026_competition", "physical_competition.launch"
        ] + ["{}:={}".format(name, value) for name, value in sorted(launch_arguments.items())]
        process = subprocess.Popen(command, start_new_session=True)
        watcher = threading.Thread(target=watch_stdin, name="ssh-stdin-watch", daemon=True)
        watcher.start()
        deadline = time.monotonic() + arguments.startup_timeout
        while not stopping.is_set() and time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                return returncode
            try:
                ping = subprocess.run(
                    ["rosnode", "ping", "-c", "1", "/competition_flow"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                )
            except (OSError, subprocess.TimeoutExpired):
                ping = None
            if ping is not None and ping.returncode == 0:
                break
            time.sleep(0.2)
        else:
            if stopping.is_set():
                return 0
            fail("physical competition did not become ready within {:.1f}s".format(
                arguments.startup_timeout))
        print("ROBOT_COMPETITION_READY bridge={}".format(
            launch_arguments.get("sim_bridge_host", "disabled")), flush=True)
        while not stopping.is_set():
            returncode = process.poll()
            if returncode is not None:
                return returncode
            time.sleep(0.2)
        return 0
    except Exception as exc:
        print("ROBOT_COMPETITION_FAILED {}".format(exc), file=sys.stderr, flush=True)
        return 1
    finally:
        stop_process_group(process)


if __name__ == "__main__":
    sys.exit(main())
