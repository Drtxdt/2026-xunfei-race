#!/usr/bin/env python3
"""Pure helpers for launching the bundled simulator in an isolated ROS environment."""

import os
import signal
import socket
import subprocess


class LocalSimConfigError(ValueError):
    """Raised when the local simulator configuration is unsafe or incomplete."""


ROS_SETUP = "/opt/ros/noetic/setup.bash"
TERMINATION_STEPS = (
    (signal.SIGINT, 5.0),
    (signal.SIGTERM, 3.0),
    (getattr(signal, "SIGKILL", 9), 1.0),
)


def validate_port(value, name):
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise LocalSimConfigError("{} must be an integer".format(name))
    if port < 1 or port > 65535:
        raise LocalSimConfigError("{} must be between 1 and 65535".format(name))
    return port


def validate_workspace(path):
    raw_path = str(path or "").strip()
    if not raw_path:
        raise LocalSimConfigError(
            "sim_workspace is empty; export UCAR_SIM_WS to the simulator gazebo_ws directory"
        )

    workspace = os.path.realpath(os.path.abspath(os.path.expanduser(raw_path)))
    required = (
        workspace,
        os.path.join(workspace, "devel", "setup.bash"),
        os.path.join(workspace, "src", "car3", "package.xml"),
    )
    labels = ("workspace directory", "devel/setup.bash", "src/car3/package.xml")
    for candidate, label in zip(required, labels):
        exists = os.path.isdir(candidate) if label == "workspace directory" else os.path.isfile(candidate)
        if not exists:
            raise LocalSimConfigError("invalid simulator workspace: missing {} at {}".format(label, candidate))
    return workspace


def clean_seed_environment(source=None):
    """Return only host/session variables needed before sourcing the simulator overlay."""
    source = dict(os.environ if source is None else source)
    exact = {
        "DBUS_SESSION_BUS_ADDRESS",
        "DISPLAY",
        "HOME",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "USER",
        "WAYLAND_DISPLAY",
        "XAUTHORITY",
        "XDG_RUNTIME_DIR",
    }
    prefixes = ("GAZEBO_", "IGN_", "NVIDIA_", "QT_", "__GL_")
    result = {
        key: value
        for key, value in source.items()
        if key in exact or any(key.startswith(prefix) for prefix in prefixes)
    }
    result.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    return result


def parse_null_environment(payload):
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="surrogateescape")
    result = {}
    for item in payload.split("\0"):
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key] = value
    return result


def build_isolated_environment(workspace, master_port, source=None, runner=subprocess.run):
    workspace = validate_workspace(workspace)
    master_port = validate_port(master_port, "sim_master_port")
    sim_setup = os.path.join(workspace, "devel", "setup.bash")
    if not os.path.isfile(ROS_SETUP):
        raise LocalSimConfigError("ROS Noetic setup not found at {}".format(ROS_SETUP))

    seed = clean_seed_environment(source)
    command = [
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        'source "$1" && source "$2" && env -0',
        "local-sim-env",
        ROS_SETUP,
        sim_setup,
    ]
    completed = runner(command, env=seed, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise LocalSimConfigError("failed to source simulator workspace: {}".format(stderr or "unknown error"))

    environment = parse_null_environment(completed.stdout)
    environment["ROS_MASTER_URI"] = "http://127.0.0.1:{}".format(master_port)
    environment["ROS_HOSTNAME"] = "127.0.0.1"
    environment.pop("ROS_IP", None)
    return environment


def build_launch_command(gui, bridge_host, bridge_port):
    bridge_port = validate_port(bridge_port, "sim_bridge_port")
    if bridge_host != "127.0.0.1":
        raise LocalSimConfigError("the managed local simulator bridge must bind to 127.0.0.1")
    return [
        "roslaunch",
        "car3",
        "task3.launch",
        "target:=wait",
        "start_bridge:=true",
        "gui:={}".format("true" if gui else "false"),
        "bridge_host:={}".format(bridge_host),
        "bridge_port:={}".format(bridge_port),
    ]


def ensure_port_available(host, port, name):
    port = validate_port(port, name)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
    except OSError as exc:
        raise LocalSimConfigError("{} {}:{} is unavailable: {}".format(name, host, port, exc))
    finally:
        probe.close()
    return port
