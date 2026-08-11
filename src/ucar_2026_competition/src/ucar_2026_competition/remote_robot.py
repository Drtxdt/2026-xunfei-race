#!/usr/bin/env python3
"""Pure helpers for the Ubuntu-to-robot SSH deployment boundary."""

import base64
import ipaddress
import json
import math
import os
import posixpath
import re
import shlex
import subprocess


class RobotDeploymentError(ValueError):
    """Raised when the robot deployment cannot be started safely."""


LAUNCH_ARG_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def validate_competition_workspace(path):
    value = str(path or "").strip()
    if not value:
        raise RobotDeploymentError(
            "competition workspace is empty; export UCAR_COMPETITION_WS")
    workspace = os.path.realpath(os.path.abspath(os.path.expanduser(value)))
    required = (
        (os.path.join(workspace, ".git"), os.path.isdir, ".git"),
        (os.path.join(workspace, "devel", "setup.bash"), os.path.isfile, "devel/setup.bash"),
        (
            os.path.join(
                workspace, "src", "ucar_2026_competition", "launch",
                "physical_competition.launch"),
            os.path.isfile,
            "physical_competition.launch",
        ),
    )
    for candidate, predicate, label in required:
        if not predicate(candidate):
            raise RobotDeploymentError(
                "invalid competition workspace: missing {} at {}".format(label, candidate))
    return workspace


def repository_revision(workspace, runner=subprocess.run):
    revision = runner(
        ["git", "-C", workspace, "rev-parse", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if revision.returncode != 0:
        raise RobotDeploymentError(
            "cannot read competition repository revision: {}".format(
                revision.stderr.strip() or "git rev-parse failed"))
    status = runner(
        ["git", "-C", workspace, "status", "--porcelain"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if status.returncode != 0:
        raise RobotDeploymentError(
            "cannot inspect competition repository: {}".format(
                status.stderr.strip() or "git status failed"))
    if status.stdout.strip():
        raise RobotDeploymentError(
            "Ubuntu competition repository has uncommitted changes; commit or clean it before launch")
    return revision.stdout.strip()


def encode_launch_arguments(arguments):
    normalized = {}
    for name, value in dict(arguments or {}).items():
        if not LAUNCH_ARG_NAME.match(str(name)):
            raise RobotDeploymentError("invalid launch argument name: {}".format(name))
        if isinstance(value, bool):
            value = "true" if value else "false"
        else:
            value = str(value)
        if "\0" in value or "\n" in value or "\r" in value:
            raise RobotDeploymentError("invalid control character in launch argument {}".format(name))
        normalized[str(name)] = value
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


def decode_launch_arguments(payload):
    try:
        decoded = base64.urlsafe_b64decode(str(payload).encode("ascii"))
        values = json.loads(decoded.decode("utf-8"))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise RobotDeploymentError("invalid encoded launch arguments: {}".format(exc))
    if not isinstance(values, dict):
        raise RobotDeploymentError("encoded launch arguments must contain an object")
    result = {}
    for name, value in values.items():
        if not LAUNCH_ARG_NAME.match(str(name)) or not isinstance(value, str):
            raise RobotDeploymentError("invalid encoded launch argument")
        result[name] = value
    return result


def resolve_physical_launch_arguments(arguments, ssh_connection):
    result = dict(arguments or {})
    simulation_enabled = result.get("enable_simulation", "true").lower() == "true"
    external_bridge = result.pop("use_external_sim_bridge", "false").lower() == "true"
    if not simulation_enabled:
        return result
    if external_bridge:
        if not result.get("sim_bridge_host", "").strip():
            raise RobotDeploymentError("external simulation bridge host is missing")
        return result

    connection = str(ssh_connection or "").split()
    if len(connection) != 4:
        raise RobotDeploymentError(
            "SSH_CONNECTION is unavailable; cannot determine Ubuntu bridge address")
    try:
        client_address = ipaddress.ip_address(connection[0])
    except ValueError:
        raise RobotDeploymentError("SSH_CONNECTION contains an invalid client address")
    if client_address.is_loopback or client_address.is_unspecified:
        raise RobotDeploymentError("SSH client address is not reachable from the robot")
    result["sim_bridge_host"] = str(client_address)
    return result


def build_remote_agent_command(
        workspace, environment_file, revision, launch_arguments, startup_timeout=30.0):
    workspace = str(workspace or "").strip()
    environment_file = str(environment_file or "").strip()
    if not workspace.startswith("/"):
        raise RobotDeploymentError("robot workspace must be an absolute Linux path")
    if not environment_file.startswith("/"):
        raise RobotDeploymentError("robot environment file must be an absolute Linux path")
    encoded = encode_launch_arguments(launch_arguments)
    setup = posixpath.join(workspace, "devel", "setup.bash")
    agent_arguments = [
        "rosrun", "ucar_2026_competition", "robot_competition_agent.py",
        "--workspace", workspace,
        "--environment-file", environment_file,
        "--expected-revision", revision,
        "--launch-arguments", encoded,
        "--startup-timeout", str(float(startup_timeout)),
    ]
    script = " && ".join((
        "source /opt/ros/noetic/setup.bash",
        "source {}".format(shlex.quote(environment_file)),
        "source {}".format(shlex.quote(setup)),
        "export ROS_MASTER_URI=http://127.0.0.1:11311",
        "unset ROS_IP ROS_HOSTNAME",
        "exec {}".format(" ".join(shlex.quote(value) for value in agent_arguments)),
    ))
    return "exec /bin/bash --noprofile --norc -c {}".format(shlex.quote(script))


def ssh_base_command(target, connect_timeout, forward_x11=False):
    value = str(target or "").strip()
    if not value:
        raise RobotDeploymentError(
            "robot SSH target is empty; export UCAR_ROBOT_HOST as user@host")
    if value.startswith("-") or any(character.isspace() for character in value):
        raise RobotDeploymentError("invalid robot SSH target: {}".format(value))
    command = [
        "ssh", "-T",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout={}".format(max(1, int(math.ceil(float(connect_timeout))))),
        "-o", "ServerAliveInterval=2",
        "-o", "ServerAliveCountMax=3",
    ]
    if forward_x11:
        command.append("-Y")
    command.append(value)
    return command


def remote_preflight_script(workspace, environment_file, expected_revision):
    values = {
        "workspace": workspace,
        "environment_file": environment_file,
        "expected_revision": expected_revision,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(values, separators=(",", ":")).encode("utf-8")).decode("ascii")
    return "python3 - {} <<'PY'\n".format(shlex.quote(encoded)) + """\
import base64, json, os, stat, subprocess, sys
cfg = json.loads(base64.urlsafe_b64decode(sys.argv[1]).decode('utf-8'))
ws = cfg['workspace']
env_file = cfg['environment_file']
required = [
    '/opt/ros/noetic/setup.bash',
    os.path.join(ws, 'devel', 'setup.bash'),
    os.path.join(ws, 'src', 'ucar_2026_competition', 'launch', 'physical_competition.launch'),
    os.path.join(ws, 'devel', 'lib', 'ucar_2026_competition', 'robot_competition_agent.py'),
    env_file,
]
missing = [path for path in required if not os.path.isfile(path)]
if missing:
    raise SystemExit('missing remote file(s): ' + ', '.join(missing))
mode = stat.S_IMODE(os.stat(env_file).st_mode)
if mode != 0o600:
    raise SystemExit('%s must have mode 600 (actual %o)' % (env_file, mode))
revision = subprocess.check_output(['git', '-C', ws, 'rev-parse', 'HEAD'], text=True).strip()
status = subprocess.check_output(['git', '-C', ws, 'status', '--porcelain'], text=True).strip()
if status:
    raise SystemExit('robot competition repository has uncommitted changes')
if revision != cfg['expected_revision']:
    raise SystemExit('competition revision mismatch: ubuntu=%s robot=%s' % (
        cfg['expected_revision'], revision))
print('REMOTE_PREFLIGHT_OK ' + revision)
PY"""
