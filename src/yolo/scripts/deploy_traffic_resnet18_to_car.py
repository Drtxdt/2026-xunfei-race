#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deploy the scoped traffic classifier package and models over SSH."""

from __future__ import annotations

import argparse
import os
import posixpath
from pathlib import Path


PACKAGE_FILES = (
    "src/ucar_2026_traffic_light_rknn_test/CMakeLists.txt",
    "src/ucar_2026_traffic_light_rknn_test/package.xml",
    "src/ucar_2026_traffic_light_rknn_test/setup.py",
    "src/ucar_2026_traffic_light_rknn_test/README.md",
    "src/ucar_2026_traffic_light_rknn_test/config/traffic_light_rknn_test.yaml",
    "src/ucar_2026_traffic_light_rknn_test/launch/traffic_light_rknn_x11_speak_test.launch",
    "src/ucar_2026_traffic_light_rknn_test/scripts/check_traffic_light_rknn_test.py",
    "src/ucar_2026_traffic_light_rknn_test/scripts/traffic_light_rknn_test_node.py",
    "src/ucar_2026_traffic_light_rknn_test/src/ucar_2026_traffic_light_rknn_test/__init__.py",
    "src/ucar_2026_traffic_light_rknn_test/src/ucar_2026_traffic_light_rknn_test/classifier.py",
    "src/ucar_2026_traffic_light_rknn_test/test/test_traffic_light_classifier.py",
    "src/yolo/models/traffic_resnet18_rk3588_int8.rknn",
    "src/yolo/models/traffic_resnet18_rk3588_fp16.rknn",
)
EXECUTABLES = (
    "src/ucar_2026_traffic_light_rknn_test/scripts/check_traffic_light_rknn_test.py",
    "src/ucar_2026_traffic_light_rknn_test/scripts/traffic_light_rknn_test_node.py",
)


def execute(client, command):
    stdin, stdout, stderr = client.exec_command(command)
    del stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    print(out, end="" if not out or out.endswith("\n") else "\n")
    print(err, end="" if not err or err.endswith("\n") else "\n")
    if code != 0:
        raise RuntimeError("remote command failed ({}): {}".format(code, command))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--host", default="192.168.1.6")
    parser.add_argument("--user", default="ucar")
    parser.add_argument("--password-env", default="UCAR_SSH_PASSWORD")
    parser.add_argument("--remote-root", default="/home/ucar/2026-xunfei-race")
    parser.add_argument("--ros-setup", default="/home/ucar/ucar_ws/devel/setup.bash")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError("missing {}".format(args.password_env))
    local_root = Path(args.root).expanduser().resolve()
    for relative in PACKAGE_FILES:
        if not (local_root / relative).is_file():
            raise FileNotFoundError(str(local_root / relative))

    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=10)
    try:
        execute(client, "cd {} && git status --short".format(args.remote_root))
        sftp = client.open_sftp()
        try:
            directories = sorted(
                {posixpath.dirname(posixpath.join(args.remote_root, path)) for path in PACKAGE_FILES}
            )
            for directory in directories:
                execute(client, "mkdir -p '{}'".format(directory))
            for relative in PACKAGE_FILES:
                source = local_root / relative
                destination = posixpath.join(args.remote_root, relative.replace("\\", "/"))
                print("[UPLOAD]", relative)
                sftp.put(str(source), destination)
                if relative in EXECUTABLES:
                    sftp.chmod(destination, 0o755)
        finally:
            sftp.close()
        if not args.skip_build:
            execute(
                client,
                "bash -lc 'source {} && cd {} && "
                "catkin_make --force-cmake --pkg ucar_2026_traffic_light_rknn_test'".format(
                    args.ros_setup,
                    args.remote_root
                ),
            )
        execute(
            client,
            "bash -lc 'source {1} && source {0}/devel/setup.bash && "
            "python3 -m nose {0}/src/ucar_2026_traffic_light_rknn_test/test/test_traffic_light_classifier.py && "
            "roslaunch --files ucar_2026_traffic_light_rknn_test traffic_light_rknn_x11_speak_test.launch && "
            "rosrun ucar_2026_traffic_light_rknn_test check_traffic_light_rknn_test.py'".format(
                args.remote_root,
                args.ros_setup,
            ),
        )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
