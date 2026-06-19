#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick runtime diagnostics for the factory-sign RKNN test."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List


def run(cmd: List[str], timeout: float = 4.0) -> str:
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
            universal_newlines=True,
        )
    except Exception as exc:
        return "ERROR: %s" % exc
    return completed.stdout.strip()


def print_check(label: str, ok: bool, detail: str = "") -> None:
    prefix = "[ OK ]" if ok else "[FAIL]"
    if detail:
        print("%s %-30s %s" % (prefix, label + ":", detail))
    else:
        print("%s %s" % (prefix, label))


def main() -> int:
    print("=== RKNN factory sign test diagnostics ===")

    yolo_path = run(["rospack", "find", "yolo"])
    if not yolo_path or yolo_path.startswith("ERROR"):
        print_check("rospack find yolo", False, yolo_path or "empty")
        return 1
    print_check("yolo package", True, yolo_path)

    model_path = os.path.join(yolo_path, "models", "factory_sign_3cls_fp16.rknn")
    print_check("factory sign model", os.path.isfile(model_path), model_path)

    alt_model_path = os.path.join(yolo_path, "models", "best.rknn")
    if os.path.isfile(alt_model_path):
        print("[INFO] alternate model present: %s" % alt_model_path)

    config_path = os.path.join(yolo_path, "config", "factory_sign_rknn_test.yaml")
    launch_path = os.path.join(yolo_path, "launch", "factory_sign_rknn_test.launch")
    print_check("config file", os.path.isfile(config_path), config_path)
    print_check("launch file", os.path.isfile(launch_path), launch_path)

    try:
        import rknnlite.api  # noqa: F401
        print_check("rknnlite.api import", True)
    except Exception as exc:
        print_check("rknnlite.api import", False, str(exc))

    nodes = run(["rosnode", "list"])
    for node in [
        "/factory_sign_rknn_test_node",
        "/factory_sign_mjpeg_server",
        "/voice_speak_node",
        "/competition_announcer",
        "/factory_sign_rknn_x11_viewer",
    ]:
        print("[INFO] node %-36s %s" % (node, "yes" if node in nodes.splitlines() else "no"))

    topics = run(["rostopic", "list"])
    for topic in [
        "/usb_cam/image_raw",
        "/factory_sign_rknn_test/detections",
        "/factory_sign_rknn_test/debug_image",
        "/factory_sign_rknn_test/status",
        "/speak",
    ]:
        print("[INFO] topic %-38s %s" % (topic, "yes" if topic in topics.splitlines() else "no"))

    status = run(["rostopic", "echo", "-n1", "/factory_sign_rknn_test/status"], timeout=2.0)
    if status:
        print("[INFO] status sample:")
        print(status[:500])
    else:
        print("[WARN] no status sample")

    sample = run(["rostopic", "echo", "-n1", "/factory_sign_rknn_test/detections"], timeout=3.0)
    if sample:
        print("[INFO] detection sample:")
        print(sample[:1600])
    else:
        print("[WARN] no detection sample")

    return 0


if __name__ == "__main__":
    sys.exit(main())
