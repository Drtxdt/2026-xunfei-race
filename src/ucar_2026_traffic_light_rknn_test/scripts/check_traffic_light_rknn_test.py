#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick runtime diagnostics for the RKNN traffic-light test."""

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


def main() -> int:
    print("=== RKNN traffic light test diagnostics ===")
    failed = False

    yolo_path = run(["rospack", "find", "yolo"])
    if not yolo_path or yolo_path.startswith("ERROR"):
        print("[FAIL] rospack find yolo: %s" % (yolo_path or "empty"))
        return 1
    print("[ OK ] yolo package: %s" % yolo_path)

    model_path = os.path.join(
        yolo_path, "models", "traffic_resnet18_rk3588_int8.rknn"
    )
    if os.path.isfile(model_path):
        size_mib = os.path.getsize(model_path) / float(1024 * 1024)
        if size_mib < 1.0:
            print("[FAIL] RKNN model is unexpectedly small: %.2f MiB" % size_mib)
            failed = True
        else:
            print("[ OK ] RKNN model exists: %s (%.2f MiB)" % (model_path, size_mib))
    else:
        print("[FAIL] RKNN model missing: %s" % model_path)
        failed = True

    try:
        import rknnlite.api  # noqa: F401
        print("[ OK ] rknnlite.api import works")
    except Exception as exc:
        print("[FAIL] rknnlite.api import failed: %s" % exc)
        failed = True

    nodes = run(["rosnode", "list"])
    for node in [
        "traffic_light_rknn_test_node",
        "voice_speak_node",
        "competition_announcer",
    ]:
        print("[INFO] node %-34s %s" % (node, "yes" if node in nodes else "no"))

    topics = run(["rostopic", "list"])
    for topic in [
        "/usb_cam/image_raw",
        "/traffic_light_rknn_test/detections",
        "/traffic_light_rknn_test/debug_image",
        "/traffic_light_rknn_test/status",
        "/speak",
    ]:
        print("[INFO] topic %-38s %s" % (topic, "yes" if topic in topics.splitlines() else "no"))

    sample = run(["rostopic", "echo", "-n1", "/traffic_light_rknn_test/detections"], timeout=3.0)
    if sample:
        print("[INFO] detection sample:")
        print(sample[:1200])
    else:
        print("[WARN] no detection sample")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
