#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick runtime diagnostics for the traffic-light speech test."""

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
    print("=== Traffic light speech test diagnostics ===")

    yolo_path = run(["rospack", "find", "yolo"])
    if not yolo_path or yolo_path.startswith("ERROR"):
        print("[FAIL] rospack find yolo: %s" % (yolo_path or "empty"))
        return 1
    print("[ OK ] yolo package: %s" % yolo_path)

    model_path = os.path.join(yolo_path, "models", "best.pt")
    if os.path.isfile(model_path):
        print("[ OK ] model exists: %s" % model_path)
    else:
        print("[FAIL] model missing: %s" % model_path)

    nodes = run(["rosnode", "list"])
    print("[INFO] traffic_light node present: %s" % ("yes" if "traffic_light_inference" in nodes else "no"))
    print("[INFO] voice_speak_node present: %s" % ("yes" if "voice_speak_node" in nodes else "no"))

    topics = run(["rostopic", "list"])
    for topic in ["/usb_cam/image_raw", "/traffic_light/detections", "/traffic_light/status", "/speak"]:
        print("[INFO] topic %-32s %s" % (topic, "yes" if topic in topics.splitlines() else "no"))

    status = run(["rostopic", "echo", "-n1", "/traffic_light/status"], timeout=3.0)
    if status:
        print("[INFO] /traffic_light/status:")
        print(status)
    else:
        print("[WARN] no /traffic_light/status sample")

    det = run(["rostopic", "echo", "-n1", "/traffic_light/detections"], timeout=3.0)
    if det:
        print("[INFO] /traffic_light/detections sample:")
        print(det[:1000])
    else:
        print("[WARN] no /traffic_light/detections sample")

    return 0


if __name__ == "__main__":
    sys.exit(main())
