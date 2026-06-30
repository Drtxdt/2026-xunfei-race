#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check the local PaddleOCR runtime used by factory_sign_ppocr_test."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


CHECK_CODE = r"""
import sys
print("python:", sys.executable)
print("version:", sys.version.replace("\n", " "))
try:
    import paddle
    print("paddle:", getattr(paddle, "__version__", "unknown"))
except Exception as exc:
    print("paddle import failed:", exc)
    raise
try:
    import paddleocr
    print("paddleocr:", getattr(paddleocr, "__version__", "unknown"))
    from paddleocr import PaddleOCR
except Exception as exc:
    print("paddleocr import failed:", exc)
    raise
attempts = [
    dict(lang="ch", ocr_version="PP-OCRv5", text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="PP-OCRv5_mobile_rec", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False),
    dict(lang="ch", ocr_version="PP-OCRv5", text_detection_model_name="PP-OCRv5_mobile_det", text_recognition_model_name="PP-OCRv5_mobile_rec", use_angle_cls=False, show_log=False),
    dict(lang="ch", ocr_version="PP-OCRv5", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False),
    dict(lang="ch", ocr_version="PP-OCRv5", use_angle_cls=False, show_log=False),
    dict(lang="ch", use_angle_cls=False, show_log=False),
    dict(lang="ch"),
]
last = None
for kwargs in attempts:
    try:
        ocr = PaddleOCR(**kwargs)
        print("PaddleOCR init ok with:", kwargs)
        if not hasattr(ocr, "ocr"):
            print("PaddleOCR legacy ocr() API not found; use launch ocr_api:=predict only for diagnosis")
            raise SystemExit(3)
        import cv2
        import numpy as np

        img = np.full((120, 320, 3), 255, dtype=np.uint8)
        cv2.putText(img, "food", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
        started = __import__("time").time()
        result = ocr.ocr(img, cls=False)
        elapsed_ms = int((__import__("time").time() - started) * 1000)
        print("PaddleOCR legacy ocr smoke ok: type={} elapsed_ms={}".format(type(result).__name__, elapsed_ms))
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc:
        last = exc
print("PaddleOCR init failed:", last)
raise SystemExit(2)
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="/home/ucar/ppocrv6_env/bin/python3", help="local PaddleOCR Python interpreter")
    parser.add_argument("--timeout-sec", type=float, default=180.0, help="maximum seconds for init plus one OCR smoke request")
    args = parser.parse_args()

    python = os.path.expanduser(os.path.expandvars(args.python))
    if not os.path.exists(python):
        print("[ERROR] python not found:", python)
        print("Create a local PaddleOCR env first, then pass --python or launch paddle_python:=...")
        return 2
    proc = subprocess.Popen([python, "-c", CHECK_CODE], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    try:
        output, _ = proc.communicate(timeout=args.timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        if output:
            sys.stdout.write(output)
        print("[ERROR] PaddleOCR init/OCR smoke timed out after {:.1f}s".format(args.timeout_sec))
        print("[ERROR] This means the local PaddleOCR Python inference path is hanging before ROS is involved.")
        return 124
    if output:
        sys.stdout.write(output)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
