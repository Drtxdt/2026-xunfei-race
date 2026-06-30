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
    dict(lang="ch", ocr_version="PP-OCRv5", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False),
    dict(lang="ch", ocr_version="PP-OCRv5", use_angle_cls=False, show_log=False),
    dict(lang="ch", use_angle_cls=False, show_log=False),
    dict(lang="ch"),
]
last = None
for kwargs in attempts:
    try:
        PaddleOCR(**kwargs)
        print("PaddleOCR init ok with:", kwargs)
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
    args = parser.parse_args()

    python = os.path.expanduser(os.path.expandvars(args.python))
    if not os.path.exists(python):
        print("[ERROR] python not found:", python)
        print("Create a local PaddleOCR env first, then pass --python or launch paddle_python:=...")
        return 2
    proc = subprocess.Popen([python, "-c", CHECK_CODE], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
