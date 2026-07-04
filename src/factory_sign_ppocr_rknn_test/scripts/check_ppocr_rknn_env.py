#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick diagnostics for factory_sign_ppocr_rknn_test."""

from __future__ import annotations

import argparse
import os
import sys


def package_dir() -> str:
    try:
        import rospkg

        return rospkg.RosPack().get_path("factory_sign_ppocr_rknn_test")
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve(path: str, default_name: str) -> str:
    if path:
        return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    return os.path.join(package_dir(), "models", default_name)


def status(ok: bool, message: str) -> None:
    print("[ OK ] {}".format(message) if ok else "[FAIL] {}".format(message))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-model", default="", help="ppocrv4_det.rknn path")
    parser.add_argument("--rec-model", default="", help="ppocrv4_rec.rknn path")
    parser.add_argument("--keys", default="", help="ppocr_keys_v1.txt path")
    parser.add_argument("--mode", default="rec_only", choices=("rec_only", "system"))
    args = parser.parse_args()

    root = package_dir()
    print("package:", root)
    det_model = resolve(args.det_model, "ppocrv4_det.rknn")
    rec_model = resolve(args.rec_model, "ppocrv4_rec.rknn")
    keys = resolve(args.keys, "ppocr_keys_v1.txt")

    ok = True
    status(os.path.isfile(rec_model), "rec model: {}".format(rec_model))
    ok = ok and os.path.isfile(rec_model)
    if args.mode == "system":
        status(os.path.isfile(det_model), "det model: {}".format(det_model))
        ok = ok and os.path.isfile(det_model)
    status(os.path.isfile(keys), "keys file: {}".format(keys))
    ok = ok and os.path.isfile(keys)

    try:
        from rknnlite.api import RKNNLite  # noqa: F401

        status(True, "rknnlite.api.RKNNLite importable")
    except Exception as exc:
        status(False, "rknnlite.api.RKNNLite import failed: {}".format(exc))
        return 2
    if not ok:
        print("")
        print("Place assets under {}/models or pass explicit paths.".format(root))
        return 2

    sys.path.insert(0, os.path.dirname(__file__))
    from factory_sign_ppocr_rknn_node import CTCLabelDecoder, RknnRuntime, repair_ros_logging

    repair_ros_logging()
    try:
        decoder = CTCLabelDecoder(keys)
        status(True, "keys loaded: {} characters".format(len(decoder.characters)))
    except Exception as exc:
        status(False, "keys load failed: {}".format(exc))
        return 2

    import numpy as np

    rec = None
    det = None
    try:
        rec = RknnRuntime(rec_model)
        rec_input = np.full((1, 48, 320, 3), 255, dtype=np.uint8)
        rec_outputs = rec.infer(rec_input)
        status(bool(rec_outputs), "rec dummy inference output shapes: {}".format([getattr(x, "shape", None) for x in rec_outputs or []]))
        if args.mode == "system":
            det = RknnRuntime(det_model)
            det_input = np.full((1, 480, 480, 3), 255, dtype=np.uint8)
            det_outputs = det.infer(det_input)
            status(bool(det_outputs), "det dummy inference output shapes: {}".format([getattr(x, "shape", None) for x in det_outputs or []]))
    except Exception as exc:
        status(False, "RKNN load/inference failed: {}".format(exc))
        return 2
    finally:
        if rec is not None:
            rec.release()
        if det is not None:
            det.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
