#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the RKNN PPOCR recognizer on one local image."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--det-model", default="")
    parser.add_argument("--rec-model", default="")
    parser.add_argument("--keys", default="")
    parser.add_argument("--mode", default="ppocr_rknn_rec_only", choices=("ppocr_rknn_rec_only", "ppocr_rknn_system"))
    parser.add_argument("--roi-scale", type=float, default=0.8)
    parser.add_argument("--rec-resize-mode", default="stretch", choices=("stretch", "pad"))
    parser.add_argument("--no-global-rec-candidates", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from factory_sign_ppocr_rknn_node import FactorySignKeywordClassifier, PPOCRRknnRecognizer, repair_ros_logging

    import cv2

    image_path = os.path.abspath(os.path.expanduser(os.path.expandvars(args.image)))
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] cannot read image:", image_path)
        return 2
    h, w = image.shape[:2]
    scale = min(max(args.roi_scale, 0.1), 1.0)
    roi_w = int(w * scale)
    roi_h = int(h * scale)
    x0 = max(0, (w - roi_w) // 2)
    y0 = max(0, (h - roi_h) // 2)
    roi = image[y0 : y0 + roi_h, x0 : x0 + roi_w]

    repair_ros_logging()
    recognizer = PPOCRRknnRecognizer(
        det_model_path=resolve(args.det_model, "ppocrv4_det.rknn"),
        rec_model_path=resolve(args.rec_model, "ppocrv4_rec.rknn"),
        keys_path=resolve(args.keys, "ppocr_keys_v1.txt"),
        mode=args.mode,
        min_score=0.0,
        det_binary_thresh=0.30,
        det_box_thresh=0.55,
        det_input_size=480,
        rec_image_height=48,
        rec_image_width=320,
        rec_resize_mode=args.rec_resize_mode,
        max_rec_crops=3,
        use_global_rec_candidates=not args.no_global_rec_candidates,
        box_padding_x=0.15,
        box_padding_y=0.35,
        global_fallback_crops=1,
        small_crop_retry=True,
        small_crop_max_height=20,
    )
    try:
        result = recognizer.recognize(roi)
        category = FactorySignKeywordClassifier().classify(result.raw_text)
        print("image:", image_path)
        print("mode:", args.mode)
        print("raw_text:", result.raw_text)
        print("category:", category)
        print("confidence:", "{:.4f}".format(result.confidence))
        print("elapsed_ms:", result.elapsed_ms, "det_ms:", result.det_ms, "rec_ms:", result.rec_ms)
        print("error:", result.error)
        for idx, item in enumerate(result.texts):
            print("text[{}]: score={:.4f} text={!r} box={}".format(idx, item.score, item.text, item.box))
    finally:
        recognizer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
