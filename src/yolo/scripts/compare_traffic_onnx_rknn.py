#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare ONNX and RKNN traffic classifiers over full val/test splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CLASS_NAMES = (
    "green_left",
    "green_right",
    "green_straight",
    "red_light",
    "background",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Gate traffic RKNN against ONNX.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--rknn", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--margin", type=float, default=0.12)
    return parser.parse_args()


def stable_softmax(logits):
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != len(CLASS_NAMES) or not np.all(np.isfinite(values)):
        raise RuntimeError("invalid logits: shape={} values={}".format(values.shape, values))
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return (exp / np.sum(exp)).astype(np.float32)


def load_rgb(path):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cannot decode {}".format(path))
    height, width = bgr.shape[:2]
    y1 = max(0, min(height - 1, int(round(height * 0.18))))
    y2 = max(y1 + 1, min(height, int(round(height * 0.72))))
    roi_rgb = cv2.cvtColor(bgr[y1:y2, 0:width], cv2.COLOR_BGR2RGB)
    return np.asarray(Image.fromarray(roi_rgb).resize((320, 160), Image.BILINEAR))


def enumerate_samples(root):
    samples = []
    for split in ("val", "test"):
        for class_id, class_name in enumerate(CLASS_NAMES):
            directory = root / split / class_name
            paths = sorted(
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            samples.extend((split, class_id, class_name, path) for path in paths)
    return samples


def accepted(probabilities, confidence, margin):
    order = np.argsort(probabilities)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    score = float(probabilities[top1])
    gap = score - float(probabilities[top2])
    # Evaluation coverage includes a confident background classification. The
    # ROS decision layer separately rejects background from active consensus.
    valid = score >= confidence and gap >= margin
    return top1, score, gap, valid


def summarize(rows, prefix, confidence, margin):
    summary = {}
    for split in ("val", "test"):
        subset = [row for row in rows if row["split"] == split]
        agreement = sum(row["onnx_top1"] == row[prefix + "_top1"] for row in subset)
        accepted_rows = [row for row in subset if row[prefix + "_accepted"]]
        accepted_correct = sum(row[prefix + "_top1"] == row["actual_id"] for row in accepted_rows)
        per_class = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            own = [row for row in subset if row["actual_id"] == class_id]
            own_accepted = [row for row in own if row[prefix + "_accepted"]]
            per_class[class_name] = {
                "total": len(own),
                "accepted": len(own_accepted),
                "coverage": len(own_accepted) / float(max(1, len(own))),
                "accepted_correct": sum(
                    row[prefix + "_top1"] == class_id for row in own_accepted
                ),
            }
        summary[split] = {
            "total": len(subset),
            "top1_agreement_with_onnx": agreement / float(max(1, len(subset))),
            "accepted": len(accepted_rows),
            "coverage": len(accepted_rows) / float(max(1, len(subset))),
            "accepted_accuracy": accepted_correct / float(max(1, len(accepted_rows))),
            "per_class": per_class,
        }
    return summary


def main():
    args = parse_args()
    data_root = Path(args.data).expanduser().resolve()
    onnx_path = Path(args.onnx).expanduser().resolve()
    rknn_path = Path(args.rknn).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    import onnxruntime as ort
    from rknn.api import RKNN
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError("load_rknn failed: {}".format(ret))
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError("init_runtime failed: {}".format(ret))

    rows = []
    try:
        samples = enumerate_samples(data_root)
        for index, (split, actual_id, actual_name, path) in enumerate(samples, 1):
            rgb = load_rgb(path)
            normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
            nchw = np.transpose(normalized, (2, 0, 1))[None, ...]
            onnx_logits = session.run(None, {input_name: nchw})[0]
            onnx_probs = stable_softmax(onnx_logits)
            outputs = rknn.inference(inputs=[rgb])
            if not outputs:
                raise RuntimeError("RKNN returned no output for {}".format(path))
            rknn_probs = stable_softmax(outputs[0])
            onnx_top1, onnx_conf, onnx_margin, onnx_valid = accepted(
                onnx_probs, args.confidence, args.margin
            )
            rknn_top1, rknn_conf, rknn_margin, rknn_valid = accepted(
                rknn_probs, args.confidence, args.margin
            )
            row = {
                "split": split,
                "path": str(path),
                "actual_id": actual_id,
                "actual_name": actual_name,
                "onnx_top1": onnx_top1,
                "onnx_class": CLASS_NAMES[onnx_top1],
                "onnx_confidence": onnx_conf,
                "onnx_margin": onnx_margin,
                "onnx_accepted": onnx_valid,
                "rknn_top1": rknn_top1,
                "rknn_class": CLASS_NAMES[rknn_top1],
                "rknn_confidence": rknn_conf,
                "rknn_margin": rknn_margin,
                "rknn_accepted": rknn_valid,
                "probability_mae": float(np.mean(np.abs(onnx_probs - rknn_probs))),
                "onnx_probabilities": json.dumps(onnx_probs.tolist()),
                "rknn_probabilities": json.dumps(rknn_probs.tolist()),
            }
            rows.append(row)
            if index % 100 == 0 or index == len(samples):
                print("[INFO] compared {}/{}".format(index, len(samples)))
    finally:
        rknn.release()

    csv_path = output_root / (rknn_path.stem + "_per_image.csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "onnx": str(onnx_path),
        "rknn": str(rknn_path),
        "confidence_threshold": args.confidence,
        "margin_threshold": args.margin,
        "probability_mae_mean": float(np.mean([row["probability_mae"] for row in rows])),
        "probability_mae_max": float(np.max([row["probability_mae"] for row in rows])),
        "metrics": summarize(rows, "rknn", args.confidence, args.margin),
    }
    report_path = output_root / (rknn_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[OK]", csv_path)
    print("[OK]", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
