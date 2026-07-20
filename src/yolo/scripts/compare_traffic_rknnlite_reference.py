#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run an exported RKNN on RK3588 and gate it against ONNX reference CSV."""

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


def stable_softmax(logits):
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size != 5 or not np.all(np.isfinite(values)):
        raise RuntimeError("invalid RKNN logits shape={} values={}".format(values.shape, values))
    exp_values = np.exp(values - np.max(values))
    return (exp_values / np.sum(exp_values)).astype(np.float32)


def load_rgb(path):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cannot decode {}".format(path))
    height, width = bgr.shape[:2]
    y1 = max(0, min(height - 1, int(round(height * 0.18))))
    y2 = max(y1 + 1, min(height, int(round(height * 0.72))))
    roi_rgb = cv2.cvtColor(bgr[y1:y2, :width], cv2.COLOR_BGR2RGB)
    return np.asarray(Image.fromarray(roi_rgb).resize((320, 160), Image.BILINEAR))


def decision(probabilities, confidence, margin):
    order = np.argsort(probabilities)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    score = float(probabilities[top1])
    gap = score - float(probabilities[top2])
    return top1, score, gap, score >= confidence and gap >= margin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--rknn", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--margin", type=float, default=0.12)
    args = parser.parse_args()

    from rknnlite.api import RKNNLite

    data_root = Path(args.data).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with Path(args.reference).open("r", encoding="utf-8", newline="") as handle:
        references = list(csv.DictReader(handle))

    rknn = RKNNLite()
    if rknn.load_rknn(str(Path(args.rknn).expanduser().resolve())) != 0:
        raise RuntimeError("load_rknn failed")
    if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        raise RuntimeError("init_runtime failed")

    rows = []
    output_dtype = None
    try:
        for index, reference in enumerate(references, 1):
            rgb = load_rgb(data_root / reference["path"])
            # RKNNLite 1.4 requires an explicit four-dimensional NHWC batch.
            batch = np.ascontiguousarray(rgb[None, ...], dtype=np.uint8)
            outputs = rknn.inference(inputs=[batch])
            if outputs is None or len(outputs) == 0:
                raise RuntimeError("RKNN returned no output for {}".format(reference["path"]))
            output = np.asarray(outputs[0])
            output_dtype = str(output.dtype)
            probabilities = stable_softmax(output)
            onnx_probabilities = np.asarray(json.loads(reference["onnx_probabilities"]), dtype=np.float32)
            top1, confidence, margin, valid = decision(
                probabilities, args.confidence, args.margin
            )
            rows.append(
                {
                    **reference,
                    "rknn_top1": top1,
                    "rknn_class": CLASS_NAMES[top1],
                    "rknn_confidence": confidence,
                    "rknn_margin": margin,
                    "rknn_accepted": int(valid),
                    "probability_mae": float(np.mean(np.abs(onnx_probabilities - probabilities))),
                    "rknn_probabilities": json.dumps(probabilities.tolist(), separators=(",", ":")),
                    "rknn_output_dtype": output_dtype,
                }
            )
            if index % 100 == 0 or index == len(references):
                print("[INFO] RKNNLite {}/{}".format(index, len(references)))
    finally:
        rknn.release()

    metrics = {}
    unsafe = defaultdict(int)
    for split in ("val", "test"):
        subset = [row for row in rows if row["split"] == split]
        accepted_rows = [row for row in subset if bool(int(row["rknn_accepted"]))]
        per_class = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            own = [row for row in subset if int(row["actual_id"]) == class_id]
            own_accepted = [row for row in own if bool(int(row["rknn_accepted"]))]
            per_class[class_name] = {
                "total": len(own),
                "accepted": len(own_accepted),
                "coverage": len(own_accepted) / float(max(1, len(own))),
                "accepted_correct": sum(int(row["rknn_top1"]) == class_id for row in own_accepted),
            }
        metrics[split] = {
            "total": len(subset),
            "top1_agreement_with_onnx": sum(
                int(row["onnx_top1"]) == int(row["rknn_top1"]) for row in subset
            ) / float(max(1, len(subset))),
            "coverage": len(accepted_rows) / float(max(1, len(subset))),
            "accepted_accuracy": sum(
                int(row["actual_id"]) == int(row["rknn_top1"]) for row in accepted_rows
            ) / float(max(1, len(accepted_rows))),
            "per_class": per_class,
        }
        for row in accepted_rows:
            actual, predicted = int(row["actual_id"]), int(row["rknn_top1"])
            if actual in (0, 1) and predicted in (0, 1) and actual != predicted:
                unsafe[split + "_left_right_swap"] += 1
            if actual == 3 and predicted in (0, 1, 2):
                unsafe[split + "_red_as_green"] += 1

    gates = {
        "test_agreement_at_least_99_5": metrics["test"]["top1_agreement_with_onnx"] >= 0.995,
        "test_accepted_accuracy_100": metrics["test"]["accepted_accuracy"] == 1.0,
        "test_coverage_100": metrics["test"]["coverage"] == 1.0,
        "val_accepted_accuracy_100": metrics["val"]["accepted_accuracy"] == 1.0,
        "val_coverage_at_least_83_6": metrics["val"]["coverage"] >= 0.836,
        "val_left_coverage_at_least_64": metrics["val"]["per_class"]["green_left"]["coverage"] >= 0.64,
        "val_right_coverage_at_least_64": metrics["val"]["per_class"]["green_right"]["coverage"] >= 0.64,
        "no_unsafe_confusions": sum(unsafe.values()) == 0,
    }
    report = {
        "rknn": str(args.rknn),
        "rknn_output_dtype": output_dtype,
        "confidence_threshold": args.confidence,
        "margin_threshold": args.margin,
        "probability_mae_mean": float(np.mean([row["probability_mae"] for row in rows])),
        "probability_mae_max": float(np.max([row["probability_mae"] for row in rows])),
        "unsafe_confusions": dict(unsafe),
        "metrics": metrics,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    csv_path = output_root / "per_image.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report_path = output_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
