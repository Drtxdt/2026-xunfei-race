#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export per-image ONNX probabilities for RK3588 RKNNLite gating."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from compare_traffic_onnx_rknn import (
    CLASS_NAMES,
    MEAN,
    STD,
    accepted,
    enumerate_samples,
    load_rgb,
    stable_softmax,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--margin", type=float, default=0.12)
    args = parser.parse_args()

    data_root = Path(args.data).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(Path(args.onnx).expanduser().resolve()), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    rows = []
    samples = enumerate_samples(data_root)
    for index, (split, actual_id, actual_name, path) in enumerate(samples, 1):
        rgb = load_rgb(path)
        normalized = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        nchw = np.transpose(normalized, (2, 0, 1))[None, ...]
        logits = session.run(None, {input_name: nchw})[0]
        probabilities = stable_softmax(logits)
        top1, confidence, margin, valid = accepted(
            probabilities, args.confidence, args.margin
        )
        rows.append(
            {
                "split": split,
                "path": path.relative_to(data_root).as_posix(),
                "actual_id": actual_id,
                "actual_name": actual_name,
                "onnx_top1": top1,
                "onnx_class": CLASS_NAMES[top1],
                "onnx_confidence": confidence,
                "onnx_margin": margin,
                "onnx_accepted": int(valid),
                "onnx_probabilities": json.dumps(probabilities.tolist(), separators=(",", ":")),
            }
        )
        if index % 100 == 0 or index == len(samples):
            print("[INFO] ONNX reference {}/{}".format(index, len(samples)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("[OK]", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
