#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert fixed-shape traffic ResNet18 ONNX to RK3588 RKNN."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


CLASS_NAMES = (
    "green_left",
    "green_right",
    "green_straight",
    "red_light",
    "background",
)
INPUT_SHAPE = [1, 3, 160, 320]
OUTPUT_SHAPE = [1, 5]
MEAN_VALUES = [[123.675, 116.28, 103.53]]
STD_VALUES = [[58.395, 57.12, 57.375]]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert traffic ResNet18 ONNX to RKNN.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_onnx(path):
    import onnx
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    input_dims = [item.dim_value for item in model.graph.input[0].type.tensor_type.shape.dim]
    output_dims = [item.dim_value for item in model.graph.output[0].type.tensor_type.shape.dim]
    opsets = [item.version for item in model.opset_import]
    if input_dims != INPUT_SHAPE:
        raise RuntimeError("unexpected ONNX input shape: {}".format(input_dims))
    if output_dims != OUTPUT_SHAPE:
        raise RuntimeError("unexpected ONNX output shape: {}".format(output_dims))
    if 12 not in opsets:
        raise RuntimeError("expected ONNX opset 12, got {}".format(opsets))
    return input_dims, output_dims, opsets


def make_toolkit14_compatible_onnx(source_path):
    """Remove no-op attributes which RKNN Toolkit 1.4 rejects.

    PyTorch writes ``dilations=[1, 1]`` on MaxPool for opset 12.  Unit
    dilation is the ONNX default and removing it does not change inference,
    but Toolkit 1.4's MaxPool importer treats the explicit attribute as
    unsupported.
    """
    import onnx

    model = onnx.load(str(source_path))
    changed = 0
    for node in model.graph.node:
        if node.op_type != "MaxPool":
            continue
        for index in range(len(node.attribute) - 1, -1, -1):
            attribute = node.attribute[index]
            if attribute.name != "dilations":
                continue
            values = list(attribute.ints)
            if any(value != 1 for value in values):
                raise RuntimeError(
                    "RKNN Toolkit 1.4 cannot import non-unit MaxPool dilation: {}".format(values)
                )
            del node.attribute[index]
            changed += 1
    fd, compatible_name = tempfile.mkstemp(prefix="traffic_resnet18_rknn_", suffix=".onnx")
    os.close(fd)
    compatible_path = Path(compatible_name)
    onnx.save(model, str(compatible_path))
    onnx.checker.check_model(onnx.load(str(compatible_path)))
    return compatible_path, changed


def main():
    args = parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve() if args.dataset else None
    if not onnx_path.is_file():
        raise FileNotFoundError(str(onnx_path))
    if args.quant and (dataset_path is None or not dataset_path.is_file()):
        raise RuntimeError("--quant requires --dataset")
    input_shape, output_shape, opsets = validate_onnx(onnx_path)

    compatible_onnx_path, compatibility_edits = make_toolkit14_compatible_onnx(onnx_path)
    if compatibility_edits:
        print("[INFO] removed {} no-op MaxPool dilations attribute(s)".format(compatibility_edits))

    from rknn.api import RKNN
    rknn = RKNN(verbose=bool(args.verbose))
    try:
        ret = rknn.config(
            target_platform=args.target,
            mean_values=MEAN_VALUES,
            std_values=STD_VALUES,
            quant_img_RGB2BGR=False,
            optimization_level=3,
        )
        if ret != 0:
            raise RuntimeError("rknn.config failed: {}".format(ret))
        ret = rknn.load_onnx(model=str(compatible_onnx_path))
        if ret != 0:
            raise RuntimeError("rknn.load_onnx failed: {}".format(ret))
        previous_cwd = os.getcwd()
        try:
            if args.quant:
                os.chdir(str(dataset_path.parent))
            ret = rknn.build(
                do_quantization=bool(args.quant),
                dataset=dataset_path.name if args.quant else None,
            )
        finally:
            os.chdir(previous_cwd)
        if ret != 0:
            raise RuntimeError("rknn.build failed: {}".format(ret))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError("rknn.export_rknn failed: {}".format(ret))
    finally:
        rknn.release()
        try:
            compatible_onnx_path.unlink()
        except OSError:
            pass

    metadata = {
        "model": "traffic_resnet18",
        "target": args.target,
        "quantized": bool(args.quant),
        "onnx": str(onnx_path),
        "rknn": str(output_path),
        "onnx_opsets": opsets,
        "toolkit14_compatibility_edits": compatibility_edits,
        "input_shape_nchw": input_shape,
        "runtime_input_shape_nhwc": [1, 160, 320, 3],
        "runtime_input_dtype": "uint8",
        "output_shape": output_shape,
        "classes": list(CLASS_NAMES),
        "preprocess": {
            "source_camera_mirrored": True,
            "runtime_horizontal_flip": True,
            "crop_top": 0.18,
            "crop_bottom": 0.72,
            "resize_width": 320,
            "resize_height": 160,
            "color": "RGB",
            "mean_values": MEAN_VALUES,
            "std_values": STD_VALUES,
            "normalization_embedded_in_rknn": True,
        },
        "calibration_dataset": str(dataset_path) if args.quant else None,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[OK] RKNN:", output_path)
    print("[OK] metadata:", metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
