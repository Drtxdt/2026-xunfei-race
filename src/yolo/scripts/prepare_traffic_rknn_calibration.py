#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a train-only, class-balanced RKNN INT8 calibration set."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


CLASS_NAMES = (
    "green_left",
    "green_right",
    "green_straight",
    "red_light",
    "background",
)
TRAIN_SESSIONS = (
    "session_01_normal_light",
    "session_02_bright",
    "session_03_dim",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare RKNN traffic calibration images.")
    parser.add_argument("--data", required=True, help="Corrected classification dataset root.")
    parser.add_argument("--output", required=True, help="New calibration directory.")
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--crop-top", type=float, default=0.18)
    parser.add_argument("--crop-bottom", type=float, default=0.72)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--input-height", type=int, default=160)
    return parser.parse_args()


def distribute(total, buckets):
    base, remainder = divmod(total, buckets)
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def select_evenly(paths, count):
    if count <= 0:
        return []
    if len(paths) < count:
        raise ValueError("need {} images, found {}".format(count, len(paths)))
    if count == 1:
        return [paths[len(paths) // 2]]
    indices = [int(round(index * (len(paths) - 1) / float(count - 1))) for index in range(count)]
    return [paths[index] for index in indices]


def main():
    args = parse_args()
    data_root = Path(args.data).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if not (data_root / "train").is_dir():
        raise RuntimeError("missing train directory: {}".format(data_root / "train"))
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("output must be absent or empty: {}".format(output_root))
    if not (0.0 <= args.crop_top < args.crop_bottom <= 1.0):
        raise ValueError("invalid crop range")
    output_root.mkdir(parents=True, exist_ok=True)
    image_root = output_root / "images"
    image_root.mkdir()

    quotas = distribute(args.per_class, len(TRAIN_SESSIONS))
    manifest = []
    counts = defaultdict(int)
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_dir = data_root / "train" / class_name
        all_paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for session, quota in zip(TRAIN_SESSIONS, quotas):
            session_paths = [path for path in all_paths if path.name.startswith(session + "__")]
            for source in select_evenly(session_paths, quota):
                with Image.open(str(source)) as image:
                    image = image.convert("RGB")
                    width, height = image.size
                    y1 = max(0, min(height - 1, int(round(height * args.crop_top))))
                    y2 = max(y1 + 1, min(height, int(round(height * args.crop_bottom))))
                    prepared = image.crop((0, y1, width, y2)).resize(
                        (args.input_width, args.input_height), Image.BILINEAR
                    )
                    destination = image_root / (
                        "{:02d}_{}__{}.png".format(class_id, class_name, source.stem)
                    )
                    prepared.save(str(destination), format="PNG")
                manifest.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "session": session,
                    "source": str(source),
                    "prepared": str(destination),
                })
                counts[class_name] += 1

    expected = args.per_class * len(CLASS_NAMES)
    if len(manifest) != expected or any(counts[name] != args.per_class for name in CLASS_NAMES):
        raise RuntimeError("calibration count mismatch: {}".format(dict(counts)))
    dataset_path = output_root / "dataset.txt"
    dataset_path.write_text(
        "\n".join(
            Path(item["prepared"]).resolve().relative_to(output_root).as_posix()
            for item in manifest
        ) + "\n",
        encoding="utf-8",
    )
    report = {
        "source": str(data_root),
        "split": "train only",
        "classes": list(CLASS_NAMES),
        "per_class": args.per_class,
        "total": len(manifest),
        "session_quotas_per_class": dict(zip(TRAIN_SESSIONS, quotas)),
        "crop_top": args.crop_top,
        "crop_bottom": args.crop_bottom,
        "input_width": args.input_width,
        "input_height": args.input_height,
        "color": "RGB image files",
        "manifest": manifest,
    }
    (output_root / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[OK] calibration images:", len(manifest))
    print("[OK] dataset list:", dataset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
