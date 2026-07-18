#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Correct mirrored captures and build a session-separated classification dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = (
    "green_left",
    "green_right",
    "green_straight",
    "red_light",
    "background",
)
SESSION_SPLITS = {
    "session_01_normal_light": "train",
    "session_02_bright": "train",
    "session_03_dim": "train",
    "session_04_background_changed": "val",
    "session_05_final_field": "test",
}
EXPECTED_PER_CLASS = {
    "session_01_normal_light": 120,
    "session_02_bright": 120,
    "session_03_dim": 110,
    "session_04_background_changed": 100,
    "session_05_final_field": 50,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_image(path):
    """Read an image through NumPy so Windows Unicode paths work with OpenCV."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def write_jpeg(path, image, quality=95):
    """Write JPEG through NumPy so Windows Unicode paths work with OpenCV."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return False
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flip mirrored traffic-light images once and organize a 5-class dataset."
    )
    parser.add_argument("--source", required=True, help="Raw five-session capture root.")
    parser.add_argument("--output", required=True, help="New corrected dataset root; must be empty.")
    parser.add_argument("--crop-top", type=float, default=0.18)
    parser.add_argument("--crop-bottom", type=float, default=0.72)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--input-height", type=int, default=160)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit counts other than the planned 120/120/110/100/50 per class.",
    )
    return parser.parse_args()


def image_paths(directory):
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vertical_band(image, top, bottom, output_size):
    height, width = image.shape[:2]
    y1 = max(0, min(height - 1, int(round(height * top))))
    y2 = max(y1 + 1, min(height, int(round(height * bottom))))
    cropped = image[y1:y2, 0:width]
    return cv2.resize(cropped, output_size, interpolation=cv2.INTER_AREA)


def make_preview(samples, output_path, crop_top, crop_bottom, input_size):
    if not samples:
        return
    tile_width, tile_height = input_size
    columns = 4
    rows = (len(samples) + columns - 1) // columns
    sheet = np.full((rows * (tile_height + 24), columns * tile_width, 3), 32, np.uint8)
    for index, (image_path, label) in enumerate(samples):
        image = read_image(image_path)
        if image is None:
            raise RuntimeError("cannot decode preview image: {}".format(image_path))
        tile = vertical_band(image, crop_top, crop_bottom, input_size)
        row, column = divmod(index, columns)
        x = column * tile_width
        y = row * (tile_height + 24)
        sheet[y:y + tile_height, x:x + tile_width] = tile
        cv2.putText(
            sheet,
            label[:44],
            (x + 4, y + tile_height + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if not write_jpeg(output_path, sheet, 95):
        raise RuntimeError("failed to write preview: {}".format(output_path))


def main():
    args = parse_args()
    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if not source_root.is_dir():
        print("[FAIL] source dataset does not exist:", source_root)
        return 2
    if output_root.exists() and any(output_root.iterdir()):
        print("[FAIL] output must be absent or empty; refusing to overwrite:", output_root)
        return 2
    if not (0.0 <= args.crop_top < args.crop_bottom <= 1.0):
        print("[FAIL] crop range must satisfy 0 <= top < bottom <= 1")
        return 2
    if args.input_width <= 0 or args.input_height <= 0:
        print("[FAIL] input dimensions must be positive")
        return 2

    errors = []
    planned = []
    for session, split in SESSION_SPLITS.items():
        for class_name in CLASS_NAMES:
            directory = source_root / session / class_name
            paths = image_paths(directory)
            expected = EXPECTED_PER_CLASS[session]
            if not directory.is_dir():
                errors.append("missing directory: {}".format(directory))
            if not args.allow_incomplete and len(paths) != expected:
                errors.append(
                    "{} / {} has {} images, expected {}".format(
                        session, class_name, len(paths), expected
                    )
                )
            for path in paths:
                planned.append((session, split, class_name, path))
    if errors:
        print("[FAIL] source validation failed:")
        for message in errors:
            print(" -", message)
        return 1
    if not planned:
        print("[FAIL] no images found")
        return 1

    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)
    preview_root = output_root / "crop_preview"
    preview_root.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(lambda: defaultdict(int))
    preview_samples = defaultdict(list)
    preview_counts = defaultdict(int)
    manifest_rows = []
    seen_destinations = set()
    for index, (session, split, class_name, source_path) in enumerate(planned, 1):
        image = read_image(source_path)
        if image is None:
            raise RuntimeError("cannot decode image: {}".format(source_path))
        height, width = image.shape[:2]
        if (width, height) != (640, 480):
            raise RuntimeError(
                "{} has size {}x{}, expected 640x480".format(source_path, width, height)
            )

        # Exactly one correction: raw capture is mirrored, output follows physical direction.
        corrected = cv2.flip(image, 1)
        destination_name = "{}__{}.jpg".format(session, source_path.stem)
        destination = output_root / split / class_name / destination_name
        if str(destination).lower() in seen_destinations:
            raise RuntimeError("destination collision: {}".format(destination))
        seen_destinations.add(str(destination).lower())
        if not write_jpeg(destination, corrected, args.jpeg_quality):
            raise RuntimeError("failed to write: {}".format(destination))

        counts[split][class_name] += 1
        manifest_rows.append({
            "split": split,
            "session": session,
            "class_name": class_name,
            "source": str(source_path),
            "destination": str(destination),
            "source_sha256": sha256(source_path),
            "destination_sha256": sha256(destination),
            "horizontal_flip_applied": "true",
            "width": width,
            "height": height,
        })
        preview_key = (class_name, session)
        if preview_counts[preview_key] < 3:
            preview_samples[class_name].append((destination, "{} {}".format(split, destination_name)))
            preview_counts[preview_key] += 1
        if index % 250 == 0 or index == len(planned):
            print("[INFO] processed {}/{}".format(index, len(planned)))

    with (output_root / "dataset_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    preprocess = {
        "model_family": "resnet18",
        "classes": list(CLASS_NAMES),
        "source_camera_is_mirrored": True,
        "dataset_horizontal_flip_applied": True,
        "runtime_horizontal_flip_required": True,
        "crop": {
            "type": "full_width_vertical_band",
            "top_normalized": args.crop_top,
            "bottom_normalized": args.crop_bottom,
        },
        "input": {
            "width": args.input_width,
            "height": args.input_height,
            "color": "RGB",
        },
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "random_horizontal_flip": False,
    }
    (output_root / "preprocess.json").write_text(
        json.dumps(preprocess, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "source": str(source_root),
        "output": str(output_root),
        "total_images": len(manifest_rows),
        "counts": {split: dict(values) for split, values in counts.items()},
        "classes": list(CLASS_NAMES),
        "raw_source_preserved": True,
        "horizontal_flip_applied_exactly_once": True,
    }
    (output_root / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    for class_name, samples in preview_samples.items():
        make_preview(
            samples,
            preview_root / (class_name + ".jpg"),
            args.crop_top,
            args.crop_bottom,
            (args.input_width, args.input_height),
        )

    print("[OK] corrected classification dataset:", output_root)
    print("[INFO] images:", len(manifest_rows))
    print("[INFO] original raw dataset was not modified")
    print("[INFO] inspect crop previews:", preview_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
