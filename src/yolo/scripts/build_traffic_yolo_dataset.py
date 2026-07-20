#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate LabelImg annotations and build a leak-free YOLOv5 dataset by session."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from yolo_tools.traffic_dataset import (
    CAPTURE_CLASSES,
    CLASS_NAMES,
    CLASS_TO_ID,
    contact_sheet,
    count_by_split_and_class,
    difference_hash,
    draw_boxes,
    file_sha256,
    hamming_distance,
    image_files,
    label_text,
    load_split_manifest,
    parse_yolo_label,
    safe_name,
    write_data_yaml,
)


TARGET_COUNTS = {
    "train": 350,
    "val": 100,
    "test": 50,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build the final YOLOv5 traffic-light dataset.")
    parser.add_argument("--raw-root", default="~/traffic_dataset_raw")
    parser.add_argument("--output", default="~/traffic_yolov5")
    parser.add_argument("--splits", default="", help="Default: <raw-root>/traffic_splits.json")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    parser.add_argument("--allow-other-size", action="store_true")
    parser.add_argument(
        "--allow-incomplete-counts",
        action="store_true",
        help="Allow a dataset whose per-class split counts differ from 350/100/50.",
    )
    parser.add_argument("--skip-qa-sheets", action="store_true")
    return parser.parse_args()


def collect_records(raw_root: Path, split_manifest, expected_size, allow_other_size):
    records, errors, warnings = [], [], []
    hashes = defaultdict(list)
    near_rows = []
    for split, sessions in split_manifest.items():
        for session in sessions:
            session_dir = raw_root / session
            if not session_dir.is_dir():
                errors.append("missing session directory: {}".format(session_dir))
                continue
            for capture_class in CAPTURE_CLASSES:
                class_dir = session_dir / capture_class
                paths = image_files(class_dir)
                previous_hash = None
                previous_path = None
                for path in paths:
                    image = cv2.imread(str(path))
                    if image is None:
                        errors.append("cannot decode image: {}".format(path))
                        continue
                    height, width = image.shape[:2]
                    if not allow_other_size and (width, height) != expected_size:
                        errors.append(
                            "{} has size {}x{}, expected {}x{}".format(
                                path, width, height, expected_size[0], expected_size[1]
                            )
                        )
                    label_path = path.with_suffix(".txt")
                    if label_path.exists():
                        try:
                            boxes = parse_yolo_label(label_path.read_text(encoding="utf-8"), str(label_path))
                        except (OSError, UnicodeError, ValueError) as exc:
                            errors.append(str(exc))
                            continue
                    else:
                        errors.append("missing label file: {}".format(label_path))
                        continue

                    if capture_class == "background" and boxes:
                        errors.append("background image must have an empty label: {}".format(label_path))
                    if capture_class != "background":
                        expected_class = CLASS_TO_ID[capture_class]
                        if not boxes:
                            errors.append("positive image has no box: {}".format(label_path))
                        elif not any(int(box[0]) == expected_class for box in boxes):
                            errors.append(
                                "label has no object matching folder '{}': {}".format(capture_class, label_path)
                            )
                        elif len(boxes) > 1:
                            warnings.append(
                                "multiple panels require manual review (different classes are allowed): {}".format(
                                    label_path
                                )
                            )
                    for box in boxes:
                        box_width = float(box[3]) * width
                        box_height = float(box[4]) * height
                        if min(box_width, box_height) < 20.0:
                            warnings.append(
                                "very small box {:.1f}x{:.1f}px: {}".format(box_width, box_height, label_path)
                            )
                        if max(box_width, box_height) > 200.0:
                            warnings.append(
                                "very large box {:.1f}x{:.1f}px: {}".format(box_width, box_height, label_path)
                            )

                    digest = file_sha256(path)
                    hashes[digest].append((split, path, capture_class))
                    perceptual_hash = difference_hash(image)
                    if previous_hash is not None:
                        near_rows.append({
                            "session": session,
                            "class": capture_class,
                            "left": str(previous_path),
                            "right": str(path),
                            "distance": hamming_distance(previous_hash, perceptual_hash),
                        })
                    previous_hash, previous_path = perceptual_hash, path
                    records.append({
                        "split": split,
                        "session": session,
                        "capture_class": capture_class,
                        "image": path,
                        "boxes": boxes,
                        "width": width,
                        "height": height,
                    })

    for digest, occurrences in hashes.items():
        if len(occurrences) < 2:
            continue
        splits = {item[0] for item in occurrences}
        capture_classes = {item[2] for item in occurrences}
        paths = [str(item[1]) for item in occurrences]
        message = "exact duplicate {}: {}".format(digest[:12], paths)
        if len(splits) > 1:
            errors.append("cross-split " + message)
        elif len(capture_classes) > 1:
            errors.append("cross-class " + message)
        else:
            warnings.append(message)
    return records, errors, warnings, near_rows


def write_qa_sheets(records, qa_root: Path):
    groups = defaultdict(list)
    for record in records:
        groups[(record["split"], record["capture_class"])].append(record)
    qa_root.mkdir(parents=True, exist_ok=True)
    for (split, capture_class), items in sorted(groups.items()):
        for page_index in range(0, len(items), 24):
            page = items[page_index:page_index + 24]
            images, labels = [], []
            for record in page:
                image = cv2.imread(str(record["destination_image"]))
                images.append(draw_boxes(image, record["boxes"]))
                labels.append(Path(record["destination_image"]).name)
            sheet = contact_sheet(images, labels, columns=4)
            output = qa_root / "{}_{}_{:03d}.jpg".format(split, capture_class, page_index // 24 + 1)
            cv2.imwrite(str(output), sheet)


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    split_path = Path(args.splits).expanduser().resolve() if args.splits else raw_root / "traffic_splits.json"
    if not raw_root.is_dir():
        print("[FAIL] raw root does not exist:", raw_root)
        return 2
    if not split_path.is_file():
        print("[FAIL] split manifest does not exist:", split_path)
        return 2
    if output_root.exists() and any(output_root.iterdir()):
        print("[FAIL] output must be absent or empty; refusing to overwrite:", output_root)
        return 2

    try:
        split_manifest = load_split_manifest(split_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("[FAIL] invalid split manifest:", exc)
        return 2
    records, errors, warnings, near_rows = collect_records(
        raw_root,
        split_manifest,
        (args.width, args.height),
        bool(args.allow_other_size),
    )
    near_rows = [row for row in near_rows if int(row["distance"]) <= args.near_duplicate_distance]
    if not records:
        errors.append("no images found in manifest sessions")
    counts = count_by_split_and_class(records)
    if not args.allow_incomplete_counts:
        for split, target in TARGET_COUNTS.items():
            for capture_class in CAPTURE_CLASSES:
                actual = int(counts.get(split, {}).get(capture_class, 0))
                if actual != target:
                    errors.append(
                        "{} / {} has {} images, expected {} (use --allow-incomplete-counts only for a draft)".format(
                            split, capture_class, actual, target
                        )
                    )
    if errors:
        print("[FAIL] dataset validation failed with {} error(s):".format(len(errors)))
        for message in errors[:100]:
            print(" -", message)
        if len(errors) > 100:
            print(" - ... {} more".format(len(errors) - 100))
        return 1

    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for record in records:
        source_image = Path(record["image"])
        stem = "{}__{}__{}".format(
            safe_name(record["session"]), safe_name(record["capture_class"]), safe_name(source_image.stem)
        )
        destination_image = output_root / "images" / record["split"] / (stem + source_image.suffix.lower())
        destination_label = output_root / "labels" / record["split"] / (stem + ".txt")
        shutil.copy2(str(source_image), str(destination_image))
        destination_label.write_text(label_text(record["boxes"]) + ("\n" if record["boxes"] else ""), encoding="utf-8")
        record["destination_image"] = destination_image
        record["destination_label"] = destination_label

    write_data_yaml(output_root)
    (output_root / "predefined_classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    report = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "splits": split_manifest,
        "counts": counts,
        "total_images": len(records),
        "warnings": warnings,
        "near_duplicate_pairs": len(near_rows),
    }
    (output_root / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "near_duplicates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("session", "class", "left", "right", "distance"))
        writer.writeheader()
        writer.writerows(near_rows)
    if not args.skip_qa_sheets:
        write_qa_sheets(records, output_root / "qa")

    print("[OK] built dataset:", output_root)
    print("[INFO] total images:", len(records))
    print("[INFO] counts:", json.dumps(report["counts"], ensure_ascii=False, sort_keys=True))
    print("[WARN] validation warnings:", len(warnings))
    print("[WARN] near-duplicate adjacent pairs:", len(near_rows))
    print("[INFO] inspect:", output_root / "dataset_report.json")
    print("[INFO] inspect:", output_root / "qa")
    return 0


if __name__ == "__main__":
    sys.exit(main())
