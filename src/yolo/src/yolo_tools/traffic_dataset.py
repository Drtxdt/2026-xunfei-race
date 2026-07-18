#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure helpers for validating and building the traffic-light YOLOv5 dataset."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


CLASS_NAMES = ("green_left", "green_right", "green_straight", "red_light")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
CAPTURE_CLASSES = CLASS_NAMES + ("background",)
IMAGE_SUFFIXES = frozenset((".jpg", ".jpeg", ".png", ".bmp"))


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "item"


def image_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_yolo_label(text: str, source: str = "<label>") -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    for line_number, raw_line in enumerate(str(text).splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError("{}:{} expected 5 columns, got {}".format(source, line_number, len(parts)))
        try:
            class_value = float(parts[0])
            class_id = int(class_value)
            values = [float(value) for value in parts[1:]]
        except ValueError:
            raise ValueError("{}:{} contains a non-numeric value".format(source, line_number))
        if class_value != class_id or class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError("{}:{} class_id must be an integer in 0..3".format(source, line_number))
        cx, cy, width, height = values
        if not all(np.isfinite(values)):
            raise ValueError("{}:{} contains a non-finite coordinate".format(source, line_number))
        if width <= 0.0 or height <= 0.0:
            raise ValueError("{}:{} width and height must be positive".format(source, line_number))
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and width <= 1.0 and height <= 1.0):
            raise ValueError("{}:{} normalized values must be within 0..1".format(source, line_number))
        if cx - width / 2.0 < -1e-6 or cx + width / 2.0 > 1.0 + 1e-6:
            raise ValueError("{}:{} box exceeds horizontal image bounds".format(source, line_number))
        if cy - height / 2.0 < -1e-6 or cy + height / 2.0 > 1.0 + 1e-6:
            raise ValueError("{}:{} box exceeds vertical image bounds".format(source, line_number))
        boxes.append((class_id, cx, cy, width, height))
    return boxes


def label_text(boxes: Sequence[Sequence[float]]) -> str:
    return "\n".join(
        "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
            int(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
        )
        for box in boxes
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bool(bit))
    return value


def hamming_distance(left: int, right: int) -> int:
    return int(left ^ right).bit_count() if hasattr(int, "bit_count") else bin(left ^ right).count("1")


def load_split_manifest(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = ("train", "val", "test")
    result = {}
    seen = set()
    for split in expected:
        values = data.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError("split manifest must contain a non-empty '{}' list".format(split))
        sessions = [str(value).strip() for value in values if str(value).strip()]
        if len(sessions) != len(values):
            raise ValueError("split '{}' contains an empty session name".format(split))
        overlap = seen.intersection(sessions)
        if overlap:
            raise ValueError("sessions assigned to multiple splits: {}".format(sorted(overlap)))
        seen.update(sessions)
        result[split] = sessions
    return result


def write_data_yaml(output_root: Path) -> None:
    content = """# Generated YOLOv5 traffic-light dataset config
path: {path}
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: green_left
  1: green_right
  2: green_straight
  3: red_light
""".format(path=output_root.resolve().as_posix())
    (output_root / "data.yaml").write_text(content, encoding="utf-8")


def draw_boxes(image: np.ndarray, boxes: Sequence[Sequence[float]]) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    colors = ((0, 255, 0), (255, 255, 0), (0, 255, 255), (0, 0, 255))
    for class_id, cx, cy, box_width, box_height in boxes:
        x1 = int(round((cx - box_width / 2.0) * width))
        y1 = int(round((cy - box_height / 2.0) * height))
        x2 = int(round((cx + box_width / 2.0) * width))
        y2 = int(round((cy + box_height / 2.0) * height))
        color = colors[int(class_id)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            CLASS_NAMES[int(class_id)],
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def contact_sheet(images: Sequence[np.ndarray], labels: Sequence[str], columns: int = 4) -> np.ndarray:
    tile_width, tile_height = 240, 200
    rows = max(1, (len(images) + columns - 1) // columns)
    sheet = np.full((rows * tile_height, columns * tile_width, 3), 32, dtype=np.uint8)
    for index, image in enumerate(images):
        resized = cv2.resize(image, (tile_width, 180), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        y, x = row * tile_height, column * tile_width
        sheet[y:y + 180, x:x + tile_width] = resized
        cv2.putText(
            sheet,
            labels[index][:38],
            (x + 4, y + 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return sheet


def count_by_split_and_class(records: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, int]]:
    counts = defaultdict(lambda: defaultdict(int))
    for record in records:
        counts[str(record["split"])][str(record["capture_class"])] += 1
    return {split: dict(values) for split, values in counts.items()}
