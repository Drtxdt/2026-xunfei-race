#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Initialize the five-session traffic-light capture and LabelImg workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yolo_tools.traffic_dataset import CAPTURE_CLASSES, CLASS_NAMES


SESSIONS = (
    "session_01_normal_light",
    "session_02_bright",
    "session_03_dim",
    "session_04_background_changed",
    "session_05_final_field",
)


def write_if_missing(path: Path, content: str) -> bool:
    if path.exists():
        return False
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Create traffic-light capture session directories.")
    parser.add_argument("--root", default="~/traffic_dataset_raw", help="Raw dataset root.")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for session in SESSIONS:
        for class_name in CAPTURE_CLASSES:
            (root / session / class_name).mkdir(parents=True, exist_ok=True)

    write_if_missing(root / "predefined_classes.txt", "\n".join(CLASS_NAMES) + "\n")
    split_manifest = {
        "train": list(SESSIONS[:3]),
        "val": [SESSIONS[3]],
        "test": [SESSIONS[4]],
    }
    write_if_missing(
        root / "traffic_splits.json",
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    quotas = {
        SESSIONS[0]: 120,
        SESSIONS[1]: 120,
        SESSIONS[2]: 110,
        SESSIONS[3]: 100,
        SESSIONS[4]: 50,
    }
    write_if_missing(
        root / "capture_quotas.json",
        json.dumps({"per_class": quotas}, ensure_ascii=False, indent=2) + "\n",
    )
    print("[OK] initialized:", root)
    print("[INFO] LabelImg classes:", root / "predefined_classes.txt")
    print("[INFO] split manifest  :", root / "traffic_splits.json")
    print("[INFO] quotas per class:", root / "capture_quotas.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
