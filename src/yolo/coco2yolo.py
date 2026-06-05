#!/usr/bin/env python3
"""Convert COCO JSON annotations to YOLO txt format with train/val split."""

import json
import os
import shutil
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────
COCO_JSON = Path("2728601_1780572667/2728601_1780572667/Annotations/coco_info.json")
IMAGE_DIR = Path("2728601_1780572667/2728601_1780572667/Images")
OUTPUT_DIR = Path("yolo_dataset")
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# Category mapping: COCO category_id -> YOLO class_id (0-indexed)
# Names from the dataset: 绿色左箭头, 绿色右箭头, 绿色上箭头, 红色停止
CLASS_NAMES = {
    0: "green_left",
    1: "green_right",
    2: "green_straight",
    3: "red_light",
}

def coco_to_yolo_bbox(bbox, img_w, img_h):
    """Convert [x, y, w, h] absolute to YOLO normalized [cx, cy, w, h]."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh

def main():
    random.seed(RANDOM_SEED)

    # Load COCO JSON
    with open(COCO_JSON, encoding="utf-8") as f:
        data = json.load(f)

    # Build image_id -> image_info map
    img_map = {img["id"]: img for img in data["images"]}

    # Group annotations by image_id
    anns_by_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    # Collect all image_ids with annotations
    all_img_ids = list(anns_by_image.keys())
    random.shuffle(all_img_ids)

    split_idx = int(len(all_img_ids) * TRAIN_RATIO)
    train_ids = set(all_img_ids[:split_idx])
    val_ids = set(all_img_ids[split_idx:])

    print(f"Total images: {len(all_img_ids)}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Create output directories
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process each image
    for img_id, anns in anns_by_image.items():
        img_info = img_map[img_id]
        fname = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]
        split = "train" if img_id in train_ids else "val"

        src_path = IMAGE_DIR / fname
        dst_path = OUTPUT_DIR / "images" / split / fname
        label_path = OUTPUT_DIR / "labels" / split / (Path(fname).stem + ".txt")

        # Copy image
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"WARNING: missing image {src_path}")

        # Write YOLO labels
        lines = []
        for ann in anns:
            cat_id = ann["category_id"]
            cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    # Generate data.yaml for YOLOv5
    yaml_content = f"""# YOLOv5 dataset config
path: {OUTPUT_DIR.resolve().as_posix()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {json.dumps([CLASS_NAMES[i] for i in sorted(CLASS_NAMES)])}
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nDone! Output: {OUTPUT_DIR}")
    print(f"data.yaml written to: {OUTPUT_DIR / 'data.yaml'}")

if __name__ == "__main__":
    main()
