#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate a five-class ResNet18 traffic-light classifier."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet18


CLASS_NAMES = (
    "green_left",
    "green_right",
    "green_straight",
    "red_light",
    "background",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VerticalBandCrop(object):
    def __init__(self, top, bottom):
        self.top = float(top)
        self.bottom = float(bottom)

    def __call__(self, image):
        width, height = image.size
        y1 = max(0, min(height - 1, int(round(height * self.top))))
        y2 = max(y1 + 1, min(height, int(round(height * self.bottom))))
        return image.crop((0, y1, width, y2))


class FixedClassDataset(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        for class_id, class_name in enumerate(CLASS_NAMES):
            directory = self.root / class_name
            if not directory.is_dir():
                raise RuntimeError("missing class directory: {}".format(directory))
            paths = sorted(
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not paths:
                raise RuntimeError("class directory has no images: {}".format(directory))
            self.samples.extend((path, class_id) for path in paths)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(str(path)) as image:
            image = image.convert("RGB")
        return self.transform(image), target


def parse_args():
    parser = argparse.ArgumentParser(description="Train traffic-light ResNet18 classification.")
    parser.add_argument("--data", required=True, help="Corrected train/val/test dataset root.")
    parser.add_argument("--output", default="runs/traffic_resnet18")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--input-height", type=int, default=160)
    parser.add_argument("--crop-top", type=float, default=0.18)
    parser.add_argument("--crop-bottom", type=float, default=0.72)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--device", default="", help="Default: cuda if available, otherwise cpu.")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(args):
    common_resize = transforms.Resize(
        (args.input_height, args.input_width),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    train_transform = transforms.Compose([
        VerticalBandCrop(args.crop_top, args.crop_bottom),
        common_resize,
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
        transforms.RandomAffine(
            degrees=4.0,
            translate=(0.04, 0.05),
            scale=(0.92, 1.08),
            interpolation=InterpolationMode.BILINEAR,
            fill=(114, 114, 114),
        ),
        # Intentionally no RandomHorizontalFlip: left/right semantics must never swap.
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        VerticalBandCrop(args.crop_top, args.crop_bottom),
        common_resize,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def build_model(pretrained):
    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except (ImportError, AttributeError):
            model = resnet18(pretrained=True)
    else:
        try:
            model = resnet18(weights=None)
        except TypeError:
            model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def confusion_metrics(confusion):
    metrics = {}
    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = float(confusion[index, index])
        predicted = float(confusion[:, index].sum())
        actual = float(confusion[index, :].sum())
        metrics[class_name] = {
            "precision": true_positive / predicted if predicted else 0.0,
            "recall": true_positive / actual if actual else 0.0,
            "support": int(actual),
        }
    return metrics


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                amp_context = torch.amp.autocast(device_type=device.type, enabled=use_amp)
            else:
                amp_context = torch.cuda.amp.autocast(enabled=use_amp)
            with amp_context:
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        predictions = logits.argmax(dim=1)
        batch_size = int(targets.size(0))
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((predictions == targets).sum().item())
        total_samples += batch_size
        for actual, predicted in zip(targets.detach().cpu().tolist(), predictions.detach().cpu().tolist()):
            confusion[int(actual), int(predicted)] += 1
    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / float(max(1, total_samples)),
        "confusion": confusion,
        "per_class": confusion_metrics(confusion),
    }


def save_confusion(path, confusion):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted"] + list(CLASS_NAMES))
        for class_name, row in zip(CLASS_NAMES, confusion.tolist()):
            writer.writerow([class_name] + row)


def export_onnx_model(model, output_path, input_height, input_width):
    model.eval().cpu()
    dummy = torch.zeros(1, 3, input_height, input_width)
    kwargs = {
        "input_names": ["images"],
        "output_names": ["logits"],
        "opset_version": 12,
        "do_constant_folding": True,
    }
    try:
        # PyTorch 2.6+ defaults to the dynamo exporter and may silently upgrade
        # the opset. The legacy exporter is more predictable for RKNN conversion.
        torch.onnx.export(model, dummy, str(output_path), dynamo=False, **kwargs)
    except TypeError:
        # Compatibility with older PyTorch versions that do not expose `dynamo`.
        torch.onnx.export(model, dummy, str(output_path), **kwargs)
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
    except ImportError:
        print("[WARN] onnx package is not installed; exported file was not checker-validated")


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    seed_everything(args.seed)
    data_root = Path(args.data).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not (0.0 <= args.crop_top < args.crop_bottom <= 1.0):
        raise ValueError("crop range must satisfy 0 <= top < bottom <= 1")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.type == "cuda" and not args.no_amp
    pin_memory = device.type == "cuda"

    train_transform, eval_transform = build_transforms(args)
    datasets = {
        "train": FixedClassDataset(data_root / "train", train_transform),
        "val": FixedClassDataset(data_root / "val", eval_transform),
        "test": FixedClassDataset(data_root / "test", eval_transform),
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=pin_memory, generator=generator,
            persistent_workers=args.workers > 0,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=pin_memory,
            persistent_workers=args.workers > 0,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=pin_memory,
            persistent_workers=args.workers > 0,
        ),
    }
    model = build_model(not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metadata = {
        "model": "resnet18",
        "classes": list(CLASS_NAMES),
        "class_to_id": {name: index for index, name in enumerate(CLASS_NAMES)},
        "input_width": args.input_width,
        "input_height": args.input_height,
        "crop_top": args.crop_top,
        "crop_bottom": args.crop_bottom,
        "color": "RGB",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "runtime_horizontal_flip_required": True,
        "random_horizontal_flip": False,
        "dataset_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "seed": args.seed,
    }
    (output_root / "training_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    history = []
    best_accuracy = -1.0
    epochs_without_improvement = 0
    checkpoint_path = output_root / "best.pt"
    print("[INFO] device:", device)
    print("[INFO] dataset sizes:", metadata["dataset_sizes"])
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_result = run_epoch(
            model, loaders["train"], criterion, device,
            optimizer=optimizer, scaler=scaler, use_amp=use_amp,
        )
        val_result = run_epoch(
            model, loaders["val"], criterion, device, use_amp=use_amp
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "val_loss": val_result["loss"],
            "val_accuracy": val_result["accuracy"],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            "epoch {}/{} train_loss={:.4f} train_acc={:.4f} "
            "val_loss={:.4f} val_acc={:.4f} time={:.1f}s".format(
                epoch, args.epochs, row["train_loss"], row["train_accuracy"],
                row["val_loss"], row["val_accuracy"], row["seconds"]
            )
        )
        if val_result["accuracy"] > best_accuracy + 1e-6:
            best_accuracy = val_result["accuracy"]
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "metadata": metadata,
                "epoch": epoch,
                "val_accuracy": best_accuracy,
            }, str(checkpoint_path))
            save_confusion(output_root / "best_val_confusion.csv", val_result["confusion"])
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("[INFO] early stopping after {} epochs without improvement".format(args.patience))
                break

    with (output_root / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = run_epoch(model, loaders["test"], criterion, device, use_amp=use_amp)
    save_confusion(output_root / "test_confusion.csv", test_result["confusion"])
    test_report = {
        "loss": test_result["loss"],
        "accuracy": test_result["accuracy"],
        "per_class": test_result["per_class"],
        "confusion": test_result["confusion"].tolist(),
        "best_epoch": checkpoint["epoch"],
        "best_val_accuracy": checkpoint["val_accuracy"],
    }
    (output_root / "test_report.json").write_text(
        json.dumps(test_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[RESULT] test_accuracy={:.4f}".format(test_result["accuracy"]))
    for class_name in CLASS_NAMES:
        item = test_result["per_class"][class_name]
        print(
            "[RESULT] {:>14s} precision={:.4f} recall={:.4f} support={}".format(
                class_name, item["precision"], item["recall"], item["support"]
            )
        )

    if args.export_onnx:
        onnx_path = output_root / "traffic_resnet18.onnx"
        export_onnx_model(model, onnx_path, args.input_height, args.input_width)
        print("[OK] ONNX exported:", onnx_path)
    print("[OK] best checkpoint:", checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
