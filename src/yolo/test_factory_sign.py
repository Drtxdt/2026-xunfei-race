#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
厂区标识牌 RKNN 模型测试脚本
============================
测试 factory_sign_3cls_fp16.rknn 在小车 NPU 上的推理效果。

用法:
  # 测试单张图片
  python3 test_factory_sign.py --source ./test.jpg

  # 测试图片目录（批量）
  python3 test_factory_sign.py --source ./test_images/

  # 摄像头实时测试
  python3 test_factory_sign.py --source camera

  # 指定模型和阈值
  python3 test_factory_sign.py --model models/factory_sign_3cls_fp16.rknn \
      --source ./test.jpg --conf 0.3 --nms 0.45

  # 保存标注结果
  python3 test_factory_sign.py --source ./test_images/ --output ./results/

选项:
  --model    RKNN 模型路径          (默认: models/factory_sign_3cls_fp16.rknn)
  --source   图片来源               (图片路径 / 目录 / camera)
  --output   标注图片输出目录        (默认: 不保存)
  --conf     置信度阈值             (默认: 0.3)
  --nms      NMS IoU 阈值           (默认: 0.45)
  --img-size 模型输入尺寸           (默认: 640)
  --camera   摄像头设备 ID          (默认: 0)

类别: food(食品加工车间) / electronic(电子产品生产车间) / daily(日用品加工车间)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ============================================================
# 常量 — YOLOv5 后处理参数
# ============================================================

CLASS_NAMES = ["food", "electronic", "daily"]
CLASS_LABELS = ["食品加工车间", "电子产品生产车间", "日用品加工车间"]

ANCHORS = np.array(
    [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
     [59, 119], [116, 90], [156, 198], [373, 326]],
    dtype=np.float32,
)
MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

COLORS = [
    (0, 255, 0),    # food — 绿
    (255, 191, 0),  # electronic — 蓝
    (0, 255, 255),  # daily — 黄
]


# ============================================================
# 工具函数
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(im: np.ndarray, new_shape: int):
    """等比缩放 + 黑边填充到 new_shape × new_shape"""
    h, w = im.shape[:2]
    r = min(float(new_shape) / h, float(new_shape) / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape - new_w) / 2.0, (new_shape - new_h) / 2.0
    if (w, h) != (new_w, new_h):
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return im, r, (dw, dh)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2.0
    y[:, 1] = x[:, 1] - x[:, 3] / 2.0
    y[:, 2] = x[:, 0] + x[:, 2] / 2.0
    y[:, 3] = x[:, 1] + x[:, 3] / 2.0
    return y


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep, dtype=np.int64)


# ============================================================
# YOLOv5 后处理
# ============================================================

def process_head(output: np.ndarray, mask: List[int], input_size: int):
    """处理单个检测头"""
    anchors = ANCHORS[mask].reshape(1, 1, 3, 2)
    gh, gw = output.shape[:2]
    box_xy = (output[..., 0:2] * 2.0 - 0.5 +
              np.stack(np.meshgrid(np.arange(gw), np.arange(gh), indexing="xy"), axis=-1)
              .reshape(gh, gw, 1, 2)) * (input_size / float(gh))
    box_wh = np.power(output[..., 2:4] * 2.0, 2.0) * anchors
    return np.concatenate((box_xy, box_wh), axis=-1), output[..., 4:5], output[..., 5:]


def postprocess(outputs, input_size: int, conf_thresh: float,
                nms_thresh: float, num_classes: int):
    """YOLOv5 完整后处理: decode + NMS"""

    # ---- 单输出路径 (Detect 层直接输出的 flat 格式) ----
    arr = np.asarray(outputs[0]).squeeze()
    if arr.ndim == 2:
        actual_nc = arr.shape[-1] - 5
        boxes = arr[:, :4].astype(np.float32)
        obj = arr[:, 4].astype(np.float32)
        cls_probs = arr[:, 5:].astype(np.float32)

        # 自动判断是否需要 sigmoid
        if obj.max() > 1.0 or obj.min() < 0.0:
            obj = sigmoid(obj)
        if cls_probs.max() > 1.0 or cls_probs.min() < 0.0:
            cls_probs = sigmoid(cls_probs)

        class_ids = np.argmax(cls_probs, axis=1)
        scores = obj * cls_probs[np.arange(len(cls_probs)), class_ids]

        # 自动检测模型输入尺寸
        for sz in (320, 416, 512, 640):
            grids = (sz // 8) ** 2 + (sz // 16) ** 2 + (sz // 32) ** 2
            if arr.shape[0] == grids * 3:
                boxes = boxes * (float(input_size) / float(sz))
                break

        keep = np.where(scores >= conf_thresh)[0]
        if keep.size == 0:
            return None, None, None
        boxes = xywh2xyxy(boxes[keep])
        class_ids = class_ids[keep]
        scores = scores[keep]

        # 逐类 NMS
        kept_boxes, kept_classes, kept_scores = [], [], []
        for cls_id in set(class_ids.tolist()):
            idx = np.where(class_ids == cls_id)
            nms_idx = nms_boxes(boxes[idx], scores[idx], nms_thresh)
            kept_boxes.append(boxes[idx][nms_idx])
            kept_classes.append(np.full(len(nms_idx), cls_id, dtype=np.int64))
            kept_scores.append(scores[idx][nms_idx])
        if not kept_boxes:
            return None, None, None
        return (np.concatenate(kept_boxes), np.concatenate(kept_classes),
                np.concatenate(kept_scores))

    # ---- 3-head 输出路径 ----
    all_boxes, all_classes, all_scores = [], [], []
    for output, mask in zip(outputs[:3], MASKS):
        arr_hwc = np.asarray(output).squeeze()
        if arr_hwc.ndim == 3 and arr_hwc.shape[-1] % 3 == 0:
            h, w, c = arr_hwc.shape
            arr_hwc = arr_hwc.reshape(h, w, 3, c // 3)
        boxes, obj, cls_probs = process_head(arr_hwc, mask, input_size)
        if obj.max() > 1.0 or obj.min() < 0.0:
            obj = sigmoid(obj)
        if cls_probs.max() > 1.0 or cls_probs.min() < 0.0:
            cls_probs = sigmoid(cls_probs)
        flat_boxes = boxes.reshape(-1, 4)
        flat_obj = obj.reshape(-1)
        flat_cls = cls_probs.reshape(-1, cls_probs.shape[-1])
        pos = np.where(flat_obj >= conf_thresh)
        if pos[0].size == 0:
            continue
        fb = flat_boxes[pos]
        fc = np.argmax(flat_cls[pos], axis=-1)
        fs = flat_obj[pos] * np.max(flat_cls[pos], axis=-1)
        all_boxes.append(fb)
        all_classes.append(fc)
        all_scores.append(fs)
    if not all_boxes:
        return None, None, None
    boxes = xywh2xyxy(np.concatenate(all_boxes))
    classes = np.concatenate(all_classes)
    scores = np.concatenate(all_scores)
    kept_boxes, kept_classes, kept_scores = [], [], []
    for cls_id in set(classes.tolist()):
        idx = np.where(classes == cls_id)
        nms_idx = nms_boxes(boxes[idx], scores[idx], nms_thresh)
        kept_boxes.append(boxes[idx][nms_idx])
        kept_classes.append(np.full(len(nms_idx), cls_id, dtype=np.int64))
        kept_scores.append(scores[idx][nms_idx])
    if not kept_boxes:
        return None, None, None
    return (np.concatenate(kept_boxes), np.concatenate(kept_classes),
            np.concatenate(kept_scores))


def scale_boxes(boxes: np.ndarray, ratio: float, pad: Tuple[float, float],
                frame_shape: Tuple[int, int, int]) -> np.ndarray:
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    h, w = frame_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    return boxes


# ============================================================
# 模型加载
# ============================================================

def load_model(model_path: str):
    """加载 RKNN 模型，优先 rknnlite(NPU) → rknn-toolkit2(模拟器)"""
    # 优先 NPU 直推
    try:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        print(f"[MODEL] rknnlite (NPU): {model_path}")
        return rknn
    except ImportError:
        pass

    # 回退模拟器
    try:
        from rknn.api import RKNN
        rknn = RKNN(verbose=False)
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = rknn.init_runtime(target='rk3588')
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        print(f"[MODEL] rknn-toolkit2 (simulator): {model_path}")
        return rknn
    except ImportError:
        pass

    raise RuntimeError("No RKNN runtime. Install rknnlite or rknn-toolkit2.")


# ============================================================
# 推理
# ============================================================

def infer(rknn, frame: np.ndarray, input_size: int,
          conf_thresh: float, nms_thresh: float):
    """单帧推理，返回检测框列表"""
    img, ratio, pad = letterbox(frame, input_size)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outputs = rknn.inference(inputs=[np.expand_dims(rgb, axis=0)])
    boxes, classes, scores = postprocess(
        outputs, input_size, conf_thresh, nms_thresh, len(CLASS_NAMES))
    if boxes is not None:
        boxes = scale_boxes(boxes, ratio, pad, frame.shape)
    return boxes, classes, scores


# ============================================================
# 可视化
# ============================================================

def draw(frame: np.ndarray, boxes, classes, scores, elapsed_ms: float) -> np.ndarray:
    annot = frame.copy()
    if boxes is None:
        cv2.putText(annot, "No detection", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        for box, cls_id, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = [int(v) for v in box]
            cls_id = int(cls_id)
            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)
            label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
            cv2.putText(annot, label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(annot, f"{elapsed_ms:.0f}ms", (8, annot.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return annot


# ============================================================
# 测试模式
# ============================================================

def test_image(rknn, image_path: str, args) -> None:
    """测试单张图片"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return
    t0 = time.time()
    boxes, classes, scores = infer(rknn, frame, args.img_size, args.conf, args.nms)
    elapsed = (time.time() - t0) * 1000

    if boxes is None:
        print(f"  No detection | {elapsed:.0f}ms")
    else:
        for box, cls_id, score in zip(boxes, classes, scores):
            cls_id = int(cls_id)
            x1, y1, x2, y2 = [int(v) for v in box]
            print(f"  {CLASS_NAMES[cls_id]:12s} ({CLASS_LABELS[cls_id]})  "
                  f"conf={score:.3f}  bbox=[{x1},{y1},{x2},{y2}]  {elapsed:.0f}ms")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        out_name = os.path.join(args.output, os.path.basename(image_path))
        annot = draw(frame, boxes, classes, scores, elapsed)
        cv2.imwrite(out_name, annot)
        print(f"  Saved: {out_name}")

    if args.show:
        annot = draw(frame, boxes, classes, scores, elapsed)
        cv2.imshow("Factory Sign Test (Q=quit)", annot)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_directory(rknn, source_dir: str, args) -> None:
    """批量测试图片目录"""
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = sorted([f for f in Path(source_dir).iterdir() if f.suffix.lower() in exts])
    if not files:
        print(f"[ERROR] No images in {source_dir}")
        return

    print(f"\n{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Source: {source_dir} ({len(files)} images)")
    print(f"  Conf={args.conf}  NMS={args.nms}  Size={args.img_size}")
    print(f"{'='*60}\n")

    stats = {name: 0 for name in CLASS_NAMES}
    total_time, no_det = 0.0, 0
    output_shapes_logged = False

    for i, img_path in enumerate(files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        t0 = time.time()
        boxes, classes, scores = infer(rknn, frame, args.img_size, args.conf, args.nms)
        elapsed = (time.time() - t0) * 1000
        total_time += elapsed

        if not output_shapes_logged:
            output_shapes_logged = True

        if boxes is not None:
            for cls_id in classes:
                name = CLASS_NAMES[int(cls_id)]
                stats[name] = stats.get(name, 0) + 1
        else:
            no_det += 1

        det_str = " | ".join(
            f"{CLASS_NAMES[int(c)]} {s:.2f}" for c, s in zip(classes, scores)[:3]
        ) if boxes is not None else "x"
        print(f"  [{i+1:>3}/{len(files)}] {img_path.name:45s} {elapsed:5.0f}ms  {det_str}")

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            annot = draw(frame, boxes, classes, scores, elapsed)
            cv2.imwrite(os.path.join(args.output, img_path.name), annot)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  Images: {len(files)}  |  No detection: {no_det}  "
          f"|  Avg: {total_time/len(files):.0f}ms")
    for name in CLASS_NAMES:
        bar = "#" * min(stats[name], 40)
        print(f"  {name:12s} ({CLASS_LABELS[CLASS_NAMES.index(name)]:10s}): "
              f"{stats[name]:>4d}  {bar}")
    print(f"{'='*60}")


def test_camera(rknn, args) -> None:
    """摄像头实时测试"""
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera /dev/video{args.camera}")
        return

    print(f"\n  Camera /dev/video{args.camera} | Q=quit S=save\n"
          f"  Classes: {CLASS_NAMES} | Conf={args.conf}\n")

    frame_count, fps, t_fps = 0, 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        boxes, classes, scores = infer(rknn, frame, args.img_size, args.conf, args.nms)
        elapsed = (time.time() - t0) * 1000

        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10000.0 / (time.time() - t_fps) / 10
            t_fps = time.time()

        annot = draw(frame, boxes, classes, scores, elapsed)
        cv2.putText(annot, f"FPS: {fps:.1f}", (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if args.show or True:  # camera mode always show
            cv2.imshow("Factory Sign Test (Q=quit, S=save)", annot)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"factory_sign_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fname, annot)
            print(f"  Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="厂区标识牌 RKNN 模型测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
类别映射:
  food       — 食品加工车间
  electronic — 电子产品生产车间
  daily      — 日用品加工车间

示例:
  python3 test_factory_sign.py --source ./test.jpg
  python3 test_factory_sign.py --source ./test_images/ --output ./results/
  python3 test_factory_sign.py --source camera --conf 0.3
  python3 test_factory_sign.py --model best.rknn --source ./test.jpg
        """)
    parser.add_argument("--model", default="models/factory_sign_3cls_fp16.rknn",
                        help="RKNN 模型路径 (默认: models/factory_sign_3cls_fp16.rknn)")
    parser.add_argument("--source", required=True,
                        help="图片来源: 图片路径 / 目录 / camera")
    parser.add_argument("--output", default=None,
                        help="标注图片输出目录 (默认: 不保存)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="置信度阈值 (默认: 0.3)")
    parser.add_argument("--nms", type=float, default=0.45,
                        help="NMS IoU 阈值 (默认: 0.45)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="模型输入尺寸 (默认: 640)")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头设备 ID (默认: 0)")
    parser.add_argument("--show", action="store_true",
                        help="显示标注图片窗口 (图片模式)")
    args = parser.parse_args()

    # 检查模型
    if not os.path.isfile(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print(f"  Place the .rknn file in models/ or use --model")
        sys.exit(1)

    print(f"Loading: {args.model}")
    rknn = load_model(args.model)

    try:
        src = args.source.lower()
        if src == "camera":
            test_camera(rknn, args)
        elif os.path.isfile(args.source):
            test_image(rknn, args.source, args)
        elif os.path.isdir(args.source):
            test_directory(rknn, args.source, args)
        else:
            print(f"[ERROR] Source not found: {args.source}")
            sys.exit(1)
    finally:
        try:
            rknn.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
