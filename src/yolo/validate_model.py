#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RKNN YOLOv5 模型验证与 ROS 推理节点
=====================================
双模式工具: CLI 离线验证 + ROS 在线推理(含共识滤波/语音/调试图像)

用法:
  # ---- CLI 模式（与之前兼容） ----
  # 验证厂区标识牌模型（3类：food/electronic/daily）
  python3 validate_model.py \
      --model models/factory_sign_3cls.rknn \
      --classes food,electronic,daily \
      --source ./test_images/

  # 验证红绿灯模型（4类：red_light/green_straight/green_left/green_right）
  python3 validate_model.py \
      --model models/best.rknn \
      --classes green_left,green_right,green_straight,red_light \
      --source ./test_images/

  # 使用摄像头实时验证
  python3 validate_model.py \
      --model models/factory_sign_3cls.rknn \
      --classes food,electronic,daily \
      --source camera --camera-id 0

  # ---- ROS 模式（上车部署） ----
  # 作为 ROS 节点运行，订阅摄像头话题，发布检测/语音/调试图像
  python3 validate_model.py --ros --model models/factory_sign_3cls.rknn \
      --classes food,electronic,daily

  # 通过 ROS param 配置（配合 launch 文件使用）
  python3 validate_model.py --ros
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 常量
# ============================================================

ANCHORS = np.array(
    [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
     [59, 119], [116, 90], [156, 198], [373, 326]],
    dtype=np.float32,
)
MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

DEFAULT_COLORS = [
    (0, 255, 0), (255, 255, 0), (0, 255, 255),
    (0, 0, 255), (255, 0, 255), (255, 0, 0), (128, 0, 255),
]


# ============================================================
# 工具函数
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(im: np.ndarray, new_shape: int, color=(0, 0, 0)):
    shape = im.shape[:2]
    r = min(float(new_shape) / shape[0], float(new_shape) / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]
    dw /= 2.0; dh /= 2.0
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2.0
    y[:, 1] = x[:, 1] - x[:, 3] / 2.0
    y[:, 2] = x[:, 0] + x[:, 2] / 2.0
    y[:, 3] = x[:, 1] + x[:, 3] / 2.0
    return y


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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


def infer_input_size_from_count(count: int) -> Optional[int]:
    """从单输出 25200/6300/... 行数反推 YOLOv5 输入尺寸"""
    for size in (320, 416, 512, 640):
        grids = (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2
        if count == grids * 3:
            return size
    return None


# ============================================================
# YOLOv5 后处理
# ============================================================

def process_output(output: np.ndarray, mask: List[int], input_size: int):
    anchors = ANCHORS[mask].reshape(1, 1, 3, 2)
    grid_h, grid_w = output.shape[:2]
    box_confidence = output[..., 4:5]
    box_class_probs = output[..., 5:]
    if box_confidence.max() > 1.0 or box_class_probs.max() > 1.0:
        box_confidence = sigmoid(box_confidence)
        box_class_probs = sigmoid(box_class_probs)
    grid_y, grid_x = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    grid = np.stack((grid_x, grid_y), axis=-1).reshape(grid_h, grid_w, 1, 2)
    box_xy = (output[..., 0:2] * 2.0 - 0.5 + grid) * (input_size / float(grid_h))
    box_wh = np.power(output[..., 2:4] * 2.0, 2.0) * anchors
    return np.concatenate((box_xy, box_wh), axis=-1), box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs, conf_thresh):
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])
    pos = np.where(box_confidences >= conf_thresh)
    boxes, box_confidences, box_class_probs = boxes[pos], box_confidences[pos], box_class_probs[pos]
    if boxes.size == 0:
        return np.empty((0, 4)), np.empty((0,), dtype=np.int64), np.empty((0,))
    class_scores = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    pos = np.where(class_scores >= conf_thresh)
    boxes, classes, scores = boxes[pos], classes[pos], (class_scores * box_confidences)[pos]
    return boxes, classes, scores


def normalize_output(output: np.ndarray) -> np.ndarray:
    """将 RKNN 输出归一化为 H×W×3×(5+nc) 形状"""
    arr = np.asarray(output)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"unsupported RKNN output ndim={arr.ndim} shape={arr.shape}")
    if arr.shape[0] % 3 == 0 and arr.shape[1] == arr.shape[2]:
        c, h, w = arr.shape
        return np.transpose(arr.reshape(3, c // 3, h, w), (2, 3, 0, 1))
    if arr.shape[-1] % 3 == 0 and arr.shape[0] == arr.shape[1]:
        h, w, c = arr.shape
        return arr.reshape(h, w, 3, c // 3)
    raise ValueError(f"unsupported RKNN output shape={arr.shape}")


def yolov5_post_process_3head(outputs, input_size, conf_thresh, nms_thresh, num_classes):
    """3 特征图后处理路径（640 等大尺寸模型）"""
    processed = [normalize_output(o) for o in outputs[:3]]
    boxes_list, classes_list, scores_list = [], [], []
    for output, mask in zip(processed, MASKS):
        b, bc, bp = process_output(output, mask, input_size)
        b, c, s = filter_boxes(b, bc, bp, conf_thresh)
        if b.size:
            boxes_list.append(b)
            classes_list.append(c)
            scores_list.append(s)
    if not boxes_list:
        return None, None, None

    boxes = xywh2xyxy(np.concatenate(boxes_list))
    classes = np.concatenate(classes_list)
    scores = np.concatenate(scores_list)

    kept_boxes, kept_classes, kept_scores = [], [], []
    for cls_id in set(classes.tolist()):
        idx = np.where(classes == cls_id)
        keep = nms_boxes(boxes[idx], scores[idx], nms_thresh)
        kept_boxes.append(boxes[idx][keep])
        kept_classes.append(np.full(len(keep), cls_id, dtype=np.int64))
        kept_scores.append(scores[idx][keep])
    if not kept_boxes:
        return None, None, None
    return np.concatenate(kept_boxes), np.concatenate(kept_classes), np.concatenate(kept_scores)


def yolov5_post_process_flat(output_arr: np.ndarray, conf_thresh, nms_thresh, num_classes):
    """单输出后处理路径（320 等 Detect 层直接输出的模型）"""
    arr = np.asarray(output_arr)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.shape[-1] != 5 + num_classes:
        return None, None, None

    boxes = arr[:, :4].astype(np.float32)
    obj = arr[:, 4].astype(np.float32)
    class_probs = arr[:, 5:].astype(np.float32)
    if obj.size and (obj.max() > 1.0 or obj.min() < 0.0):
        obj = sigmoid(obj)
    if class_probs.size and (class_probs.max() > 1.0 or class_probs.min() < 0.0):
        class_probs = sigmoid(class_probs)

    class_ids = np.argmax(class_probs, axis=1)
    class_scores = class_probs[np.arange(class_probs.shape[0]), class_ids]
    scores = obj * class_scores
    keep = np.where(scores >= conf_thresh)[0]
    if keep.size == 0:
        return None, None, None

    boxes = xywh2xyxy(boxes[keep])
    class_ids = class_ids[keep]
    scores = scores[keep]

    kept_boxes, kept_classes, kept_scores = [], [], []
    for cls_id in set(class_ids.tolist()):
        idx = np.where(class_ids == cls_id)
        keep_idx = nms_boxes(boxes[idx], scores[idx], nms_thresh)
        kept_boxes.append(boxes[idx][keep_idx])
        kept_classes.append(np.full(len(keep_idx), cls_id, dtype=np.int64))
        kept_scores.append(scores[idx][keep_idx])
    if not kept_boxes:
        return None, None, None
    return np.concatenate(kept_boxes), np.concatenate(kept_classes), np.concatenate(kept_scores)


def yolov5_post_process(outputs, input_size, conf_thresh, nms_thresh, num_classes):
    """自动选择 3-head 或单输出后处理路径"""
    # 尝试单输出路径
    single = yolov5_post_process_flat(outputs[0], conf_thresh, nms_thresh, num_classes)
    if single is not None and single[0] is not None:
        boxes, classes, scores = single
        model_size = infer_input_size_from_count(np.asarray(outputs[0]).squeeze().shape[0])
        if model_size and model_size != input_size:
            scale = float(input_size) / float(model_size)
            boxes = boxes * scale
        return boxes, classes, scores

    # 如果 flat 路径确认这是单输出模型（ndim=2 且列数>=5），
    # 说明只是当前帧无检测，不要回退到 3-head 路径
    arr = np.asarray(outputs[0]).squeeze()
    if arr.ndim == 2 and arr.shape[-1] >= 5:
        return None, None, None

    # 回退到 3-head 路径
    return yolov5_post_process_3head(outputs, input_size, conf_thresh, nms_thresh, num_classes)


def scale_boxes(boxes, ratio, pad, frame_shape):
    dw, dh = pad
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= ratio
    h, w = frame_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    return boxes


# ============================================================
# 共识滤波器
# ============================================================

class ConsensusFilter:
    """多帧共识滤波 + EMA 平滑

    解决 INT8 量化引起的帧间抖动和类别跳变:
    - N 帧确认后才锁定目标类别
    - M 帧连续未检测到才释放
    - EMA 平滑置信度
    - 超时自动重置
    """

    def __init__(self, num_classes: int, confirm_frames: int = 3,
                 release_frames: int = 3, timeout: float = 1.0,
                 ema_alpha: float = 0.3):
        self.num_classes = num_classes
        self.confirm_frames = confirm_frames
        self.release_frames = release_frames
        self.timeout = timeout
        self.ema_alpha = ema_alpha

        self.class_hits = {i: 0 for i in range(num_classes)}
        self.consensus_class: Optional[int] = None
        self.consensus_confidence: float = 0.0
        self.consensus_held: int = 0
        self.consensus_started_at: float = 0.0
        self.release_counter: int = 0

    def update(self, detections: List[Dict[str, Any]]) -> None:
        """输入当前帧的检测结果，更新共识状态"""
        # 统计各类别最佳置信度
        best_by_class: Dict[int, float] = {}
        for det in detections:
            cls_id = int(det["class_id"])
            conf = float(det["confidence"])
            if conf > best_by_class.get(cls_id, 0.0):
                best_by_class[cls_id] = conf

        # 更新命中计数
        for cls_id in range(self.num_classes):
            self.class_hits[cls_id] = self.class_hits[cls_id] + 1 if cls_id in best_by_class else 0

        # 超时重置
        if self.consensus_class is not None and \
           time.time() - self.consensus_started_at > self.timeout:
            self.reset()

        # 找最佳候选
        best_cls = None
        best_conf = 0.0
        for cls_id, hits in self.class_hits.items():
            if hits >= self.confirm_frames and best_by_class.get(cls_id, 0.0) > best_conf:
                best_cls = cls_id
                best_conf = best_by_class[cls_id]

        if best_cls is not None:
            if self.consensus_class == best_cls:
                self.consensus_confidence = (
                    self.ema_alpha * best_conf +
                    (1.0 - self.ema_alpha) * self.consensus_confidence
                )
                self.consensus_held += 1
                self.release_counter = 0
            else:
                self.consensus_class = best_cls
                self.consensus_confidence = best_conf
                self.consensus_held = 1
                self.consensus_started_at = time.time()
                self.release_counter = 0
            return

        # 无确认 → 累计释放计数
        if self.consensus_class is not None:
            self.release_counter += 1
            if self.release_counter >= self.release_frames:
                self.reset()

    def reset(self) -> None:
        self.consensus_class = None
        self.consensus_confidence = 0.0
        self.consensus_held = 0
        self.consensus_started_at = 0.0
        self.release_counter = 0

    @property
    def active(self) -> bool:
        return self.consensus_class is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.consensus_class,
            "confidence": round(self.consensus_confidence, 4),
            "active": self.active,
            "held_frames": self.consensus_held,
        }


# ============================================================
# RKNN 模型加载
# ============================================================

def load_rknn_model(model_path: str, verbose: bool = True):
    """加载 RKNN 模型，自动尝试 rknnlite（NPU）和 rknn-toolkit2（PC模拟）"""
    # 优先 rknnlite（ARM64 小车）
    try:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        if verbose:
            print(f"[RKNN] Loaded via rknnlite (NPU): {model_path}")
        return rknn, "npu"
    except ImportError:
        pass

    # 回退 rknn-toolkit2（x86_64 PC 模拟器）
    try:
        from rknn.api import RKNN
        rknn = RKNN(verbose=False)
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {ret}")
        ret = rknn.init_runtime(target='rk3588')
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")
        if verbose:
            print(f"[RKNN] Loaded via rknn-toolkit2 (simulator): {model_path}")
        return rknn, "simulator"
    except ImportError:
        pass

    raise RuntimeError("No RKNN runtime available. Install rknnlite or rknn-toolkit2.")


# ============================================================
# 检测结果构建
# ============================================================

def build_detections(boxes, classes, scores, class_names: List[str]) -> List[Dict[str, Any]]:
    if boxes is None:
        return []
    dets = []
    for box, cls_id, score in zip(boxes, classes, scores):
        cls_id = int(cls_id)
        if cls_id < 0 or cls_id >= len(class_names):
            continue
        dets.append({
            "class_id": cls_id,
            "class_name": class_names[cls_id],
            "confidence": float(score),
            "bbox": [float(x) for x in box],
        })
    dets.sort(key=lambda item: item["confidence"], reverse=True)
    return dets


# ============================================================
# 单张推理
# ============================================================

def infer_frame(rknn, frame: np.ndarray, class_names: List[str],
                conf_thresh: float, nms_thresh: float, input_size: int,
                output_shapes_logged: bool = False):
    """对单帧执行推理，返回 boxes/classes/scores 和预处理参数"""
    img, ratio, pad = letterbox(frame, input_size)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    outputs = rknn.inference(inputs=[np.expand_dims(rgb, axis=0)])

    if not output_shapes_logged:
        shapes = [getattr(o, "shape", None) for o in outputs]
        print(f"[RKNN] output shapes: {shapes}")
        output_shapes_logged = True

    boxes, classes, scores = yolov5_post_process(
        outputs, input_size, conf_thresh, nms_thresh, len(class_names))

    if boxes is not None:
        boxes = scale_boxes(boxes, ratio, pad, frame.shape)

    return boxes, classes, scores, output_shapes_logged


# ============================================================
# 标注绘制
# ============================================================

def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]],
                    class_names: List[str], consensus: Optional[ConsensusFilter] = None,
                    inf_time_ms: float = 0.0) -> np.ndarray:
    annot = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        cls_id = int(det["class_id"])
        name = det["class_name"]
        color = DEFAULT_COLORS[cls_id % len(DEFAULT_COLORS)]
        cv2.rectangle(annot, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annot, f"{name} {det['confidence']:.2f}",
                    (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 共识状态
    if consensus is not None:
        if consensus.active:
            text = f"LOCK: {class_names[consensus.consensus_class]} {consensus.consensus_confidence:.2f}"
        else:
            text = "LOCK: none"
        cv2.putText(annot, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if inf_time_ms > 0:
        cv2.putText(annot, f"{inf_time_ms:.0f}ms", (8, annot.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return annot


# ============================================================
# CLI 模式: 图片目录验证
# ============================================================

def validate_directory(rknn, source_dir: str, class_names: List[str],
                       conf_thresh: float, nms_thresh: float,
                       output_dir: Optional[str], input_size: int):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([
        f for f in Path(source_dir).iterdir() if f.suffix.lower() in exts
    ])
    if not image_files:
        print(f"[ERROR] No images in {source_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Model: {class_names} | Images: {len(image_files)}")
    print(f"Conf: {conf_thresh} | NMS: {nms_thresh} | Size: {input_size}")
    print(f"{'='*60}\n")

    out_path = Path(output_dir) if output_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    output_shapes_logged = False
    stats = {name: 0 for name in class_names}
    total_det, total_time, no_detection = 0, 0.0, 0

    for i, img_path in enumerate(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [{i+1}/{len(image_files)}] SKIP: {img_path.name}")
            continue

        t0 = time.time()
        boxes, classes, scores, output_shapes_logged = infer_frame(
            rknn, image, class_names, conf_thresh, nms_thresh, input_size,
            output_shapes_logged)
        elapsed = (time.time() - t0) * 1000
        total_time += elapsed

        dets = build_detections(boxes, classes, scores, class_names)
        for d in dets:
            stats[d["class_name"]] = stats.get(d["class_name"], 0) + 1
        total_det += len(dets)
        if not dets:
            no_detection += 1

        det_str = " | ".join(f"{d['class_name']} {d['confidence']:.2f}" for d in dets[:3])
        if len(dets) > 3:
            det_str += f" ... +{len(dets)-3}"
        print(f"  [{i+1:>3}/{len(image_files)}] {img_path.name:40s} {elapsed:5.0f}ms  "
              f"{'V' if dets else 'x (none)'}  {det_str}")

        if out_path:
            annot = draw_detections(image, dets, class_names, None, elapsed)
            cv2.imwrite(str(out_path / img_path.name), annot)

    print(f"\n{'='*60}")
    print("Report")
    print(f"{'='*60}")
    print(f"Total images:   {len(image_files)}")
    print(f"Total boxes:    {total_det}")
    print(f"No detection:   {no_detection} ({100*no_detection/len(image_files):.1f}%)")
    print(f"Avg time:       {total_time/len(image_files):.1f}ms")
    print()
    for name in class_names:
        bar = "#" * min(stats[name] // 2, 30)
        print(f"  {name:20s}: {stats[name]:>4d}  {bar}")
    print(f"{'='*60}")
    if out_path:
        print(f"\nAnnotated images saved to: {out_path}")


# ============================================================
# CLI 模式: 摄像头实时验证
# ============================================================

def validate_camera(rknn, class_names: List[str], conf_thresh: float,
                    nms_thresh: float, camera_id: int, input_size: int,
                    consensus: bool = False):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera /dev/video{camera_id}")
        return

    cf = ConsensusFilter(len(class_names)) if consensus else None
    mode = "CLI + consensus" if consensus else "CLI"
    print(f"\nCamera validation ({mode}) | Q=quit S=save")
    print(f"Classes: {class_names} | Conf: {conf_thresh}\n")

    frame_count, fps, t_fps = 0, 0.0, time.time()
    output_shapes_logged = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.time()
        boxes, classes, scores, output_shapes_logged = infer_frame(
            rknn, frame, class_names, conf_thresh, nms_thresh, input_size,
            output_shapes_logged)
        elapsed = (time.time() - t0) * 1000

        dets = build_detections(boxes, classes, scores, class_names)
        if cf is not None:
            cf.update(dets)

        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10000.0 / (time.time() - t_fps) / 10
            t_fps = time.time()

        annot = draw_detections(frame, dets, class_names, cf, elapsed)
        cv2.putText(annot, f"FPS: {fps:.1f}", (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Model Validation (Q=quit, S=save)", annot)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"validate_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, annot)
            print(f"  Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# ROS 模式
# ============================================================

def _repair_ros_logging():
    """修复 rospy 日志级别映射"""
    levels = {
        "CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR,
        "WARNING": logging.WARNING, "WARN": logging.WARNING,
        "INFO": logging.INFO, "DEBUG": logging.DEBUG, "NOTSET": logging.NOTSET,
    }
    for name, value in levels.items():
        logging.addLevelName(value, name)
    try:
        import rosgraph.roslogging as roslogging
        mapping = getattr(roslogging, "_logging_to_rospy_names", None)
        if isinstance(mapping, dict):
            # Ensure WARN / FATAL can be looked up by the rospy log handler
            for short, full in {"D": "DEBUG", "I": "INFO", "W": "WARNING",
                                 "E": "ERROR", "F": "CRITICAL"}.items():
                if short not in mapping and full in mapping:
                    mapping[short] = mapping[full]
            # Backfill: if we renamed WARNING -> WARN, ensure mapping has it
            if "WARNING" in mapping and "WARN" not in mapping:
                mapping["WARN"] = mapping["WARNING"]
            if "CRITICAL" in mapping and "FATAL" not in mapping:
                mapping["FATAL"] = mapping["CRITICAL"]
    except Exception:
        pass


class RknnRosNode:
    """RKNN YOLOv5 ROS 推理节点（通用，支持任意类别数）"""

    def __init__(self, class_names: List[str], model_path: str,
                 input_size: int, conf_thresh: float, nms_thresh: float):
        _repair_ros_logging()
        import rospy
        import rospkg
        from cv_bridge import CvBridge
        from std_msgs.msg import String
        from sensor_msgs.msg import Image as RosImage

        self._String = String
        self._RosImage = RosImage

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model_path = self._resolve_model_path(model_path)
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        rospy.init_node("rknn_validate_node")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_stamp = 0.0
        self.shutdown = False
        self.output_shapes_logged = False
        self.inference_ms = 0.0
        self._error_count = 0
        self._max_reloads = 3

        # 读取 ROS params（可被 launch 文件覆盖）
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.detections_topic = rospy.get_param("~detections_topic", "/yolo/detections")
        self.debug_image_topic = rospy.get_param("~debug_image_topic", "/yolo/debug_image")
        self.status_topic = rospy.get_param("~status_topic", "/yolo/status")
        self.inference_rate = float(rospy.get_param("~inference_rate", 10.0))
        self.flip = bool(rospy.get_param("~flip", False))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))
        self.image_timeout = float(rospy.get_param("~image_timeout", 5.0))
        self.confirm_frames = int(rospy.get_param("~consensus_confirm_frames", 3))
        self.release_frames = int(rospy.get_param("~consensus_release_frames", 3))
        self.consensus_timeout = float(rospy.get_param("~consensus_timeout", 1.0))
        self.ema_alpha = float(rospy.get_param("~consensus_ema_alpha", 0.3))
        self.enable_speech = bool(rospy.get_param("~enable_speech", True))
        self.announce_service = rospy.get_param("~announce_service", "/competition_speech/announce")
        self.announce_timeout = float(rospy.get_param("~announce_service_timeout_sec", 0.5))
        self.speak_topic = rospy.get_param("~speak_topic", "/speak")
        self.use_announce_service = bool(rospy.get_param("~use_announce_service", True))
        self.fallback_to_speak_topic = bool(rospy.get_param("~fallback_to_speak_topic", True))
        self.repeat_same = bool(rospy.get_param("~repeat_same", True))
        self.min_speech_interval = float(rospy.get_param("~min_speech_interval_sec", 2.0))
        self.speech_wait = bool(rospy.get_param("~speech_wait", False))
        self.speech_texts = self._load_speech_texts()

        # 共识滤波器
        self.consensus_filter = ConsensusFilter(
            self.num_classes, self.confirm_frames,
            self.release_frames, self.consensus_timeout, self.ema_alpha)
        self.last_spoken_class: Optional[str] = None
        self.last_spoken_at: float = 0.0

        # 加载模型
        self.rknn = self._load_rknn_ros()

        # ROS 接口
        self.speak_pub = rospy.Publisher(self.speak_topic, String, queue_size=1)
        self.det_pub = rospy.Publisher(self.detections_topic, String, queue_size=1)
        self.debug_pub = rospy.Publisher(self.debug_image_topic, RosImage, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=1, latch=True)
        self.image_sub = rospy.Subscriber(
            self.image_topic, RosImage, self._image_cb, queue_size=1, buff_size=2 ** 24)
        rospy.on_shutdown(self._on_shutdown)
        self._publish_status("tracking")
        rospy.loginfo("rknn_validate_node ready: model=%s classes=%s",
                       self.model_path, self.class_names)

    def _resolve_model_path(self, param_path: str) -> str:
        import rospkg
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(param_path)))
        if path and os.path.isfile(path):
            return path
        try:
            yolo_dir = rospkg.RosPack().get_path("yolo")
            default = os.path.join(yolo_dir, "models", "best.rknn")
            if os.path.isfile(default):
                return default
        except Exception:
            pass
        import rospy
        rospy.logfatal("RKNN model not found. Set ~model_path or place best.rknn in yolo/models")
        sys.exit(1)

    def _load_speech_texts(self) -> Dict[str, str]:
        import rospy
        texts = {}
        for name in self.class_names:
            param_key = f"~speech_text_{name}"
            default = f"识别到{name}。"
            texts[name] = rospy.get_param(param_key, default)
        return texts

    def _load_rknn_ros(self):
        import rospy
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:
            _repair_ros_logging()
            rospy.logfatal("Failed to import rknnlite.api: %s", exc)
            sys.exit(1)
        _repair_ros_logging()
        rknn = RKNNLite()
        _repair_ros_logging()
        rospy.loginfo("Loading RKNN model: %s", self.model_path)
        ret = rknn.load_rknn(self.model_path)
        _repair_ros_logging()
        if ret != 0:
            rospy.logfatal("load_rknn failed: %s", ret)
            sys.exit(ret)
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        _repair_ros_logging()
        if ret != 0:
            rospy.logfatal("init_runtime failed: %s", ret)
            sys.exit(ret)
        return rknn

    def _image_cb(self, msg):
        import rospy
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "cv_bridge error: %s", exc)
            return
        if self.flip:
            frame = cv2.flip(frame, 1)
        stamp = msg.header.stamp.to_sec() if msg.header.stamp else time.time()
        with self.lock:
            self.latest_frame = frame
            self.latest_stamp = stamp

    def run(self) -> None:
        import rospy
        rate = rospy.Rate(self.inference_rate)
        while not rospy.is_shutdown() and not self.shutdown:
            started = time.time()
            self._infer_once()
            self.inference_ms = (time.time() - started) * 1000.0
            rate.sleep()

    def _infer_once(self) -> None:
        import rospy
        with self.lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()
            stamp = self.latest_stamp

        if time.time() - stamp > self.image_timeout:
            self._publish_status("no_image")
            self.consensus_filter.reset()
            return

        try:
            boxes, classes, scores, self.output_shapes_logged = infer_frame(
                self.rknn, frame, self.class_names,
                self.conf_thresh, self.nms_thresh, self.input_size,
                self.output_shapes_logged)
            self._error_count = 0  # 成功则清零
        except Exception as exc:
            _repair_ros_logging()
            self._error_count += 1
            rospy.logerr_throttle(2.0, "RKNN inference failed (#%d/%d): %s",
                                  self._error_count, self._max_reloads, exc)
            if self._error_count <= self._max_reloads:
                self._publish_status("error")
                self._reload_model()
            else:
                self._publish_status("persistent_error")
                rospy.logerr_throttle(2.0,
                    "Max reloads (%d) exceeded. Sleeping; check model/config.",
                    self._max_reloads)
            return

        dets = build_detections(boxes, classes, scores, self.class_names)
        previous = self.consensus_filter.consensus_class
        self.consensus_filter.update(dets)
        self._publish_detections(dets, stamp)
        self._publish_debug_image(frame, dets)
        if self.consensus_filter.consensus_class != previous or self.repeat_same:
            self._maybe_speak()
        self._publish_status("tracking")

    def _reload_model(self) -> None:
        import rospy
        rospy.logwarn("Attempting model reload...")
        try:
            if self.rknn is not None:
                self.rknn.release()
        except Exception:
            pass
        try:
            self.rknn = self._load_rknn_ros()
            self.output_shapes_logged = False
            rospy.loginfo("Model reloaded successfully")
        except Exception as exc:
            rospy.logerr("Model reload failed: %s", exc)

    def _publish_detections(self, dets: List[Dict[str, Any]], stamp: float) -> None:
        import rospy
        payload = {
            "header": {"stamp": stamp},
            "raw_detections": [
                {"class_name": d["class_name"], "class_id": d["class_id"],
                 "confidence": round(d["confidence"], 4),
                 "bbox": [round(x, 1) for x in d["bbox"]]}
                for d in dets
            ],
            "consensus": {
                "class_name": self.class_names[self.consensus_filter.consensus_class]
                if self.consensus_filter.active else None,
                **self.consensus_filter.to_dict(),
            },
            "diagnostics": {
                "fps": round(1000.0 / max(self.inference_ms, 1.0), 1),
                "inference_ms": round(self.inference_ms, 1),
            },
        }
        self.det_pub.publish(self._String(
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))

    def _publish_debug_image(self, frame: np.ndarray, dets: List[Dict[str, Any]]) -> None:
        import rospy
        if not self.publish_debug:
            return
        annot = draw_detections(frame, dets, self.class_names, self.consensus_filter, self.inference_ms)
        try:
            msg = self.bridge.cv2_to_imgmsg(annot, "bgr8")
            msg.header.stamp = rospy.Time.now()
            self.debug_pub.publish(msg)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "debug image publish failed: %s", exc)

    def _maybe_speak(self) -> None:
        import rospy
        if not self.enable_speech or not self.consensus_filter.active:
            return
        class_name = self.class_names[self.consensus_filter.consensus_class]
        now = time.time()
        if not self.repeat_same and class_name == self.last_spoken_class:
            return
        if now - self.last_spoken_at < self.min_speech_interval:
            return
        self.last_spoken_class = class_name
        self.last_spoken_at = now
        text = self.speech_texts.get(class_name, f"识别到{class_name}。")
        if self._try_announce_service(class_name, text):
            return
        if self.fallback_to_speak_topic:
            rospy.logwarn("Fallback to %s: %s", self.speak_topic, text)
            self.speak_pub.publish(self._String(data=text))

    def _try_announce_service(self, class_name: str, text: str) -> bool:
        import rospy
        if not self.use_announce_service:
            return False
        try:
            from ucar_2026_competition_speech.srv import Announce
        except Exception:
            return False
        try:
            rospy.wait_for_service(self.announce_service, timeout=self.announce_timeout)
            announce = rospy.ServiceProxy(self.announce_service, Announce)
            res = announce("custom", "", "", class_name, text, self.speech_wait)
            if bool(res.success):
                rospy.loginfo("Announced via %s: %s", self.announce_service, res.speech_text)
                return True
            rospy.logwarn("Announce service failed: %s", res.message)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Announce service unavailable: %s", exc)
        return False

    def _publish_status(self, status: str) -> None:
        self.status_pub.publish(self._String(data=status))

    def _on_shutdown(self) -> None:
        self.shutdown = True
        self._publish_status("shutdown")
        try:
            if self.rknn is not None:
                self.rknn.release()
        except Exception:
            pass


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RKNN YOLOv5 模型验证与 ROS 推理节点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # CLI image directory
  python3 validate_model.py --model models/factory_sign_3cls.rknn \\
      --classes food,electronic,daily --source ./test_images/

  # CLI camera with consensus filtering
  python3 validate_model.py --model models/best.rknn \\
      --classes green_left,green_right,green_straight,red_light \\
      --source camera --consensus

  # ROS node (params from launch file)
  python3 validate_model.py --ros --model models/factory_sign_3cls.rknn \\
      --classes food,electronic,daily
""")
    parser.add_argument("--model", required=True, help="RKNN model path")
    parser.add_argument("--classes", required=True,
                        help="Class names, comma-separated, e.g.: food,electronic,daily")
    parser.add_argument("--source", default="camera",
                        help="Image directory path, or 'camera' (default: camera)")
    parser.add_argument("--output", default=None, help="Output dir for annotated images (CLI dir mode)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--img-size", type=int, default=640, help="Model input size (default: 640)")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--ros", action="store_true", help="Run as ROS node")
    parser.add_argument("--consensus", action="store_true",
                        help="Enable consensus filtering in CLI camera mode")

    args = parser.parse_args()
    class_names = [s.strip() for s in args.classes.split(",")]

    if args.ros:
        # ROS 模式: model_path 可为空，由 ROS param 或自动发现提供
        node = RknnRosNode(class_names, args.model, args.img_size, args.conf, args.nms)
        node.run()
        return

    # CLI 模式
    print(f"Loading: {args.model}")
    print(f"Classes: {class_names}")
    rknn, runtime = load_rknn_model(args.model)

    try:
        if args.source.lower() == "camera":
            validate_camera(rknn, class_names, args.conf, args.nms,
                            args.camera_id, args.img_size, args.consensus)
        else:
            if not os.path.isdir(args.source):
                print(f"[ERROR] Directory not found: {args.source}")
                sys.exit(1)
            validate_directory(rknn, args.source, class_names, args.conf, args.nms,
                               args.output, args.img_size)
    finally:
        try:
            rknn.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
