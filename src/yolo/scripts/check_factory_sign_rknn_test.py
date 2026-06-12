#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
厂区标识牌 RKNN 推理节点诊断脚本
================================
检查模型文件、依赖库、ROS 话题、推理功能是否正常。

用法:
  python3 check_factory_sign_rknn_test.py
  python3 check_factory_sign_rknn_test.py --model models/factory_sign_3cls.rknn
  python3 check_factory_sign_rknn_test.py --ros         # 检查 ROS 节点和话题
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

CLASS_NAMES = ["food", "electronic", "daily"]
MODEL_CANDIDATES = [
    "models/factory_sign_3cls.rknn",
    "models/best.rknn",
]
TEST_IMAGE_SIZE = 640


def green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def ok(msg: str) -> None:
    print(f"  {green('[OK]')} {msg}")


def fail(msg: str) -> None:
    print(f"  {red('[FAIL]')} {msg}")


def warn(msg: str) -> None:
    print(f"  {yellow('[WARN]')} {msg}")


def info(msg: str) -> None:
    print(f"  [INFO] {msg}")


def find_package_path() -> Optional[str]:
    """查找 yolo 包路径"""
    try:
        import rospkg
        return rospkg.RosPack().get_path("yolo")
    except Exception:
        pass
    # 回退: 从脚本位置推断
    script_dir = Path(__file__).resolve().parent
    for candidate in [
        script_dir.parent,
        script_dir.parent.parent / "yolo",
    ]:
        if (candidate / "package.xml").exists():
            return str(candidate)
    return None


def check_package(pkg_path: Optional[str]) -> bool:
    print("\n--- 1. 包路径 ---")
    if pkg_path is None:
        fail("Cannot find yolo package path")
        return False
    ok(f"yolo package: {pkg_path}")

    # 检查关键文件
    checks = [
        ("package.xml", "Package manifest"),
        ("CMakeLists.txt", "Build config"),
        ("validate_model.py", "Inference node"),
        ("models", "Models directory"),
        ("config/factory_sign_rknn_test.yaml", "Config file"),
        ("launch/factory_sign_rknn_test.launch", "Launch file"),
    ]
    all_ok = True
    for rel_path, desc in checks:
        full = os.path.join(pkg_path, rel_path)
        if os.path.exists(full):
            ok(f"{desc}: {rel_path}")
        else:
            if rel_path in ("config/factory_sign_rknn_test.yaml", "launch/factory_sign_rknn_test.launch"):
                warn(f"{desc}: {rel_path} (not created yet)")
            else:
                fail(f"{desc}: {rel_path}")
                all_ok = False
    return all_ok


def check_model(pkg_path: str, model_arg: Optional[str]) -> Optional[str]:
    print("\n--- 2. 模型文件 ---")
    if model_arg and os.path.isfile(model_arg):
        model_path = os.path.abspath(model_arg)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        ok(f"Model: {model_path} ({size_mb:.1f} MB)")
        return model_path

    for cand in MODEL_CANDIDATES:
        full = os.path.join(pkg_path, cand)
        if os.path.isfile(full):
            size_mb = os.path.getsize(full) / (1024 * 1024)
            ok(f"Model found: {cand} ({size_mb:.1f} MB)")
            return full

    fail("No .rknn model found")
    info("Place factory_sign_3cls.rknn in yolo/models/ or use --model")
    return None


def check_rknn_import() -> bool:
    print("\n--- 3. RKNN 依赖 ---")
    try:
        from rknnlite.api import RKNNLite
        ok("rknnlite.api.RKNNLite importable (NPU)")
        return True
    except ImportError:
        fail("rknnlite not available (not on the car?)")
        pass

    try:
        from rknn.api import RKNN
        ok("rknn-toolkit2.api.RKNN importable (simulator)")
        return True
    except ImportError:
        fail("rknn-toolkit2 not available")

    return False


def check_model_load(model_path: str) -> bool:
    print("\n--- 4. 模型加载 ---")
    try:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
    except ImportError:
        try:
            from rknn.api import RKNN
            rknn = RKNN(verbose=False)
        except ImportError:
            fail("Cannot import RKNN")
            return False

    try:
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            fail(f"load_rknn returned {ret}")
            return False
        ok("load_rknn: success")

        try:
            ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        except Exception:
            try:
                ret = rknn.init_runtime(target='rk3588')
            except Exception:
                pass

        if ret == 0:
            ok("init_runtime: success")
        else:
            warn(f"init_runtime returned {ret} (may be OK on PC simulator)")

        # 模型信息
        try:
            info(f"SDK version: {rknn.get_sdk_version()}")
        except Exception:
            pass

        rknn.release()
        ok("Model released cleanly")
    except Exception as exc:
        fail(f"Model load failed: {exc}")
        return False
    return True


def check_inference(model_path: str) -> bool:
    print("\n--- 5. 推理测试 ---")
    try:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        rknn.load_rknn(model_path)
        rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    except ImportError:
        from rknn.api import RKNN
        rknn = RKNN(verbose=False)
        rknn.load_rknn(model_path)
        rknn.init_runtime(target='rk3588')

    # 生成测试图像（纯黑）
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    t0 = time.time()
    outputs = rknn.inference(inputs=[test_img])
    elapsed = (time.time() - t0) * 1000

    shapes = [getattr(o, "shape", None) for o in outputs]
    info(f"Output shapes: {shapes}")
    ok(f"Inference: {elapsed:.1f}ms")

    # 检查输出形状
    output0 = np.asarray(outputs[0]).squeeze()
    if output0.ndim == 2 and output0.shape[1] == 5 + len(CLASS_NAMES):
        info(f"Detected flat output ({output0.shape[0]} boxes, {output0.shape[1]} dims)")
    elif output0.ndim == 3:
        info(f"Detected 3-head output (H={output0.shape[0]}, W={output0.shape[1]}, C={output0.shape[2]})")
    else:
        warn(f"Unexpected output shape: {output0.shape}")

    rknn.release()
    ok("Inference test passed")
    return True


def check_camera() -> bool:
    print("\n--- 6. 摄像头 ---")
    for dev_id in range(4):
        cap = cv2.VideoCapture(dev_id)
        if cap.isOpened():
            ok(f"/dev/video{dev_id} available")
            ret, frame = cap.read()
            if ret:
                ok(f"Frame read: {frame.shape[1]}x{frame.shape[0]}")
            else:
                warn("Frame read failed")
            cap.release()
            return True
        cap.release()
    warn("No camera found (/dev/video0-3)")
    return True  # not fatal


def check_ros_topics() -> bool:
    print("\n--- 7. ROS 节点与话题 ---")
    try:
        import rospy
    except ImportError:
        fail("rospy not available")
        return False

    try:
        node_names = rospy.get_node_names()
    except Exception:
        warn("ROS master not running? Skip ROS checks")
        return True

    # 检查相关节点
    checks = [
        ("/usb_cam", "USB camera driver"),
        ("/voice_speak_node", "TTS speech node"),
        ("/factory_sign_rknn_test_node", "Factory sign inference node"),
        ("/traffic_light_rknn_test_node", "Traffic light inference node"),
    ]
    for node_name, desc in checks:
        found = any(node_name in n for n in node_names)
        if found:
            ok(f"{desc}: {node_name}")
        else:
            info(f"{desc}: not running")

    # 检查话题
    try:
        topics = dict(rospy.get_published_topics())
        topic_checks = [
            ("/usb_cam/image_raw", "Camera image"),
            ("/factory_sign_rknn_test/detections", "Factory sign detections"),
            ("/factory_sign_rknn_test/debug_image", "Factory sign debug image"),
            ("/factory_sign_rknn_test/status", "Factory sign status"),
            ("/speak", "Speech output"),
        ]
        for topic, desc in topic_checks:
            if topic in topics:
                ok(f"{desc}: {topic} ({topics[topic]})")
            else:
                info(f"{desc}: not published")
    except Exception:
        pass
    return True


def check_detection_sample(model_path: str) -> bool:
    """用带标识的模拟图像测试检测输出格式"""
    print("\n--- 8. 检测输出格式 ---")
    try:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        rknn.load_rknn(model_path)
        rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    except ImportError:
        from rknn.api import RKNN
        rknn = RKNN(verbose=False)
        rknn.load_rknn(model_path)
        rknn.init_runtime(target='rk3588')

    # 生成一张有简单矩形图案的测试图
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (100, 100, 300, 300), (128, 128, 128), -1)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    outputs = rknn.inference(inputs=[test_img_rgb])
    rknn.release()

    # 验证输出格式
    output0 = np.asarray(outputs[0]).squeeze()
    num_classes = len(CLASS_NAMES)
    if output0.ndim == 2 and output0.shape[1] == 5 + num_classes:
        ok(f"Flat output: {output0.shape[0]} candidates × (5+{num_classes}) dims")
    elif output0.ndim == 3 and output0.shape[2] == 3 * (5 + num_classes):
        ok(f"First head HWC: {output0.shape}")
    else:
        warn(f"Unexpected format: ndim={output0.ndim} shape={output0.shape}")
        info("Model classes may differ from expected 3 (food, electronic, daily)")

    # 打印几条样例输出
    flat = output0.reshape(-1, output0.shape[-1]) if output0.ndim <= 3 else output0[:5]
    info(f"Sample output (first 5 rows):\n{flat[:5]}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Factory Sign RKNN Diagnostics")
    parser.add_argument("--model", default=None, help="RKNN model path (auto-find if empty)")
    parser.add_argument("--ros", action="store_true", help="Check ROS nodes and topics")
    parser.add_argument("--all", action="store_true", help="Run all checks (default)")
    args = parser.parse_args()
    run_all = args.all or (not args.ros)  # default: run all except ROS unless --ros

    print("=" * 55)
    print("  Factory Sign RKNN Test Diagnostics")
    print("  Classes:", CLASS_NAMES)
    print("=" * 55)

    pkg_path = find_package_path()
    all_ok = True

    # 1. Package
    if not check_package(pkg_path):
        all_ok = False

    # 2. Model file
    model_path = check_model(pkg_path or ".", args.model)
    if model_path is None:
        print(f"\n{red('Model not found. Abort further checks.')}")
        sys.exit(1)

    # 3. RKNN import
    if not check_rknn_import():
        all_ok = False

    # 4. Model load
    if not check_model_load(model_path):
        all_ok = False

    # 5. Inference
    if not check_inference(model_path):
        all_ok = False

    # 6. Camera
    check_camera()

    # 7. ROS topics (only with --ros or --all)
    if args.ros or args.all:
        check_ros_topics()

    # 8. Detection format
    check_detection_sample(model_path)

    print(f"\n{'=' * 55}")
    if all_ok:
        print(green("  All critical checks passed."))
    else:
        print(red("  Some checks FAILED. Review above."))
    print(f"{'=' * 55}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
