#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import os
import sys
import types
import unittest

import numpy as np


def _install_ros_stubs():
    rospy = types.ModuleType("rospy")
    rospy.get_param = lambda name, default=None: default
    rospy.logfatal = lambda *args, **kwargs: None
    sys.modules.setdefault("rospy", rospy)

    rospkg = types.ModuleType("rospkg")
    rospkg.RosPack = lambda: types.SimpleNamespace(get_path=lambda package: "")
    sys.modules.setdefault("rospkg", rospkg)

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = object
    cv_bridge.CvBridgeError = Exception
    sys.modules.setdefault("cv_bridge", cv_bridge)

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = object
    sensor_msgs.msg = sensor_msgs_msg
    sys.modules.setdefault("sensor_msgs", sensor_msgs)
    sys.modules.setdefault("sensor_msgs.msg", sensor_msgs_msg)

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = object
    std_msgs.msg = std_msgs_msg
    sys.modules.setdefault("std_msgs", std_msgs)
    sys.modules.setdefault("std_msgs.msg", std_msgs_msg)


def _load_node_module():
    _install_ros_stubs()
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "factory_sign_rknn_test_node.py",
    )
    spec = importlib.util.spec_from_file_location("factory_sign_rknn_test_node", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FactorySignRknnInputTest(unittest.TestCase):
    def test_repair_logging_levels_backfills_warn_mapping(self):
        rosgraph = types.ModuleType("rosgraph")
        roslogging = types.ModuleType("rosgraph.roslogging")
        roslogging._logging_to_rospy_names = {
            "DEBUG": "debug",
            "INFO": "info",
            "WARNING": "warn",
            "ERROR": "error",
            "CRITICAL": "fatal",
        }
        rosgraph.roslogging = roslogging
        sys.modules["rosgraph"] = rosgraph
        sys.modules["rosgraph.roslogging"] = roslogging

        module = _load_node_module()
        module.repair_logging_levels()

        self.assertEqual("warn", roslogging._logging_to_rospy_names["WARN"])
        self.assertEqual("warn", roslogging._logging_to_rospy_names["W"])

    def test_prepare_rknn_input_adds_batch_dimension(self):
        module = _load_node_module()
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        tensor = module.prepare_rknn_input(image)

        self.assertEqual((1, 640, 640, 3), tensor.shape)
        self.assertEqual(np.uint8, tensor.dtype)
        np.testing.assert_array_equal(image, tensor[0])

    def test_prepare_rknn_input_leaves_existing_batch_dimension(self):
        module = _load_node_module()
        image = np.zeros((1, 640, 640, 3), dtype=np.uint8)

        tensor = module.prepare_rknn_input(image)

        self.assertIs(tensor, image)

    def test_single_output_post_accepts_model_with_extra_class_output(self):
        module = _load_node_module()
        node = object.__new__(module.FactorySignRknnTestNode)
        node.conf_thresh = 0.3
        node.nms_iou = 0.45
        node.model_class_indices = [0, 1, 2]
        node.last_model_top_scores = []

        arr = np.zeros((25200, 9), dtype=np.float32)
        arr[0, :4] = [320.0, 320.0, 100.0, 80.0]
        arr[0, 4] = 0.9
        arr[0, 5:9] = [0.8, 0.1, 0.05, 0.99]

        boxes, classes, scores, model_size = node.single_output_post(arr)

        self.assertEqual(640, model_size)
        self.assertEqual([0], classes.tolist())
        self.assertEqual((1, 4), boxes.shape)
        self.assertAlmostEqual(0.72, float(scores[0]), places=5)
        self.assertEqual(3, node.last_model_top_scores[0][0])

    def test_single_output_post_can_map_fourth_model_class_to_daily(self):
        module = _load_node_module()
        node = object.__new__(module.FactorySignRknnTestNode)
        node.conf_thresh = 0.3
        node.nms_iou = 0.45
        node.model_class_indices = [0, 1, 3]
        node.last_model_top_scores = []

        arr = np.zeros((25200, 9), dtype=np.float32)
        arr[0, :4] = [320.0, 320.0, 100.0, 80.0]
        arr[0, 4] = 0.9
        arr[0, 5:9] = [0.1, 0.1, 0.05, 0.99]

        boxes, classes, scores, model_size = node.single_output_post(arr)

        self.assertEqual(640, model_size)
        self.assertEqual([2], classes.tolist())
        self.assertEqual((1, 4), boxes.shape)
        self.assertAlmostEqual(0.891, float(scores[0]), places=5)


if __name__ == "__main__":
    unittest.main()
