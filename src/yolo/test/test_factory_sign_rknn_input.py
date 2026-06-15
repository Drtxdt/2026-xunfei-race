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


if __name__ == "__main__":
    unittest.main()
