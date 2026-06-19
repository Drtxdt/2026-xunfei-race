import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def install_ros_stubs():
    rospy = types.ModuleType("rospy")
    rospy.get_param = lambda name, default=None: default
    rospy.init_node = lambda *a, **k: None
    rospy.logfatal = lambda *a, **k: None
    rospy.loginfo = lambda *a, **k: None
    rospy.loginfo_throttle = lambda *a, **k: None
    rospy.logwarn = lambda *a, **k: None
    rospy.logwarn_throttle = lambda *a, **k: None
    rospy.logerr_throttle = lambda *a, **k: None
    rospy.on_shutdown = lambda *a, **k: None
    rospy.is_shutdown = lambda: True
    rospy.Time = types.SimpleNamespace(now=lambda: None)
    rospy.Rate = lambda hz: types.SimpleNamespace(sleep=lambda: None)
    rospy.Publisher = lambda *a, **k: types.SimpleNamespace(publish=lambda *a, **k: None)
    rospy.Subscriber = lambda *a, **k: None
    rospy.wait_for_service = lambda *a, **k: None
    rospy.ServiceProxy = lambda *a, **k: None
    sys.modules["rospy"] = rospy

    rospkg = types.ModuleType("rospkg")
    rospkg.RosPack = lambda: types.SimpleNamespace(get_path=lambda name: str(ROOT))
    sys.modules["rospkg"] = rospkg

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = object
    cv_bridge.CvBridgeError = Exception
    sys.modules["cv_bridge"] = cv_bridge

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = object
    sys.modules["sensor_msgs"] = sensor_msgs
    sys.modules["sensor_msgs.msg"] = sensor_msgs_msg

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.String = type("String", (), {"__init__": lambda self, data="": setattr(self, "data", data)})
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs_msg

    speech_pkg = types.ModuleType("ucar_2026_competition_speech")
    speech_srv = types.ModuleType("ucar_2026_competition_speech.srv")
    speech_srv.Announce = object
    sys.modules["ucar_2026_competition_speech"] = speech_pkg
    sys.modules["ucar_2026_competition_speech.srv"] = speech_srv


install_ros_stubs()
import signboard_rknn_test_node as node  # noqa: E402


def test_class_names_match_checkpoint_output_order():
    assert node.CLASS_NAMES == ["daily_necessities", "electronics", "food_processing"]
    assert node.SPEECH_TEXT["daily_necessities"] == "日用品加工车间。"
    assert node.SPEECH_TEXT["electronics"] == "电子产品生产车间。"
    assert node.SPEECH_TEXT["food_processing"] == "食品加工车间。"


def test_classify_logits_returns_raw_probs_argmax_and_name():
    result = node.classify_logits(np.array([3.0, 1.0, 0.0], dtype=np.float32))

    assert result["class_id"] == 0
    assert result["class_name"] == "daily_necessities"
    assert result["confidence"] > 0.8
    assert result["logits"] == [3.0, 1.0, 0.0]
    assert set(result["probs"].keys()) == {"daily_necessities", "electronics", "food_processing"}


def test_parse_fixed_roi_accepts_xywh_and_disabled_values():
    assert node.parse_fixed_roi("") is None
    assert node.parse_fixed_roi("none") is None
    assert node.parse_fixed_roi("10,20,30,40") == (10, 20, 40, 60)


def test_crop_fixed_roi_clamps_to_frame_bounds():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    roi, box = node.crop_fixed_roi(frame, (-10, 20, 250, 120))

    assert box == (0, 20, 199, 99)
    assert roi.shape == (79, 199, 3)
