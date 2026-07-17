#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import actionlib
import tf
import dynamic_reconfigure.client
import json
import math
import os
import threading
import sys

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Quaternion, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool, String

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from navigator_logic import (
    build_observation_candidates,
    center_angular_command,
    center_step_angle,
    footprint_max_cost,
    normalize_angle,
    parking_footprint_inside,
    scan_dwell_deadline,
    split_scan_angle,
)


def euler_to_quaternion(yaw):
    """将 yaw 角转换为 geometry_msgs/Quaternion"""
    q = tf.transformations.quaternion_from_euler(0, 0, yaw)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def quaternion_to_yaw(q):
    """从四元数提取 yaw 角，支持 geometry_msgs/Quaternion 或 (x, y, z, w) 列表/元组"""
    if isinstance(q, (list, tuple)):
        x, y, z, w = q
    else:
        x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


class VisionTriggeredNavigator(object):
    """
    三阶段导航节点：
      1. 巡航（依次访问预标点，costmap 实时判断可行性，跳点，到站自转）
      2. 视觉触发（键盘或视觉话题触发，射线与长方形围墙求交，内法向偏移 0.4m）
      3. 导航到结束点
    """

    def __init__(self):
        rospy.init_node("vision_triggered_navigator")

        # ---------- 参数读取 ----------
        # 参考坐标系
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")

        # 视觉触发长方形围墙（4 个角点，带手标误差，程序取 min/max 包围盒）
        # 目标点位于墙内侧 vision_offset 处，车头垂直指向墙外
        self.vision_rect_corners = rospy.get_param(
            "~vision_rect_corners",
            [[-2.2311, -1.2505],
             [2.8000, -1.1940],
             [-2.2197, -3.2746],
             [2.7739, -3.2186]])
        self._build_rect_bounds()

        # 视觉目标安全偏移距离
        self.vision_offset = rospy.get_param("~vision_offset", 0.4)

        # 触发模式："keyboard" 或 "vision"
        self.trigger_mode = rospy.get_param("~trigger_mode", "keyboard")
        self.vision_topic = rospy.get_param("~vision_topic", "/vision/detected")
        self.status_topic = rospy.get_param(
            "~status_topic", "/vision_triggered_navigator/status")
        self.navigate_to_end_after_trigger = rospy.get_param(
            "~navigate_to_end_after_trigger", True)

        # 任务2专用的覆盖优先模式。默认关闭，保证原独立导航行为不变。
        self.coverage_search_mode = rospy.get_param("~coverage_search_mode", False)
        self.target_topic = rospy.get_param("~target_topic", "/vision/target")
        self.coverage_candidate_offsets = rospy.get_param(
            "~coverage_candidate_offsets",
            [[0.0, 0.0], [0.0, 0.28], [0.0, -0.28],
             [0.28, 0.0], [-0.28, 0.0]])
        self.coverage_min_wall_clearance = rospy.get_param(
            "~coverage_min_wall_clearance", 0.36)
        self.coverage_accept_radius = rospy.get_param("~coverage_accept_radius", 0.45)
        self.coverage_goal_timeout = rospy.get_param("~coverage_goal_timeout_sec", 25.0)
        self.coverage_stall_timeout = rospy.get_param("~coverage_stall_timeout_sec", 6.0)
        self.coverage_min_progress = rospy.get_param("~coverage_min_progress_m", 0.05)
        self.coverage_revisit_passes = int(rospy.get_param("~coverage_revisit_passes", 1))
        self.max_coverage_anchors = int(rospy.get_param("~max_coverage_anchors", 0))
        self.center_only = rospy.get_param("~center_only", False)
        self.coverage_scan_settle = rospy.get_param("~coverage_scan_settle_sec", 0.35)
        self.coverage_scan_step_angle = math.radians(max(
            1.0, float(rospy.get_param("~coverage_scan_step_deg", 20.0))))
        self.coverage_scan_angular_speed = abs(float(rospy.get_param(
            "~coverage_scan_angular_speed", 0.35)))
        self.coverage_scan_dwell = max(0.0, float(rospy.get_param(
            "~coverage_scan_dwell_sec", 0.65)))
        self.coverage_candidate_hold = max(0.0, float(rospy.get_param(
            "~coverage_candidate_hold_sec", 1.2)))
        self.coverage_scan_max_dwell = max(
            self.coverage_scan_dwell,
            float(rospy.get_param("~coverage_scan_max_dwell_sec", 2.0)))
        self.coverage_scan_pose_timeout = max(0.1, float(rospy.get_param(
            "~coverage_scan_pose_timeout_sec", 0.5)))
        self.coverage_scan_step_timeout_margin = max(0.1, float(rospy.get_param(
            "~coverage_scan_step_timeout_margin_sec", 2.0)))
        self.robot_footprint_radius = rospy.get_param("~robot_footprint_radius", 0.215)
        self.lethal_cost = int(rospy.get_param("~lethal_cost", 253))

        # OCR命中后先将目标框居中，再使用车头射线计算最终停泊点。
        self.target_center_tolerance = rospy.get_param("~target_center_tolerance", 0.08)
        self.target_center_required_hits = int(rospy.get_param(
            "~target_center_required_hits", 2))
        self.target_center_timeout = rospy.get_param("~target_center_timeout_sec", 12.0)
        self.target_bbox_stale = rospy.get_param("~target_bbox_stale_sec", 0.8)
        self.target_center_min_speed = rospy.get_param("~target_center_min_speed", 0.08)
        self.target_center_max_speed = rospy.get_param("~target_center_max_speed", 0.18)
        self.target_center_steering_sign = rospy.get_param(
            "~target_center_steering_sign", -1.0)
        self.target_center_coarse_step = math.radians(abs(float(rospy.get_param(
            "~target_center_coarse_step_deg", 4.0))))
        self.target_center_fine_step = math.radians(abs(float(rospy.get_param(
            "~target_center_fine_step_deg", 2.0))))
        self.target_center_fine_threshold = abs(float(rospy.get_param(
            "~target_center_fine_threshold", 0.20)))
        self.target_center_start_speed = abs(float(rospy.get_param(
            "~target_center_start_speed", 0.20)))
        self.target_center_step_max_speed = max(
            self.target_center_start_speed,
            abs(float(rospy.get_param("~target_center_step_max_speed", 0.35))))
        self.target_center_speed_increment = abs(float(rospy.get_param(
            "~target_center_speed_increment", 0.05)))
        self.target_center_motion_window = max(0.1, float(rospy.get_param(
            "~target_center_motion_window_sec", 0.6)))
        self.target_center_min_progress = math.radians(abs(float(rospy.get_param(
            "~target_center_min_progress_deg", 0.5))))
        self.target_center_settle = max(0.0, float(rospy.get_param(
            "~target_center_settle_sec", 0.25)))
        self.target_center_reverse_threshold = abs(float(rospy.get_param(
            "~target_center_reverse_threshold", 0.03)))
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.odom_stale = max(0.1, float(rospy.get_param(
            "~odom_stale_sec", 0.5)))
        self.camera_boresight_yaw_offset = rospy.get_param(
            "~camera_boresight_yaw_offset", 0.0)
        self.arrival_hold_sec = rospy.get_param("~arrival_hold_sec", 0.6)

        # 任务2专用的50cm停车框。独立导航默认不启用验证并继续使用vision_offset。
        self.validate_parking_box = rospy.get_param("~validate_parking_box", False)
        self.parking_box_width = abs(float(rospy.get_param("~parking_box_width", 0.50)))
        self.parking_box_depth = abs(float(rospy.get_param("~parking_box_depth", 0.50)))
        self.parking_goal_offset = abs(float(rospy.get_param(
            "~parking_goal_offset", self.vision_offset)))
        self.parking_xy_tolerance = abs(float(rospy.get_param(
            "~parking_xy_tolerance", 0.04)))
        self.parking_yaw_tolerance = abs(float(rospy.get_param(
            "~parking_yaw_tolerance", 0.06)))
        self.parking_validation_margin = max(0.0, float(rospy.get_param(
            "~parking_validation_margin", 0.01)))
        self.footprint_half_length = abs(float(rospy.get_param(
            "~footprint_half_length", 0.171)))
        self.footprint_half_width = abs(float(rospy.get_param(
            "~footprint_half_width", 0.128)))
        self.local_planner_reconfigure_ns = rospy.get_param(
            "~local_planner_reconfigure_ns", "/move_base/TebLocalPlannerROS")

        # move_base 与 costmap
        self.move_base_server = rospy.get_param("~move_base_server", "/move_base")
        self.costmap_topic = rospy.get_param("~costmap_topic", "/move_base/local_costmap/costmap")
        self.cost_threshold = rospy.get_param("~cost_threshold", 100)
        self.feasibility_check_rate = rospy.get_param("~feasibility_check_rate", 1.0)

        # 自转参数
        self.rotation_speed = abs(rospy.get_param("~rotation_speed", 0.5))  # 左转为正，rad/s
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")

        # 巡航点与结束点
        self.patrol_points = rospy.get_param("~patrol_points", [])
        self.end_goal = rospy.get_param("~end_goal",
                                        {"x": 0.3195, "y": -3.2703, "yaw": -1.5596})

        # 初始位姿（用于 AMCL 2D Pose Estimate）
        self.publish_initial_pose = rospy.get_param("~publish_initial_pose", True)
        self.initial_pose = rospy.get_param("~initial_pose", {"x": 0.0, "y": 0.0, "yaw": 0.0})

        # ---------- ROS 通信 ----------
        self.tf_listener = tf.TransformListener()
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=10, latch=True)

        self.costmap = None
        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self._costmap_cb)
        self.odom_yaw = None
        self.odom_received_at = 0.0
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)

        self.target_error = None
        self.target_payload_at = 0.0
        self.last_target_payload = None
        if self.coverage_search_mode:
            rospy.Subscriber(self.target_topic, String, self._target_cb, queue_size=10)

        self.triggered = False
        if self.trigger_mode == "vision":
            rospy.Subscriber(self.vision_topic, Bool, self._vision_cb)
            rospy.loginfo("[vision_triggered_navigator] 触发模式：视觉话题 <%s>", self.vision_topic)
        elif self.trigger_mode == "keyboard":
            t = threading.Thread(target=self._keyboard_thread)
            t.daemon = True
            t.start()
            rospy.loginfo("[vision_triggered_navigator] 触发模式：键盘回车")
        else:
            rospy.logerr("[vision_triggered_navigator] 未知触发模式 '%s'，仅支持 keyboard/vision", self.trigger_mode)

        # 连接 move_base action server
        self.move_base_client = actionlib.SimpleActionClient(self.move_base_server, MoveBaseAction)
        rospy.loginfo("[vision_triggered_navigator] 等待 move_base 服务器...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("[vision_triggered_navigator] move_base 服务器已连接.")

        # 当前目标记录（用于定时器检查）
        self.current_goal_x = None
        self.current_goal_y = None
        self.current_goal_infeasible = False
        self.current_goal_near_enough = False
        self.current_goal_timed_out = False
        self.parking_wall_point = None
        self.parking_inward_normal = None
        self._planner_client = None
        self._saved_planner_tolerances = None
        rospy.on_shutdown(self._restore_final_tolerances)
        self._publish_status("ready")

    # ------------------------------------------------------------------
    # 回调与工具函数
    # ------------------------------------------------------------------
    def _costmap_cb(self, msg):
        """保存最新 costmap"""
        self.costmap = msg

    def _odom_cb(self, msg):
        """Keep a fresh, local yaw source for closed-loop centering steps."""
        self.odom_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.odom_received_at = rospy.get_time()

    def _publish_status(self, status):
        """发布简洁、稳定的流程状态，供比赛总控监听。"""
        self.status_pub.publish(String(data=status))

    def _vision_cb(self, msg):
        """视觉触发回调"""
        if msg.data and not self.triggered:
            rospy.loginfo("[vision_triggered_navigator] 收到视觉触发信号，打断当前导航.")
            self.triggered = True
            self._publish_status("triggered")
            self.cancel_goal()

    def _target_cb(self, msg):
        """保存当前目标厂牌在完整图像中的水平位置。"""
        try:
            payload = json.loads(msg.data)
            center_x = float(payload.get("target_center_x"))
            image_width = float(payload.get("image_width"))
            if image_width <= 1.0:
                return
            self.target_error = (center_x - image_width * 0.5) / (image_width * 0.5)
            self.target_payload_at = rospy.get_time()
            self.last_target_payload = payload
        except (TypeError, ValueError, KeyError):
            return

    def publish_initial_pose_to_amcl(self):
        """发布 /initialpose 给 AMCL 做初始定位"""
        if not self.publish_initial_pose:
            return

        x = self.initial_pose.get("x", 0.0)
        y = self.initial_pose.get("y", 0.0)
        yaw = self.initial_pose.get("yaw", 0.0)

        pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
        rospy.loginfo("[vision_triggered_navigator] 等待 /initialpose 订阅者...")
        wait_start = rospy.Time.now()
        while pub.get_num_connections() == 0:
            if (rospy.Time.now() - wait_start).to_sec() > 5.0:
                rospy.logwarn("[vision_triggered_navigator] 5秒内没有 /initialpose 订阅者，继续执行...")
                break
            rospy.sleep(0.1)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = euler_to_quaternion(yaw)
        msg.pose.covariance = [
            0.1, 0,   0,   0,   0,   0,
            0,   0.1, 0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0.04
        ]

        pub.publish(msg)
        rospy.loginfo("[vision_triggered_navigator] 已发布初始位姿: x=%.4f, y=%.4f, yaw=%.4f", x, y, yaw)
        rospy.sleep(3)

    def _keyboard_thread(self):
        """键盘回车触发线程"""
        # 兼容 Python 2/3
        if sys.version_info[0] >= 3:
            input_func = input
        else:
            input_func = raw_input

        rospy.loginfo("[vision_triggered_navigator] 按回车键触发视觉导航...")
        while not rospy.is_shutdown() and not self.triggered:
            try:
                input_func()
                if not self.triggered:
                    rospy.loginfo("[vision_triggered_navigator] 键盘触发，打断当前导航.")
                    self.triggered = True
                    self.cancel_goal()
            except EOFError:
                rospy.sleep(0.5)

    def _build_rect_bounds(self):
        """从手标角点构建轴对齐包围盒与四堵墙"""
        xs = [p[0] for p in self.vision_rect_corners] # 去四个角点的 x 坐标
        ys = [p[1] for p in self.vision_rect_corners] # 去四个角点的 y 坐标
        self.rect_x_min = min(xs)
        self.rect_x_max = max(xs)
        self.rect_y_min = min(ys)
        self.rect_y_max = max(ys)

        # 四堵墙：(起点，终点，向内法向量)
        self.walls = [
            ((self.rect_x_min, self.rect_y_min), (self.rect_x_min, self.rect_y_max), (1.0, 0.0)),   # 左
            ((self.rect_x_max, self.rect_y_min), (self.rect_x_max, self.rect_y_max), (-1.0, 0.0)),  # 右
            ((self.rect_x_min, self.rect_y_min), (self.rect_x_max, self.rect_y_min), (0.0, 1.0)),   # 下
            ((self.rect_x_min, self.rect_y_max), (self.rect_x_max, self.rect_y_max), (0.0, -1.0)),  # 上
        ]

    def _get_robot_pose(self, frame_id):
        """查询 map -> frame_id 的位姿，返回 (x, y, yaw)；失败返回 None"""
        try:
            self.tf_listener.waitForTransform(self.map_frame, frame_id, rospy.Time(0), rospy.Duration(0.5))
            (trans, rot) = self.tf_listener.lookupTransform(self.map_frame, frame_id, rospy.Time(0))
            yaw = quaternion_to_yaw(rot)
            return trans[0], trans[1], yaw
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, "[vision_triggered_navigator] TF 查询 %s -> %s 失败: %s",
                                   self.map_frame, frame_id, str(e))
            return None

    def _get_cost_at(self, x, y):
        """查询 costmap 中 (x,y) 的代价值，未收到/越界返回 -1"""
        if self.costmap is None:
            return -1

        info = self.costmap.info
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)

        if mx < 0 or mx >= info.width or my < 0 or my >= info.height:
            return -1

        idx = my * info.width + mx
        raw = self.costmap.data[idx]
        cost = raw & 0xFF  # costmap 的 int8 需要转无符号才是 0~255
        if cost == 255:    # NO_INFORMATION
            return -1
        rospy.loginfo_throttle(5.0, "[vision_triggered_navigator] 查询 costmap (%.3f, %.3f) -> cost=%d", x, y, cost)
        return cost

    def _map_point_in_costmap_frame(self, x, y):
        """Transform a map-frame point into the latest costmap frame."""
        if self.costmap is None:
            return None
        frame = self.costmap.header.frame_id or self.map_frame
        if frame == self.map_frame:
            return float(x), float(y)
        point = PoseStamped()
        point.header.frame_id = self.map_frame
        point.header.stamp = rospy.Time(0)
        point.pose.position.x = float(x)
        point.pose.position.y = float(y)
        point.pose.orientation.w = 1.0
        try:
            self.tf_listener.waitForTransform(
                frame, self.map_frame, rospy.Time(0), rospy.Duration(0.3))
            transformed = self.tf_listener.transformPose(frame, point)
            return transformed.pose.position.x, transformed.pose.position.y
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                2.0, "[vision_triggered_navigator] 无法将map目标转换到costmap坐标系%s: %s",
                frame, str(exc))
            return None

    def _coverage_pose_cost(self, x, y):
        """Evaluate the whole robot footprint; unknown means 'let move_base decide'."""
        if self.costmap is None:
            return False, -1, False
        point = self._map_point_in_costmap_frame(x, y)
        if point is None:
            return False, -1, False
        info = self.costmap.info
        return footprint_max_cost(
            self.costmap.data,
            info.width,
            info.height,
            info.resolution,
            info.origin.position.x,
            info.origin.position.y,
            point[0],
            point[1],
            self.robot_footprint_radius,
            self.lethal_cost,
        )

    def is_goal_feasible(self, x, y):
        """判断目标点是否可行：cost 已知且小于阈值"""
        cost = self._get_cost_at(x, y)
        if cost < 0:
            rospy.logwarn_throttle(2.0,
                "[vision_triggered_navigator] 目标 (%.3f, %.3f) 代价未知，按可行处理", x, y)
            return True
        if cost >= self.cost_threshold:
            rospy.logwarn("[vision_triggered_navigator] 目标 (%.3f, %.3f) 代价 %d >= 阈值 %d，不可行",
                          x, y, cost, self.cost_threshold)
            return False
        return True

    def cancel_goal(self):
        """取消当前 move_base 目标"""
        if self.move_base_client.get_state() in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
            rospy.logwarn("[vision_triggered_navigator] 取消当前 move_base 目标.")
            self.move_base_client.cancel_goal()

    def _wait_navigation_idle(self, timeout=2.0):
        """Do not publish direct cmd_vel until move_base has relinquished control."""
        deadline = rospy.get_time() + max(0.0, float(timeout))
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.get_time() < deadline:
            state = self.move_base_client.get_state()
            if state not in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
                self.cmd_vel_pub.publish(Twist())
                return True
            self.move_base_client.cancel_goal()
            self.cmd_vel_pub.publish(Twist())
            rate.sleep()
        state = self.move_base_client.get_state()
        return state not in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]

    def _check_current_goal_cb(self, event):
        """定时器回调：检查当前导航目标是否仍然可行"""
        if self.current_goal_infeasible:
            return
        if self.current_goal_x is None or self.current_goal_y is None:
            return
        if not self.is_goal_feasible(self.current_goal_x, self.current_goal_y):
            rospy.logwarn("[vision_triggered_navigator] 当前导航目标中途变得不可行，取消并跳点.")
            self.current_goal_infeasible = True
            self.cancel_goal()

    # ------------------------------------------------------------------
    # 动作执行
    # ------------------------------------------------------------------
    def send_goal(self, x, y, yaw):
        """
        发送导航目标并等待结果。
        等待期间启动定时器实时检查目标可行性。
        返回 move_base 终态（SUCCEEDED / PREEMPTED / ABORTED / ...）
        """
        self.current_goal_x = x
        self.current_goal_y = y
        self.current_goal_infeasible = False
        self.current_goal_near_enough = False
        self.current_goal_timed_out = False

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation = euler_to_quaternion(yaw)

        rospy.loginfo("[vision_triggered_navigator] 发送导航目标: x=%.4f y=%.4f yaw=%.4f", x, y, yaw)
        self.move_base_client.send_goal(goal)

        timer = None
        if not self.coverage_search_mode:
            # 仅保留给原有独立导航模式；任务2覆盖模式不再按单栅格cost取消目标。
            period = rospy.Duration(1.0 / max(self.feasibility_check_rate, 0.1))
            timer = rospy.Timer(period, self._check_current_goal_cb)

        started = rospy.get_time()
        last_progress_at = started
        best_distance = None
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            state = self.move_base_client.get_state()
            if state not in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
                break
            if self.coverage_search_mode and not self.triggered:
                pose = self._get_robot_pose(self.base_frame)
                if pose is not None:
                    distance = math.hypot(float(x) - pose[0], float(y) - pose[1])
                    if best_distance is None or distance <= best_distance - self.coverage_min_progress:
                        best_distance = distance
                        last_progress_at = rospy.get_time()

                    known, max_cost, blocked = self._coverage_pose_cost(x, y)
                    if blocked and distance <= self.coverage_accept_radius:
                        rospy.logwarn(
                            "[vision_triggered_navigator] 精确观察点被障碍占据(cost=%d)，"
                            "当前位置距锚点%.2fm，改在安全当前位置观察.", max_cost, distance)
                        self.current_goal_near_enough = True
                        self.cancel_goal()
                        break

                    now = rospy.get_time()
                    if (now - last_progress_at >= self.coverage_stall_timeout and
                            now - started >= self.coverage_stall_timeout):
                        self.current_goal_near_enough = distance <= self.coverage_accept_radius
                        self.current_goal_timed_out = True
                        rospy.logwarn(
                            "[vision_triggered_navigator] 观察点导航连续%.1fs无进展，"
                            "distance=%.2f near_enough=%s，切换观察候选.",
                            self.coverage_stall_timeout, distance,
                            self.current_goal_near_enough)
                        self.cancel_goal()
                        break

                if rospy.get_time() - started >= self.coverage_goal_timeout:
                    if pose is not None:
                        distance = math.hypot(float(x) - pose[0], float(y) - pose[1])
                        self.current_goal_near_enough = distance <= self.coverage_accept_radius
                    self.current_goal_timed_out = True
                    rospy.logwarn(
                        "[vision_triggered_navigator] 观察候选超过%.1fs，near_enough=%s，取消.",
                        self.coverage_goal_timeout, self.current_goal_near_enough)
                    self.cancel_goal()
                    break
            rate.sleep()

        if timer is not None:
            timer.shutdown()
        if self.current_goal_near_enough or self.current_goal_timed_out:
            self.move_base_client.wait_for_result(rospy.Duration(1.0))
        final_state = self.move_base_client.get_state()

        if final_state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("[vision_triggered_navigator] 导航目标到达.")
        elif final_state == actionlib.GoalStatus.PREEMPTED:
            rospy.logwarn("[vision_triggered_navigator] 导航目标被抢占/取消.")
        elif final_state == actionlib.GoalStatus.ABORTED:
            rospy.logerr("[vision_triggered_navigator] 导航目标被终止.")
        else:
            rospy.logwarn("[vision_triggered_navigator] 导航结束，状态: %s", str(final_state))

        return final_state

    def _hold_scan_step(self, step_label, candidate_since):
        """Publish zero velocity while OCR consumes stable frames at one heading."""
        started = rospy.get_time()
        deadline = scan_dwell_deadline(
            started,
            self.coverage_scan_dwell,
            self.target_payload_at if self.target_payload_at >= candidate_since else 0.0,
            self.coverage_candidate_hold,
            self.coverage_scan_max_dwell,
        )
        extension_logged = False
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and not self.triggered:
            self.cmd_vel_pub.publish(Twist())
            candidate_at = self.target_payload_at
            new_deadline = scan_dwell_deadline(
                started,
                self.coverage_scan_dwell,
                candidate_at if candidate_at >= candidate_since else 0.0,
                self.coverage_candidate_hold,
                self.coverage_scan_max_dwell,
            )
            if new_deadline > deadline:
                deadline = new_deadline
            if (not extension_logged and candidate_at >= candidate_since and
                    deadline > started + self.coverage_scan_dwell + 1e-3):
                extension_logged = True
                rospy.loginfo(
                    "[vision_triggered_navigator] %s 收到目标候选，停车延长确认至%.2fs.",
                    step_label, deadline - started)
            if rospy.get_time() >= deadline:
                break
            rate.sleep()
        self.cmd_vel_pub.publish(Twist())

    def _step_scan(self, direction, duration):
        """Run a TF-closed-loop stop-and-look sweep for task2 coverage mode."""
        if duration <= 0:
            return True
        if self.coverage_scan_angular_speed <= 0.0:
            rospy.logerr("[vision_triggered_navigator] 步进扫描角速度必须大于0.")
            return False

        direction_sign = 1.0 if direction == "left" else -1.0
        total_angle = abs(self.rotation_speed * float(duration))
        steps = split_scan_angle(total_angle, self.coverage_scan_step_angle)
        rospy.loginfo(
            "[vision_triggered_navigator] 步进扫描 %s: total=%.1fdeg steps=%d "
            "speed=%.2frad/s dwell=%.2fs",
            direction, math.degrees(total_angle), len(steps),
            self.coverage_scan_angular_speed, self.coverage_scan_dwell)

        for step_index, step_angle in enumerate(steps, 1):
            if self.triggered:
                self.cmd_vel_pub.publish(Twist())
                return True
            start_pose = self._get_robot_pose(self.base_frame)
            if start_pose is None:
                self.cmd_vel_pub.publish(Twist())
                rospy.logerr(
                    "[vision_triggered_navigator] 步进扫描%d/%d起始TF不可用，延期当前锚点.",
                    step_index, len(steps))
                return False

            step_started = rospy.get_time()
            last_pose_at = step_started
            handled_candidate_at = self.target_payload_at
            deadline = (step_started + step_angle / self.coverage_scan_angular_speed +
                        self.coverage_scan_step_timeout_margin)
            max_progress = 0.0
            twist = Twist()
            twist.angular.z = self.coverage_scan_angular_speed * direction_sign
            rate = rospy.Rate(20)

            while not rospy.is_shutdown() and not self.triggered:
                now = rospy.get_time()
                if self.target_payload_at > handled_candidate_at:
                    handled_candidate_at = self.target_payload_at
                    self.cmd_vel_pub.publish(Twist())
                    rospy.loginfo(
                        "[vision_triggered_navigator] 步进%d/%d转动中捕获目标候选，立即停车确认.",
                        step_index, len(steps))
                    pause_started = rospy.get_time()
                    self._hold_scan_step(
                        "步进{}/{}".format(step_index, len(steps)), step_started)
                    deadline += max(0.0, rospy.get_time() - pause_started)
                    if self.triggered:
                        return True
                    handled_candidate_at = self.target_payload_at

                pose = self._get_robot_pose(self.base_frame)
                if pose is not None:
                    last_pose_at = now
                    progress = normalize_angle(pose[2] - start_pose[2]) * direction_sign
                    max_progress = max(max_progress, progress)
                    if max_progress >= step_angle:
                        break
                elif now - last_pose_at >= self.coverage_scan_pose_timeout:
                    self.cmd_vel_pub.publish(Twist())
                    rospy.logerr(
                        "[vision_triggered_navigator] 步进扫描TF超过%.2fs未更新，"
                        "立即停车并延期当前锚点.", self.coverage_scan_pose_timeout)
                    return False

                if now >= deadline:
                    self.cmd_vel_pub.publish(Twist())
                    rospy.logerr(
                        "[vision_triggered_navigator] 步进扫描%d/%d超时，"
                        "progress=%.1f/%.1fdeg，延期当前锚点.",
                        step_index, len(steps), math.degrees(max_progress),
                        math.degrees(step_angle))
                    return False
                self.cmd_vel_pub.publish(twist)
                rate.sleep()

            self.cmd_vel_pub.publish(Twist())
            if self.triggered:
                return True
            pose = self._get_robot_pose(self.base_frame)
            heading = math.degrees(pose[2]) if pose is not None else float("nan")
            rospy.loginfo(
                "[vision_triggered_navigator] 步进%d/%d到位 heading=%.1fdeg，停车识别.",
                step_index, len(steps), heading)
            self._hold_scan_step(
                "步进{}/{}".format(step_index, len(steps)), step_started)
        return True

    def rotate(self, direction, duration):
        """
        发布角速度使机器人自转。
        direction: "left" 左转，"right" 右转
        duration: 保持时间（秒）
        """
        if duration <= 0:
            return True

        if self.coverage_search_mode:
            return self._step_scan(direction, duration)

        twist = Twist()
        direction_sign = 1.0 if direction == "left" else -1.0
        twist.angular.z = self.rotation_speed * direction_sign

        rospy.loginfo("[vision_triggered_navigator] 自转 %s, 速度 %.2f rad/s, 保持 %.2f s",
                      direction, twist.angular.z, duration)

        start = rospy.Time.now()
        time_limit = duration
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start).to_sec()
            # 轮询时检查是否被视觉/键盘触发打断，打断立即停止
            if elapsed >= time_limit or self.triggered:
                break
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # 发送零速停止
        self.cmd_vel_pub.publish(Twist())
        return True

    def perform_rotations(self, rotations):
        """顺序执行一组自转动作"""
        for rot in rotations:
            if self.triggered:
                break
            direction = rot.get("direction", "left")
            duration = rot.get("duration", 0.0)
            if not self.rotate(direction, duration):
                return False
        return True

    def _observation_candidates(self, point):
        bounds = (self.rect_x_min, self.rect_x_max,
                  self.rect_y_min, self.rect_y_max)
        positions = build_observation_candidates(
            point["x"], point["y"], self.coverage_candidate_offsets,
            bounds, self.coverage_min_wall_clearance)
        return [(x, y, point["yaw"]) for x, y in positions]

    def _visit_coverage_point(self, point, patrol_idx, revisit=False):
        """Observe one logical wall segment without silently losing coverage."""
        candidates = self._observation_candidates(point)
        label = "重访" if revisit else "首访"
        for candidate_idx, (x, y, yaw) in enumerate(candidates):
            if self.triggered:
                return "triggered"
            known, max_cost, blocked = self._coverage_pose_cost(x, y)
            if known and blocked:
                rospy.logwarn(
                    "[vision_triggered_navigator] 锚点%d %s候选%d footprint被占(cost=%d)，换候选.",
                    patrol_idx + 1, label, candidate_idx + 1, max_cost)
                continue
            rospy.loginfo(
                "[vision_triggered_navigator] 锚点%d %s候选%d/%d: (%.3f, %.3f, %.3f)",
                patrol_idx + 1, label, candidate_idx + 1, len(candidates), x, y, yaw)
            result = self.send_goal(x, y, yaw)
            if self.triggered:
                return "triggered"
            if result == actionlib.GoalStatus.SUCCEEDED or self.current_goal_near_enough:
                if not self._wait_navigation_idle():
                    rospy.logerr(
                        "[vision_triggered_navigator] move_base未在期限内释放控制权，"
                        "禁止执行观察自转，改试下一候选.")
                    continue
                self.cmd_vel_pub.publish(Twist())
                initial_hold_at = rospy.get_time()
                self._hold_scan_step(
                    "锚点{}初始朝向".format(patrol_idx + 1),
                    initial_hold_at - max(self.target_bbox_stale,
                                          self.coverage_scan_dwell))
                if self.triggered:
                    return "triggered"
                if not self.perform_rotations(point.get("rotations", [])):
                    rospy.logwarn(
                        "[vision_triggered_navigator] 锚点%d步进扫描未完成，延期重访.",
                        patrol_idx + 1)
                    return "deferred"
                if self.triggered:
                    return "triggered"
                rospy.loginfo(
                    "[vision_triggered_navigator] coverage anchor=%d state=covered candidate=%d revisit=%s",
                    patrol_idx + 1, candidate_idx + 1, revisit)
                return "covered"
        rospy.logwarn(
            "[vision_triggered_navigator] coverage anchor=%d state=deferred revisit=%s",
            patrol_idx + 1, revisit)
        return "deferred"

    def _odom_is_fresh(self):
        return (self.odom_yaw is not None and
                rospy.get_time() - self.odom_received_at <= self.odom_stale)

    def _rotate_center_step(self, direction, target_angle):
        """Rotate one small odometry-closed-loop step, ramping through deadband."""
        if not self._odom_is_fresh():
            rospy.logerr("[vision_triggered_navigator] /odom超过%.2fs未更新，拒绝居中转动.",
                         self.odom_stale)
            return False

        direction = 1.0 if direction >= 0.0 else -1.0
        start_yaw = self.odom_yaw
        speed = self.target_center_start_speed
        window_started = rospy.get_time()
        window_yaw = start_yaw
        ramp_steps = int(math.ceil(
            max(0.0, self.target_center_step_max_speed - self.target_center_start_speed) /
            max(self.target_center_speed_increment, 1e-3)))
        deadline = rospy.get_time() + max(
            2.0,
            target_angle / max(self.target_center_start_speed, 0.01) +
            (ramp_steps + 2) * self.target_center_motion_window + 0.5)
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and rospy.get_time() < deadline:
            if not self._odom_is_fresh():
                self.cmd_vel_pub.publish(Twist())
                rospy.logerr("[vision_triggered_navigator] 居中步进期间/odom失效，立即停车.")
                return False

            progress = abs(normalize_angle(self.odom_yaw - start_yaw))
            if progress >= target_angle:
                self.cmd_vel_pub.publish(Twist())
                rospy.loginfo(
                    "[vision_triggered_navigator] 居中步进完成 angle=%.2fdeg speed=%.2f",
                    math.degrees(progress), speed)
                return True

            now = rospy.get_time()
            if now - window_started >= self.target_center_motion_window:
                window_progress = abs(normalize_angle(self.odom_yaw - window_yaw))
                if window_progress < self.target_center_min_progress:
                    if speed + 1e-6 < self.target_center_step_max_speed:
                        speed = min(
                            self.target_center_step_max_speed,
                            speed + self.target_center_speed_increment)
                        rospy.logwarn(
                            "[vision_triggered_navigator] 角速度未越过底盘死区，提升至%.2frad/s.",
                            speed)
                    else:
                        self.cmd_vel_pub.publish(Twist())
                        rospy.logerr(
                            "[vision_triggered_navigator] 已到%.2frad/s仍无里程计转角，居中失败.",
                            speed)
                        return False
                window_started = now
                window_yaw = self.odom_yaw

            twist = Twist()
            twist.angular.z = direction * speed
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        self.cmd_vel_pub.publish(Twist())
        rospy.logerr("[vision_triggered_navigator] 居中单步转动超时.")
        return False

    def _wait_fresh_target(self, previous_stamp, deadline):
        """Wait stopped for an OCR box newer than the one used for the last step."""
        rate = rospy.Rate(30)
        while not rospy.is_shutdown() and rospy.get_time() < deadline:
            self.cmd_vel_pub.publish(Twist())
            if self.target_payload_at > previous_stamp:
                return True
            rate.sleep()
        return False

    def _centering_failure(self, message):
        self.cmd_vel_pub.publish(Twist())
        self._publish_status("centering_failed")
        rospy.logerr("[vision_triggered_navigator] %s", message)
        return False

    def _center_visual_target(self):
        """Center the OCR box with stop-look odometry-closed-loop angular steps."""
        if not self.coverage_search_mode:
            return True
        self._publish_status("target_centering")
        if not self._wait_navigation_idle():
            return self._centering_failure("move_base未释放控制权，拒绝视觉居中.")
        if not self._odom_is_fresh():
            return self._centering_failure("视觉居中开始前/odom不可用.")

        deadline = rospy.get_time() + self.target_center_timeout
        centered_hits = 0
        last_centered_stamp = 0.0
        steering_sign = self.target_center_steering_sign
        reversed_once = False
        must_improve_after_reverse = False

        while not rospy.is_shutdown() and rospy.get_time() < deadline:
            age = rospy.get_time() - self.target_payload_at
            if self.target_error is None or age > self.target_bbox_stale:
                return self._centering_failure("目标框丢失或超过时效，停止而不恢复巡航.")
            if not self._odom_is_fresh():
                return self._centering_failure("视觉居中期间/odom超过时效.")

            if abs(self.target_error) <= self.target_center_tolerance:
                self.cmd_vel_pub.publish(Twist())
                if self.target_payload_at > last_centered_stamp:
                    last_centered_stamp = self.target_payload_at
                    centered_hits += 1
                    rospy.loginfo(
                        "[vision_triggered_navigator] target centered error=%.3f hits=%d/%d",
                        self.target_error, centered_hits, self.target_center_required_hits)
                if centered_hits >= self.target_center_required_hits:
                    self._hold_stopped(self.target_center_settle)
                    return True
                if not self._wait_fresh_target(last_centered_stamp, min(
                        deadline, rospy.get_time() + self.target_bbox_stale)):
                    return self._centering_failure("居中后未收到第二帧新目标框.")
                continue

            centered_hits = 0
            before_error = float(self.target_error)
            before_stamp = self.target_payload_at
            step_angle = center_step_angle(
                before_error,
                self.target_center_tolerance,
                self.target_center_fine_threshold,
                self.target_center_coarse_step,
                self.target_center_fine_step,
            )
            direction = (1.0 if steering_sign >= 0.0 else -1.0) * math.copysign(
                1.0, before_error)
            rospy.loginfo(
                "[vision_triggered_navigator] 居中步进 error=%.3f step=%.1fdeg direction=%+.0f",
                before_error, math.degrees(step_angle), direction)
            if not self._rotate_center_step(direction, step_angle):
                return self._centering_failure("底盘未完成视觉居中步进.")
            self._hold_stopped(self.target_center_settle)
            settled_at = rospy.get_time()
            if not self._wait_fresh_target(
                    max(before_stamp, settled_at),
                    min(deadline, rospy.get_time() + self.target_bbox_stale)):
                return self._centering_failure("步进后未收到新的目标框.")

            after_error = float(self.target_error)
            improvement = abs(before_error) - abs(after_error)
            rospy.loginfo(
                "[vision_triggered_navigator] 居中反馈 before=%.3f after=%.3f improvement=%.3f",
                before_error, after_error, improvement)
            if must_improve_after_reverse:
                if improvement <= 0.0:
                    return self._centering_failure("自动反向后误差仍未减小，停止居中.")
                must_improve_after_reverse = False
            elif improvement < -self.target_center_reverse_threshold:
                if reversed_once:
                    return self._centering_failure("目标误差再次增大，停止居中.")
                steering_sign *= -1.0
                reversed_once = True
                must_improve_after_reverse = True
                rospy.logwarn(
                    "[vision_triggered_navigator] 首次步进使误差增大，自动反转居中方向为%+.0f.",
                    steering_sign)

        return self._centering_failure("目标居中超过%.1fs，车辆保持停车." %
                                       self.target_center_timeout)

    def _hold_stopped(self, duration):
        deadline = rospy.get_time() + max(0.0, float(duration))
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.get_time() < deadline:
            self.cmd_vel_pub.publish(Twist())
            rate.sleep()

    def _tighten_final_tolerances(self):
        """Temporarily tighten TEB only for the 50cm task2 parking goal."""
        if not self.validate_parking_box:
            return True
        try:
            if self._planner_client is None:
                self._planner_client = dynamic_reconfigure.client.Client(
                    self.local_planner_reconfigure_ns, timeout=3.0)
            current = self._planner_client.get_configuration(timeout=3.0)
            self._saved_planner_tolerances = {
                "xy_goal_tolerance": current.get("xy_goal_tolerance", 0.15),
                "yaw_goal_tolerance": current.get("yaw_goal_tolerance", 0.1),
                "free_goal_vel": current.get("free_goal_vel", False),
            }
            updated = self._planner_client.update_configuration({
                "xy_goal_tolerance": self.parking_xy_tolerance,
                "yaw_goal_tolerance": self.parking_yaw_tolerance,
                "free_goal_vel": False,
            })
            rospy.loginfo(
                "[vision_triggered_navigator] 最终停泊临时收紧TEB容差 xy=%.3f yaw=%.3f",
                float(updated.get("xy_goal_tolerance", self.parking_xy_tolerance)),
                float(updated.get("yaw_goal_tolerance", self.parking_yaw_tolerance)))
            return True
        except Exception as exc:
            self._saved_planner_tolerances = None
            rospy.logerr("[vision_triggered_navigator] 无法收紧最终停泊TEB容差: %s", str(exc))
            return False

    def _restore_final_tolerances(self):
        """Restore navigation-team TEB values after task2 parking."""
        saved = self._saved_planner_tolerances
        self._saved_planner_tolerances = None
        if not saved or self._planner_client is None:
            return
        try:
            self._planner_client.update_configuration(saved)
            rospy.loginfo(
                "[vision_triggered_navigator] 已恢复TEB容差 xy=%.3f yaw=%.3f",
                float(saved["xy_goal_tolerance"]),
                float(saved["yaw_goal_tolerance"]))
        except Exception as exc:
            rospy.logerr("[vision_triggered_navigator] 恢复TEB容差失败: %s", str(exc))

    def _validate_parking_pose(self):
        """Require the full configured footprint to be inside the 50cm box."""
        if not self.validate_parking_box:
            return True
        if self.parking_wall_point is None or self.parking_inward_normal is None:
            rospy.logerr("[vision_triggered_navigator] 缺少停泊框几何，无法验证.")
            return False
        pose = self._get_robot_pose(self.base_frame)
        if pose is None:
            rospy.logerr("[vision_triggered_navigator] 无法获取最终位姿，停泊验证失败.")
            return False
        valid = parking_footprint_inside(
            pose,
            self.parking_wall_point,
            self.parking_inward_normal,
            self.parking_box_width,
            self.parking_box_depth,
            self.footprint_half_length,
            self.footprint_half_width,
            self.parking_validation_margin,
        )
        rospy.loginfo(
            "[vision_triggered_navigator] 停泊框验证 pose=(%.4f, %.4f, %.4f) "
            "box=%.2fx%.2f full_footprint_inside=%s",
            pose[0], pose[1], pose[2], self.parking_box_width,
            self.parking_box_depth, valid)
        return valid

    # ------------------------------------------------------------------
    # 视觉触发目标计算
    # ------------------------------------------------------------------
    def compute_vision_goal(self):
        """
        根据 base_link 车头正方向射线与长方形围墙求交，
        交点沿内法向回退 vision_offset，返回 (x, y, yaw)。
        """
        pose = self._get_robot_pose(self.base_frame)
        if pose is None:
            rospy.logerr("[vision_triggered_navigator] 无法获取机器人位置，无法计算视觉目标.")
            return None
        px, py, yaw = pose
        yaw = normalize_angle(yaw + self.camera_boresight_yaw_offset)
        dx = math.cos(yaw)
        dy = math.sin(yaw)

        rospy.loginfo("[vision_triggered_navigator] 射线起点 (%.4f, %.4f), 方向 yaw=%.4f",
                      px, py, yaw)

        # 射线与 4 堵墙求最近正交点
        best_t = float('inf')
        best_normal = None
        best_point = None

        for a, b, normal in self.walls:
            t = self._ray_segment_intersection((px, py), (dx, dy), a, b)
            if t is not None and t > 1e-6 and t < best_t:
                best_t = t
                best_normal = normal
                best_point = (px + t * dx, py + t * dy)

        if best_point is None:
            rospy.logerr("[vision_triggered_navigator] 射线与围墙无交点，无法计算视觉目标.")
            return None

        ix, iy = best_point
        nx, ny = best_normal
        offset = self.parking_goal_offset
        gx = ix + nx * offset
        gy = iy + ny * offset
        # 车头垂直指向墙外（与内法向相反）
        gyaw = math.atan2(-ny, -nx)

        rospy.loginfo("[vision_triggered_navigator] 墙交点 (%.4f, %.4f), 外法向 (%.2f, %.2f), "
                      "目标点 (%.4f, %.4f, yaw=%.4f)",
                      ix, iy, -nx, -ny, gx, gy, gyaw)
        self.parking_wall_point = (ix, iy)
        self.parking_inward_normal = (nx, ny)
        return gx, gy, gyaw

    @staticmethod
    def _ray_segment_intersection(origin, direction, a, b):
        """
        2D 射线与线段相交，返回射线参数 t；不相交返回 None。
        origin: (x, y)
        direction: (dx, dy)，无需单位化
        a, b: 线段端点
        """
        ox, oy = origin
        dx, dy = direction
        ax, ay = a
        bx, by = b

        vx = bx - ax
        vy = by - ay

        denom = vx * (-dy) - vy * (-dx)  # = vx*(-dy) + vy*dx
        # 等价于 cross(v, d_perp)
        denom = -vx * dy + vy * dx
        if abs(denom) < 1e-9:
            return None

        wx = ox - ax
        wy = oy - ay

        # t = cross(v, w) / denom
        t = (vx * wy - vy * wx) / denom
        # u = cross(w, d_perp) / denom, d_perp = (-dy, dx)
        u = (wx * (-dy) - wy * (-dx)) / denom
        u = (-wx * dy + wy * dx) / denom

        if u < -1e-6 or u > 1.0 + 1e-6:
            return None
        return t

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self):
        rospy.loginfo("[vision_triggered_navigator] 节点启动，开始三阶段导航.")
        self._publish_status("patrolling")

        # 步骤 0：给 AMCL 发送初始位姿
        self.publish_initial_pose_to_amcl()

        state = "PATROL"
        patrol_idx = 0
        coverage_count = len(self.patrol_points)
        if self.max_coverage_anchors > 0:
            coverage_count = min(coverage_count, self.max_coverage_anchors)
        coverage_order = list(range(coverage_count))
        coverage_position = 0
        coverage_deferred = []
        coverage_revisit_pass = 0

        while not rospy.is_shutdown():
            # 一旦被触发，立即切换到视觉阶段
            if self.triggered and state == "PATROL":
                rospy.loginfo("[vision_triggered_navigator] 巡航被打断，进入视觉触发阶段.")
                state = "VISION"
                continue

            if state == "PATROL":
                if self.coverage_search_mode:
                    if coverage_position >= len(coverage_order):
                        if (coverage_deferred and
                                coverage_revisit_pass < self.coverage_revisit_passes):
                            coverage_revisit_pass += 1
                            coverage_order = list(coverage_deferred)
                            coverage_deferred = []
                            coverage_position = 0
                            self._publish_status("revisiting")
                            rospy.logwarn(
                                "[vision_triggered_navigator] 开始覆盖重访%d/%d，锚点=%s",
                                coverage_revisit_pass, self.coverage_revisit_passes,
                                [idx + 1 for idx in coverage_order])
                            continue
                        rospy.logerr(
                            "[vision_triggered_navigator] 所有墙段扫描完成但未锁定目标；"
                            "未覆盖锚点=%s", [idx + 1 for idx in coverage_deferred])
                        self._publish_status("failed")
                        break

                    point_idx = coverage_order[coverage_position]
                    point = self.patrol_points[point_idx]
                    rospy.loginfo(
                        "[vision_triggered_navigator] === 覆盖锚点 %d / %d，逻辑编号%d ===",
                        coverage_position + 1, len(coverage_order), point_idx + 1)
                    outcome = self._visit_coverage_point(
                        point, point_idx, coverage_revisit_pass > 0)
                    if outcome == "triggered":
                        state = "VISION"
                        continue
                    if outcome == "deferred":
                        coverage_deferred.append(point_idx)
                    coverage_position += 1
                    continue

                if patrol_idx >= len(self.patrol_points):
                    if not self.navigate_to_end_after_trigger:
                        rospy.logerr("[vision_triggered_navigator] 巡航点全部完成但未识别到目标厂牌.")
                        self._publish_status("failed")
                        break
                    rospy.loginfo("[vision_triggered_navigator] 巡航点全部完成，进入结束点阶段.")
                    state = "END"
                    continue

                point = self.patrol_points[patrol_idx]
                x = point["x"]
                y = point["y"]
                yaw = point["yaw"]
                rotations = point.get("rotations", [])

                rospy.loginfo("[vision_triggered_navigator] === 巡航点 %d / %d ===",
                              patrol_idx + 1, len(self.patrol_points))

                # 发送前检查可行性
                if not self.is_goal_feasible(x, y):
                    rospy.logwarn("[vision_triggered_navigator] 巡航点 %d 初始不可行，跳过.", patrol_idx + 1)
                    patrol_idx += 1
                    continue

                result = self.send_goal(x, y, yaw)

                # 若触发被打断
                if self.triggered:
                    rospy.loginfo("[vision_triggered_navigator] 巡航点 %d 导航中被触发，进入视觉阶段.",
                                  patrol_idx + 1)
                    state = "VISION"
                    continue

                # 若因中途代价变高被取消
                if self.current_goal_infeasible:
                    rospy.logwarn("[vision_triggered_navigator] 巡航点 %d 中途不可行，跳到下一目标.",
                                  patrol_idx + 1)
                    patrol_idx += 1
                    continue

                # 成功到达则执行自转
                if result == actionlib.GoalStatus.SUCCEEDED:
                    self.perform_rotations(rotations)
                    patrol_idx += 1
                else:
                    rospy.logwarn("[vision_triggered_navigator] 巡航点 %d 导航未成功，跳过.", patrol_idx + 1)
                    patrol_idx += 1

            elif state == "VISION":
                rospy.loginfo("[vision_triggered_navigator] === 视觉触发阶段 ===")
                self.cancel_goal()
                self._hold_stopped(self.coverage_scan_settle)
                self._publish_status("target_locked")
                if not self._center_visual_target():
                    rospy.logerr("[vision_triggered_navigator] 目标锁定后居中失败，车辆保持停车.")
                    break
                if self.center_only:
                    self._hold_stopped(self.arrival_hold_sec)
                    self._publish_status("centered")
                    rospy.logwarn(
                        "[vision_triggered_navigator] center_only=true：仅完成居中，不执行50cm框停泊.")
                    break
                self._publish_status("parking_approaching")
                goal = self.compute_vision_goal()
                if goal is not None:
                    gx, gy, gyaw = goal
                else:
                    rospy.logerr("[vision_triggered_navigator] 视觉目标计算失败.")
                    self._publish_status("failed")
                    break
                if not self._tighten_final_tolerances():
                    self._publish_status("parking_validation_failed")
                    self._hold_stopped(self.arrival_hold_sec)
                    break
                parking_valid = False
                try:
                    result = self.send_goal(gx, gy, gyaw)
                    if result != actionlib.GoalStatus.SUCCEEDED:
                        self._publish_status("failed")
                        break
                    self._hold_stopped(self.arrival_hold_sec)
                    self._publish_status("parking_verifying")
                    parking_valid = self._validate_parking_pose()
                finally:
                    self._restore_final_tolerances()
                if not parking_valid:
                    self._publish_status("parking_validation_failed")
                    self._hold_stopped(self.arrival_hold_sec)
                    rospy.logerr(
                        "[vision_triggered_navigator] move_base已结束，但完整footprint不在50cm框内.")
                    break
                if self.navigate_to_end_after_trigger:
                    state = "END"
                else:
                    self._hold_stopped(self.arrival_hold_sec)
                    self._publish_status("arrived")
                    rospy.loginfo("[vision_triggered_navigator] 已抵达厂牌，按配置不前往结束点.")
                    break

            elif state == "END":
                rospy.loginfo("[vision_triggered_navigator] === 结束点阶段 ===")
                x = self.end_goal["x"]
                y = self.end_goal["y"]
                yaw = self.end_goal["yaw"]
                result = self.send_goal(x, y, yaw)
                if result == actionlib.GoalStatus.SUCCEEDED:
                    self._publish_status("completed")
                    rospy.loginfo("[vision_triggered_navigator] 全部流程结束.")
                else:
                    self._publish_status("failed")
                break

            else:
                break

        self.cmd_vel_pub.publish(Twist())


def main():
    node = VisionTriggeredNavigator()
    node.run()


if __name__ == "__main__":
    main()
