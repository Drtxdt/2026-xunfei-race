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
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger, TriggerResponse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from navigator_logic import (
    build_quadrilateral_walls,
    docking_command,
    docking_pose_errors,
    docking_within_tolerance,
    lidar_base_wall_distance,
    lidar_requires_stop,
    center_step_angle,
    costmap_value_at,
    exact_observation_target,
    footprint_max_cost,
    latch_trigger,
    normalize_angle,
    parking_footprint_margins,
    parking_goal_from_wall,
    ray_segment_intersection,
    scan_dwell_deadline,
    sensor_is_fresh,
    should_skip_coverage_anchor,
    split_scan_angle,
    staging_motion_is_rotation_stall,
    wall_normal_distance,
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
        self.trigger_service_name = rospy.get_param(
            "~trigger_service", "/vision_triggered_navigator/trigger_target")
        self.navigate_to_end_after_trigger = rospy.get_param(
            "~navigate_to_end_after_trigger", True)

        # 任务2专用的覆盖优先模式。默认关闭，保证原独立导航行为不变。
        self.coverage_search_mode = rospy.get_param("~coverage_search_mode", False)
        self.target_topic = rospy.get_param("~target_topic", "/vision/target")
        self.coverage_goal_timeout = rospy.get_param("~coverage_goal_timeout_sec", 25.0)
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
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
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
        self.parking_staging_offset = abs(float(rospy.get_param(
            "~parking_staging_offset", 0.55)))
        self.parking_staging_timeout = max(1.0, float(rospy.get_param(
            "~parking_staging_timeout_sec", 20.0)))
        self.parking_staging_acceptance = max(0.01, float(rospy.get_param(
            "~parking_staging_acceptance", 0.12)))
        self.parking_staging_watchdog_window = max(0.5, float(rospy.get_param(
            "~parking_staging_watchdog_window_sec", 2.0)))
        self.parking_staging_min_progress = max(0.0, float(rospy.get_param(
            "~parking_staging_min_progress", 0.03)))
        self.parking_staging_max_rotation = math.radians(abs(float(rospy.get_param(
            "~parking_staging_max_rotation_deg", 45.0))))
        self.parking_docking_timeout = max(1.0, float(rospy.get_param(
            "~parking_docking_timeout_sec", 15.0)))
        self.parking_dock_max_x = abs(float(rospy.get_param(
            "~parking_dock_max_x", 0.10)))
        self.parking_dock_max_y = abs(float(rospy.get_param(
            "~parking_dock_max_y", 0.06)))
        self.parking_dock_max_yaw = abs(float(rospy.get_param(
            "~parking_dock_max_yaw", 0.15)))
        self.parking_dock_normal_tolerance = abs(float(rospy.get_param(
            "~parking_dock_normal_tolerance", 0.015)))
        self.parking_dock_tangent_tolerance = abs(float(rospy.get_param(
            "~parking_dock_tangent_tolerance", 0.02)))
        self.parking_dock_yaw_tolerance = abs(float(rospy.get_param(
            "~parking_dock_yaw_tolerance", 0.035)))
        self.parking_dock_stable_sec = max(0.1, float(rospy.get_param(
            "~parking_dock_stable_sec", 0.5)))
        self.parking_min_wall_distance = abs(float(rospy.get_param(
            "~parking_min_wall_distance", 0.19)))
        self.parking_lidar_stop_distance = abs(float(rospy.get_param(
            "~parking_lidar_stop_distance", 0.15)))
        self.parking_lidar_forward_offset = float(rospy.get_param(
            "~parking_lidar_forward_offset", 0.08))
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.scan_stale = max(0.1, float(rospy.get_param(
            "~scan_stale_sec", 0.5)))
        self.scan_front_half_angle = math.radians(abs(float(rospy.get_param(
            "~scan_front_half_angle_deg", 15.0))))
        self.parking_normal_offset = float(rospy.get_param(
            "~parking_normal_offset", 0.0))
        self.parking_tangent_offset = float(rospy.get_param(
            "~parking_tangent_offset", 0.0))
        self.parking_xy_tolerance = abs(float(rospy.get_param(
            "~parking_xy_tolerance", 0.04)))
        self.parking_yaw_tolerance = abs(float(rospy.get_param(
            "~parking_yaw_tolerance", 0.06)))
        self.parking_validation_margin = max(0.0, float(rospy.get_param(
            "~parking_validation_margin", 0.01)))
        self.parking_required_margin = max(0.0, float(rospy.get_param(
            "~parking_required_margin", 0.02)))
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
        self.odom_pose = None
        self.odom_frame_from_msg = self.odom_frame
        self.odom_received_at = 0.0
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=10)
        self.scan_front_min = None
        self.scan_received_at = 0.0
        rospy.Subscriber(self.scan_topic, LaserScan, self._scan_cb, queue_size=1)

        self.target_error = None
        self.target_payload_at = 0.0
        self.last_target_payload = None
        if self.coverage_search_mode:
            rospy.Subscriber(self.target_topic, String, self._target_cb, queue_size=10)

        self.triggered = False
        self.trigger_lock = threading.Lock()

        # 连接 move_base action server
        self.move_base_client = actionlib.SimpleActionClient(self.move_base_server, MoveBaseAction)
        rospy.loginfo("[vision_triggered_navigator] 等待 move_base 服务器...")
        self.move_base_client.wait_for_server()
        rospy.loginfo("[vision_triggered_navigator] move_base 服务器已连接.")

        # 当前目标记录（用于定时器检查）
        self.current_goal_x = None
        self.current_goal_y = None
        self.current_goal_infeasible = False
        self.current_goal_timed_out = False
        self.parking_wall_point = None
        self.parking_inward_normal = None
        self.parking_wall_name = None
        self._planner_client = None
        self._saved_planner_tolerances = None
        rospy.on_shutdown(self._restore_final_tolerances)
        self._publish_status("ready")
        # 只有在 action client 和全部状态完成初始化后才接收触发，避免启动竞态。
        self.trigger_service = rospy.Service(
            self.trigger_service_name, Trigger, self._trigger_service_cb)
        rospy.loginfo("[vision_triggered_navigator] 可靠触发服务已就绪: %s",
                      self.trigger_service_name)
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

    # ------------------------------------------------------------------
    # 回调与工具函数
    # ------------------------------------------------------------------
    def _costmap_cb(self, msg):
        """保存最新 costmap"""
        self.costmap = msg

    def _odom_cb(self, msg):
        """Keep a fresh odom pose for centering and final docking."""
        self.odom_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.odom_pose = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            self.odom_yaw,
        )
        if msg.header.frame_id:
            self.odom_frame_from_msg = msg.header.frame_id
        self.odom_received_at = rospy.get_time()

    def _scan_cb(self, msg):
        """Store the nearest valid range in the forward scan sector."""
        nearest = None
        angle = float(msg.angle_min)
        for value in msg.ranges:
            base_angle = normalize_angle(angle)
            if abs(base_angle) <= self.scan_front_half_angle:
                distance = float(value)
                if (math.isfinite(distance) and
                        distance >= float(msg.range_min) and
                        distance <= float(msg.range_max)):
                    nearest = distance if nearest is None else min(nearest, distance)
            angle += float(msg.angle_increment)
        self.scan_front_min = nearest
        self.scan_received_at = rospy.get_time()

    def _publish_status(self, status):
        """发布简洁、稳定的流程状态，供比赛总控监听。"""
        self.status_pub.publish(String(data=status))

    def _accept_trigger(self, source):
        """Idempotently latch a target trigger and cancel active navigation."""
        with self.trigger_lock:
            self.triggered, accepted = latch_trigger(self.triggered)
            if not accepted:
                rospy.loginfo("[vision_triggered_navigator] %s触发重复到达，保持已锁存状态.",
                              source)
                return False
        rospy.loginfo("[vision_triggered_navigator] 收到%s触发，打断当前导航.", source)
        self._publish_status("triggered")
        self.cancel_goal()
        return True

    def _vision_cb(self, msg):
        """视觉触发回调"""
        if msg.data:
            self._accept_trigger("视觉话题")

    def _trigger_service_cb(self, _request):
        """Reliably acknowledge competition target lock requests."""
        accepted = self._accept_trigger("目标服务")
        if accepted:
            return TriggerResponse(True, "target trigger accepted and latched")
        return TriggerResponse(True, "target trigger was already latched")

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
                    self._accept_trigger("键盘")
            except EOFError:
                rospy.sleep(0.5)

    def _build_rect_bounds(self):
        """Build a safety AABB plus the four measured, possibly skewed walls."""
        xs = [p[0] for p in self.vision_rect_corners] # 去四个角点的 x 坐标
        ys = [p[1] for p in self.vision_rect_corners] # 去四个角点的 y 坐标
        self.rect_x_min = min(xs)
        self.rect_x_max = max(xs)
        self.rect_y_min = min(ys)
        self.rect_y_max = max(ys)

        # 最终停泊使用实测四边形墙段；AABB不参与观察点或停车目标生成。
        self.walls = build_quadrilateral_walls(self.vision_rect_corners)

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
        """查询map目标在costmap中的代价；未知、TF失败或越界返回-1。"""
        if self.costmap is None:
            return -1

        point = self._map_point_in_costmap_frame(x, y)
        if point is None:
            return -1
        cost_x, cost_y = point
        info = self.costmap.info
        cost = costmap_value_at(
            self.costmap.data, info.width, info.height, info.resolution,
            info.origin.position.x, info.origin.position.y, cost_x, cost_y)
        if cost < 0:
            return -1
        rospy.loginfo_throttle(
            5.0,
            "[vision_triggered_navigator] 查询costmap map=(%.3f, %.3f) %s=(%.3f, %.3f) -> cost=%d",
            x, y, self.costmap.header.frame_id or self.map_frame,
            cost_x, cost_y, cost)
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
                2.0,
                "[vision_triggered_navigator] map目标(%.4f, %.4f)无法转换到costmap坐标系%s: %s；按未知代价处理",
                x, y, frame, str(exc))
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
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            state = self.move_base_client.get_state()
            if state not in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
                break
            if self.coverage_search_mode and not self.triggered:
                if rospy.get_time() - started >= self.coverage_goal_timeout:
                    self.current_goal_timed_out = True
                    rospy.logwarn(
                        "[vision_triggered_navigator] 精确观察点(%.4f, %.4f, %.4f)导航超过%.1fs，取消并进入下一原始锚点.",
                        x, y, yaw, self.coverage_goal_timeout)
                    self.cancel_goal()
                    break
            rate.sleep()

        if timer is not None:
            timer.shutdown()
        if self.current_goal_timed_out:
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

    def _visit_coverage_point(self, point, patrol_idx):
        """Visit one calibrated anchor once, then perform its original scan."""
        x, y, yaw = exact_observation_target(point)
        if self.triggered:
            return "triggered"

        known, max_cost, _blocked = self._coverage_pose_cost(x, y)
        if should_skip_coverage_anchor(known, max_cost, self.lethal_cost):
            self.cmd_vel_pub.publish(Twist())
            rospy.logwarn(
                "[vision_triggered_navigator] 精确锚点%d footprint被锥桶占据(cost=%d)，仅跳过该原始点: (%.4f, %.4f, %.4f).",
                patrol_idx + 1, max_cost, x, y, yaw)
            return "skipped_blocked"

        rospy.loginfo(
            "[vision_triggered_navigator] 精确锚点%d: (%.4f, %.4f, %.4f)",
            patrol_idx + 1, x, y, yaw)
        result = self.send_goal(x, y, yaw)
        if self.triggered:
            return "triggered"
        if result != actionlib.GoalStatus.SUCCEEDED:
            self.cmd_vel_pub.publish(Twist())
            rospy.logwarn(
                "[vision_triggered_navigator] 精确锚点%d导航未成功(state=%s timeout=%s)，不生成替代坐标，进入下一原始点.",
                patrol_idx + 1, str(result), self.current_goal_timed_out)
            return "skipped_failed"
        if not self._wait_navigation_idle():
            self.cmd_vel_pub.publish(Twist())
            rospy.logerr(
                "[vision_triggered_navigator] 精确锚点%d到达后move_base未释放控制权，禁止观察自转并进入下一原始点.",
                patrol_idx + 1)
            return "skipped_failed"

        self.cmd_vel_pub.publish(Twist())
        initial_hold_at = rospy.get_time()
        self._hold_scan_step(
            "锚点{}初始朝向".format(patrol_idx + 1),
            initial_hold_at - max(self.target_bbox_stale,
                                  self.coverage_scan_dwell))
        if self.triggered:
            return "triggered"
        if not self.perform_rotations(point.get("rotations", [])):
            self.cmd_vel_pub.publish(Twist())
            rospy.logwarn(
                "[vision_triggered_navigator] 精确锚点%d步进扫描未完成，不重访，进入下一原始点.",
                patrol_idx + 1)
            return "skipped_scan_failed"
        if self.triggered:
            return "triggered"
        rospy.loginfo(
            "[vision_triggered_navigator] coverage anchor=%d state=covered exact=true",
            patrol_idx + 1)
        return "covered"

    def _odom_is_fresh(self):
        return (self.odom_yaw is not None and sensor_is_fresh(
            self.odom_received_at, rospy.get_time(), self.odom_stale))

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

    def _transform_map_pose(self, target_frame, pose):
        """Transform one map pose into target_frame, returning an xyz tuple."""
        stamped = PoseStamped()
        stamped.header.frame_id = self.map_frame
        stamped.header.stamp = rospy.Time(0)
        stamped.pose.position.x = float(pose[0])
        stamped.pose.position.y = float(pose[1])
        stamped.pose.orientation = euler_to_quaternion(float(pose[2]))
        try:
            self.tf_listener.waitForTransform(
                target_frame, self.map_frame, rospy.Time(0), rospy.Duration(0.5))
            transformed = self.tf_listener.transformPose(target_frame, stamped)
            return (
                transformed.pose.position.x,
                transformed.pose.position.y,
                quaternion_to_yaw(transformed.pose.orientation),
            )
        except (tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException) as exc:
            rospy.logerr(
                "[vision_triggered_navigator] 无法将map停泊位姿转换到%s: %s",
                target_frame, str(exc))
            return None

    def _transform_wall_geometry(self, target_frame):
        """Transform wall centre and inward normal from map into target_frame."""
        if self.parking_wall_point is None or self.parking_inward_normal is None:
            return None
        wx, wy = self.parking_wall_point
        nx, ny = self.parking_inward_normal
        wall = self._transform_map_pose(target_frame, (wx, wy, 0.0))
        inward = self._transform_map_pose(
            target_frame, (wx + nx, wy + ny, 0.0))
        if wall is None or inward is None:
            return None
        normal_x = inward[0] - wall[0]
        normal_y = inward[1] - wall[1]
        length = math.hypot(normal_x, normal_y)
        if length <= 1e-6:
            rospy.logerr("[vision_triggered_navigator] odom墙面法向量长度为0.")
            return None
        return (wall[0], wall[1]), (normal_x / length, normal_y / length)

    def _make_move_base_goal(self, x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = float(x)
        goal.target_pose.pose.position.y = float(y)
        goal.target_pose.pose.orientation = euler_to_quaternion(float(yaw))
        return goal

    def _navigate_to_parking_staging(self, goal):
        """Use move_base only to reach a safe staging pose, with spin watchdog."""
        x, y, yaw = [float(value) for value in goal]
        pose = self._get_robot_pose(self.base_frame)
        if pose is None:
            rospy.logerr("[vision_triggered_navigator] 无法获取预停点起始位姿.")
            return False
        if math.hypot(x - pose[0], y - pose[1]) <= self.parking_staging_acceptance:
            rospy.loginfo("[vision_triggered_navigator] 已位于预停点%.2fm范围内，跳过move_base.",
                          self.parking_staging_acceptance)
            return self._wait_navigation_idle()

        rospy.loginfo(
            "[vision_triggered_navigator] 发送预停点: x=%.4f y=%.4f yaw=%.4f timeout=%.1fs",
            x, y, yaw, self.parking_staging_timeout)
        self.move_base_client.send_goal(self._make_move_base_goal(x, y, yaw))
        started = rospy.get_time()
        window_started = started
        window_pose = pose
        last_yaw = pose[2]
        yaw_accumulated = 0.0
        reached = False
        failure_reason = ""
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            pose = self._get_robot_pose(self.base_frame)
            if pose is not None:
                distance = math.hypot(x - pose[0], y - pose[1])
                yaw_accumulated += abs(normalize_angle(pose[2] - last_yaw))
                last_yaw = pose[2]
                if distance <= self.parking_staging_acceptance:
                    reached = True
                    break
                if rospy.get_time() - window_started >= self.parking_staging_watchdog_window:
                    moved = math.hypot(pose[0] - window_pose[0],
                                       pose[1] - window_pose[1])
                    if staging_motion_is_rotation_stall(
                            moved, yaw_accumulated,
                            self.parking_staging_min_progress,
                            self.parking_staging_max_rotation):
                        failure_reason = (
                            "预停点出现原地旋转: %.1fs位移%.3fm累计转角%.1fdeg" %
                            (self.parking_staging_watchdog_window, moved,
                             math.degrees(yaw_accumulated)))
                        break
                    window_started = rospy.get_time()
                    window_pose = pose
                    yaw_accumulated = 0.0

            state = self.move_base_client.get_state()
            if state not in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
                failure_reason = "move_base预停点提前结束(state=%s)" % str(state)
                break
            if rospy.get_time() - started >= self.parking_staging_timeout:
                failure_reason = "预停点导航超过%.1fs" % self.parking_staging_timeout
                break
            rate.sleep()

        self.move_base_client.cancel_goal()
        idle = self._wait_navigation_idle(timeout=2.0)
        self.cmd_vel_pub.publish(Twist())
        if reached and idle:
            rospy.loginfo("[vision_triggered_navigator] 预停点交接完成，move_base已释放控制权.")
            return True
        if not idle:
            failure_reason = "move_base未释放/cmd_vel控制权"
        rospy.logerr("[vision_triggered_navigator] parking_staging_failed: %s",
                     failure_reason or "未知原因")
        return False

    def _run_parking_docking(self, map_goal):
        """Finish the last short approach in a locked odom frame."""
        if not self._wait_navigation_idle(timeout=2.0):
            rospy.logerr("[vision_triggered_navigator] move_base仍占用控制权，拒绝直接停泊.")
            return False
        if not self._odom_is_fresh() or self.odom_pose is None:
            rospy.logerr("[vision_triggered_navigator] /odom不新鲜，拒绝直接停泊.")
            return False
        if (not sensor_is_fresh(self.scan_received_at, rospy.get_time(),
                                self.scan_stale) or
                self.scan_front_min is None):
            rospy.logerr("[vision_triggered_navigator] /scan不新鲜或前向无有效量程，拒绝直接停泊.")
            return False

        odom_frame = self.odom_frame_from_msg or self.odom_frame
        target = self._transform_map_pose(odom_frame, map_goal)
        wall_geometry = self._transform_wall_geometry(odom_frame)
        if target is None or wall_geometry is None:
            return False
        wall_point, inward_normal = wall_geometry
        rospy.loginfo(
            "[vision_triggered_navigator] 锁定odom停泊目标 frame=%s target=(%.4f,%.4f,%.4f)",
            odom_frame, target[0], target[1], target[2])

        deadline = rospy.get_time() + self.parking_docking_timeout
        stable_since = None
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.get_time() < deadline:
            if not self._odom_is_fresh() or self.odom_pose is None:
                self.cmd_vel_pub.publish(Twist())
                rospy.logerr("[vision_triggered_navigator] 停泊期间/odom超过%.2fs未更新.",
                             self.odom_stale)
                return False
            if (not sensor_is_fresh(self.scan_received_at, rospy.get_time(),
                                    self.scan_stale) or
                    self.scan_front_min is None):
                self.cmd_vel_pub.publish(Twist())
                rospy.logerr("[vision_triggered_navigator] 停泊期间/scan超过%.2fs未更新.",
                             self.scan_stale)
                return False

            errors = docking_pose_errors(self.odom_pose, target)
            distance_to_wall = wall_normal_distance(
                self.odom_pose, wall_point, inward_normal)
            lidar_base_distance = lidar_base_wall_distance(
                self.scan_front_min, self.parking_lidar_forward_offset)
            if docking_within_tolerance(
                    errors,
                    self.parking_dock_normal_tolerance,
                    self.parking_dock_tangent_tolerance,
                    self.parking_dock_yaw_tolerance):
                self.cmd_vel_pub.publish(Twist())
                if stable_since is None:
                    stable_since = rospy.get_time()
                elif rospy.get_time() - stable_since >= self.parking_dock_stable_sec:
                    rospy.loginfo(
                        "[vision_triggered_navigator] odom停泊收敛 stable=%.2fs errors=(%.3f,%.3f,%.3f)",
                        self.parking_dock_stable_sec,
                        errors[0], errors[1], errors[2])
                    return True
                rate.sleep()
                continue
            stable_since = None

            if distance_to_wall < self.parking_min_wall_distance:
                self.cmd_vel_pub.publish(Twist())
                rospy.logerr(
                    "[vision_triggered_navigator] 几何墙距%.3fm小于硬限%.3fm，立即停车.",
                    distance_to_wall, self.parking_min_wall_distance)
                return False
            if lidar_requires_stop(
                    self.scan_front_min, lidar_base_distance, distance_to_wall,
                    self.parking_lidar_stop_distance):
                self.cmd_vel_pub.publish(Twist())
                rospy.logerr(
                    "[vision_triggered_navigator] 雷达近障碍触发停车: base等效=%.3fm raw=%.3f geometry=%.3f offset=%+.3f limit=%.3f.",
                    lidar_base_distance, self.scan_front_min,
                    distance_to_wall,
                    self.parking_lidar_forward_offset,
                    self.parking_lidar_stop_distance)
                return False

            command = docking_command(
                errors,
                self.parking_dock_normal_tolerance,
                self.parking_dock_tangent_tolerance,
                self.parking_dock_yaw_tolerance,
                self.parking_dock_max_x,
                self.parking_dock_max_y,
                self.parking_dock_max_yaw,
            )
            twist = Twist()
            twist.linear.x, twist.linear.y, twist.angular.z = command
            self.cmd_vel_pub.publish(twist)
            rospy.loginfo_throttle(
                0.5,
                "[vision_triggered_navigator] docking errors=(forward=%+.3f lateral=%+.3f yaw=%+.3f) cmd=(%+.3f,%+.3f,%+.3f) wall=%.3f lidar_base=%.3f",
                errors[0], errors[1], errors[2], command[0], command[1], command[2],
                distance_to_wall, lidar_base_distance)
            rate.sleep()

        self.cmd_vel_pub.publish(Twist())
        rospy.logerr("[vision_triggered_navigator] 停泊闭环超过%.1fs仍未收敛.",
                     self.parking_docking_timeout)
        return False

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
        diagnostics = parking_footprint_margins(
            pose,
            self.parking_wall_point,
            self.parking_inward_normal,
            self.parking_box_width,
            self.parking_box_depth,
            self.footprint_half_length,
            self.footprint_half_width,
            self.parking_validation_margin,
        )
        # diagnostics margins already exclude the legacy validation margin;
        # add it back so parking_required_margin is the physical box margin.
        minimum_margin = self.parking_validation_margin + min(
            float(diagnostics.get("near_margin", float("-inf"))),
            float(diagnostics.get("far_margin", float("-inf"))),
            float(diagnostics.get("side_margin", float("-inf"))),
        )
        valid = (bool(diagnostics.get("inside")) and
                 minimum_margin >= self.parking_required_margin)
        rospy.loginfo(
            "[vision_triggered_navigator] 停泊框验证 wall=%s pose=(%.4f, %.4f, %.4f) "
            "box=%.2fx%.2f normal=[%.3f,%.3f] tangent_abs=%.3f "
            "error(normal=%+.3f tangent=%+.3f) "
            "margins(near=%.3f far=%.3f side=%.3f min=%.3f required=%.3f) full_footprint_inside=%s",
            self.parking_wall_name or "unknown",
            pose[0], pose[1], pose[2], self.parking_box_width,
            self.parking_box_depth,
            float(diagnostics.get("normal_min", float("nan"))),
            float(diagnostics.get("normal_max", float("nan"))),
            float(diagnostics.get("tangent_abs_max", float("nan"))),
            float(diagnostics.get("normal_error", float("nan"))),
            float(diagnostics.get("tangent_error", float("nan"))),
            float(diagnostics.get("near_margin", float("nan"))),
            float(diagnostics.get("far_margin", float("nan"))),
            float(diagnostics.get("side_margin", float("nan"))),
            minimum_margin, self.parking_required_margin,
            valid)
        for index, corner in enumerate(diagnostics.get("corners", []), 1):
            rospy.loginfo(
                "[vision_triggered_navigator] footprint corner%d map=(%.3f,%.3f) "
                "normal=%.3f tangent=%.3f margins(near=%.3f far=%.3f side=%.3f)",
                index, corner["x"], corner["y"],
                corner["normal"], corner["tangent"],
                corner["near_margin"], corner["far_margin"],
                corner["side_margin"])
        return valid

    # ------------------------------------------------------------------
    # 视觉触发目标计算
    # ------------------------------------------------------------------
    def compute_vision_goal(self):
        """
        根据 base_link 车头正方向射线与实测四边形围墙求交，
        沿墙法向和切向连续计算停车目标，返回 (x, y, yaw)。
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
        best_wall_name = None

        for wall_name, a, b, normal in self.walls:
            t = ray_segment_intersection((px, py), (dx, dy), a, b)
            if t is not None and t < best_t:
                best_t = t
                best_normal = normal
                best_point = (px + t * dx, py + t * dy)
                best_wall_name = wall_name

        if best_point is None:
            rospy.logerr("[vision_triggered_navigator] 射线与围墙无交点，无法计算视觉目标.")
            return None

        ix, iy = best_point
        nx, ny = best_normal
        gx, gy, gyaw = parking_goal_from_wall(
            best_point,
            best_normal,
            self.parking_goal_offset,
            self.parking_normal_offset,
            self.parking_tangent_offset,
        )

        rospy.loginfo(
            "[vision_triggered_navigator] 墙段=%s 交点=(%.4f,%.4f) "
            "内法向=(%.4f,%.4f) normal_offset=%+.3f tangent_offset=%+.3f "
            "目标点=(%.4f,%.4f,yaw=%.4f)",
            best_wall_name, ix, iy, nx, ny,
            self.parking_normal_offset, self.parking_tangent_offset,
            gx, gy, gyaw)
        self.parking_wall_point = (ix, iy)
        self.parking_inward_normal = (nx, ny)
        self.parking_wall_name = best_wall_name
        return gx, gy, gyaw

    def compute_staging_goal(self):
        """Build the safe move_base handoff pose from the locked wall geometry."""
        if self.parking_wall_point is None or self.parking_inward_normal is None:
            return None
        return parking_goal_from_wall(
            self.parking_wall_point,
            self.parking_inward_normal,
            self.parking_staging_offset,
            self.parking_normal_offset,
            self.parking_tangent_offset,
        )

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
        coverage_position = 0

        while not rospy.is_shutdown():
            # 一旦被触发，立即切换到视觉阶段
            if self.triggered and state == "PATROL":
                rospy.loginfo("[vision_triggered_navigator] 巡航被打断，进入视觉触发阶段.")
                state = "VISION"
                continue

            if state == "PATROL":
                if self.coverage_search_mode:
                    if coverage_position >= coverage_count:
                        rospy.logerr(
                            "[vision_triggered_navigator] %d个精确观察点已按原顺序处理完成，但未锁定目标.",
                            coverage_count)
                        self._publish_status("failed")
                        break

                    point_idx = coverage_position
                    point = self.patrol_points[point_idx]
                    rospy.loginfo(
                        "[vision_triggered_navigator] === 覆盖锚点 %d / %d，逻辑编号%d ===",
                        coverage_position + 1, coverage_count, point_idx + 1)
                    outcome = self._visit_coverage_point(point, point_idx)
                    if outcome == "triggered":
                        state = "VISION"
                        continue
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
                goal = self.compute_vision_goal()
                if goal is not None:
                    gx, gy, gyaw = goal
                else:
                    rospy.logerr("[vision_triggered_navigator] 视觉目标计算失败.")
                    self._publish_status("failed")
                    break
                staging_goal = self.compute_staging_goal()
                if staging_goal is None:
                    self._publish_status("parking_staging_failed")
                    self._hold_stopped(self.arrival_hold_sec)
                    break
                self._publish_status("parking_staging")
                if not self._navigate_to_parking_staging(staging_goal):
                    self._publish_status("parking_staging_failed")
                    self._hold_stopped(self.arrival_hold_sec)
                    break
                self._publish_status("parking_docking")
                if not self._run_parking_docking((gx, gy, gyaw)):
                    self._publish_status("parking_docking_failed")
                    self._hold_stopped(self.arrival_hold_sec)
                    break
                self._hold_stopped(self.arrival_hold_sec)
                self._publish_status("parking_verifying")
                parking_valid = self._validate_parking_pose()
                if not parking_valid:
                    self._publish_status("parking_validation_failed")
                    self._hold_stopped(self.arrival_hold_sec)
                    rospy.logerr(
                        "[vision_triggered_navigator] 低速闭环已结束，但完整footprint未达到50cm框2cm余量要求.")
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
