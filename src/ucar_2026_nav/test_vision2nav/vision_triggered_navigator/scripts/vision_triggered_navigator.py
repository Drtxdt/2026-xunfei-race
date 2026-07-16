#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import actionlib
import tf
import math
import threading
import sys

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Quaternion, Twist, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool


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

        self.costmap = None
        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self._costmap_cb)

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

    # ------------------------------------------------------------------
    # 回调与工具函数
    # ------------------------------------------------------------------
    def _costmap_cb(self, msg):
        """保存最新 costmap"""
        self.costmap = msg

    def _vision_cb(self, msg):
        """视觉触发回调"""
        if msg.data and not self.triggered:
            rospy.loginfo("[vision_triggered_navigator] 收到视觉触发信号，打断当前导航.")
            self.triggered = True
            self.cancel_goal()

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

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation = euler_to_quaternion(yaw)

        rospy.loginfo("[vision_triggered_navigator] 发送导航目标: x=%.4f y=%.4f yaw=%.4f", x, y, yaw)
        self.move_base_client.send_goal(goal)

        # 启动定时器，周期性检查当前目标可行性
        period = rospy.Duration(1.0 / max(self.feasibility_check_rate, 0.1))
        # 独立线程启动定时器，避免阻塞主循环
        timer = rospy.Timer(period, self._check_current_goal_cb)

        # 只要不被视觉、键盘、代价变高触发，就一直轮询等待导航完成
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            state = self.move_base_client.get_state()
            if state not in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
                break
            rate.sleep()

        timer.shutdown()
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

    def rotate(self, direction, duration):
        """
        发布角速度使机器人自转。
        direction: "left" 左转，"right" 右转
        duration: 保持时间（秒）
        """
        if duration <= 0:
            return

        twist = Twist()
        twist.angular.z = self.rotation_speed if direction == "left" else -self.rotation_speed

        rospy.loginfo("[vision_triggered_navigator] 自转 %s, 速度 %.2f rad/s, 保持 %.2f s",
                      direction, twist.angular.z, duration)

        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start).to_sec()
            # 轮询时检查是否被视觉/键盘触发打断，打断立即停止
            if elapsed >= duration or self.triggered:
                break
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # 发送零速停止
        self.cmd_vel_pub.publish(Twist())

    def perform_rotations(self, rotations):
        """顺序执行一组自转动作"""
        for rot in rotations:
            if self.triggered:
                break
            direction = rot.get("direction", "left")
            duration = rot.get("duration", 0.0)
            self.rotate(direction, duration)

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
        gx = ix + nx * self.vision_offset
        gy = iy + ny * self.vision_offset
        # 车头垂直指向墙外（与内法向相反）
        gyaw = math.atan2(-ny, -nx)

        rospy.loginfo("[vision_triggered_navigator] 墙交点 (%.4f, %.4f), 外法向 (%.2f, %.2f), "
                      "目标点 (%.4f, %.4f, yaw=%.4f)",
                      ix, iy, -nx, -ny, gx, gy, gyaw)
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

        # 步骤 0：给 AMCL 发送初始位姿
        self.publish_initial_pose_to_amcl()

        state = "PATROL"
        patrol_idx = 0

        while not rospy.is_shutdown():
            # 一旦被触发，立即切换到视觉阶段
            if self.triggered and state == "PATROL":
                rospy.loginfo("[vision_triggered_navigator] 巡航被打断，进入视觉触发阶段.")
                state = "VISION"
                continue

            if state == "PATROL":
                if patrol_idx >= len(self.patrol_points):
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
                goal = self.compute_vision_goal()
                if goal is not None:
                    gx, gy, gyaw = goal
                    self.send_goal(gx, gy, gyaw)
                else:
                    rospy.logerr("[vision_triggered_navigator] 视觉目标计算失败，直接进入结束点阶段.")
                state = "END"

            elif state == "END":
                rospy.loginfo("[vision_triggered_navigator] === 结束点阶段 ===")
                x = self.end_goal["x"]
                y = self.end_goal["y"]
                yaw = self.end_goal["yaw"]
                self.send_goal(x, y, yaw)
                rospy.loginfo("[vision_triggered_navigator] 全部流程结束.")
                break

            else:
                break

        rospy.spin()


def main():
    node = VisionTriggeredNavigator()
    node.run()


if __name__ == "__main__":
    main()
