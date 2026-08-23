#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""黄线停车 staging 点导航精度测试节点。

流程：
1. 按 yaml 中的初始位姿猜测 + 误差参数计算协方差，发布 /initialpose；
2. 等待 initial_pose_settle_sec 秒让 AMCL 收敛；
3. 通过 move_base 导航到 traffic_x/y/yaw 标定点；
4. 结束后用 TF 读取 map->base_link 实际位姿，打印与目标点的残差，
   供人工对照肉眼结果精调 traffic_x/y/yaw。

参数只经 config/yellow_stop_nav_test.yaml 传入，launch 不做任何覆盖。
节点完成一次后自动退出；把车摆回起点后可直接
`rosrun test_yellow_stop yellow_stop_nav_test_node.py` 重测。
"""

import math

import actionlib
import rospy
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def quaternion_from_yaw(yaw):
    half = float(yaw) * 0.5
    return math.sin(half), math.cos(half)


class YellowStopNavTest(object):
    def __init__(self):
        # 初始位姿猜测：车在 map 系下的实际摆放位姿【标定】
        self.initial_x = float(rospy.get_param("~initial_pose_x", 0.0))
        self.initial_y = float(rospy.get_param("~initial_pose_y", 0.0))
        self.initial_yaw = float(rospy.get_param("~initial_pose_yaw", 0.0))
        # 初始位姿误差估计：用于计算猜测协方差
        self.xy_sigma = float(rospy.get_param("~initial_pose_xy_sigma_m", 0.10))
        self.yaw_sigma = math.radians(float(rospy.get_param(
            "~initial_pose_yaw_sigma_deg", 10.0)))
        self.subscriber_wait_sec = float(rospy.get_param(
            "~initial_pose_subscriber_wait_sec", 5.0))
        self.settle_sec = float(rospy.get_param(
            "~initial_pose_settle_sec", 3.0))
        # 被测的黄线停车 staging 标定点（map 系）
        self.goal_x = float(rospy.get_param("~traffic_x", 0.2395))
        self.goal_y = float(rospy.get_param("~traffic_y", -3.10))
        self.goal_yaw = float(rospy.get_param("~traffic_yaw", -1.5596))
        self.goal_timeout = float(rospy.get_param(
            "~move_base_timeout_sec", 60.0))

        self.initialpose_pub = rospy.Publisher(
            "/initialpose", PoseWithCovarianceStamped, queue_size=1,
            latch=True)
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.move_base = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)

    def publish_initial_pose(self):
        wait_deadline = rospy.get_time() + self.subscriber_wait_sec
        while (self.initialpose_pub.get_num_connections() == 0 and
               rospy.get_time() < wait_deadline and not rospy.is_shutdown()):
            rospy.sleep(0.1)
        if self.initialpose_pub.get_num_connections() == 0:
            rospy.logwarn(
                "【黄线测试】%.1fs 内没有 /initialpose 订阅者（AMCL 未就绪？），"
                "仍尝试发布", self.subscriber_wait_sec)

        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.pose.position.x = self.initial_x
        pose.pose.pose.position.y = self.initial_y
        pose.pose.pose.position.z = 0.0
        sin_half, cos_half = quaternion_from_yaw(self.initial_yaw)
        pose.pose.pose.orientation.z = sin_half
        pose.pose.pose.orientation.w = cos_half
        # 由误差参数（标准差）计算协方差：x/y 取 xy_sigma^2，yaw 取 yaw_sigma^2
        xy_var = self.xy_sigma ** 2
        yaw_var = self.yaw_sigma ** 2
        cov = [0.0] * 36
        cov[0] = xy_var
        cov[7] = xy_var
        cov[35] = yaw_var
        pose.pose.covariance = cov
        self.initialpose_pub.publish(pose)
        rospy.loginfo(
            "【黄线测试】已发布初始位姿猜测: x=%.4f y=%.4f yaw=%.4f "
            "(xy_σ=%.3fm yaw_σ=%.1f°)，等待 %.1fs 让 AMCL 收敛",
            self.initial_x, self.initial_y, self.initial_yaw,
            self.xy_sigma, math.degrees(self.yaw_sigma), self.settle_sec)
        rospy.sleep(self.settle_sec)

    def navigate_to_goal(self):
        if not self.move_base.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("【黄线测试】move_base action server 不可用，测试终止")
            return False
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = self.goal_x
        goal.target_pose.pose.position.y = self.goal_y
        sin_half, cos_half = quaternion_from_yaw(self.goal_yaw)
        goal.target_pose.pose.orientation.z = sin_half
        goal.target_pose.pose.orientation.w = cos_half
        rospy.loginfo(
            "【黄线测试】发送导航目标: x=%.4f y=%.4f yaw=%.4f（超时 %.0fs）",
            self.goal_x, self.goal_y, self.goal_yaw, self.goal_timeout)
        self.move_base.send_goal(goal)
        finished = self.move_base.wait_for_result(
            rospy.Duration(self.goal_timeout))
        if not finished:
            self.move_base.cancel_goal()
            rospy.logerr("【黄线测试】导航超时（%.0fs），已取消目标", self.goal_timeout)
            return False
        state = self.move_base.get_state()
        rospy.loginfo("【黄线测试】导航结束，move_base 状态码=%d（3=成功）", state)
        return state == 3

    def report_residual(self):
        """用 TF 读取最终实际位姿，打印与目标点的残差供调参对照。"""
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_link", rospy.Time(0), rospy.Duration(1.0))
        except tf2_ros.TransformException as exc:
            rospy.logwarn("【黄线测试】无法获取 map->base_link，跳过残差报告: %s",
                          exc)
            return
        t = transform.transform.translation
        q = transform.transform.rotation
        actual_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        dx = self.goal_x - t.x
        dy = self.goal_y - t.y
        dyaw = math.atan2(math.sin(self.goal_yaw - actual_yaw),
                          math.cos(self.goal_yaw - actual_yaw))
        rospy.loginfo(
            "【黄线测试】实际位姿: x=%.4f y=%.4f yaw=%.4f",
            t.x, t.y, actual_yaw)
        rospy.loginfo(
            "【黄线测试】目标残差: dx=%+.4fm dy=%+.4fm dyaw=%+.2f° "
            "（正 dx/dy = 实际位姿未到目标，需相应调大 traffic_x/y）",
            dx, dy, math.degrees(dyaw))

    def run(self):
        self.publish_initial_pose()
        if rospy.is_shutdown():
            return
        if self.navigate_to_goal():
            self.report_residual()
        rospy.loginfo("【黄线测试】测试完成，节点退出；"
                      "摆车后可 rosrun 本节点重测")


def main():
    rospy.init_node("yellow_stop_nav_test")
    YellowStopNavTest().run()


if __name__ == "__main__":
    main()
