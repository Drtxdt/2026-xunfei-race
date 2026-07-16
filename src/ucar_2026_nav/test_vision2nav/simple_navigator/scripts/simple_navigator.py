#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped, Quaternion, PoseWithCovarianceStamped
import tf


def euler_to_quaternion(yaw):
    """将 yaw 角转换为 geometry_msgs/Quaternion"""
    q = tf.transformations.quaternion_from_euler(0, 0, yaw)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


def get_goal_from_params():
    """
    读取目标点参数。
    优先级：
      1. /map_goal_picker/goal_* （由 map_goal_picker 写入）
      2. 当前节点私有参数 goal_* （通过 launch / YAML 加载）
    """
    goal_x = None
    goal_y = None
    goal_yaw = None
    source = ""

    # 尝试从 map_goal_picker 的参数读取
    if rospy.has_param("/map_goal_picker/goal_x"):
        goal_x = rospy.get_param("/map_goal_picker/goal_x")
        goal_y = rospy.get_param("/map_goal_picker/goal_y")
        goal_yaw = rospy.get_param("/map_goal_picker/goal_yaw")
        source = "parameter server (/map_goal_picker)"
        rospy.loginfo("[simple_navigator] Loaded goal from parameter server (map_goal_picker).")
    else:
        # 回退到私有参数
        goal_x = rospy.get_param("~goal_x", 0.0)
        goal_y = rospy.get_param("~goal_y", 0.0)
        goal_yaw = rospy.get_param("~goal_yaw", 0.0)
        source = "private parameter (YAML default)"
        rospy.logwarn("[simple_navigator] /map_goal_picker/goal_* not found. Using default YAML values.")

    rospy.loginfo("[simple_navigator] Goal from %s:", source)
    rospy.loginfo("[simple_navigator]   x   = %.4f", goal_x)
    rospy.loginfo("[simple_navigator]   y   = %.4f", goal_y)
    rospy.loginfo("[simple_navigator]   yaw = %.4f rad", goal_yaw)

    return goal_x, goal_y, goal_yaw


def send_goal_to_move_base(x, y, yaw):
    """通过 actionlib 发送导航目标到 move_base"""
    move_base_server = rospy.get_param("~move_base_server", "/move_base")
    map_frame = rospy.get_param("~map_frame", "map")

    client = actionlib.SimpleActionClient(move_base_server, MoveBaseAction)
    rospy.loginfo("[simple_navigator] 等待 move_base 服务器...")
    client.wait_for_server()
    rospy.loginfo("[simple_navigator] move_base 服务器已连接.")

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = map_frame
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation = euler_to_quaternion(yaw)

    rospy.loginfo("[simple_navigator] 发送导航目标到 move_base...")
    client.send_goal(goal)

    # 等待结果，允许通过 Ctrl+C 中断
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and client.get_state() in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
        rate.sleep()

    result_state = client.get_state()
    if result_state == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("[simple_navigator] 导航成功!")
    elif result_state == actionlib.GoalStatus.PREEMPTED:
        rospy.logwarn("[simple_navigator] 导航被抢占.")
    elif result_state == actionlib.GoalStatus.ABORTED:
        rospy.logerr("[simple_navigator] 导航被终止.")
    else:
        rospy.logerr("[simple_navigator] 导航结束，状态为: %s", str(result_state))

    return result_state == actionlib.GoalStatus.SUCCEEDED


def publish_initial_pose(x=0.0, y=0.0, yaw=0.0, frame_id="map"):
    """
    发布 2D Pose Estimate 到 /initialpose，供 AMCL 做初始定位。
    等价于在 rviz 中点击 '2D Pose Estimate' 工具。
    """
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)

    # 等待订阅者连接（AMCL 节点）
    rospy.loginfo("[simple_navigator] 等待 /initialpose 订阅者...")
    wait_start = rospy.Time.now()
    while pub.get_num_connections() == 0:
        if (rospy.Time.now() - wait_start).to_sec() > 5.0:
            rospy.logwarn("[simple_navigator] 5秒内没有 /initialpose 订阅者，继续执行...")
            break
        rospy.sleep(0.1)

    msg = PoseWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id

    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0
    msg.pose.pose.orientation = euler_to_quaternion(yaw)

    # covariance 矩阵：初始定位不确定性，可自行调整
    # [x, y, z, roll, pitch, yaw]
    msg.pose.covariance = [
        0.1, 0,   0,   0,   0,   0,
        0,   0.1, 0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0.04
    ]

    pub.publish(msg)
    rospy.loginfo("[simple_navigator] 已发布初始定位 /initialpose: x=%.2f, y=%.2f, yaw=%.2f", x, y, yaw)
    rospy.sleep(1.0)  # 给 AMCL 一点处理时间


def main():
    rospy.init_node("simple_navigator")
    rospy.loginfo("[simple_navigator] Node started.")

    # ===== 步骤1：给 AMCL 发送初始定位 =====
    publish_initial_pose(x=0.0, y=0.0, yaw=0.0)

    # 读取目标点
    x, y, yaw = get_goal_from_params()

    # 执行单点导航
    success = send_goal_to_move_base(x, y, yaw)

    if success:
        rospy.loginfo("[simple_navigator] 导航完成.")
    else:
        rospy.logerr("[simple_navigator] 导航失败.")


if __name__ == "__main__":
    main()
