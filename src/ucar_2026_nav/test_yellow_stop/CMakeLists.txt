cmake_minimum_required(VERSION 3.0.2)
project(test_yellow_stop)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  geometry_msgs
  move_base_msgs
  rospy
  std_msgs
  tf2_ros
)

catkin_package()

catkin_install_python(
  PROGRAMS scripts/yellow_stop_nav_test_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
