cmake_minimum_required(VERSION 2.8.3)
project(simple_navigator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  move_base_msgs
  geometry_msgs
  actionlib
  tf
)

catkin_package(
  CATKIN_DEPENDS rospy move_base_msgs geometry_msgs actionlib tf
)

catkin_install_python(PROGRAMS scripts/simple_navigator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
