cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_line_follow)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/line_follow_node.py
  scripts/parking_debug_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
