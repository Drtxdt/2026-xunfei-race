cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_traffic_stop_line)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/traffic_stop_line_node.py
  scripts/stop_line_logic.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
