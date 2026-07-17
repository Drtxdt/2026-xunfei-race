cmake_minimum_required(VERSION 2.8.3)
project(vision_triggered_navigator)

find_package(catkin REQUIRED COMPONENTS
  rospy
  move_base_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  actionlib
  tf
  std_msgs
  dynamic_reconfigure
  std_srvs
)

catkin_package(
  CATKIN_DEPENDS rospy move_base_msgs geometry_msgs nav_msgs sensor_msgs actionlib tf std_msgs dynamic_reconfigure std_srvs
)

catkin_install_python(PROGRAMS scripts/vision_triggered_navigator.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES scripts/navigator_logic.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
