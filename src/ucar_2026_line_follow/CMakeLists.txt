cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_line_follow)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(right_line_follow_cpp_node
  src/right_line_follow_cpp_node.cpp
)

target_link_libraries(right_line_follow_cpp_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/camera_window_debug_node.py
  scripts/line_follow_node.py
  scripts/parking_debug_node.py
  scripts/line_follow_straight_right.py
  scripts/parking_debug_straight_right.py
  scripts/right_debug_relay_node.py
  scripts/right_line_follow_node.py
  scripts/right_parking_debug_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS right_line_follow_cpp_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
