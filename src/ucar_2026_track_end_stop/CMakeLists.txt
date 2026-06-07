cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_track_end_stop)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(track_end_stop_node
  src/track_end_stop_node.cpp
)
target_link_libraries(track_end_stop_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(right_track_end_stop_node
  src/right_track_end_stop_node.cpp
)
target_link_libraries(right_track_end_stop_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(stable_right_track_end_stop_node
  src/stable_right_track_end_stop_node.cpp
)
target_link_libraries(stable_right_track_end_stop_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

install(TARGETS track_end_stop_node right_track_end_stop_node stable_right_track_end_stop_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
