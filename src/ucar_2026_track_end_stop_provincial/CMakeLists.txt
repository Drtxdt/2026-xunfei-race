cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_track_end_stop_provincial)

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

# CMake targets must be workspace-unique because the original
# ucar_2026_track_end_stop package is still present.  We keep the binary
# names identical so the launch files can use the same type="..." values.
add_executable(provincial_track_end_stop_node
  src/track_end_stop_node.cpp
)
set_target_properties(provincial_track_end_stop_node PROPERTIES OUTPUT_NAME track_end_stop_node)
target_link_libraries(provincial_track_end_stop_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(provincial_right_track_end_stop_node
  src/right_track_end_stop_node.cpp
)
set_target_properties(provincial_right_track_end_stop_node PROPERTIES OUTPUT_NAME right_track_end_stop_node)
target_link_libraries(provincial_right_track_end_stop_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(provincial_stable_right_track_end_stop_node
  src/stable_right_track_end_stop_node.cpp
)
set_target_properties(provincial_stable_right_track_end_stop_node PROPERTIES OUTPUT_NAME stable_right_track_end_stop_node)
target_link_libraries(provincial_stable_right_track_end_stop_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

install(TARGETS provincial_track_end_stop_node provincial_right_track_end_stop_node provincial_stable_right_track_end_stop_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
