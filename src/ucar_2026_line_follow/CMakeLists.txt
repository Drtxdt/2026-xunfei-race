cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_line_follow)

add_compile_options(-std=c++14)

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

# 原有的 C++ 节点，仅在源文件存在时添加
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/right_line_follow_cpp_node.cpp)
  add_executable(right_line_follow_cpp_node
    src/right_line_follow_cpp_node.cpp
  )
  target_link_libraries(right_line_follow_cpp_node
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/left_line_follow_cpp_node.cpp)
  add_executable(left_line_follow_cpp_node
    src/left_line_follow_cpp_node.cpp
  )
  target_link_libraries(left_line_follow_cpp_node
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )
endif()

# 新增：您的 stable_right_follow 节点
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/stable_right_follow.cpp)
  add_executable(stable_right_follow
    src/stable_right_follow.cpp
  )
  target_link_libraries(stable_right_follow
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
  )
endif()

# Python 脚本安装（保持不变）
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

# 安装所有存在的 C++ 可执行文件
set(INSTALL_TARGETS "")
if(TARGET right_line_follow_cpp_node)
  list(APPEND INSTALL_TARGETS right_line_follow_cpp_node)
endif()
if(TARGET left_line_follow_cpp_node)
  list(APPEND INSTALL_TARGETS left_line_follow_cpp_node)
endif()
if(TARGET stable_right_follow)
  list(APPEND INSTALL_TARGETS stable_right_follow)
endif()

if(INSTALL_TARGETS)
  install(TARGETS ${INSTALL_TARGETS}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
endif()

# 安装 config 和 launch 目录
install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)