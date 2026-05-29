cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_aikit_tts)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  std_msgs
)

catkin_package()

set(AIKIT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/aikit)
set(AIKIT_LIB_DIR ${AIKIT_ROOT}/libs)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${AIKIT_ROOT}/include
)

add_executable(aikit_speak_node src/aikit_speak_node.cpp)
target_link_libraries(aikit_speak_node
  ${catkin_LIBRARIES}
  ${AIKIT_LIB_DIR}/libaikit.so
  pthread
  dl
)
set_target_properties(aikit_speak_node PROPERTIES
  BUILD_RPATH "${AIKIT_LIB_DIR}"
  INSTALL_RPATH "${AIKIT_LIB_DIR}"
)

install(TARGETS aikit_speak_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch third_party
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
