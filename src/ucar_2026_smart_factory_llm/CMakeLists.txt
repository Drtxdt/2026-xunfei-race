cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_smart_factory_llm)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  message_generation
  move_base_msgs
  rospy
  std_msgs
  ucar_2026_competition_speech
)

add_service_files(
  FILES
  ReasonPickupOrder.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS actionlib actionlib_msgs geometry_msgs message_runtime move_base_msgs rospy std_msgs ucar_2026_competition_speech
)

catkin_install_python(PROGRAMS
  scripts/reason_pickup_server.py
  scripts/reason_and_speak_once.py
  scripts/qr_to_llm_speak_once.py
  scripts/task1_full_once.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
