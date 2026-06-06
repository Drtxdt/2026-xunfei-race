cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_competition_speech)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  std_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  Announce.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy std_msgs
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test)
endif()

catkin_install_python(PROGRAMS
  scripts/competition_announcer.py
  scripts/announce_once.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
