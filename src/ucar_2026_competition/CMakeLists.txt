cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_competition)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  move_base_msgs
  nav_msgs
  sensor_msgs
  rospy
  std_msgs
  std_srvs
  ucar_2026_competition_speech
  ucar_2026_smart_factory_llm
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/competition_flow.py
  scripts/external_voice_nodes.py
  scripts/fixed_command_iat.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
