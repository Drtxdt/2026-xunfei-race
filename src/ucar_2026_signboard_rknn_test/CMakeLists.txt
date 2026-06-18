cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_signboard_rknn_test)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  ucar_2026_competition_speech
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/check_signboard_rknn_test.py
  scripts/signboard_rknn_test_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
