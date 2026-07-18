cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_traffic_light_rknn_test)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
  ucar_2026_competition_speech
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/check_traffic_light_rknn_test.py
  scripts/traffic_light_rknn_test_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_traffic_light_classifier.py)
endif()
