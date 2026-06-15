cmake_minimum_required(VERSION 3.0.2)
project(yolo)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  test_factory_sign.py
  validate_model.py
  scripts/factory_sign_rknn_test_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config models scripts
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
