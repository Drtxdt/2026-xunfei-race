cmake_minimum_required(VERSION 3.0.2)
project(factory_sign_ppocr_rknn_test)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/factory_sign_ppocr_rknn_node.py
  scripts/check_ppocr_rknn_env.py
  scripts/test_image_ppocr_rknn.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
