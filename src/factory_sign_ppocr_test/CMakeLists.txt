cmake_minimum_required(VERSION 3.0.2)
project(factory_sign_ppocr_test)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/factory_sign_ppocr_node.py
  scripts/ppocr_worker.py
  scripts/check_ppocr_env.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
