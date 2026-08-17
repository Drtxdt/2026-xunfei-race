cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_national_competition)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  geometry_msgs
  move_base_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
)

catkin_python_setup()
catkin_package()

catkin_install_python(
  PROGRAMS scripts/national_flow_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_logic.py)
endif()
