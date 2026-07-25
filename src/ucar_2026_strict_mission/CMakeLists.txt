cmake_minimum_required(VERSION 3.0.2)
project(ucar_2026_strict_mission)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  move_base_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
)

catkin_python_setup()
catkin_package()

catkin_install_python(
  PROGRAMS scripts/strict_mission_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_logic.py)
endif()
