cmake_minimum_required(VERSION 3.0.2)
project(cloud_aligner)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  tf
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/alignment_node.py
  scripts/prepare_reference.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
