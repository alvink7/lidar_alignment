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

# ---------------------------------------------------------------------------
# Scan Context C++ core (place recognition front-end) -> scancontext_py.so
# See scancontext/README / cloud_aligner scan-context integration spec.
# ---------------------------------------------------------------------------
find_package(PCL REQUIRED COMPONENTS common)
find_package(Eigen3 REQUIRED)

execute_process(
  COMMAND python3 -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

pybind11_add_module(scancontext_py
  scancontext/cpp/binding.cpp
  scancontext/cpp/Scancontext.cpp
)
target_include_directories(scancontext_py PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/scancontext/cpp
)
target_link_libraries(scancontext_py PRIVATE ${PCL_LIBRARIES})
# Land the extension next to the Python package sources so
# `import scancontext_py` works without installing.
set_target_properties(scancontext_py PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scancontext/python
)
