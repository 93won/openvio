# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seungwon Choi. All rights reserved.

cmake_minimum_required(VERSION 3.16)
project(openvio_ros1 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs nav_msgs geometry_msgs tf2_ros)
catkin_package()

# The OpenVIO library from the repository root (its own flags fix the floating-point rounding), linked statically into
# the node like the ROS 2 one (catkin turns BUILD_SHARED_LIBS on)
set(BUILD_SHARED_LIBS OFF)
set(OPENVIO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
add_subdirectory(${OPENVIO_ROOT} openvio EXCLUDE_FROM_ALL)

add_executable(openvio_node src/openvio_node.cc ${OPENVIO_ROOT}/ros/common/online_odometry.cc)
target_include_directories(openvio_node PRIVATE ${OPENVIO_ROOT}/ros/common ${catkin_INCLUDE_DIRS})
# Same instruction set and rounding as the library: the node converts the estimate (e.g. the quaternion it writes)
separate_arguments(OPENVIO_ARCH UNIX_COMMAND "${OPENVIO_ARCH_FLAGS}")
target_compile_options(openvio_node PRIVATE ${OPENVIO_ARCH} -ffp-contract=off)
target_compile_definitions(openvio_node PRIVATE EIGEN_NO_CPUID)
target_link_libraries(openvio_node vio ${catkin_LIBRARIES})
# OpenVIO's bundled spdlog must not replace the symbols of a spdlog that another library loads
set_target_properties(vio openvio_node PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

install(TARGETS openvio_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch rviz DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY ${OPENVIO_ROOT}/config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
