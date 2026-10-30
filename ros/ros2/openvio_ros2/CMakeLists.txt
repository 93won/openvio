# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Seungwon Choi. All rights reserved.

cmake_minimum_required(VERSION 3.16)
project(openvio_ros2 LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# The OpenVIO library from the repository root (its own flags fix the floating-point rounding)
set(OPENVIO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
add_subdirectory(${OPENVIO_ROOT} openvio EXCLUDE_FROM_ALL)

add_executable(openvio_node src/openvio_node.cc ${OPENVIO_ROOT}/ros/common/online_odometry.cc)
target_include_directories(openvio_node PRIVATE ${OPENVIO_ROOT}/ros/common)
# Same instruction set and rounding as the library: the node converts the estimate (e.g. the quaternion it writes)
separate_arguments(OPENVIO_ARCH UNIX_COMMAND "${OPENVIO_ARCH_FLAGS}")
target_compile_options(openvio_node PRIVATE ${OPENVIO_ARCH} -ffp-contract=off)
target_compile_definitions(openvio_node PRIVATE EIGEN_NO_CPUID)
target_link_libraries(openvio_node vio rclcpp::rclcpp tf2_ros::tf2_ros ${sensor_msgs_TARGETS} ${nav_msgs_TARGETS}
  ${geometry_msgs_TARGETS})
# OpenVIO's bundled spdlog must not replace the symbols of the spdlog that ROS logging loads (they differ in version)
set_target_properties(vio openvio_node PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

install(TARGETS openvio_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME})
install(DIRECTORY ${OPENVIO_ROOT}/config DESTINATION share/${PROJECT_NAME})

ament_package()
