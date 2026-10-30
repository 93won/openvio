# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Seungwon Choi. All rights reserved.

cmake_minimum_required(VERSION 3.16)
project(vio LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Fixed rounding: no fused multiply-add contraction, one portable instruction set, and Eigen's fixed default cache
# sizes for blocking matrix products (the sizes it would query differ between the core types of a hybrid CPU, which
# changed the summation order and the last digits from run to run)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march=x86-64-v3 OPENVIO_HAVE_X86_64_V3)
if(OPENVIO_HAVE_X86_64_V3)
  set(OPENVIO_ARCH_FLAGS "-march=x86-64-v3")
else()
  # The x86-64-v3 instruction set spelled out, for compilers before GCC 11 (e.g. Ubuntu 20.04)
  set(OPENVIO_ARCH_FLAGS "-march=x86-64 -mcx16 -msahf -mpopcnt -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mbmi \
-mbmi2 -mf16c -mfma -mlzcnt -mmovbe -mxsave")
endif()
set(OPENVIO_ARCH_FLAGS "${OPENVIO_ARCH_FLAGS}" CACHE INTERNAL "Instruction set of OpenVIO (also used by the ROS nodes)")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG ${OPENVIO_ARCH_FLAGS} -ffp-contract=off")
add_compile_definitions(EIGEN_NO_CPUID)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PNG REQUIRED)

# Third-party libraries
set(BUILD_EXAMPLES OFF CACHE BOOL "Build Ceres examples" FORCE)
set(BUILD_BENCHMARKS OFF CACHE BOOL "Build Ceres benchmarks" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build Ceres tests" FORCE)
set(PROVIDE_UNINSTALL_TARGET OFF CACHE BOOL "Disable Ceres uninstall target" FORCE)
set(SUITESPARSE OFF CACHE BOOL "Disable SuiteSparse to avoid version conflicts" FORCE)
set(CXSPARSE OFF CACHE BOOL "Disable CXSparse" FORCE)
add_subdirectory(thirdparty/ceres-solver EXCLUDE_FROM_ALL)
add_subdirectory(thirdparty/spdlog EXCLUDE_FROM_ALL)

# Library
add_library(vio
  src/vio/estimators/estimator.cc
  src/vio/estimators/imu_propagation.cc
  src/vio/estimators/initializer.cc
  src/vio/estimators/startup_calibration.cc
  src/vio/estimators/visual_update.cc
  src/vio/feature/feature_tracker.cc
  src/vio/feature/landmark_tracker.cc
  src/vio/geometry/inverse_depth.cc
  src/vio/geometry/triangulation.cc
  src/vio/odometry/visual_inertial_odometry.cc
  src/vio/pipeline/config_loader.cc
  src/vio/pipeline/odometry_pipeline.cc
  src/vio/scene/camera.cc
  src/vio/scene/frame.cc
  src/vio/sensor/camera_model.cc
  src/vio/sensor/equirectangular_camera_model.cc
  src/vio/sensor/fisheye_equidistant_camera_model.cc
  src/vio/sensor/imu_interval.cc
  src/vio/sensor/pinhole_radtan_camera_model.cc
)
target_include_directories(vio PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/ceres-solver/include
  ${CMAKE_CURRENT_BINARY_DIR}/thirdparty/ceres-solver/config
)
target_link_libraries(vio PUBLIC ${OpenCV_LIBS} PNG::PNG ceres spdlog::spdlog_header_only)

# Command-line tool: vio odometry | vio gui
add_executable(vio_main src/vio/apps/vio.cc)
target_link_libraries(vio_main PRIVATE vio)
set_target_properties(vio_main PROPERTIES OUTPUT_NAME vio RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Viewer (ImGui/ImPlot), built when GLFW is available
find_package(glfw3 3.3 QUIET)
if(glfw3_FOUND)
  set(OpenGL_GL_PREFERENCE GLVND)
  find_package(OpenGL REQUIRED)
  add_library(imgui STATIC
    thirdparty/imgui/imgui.cpp
    thirdparty/imgui/imgui_draw.cpp
    thirdparty/imgui/imgui_tables.cpp
    thirdparty/imgui/imgui_widgets.cpp
    thirdparty/imgui/backends/imgui_impl_glfw.cpp
    thirdparty/imgui/backends/imgui_impl_opengl3.cpp
    thirdparty/implot/implot.cpp
    thirdparty/implot/implot_items.cpp
  )
  target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/implot
  )
  target_link_libraries(imgui PUBLIC glfw OpenGL::GL ${CMAKE_DL_LIBS})

  add_library(vio_ui STATIC src/vio/ui/viewer.cc)
  target_link_libraries(vio_ui PUBLIC vio imgui)
  target_link_libraries(vio_main PRIVATE vio_ui)
  target_compile_definitions(vio_main PRIVATE VIO_WITH_GUI)
else()
  message(STATUS "GLFW not found: 'vio gui' is not built")
endif()
