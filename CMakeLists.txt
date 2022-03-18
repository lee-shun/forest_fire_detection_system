# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: CMakeLists.txt
#
#   @Author: Shun Li
#
#   @Date: 2022-03-18
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------

cmake_minimum_required(VERSION 2.8.3)
project(forest_fire_detection_system)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS "-std=c++11 -g -ggdb ${CMAKE_CXX_FLAGS}")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# ros package
find_package(
  catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  tf2
  tf2_ros
  image_transport
  dji_osdk_ros
  geometry_msgs
  message_generation
  nav_msgs
  nmea_msgs
  std_msgs
  message_filters # for message_filter
)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# yaml
find_library(YAML_CPP_LIBRARIES yaml-cpp)

# cv_bridge
set(cv_bridge_DIR ~/cv_bridge_ws/install/share/cv_bridge/cmake)
find_package(cv_bridge REQUIRED)
include_directories(${cv_bridge_INCLUDE_DIRS})
link_directories(${cv_bridge_LIBRARY_DIRS})

# dji_osdk_ros
link_directories("~/catkin_ws/devel/lib")
add_definitions(-DUSE_OPEN_CV_CONTRIB)
add_definitions(-DUSE_GPU)
add_definitions(-DADVANCED_SENSING)

# stereo_camera_vo
include_directories("3rdparty/stereo_camera_vo/include")
link_directories("3rdparty/stereo_camera_vo/lib")

# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# fmt
find_package(fmt REQUIRED)
include_directories(${fmt_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

set(
  VO_THIRD_PARTY_LIBS
  ${OpenCV_LIBS}
  ${Sophus_LIBRARIES}
  ${Pangolin_LIBRARIES} GL GLU GLEW glut
  g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
  pthread
  ${CSPARSE_LIBRARY}
  fmt::fmt
  ${GTEST_BOTH_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  
  stereo_camera_vo_common
  stereo_camera_vo_module
  stereo_camera_vo_tool
)


add_message_files(
  FILES
  SingleFireFuse.msg
  SingleFireIR.msg
  SingleFireRGB.msg
)

add_service_files(
  FILES
  SegmentSmoke.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  roscpp
  rospy
  roslib
  tf2
  tf2_ros
  std_msgs
  dji_osdk_ros
  geometry_msgs
  nav_msgs
  nmea_msgs
  message_runtime
)

###########
## Build ##
###########

add_subdirectory(src)
