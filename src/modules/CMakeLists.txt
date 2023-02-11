#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: CMakeLists.txt
#
#   @Author: Shun Li
#
#   @Date: 2021-10-22
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
#------------------------------------------------------------------------------



file(GLOB_RECURSE MODULE_PATH_LIST *.cpp)
set(${PROJECT_NAME}_module)

add_library(${PROJECT_NAME}_module ${MODULE_PATH_LIST})
add_dependencies(
  ${PROJECT_NAME}_module
  ${PROJECT_NAME}_generate_messages_cpp
)
target_link_libraries(
  ${PROJECT_NAME}_module
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  dji_osdk_ros
  ${catkin_LIBRARIES}
)
