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

file(GLOB_RECURSE SRC_LIST *.cpp)

foreach(EACH_SRC ${SRC_LIST})

  string(REGEX REPLACE ".+/(.+)\\..*" "\\1" EACH_EXE ${EACH_SRC})

  add_executable(${EACH_EXE} ${EACH_SRC})

  target_link_libraries(
    ${EACH_EXE}
    ${catkin_LIBRARIES}
    ${PROJECT_NAME}_module
    # ${PATH_PLAN_THIRD_PARTY_LIBS}
  )

endforeach()
