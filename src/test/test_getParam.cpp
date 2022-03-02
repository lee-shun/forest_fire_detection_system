/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_getParam.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-11-10
 *
 *   @Description:
 *
 *******************************************************************************/

#include <ros/package.h>
#include <ros/ros.h>
#include <tools/PrintControl/PrintCtrlMacro.h>

#include <tools/SystemLib.hpp>
int main(int argc, char** argv) {
  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");
  const std::string config_path = package_path + "/config/ZigzagPathShape.yaml";
  PRINT_INFO("get from %s", config_path.c_str());
  YAML::Node node = YAML::LoadFile(config_path);
  int num = FFDS::TOOLS::getParam(node, "num", 10);
  float len = FFDS::TOOLS::getParam(node, "len", 40.0);
  float wid = FFDS::TOOLS::getParam(node, "wid", 10.0);
  float height = FFDS::TOOLS::getParam(node, "height", 15.0);

  return 0;
}
