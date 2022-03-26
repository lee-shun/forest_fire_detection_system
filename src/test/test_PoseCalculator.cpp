/*******************************************************************************
*   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
*
*   @Filename: test_PoseCalculator.cpp
*
*   @Author: Shun Li
*
*   @Email: 2015097272@qq.com
*
*   @Date: 2022-03-18
*
*   @Description: 
*
*******************************************************************************/


#include "modules/PoseCalculator/PoseCalculator.hpp"
int main(int argc, char** argv) {
  ros::init(argc, argv, "test_pose_calculator_node");
  ros::NodeHandle nh;

  FFDS::MODULES::PoseCalculator pose_calculator;

  while (ros::ok()) {
    // pose_calculator.Step();
  }
return 0;
}
