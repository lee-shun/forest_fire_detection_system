/*******************************************************************************
 *   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PreciselyWaterDropping.cpp
 *
 *   @Author: ShunLi
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 03/07/2023
 *
 *   @Description: this is the water dropping application with the help from
 *   ifrared images.
 *
 *******************************************************************************/

#include "app/precisely_water_dropping/PreciselyWaterDropping.hpp"
#include "modules/GimbalCameraOperator/GimbalCameraOperator.hpp"
#include "modules/InfraredDroneController/InfraredDroneController.hpp"

namespace FFDS {
namespace APP {
PreciselyWaterDropper::PreciselyWaterDropper(const int target_x,
                                             const int target_y,
                                             const int ctrl_times,
                                             const int ctrl_threshold)
    : target_x_(target_x),
      target_y_(target_y),
      ctrl_times_(ctrl_times),
      ctrl_threshold_(ctrl_threshold) {}
void PreciselyWaterDropper::run() {
  // STEP: 0 reset the gimbal
  MODULES::GimbalCameraOperator gimbal_operator;
  if (gimbal_operator.resetGimbal()) {
    PRINT_INFO("Successfuly reset the gimbal!");
  } else {
    PRINT_ERROR("Failed to reset the gimbal, quit!")
    return;
  }
  // STEP: 1 roate the camera to face down.
  // TODO(lee-shun): may be -90 here.
  if (gimbal_operator.rotateByDeg(90, 0, 0, false)) {
    PRINT_INFO("Successfuly rotate the gimbal!");
  } else {
    PRINT_ERROR("Failed to rotate the gimbal, quit!")
    return;
  }

  // STEP: 2 control the UAV to fly  along x and y with the target point on
  // image.
  MODULES::InfraredDroneController drone_controller;
  if (drone_controller.ctrlDroneMoveByTarget(target_x_, target_y_, ctrl_times_,
                                             ctrl_threshold_)) {
    PRINT_INFO("water can be drop now!")
  } else {
    PRINT_WARN("need more control times or a looser threshold!");
  }

  // STEP: 3 confirm to drop water by hand.

  std::string water_dropped;
  std::cin >> water_dropped;

  // STEP: 4 return home!
}
}  // namespace APP
}  // namespace FFDS

int main(int agrc, char* argv[]) {
  // STEP：1 fly around the water and then come close to the water...
  // STEP：2 precisely drop the water.
  FFDS::APP::PreciselyWaterDropper dropper(200, 200, 10, 10);
  dropper.run();
}
