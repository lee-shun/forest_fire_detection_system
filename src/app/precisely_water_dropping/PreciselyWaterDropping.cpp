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

namespace FFDS {
namespace APP {
PreciselyWaterDropper::PreciselyWaterDropper(const int ctrl_times,
                                             const int target_x,
                                             const int targt_y) {}
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

}  // namespace APP
}  // namespace APP

// TODO(lee-shun): modify this as a server, provide the water dropping service.
int main(int agrc, char* argv[]) {
  FFDS::APP::PreciselyWaterDropper dropper(10, 200, 200);
  dropper.run();
}
