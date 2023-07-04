/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: InfraredDroneController.cpp
 *
 *   @Author: ShunLi
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 04/07/2023
 *
 *   @Description:
 *
 *******************************************************************************/

#include "modules/InfraredDroneController/InfraredDroneController.hpp"
#include "modules/BasicController/PIDController.hpp"

namespace FFDS {
namespace MODULES {
bool InfraredDroneController::ctrlDroneMoveByTarget(const int target_x,
                                                    const int target_y,
                                                    const int max_ctrl_times,
                                                    const int threshold) {
  // TODO(lee-shun): here
  return true;
}
}  // namespace MODULES
}  // namespace FFDS
