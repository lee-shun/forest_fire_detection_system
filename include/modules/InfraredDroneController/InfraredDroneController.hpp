/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: InfraredDroneController.hpp
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

#include <ros/ros.h>
#include "forest_fire_detection_system/SingleFireIR.h"

namespace FFDS {
namespace MODULES {
class InfraredDroneController {
 public:
  InfraredDroneController() {
    singleFirePosIRSub =
        nh.subscribe("forest_fire_detection_system/single_fire_in_ir_image", 10,
                     &InfraredDroneController::singleFireIRCallback, this);
  }

  // return true if ctrl the drone error between the given threshold and
  // max_ctrl_times.
  bool ctrlDroneMoveByTarget(const int target_x, const int target_y,
                             const int max_ctrl_times, const int threshold);

 private:
  ros::NodeHandle nh;
  ros::Subscriber singleFirePosIRSub;
  forest_fire_detection_system::SingleFireIR heatPosPix;

  void singleFireIRCallback(
      const forest_fire_detection_system::SingleFireIR::ConstPtr&
          firePosition) {
    heatPosPix = *firePosition;
  }
};
}  // namespace MODULES
}  // namespace FFDS
