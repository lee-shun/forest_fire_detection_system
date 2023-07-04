/*******************************************************************************
 *   Copyright (C) 2023 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PreciselyWaterDropping.hpp
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

#include <ros/ros.h>

namespace FFDS {
namespace APP {
class PreciselyWaterDropper {
 private:
  int target_x_;
  int target_y_;
  int ctrl_times_;
  int ctrl_threshold_;

 public:
  PreciselyWaterDropper(const int target_x, const int target_y,
                        const int ctrl_times, const int ctrl_threshold);
  void run();
};
}  // namespace APP
}  // namespace FFDS
