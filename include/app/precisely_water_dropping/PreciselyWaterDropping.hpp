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
#include <utility>

namespace FFDS {
namespace APP {
class PreciselyWaterDropper {
 private:
 public:
  PreciselyWaterDropper(const int ctrl_times, const int target_x,
                        const int targt_y);
  void run();
};
}  // namespace APP
}  // namespace FFDS
