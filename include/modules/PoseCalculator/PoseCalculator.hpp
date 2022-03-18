/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PoseCalculator.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-03-17
 *
 *   @Description:
 *
 *******************************************************************************/

#ifndef INCLUDE_MODULES_POSECALCULATOR_POSECALCULATOR_HPP_
#define INCLUDE_MODULES_POSECALCULATOR_POSECALCULATOR_HPP_

#include <stereo_camera_vo/common/camera.h>
#include <stereo_camera_vo/common/frame.h>
#include <stereo_camera_vo/module/frontend.h>

#include <memory>

#include <sophus/se3.hpp>

namespace FFDS {
namespace MODULES {
class PoseCalculator {
 public:
  PoseCalculator() {
    /**
     * Step: 1 create the camera instances
     * */

    stereo_camera_vo::common::Camera::Ptr camera_left =
        std::make_shared<stereo_camera_vo::common::Camera>(1, 1, 1, 1, 1,
                                                           Sophus::SE3d());
    stereo_camera_vo::common::Camera::Ptr camera_right =
        std::make_shared<stereo_camera_vo::common::Camera>(1, 1, 1, 1, 1,
                                                           Sophus::SE3d());
    /**
     * Step: 2 create the frontend instance
     * */
    frontend_ = std::make_shared<stereo_camera_vo::module::Frontend>(
        camera_left, camera_right, true);
  }

  void Step(stereo_camera_vo::common::Frame::Ptr current_frame) {}

 private:
  stereo_camera_vo::module::Frontend::Ptr frontend_{nullptr};
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_POSECALCULATOR_POSECALCULATOR_HPP_
