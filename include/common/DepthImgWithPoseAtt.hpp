/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: DepthImgWithPoseAtt.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-02-15
 *
 *   @Description:
 *
 *******************************************************************************/

#ifndef INCLUDE_COMMON_DEPTHIMGWITHPOSEATT_HPP_
#define INCLUDE_COMMON_DEPTHIMGWITHPOSEATT_HPP_

#include <sensor_msgs/PointCloud2.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/core.hpp>

namespace FFDS {
namespace COMMON {
struct DepthImgWithPoseAtt {
  DepthImgWithPoseAtt() {}
  sensor_msgs::PointCloud2 pt_cloud;
  Eigen::Quaterniond rotation;
  Eigen::Vector3d translation;
  Eigen::Vector3d gps;
};
}  // namespace COMMON
}  // namespace FFDS

#endif  // INCLUDE_COMMON_DEPTHIMGWITHPOSEATT_HPP_
