/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PolygonalPathPlanner.cpp
 *
 *   @Author: ShunLi
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 23/10/2022
 *
 *   @Description:
 *
 *******************************************************************************/

#include "modules/PathPlanner/PolygonalPathPlanner.hpp"
#include "tools/PrintControl/PrintCtrlMacro.h"

namespace FFDS {
namespace MODULES {

std::vector<dji_osdk_ros::WaypointV2>& PolygonalPathPlanner::getWpV2Vec() {
  GenLocalPos(height_);
  FeedWp2Vec();

  return wp_v2_vec_;
}

// BUG: here double line of the local pos!
std::vector<FFDS::COMMON::LocalPosition<double>>&
PolygonalPathPlanner::getLocalPosVec() {
  GenLocalPos(height_);
  return local_pos_vec_;
}

void PolygonalPathPlanner::FindStartPos(double start_loc[2],
                                        double home_loc[2]) {
  // take the center as the local ref...NED
  double c[2], h[2], m[2];  // now, m is the local pos of home under the center
  c[0] = center_.latitude;
  c[1] = center_.longitude;
  h[0] = home_.latitude;
  h[1] = home_.longitude;

  TOOLS::LatLong2Meter(c, h, m);
  home_loc[0] = m[0];
  home_loc[1] = m[1];

  double s1[2] = {0.0, 0.0}, s2[2] = {0.0, 0.0};

  if (TOOLS::isEquald(m[0], 0.0)) {
    // N == 0
    s1[0] = 0.0;
    s1[1] = radius_;
    s2[0] = 0.0;
    s2[1] = -radius_;
  } else if (TOOLS::isEquald(m[1], 0.0)) {
    // E == 0
    s1[0] = radius_;
    s1[1] = 0.0;
    s2[0] = -radius_;
    s2[1] = 0.0;

  } else {
    double k = m[1] / m[0];  // e/n
    s1[0] = radius_ / sqrt(1 + k * k);
    s1[1] = k * s1[0];
    s2[0] = -radius_ / sqrt(1 + k * k);
    s2[1] = k * s2[0];
  }

  // find the nearest one
  auto euler_dis = [m](double cur[2]) {
    double x = cur[0] - m[0], y = cur[1] - m[1];
    return (x * x + y * y);
  };

  auto tmp = euler_dis(s1) < euler_dis(s2) ? s1 : s2;
  start_loc[0] = tmp[0];
  start_loc[1] = tmp[1];
};

void PolygonalPathPlanner::GenLocalPos(const float height) {
  // clear the vec first.
  local_pos_vec_.clear();

  double start_loc[2], home_loc[2];
  FindStartPos(start_loc, home_loc);
  local_pos_vec_.push_back(
      COMMON::LocalPosition<double>(home_loc[0], home_loc[1], height));
  local_pos_vec_.push_back(
      COMMON::LocalPosition<double>(start_loc[0], start_loc[1], height));

  float each_rad = 2 * M_PI / num_of_wps_;
  float cur_rad = 0.0;
  while (cur_rad + each_rad <= 2 * M_PI) {
    cur_rad += each_rad;

    double angle = cur_rad + std::atan2(start_loc[1], start_loc[0]);
    double cur_pos[2];
    CalLocalWpFrom(angle, cur_pos);
    local_pos_vec_.push_back(
        COMMON::LocalPosition<double>(cur_pos[0], cur_pos[1], height));
  }
}

// counter clockwise, as the define the angle
void PolygonalPathPlanner::CalLocalWpFrom(const float rad, double cur[2]) {
  cur[0] = radius_ * std::cos(rad);
  cur[1] = radius_ * std::sin(rad);
}

void PolygonalPathPlanner::FeedWp2Vec() {
  // clear the vec first
  wp_v2_vec_.clear();

  dji_osdk_ros::WaypointV2 wpV2;
  MODULES::WpV2Operator::setWaypointV2Defaults(&wpV2);

  double ref[3], result[3];
  ref[0] = center_.latitude;
  ref[1] = center_.longitude;
  ref[2] = center_.altitude;

  for (int i = 0; i < local_pos_vec_.size(); ++i) {
    MODULES::WpV2Operator::setWaypointV2Defaults(&wpV2);

    // STEP: 1 cal the pos
    TOOLS::Meter2LatLongAlt<double>(ref, local_pos_vec_[i], result);
    wpV2.latitude = TOOLS::Deg2Rad(result[0]);
    wpV2.longitude = TOOLS::Deg2Rad(result[1]);
    wpV2.relativeHeight = local_pos_vec_[i].z;

    // STEP: 2 set the direction
    wpV2.headingMode =
        dji_osdk_ros::WaypointV2::DJIWaypointV2HeadingWaypointCustom;

    // NOTE: calculation!
    float x = local_pos_vec_[i].x, y = local_pos_vec_[i].y;
    float abs_ang = std::abs(std::atan2(y, x));
    if (x > 0 && y > 0) {
      // 1
      wpV2.heading = TOOLS::Rad2Deg(abs_ang - M_PI);
    } else if (x < 0 && y > 0) {
      // 2
      wpV2.heading = TOOLS::Rad2Deg(abs_ang - M_PI);
    } else if (x < 0 && y < 0) {
      // 3
      wpV2.heading = TOOLS::Rad2Deg(M_PI - abs_ang);
    } else {
      // 4
      wpV2.heading = TOOLS::Rad2Deg(M_PI - abs_ang);
    }

    // STEP: 3 set the velocity
    wpV2.autoFlightSpeed = velocity_;

    wp_v2_vec_.push_back(wpV2);
  }
}

}  // namespace MODULES
}  // namespace FFDS
