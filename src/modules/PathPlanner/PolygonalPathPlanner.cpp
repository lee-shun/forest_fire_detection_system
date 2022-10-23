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

namespace FFDS {
namespace MODULES {

std::vector<dji_osdk_ros::WaypointV2>& PolygonalPathPlanner::getWpV2Vec() {
  return wp_v2_vec_;
}

void PolygonalPathPlanner::FindStartPos(double s[2]) {
  // take the center as the local ref...NED
  double c[2], h[2], m[2];
  c[0] = center_.latitude;
  c[1] = center_.longitude;
  h[0] = home_.latitude;
  h[1] = home_.longitude;

  TOOLS::LatLong2Meter(c, h, m);

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
    return sqrt(x * x + y * y);
  };

  s =  euler_dis(s1) < euler_dis(s2) ? s1 : s2;
};

}  // namespace MODULES
}  // namespace FFDS
