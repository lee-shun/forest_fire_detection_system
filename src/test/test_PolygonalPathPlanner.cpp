/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_PolygonalPathPlanner.cpp
 *
 *   @Author: ShunLi
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 26/10/2022
 *
 *   @Description:
 *
 *******************************************************************************/

#include "modules/PathPlanner/PolygonalPathPlanner.hpp"
#include "tools/GoogleEarthPath.hpp"

int main(int argc, char** argv) {
  // home position
  sensor_msgs::NavSatFix home;
  home.altitude = 25.445088;
  home.latitude = 45.4550769;
  home.longitude = -73.915427;

  // center of the fire
  sensor_msgs::NavSatFix center;
  center.altitude = 25.5387778;
  center.latitude = 45.4552318;
  center.longitude = -73.9149486;

  FFDS::MODULES::PolygonalPathPlanner planner(home, center, 20, 15.0, 15.0, 0.5);
  auto waypointVec = planner.getWpV2Vec();

  FFDS::TOOLS::GoogleEarthPath path("/home/ls/polygonal_path.kml", "polygonal_path");
  double longitude, latitude;

  for (int i = 0; i < waypointVec.size(); ++i) {
    latitude = FFDS::TOOLS::Rad2Deg(waypointVec[i].latitude);
    longitude = FFDS::TOOLS::Rad2Deg(waypointVec[i].longitude);
    path.addPoint(longitude, latitude);
  }

  return 0;
}
