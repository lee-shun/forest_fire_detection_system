/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PositionHelper.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-01-13
 *
 *   @Description:
 *
 *******************************************************************************/

#ifndef INCLUDE_TOOLS_POSITIONHELPER_HPP_
#define INCLUDE_TOOLS_POSITIONHELPER_HPP_

#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <tools/PrintControl/PrintCtrlMacro.h>

#include <cmath>
#include <common/CommonTypes.hpp>
#include <tools/MathLib.hpp>

namespace FFDS {
namespace TOOLS {
class PositionHelper {
 public:
  PositionHelper() {
    gps_location_sub = nh.subscribe<sensor_msgs::NavSatFix>(
        "dji_osdk_ros/gps_position", 10, &PositionHelper::gpsCallback, this);
    ros::Duration(1.0).sleep();
  }
  sensor_msgs::NavSatFix getAverageGPS(const int average_times);

  sensor_msgs::NavSatFix getTargetGPS(const sensor_msgs::NavSatFix& myGPos,
                                      const COMMON::PosAngle<double> angle,
                                      const double distance);

  COMMON::LocalPosition<double> getTargetRelativePos(
      const COMMON::PosAngle<double> angle, const double distance);

 private:
  ros::NodeHandle nh;
  ros::Subscriber gps_location_sub;
  sensor_msgs::NavSatFix gps_position;

  void gpsCallback(const sensor_msgs::NavSatFix::ConstPtr& gps_msg) {
    gps_position = *gps_msg;
  }
};

sensor_msgs::NavSatFix PositionHelper::getAverageGPS(
    const int average_times) {
  sensor_msgs::NavSatFix homeGPos;

  for (int i = 0; (i < average_times) && ros::ok(); i++) {
    ros::spinOnce();
    if (TOOLS::isEquald(0.0, gps_position.latitude) ||
        TOOLS::isEquald(0.0, gps_position.longitude) ||
        TOOLS::isEquald(0.0, gps_position.altitude)) {
      PRINT_WARN("zero in gps_position, waiting for normal gps position!");
      i = 0;
      continue;
    }
    homeGPos.latitude += gps_position.latitude;
    homeGPos.longitude += gps_position.longitude;
    homeGPos.altitude += gps_position.altitude;

    ros::Rate(10).sleep();
  }
  homeGPos.latitude = homeGPos.latitude / average_times;
  homeGPos.longitude = homeGPos.longitude / average_times;
  homeGPos.altitude = homeGPos.altitude / average_times;

  return homeGPos;
}

inline COMMON::LocalPosition<double> PositionHelper::getTargetRelativePos(
    const COMMON::PosAngle<double> angle, const double distance) {
  return COMMON::LocalPosition<double>(
      (distance * cos(angle.beta) * cos(angle.alpha)),
      (distance * cos(angle.beta) * sin(angle.alpha)),
      (distance * sin(angle.beta)));
}

sensor_msgs::NavSatFix PositionHelper::getTargetGPS(
    const sensor_msgs::NavSatFix& myGPos, const COMMON::PosAngle<double> angle,
    const double distance) {
  COMMON::LocalPosition<double> rel_local_pos =
      getTargetRelativePos(angle, distance);

  COMMON::GPSPosition<double> refGPS;
  refGPS.lon = myGPos.longitude;
  refGPS.lat = myGPos.latitude;
  refGPS.alt = myGPos.altitude;

  COMMON::GPSPosition<double> resGPS =
      TOOLS::Meter2LatLongAlt(refGPS, rel_local_pos);

  sensor_msgs::NavSatFix finalGPS;
  finalGPS.longitude = resGPS.lon;
  finalGPS.latitude = refGPS.lat;
  finalGPS.altitude = refGPS.alt;

  return finalGPS;
}

}  // namespace TOOLS
}  // namespace FFDS

#endif  // INCLUDE_TOOLS_POSITIONHELPER_HPP_
