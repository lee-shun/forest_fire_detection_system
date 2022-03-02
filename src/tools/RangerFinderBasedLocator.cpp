/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: RangerFinderBasedLocator.cpp
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

#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <tools/PrintControl/PrintCtrlMacro.h>

#include <modules/GimbalCameraOperator/GimbalCameraOperator.hpp>
#include <tools/PositionHelper.hpp>
#include <tools/PrintControl/FileWritter.hpp>
#include <tools/SystemLib.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "rangerfinder_based_locator_node");

  FFDS::TOOLS::PositionHelper posHelper;
  FFDS::MODULES::GimbalCameraOperator gcOperator;
  FFDS::TOOLS::FileWritter locatorWriter("ranger_finder_based_locator.csv", 8);
  locatorWriter.new_open();

  FFDS::COMMON::PosAngle<double> position_angle;
  double ranger_distance;
  PRINT_INFO("Please input the rangerfinder distance: ");
  std::cin >> ranger_distance;

  sensor_msgs::NavSatFix ref_pos = posHelper.getAverageGPS(10);
  geometry_msgs::Vector3Stamped gimbal_angle =
      gcOperator.getAverageGimbalAtt(10);

  locatorWriter.write("time_ms", "ranger_distance", "reference_pos.lon",
                      "reference_pos.lat", "reference_pos.alt",
                      "gimbal_angle_x", "gimbal_angle_y", "gimbal_angle_z");
  locatorWriter.write(FFDS::TOOLS::getSysTime(), ranger_distance,
                      ref_pos.longitude, ref_pos.latitude, ref_pos.altitude,
                      gimbal_angle.vector.x, gimbal_angle.vector.y,
                      gimbal_angle.vector.z);

  /**
   * convert from gimbal angle to position angle
   * */
  /* WARN: the gimbal x is pitch, y is roll, z is yaw, it's left hand
   * WARN: rule??? YOU GOT BE KIDDING ME! */
  position_angle.alpha = gimbal_angle.vector.z;
  position_angle.beta = gimbal_angle.vector.x;

  /* FIXME: this getTargetGPS() function is not tested yet, the results may
   * suffer bugs!
   * */
  PRINT_WARN(
      "this calculate fire posititon function is not tested yet, the results "
      "may suffer bugs! But can record the Gimabl angle, current pos and "
      "ranger_distance!");
  sensor_msgs::NavSatFix fire_pos =
      posHelper.getTargetGPS(ref_pos, position_angle, ranger_distance);

  locatorWriter.write("fire_pos.lon", "fire_pos.lat", "fire_pos.alt");
  locatorWriter.write(fire_pos.longitude, fire_pos.latitude, fire_pos.altitude);
  locatorWriter.close();

  PRINT_DEBUG("Fire pos lon: %.8lf, lat: %.8lf, alt: %.8lf.",
              fire_pos.longitude, fire_pos.latitude, fire_pos.altitude)

  return 0;
}
