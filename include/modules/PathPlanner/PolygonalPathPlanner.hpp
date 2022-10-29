/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PolygonalPathPlanner.hpp
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

#ifndef INCLUDE_MODULES_PATHPLANNER_POLYGONALPATHPLANNER_HPP_
#define INCLUDE_MODULES_PATHPLANNER_POLYGONALPATHPLANNER_HPP_

#include <dji_osdk_ros/WaypointV2.h>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>

#include <common/CommonTypes.hpp>
#include <modules/PathPlanner/PathPlannerBase.hpp>
#include <modules/WayPointOperator/WpV2Operator.hpp>
#include <tools/MathLib.hpp>
#include <vector>

namespace FFDS {
namespace MODULES {
class PolygonalPathPlanner {
 public:
  // degree in rads
  PolygonalPathPlanner(sensor_msgs::NavSatFix home,
                       sensor_msgs::NavSatFix center, int num_of_wps,
                       float radius, float height, float velocity)
      : home_(home),
        center_(center),
        num_of_wps_(num_of_wps),
        radius_(radius),
        height_(height),
        velocity_(velocity) {}

  std::vector<dji_osdk_ros::WaypointV2>& getWpV2Vec();
  std::vector<FFDS::COMMON::LocalPosition<double>>& getLocalPosVec();

 private:
  void FindStartPos(double start_loc[2], double home_loc[2]);

  void GenLocalPos(const float height);

  void CalLocalWpFrom(const float rad, double cur[2]);

  void FeedWp2Vec();

  sensor_msgs::NavSatFix home_;
  sensor_msgs::NavSatFix center_;
  int num_of_wps_;
  float radius_;
  float height_;
  float velocity_;

  /* the NavSatFix is float64==double */
  std::vector<COMMON::LocalPosition<double>> local_pos_vec_;
  std::vector<dji_osdk_ros::WaypointV2> wp_v2_vec_;
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_PATHPLANNER_POLYGONALPATHPLANNER_HPP_
