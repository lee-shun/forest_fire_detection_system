/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: ZigzagPathPlanner.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-26
 *
 *   @Description:
 *
 ******************************************************************************/

/**
 * FIXME: Need to redefine the class variables and methods
 * */

#ifndef INCLUDE_MODULES_PATHPLANNER_ZIGZAGPATHPLANNER_HPP_
#define INCLUDE_MODULES_PATHPLANNER_ZIGZAGPATHPLANNER_HPP_

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

class ZigzagPathPlanner : public MODULES::PathPlannerBase {
 public:
  ZigzagPathPlanner() {}
  ZigzagPathPlanner(sensor_msgs::NavSatFix home, int num, float len, float wid,
                    float height)
      : zigzagNum(num),
        zigzagLen(len),
        zigzagWid(wid),
        zigzagHeight(height),
        homeGPos(home) {}

  ~ZigzagPathPlanner() {}

  void setParams(sensor_msgs::NavSatFix home, int num, float len, float wid,
                 float height);

  std::vector<dji_osdk_ros::WaypointV2>& getWpV2Vec(bool useInitHeadDirection,
                                                    const float homeHeadRad);

  std::vector<COMMON::LocalPosition<double>>& getLocalPosVec(
      bool useInitHeadDirection, const float homeHeadRad);

 private:
  int zigzagNum{0};
  float zigzagLen{0.0};
  float zigzagWid{0.0};
  float zigzagHeight{0.0};
  sensor_msgs::NavSatFix homeGPos;

  /* the NavSatFix is float64==double */
  std::vector<COMMON::LocalPosition<double>> LocalPosVec;
  std::vector<dji_osdk_ros::WaypointV2> wpV2Vec;

  /**
   * NOTE: we want the M300 initial heading as the positive direction.
   * NOTE: This coordinates is defined as H(ead)Earth coordinates by Shun.
   * NOTE: There is only one difference between HEarth and Earth, the
   * NOTE: init-heading angle.
   **/

  /* generate the local as the same, treat it in HEarth or Earth local position.
   */
  void calLocalPos();

  void Earth2HEarth(float homeHeadRad);

  void feedWp2Vec();
};

}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_PATHPLANNER_ZIGZAGPATHPLANNER_HPP_
