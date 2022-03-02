/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: ZigzagPathPlanner.cpp
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

#include <modules/PathPlanner/ZigzagPathPlanner.hpp>

void FFDS::MODULES::ZigzagPathPlanner::setParams(sensor_msgs::NavSatFix home,
                                                 int num, float len, float wid,
                                                 float height) {
  homeGPos = home;
  zigzagNum = num;
  zigzagLen = len;
  zigzagWid = wid;
  zigzagHeight = height;
}

void FFDS::MODULES::ZigzagPathPlanner::calLocalPos() {
  float each_len = zigzagLen / zigzagNum;
  int point_num = 2 * (zigzagNum + 1);
  COMMON::LocalPosition<double> pos;

  bool is_lower_left = true;
  bool is_upper_left = false;
  bool is_lower_right = false;
  bool is_upper_right = false;

  /* add the first point above the home position */
  pos.x = 0.0;
  pos.y = 0.0;
  pos.z = zigzagHeight;
  LocalPosVec.push_back(pos);

  for (int i = 0; i < point_num - 1; ++i) {
    pos.z = zigzagHeight;

    if (is_lower_left) {
      pos.x += 0.0;
      pos.y += zigzagWid;
      LocalPosVec.push_back(pos);

      is_lower_left = false;
      is_upper_left = false;
      is_lower_right = true;
      is_upper_right = false;
    } else if (is_lower_right) {
      pos.x += each_len;
      pos.y += 0.0;
      LocalPosVec.push_back(pos);

      is_lower_left = false;
      is_upper_left = false;
      is_lower_right = false;
      is_upper_right = true;
    } else if (is_upper_right) {
      pos.x += 0.0;
      pos.y += -zigzagWid;
      LocalPosVec.push_back(pos);

      is_lower_left = false;
      is_upper_left = true;
      is_lower_right = false;
      is_upper_right = false;
    } else if (is_upper_left) {
      pos.x += each_len;
      pos.y += 0.0;
      LocalPosVec.push_back(pos);

      is_lower_left = true;
      is_upper_left = false;
      is_lower_right = false;
      is_upper_right = false;
    } else {
      ROS_ERROR_STREAM("the bool is wrong!");
    }
  }
}

void FFDS::MODULES::ZigzagPathPlanner::Earth2HEarth(float homeHeadRad) {
  float rot_x;
  float rot_y;

  for (int i = 0; i < LocalPosVec.size(); ++i) {
    rot_x = LocalPosVec[i].x * sin(homeHeadRad) -
            LocalPosVec[i].y * cos(homeHeadRad);

    rot_y = LocalPosVec[i].x * cos(homeHeadRad) +
            LocalPosVec[i].y * sin(homeHeadRad);

    LocalPosVec[i].x = rot_x;
    LocalPosVec[i].y = rot_y;
    LocalPosVec[i].z = LocalPosVec[i].z;
  }
}

void FFDS::MODULES::ZigzagPathPlanner::feedWp2Vec() {
  dji_osdk_ros::WaypointV2 wpV2;
  MODULES::WpV2Operator::setWaypointV2Defaults(&wpV2);

  double ref[3], result[3];
  ref[0] = homeGPos.latitude;
  ref[1] = homeGPos.longitude;
  ref[2] = homeGPos.altitude;

  for (int i = 0; i < LocalPosVec.size(); ++i) {
    MODULES::WpV2Operator::setWaypointV2Defaults(&wpV2);

    /* NOTE: gps is represented by rad in DJI, but use degree in the
     * NOTE: HotpointMission ......
     * NOTE: use x->latitude(north and south), use y->longitude(west and east)
     * NOTE: so we use loal as NED.
     */
    TOOLS::Meter2LatLongAlt<double>(ref, LocalPosVec[i], result);
    wpV2.latitude = TOOLS::Deg2Rad(result[0]);
    wpV2.longitude = TOOLS::Deg2Rad(result[1]);
    wpV2.relativeHeight = LocalPosVec[i].z;

    wpV2Vec.push_back(wpV2);
  }
}

std::vector<dji_osdk_ros::WaypointV2>&
FFDS::MODULES::ZigzagPathPlanner::getWpV2Vec(bool useInitHeadDirection,
                                             const float homeHeadRad) {
  /* Step: 1 generate the local position*/
  getLocalPosVec(useInitHeadDirection, homeHeadRad);

  /* Step: 2 to global gps position*/
  feedWp2Vec();

  return wpV2Vec;
}

/**
 * Return the local position after rotation if applicable
 * */
std::vector<FFDS::COMMON::LocalPosition<double> >&
FFDS::MODULES::ZigzagPathPlanner::getLocalPosVec(bool useInitHeadDirection,
                                                 const float homeHeadRad) {
  /* Step: 1 generate the local zigzag LocalPosVec */
  calLocalPos();

  /* Step: 2 if HeadEarth to Earth? */
  if (useInitHeadDirection) {
    Earth2HEarth(homeHeadRad);
  }

  return LocalPosVec;
}
