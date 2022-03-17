/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: SingleFirePointTaskManager.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-22
 *
 *   @Description:
 *
 ******************************************************************************/

#ifndef INCLUDE_APP_SINGLE_FIRE_POINT_TASK_SINGLEFIREPOINTTASKMANAGER_HPP_
#define INCLUDE_APP_SINGLE_FIRE_POINT_TASK_SINGLEFIREPOINTTASKMANAGER_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <dji_osdk_ros/FlightTaskControl.h>
#include <dji_osdk_ros/ObtainControlAuthority.h>
#include <dji_osdk_ros/SubscribeWaypointV2Event.h>
#include <dji_osdk_ros/SubscribeWaypointV2State.h>
#include <dji_osdk_ros/WaypointV2MissionEventPush.h>
#include <dji_osdk_ros/WaypointV2MissionStatePush.h>
#include <forest_fire_detection_system/SegmentSmoke.h>
#include <forest_fire_detection_system/SingleFireIR.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <tools/PrintControl/PrintCtrlMacro.h>
#include <tools/PositionHelper.hpp>

#include <modules/GimbalCameraOperator/GimbalCameraOperator.hpp>
#include <modules/PathPlanner/ZigzagPathPlanner.hpp>
#include <modules/WayPointOperator/WpV2Operator.hpp>
#include <tools/SystemLib.hpp>
#include <tools/PrintControl/FileWritter.hpp>

namespace FFDS {

namespace APP {

class SingleFirePointTaskManager {
 private:
  ros::NodeHandle nh;

  ros::Subscriber gpsPositionSub;
  ros::Subscriber attitudeSub;
  ros::Subscriber waypointV2EventSub;
  ros::Subscriber waypointV2StateSub;
  ros::Subscriber singleFirePosIRSub;

  ros::ServiceClient task_control_client;
  ros::ServiceClient obtain_ctrl_authority_client;
  ros::ServiceClient waypointV2_mission_state_push_client;
  ros::ServiceClient waypointV2_mission_event_push_client;
  ros::ServiceClient segment_smoke_client;

  /**
   * ros msg
   * */

  sensor_msgs::NavSatFix gps_position_;
  geometry_msgs::QuaternionStamped attitude_data_;
  dji_osdk_ros::WaypointV2MissionEventPush waypoint_V2_mission_event_push_;
  dji_osdk_ros::WaypointV2MissionStatePush waypoint_V2_mission_state_push_;
  forest_fire_detection_system::SingleFireIR signleFirePos;

  /**
   * ros srv
   * */

  dji_osdk_ros::FlightTaskControl controlTask_;
  dji_osdk_ros::ObtainControlAuthority obtainCtrlAuthority;
  dji_osdk_ros::SubscribeWaypointV2Event subscribeWaypointV2Event_;
  dji_osdk_ros::SubscribeWaypointV2State subscribeWaypointV2State_;

  /**
   * member functions
   * */

  void readPathParams();
  sensor_msgs::NavSatFix getHomeGPosAverage(int times);
  Eigen::Vector3d getInitAttAverage(int times);
  void initMission(
      dji_osdk_ros::InitWaypointV2Setting *initWaypointV2SettingPtr);

  /**
   * callback functions
   * */

  void gpsPositionSubCallback(
      const sensor_msgs::NavSatFix::ConstPtr &gpsPosition);

  void attitudeSubCallback(
      const geometry_msgs::QuaternionStampedConstPtr &attitudeData);

  void waypointV2MissionEventSubCallback(
      const dji_osdk_ros::WaypointV2MissionEventPush::ConstPtr
          &waypointV2MissionEventPush);

  void waypointV2MissionStateSubCallback(
      const dji_osdk_ros::WaypointV2MissionStatePush::ConstPtr
          &waypointV2MissionStatePush);

  void singleFireIRCallback(
      const forest_fire_detection_system::SingleFireIR::ConstPtr &sfPos);

 public:
  SingleFirePointTaskManager();
  ~SingleFirePointTaskManager();

  void goHomeLand();

  void run();
};

}  // namespace APP
}  // namespace FFDS

#endif  // INCLUDE_APP_SINGLE_FIRE_POINT_TASK_SINGLEFIREPOINTTASKMANAGER_HPP_
