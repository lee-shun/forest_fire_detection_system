/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVLab. All rights reserved.
 *
 *   @Filename: test_WpV2Operator.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-09-19
 *
 *   @Description: Rewrite of the waypointv2 node example with some "clang
 *   warnings..."
 *
 ******************************************************************************/

#ifndef INCLUDE_TEST_TEST_WPV2OPERATOR_H_
#define INCLUDE_TEST_TEST_WPV2OPERATOR_H_

#include <dji_osdk_ros/GenerateWaypointV2Action.h>
#include <dji_osdk_ros/GetDroneType.h>
#include <dji_osdk_ros/InitWaypointV2Setting.h>
#include <dji_osdk_ros/ObtainControlAuthority.h>
#include <dji_osdk_ros/SubscribeWaypointV2Event.h>
#include <dji_osdk_ros/SubscribeWaypointV2State.h>
#include <dji_osdk_ros/WaypointV2MissionEventPush.h>
#include <dji_osdk_ros/WaypointV2MissionStatePush.h>
#include <dji_osdk_ros/common_type.h>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>

#include <modules/WayPointOperator/WpV2Operator.hpp>
#include <tools/MathLib.hpp>
#include <vector>

/**
 * global variable
 * */
dji_osdk_ros::GetDroneType drone_type;
dji_osdk_ros::InitWaypointV2Setting initWaypointV2Setting_;
dji_osdk_ros::UploadWaypointV2Mission uploadWaypointV2Mission_;
dji_osdk_ros::UploadWaypointV2Action uploadWaypointV2Action_;
dji_osdk_ros::DownloadWaypointV2Mission downloadWaypointV2Mission_;
dji_osdk_ros::StartWaypointV2Mission startWaypointV2Mission_;
dji_osdk_ros::StopWaypointV2Mission stopWaypointV2Mission_;
dji_osdk_ros::PauseWaypointV2Mission pauseWaypointV2Mission_;
dji_osdk_ros::ResumeWaypointV2Mission resumeWaypointV2Mission_;
dji_osdk_ros::SetGlobalCruisespeed setGlobalCruisespeed_;
dji_osdk_ros::GetGlobalCruisespeed getGlobalCruisespeed_;
dji_osdk_ros::GenerateWaypointV2Action generateWaypointV2Action_;
dji_osdk_ros::SubscribeWaypointV2Event subscribeWaypointV2Event_;
dji_osdk_ros::SubscribeWaypointV2State subscribeWaypointV2State_;

ros::ServiceClient waypointV2_init_setting_client;
ros::ServiceClient waypointV2_generate_actions_client;
ros::ServiceClient waypointV2_mission_event_push_client;
ros::ServiceClient waypointV2_mission_state_push_client;

ros::Subscriber waypointV2EventSub;
ros::Subscriber waypointV2StateSub;

ros::ServiceClient get_drone_type_client;
sensor_msgs::NavSatFix gps_position_;
dji_osdk_ros::WaypointV2MissionEventPush waypoint_V2_mission_event_push_;
dji_osdk_ros::WaypointV2MissionStatePush waypoint_V2_mission_state_push_;

void gpsPositionSubCallback(
    const sensor_msgs::NavSatFix::ConstPtr &gpsPosition);
void waypointV2MissionStateSubCallback(
    const dji_osdk_ros::WaypointV2MissionStatePush::ConstPtr
        &waypointV2MissionStatePush);
void waypointV2MissionEventSubCallback(
    const dji_osdk_ros::WaypointV2MissionEventPush::ConstPtr
        &waypointV2MissionEventPush);

std::vector<dji_osdk_ros::WaypointV2> generatePolygonWaypoints(
    const ros::NodeHandle &nh, DJI::OSDK::float32_t radius,
    uint16_t polygonNum);
bool initWaypointV2Setting(ros::NodeHandle &nh);
bool generateWaypointV2Actions(ros::NodeHandle &nh, uint16_t actionNum);

bool runWaypointV2Mission(ros::NodeHandle &nh);

#endif  // INCLUDE_TEST_TEST_WPV2OPERATOR_H_
