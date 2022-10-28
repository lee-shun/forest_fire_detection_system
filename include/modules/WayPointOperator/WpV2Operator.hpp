/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: WpV2Operator.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-25
 *
 *   @Description:
 *
 ******************************************************************************/

#ifndef INCLUDE_MODULES_WAYPOINTOPERATOR_WPV2OPERATOR_HPP_
#define INCLUDE_MODULES_WAYPOINTOPERATOR_WPV2OPERATOR_HPP_

#include <dji_osdk_ros/DownloadWaypointV2Mission.h>
#include <dji_osdk_ros/GenerateWaypointV2Action.h>
#include <dji_osdk_ros/GetGlobalCruisespeed.h>
#include <dji_osdk_ros/InitWaypointV2Setting.h>
#include <dji_osdk_ros/PauseWaypointV2Mission.h>
#include <dji_osdk_ros/ResumeWaypointV2Mission.h>
#include <dji_osdk_ros/SetGlobalCruisespeed.h>
#include <dji_osdk_ros/StartWaypointV2Mission.h>
#include <dji_osdk_ros/StopWaypointV2Mission.h>
#include <dji_osdk_ros/UploadWaypointV2Action.h>
#include <dji_osdk_ros/UploadWaypointV2Mission.h>
#include <ros/ros.h>
#include <tools/PrintControl/PrintCtrlMacro.h>

#include <dji_mission_type.hpp>
#include <dji_type.hpp>
#include <vector>

namespace FFDS {
namespace MODULES {

class WpV2Operator {
  /**
   * NOTE: when calling the operators, prepare the "content" you want to pass
   * NOTE: first.
   **/

 public:
  WpV2Operator() {
    waypointV2_init_setting_client =
        nh.serviceClient<dji_osdk_ros::InitWaypointV2Setting>(
            "dji_osdk_ros/waypointV2_initSetting");
    waypointV2_generate_actions_client =
        nh.serviceClient<dji_osdk_ros::GenerateWaypointV2Action>(
            "dji_osdk_ros/waypointV2_generateActions");
    waypointV2_upload_mission_client =
        nh.serviceClient<dji_osdk_ros::UploadWaypointV2Mission>(
            "dji_osdk_ros/waypointV2_uploadMission");
    waypointV2_upload_action_client =
        nh.serviceClient<dji_osdk_ros::UploadWaypointV2Action>(
            "dji_osdk_ros/waypointV2_uploadAction");
    waypointV2_download_mission_client =
        nh.serviceClient<dji_osdk_ros::DownloadWaypointV2Mission>(
            "dji_osdk_ros/waypointV2_downloadMission");
    waypointV2_start_mission_client =
        nh.serviceClient<dji_osdk_ros::StartWaypointV2Mission>(
            "dji_osdk_ros/waypointV2_startMission");
    waypointV2_stop_mission_client =
        nh.serviceClient<dji_osdk_ros::StopWaypointV2Mission>(
            "dji_osdk_ros/waypointV2_stopMission");
    waypointV2_pause_mission_client =
        nh.serviceClient<dji_osdk_ros::PauseWaypointV2Mission>(
            "dji_osdk_ros/waypointV2_pauseMission");
    waypointV2_resume_mission_client =
        nh.serviceClient<dji_osdk_ros::ResumeWaypointV2Mission>(
            "dji_osdk_ros/waypointV2_resumeMission");
    waypointV2_set_global_cruisespeed_client =
        nh.serviceClient<dji_osdk_ros::SetGlobalCruisespeed>(
            "dji_osdk_ros/waypointV2_setGlobalCruisespeed");
    waypointV2_get_global_cruisespeed_client =
        nh.serviceClient<dji_osdk_ros::GetGlobalCruisespeed>(
            "dji_osdk_ros/waypointV2_getGlobalCruisespeed");

    ros::Duration(3.0).sleep();
  }

  static void setWaypointV2Defaults(dji_osdk_ros::WaypointV2 *waypointV2Ptr);

  bool initWaypointV2Setting(
      dji_osdk_ros::InitWaypointV2Setting *initWaypointV2SettingPtr);

  bool generateWaypointV2Actions(
      dji_osdk_ros::GenerateWaypointV2Action *generateWaypointV2ActionPtr);

  bool uploadWaypointV2Mission(
      dji_osdk_ros::UploadWaypointV2Mission *uploadWaypointV2MissionPtr);

  bool uploadWaypointV2Action(
      dji_osdk_ros::UploadWaypointV2Action *uploadWaypointV2ActionPtr);

  bool downloadWaypointV2Mission(
      dji_osdk_ros::DownloadWaypointV2Mission *downloadWaypointV2MissionPtr,
      std::vector<dji_osdk_ros::WaypointV2> *missionPtr);

  bool startWaypointV2Mission(
      dji_osdk_ros::StartWaypointV2Mission *startWaypointV2MissionPtr);

  bool stopWaypointV2Mission(
      dji_osdk_ros::StopWaypointV2Mission *stopWaypointV2MissionPtr);

  bool pauseWaypointV2Mission(
      dji_osdk_ros::PauseWaypointV2Mission *pauseWaypointV2MissionPtr);

  bool resumeWaypointV2Mission(
      dji_osdk_ros::ResumeWaypointV2Mission *resumeWaypointV2MissionPtr);

  bool setGlobalCruiseSpeed(
      dji_osdk_ros::SetGlobalCruisespeed *setGlobalCruisespeedPtr);

  DJI::OSDK::float32_t getGlobalCruiseSpeed(
      dji_osdk_ros::GetGlobalCruisespeed *getGlobalCruisespeedPtr);

 private:
  ros::NodeHandle nh;

  ros::ServiceClient waypointV2_init_setting_client;
  ros::ServiceClient waypointV2_generate_actions_client;
  ros::ServiceClient waypointV2_upload_mission_client;
  ros::ServiceClient waypointV2_upload_action_client;
  ros::ServiceClient waypointV2_download_mission_client;
  ros::ServiceClient waypointV2_start_mission_client;
  ros::ServiceClient waypointV2_stop_mission_client;
  ros::ServiceClient waypointV2_pause_mission_client;
  ros::ServiceClient waypointV2_resume_mission_client;
  ros::ServiceClient waypointV2_set_global_cruisespeed_client;
  ros::ServiceClient waypointV2_get_global_cruisespeed_client;
};

}  // namespace MODULES

}  // namespace FFDS

#endif  // INCLUDE_MODULES_WAYPOINTOPERATOR_WPV2OPERATOR_HPP_
