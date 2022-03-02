/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_WpV2Operator.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-25
 *
 *   @Description: rewirte the dji_osdk_ros/sample/waypointV2_node.cpp
 *
 ******************************************************************************/

#include <test/test_WpV2Operator.h>

void gpsPositionSubCallback(
    const sensor_msgs::NavSatFix::ConstPtr &gpsPosition) {
  gps_position_ = *gpsPosition;
}

void waypointV2MissionEventSubCallback(
    const dji_osdk_ros::WaypointV2MissionEventPush::ConstPtr
        &waypointV2MissionEventPush) {
  waypoint_V2_mission_event_push_ = *waypointV2MissionEventPush;

  ROS_INFO("waypoint_V2_mission_event_push_.event ID :0x%x\n",
           waypoint_V2_mission_event_push_.event);

  if (waypoint_V2_mission_event_push_.event == 0x01) {
    ROS_INFO("interruptReason:0x%x\n",
             waypoint_V2_mission_event_push_.interruptReason);
  }
  if (waypoint_V2_mission_event_push_.event == 0x02) {
    ROS_INFO("recoverProcess:0x%x\n",
             waypoint_V2_mission_event_push_.recoverProcess);
  }
  if (waypoint_V2_mission_event_push_.event == 0x03) {
    ROS_INFO("finishReason:0x%x\n",
             waypoint_V2_mission_event_push_.finishReason);
  }

  if (waypoint_V2_mission_event_push_.event == 0x10) {
    ROS_INFO("current waypointIndex:%d\n",
             waypoint_V2_mission_event_push_.waypointIndex);
  }

  if (waypoint_V2_mission_event_push_.event == 0x11) {
    ROS_INFO("currentMissionExecNum:%d\n",
             waypoint_V2_mission_event_push_.currentMissionExecNum);
  }
}

void waypointV2MissionStateSubCallback(
    const dji_osdk_ros::WaypointV2MissionStatePush::ConstPtr
        &waypointV2MissionStatePush) {
  waypoint_V2_mission_state_push_ = *waypointV2MissionStatePush;

  ROS_INFO("waypointV2MissionStateSubCallback");
  ROS_INFO("missionStatePushAck->commonDataVersion:%d\n",
           waypoint_V2_mission_state_push_.commonDataVersion);
  ROS_INFO("missionStatePushAck->commonDataLen:%d\n",
           waypoint_V2_mission_state_push_.commonDataLen);
  ROS_INFO("missionStatePushAck->data.state:0x%x\n",
           waypoint_V2_mission_state_push_.state);
  ROS_INFO("missionStatePushAck->data.curWaypointIndex:%d\n",
           waypoint_V2_mission_state_push_.curWaypointIndex);
  ROS_INFO("missionStatePushAck->data.velocity:%d\n",
           waypoint_V2_mission_state_push_.velocity);
}

bool generateWaypointV2Actions(ros::NodeHandle &nh, uint16_t actionNum) {
  waypointV2_generate_actions_client =
      nh.serviceClient<dji_osdk_ros::GenerateWaypointV2Action>(
          "dji_osdk_ros/waypointV2_generateActions");
  dji_osdk_ros::WaypointV2Action actionVector;
  for (uint16_t i = 0; i < actionNum; i++) {
    actionVector.actionId = i;
    actionVector.waypointV2ActionTriggerType = dji_osdk_ros::WaypointV2Action::
        DJIWaypointV2ActionTriggerTypeSampleReachPoint;
    actionVector.waypointV2SampleReachPointTrigger.waypointIndex = i;
    actionVector.waypointV2SampleReachPointTrigger.terminateNum = 0;
    actionVector.waypointV2ACtionActuatorType =
        dji_osdk_ros::WaypointV2Action::DJIWaypointV2ActionActuatorTypeCamera;
    actionVector.waypointV2CameraActuator.actuatorIndex = 0;
    actionVector.waypointV2CameraActuator
        .DJIWaypointV2ActionActuatorCameraOperationType =
        dji_osdk_ros::WaypointV2CameraActuator::
            DJIWaypointV2ActionActuatorCameraOperationTypeTakePhoto;
    generateWaypointV2Action_.request.actions.push_back(actionVector);
  }

  waypointV2_generate_actions_client.call(generateWaypointV2Action_);

  return generateWaypointV2Action_.response.result;
}

std::vector<dji_osdk_ros::WaypointV2> generatePolygonWaypoints(
    ros::NodeHandle &nh, DJI::OSDK::float32_t radius, uint16_t polygonNum) {
  FFDS::MODULES::WpV2Operator wpv2operator;
  // Let's create a vector to store our waypoints in.
  std::vector<dji_osdk_ros::WaypointV2> waypointList;
  dji_osdk_ros::WaypointV2 startPoint;
  dji_osdk_ros::WaypointV2 waypointV2;

  wpv2operator.setWaypointV2Defaults(&startPoint);
  startPoint.latitude = gps_position_.latitude * M_PI / 180.0;
  startPoint.longitude = gps_position_.longitude * M_PI / 180.0;
  startPoint.relativeHeight = 15;
  waypointList.push_back(startPoint);

  // Iterative algorithm
  for (int i = 0; i < polygonNum; i++) {
    wpv2operator.setWaypointV2Defaults(&waypointV2);
    DJI::OSDK::float32_t angle = i * 2 * M_PI / polygonNum;
    DJI::OSDK::float32_t X = radius * cos(angle);
    DJI::OSDK::float32_t Y = radius * sin(angle);
    waypointV2.latitude = Y / FFDS::TOOLS::EARTH_R + startPoint.latitude;
    waypointV2.longitude =
        X / (FFDS::TOOLS::EARTH_R * cos(startPoint.latitude)) +
        startPoint.longitude;
    waypointV2.relativeHeight = startPoint.relativeHeight;
    waypointList.push_back(waypointV2);
  }
  waypointList.push_back(startPoint);

  return waypointList;
}

bool initWaypointV2Setting(ros::NodeHandle &nh) {
  waypointV2_init_setting_client =
      nh.serviceClient<dji_osdk_ros::InitWaypointV2Setting>(
          "dji_osdk_ros/waypointV2_initSetting");
  initWaypointV2Setting_.request.polygonNum = 6;
  initWaypointV2Setting_.request.radius = 6;
  initWaypointV2Setting_.request.actionNum = 5;

  /*! Generate actions*/
  generateWaypointV2Actions(nh, initWaypointV2Setting_.request.actionNum);
  initWaypointV2Setting_.request.waypointV2InitSettings.repeatTimes = 1;
  initWaypointV2Setting_.request.waypointV2InitSettings.finishedAction =
      initWaypointV2Setting_.request.waypointV2InitSettings
          .DJIWaypointV2MissionFinishedGoHome;
  initWaypointV2Setting_.request.waypointV2InitSettings.maxFlightSpeed = 10;
  initWaypointV2Setting_.request.waypointV2InitSettings.autoFlightSpeed = 2;
  initWaypointV2Setting_.request.waypointV2InitSettings
      .exitMissionOnRCSignalLost = 1;
  initWaypointV2Setting_.request.waypointV2InitSettings.gotoFirstWaypointMode =
      initWaypointV2Setting_.request.waypointV2InitSettings
          .DJIWaypointV2MissionGotoFirstWaypointModePointToPoint;
  initWaypointV2Setting_.request.waypointV2InitSettings.mission =
      generatePolygonWaypoints(nh, initWaypointV2Setting_.request.radius,
                               initWaypointV2Setting_.request.polygonNum);
  initWaypointV2Setting_.request.waypointV2InitSettings.missTotalLen =
      initWaypointV2Setting_.request.waypointV2InitSettings.mission.size();

  waypointV2_init_setting_client.call(initWaypointV2Setting_);
  if (initWaypointV2Setting_.response.result) {
    ROS_INFO("Init mission setting successfully!\n");
  } else {
    ROS_ERROR("Init mission setting failed!\n");
  }

  return initWaypointV2Setting_.response.result;
}

bool runWaypointV2Mission(ros::NodeHandle &nh) {
  int timeout = 1;
  bool result = false;

  FFDS::MODULES::WpV2Operator wpv2operator;

  get_drone_type_client =
      nh.serviceClient<dji_osdk_ros::GetDroneType>("get_drone_type");

  waypointV2_mission_state_push_client =
      nh.serviceClient<dji_osdk_ros::SubscribeWaypointV2Event>(
          "dji_osdk_ros/waypointV2_subscribeMissionState");

  waypointV2_mission_event_push_client =
      nh.serviceClient<dji_osdk_ros::SubscribeWaypointV2State>(
          "dji_osdk_ros/waypointV2_subscribeMissionEvent");

  waypointV2EventSub = nh.subscribe("dji_osdk_ros/waypointV2_mission_event", 10,
                                    &waypointV2MissionEventSubCallback);

  waypointV2StateSub = nh.subscribe("dji_osdk_ros/waypointV2_mission_state", 10,
                                    &waypointV2MissionStateSubCallback);

  subscribeWaypointV2Event_.request.enable_sub = true;
  subscribeWaypointV2State_.request.enable_sub = true;

  get_drone_type_client.call(drone_type);
  if (drone_type.response.drone_type !=
      static_cast<uint8_t>(dji_osdk_ros::Dronetype::M300)) {
    ROS_ERROR("This sample only supports M300!\n");
    return false;
  }

  // start publish the mission information
  waypointV2_mission_state_push_client.call(subscribeWaypointV2State_);
  waypointV2_mission_event_push_client.call(subscribeWaypointV2Event_);

  /*! init mission */

  result = initWaypointV2Setting(nh);
  if (!result) {
    return false;
  }
  sleep(timeout);

  /*! upload mission */
  result = wpv2operator.uploadWaypointV2Mission(&uploadWaypointV2Mission_);
  if (!result) {
    return false;
  }
  sleep(timeout);

  /*! download mission */
  std::vector<dji_osdk_ros::WaypointV2> mission;
  result = wpv2operator.downloadWaypointV2Mission(&downloadWaypointV2Mission_,
                                                  &mission);
  if (!result) {
    return false;
  }
  sleep(timeout);

  /*! upload  actions */
  result = wpv2operator.uploadWaypointV2Action(&uploadWaypointV2Action_);
  if (!result) {
    return false;
  }
  sleep(timeout);

  /*! start mission */
  result = wpv2operator.startWaypointV2Mission(&startWaypointV2Mission_);
  if (!result) {
    return false;
  }
  sleep(20);

  /*! set global cruise speed */
  setGlobalCruisespeed_.request.global_cruisespeed = 1.5;
  result = wpv2operator.setGlobalCruiseSpeed(&setGlobalCruisespeed_);
  if (!result) {
    return false;
  }
  sleep(timeout);

  /*! get global cruise speed */
  DJI::OSDK::float32_t globalCruiseSpeed = 0;
  globalCruiseSpeed = wpv2operator.getGlobalCruiseSpeed(&getGlobalCruisespeed_);
  sleep(timeout);

  /*! pause the mission*/
  result = wpv2operator.pauseWaypointV2Mission(&pauseWaypointV2Mission_);
  if (!result) {
    return false;
  }
  sleep(5);

  /*! resume the mission*/
  result = wpv2operator.resumeWaypointV2Mission(&resumeWaypointV2Mission_);
  if (!result) {
    return false;
  }
  sleep(20);

  return true;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "test_WpV2Operator_node");
  ros::NodeHandle nh;

  ros::Subscriber gpsPositionSub =
      nh.subscribe("dji_osdk_ros/gps_position", 10, &gpsPositionSubCallback);

  auto obtain_ctrl_authority_client =
      nh.serviceClient<dji_osdk_ros::ObtainControlAuthority>(
          "obtain_release_control_authority");

  dji_osdk_ros::ObtainControlAuthority obtainCtrlAuthority;
  obtainCtrlAuthority.request.enable_obtain = true;
  obtain_ctrl_authority_client.call(obtainCtrlAuthority);

  ros::Duration(1).sleep();
  ros::AsyncSpinner spinner(1);
  spinner.start();
  runWaypointV2Mission(nh);

  ros::waitForShutdown();
}
