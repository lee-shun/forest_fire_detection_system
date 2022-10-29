/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GrabInfoForReconstruction.cpp
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

#include <app/single_fire_point_reconstruction/GrabInfoForReconstruction.hpp>
#include <modules/PathPlanner/PolygonalPathPlanner.hpp>
#include <tools/PositionHelper.hpp>
#include <tools/SystemLib.hpp>
#include <opencv2/opencv.hpp>

namespace FFDS {
namespace APP {

GrabInfoReconstructionManager::GrabInfoReconstructionManager() {
  // STEP: 0 find home position
  FFDS::TOOLS::PositionHelper posHelper;
  home_ = posHelper.getAverageGPS(10);

  // STEP: 1 read the parameters
  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");
  const std::string config_path = package_path + "/config/reconstruction.yaml";
  PRINT_INFO("Load parameters from:%s", config_path.c_str());
  YAML::Node node = YAML::LoadFile(config_path);

  center_.longitude = TOOLS::getParam(node, "cen_long", home_.longitude);
  center_.latitude = TOOLS::getParam(node, "cen_lat", home_.latitude);
  center_.altitude = TOOLS::getParam(node, "cen_alt", home_.altitude);
  radius_ = TOOLS::getParam(node, "radius", 15.0);
  num_of_wps_ = TOOLS::getParam(node, "num_of_wps", 20);
  height_ = TOOLS::getParam(node, "height", 15.0);
  velocity_ = TOOLS::getParam(node, "velocity", 0.5);
  grab_rate_ = TOOLS::getParam(node, "grab_rate", 10.0);

  // STEP: 3 generate the save path
  root_path_ = std::getenv("HOME");
  save_path_ = root_path_ + "/m300_grabbed_data_";
  FFDS::TOOLS::shellRm(save_path_);
  // STEP: New directorys
  FFDS::TOOLS::shellMkdir(save_path_);
  FFDS::TOOLS::shellMkdir(save_path_ + "/ir");
  FFDS::TOOLS::shellMkdir(save_path_ + "/rgb");
}

void GrabInfoReconstructionManager::initWpV2Setting(
    dji_osdk_ros::InitWaypointV2Setting* initWaypointV2SettingPtr) {
  // should be changeable about the numbers and the centers
  MODULES::PolygonalPathPlanner planner(home_, center_, num_of_wps_, radius_,
                                        height_, velocity_);
  auto wp_v2_vec = planner.getWpV2Vec();
  auto local_pos_vec = planner.getLocalPosVec();

  TOOLS::FileWritter GPSplanWriter(save_path_ + "/m300_ref_GPS_path.csv", 10);
  TOOLS::FileWritter LocalplanWriter(save_path_ + "/m300_ref_local_path.csv",
                                     10);
  GPSplanWriter.new_open();
  LocalplanWriter.new_open();
  GPSplanWriter.write("long", "lat", "rel_height");
  LocalplanWriter.write("x", "y", "z");

  for (int i = 0; i < local_pos_vec.size(); ++i) {
    GPSplanWriter.write(wp_v2_vec[i].longitude, wp_v2_vec[i].latitude,
                        wp_v2_vec[i].relativeHeight);
    LocalplanWriter.write(local_pos_vec[i].x, local_pos_vec[i].y,
                          local_pos_vec[i].z);
  }
  GPSplanWriter.close();
  LocalplanWriter.close();

  /**
   * init WpV2 mission
   * */
  initWaypointV2SettingPtr->request.actionNum = wp_v2_vec.size();
  initWaypointV2SettingPtr->request.waypointV2InitSettings.repeatTimes = 1;

  initWaypointV2SettingPtr->request.waypointV2InitSettings.finishedAction =
      initWaypointV2SettingPtr->request.waypointV2InitSettings
          .DJIWaypointV2MissionFinishedGoHome;

  initWaypointV2SettingPtr->request.waypointV2InitSettings.maxFlightSpeed = 2;
  initWaypointV2SettingPtr->request.waypointV2InitSettings.autoFlightSpeed =
      velocity_;

  initWaypointV2SettingPtr->request.waypointV2InitSettings
      .exitMissionOnRCSignalLost = 1;

  initWaypointV2SettingPtr->request.waypointV2InitSettings
      .gotoFirstWaypointMode =
      initWaypointV2SettingPtr->request.waypointV2InitSettings
          .DJIWaypointV2MissionGotoFirstWaypointModePointToPoint;

  initWaypointV2SettingPtr->request.waypointV2InitSettings.mission = wp_v2_vec;

  initWaypointV2SettingPtr->request.waypointV2InitSettings.missTotalLen =
      initWaypointV2SettingPtr->request.waypointV2InitSettings.mission.size();
}

void GrabInfoReconstructionManager::generateWpV2Actions(
    dji_osdk_ros::GenerateWaypointV2Action* generateWaypointV2ActionPtr,
    int actionNum) {
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
    generateWaypointV2ActionPtr->request.actions.push_back(actionVector);
  }
}

void GrabInfoReconstructionManager::Grab() {
  // STEP: New files
  FFDS::TOOLS::FileWritter gps_writter(save_path_ + "/gps.csv", 9);
  FFDS::TOOLS::FileWritter att_writter(save_path_ + "/att.csv", 9);
  FFDS::TOOLS::FileWritter gimbal_angle_writter(
      save_path_ + "/gimbal_angle.csv", 9);
  FFDS::TOOLS::FileWritter local_pose_writter(save_path_ + "/local_pose.csv",
                                              9);
  FFDS::TOOLS::FileWritter time_writter(save_path_ + "/time_stamp.csv", 9);

  gps_writter.new_open();
  gps_writter.write("index", "lon", "lat", "alt");

  att_writter.new_open();
  att_writter.write("index", "w", "x", "y", "z");

  gimbal_angle_writter.new_open();
  gimbal_angle_writter.write("index", "pitch", "roll", "yaw");

  local_pose_writter.new_open();
  local_pose_writter.write("index", "x", "y", "z");

  time_writter.new_open();
  time_writter.write("index", "sec", "nsec");

  FFDS::MODULES::H20TIMUPoseGrabber grabber;

  int index = 0;
  while (ros::ok()) {
    if (FFDS::MODULES::H20TIMUPoseGrabber::MessageFilterStatus::EMPTY ==
        grabber.UpdateOnce())
      continue;

    ros::Time time = ros::Time::now();
    time_writter.write(index, time.sec, time.nsec);

    cv::Mat ir_img = grabber.GetIRImageOnce();
    cv::Mat rgb_img = grabber.GetRGBImageOnce();
    cv::imwrite(save_path_ + "/ir/" + std::to_string(index) + ".png", ir_img);
    cv::imwrite(save_path_ + "/rgb/" + std::to_string(index) + ".png", rgb_img);

    sensor_msgs::NavSatFix gps = grabber.GetGPSPoseOnce();
    gps_writter.write(index, gps.longitude, gps.latitude, gps.altitude);

    geometry_msgs::PointStamped local = grabber.GetLocalPosOnce();
    local_pose_writter.write(index, local.point.x, local.point.y,
                             local.point.z);

    geometry_msgs::QuaternionStamped att = grabber.GetAttOnce();
    att_writter.write(index, att.quaternion.w, att.quaternion.x,
                      att.quaternion.y, att.quaternion.z);

    auto ga = grabber.GetGimbalOnce();
    gimbal_angle_writter.write(index, ga.vector.x, ga.vector.y, ga.vector.z);

    ros::Rate(grab_rate_).sleep();
    ++index;
  }
}

void GrabInfoReconstructionManager::Run() {
  //  STEP: 0 init
  // set local reference position
  ros::ServiceClient set_local_pos_ref_client_;
  set_local_pos_ref_client_ = nh_.serviceClient<dji_osdk_ros::SetLocalPosRef>(
      "/set_local_pos_reference");
  dji_osdk_ros::SetLocalPosRef set_local_pos_reference;
  set_local_pos_ref_client_.call(set_local_pos_reference);
  if (set_local_pos_reference.response.result) {
    PRINT_INFO("set local position reference successfully!");
  } else {
    PRINT_ERROR("set local position reference failed!");
    return;
  }
  // TODO: set gimbal angle then
  // reset the gimbal and camera
  FFDS::MODULES::GimbalCameraOperator gcOperator;
  if (gcOperator.resetCameraZoom() && gcOperator.resetGimbal()) {
    PRINT_INFO("reset camera and gimbal successfully!")
  } else {
    PRINT_WARN("reset camera and gimbal failed!")
  }

  /* STEP: 1 init the wp setting, create the basic waypointV2 vector... */
  FFDS::MODULES::WpV2Operator wpV2Operator;
  dji_osdk_ros::InitWaypointV2Setting initWaypointV2Setting_;
  initWpV2Setting(&initWaypointV2Setting_);
  if (!wpV2Operator.initWaypointV2Setting(&initWaypointV2Setting_)) {
    PRINT_ERROR("init wp mission failed!");
    return;
  }
  ros::Duration(1.0).sleep();

  /* STEP: 2 upload the wp mission */
  dji_osdk_ros::UploadWaypointV2Mission uploadWaypointV2Mission_;
  if (!wpV2Operator.uploadWaypointV2Mission(&uploadWaypointV2Mission_)) {
    PRINT_ERROR("upload wp mission failed!");
    return;
  }
  ros::Duration(1.0).sleep();

  /* STEP: 3 init the wp action */
  dji_osdk_ros::GenerateWaypointV2Action generateWaypointV2Action_;
  generateWpV2Actions(&generateWaypointV2Action_,
                      initWaypointV2Setting_.request.actionNum);
  if (!wpV2Operator.generateWaypointV2Actions(&generateWaypointV2Action_)) {
    PRINT_ERROR("generate wp action failed!");
    return;
  }
  ros::Duration(1.0).sleep();

  /* STEP: 4 upload the wp action */
  dji_osdk_ros::UploadWaypointV2Action uploadWaypointV2Action_;
  if (!wpV2Operator.uploadWaypointV2Action(&uploadWaypointV2Action_)) {
    PRINT_ERROR("upload wp actions failed!");
    return;
  }
  ros::Duration(1.0).sleep();

  /* STEP: 5 start mission */
  PRINT_INFO(
      "wp_V2 mission & actions init finish, are you ready to start? y/n");
  char inputConfirm;
  std::cin >> inputConfirm;
  if (inputConfirm == 'n') {
    PRINT_WARN("exist!");
    return;
  }
  dji_osdk_ros::StartWaypointV2Mission startWaypointV2Mission_;
  if (!wpV2Operator.startWaypointV2Mission(&startWaypointV2Mission_)) {
    PRINT_ERROR("start wp v2 mission failed!");
    return;
  }
  ros::Duration(1.0).sleep();

  /* NOTE: the mission call will immediately finish, won't stop here */
  Grab();
}

}  // namespace APP
}  // namespace FFDS

int main(int argc, char** argv) {
  ros::init(argc, argv, "grad_info_reconstruction_manager_node");
  FFDS::APP::GrabInfoReconstructionManager manager;
  manager.Run();
  return 0;
}
