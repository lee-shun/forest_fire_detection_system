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

namespace FFDS {
namespace APP {

void GrabInfoReconstructionManager::initWpSetting(
    dji_osdk_ros::InitWaypointV2Setting* initWaypointV2SettingPtr) {
  // should be changeable about the numbers and the centers
  MODULES::PolygonalPathPlanner planner(home_, center_, 20, 15, 15);
  auto wp_v2_vec = planner.getWpV2Vec();
  auto local_pos_vec = planner.getLocalPosVec();

  TOOLS::FileWritter GPSplanWriter("m300_ref_GPS_path.csv", 10);
  TOOLS::FileWritter LocalplanWriter("m300_ref_Local_path.csv", 10);
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

  initWaypointV2SettingPtr->request.waypointV2InitSettings.maxFlightSpeed = 1;
  initWaypointV2SettingPtr->request.waypointV2InitSettings.autoFlightSpeed = 0.5;

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

void GrabInfoReconstructionManager::Run() {
  /* Step: 0 reset the camera and gimbal */
  FFDS::MODULES::GimbalCameraOperator gcOperator;
  if (gcOperator.resetCameraZoom() && gcOperator.resetGimbal()) {
    PRINT_INFO("reset camera and gimbal successfully!")
  } else {
    PRINT_WARN("reset camera and gimbal failed!")
  }

  /* Step: 1 init the wp setting, create the basic waypointV2 vector... */
  FFDS::MODULES::WpV2Operator wpV2Operator;
  dji_osdk_ros::InitWaypointV2Setting initWaypointV2Setting_;
  initWpSetting(&initWaypointV2Setting_);
  if (!wpV2Operator.initWaypointV2Setting(&initWaypointV2Setting_)) {
    PRINT_ERROR("Quit!");
    return;
  }
  ros::Duration(1.0).sleep();

  /* Step: 2 upload the wp mission */

  /* Step: 3 init the wp action */

  /* Step: 4 upload the wp action */

  /* Step: 5 start mission */
}

}  // namespace APP
}  // namespace FFDS

int main(int argc, char** argv) { return 0; }
