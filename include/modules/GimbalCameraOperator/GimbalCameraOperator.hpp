/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GimbalCameraOperator.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-31
 *
 *   @Description:
 *
 ******************************************************************************/

#ifndef INCLUDE_MODULES_GIMBALCAMERAOPERATOR_GIMBALCAMERAOPERATOR_HPP_
#define INCLUDE_MODULES_GIMBALCAMERAOPERATOR_GIMBALCAMERAOPERATOR_HPP_

#include <Eigen/Core>

#include <dji_osdk_ros/CameraFocusPoint.h>
#include <dji_osdk_ros/CameraSetZoomPara.h>
#include <dji_osdk_ros/CameraTapZoomPoint.h>
#include <dji_osdk_ros/GimbalAction.h>
#include <dji_osdk_ros/common_type.h>
#include <forest_fire_detection_system/SingleFireIR.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <tools/PrintControl/PrintCtrlMacro.h>

#include <modules/BasicController/PIDController.hpp>
#include <tools/MathLib.hpp>
#include <tools/PrintControl/FileWritter.hpp>
#include <tools/SystemLib.hpp>

namespace FFDS {

namespace MODULES {

class GimbalCameraOperator {
 public:
  GimbalCameraOperator() {
    singleFirePosIRSub =
        nh.subscribe("forest_fire_detection_system/single_fire_in_ir_image", 10,
                     &GimbalCameraOperator::singleFireIRCallback, this);

    gimbalAttSub = nh.subscribe("dji_osdk_ros/gimbal_angle", 10,
                                &GimbalCameraOperator::gimbalAttCallback, this);

    gimbalCtrlClient =
        nh.serviceClient<dji_osdk_ros::GimbalAction>("gimbal_task_control");

    cameraSetZoomParaClient = nh.serviceClient<dji_osdk_ros::CameraSetZoomPara>(
        "camera_task_set_zoom_para");

    cameraSetFocusPointClient =
        nh.serviceClient<dji_osdk_ros::CameraFocusPoint>(
            "camera_task_set_focus_point");

    cameraSetTapZoomPointClient =
        nh.serviceClient<dji_osdk_ros::CameraTapZoomPoint>(
            "camera_task_tap_zoom_point");
    ros::Duration(3.0).sleep();
    PRINT_INFO("initialize GimbalCameraOperator done!");
  }

  /* NOTE: The following functions should work under the zoom mode and aligned
   * NOTE: mode! */

  bool rotateByDeg(const float pitch, const float roll, const float yaw, bool is_inc);
  bool ctrlRotateGimbal(const float setPosXPix, const float setPosYPix,
                        const int times, const float tolErrPix);
  bool ctrlRotateGimbal(const int times, const float tolErrPix);
  bool resetGimbal();

  bool setCameraZoom(const float factor);
  bool resetCameraZoom();

  /* NOTE: The set Focus Point only works in the zoom mode! Not the aligned
   * NOTE: mode ... */
  bool setCameraFocusePoint(const float x, const float y);
  bool resetCameraFocusePoint();

  bool setTapZoomPoint(const float multiplier, const float x, const float y);
  bool resetTapZoomPoint();

  geometry_msgs::Vector3Stamped getAverageGimbalAtt(int average_times);

 private:
  ros::NodeHandle nh;

  ros::Subscriber singleFirePosIRSub;
  ros::Subscriber gimbalAttSub;

  ros::ServiceClient gimbalCtrlClient;
  ros::ServiceClient cameraSetZoomParaClient;
  ros::ServiceClient cameraSetFocusPointClient;
  ros::ServiceClient cameraSetTapZoomPointClient;

  dji_osdk_ros::GimbalAction gimbalAction;
  dji_osdk_ros::CameraSetZoomPara cameraSetZoomPara;
  dji_osdk_ros::CameraFocusPoint cameraFocusPoint;
  dji_osdk_ros::CameraTapZoomPoint cameraTapZoomPoint;

  geometry_msgs::Vector3Stamped gimbalAtt;
  forest_fire_detection_system::SingleFireIR heatPosPix;

  void singleFireIRCallback(
      const forest_fire_detection_system::SingleFireIR::ConstPtr& firePosition);

  void gimbalAttCallback(const geometry_msgs::Vector3Stamped::ConstPtr& att);

  void setGimbalActionDefault();

  Eigen::Vector3f camera2NED(const Eigen::Vector3f& d_attInCamera);

  PIDController pidYaw{0.015, 0.0, 0.0, false, false};
  PIDController pidPitch{0.015, 0.0, 0.0, false, false};
};

}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_GIMBALCAMERAOPERATOR_GIMBALCAMERAOPERATOR_HPP_
