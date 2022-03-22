/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GimbalCameraOperator.cpp
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

#include <modules/GimbalCameraOperator/GimbalCameraOperator.hpp>

void FFDS::MODULES::GimbalCameraOperator::gimbalAttCallback(
    const geometry_msgs::Vector3Stamped::ConstPtr& att) {
  gimbalAtt = *att;
}

void FFDS::MODULES::GimbalCameraOperator::singleFireIRCallback(
    const forest_fire_detection_system::SingleFireIR::ConstPtr& firePosition) {
  heatPosPix = *firePosition;
}

void FFDS::MODULES::GimbalCameraOperator::setGimbalActionDefault() {
  gimbalAction.request.payload_index =
      static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
  gimbalAction.request.is_reset = false;
  gimbalAction.request.pitch = 0.0;
  gimbalAction.request.roll = 0.0f;
  gimbalAction.request.yaw = 0.0;
  gimbalAction.request.rotationMode = 0;
  gimbalAction.request.time = 1.0;
}

Eigen::Vector3f FFDS::MODULES::GimbalCameraOperator::camera2NED(
    const Eigen::Vector3f& d_attInCamera) {
  float phi = TOOLS::Deg2Rad(gimbalAtt.vector.x);   /* roll angle */
  float theta = TOOLS::Deg2Rad(gimbalAtt.vector.y); /* pitch angle */

  float convert[9] = {
      1, sin(phi) * tan(theta), cos(phi) * tan(theta), 0, cos(phi), -sin(phi),
      0, sin(phi) / cos(theta), cos(phi) / cos(theta)};

  Eigen::Matrix3f eularMatrix(convert);

  return eularMatrix * d_attInCamera;
}

bool FFDS::MODULES::GimbalCameraOperator::ctrlRotateGimbal(
    const int times, const float tolErrPix) {
  PRINT_INFO("use the thermal camera image center as the target!");

  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");
  const std::string config_path = package_path + "/config/H20T_Camera.yaml";
  PRINT_INFO("get camera params from %s", config_path.c_str());
  YAML::Node node = YAML::LoadFile(config_path);

  float IR_img_width = FFDS::TOOLS::getParam(node, "pure_IR_width", 960.0);
  float IR_img_height = FFDS::TOOLS::getParam(node, "pure_IR_height", 770.0);

  return ctrlRotateGimbal(IR_img_width / 2, IR_img_height / 2, times,
                          tolErrPix);
}
/**
 * @param[in]  x and y set position on the IR image, the controlling time and
 * finally control stop error.
 * @param[out]
 * @return
 * @ref
 * @see
 * @note control the gimbal rotate by the a PID controller, no need to use the
 * focal length, control several times according to the "times" parameter
 */
bool FFDS::MODULES::GimbalCameraOperator::ctrlRotateGimbal(
    const float setPosXPix, const float setPosYPix, const int times,
    const float tolErrPix) {
  PRINT_INFO("Start controlling the gimbal using controller!");

  int ctrl_times = 0;
  pidYaw.reset();
  pidPitch.reset();

  FFDS::TOOLS::FileWritter gimbalWriter("gimbal_control_info.csv", 4);
  gimbalWriter.new_open();
  gimbalWriter.write("ctrl_times", "current_pos_x", "current_pos_y", "x_error",
                     "y_error", "pitch_control", "yaw_control");

  while (ros::ok()) {
    ros::spinOnce();

    if (heatPosPix.target_type != heatPosPix.IS_HEAT) {
      pidYaw.reset();
      pidPitch.reset();
      ctrl_times = 0;
      PRINT_WARN("not stable potential fire, control restart!")
      ros::Duration(1.0).sleep();
      continue;
    } else {
      if (ctrl_times > times) {
        PRINT_WARN("control gimbal times out after %d controlling!",
                   ctrl_times);
        return false;
      }

      PRINT_INFO("current control times: %d, tolerance: %d", ctrl_times, times);

      float errX = setPosXPix - heatPosPix.img_x;
      float errY = setPosYPix - heatPosPix.img_y;
      PRINT_DEBUG("err Yaw:%f pixel", errX);
      PRINT_DEBUG("err Pitch:%f pixel", errY);

      if (fabs(errX) <= fabs(tolErrPix) && fabs(errY) <= fabs(tolErrPix)) {
        PRINT_INFO(
            "controling gimbal finish after %d times trying with x-error: "
            "%f pixel, y-error: %f pixel!",
            ctrl_times, errX, errY);
        gimbalWriter.write(ctrl_times, heatPosPix.img_x, heatPosPix.img_y, errX,
                           errY, 0, 0);

        return true;
      }

      /* +x error -> - inc yaw */
      /* +y error -> + inc pitch */
      pidYaw.ctrl(-errX);
      pidPitch.ctrl(errY);

      /*NOTE: treat these attCam as degree */
      float d_pitchCam = pidPitch.getOutput();
      float d_yawCam = pidYaw.getOutput();

      PRINT_DEBUG("Pitch increment in Cam frame:%f deg ", d_pitchCam);
      PRINT_DEBUG("Yaw increment in Cam frame:%f deg", d_yawCam);

      /* WARN: the gimbal x is pitch, y is roll, z is yaw, it's left hand
       * WARN: rule??? YOU GOT BE KIDDING ME! */
      Eigen::Vector3f d_attCam(d_pitchCam, 0.0f, d_yawCam);

      setGimbalActionDefault();
      gimbalAction.request.is_reset = false;
      gimbalAction.request.pitch = d_attCam(0);
      gimbalAction.request.roll = d_attCam(1);
      gimbalAction.request.yaw = d_attCam(2);

      /* 0 for incremental mode, 1 for absolute mode */
      gimbalAction.request.rotationMode = 0;
      gimbalAction.request.time = 0.5;
      gimbalCtrlClient.call(gimbalAction);

      gimbalWriter.write(ctrl_times, heatPosPix.img_x, heatPosPix.img_y, errX,
                         errY, d_yawCam, d_pitchCam);

      ctrl_times += 1;

      ros::Duration(1.0).sleep();
    }
  }

  /* shutdown by keyboard */
  PRINT_WARN("stop gimbal control by keyboard...");
  return false;
}

bool FFDS::MODULES::GimbalCameraOperator::resetGimbal() {
  setGimbalActionDefault();

  gimbalAction.request.is_reset = true;
  gimbalCtrlClient.call(gimbalAction);
  return gimbalAction.response.result;
}

bool FFDS::MODULES::GimbalCameraOperator::setCameraZoom(const float factor) {
  PRINT_INFO("setting camera zoom to %f", factor);
  cameraSetZoomPara.request.payload_index =
      static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
  cameraSetZoomPara.request.factor = factor;
  cameraSetZoomParaClient.call(cameraSetZoomPara);
  return cameraSetZoomPara.response.result;
}

bool FFDS::MODULES::GimbalCameraOperator::resetCameraZoom() {
  PRINT_INFO("reset the camera zoom!")
  return setCameraZoom(2.0);
}

bool FFDS::MODULES::GimbalCameraOperator::setCameraFocusePoint(const float x,
                                                               const float y) {
  cameraFocusPoint.request.payload_index =
      static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
  cameraFocusPoint.request.x = x;
  cameraFocusPoint.request.y = y;
  cameraSetFocusPointClient.call(cameraFocusPoint);
  return cameraFocusPoint.response.result;
}

bool FFDS::MODULES::GimbalCameraOperator::resetCameraFocusePoint() {
  return setCameraFocusePoint(0.5, 0.5);
}

bool FFDS::MODULES::GimbalCameraOperator::setTapZoomPoint(
    const float multiplier, const float x, const float y) {
  cameraTapZoomPoint.request.payload_index =
      static_cast<uint8_t>(dji_osdk_ros::PayloadIndex::PAYLOAD_INDEX_0);
  cameraTapZoomPoint.request.multiplier = multiplier;
  cameraTapZoomPoint.request.x = x;
  cameraTapZoomPoint.request.y = y;
  cameraSetTapZoomPointClient.call(cameraTapZoomPoint);
  return cameraFocusPoint.response.result;
}

bool FFDS::MODULES::GimbalCameraOperator::resetTapZoomPoint() {
  return setTapZoomPoint(2.0, 0.5, 0.5);
}

geometry_msgs::Vector3Stamped
FFDS::MODULES::GimbalCameraOperator::getAverageGimbalAtt(int average_times) {
  geometry_msgs::Vector3Stamped gAtt;

  for (int i = 0; (i < average_times) && ros::ok(); i++) {
    ros::spinOnce();
    gAtt.vector.x += gimbalAtt.vector.x;
    gAtt.vector.y += gimbalAtt.vector.y;
    gAtt.vector.z += gimbalAtt.vector.z;

    ros::Rate(10).sleep();
  }
  gAtt.vector.x = gAtt.vector.x / average_times;
  gAtt.vector.y = gAtt.vector.y / average_times;
  gAtt.vector.z = gAtt.vector.z / average_times;

  return gAtt;
}
