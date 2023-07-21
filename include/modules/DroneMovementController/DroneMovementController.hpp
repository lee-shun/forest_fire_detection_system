/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: DroneMovementController.hpp
 *
 *   @Author: ShunLi
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 04/07/2023
 *
 *   @Description:
 *
 *******************************************************************************/

// dji
#include <dji_osdk_ros/FlightTaskControl.h>
#include <dji_osdk_ros/JoystickAction.h>
#include <dji_osdk_ros/ObtainControlAuthority.h>
#include <dji_osdk_ros/SetJoystickMode.h>
#include <dji_osdk_ros/common_type.h>

// ros
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <ros/ros.h>
#include <ros/time.h>

// ffds
#include "forest_fire_detection_system/SingleFireIR.h"
#include "tools/PrintControl/PrintCtrlMacro.h"

namespace FFDS {
namespace MODULES {
class DroneMovementController {
 public:
  DroneMovementController() {
    single_fire_pos_IR_sub =
        nh.subscribe("forest_fire_detection_system/single_fire_in_ir_image", 10,
                     &DroneMovementController::singleFireIRCallback, this);

    task_control_client = nh.serviceClient<dji_osdk_ros::FlightTaskControl>(
        "/flight_task_control");

    set_joystick_mode_client =
        nh.serviceClient<dji_osdk_ros::SetJoystickMode>("set_joystick_mode");

    obtain_ctrl_authority_client =
        nh.serviceClient<dji_osdk_ros::ObtainControlAuthority>(
            "obtain_release_control_authority");

    obtainCtrlAuthority.request.enable_obtain = true;
    obtain_ctrl_authority_client.call(obtainCtrlAuthority);
    PRINT_INFO("Obtain the control authority!");
  }

  // return true if ctrl the drone error between the given threshold and
  // max_ctrl_times.
  bool ctrlDroneMoveByImgTarget(const int target_x, const int target_y,
                                const int max_ctrl_times, const int threshold);

  bool ctrlDroneMOveByOffset(dji_osdk_ros::FlightTaskControl &task,
                             const dji_osdk_ros::JoystickCommand &offsetDesired,
                             float posThresholdInM, float yawThresholdInDeg);

  bool ctrlDroneReturnHome() {
    /* 4. Go home */
    ROS_INFO_STREAM("going home now");
    control_task.request.task =
        dji_osdk_ros::FlightTaskControl::Request::TASK_GOHOME;
    task_control_client.call(control_task);
    if (control_task.response.result == true) {
      ROS_INFO_STREAM("GO home successful");
      return true;
    } else {
      ROS_INFO_STREAM("Go home failed.");
      return false;
    }
  }

  bool ctrlDroneLand() {
    /* 5. Landing */
    control_task.request.task =
        dji_osdk_ros::FlightTaskControl::Request::TASK_LAND;
    ROS_INFO_STREAM(
        "Landing request sending ... need your confirmation on the remoter!");
    task_control_client.call(control_task);
    if (control_task.response.result == true) {
      ROS_INFO_STREAM("Land task successful");
      return true;
    } else {
      ROS_INFO_STREAM("Land task failed.");
      return false;
    }
  }

 private:
  ros::NodeHandle nh;
  ros::Subscriber single_fire_pos_IR_sub;
  forest_fire_detection_system::SingleFireIR heatPosPix;

  ros::ServiceClient task_control_client;
  ros::ServiceClient set_joystick_mode_client;
  ros::ServiceClient obtain_ctrl_authority_client;

  geometry_msgs::QuaternionStamped vehical_att;
  dji_osdk_ros::FlightTaskControl control_task;
  dji_osdk_ros::ObtainControlAuthority obtainCtrlAuthority;

  void singleFireIRCallback(
      const forest_fire_detection_system::SingleFireIR::ConstPtr
          &firePosition) {
    heatPosPix = *firePosition;
  }
};
}  // namespace MODULES
}  // namespace FFDS
