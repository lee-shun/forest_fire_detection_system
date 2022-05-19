/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GrabDataDepthEstimation.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-05-19
 *
 *   @Description:
 *
 *******************************************************************************/

#include "app/fire_points_depth_estimation/GrabDataDepthEstimation.hpp"
#include "modules/H20TIMUPoseGrabber/H20TIMUPoseGrabber.hpp"
#include "tools/PrintControl/FileWritter.hpp"
#include "tools/PrintControl/PrintCtrlMacro.h"
#include "tools/SystemLib.hpp"

#include <thread>

#include <opencv2/imgcodecs.hpp>

void FFDS::APP::GrabDataDepthEstimation::Grab() {
  // STEP: set local reference position
  ros::ServiceClient set_local_pos_ref_client_;
  set_local_pos_ref_client_ = nh_.serviceClient<dji_osdk_ros::SetLocalPosRef>(
      "/set_local_pos_reference");
  dji_osdk_ros::SetLocalPosRef set_local_pos_reference;
  set_local_pos_ref_client_.call(set_local_pos_reference);
  if (set_local_pos_reference.response.result) {
    PRINT_INFO("Set local position reference successfully!");
  } else {
    PRINT_ERROR("Set local position reference failed!");
    return;
  }

  // STEP: New directorys
  std::string home = std::getenv("HOME");
  std::string save_path = home + "/m300_grabbed_data";
  FFDS::TOOLS::shellRm(save_path);

  FFDS::TOOLS::shellMkdir(save_path);
  FFDS::TOOLS::shellMkdir(save_path + "/ir");
  FFDS::TOOLS::shellMkdir(save_path + "/rgb");

  // STEP: New files
  FFDS::TOOLS::FileWritter gps_writter(save_path + "/gps.csv", 8);
  FFDS::TOOLS::FileWritter att_writter(save_path + "/att.csv", 8);
  FFDS::TOOLS::FileWritter local_pose_writter(save_path + "/local_pose.csv", 8);
  FFDS::TOOLS::FileWritter time_writter(save_path + "/time_stamp.csv", 8);

  gps_writter.new_open();
  gps_writter.write("index", "lon", "lat", "alt");

  att_writter.new_open();
  att_writter.write("index", "w", "x", "y", "z");

  local_pose_writter.new_open();
  local_pose_writter.write("index", "x", "y", "z");

  time_writter.new_open();
  time_writter.write("index", "sec", "nsec");

  FFDS::MODULES::H20TIMUPoseGrabber grabber;

  int index = 0;
  while (ros::ok()) {
    grabber.UpdateOnce();
    ros::Time time = ros::Time::now();
    time_writter.write(index, time.sec, time.nsec);

    cv::Mat ir_img = grabber.GetIRImageOnce();
    cv::Mat rgb_img = grabber.GetRGBImageOnce();
    cv::imwrite(save_path + "/ir/" + std::to_string(index) + ".png", ir_img);
    cv::imwrite(save_path + "/rgb/" + std::to_string(index) + ".png", rgb_img);

    sensor_msgs::NavSatFix gps = grabber.GetGPSPoseOnce();
    gps_writter.write(index, gps.longitude, gps.latitude, gps.altitude);

    geometry_msgs::PointStamped local = grabber.GetLocalPosOnce();
    local_pose_writter.write(index, local.point.x, local.point.y,
                             local.point.z);

    geometry_msgs::QuaternionStamped att = grabber.GetAttOnce();
    att_writter.write(index, att.quaternion.w, att.quaternion.x,
                      att.quaternion.y, att.quaternion.z);

    ros::Rate(10).sleep();
    ++index;
  }
}

bool FFDS::APP::GrabDataDepthEstimation::MoveByPosOffset(
    dji_osdk_ros::FlightTaskControl &task,
    const dji_osdk_ros::JoystickCommand &offsetDesired, float posThresholdInM,
    float yawThresholdInDeg) {
  task.request.task =
      dji_osdk_ros::FlightTaskControl::Request::TASK_POSITION_AND_YAW_CONTROL;
  task.request.joystickCommand.x = offsetDesired.x;
  task.request.joystickCommand.y = offsetDesired.y;
  task.request.joystickCommand.z = offsetDesired.z;
  task.request.joystickCommand.yaw = offsetDesired.yaw;
  task.request.posThresholdInM = posThresholdInM;
  task.request.yawThresholdInDeg = yawThresholdInDeg;

  task_control_client.call(task);
  return task.response.result;
}

std::vector<dji_osdk_ros::JoystickCommand>
FFDS::APP::GrabDataDepthEstimation::GenerateOffsetCommands() {
  dji_osdk_ros::JoystickCommand command;
  std::vector<dji_osdk_ros::JoystickCommand> ctrl_vec;

  command.x = 0.0;
  command.y = 10.0;
  command.z = 0.0;
  command.yaw = 0;

  ctrl_vec.push_back(command);

  return ctrl_vec;
}

void FFDS::APP::GrabDataDepthEstimation::run(float desired_height) {
  auto command_vec = GenerateOffsetCommands();
  ROS_INFO_STREAM("Command generating finish, are you ready to take off? y/n");

  char inputChar;
  std::cin >> inputChar;

  if (inputChar == 'n') {
    ROS_INFO_STREAM("exist!");
    return;
  } else {
    /* 0. Obtain the control authority */
    ROS_INFO_STREAM("Obtain the control authority ...");
    obtainCtrlAuthority.request.enable_obtain = true;
    obtain_ctrl_authority_client.call(obtainCtrlAuthority);

    /* 1. Take off */
    ROS_INFO_STREAM("Takeoff request sending ...");
    control_task.request.task =
        dji_osdk_ros::FlightTaskControl::Request::TASK_TAKEOFF;
    task_control_client.call(control_task);

    if (control_task.response.result == false) {
      ROS_ERROR_STREAM("Takeoff task failed!");
    } else {
      ROS_INFO_STREAM("Takeoff task successful!");
      ros::Duration(2.0).sleep();

      /* 2. Move to a higher attitude */
      ROS_INFO_STREAM("Moving to a higher attitude! desired_height offset: "
                      << desired_height << " m!");
      MoveByPosOffset(control_task, {0.0, 0.0, desired_height, 0.0}, 0.8, 1);

      /* 3. Move following the offset */
      ROS_INFO_STREAM("Move by position offset request sending ...");
      for (int i = 0; ros::ok() && (i < command_vec.size()); ++i) {
        std::thread t(std::bind(&GrabDataDepthEstimation::Grab, this));
        ros::Duration(2.0).sleep();
        ROS_INFO_STREAM("Moving to the point: " << i << "!");
        MoveByPosOffset(control_task, command_vec[i], 0.8, 1);
        t.join();
      }

      /* 4. Go home */
      ROS_INFO_STREAM("going home now");
      control_task.request.task =
          dji_osdk_ros::FlightTaskControl::Request::TASK_GOHOME;
      task_control_client.call(control_task);
      if (control_task.response.result == true) {
        ROS_INFO_STREAM("GO home successful");
      } else {
        ROS_INFO_STREAM("Go home failed.");
      }

      /* 5. Landing */
      control_task.request.task =
          dji_osdk_ros::FlightTaskControl::Request::TASK_LAND;
      ROS_INFO_STREAM(
          "Landing request sending ... need your confirmation on the remoter!");
      task_control_client.call(control_task);
      if (control_task.response.result == true) {
        ROS_INFO_STREAM("Land task successful");
      } else {
        ROS_INFO_STREAM("Land task failed.");
      }
    }
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "grab_data_depth_estimation_node");
  ros::NodeHandle nh;

  return 0;
}
