/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVLab. All rights reserved.
 *
 *   @Filename: test_simple_command.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-09-19
 *
 *   @Description: use the joystick srv, simple command.
 *
 ******************************************************************************/

#ifndef INCLUDE_TEST_TEST_SIMPLE_COMMAND_HPP_
#define INCLUDE_TEST_TEST_SIMPLE_COMMAND_HPP_

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

// c++
#include <iostream>
#include <vector>

class TestSimpleCommand {
 private:
  ros::NodeHandle nh;
  ros::Time begin_time;

  ros::Subscriber vehicle_att_subscriber;
  ros::ServiceClient task_control_client;
  ros::ServiceClient set_joystick_mode_client;
  ros::ServiceClient obtain_ctrl_authority_client;

  geometry_msgs::QuaternionStamped vehical_att;
  dji_osdk_ros::FlightTaskControl control_task;
  dji_osdk_ros::ObtainControlAuthority obtainCtrlAuthority;

  /**
   * the callback functions
   * */
  void vehical_att_cb(const geometry_msgs::QuaternionStamped::ConstPtr &msg);

  /**
   * functions
   * */
  void print_vehical_att(const geometry_msgs::QuaternionStamped &att);

 public:
  TestSimpleCommand();
  ~TestSimpleCommand();

  int run(float desired_height, float zigzag_len, float zigzag_wid,
          float zigzag_num);

  std::vector<dji_osdk_ros::JoystickCommand> generate_zigzag_path(float len,
                                                                  float wid,
                                                                  float num);

  bool moveByPosOffset(dji_osdk_ros::FlightTaskControl &task,
                       const dji_osdk_ros::JoystickCommand &offsetDesired,
                       float posThresholdInM, float yawThresholdInDeg);
  void print_control_command(
      const std::vector<dji_osdk_ros::JoystickCommand> &ctrl_command_vec);
};

#endif  // INCLUDE_TEST_TEST_SIMPLE_COMMAND_HPP_
