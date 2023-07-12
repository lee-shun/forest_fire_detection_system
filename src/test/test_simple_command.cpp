/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVLab. All rights reserved.
 *
 *   @Filename: test_simple_command.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-09-19
 *
 *   @Description:
 *
 ******************************************************************************/

#include <test/test_simple_command.hpp>
#include "tools/PrintControl/PrintCtrlMacro.h"

TestSimpleCommand::TestSimpleCommand() {
  vehicle_att_subscriber = nh.subscribe<geometry_msgs::QuaternionStamped>(
      "dji_osdk_ros/attitude", 10, &TestSimpleCommand::vehical_att_cb, this);

  task_control_client =
      nh.serviceClient<dji_osdk_ros::FlightTaskControl>("/flight_task_control");

  set_joystick_mode_client =
      nh.serviceClient<dji_osdk_ros::SetJoystickMode>("set_joystick_mode");

  obtain_ctrl_authority_client =
      nh.serviceClient<dji_osdk_ros::ObtainControlAuthority>(
          "obtain_release_control_authority");
}

TestSimpleCommand::~TestSimpleCommand() {}

void TestSimpleCommand::vehical_att_cb(
    const geometry_msgs::QuaternionStamped::ConstPtr &msg) {
  vehical_att = *msg;
}

void TestSimpleCommand::print_vehical_att(
    const geometry_msgs::QuaternionStamped &att) {
  ROS_INFO("the quaternion is:\n");
  ROS_INFO("w:%.2f\n", att.quaternion.w);
  ROS_INFO("x:%.2f\n", att.quaternion.x);
  ROS_INFO("y:%.2f\n", att.quaternion.y);
  ROS_INFO("z:%.2f\n", att.quaternion.z);
}

/**
 * @param[in]
 * @param[out]
 * @return
 * @ref
 * @see
 * @note the function to generate the zegzag rectangle line command using the
 * the dji FlightTaskControl msg.
 */
std::vector<dji_osdk_ros::JoystickCommand>
TestSimpleCommand::generate_zigzag_path(float len, float wid, float num) {
  float each_len = len / num;
  int point_num = 2 * (num + 1);
  dji_osdk_ros::JoystickCommand command;
  std::vector<dji_osdk_ros::JoystickCommand> ctrl_vec;

  bool is_lower_left = true;
  bool is_upper_left = false;
  bool is_lower_right = false;
  bool is_upper_right = false;

  command.x = 0.0;
  command.y = 0.0;
  command.z = 0.0;
  command.yaw = 0.0;

  for (int i = 0; i < point_num - 1; ++i) {
    if (is_lower_left) {
      command.x = 0.0;
      command.y = wid;
      ctrl_vec.push_back(command);

      is_lower_left = false;
      is_upper_left = false;
      is_lower_right = true;
      is_upper_right = false;
    } else if (is_lower_right) {
      command.x = each_len;
      command.y = 0.0;
      ctrl_vec.push_back(command);

      is_lower_left = false;
      is_upper_left = false;
      is_lower_right = false;
      is_upper_right = true;
    } else if (is_upper_right) {
      command.x = 0.0;
      command.y = -wid;
      ctrl_vec.push_back(command);

      is_lower_left = false;
      is_upper_left = true;
      is_lower_right = false;
      is_upper_right = false;
    } else if (is_upper_left) {
      command.x = each_len;
      command.y = 0.0;
      ctrl_vec.push_back(command);

      is_lower_left = true;
      is_upper_left = false;
      is_lower_right = false;
      is_upper_right = false;
    } else {
      ROS_INFO("the bool is wrong!");
    }
  }

  return ctrl_vec;
}

void TestSimpleCommand::print_control_command(
    const std::vector<dji_osdk_ros::JoystickCommand> &ctrl_command_vec) {
  for (int i = 0; i < ctrl_command_vec.size(); ++i) {
    auto em = ctrl_command_vec[i];
    ROS_INFO_STREAM("point:" << i << "-------");
    ROS_INFO_STREAM("x:" << em.x);
    ROS_INFO_STREAM("y:" << em.y);
    ROS_INFO_STREAM("yaw:" << em.yaw);
  }
}

bool TestSimpleCommand::moveByPosOffset(
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

int TestSimpleCommand::run(float desired_height, float zigzag_len,
                           float zigzag_wid, float zigzag_num) {
  ros::Rate rate(1);
  begin_time = ros::Time::now();
  char inputChar;

  /* gererate the zigzag path */
  ROS_INFO_STREAM("desired_height: " << desired_height << " m");
  ROS_INFO_STREAM("zigzag_len: " << zigzag_len << " m");
  ROS_INFO_STREAM("zigzag_wid: " << zigzag_wid << " m");
  ROS_INFO_STREAM("zigzag_num: " << zigzag_num << " m");
  auto command_vec = generate_zigzag_path(zigzag_len, zigzag_wid, zigzag_num);
  /* print_control_command(command_vec); */

  ROS_INFO_STREAM("Command generating finish, are you ready to take off? y/n");
  std::cin >> inputChar;
  if (inputChar == 'n') {
    ROS_INFO_STREAM("exist!");
    return 0;

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
      moveByPosOffset(control_task, {0.0, 0.0, desired_height, 0.0}, 0.8, 1);

      /* 3. Move following the zigzag path */
      ROS_INFO_STREAM("Move by position offset request sending ...");
      for (int i = 0; ros::ok() && (i < command_vec.size()); ++i) {
        ROS_INFO_STREAM("Moving to the point: " << i << "!");
        moveByPosOffset(control_task, command_vec[i], 0.8, 1);
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

  return 0;
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "test_simple_command_node");

  if (argc != 5) {
    ROS_ERROR_STREAM("usage: desired_height zigzag_len zigzag_wid zigzag_num");
    return 1;

  } else {
    TestSimpleCommand node;
    node.run(std::stof(argv[1]), std::stof(argv[2]), std::stof(argv[3]),
             std::stof(argv[4]));

    return 0;
  }
}
