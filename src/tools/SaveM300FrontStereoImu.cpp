/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: SaveM300FrontStereoImu.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-03-20
 *
 *   @Description:
 *
 *******************************************************************************/

#include "modules/StereoCameraOperator/StereoCamOperator.hpp"
#include "ros/package.h"
#include "tools/SystemLib.hpp"
#include "tools/PrintControl/FileWritter.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "save_m300_front_stereo_camera_imu_node");
  ros::NodeHandle nh;

  /* Step: 1 create dataset dir */
  const std::string homedir = getenv("HOME");
  const std::string m300_stereo_data_path = homedir + "/m300_data";

  // clear path and create new
  FFDS::TOOLS::shellRm(m300_stereo_data_path);
  FFDS::TOOLS::shellMkdir(m300_stereo_data_path);

  const std::string left_img_path = m300_stereo_data_path + "/image_0";
  const std::string right_img_path = m300_stereo_data_path + "/image_1";

  // clear path and create new
  FFDS::TOOLS::shellRm(left_img_path);
  FFDS::TOOLS::shellRm(right_img_path);
  FFDS::TOOLS::shellMkdir(left_img_path);
  FFDS::TOOLS::shellMkdir(right_img_path);

  /* Step: 2 create operator */
  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");
  const std::string m300_stereo_config_path =
      package_path + "/config/m300_front_stereo_param.yaml";
  PRINT_INFO("get camera params from %s", m300_stereo_config_path.c_str());
  std::shared_ptr<FFDS::MODULES::StereoCamOperator> stereo_cam_operator =
      std::make_shared<FFDS::MODULES::StereoCamOperator>(
          m300_stereo_config_path);
  stereo_cam_operator->IfGenerateRosPtCloud(false);

  // regist the shutDownHandler
  signal(SIGINT, FFDS::MODULES::StereoCamOperator::ShutDownHandler);

  const std::string att_file_path = m300_stereo_data_path + "/pose.txt";
  FFDS::TOOLS::FileWritter att_file_writter(att_file_path, 8);
  // clear and new
  att_file_writter.new_open();
  uint32_t img_index = 1;

  while (ros::ok()) {
    stereo_cam_operator->UpdateOnce();

    cv::Mat left_img = stereo_cam_operator->GetRectLeftImgOnce();
    cv::Mat right_img = stereo_cam_operator->GetRectRightImgOnce();

    if (left_img.empty() || right_img.empty()) {
      PRINT_WARN("no valid stereo images right now!");
      continue;
    }

    if (1 == img_index) {
      PRINT_INFO("start saving dataset ...");
    }

    geometry_msgs::QuaternionStamped att_body_ros =
        stereo_cam_operator->GetAttOnce();

    // write files
    cv::imwrite(left_img_path + "/" + std::to_string(img_index) + ".png",
                left_img);
    cv::imwrite(right_img_path + "/" + std::to_string(img_index) + ".png",
                right_img);
    att_file_writter.write(att_body_ros.quaternion.w, att_body_ros.quaternion.x,
                           att_body_ros.quaternion.y,
                           att_body_ros.quaternion.z);
    ++img_index;
  }

  return 0;
}
