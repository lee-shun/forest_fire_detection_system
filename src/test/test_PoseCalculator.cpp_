/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_PoseCalculator.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-03-18
 *
 *   @Description:
 *
 *******************************************************************************/

#include <ros/ros.h>
#include <ros/package.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include <stereo_camera_vo/tool/image_preprocess.h>

#include "modules/PoseCalculator/PoseCalculator.hpp"
#include "modules/StereoCameraOperator/StereoCamOperator.hpp"

Sophus::SE3d CalDronePose(
    FFDS::MODULES::StereoCamOperator& stereo_camera_operator,
    FFDS::MODULES::PoseCalculator& pose_calculator) {
  // get images to calculate the pose
  cv::Mat left_img = stereo_camera_operator.GetRectLeftImgOnce();
  cv::Mat right_img = stereo_camera_operator.GetRectRightImgOnce();

  stereo_camera_vo::tool::GammaTransform(left_img);
  stereo_camera_vo::tool::GammaTransform(right_img);

  geometry_msgs::QuaternionStamped att_body_ros =
      stereo_camera_operator.GetAttOnce();
  Eigen::Quaterniond q;
  q.w() = att_body_ros.quaternion.w;
  q.x() = att_body_ros.quaternion.x;
  q.y() = att_body_ros.quaternion.y;
  q.z() = att_body_ros.quaternion.z;

  // pose from DJI can be extended to pose Twb
  Sophus::SE3d Twb_init(q, Eigen::Vector3d::Zero());

  std::cout << "drone pose before: \n" << Twb_init.matrix() << std::endl;

  Sophus::SE3d Tcw_init =
      FFDS::MODULES::PoseCalculator::Twb2Twc(Twb_init).inverse();
  pose_calculator.Step(left_img, right_img, &Tcw_init);
  Sophus::SE3d Twb = FFDS::MODULES::PoseCalculator::Twc2Twb(Tcw_init.inverse());

  std::cout << "drone pose after: \n" << Twb.matrix() << std::endl;

  return Twb;
}

// NOTE: the point cloud(very slow) and pose calculate should not be in the same
// loop.
int main(int argc, char** argv) {
  ros::init(argc, argv, "test_pose_calculator_node");
  ros::NodeHandle nh;

  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");

  const std::string m300_stereo_config_path =
      package_path + "/config/m300_front_stereo_param.yaml";

  // point cloud
  ros::Publisher pt_pub =
      nh.advertise<sensor_msgs::PointCloud2>("/point_cloud/output", 10);

  FFDS::MODULES::PoseCalculator pose_calculator;
  FFDS::MODULES::StereoCamOperator stereo_camera_operator(
      m300_stereo_config_path);

  // broadcast tf2
  tf2_ros::TransformBroadcaster map2uav_br;
  geometry_msgs::TransformStamped map2uav_tf;

  while (ros::ok()) {
    stereo_camera_operator.UpdateOnce();

    if (stereo_camera_operator.GetMessageFilterStatus() ==
        FFDS::MODULES::StereoCamOperator::MessageFilterStatus::EMPTY) {
      PRINT_WARN("no valid message filter! continue ...");
      continue;
    }

    sensor_msgs::PointCloud2 pt_cloud;
    pt_cloud = stereo_camera_operator.GetRosPtCloudOnce();
    pt_cloud.header.frame_id = "front_stereo_camera";
    // publish pt cloud
    pt_pub.publish(pt_cloud);

    // calculate pose via VO
    Sophus::SE3d Twb = CalDronePose(stereo_camera_operator, pose_calculator);

    map2uav_tf.header.stamp = ros::Time::now();
    map2uav_tf.header.frame_id = "map";
    map2uav_tf.child_frame_id = "uav";
    map2uav_tf.transform.translation.x = Twb.translation()[0];
    map2uav_tf.transform.translation.y = Twb.translation()[1];
    map2uav_tf.transform.translation.z = Twb.translation()[2];

    Eigen::Quaterniond rotation_wb(Twb.rotationMatrix());

    map2uav_tf.transform.rotation.w = rotation_wb.w();
    map2uav_tf.transform.rotation.x = rotation_wb.x();
    map2uav_tf.transform.rotation.y = rotation_wb.y();
    map2uav_tf.transform.rotation.z = rotation_wb.z();
    map2uav_br.sendTransform(map2uav_tf);
  }

  pose_calculator.Stop();

  return 0;
}
