/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: HandleImageTransport.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-12-06
 *
 *   @Description:
 *
 *******************************************************************************/

#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

sensor_msgs::Image recvImg;

void imageCallback(const sensor_msgs::ImageConstPtr& msg) { recvImg = *msg; }

int main(int argc, char** argv) {
  ros::init(argc, argv, "handle_image_transport_node");
  ros::NodeHandle nh;
  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe(
      "forest_fire_detection_system/main_camera_rgb_resize_image", 1,
      imageCallback);
  ros::Publisher pub = nh.advertise<sensor_msgs::Image>(
      "forest_fire_detection_system/main_camera_rgb_transport_image", 1);

  while (ros::ok()) {
    ros::spinOnce();
    pub.publish(recvImg);
  }
  return 0;
}
