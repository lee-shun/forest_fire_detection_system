/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: RGB_IRSeperator.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-11-27
 *
 *   @Description:
 *
 *******************************************************************************/

#include <modules/ImgVideoOperator/RGB_IRSeperator.hpp>

FFDS::MODULES::RGB_IRSeperator::RGB_IRSeperator() {
  imgSub = nh.subscribe("dji_osdk_ros/main_camera_images", 1,
                        &RGB_IRSeperator::imageCallback, this);
  imgIRPub =
      it.advertise("forest_fire_detection_system/main_camera_ir_image", 1);
  imgRGBPub =
      it.advertise("forest_fire_detection_system/main_camera_rgb_image", 1);
  resizeImgRGBPub = it.advertise(
      "forest_fire_detection_system/main_camera_rgb_resize_image", 1);

  ros::Duration(2.0).sleep();
}

void FFDS::MODULES::RGB_IRSeperator::imageCallback(
    const sensor_msgs::Image::ConstPtr& img) {
  rawImgPtr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
  rawImg = rawImgPtr->image;
}

void FFDS::MODULES::RGB_IRSeperator::run() {
  const std::string package_path =
      ros::package::getPath("forest_fire_detection_system");
  const std::string config_path = package_path + "/config/H20T_Camera.yaml";
  PRINT_INFO("get camera params from %s", config_path.c_str());
  YAML::Node node = YAML::LoadFile(config_path);

  int irImgWid = FFDS::TOOLS::getParam(node, "pure_IR_width", 960);
  int irImgHet = FFDS::TOOLS::getParam(node, "pure_IR_height", 770);

  int rgbImgWid = FFDS::TOOLS::getParam(node, "pure_RGB_width", 960);
  int rgbImgHet = FFDS::TOOLS::getParam(node, "pure_RGB_height", 770);

  int upperBound = FFDS::TOOLS::getParam(node, "upper_bound", 336);
  int lowerBound = FFDS::TOOLS::getParam(node, "lower_bound", 1106);

  int irUpLeft_x = 0;
  int irUpLeft_y = upperBound;

  int rgbUpLeft_x = irImgWid;
  int rgbUpLeft_y = upperBound;

  /**
   * FIXED: the hh DJI change the video size after press the "RECORD" from the
   * FIXED: remoter! YOU GOT BE KIDDING ME!
   * */
  while (ros::ok()) {
    ros::spinOnce();

    /* PRINT_DEBUG("Org mixed image shape: rows: %d, cols: %d", rawImg.rows, */
    /*             rawImg.cols); */

    cv::Mat irImg =
        rawImg(cv::Rect(irUpLeft_x, irUpLeft_y, irImgWid, irImgHet));

    cv::Mat rgbImg =
        rawImg(cv::Rect(rgbUpLeft_x, rgbUpLeft_y, rgbImgWid, rgbImgHet));

    cv::Mat resizeRgbImg;
    cv::resize(rgbImg, resizeRgbImg, cv::Size(resRGBWid, resRGBHet));

    sensor_msgs::ImagePtr irMsg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", irImg).toImageMsg();
    sensor_msgs::ImagePtr rgbMsg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", rgbImg).toImageMsg();
    sensor_msgs::ImagePtr reszieRgbMsg =
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", resizeRgbImg)
            .toImageMsg();

    irMsg->header.frame_id = "H20T_IR";
    irMsg->header.stamp = ros::Time::now();

    rgbMsg->header.frame_id = "H20T_RGB";
    rgbMsg->header.stamp = irMsg->header.stamp;

    reszieRgbMsg->header.frame_id = "H20T_RGB_RESIZE";
    reszieRgbMsg->header.stamp = irMsg->header.stamp;

    imgIRPub.publish(irMsg);
    imgRGBPub.publish(rgbMsg);
    resizeImgRGBPub.publish(reszieRgbMsg);

    ros::Rate(10).sleep();
  }
}
