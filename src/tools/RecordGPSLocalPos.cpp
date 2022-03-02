/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: RecordGPSLocalPos.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-01-07
 *
 *   @Description:
 *
 *******************************************************************************/

#include <dji_osdk_ros/SetLocalPosRef.h>
#include <geometry_msgs/PointStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>
#include <tools/PrintControl/PrintCtrlMacro.h>

#include <tools/PrintControl/FileWritter.hpp>

dji_osdk_ros::SetLocalPosRef set_local_pos_reference;
sensor_msgs::NavSatFix gps_position;
geometry_msgs::PointStamped local_position;

void callback(const sensor_msgs::NavSatFix::ConstPtr& gps_msg,
              const geometry_msgs::PointStamped::ConstPtr& local_msg) {
  gps_position = *gps_msg;
  local_position = *local_msg;
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "record_gps_node");
  ros::NodeHandle nh;

  ros::ServiceClient set_local_pos_ref_client =
      nh.serviceClient<dji_osdk_ros::SetLocalPosRef>(
          "/set_local_pos_reference");
  message_filters::Subscriber<sensor_msgs::NavSatFix> gps_sub(
      nh, "dji_osdk_ros/gps_position", 10);
  message_filters::Subscriber<geometry_msgs::PointStamped> local_pos_sub(
      nh, "dji_osdk_ros/local_position", 10);

  /**
   * bind 2 messages
   * */
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::NavSatFix, geometry_msgs::PointStamped>
      MySyncPolicy;
  /* ApproximateTime takes a queue size as its constructor argument, hence
   * MySyncPolicy(10) */
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), gps_sub,
                                                   local_pos_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));

  /**
   * start gps and local position
   * */
  set_local_pos_ref_client.call(set_local_pos_reference);
  if (set_local_pos_reference.response.result) {
    PRINT_INFO("Set local position reference successfully!");
  } else {
    PRINT_INFO("Set local position reference failed!");
  }
  ros::Duration(2.0).sleep();

  FFDS::TOOLS::FileWritter gpsPosWriter("m300_gps_pos.csv", 8);
  FFDS::TOOLS::FileWritter localPosWriter("m300_local_pos.csv", 8);
  gpsPosWriter.new_open();
  localPosWriter.new_open();
  gpsPosWriter.write("time_ms", "lon", "lat", "altitude");
  localPosWriter.write("time_ms", "E", "N", "U");

  while (ros::ok()) {
    ros::spinOnce();
    PRINT_DEBUG(
        "M300 gps position: longitude:%lf, latitude:%lf, altitude:%lf\n",
        gps_position.longitude, gps_position.latitude, gps_position.altitude);
    PRINT_DEBUG("M300 local position: x:%lf, y:%lf, z:%lf\n",
                local_position.point.x, local_position.point.y,
                local_position.point.z);
    gpsPosWriter.write(FFDS::TOOLS::getSysTime(), gps_position.longitude,
                       gps_position.latitude, gps_position.altitude);
    localPosWriter.write(FFDS::TOOLS::getSysTime(), local_position.point.x,
                         local_position.point.y, local_position.point.z);

    ros::Rate(10).sleep();
  }
  gpsPosWriter.close();
  localPosWriter.close();
  return 0;
}
