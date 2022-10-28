/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: GetCurrentGPSPos.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-01-13
 *
 *   @Description:
 *
 *******************************************************************************/

#include <sensor_msgs/NavSatFix.h>

#include <tools/PositionHelper.hpp>
#include <tools/PrintControl/FileWritter.hpp>
#include <tools/SystemLib.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "get_current_gps_pos_node");
  int average_times = 10;
  FFDS::TOOLS::PositionHelper posHelper;

  sensor_msgs::NavSatFix gps = posHelper.getAverageGPS(average_times);
  PRINT_INFO(
      "current GPS position under %d average times is lon: %.9f, lat: %.9f, "
      "alt: %.9f",
      average_times, gps.longitude, gps.latitude, gps.altitude);

  FFDS::TOOLS::FileWritter gpsWriter("fire_average_gps_pos.csv", 8);
  gpsWriter.new_open();
  gpsWriter.write("time_ms", "longitude", "latitude", "altitude");
  gpsWriter.write(FFDS::TOOLS::getSysTime(), gps.longitude, gps.latitude,
                  gps.altitude);
  gpsWriter.close();

  return 0;
}
