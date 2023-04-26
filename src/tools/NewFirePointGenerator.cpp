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
  ros::init(argc, argv, "new_fire_point_generator");
  int average_times = 10;
  FFDS::TOOLS::PositionHelper posHelper;

  sensor_msgs::NavSatFix gps = posHelper.getAverageGPS(average_times);
  PRINT_INFO(
      "当前抛水点 under %d average times is lon: %.9f, lat: %.9f, "
      "alt: %.9f",
      average_times, gps.longitude, gps.latitude, gps.altitude);

  FFDS::TOOLS::FileWritter gpsWriter("fire_average_gps_pos.csv", 8);
  gpsWriter.new_open();
  gpsWriter.write("time_ms", "longitude", "latitude", "altitude");
  gpsWriter.write(FFDS::TOOLS::getSysTime(), gps.longitude, gps.latitude,
                  gps.altitude);
  gpsWriter.close();

  // 拿到了实际的投水点
  double org_long = 0.0f, org_lat = 0.0f, new_long = 0.0f, new_lat = 0.0f;
  std::cout << "请输入原始的投水点，long, lat" << std::endl;
  std::cin >> org_long >> org_lat;

  new_long = org_long - (gps.longitude - org_long);
  new_lat = org_lat - (gps.latitude - org_lat);

  PRINT_INFO(
      "矫正之后的火点应该写入配置文件之中： under %d average times is lon: "
      "%.9f, lat: %.9f, "
      "alt: %.9f",
      average_times, new_long, new_lat, gps.altitude);

  return 0;
}
