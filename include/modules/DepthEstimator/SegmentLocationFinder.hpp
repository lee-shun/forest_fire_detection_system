/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: SegmentLocationFinder.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-05-16
 *
 *   @Description:
 *
 *******************************************************************************/

#ifndef INCLUDE_MODULES_DEPTHESTIMATOR_SEGMENTLOCATIONFINDER_HPP_
#define INCLUDE_MODULES_DEPTHESTIMATOR_SEGMENTLOCATIONFINDER_HPP_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

namespace FFDS {
namespace MODULES {

class SegmentLocationFinder {
 public:
  std::vector<cv::Point> FindLocation(const cv::Mat binary_input,
                                      const int morph_size,
                                      const bool imshow_contours = true,
                                      const bool imshow_final_rect = true) {
    cv::Mat binary = binary_input.clone();

    // STEP: 1 morphology
    cv::Mat element = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(morph_size, morph_size));

    cv::Mat after_open_binary;
    cv::morphologyEx(binary, after_open_binary, cv::MORPH_OPEN, element);

    cv::Mat after_close_binary;
    cv::morphologyEx(after_open_binary, after_close_binary, cv::MORPH_CLOSE,
                     element);

    // STEP: 2 find contours
    cv::Mat contours_img = cv::Mat::zeros(after_close_binary.rows,
                                          after_close_binary.cols, CV_8UC3);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(after_close_binary, contours, hierarchy, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
    if (imshow_contours) {
      int index = 0;
      for (; index >= 0; index = hierarchy[index][0]) {
        cv::Scalar color(255, 255, 255);
        drawContours(contours_img, contours, index, color, 0, 8, hierarchy);
      }

      cv::namedWindow("contours image:", cv::WINDOW_NORMAL);
      cv::imshow("contours image:", contours_img);
      cv::waitKey(0);
    }

    // STEP: 3 calculate rectangular center
    std::vector<cv::Point> center_of_contours;
    for (int i = 0; i < contours.size(); i++) {
      std::vector<cv::Point> points = contours[i];
      cv::RotatedRect box = minAreaRect(cv::Mat(points));
      cv::Point2f vertex[4];
      box.points(vertex);

      cv::Point s1;
      s1.x = (vertex[0].x + vertex[2].x) / 2.0;
      s1.y = (vertex[0].y + vertex[2].y) / 2.0;

      center_of_contours.push_back(s1);

      if (imshow_final_rect) {
        // draw rects
        cv::line(binary, vertex[0], vertex[1], cv::Scalar(100, 200, 211), 6,
                 cv::LINE_AA);
        cv::line(binary, vertex[1], vertex[2], cv::Scalar(100, 200, 211), 6,
                 cv::LINE_AA);
        cv::line(binary, vertex[2], vertex[3], cv::Scalar(100, 200, 211), 6,
                 cv::LINE_AA);
        cv::line(binary, vertex[3], vertex[0], cv::Scalar(100, 200, 211), 6,
                 cv::LINE_AA);

        // center
        cv::Point l, r, u, d;
        l.x = s1.x - 10;
        l.y = s1.y;

        r.x = s1.x + 10;
        r.y = s1.y;

        u.x = s1.x;
        u.y = s1.y - 10;

        d.x = s1.x;
        d.y = s1.y + 10;
        cv::line(binary, l, r, cv::Scalar(100, 200, 211), 2, cv::LINE_AA);
        cv::line(binary, u, d, cv::Scalar(100, 200, 211), 2, cv::LINE_AA);
      }
    }

    if (imshow_final_rect) {
      cv::namedWindow("final rect", cv::WINDOW_NORMAL);
      cv::imshow("final rect", binary);
      cv::waitKey(0);
    }

    return center_of_contours;
  }
};
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_DEPTHESTIMATOR_SEGMENTLOCATIONFINDER_HPP_
