/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_DepthFilter.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-05-27
 *
 *   @Description:
 *
 *******************************************************************************/

#include "modules/DepthEstimator/DepthFilter.hpp"
#include "tools/PrintControl/PrintCtrlMacro.h"
#include "tools/SystemLib.hpp"

bool ReadTranslation(const std::string filename, const int index,
                     Eigen::Vector3d* trans) {
  std::ifstream fin;
  fin.open(filename);
  if (!fin) {
    PRINT_ERROR("can not open %s in given path, no such file or directory!",
                filename.c_str());
    return false;
  }

  std::string trans_tmp;
  std::vector<double> trans_elements;
  FFDS::TOOLS::SeekToLine(fin, index + 1);
  // read each index, x, y, z, everytime
  for (int i = 0; i < 4; ++i) {
    if (!getline(fin, trans_tmp, ',')) {
      PRINT_ERROR("pose reading error! at index %d", index);
      return false;
    }
    // PRINT_DEBUG("read trans:index+xyz:%.8f", std::stod(trans_tmp));
    trans_elements.push_back(std::stod(trans_tmp));
  }

  if (trans_elements[0] != static_cast<double>(index)) {
    PRINT_INFO("mismach index of give and read! give: %d, read: %f", index,
               trans_elements[0]);
    return false;
  }

  (*trans) << trans_elements[1], trans_elements[2], trans_elements[3];

  return true;
}

int main(int argc, char** argv) {
  const std::string home = std::getenv("HOME");
  const std::string dataset_path =
      home + "/m300_depth/m300_grabbed_data_1_17.1";
  const std::string translation_path = dataset_path + "/local_pose.csv";
  const std::string img_path = dataset_path + "/rgb";

  FFDS::MODULES::DepthFilter filter;
  FFDS::MODULES::DepthFilter::Param param;

  // STEP: read the ref image
  cv::Mat ref_img = cv::imread(img_path + "/1.png", 0);
  double init_depth = 10.0;  // 深度初始值
  double init_cov2 = 10.0;   // 方差初始值
  cv::Mat depth(param.height, param.width, CV_64F, init_depth);  // 深度图
  cv::Mat depth_cov2(param.height, param.width, CV_64F,
                     init_cov2);  // 深度图方差

  // TODO: 指定ref_point
  Eigen::Vector2d ref_point(707, 386);

  // STEP: read the ref translation
  Eigen::Vector3d ref_trans;
  if (!ReadTranslation(translation_path, 1, &ref_trans)) return 1;
  Sophus::SE3d TWR(Eigen::Matrix3d::Identity(), ref_trans);

  // read updates from index(image name: 1)
  for (int i = 2; i < 10; ++i) {
    cv::Mat cur_img =
        cv::imread(img_path + "/" + std::to_string(i) + ".png", 0);
    Eigen::Vector3d cur_trans;
    if (!ReadTranslation(translation_path, i, &cur_trans)) return 1;
  Sophus::SE3d TWC(Eigen::Matrix3d::Identity(), cur_trans);

    Sophus::SE3d TCR = TWC.inverse() * TWR;

    std::cout << "TCR.rotation" << TCR.rotationMatrix() << std::endl;
    std::cout << "TCR.translation" << TCR.translation() << std::endl;

    if (ref_img.empty() || cur_img.empty()) {
      PRINT_ERROR("image index: %d is Empty!", i);
      return 1;
    }

    filter.UpdateDepth(ref_img, cur_img, TCR, ref_point, depth, depth_cov2);
  }

  // 输出最后的结果
  cv::imshow("depth_estimation", depth * 0.2);
  cv::waitKey(0);
  PRINT_INFO("depth: %f", depth.ptr<double>(707)[386]);
  return 0;
}
