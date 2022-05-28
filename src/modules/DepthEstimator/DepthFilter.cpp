/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: DepthFilter.cpp
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

#include "modules/DepthEstimator/DepthFilter.hpp"
#include "tools/PrintControl/PrintCtrlMacro.h"

// TODO: 应该在当前点的邻域内进行深度估计. 一个小的窗口之中

bool FFDS::MODULES::DepthFilter::UpdateDepth(
    const cv::Mat &ref, const cv::Mat &curr, const Sophus::SE3d &T_C_R,
    const Eigen::Vector2d pt_ref, cv::Mat &depth, cv::Mat &depth_cov2) {
  // in the window
  for (int x = std::max(0.0, pt_ref[0] - param.half_depth_win_size);
       x < pt_ref[0] + param.half_depth_win_size; ++x) {
    for (int y = std::max(0.0, pt_ref[1] - param.half_depth_win_size);
         y < pt_ref[1] + param.half_depth_win_size; ++y) {
      if (depth_cov2.ptr<double>(y)[x] < param.min_cov) {
        PRINT_INFO("current point depth estimation is converged!");
        continue;
      } else if (depth_cov2.ptr<double>(y)[x] > param.max_cov) {
        PRINT_WARN("current point depth estimation is divergent!");
        continue;
      }

      // find current point on current image via epipolar search and block
      // matching.
      Eigen::Vector2d pt_curr;
      Eigen::Vector2d epipolar_direction;
      bool ret = EpipolarSearch(
          ref, curr, T_C_R, Eigen::Vector2d(x, y), depth.ptr<double>(y)[x],
          sqrt(depth_cov2.ptr<double>(y)[x]), pt_curr, epipolar_direction);
      if (!ret) continue;

      showEpipolarMatch(ref, curr, Eigen::Vector2d(x, y), pt_curr);

      // STEP: 3 update the depth
      UpdateDepthFilter(Eigen::Vector2d(x, y), pt_curr, T_C_R,
                        epipolar_direction, depth, depth_cov2);
    }
  }
  return true;
}

// TODO: draw the epipolar line and draw the box of matching point
bool FFDS::MODULES::DepthFilter::EpipolarSearch(
    const cv::Mat &ref, const cv::Mat &curr, const Sophus::SE3d &T_C_R,
    const Eigen::Vector2d &pt_ref, const double &depth_mu,
    const double &depth_cov, Eigen::Vector2d &pt_curr,
    Eigen::Vector2d &epipolar_direction) {
  // STEP: 1 find ref point on current image
  Eigen::Vector3d f_ref = Px2Cam(pt_ref);
  f_ref.normalize();
  Eigen::Vector3d P_ref = f_ref * depth_mu;

  Eigen::Vector2d px_mean_curr = Cam2Px(T_C_R * P_ref);
  double d_min = depth_mu - 3 * depth_cov;
  double d_max = depth_mu + 3 * depth_cov;

  if (d_min < 0.1) d_min = 0.1;

  Eigen::Vector2d px_min_curr = Cam2Px(T_C_R * (f_ref * d_min));
  Eigen::Vector2d px_max_curr = Cam2Px(T_C_R * (f_ref * d_max));

  Eigen::Vector2d epipolar_line = px_max_curr - px_min_curr;
  epipolar_direction = epipolar_line;
  epipolar_direction.normalize();

  double half_length = 0.5 * epipolar_line.norm();
  if (half_length > 100) half_length = 100;

  showEpipolarLine(ref, curr, pt_ref, px_min_curr, px_max_curr);

  double best_ncc = -1.0;
  Eigen::Vector2d best_px_curr;
  for (double l = -half_length; l <= half_length; l += 0.707) {
    Eigen::Vector2d px_curr = px_mean_curr + l * epipolar_direction;

    if (!Inside(px_curr)) continue;

    double ncc = NCC(ref, curr, pt_ref, px_curr);

    if (ncc > best_ncc) {
      best_ncc = ncc;
      best_px_curr = px_curr;
    }
  }

  if (best_ncc < 0.85f) return false;

  pt_curr = best_px_curr;

  return true;
}

double FFDS::MODULES::DepthFilter::NCC(const cv::Mat &ref, const cv::Mat &curr,
                                       const Eigen::Vector2d &pt_ref,
                                       const Eigen::Vector2d &pt_curr) {
  double mean_ref = 0, mean_curr = 0;
  std::vector<double> values_ref, values_curr;

  for (int x = -param.ncc_win_size; x <= param.ncc_win_size; x++)
    for (int y = -param.ncc_win_size; y <= param.ncc_win_size; y++) {
      if (!Inside(Eigen::Vector2d(x + pt_ref(0, 0), y + pt_ref(1, 0))))
        continue;
      double value_ref =
          static_cast<double>(ref.ptr<uchar>(static_cast<int>(
              y + pt_ref(1, 0)))[static_cast<int>(x + pt_ref(0, 0))]) /
          255.0;
      mean_ref += value_ref;

      double value_curr =
          BilinearInterpolated(curr, pt_curr + Eigen::Vector2d(x, y));
      mean_curr += value_curr;

      values_ref.push_back(value_ref);
      values_curr.push_back(value_curr);
    }

  mean_ref /= param.ncc_area;
  mean_curr /= param.ncc_area;

  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < values_ref.size(); i++) {
    double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
    numerator += n;
    demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
    demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
  }

  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

bool FFDS::MODULES::DepthFilter::UpdateDepthFilter(
    const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr,
    const Sophus::SE3d &T_C_R, const Eigen::Vector2d &epipolar_direction,
    cv::Mat &depth, cv::Mat &depth_cov2) {
  Sophus::SE3d T_R_C = T_C_R.inverse();
  Eigen::Vector3d f_ref = Px2Cam(pt_ref);
  f_ref.normalize();
  Eigen::Vector3d f_curr = Px2Cam(pt_curr);
  f_curr.normalize();

  // 方程
  // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
  // f2 = R_RC * f_cur
  // 转化成下面这个矩阵方程组
  // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
  //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
  Eigen::Vector3d t = T_R_C.translation();
  Eigen::Vector3d f2 = T_R_C.so3() * f_curr;
  Eigen::Vector2d b = Eigen::Vector2d(t.dot(f_ref), t.dot(f2));
  Eigen::Matrix2d A;
  A(0, 0) = f_ref.dot(f_ref);
  A(0, 1) = -f_ref.dot(f2);
  A(1, 0) = -A(0, 1);
  A(1, 1) = -f2.dot(f2);
  Eigen::Vector2d ans = A.inverse() * b;
  Eigen::Vector3d xm = ans[0] * f_ref;       // ref 侧的结果
  Eigen::Vector3d xn = t + ans[1] * f2;      // cur 结果
  Eigen::Vector3d p_esti = (xm + xn) / 2.0;  // P的位置，取两者的平均
  double depth_estimation = p_esti.norm();   // 深度值

  // 计算不确定性（以一个像素为误差）
  Eigen::Vector3d p = f_ref * depth_estimation;
  Eigen::Vector3d a = p - t;
  double t_norm = t.norm();
  double a_norm = a.norm();
  double alpha = acos(f_ref.dot(t) / t_norm);
  double beta = acos(-a.dot(t) / (a_norm * t_norm));
  Eigen::Vector3d f_curr_prime = Px2Cam(pt_curr + epipolar_direction);
  f_curr_prime.normalize();
  double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
  double gamma = M_PI - alpha - beta_prime;
  double p_prime = t_norm * sin(beta_prime) / sin(gamma);
  double d_cov = p_prime - depth_estimation;
  double d_cov2 = d_cov * d_cov;

  // 高斯融合
  double mu = depth.ptr<double>(
      static_cast<int>(pt_ref(1, 0)))[static_cast<int>(pt_ref(0, 0))];
  double sigma2 = depth_cov2.ptr<double>(
      static_cast<int>(pt_ref(1, 0)))[static_cast<int>(pt_ref(0, 0))];

  double mu_fuse =
      (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
  double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

  depth.ptr<double>(
      static_cast<int>(pt_ref(1, 0)))[static_cast<int>(pt_ref(0, 0))] = mu_fuse;
  depth_cov2.ptr<double>(static_cast<int>(
      pt_ref(1, 0)))[static_cast<int>(pt_ref(0, 0))] = sigma_fuse2;

  return true;
}

void FFDS::MODULES::DepthFilter::showEpipolarMatch(
    const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref,
    const Eigen::Vector2d &px_curr) {
  cv::Mat ref_show, curr_show;
  cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
  cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

  cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5,
             cv::Scalar(0, 0, 250), 2);
  cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5,
             cv::Scalar(0, 0, 250), 2);

  cv::imshow("ref", ref_show);
  cv::imshow("curr", curr_show);
  cv::waitKey(1);
}

void FFDS::MODULES::DepthFilter::showEpipolarLine(
    const cv::Mat &ref, const cv::Mat &curr, const Eigen::Vector2d &px_ref,
    const Eigen::Vector2d &px_min_curr, const Eigen::Vector2d &px_max_curr) {
  cv::Mat ref_show, curr_show;
  cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
  cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

  cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5,
             cv::Scalar(0, 255, 0), 2);
  cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5,
             cv::Scalar(0, 255, 0), 2);
  cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5,
             cv::Scalar(0, 255, 0), 2);
  cv::line(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)),
           cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
           cv::Scalar(0, 255, 0), 1);

  cv::imshow("ref", ref_show);
  cv::imshow("curr", curr_show);
  cv::waitKey(1);
}
