/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: DepthFilter.hpp
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

#ifndef INCLUDE_MODULES_DEPTHESTIMATOR_DEPTHFILTER_HPP_
#define INCLUDE_MODULES_DEPTHESTIMATOR_DEPTHFILTER_HPP_

#include "Eigen/Core"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

namespace FFDS {
namespace MODULES {
class DepthFilter {
 public:
  struct Param {
    // min value to judje depth is converaged.
    double min_cov{0.1};

    // max value to judje depth is divergent.
    double max_cov{10.0};

    // estimate the depth around given point in a depth_win_size square.
    int half_depth_win_size{10};

    // the border of an image
    int boarder{0};

    // image width
    int width{960};

    // image height
    int height{770};

    double fx{2079.73910150907f};
    double fy{2080.01151697192f};
    double cx{466.470046230910f};
    double cy{394.022978676365f};

    int ncc_win_size{10};
    int ncc_area{(2 * ncc_win_size + 1) * (2 * ncc_win_size + 1)};
  };

  /**
   * update the pt_ref point depth
   * */
  bool UpdateDepth(const cv::Mat &ref, const cv::Mat &curr,
                   const Sophus::SE3d &T_C_R, const Eigen::Vector2d pt_ref,
                   cv::Mat &depth, cv::Mat &depth_cov2);

  void setParam(const Param param_in) { param = param_in; }

 private:
  /**
   * @brief
   *
   * @param[in] ref refernce image
   * @param[in] curr current image
   * @param[in] T_C_R pose from reference to current image
   * @param[in] pt_ref point on the reference image
   * @param[in] depth_mu average of the depth
   * @param[in] depth_cov
   * @param[out] pt_curr
   * @param[out] epipolar_direction
   *
   * @return
   */
  bool EpipolarSearch(const cv::Mat &ref, const cv::Mat &curr,
                      const Sophus::SE3d &T_C_R, const Eigen::Vector2d &pt_ref,
                      const double &depth_mu, const double &depth_cov,
                      Eigen::Vector2d &pt_curr,
                      Eigen::Vector2d &epipolar_direction);

  bool UpdateDepthFilter(const Eigen::Vector2d &pt_ref,
                         const Eigen::Vector2d &pt_curr,
                         const Sophus::SE3d &T_C_R,
                         const Eigen::Vector2d &epipolar_direction,
                         cv::Mat &depth, cv::Mat &depth_cov2);

  // pixel plane to normalized plane (1)
  inline Eigen::Vector3d Px2Cam(const Eigen::Vector2d px) {
    return Eigen::Vector3d((px(0, 0) - param.cx) / param.fx,
                           (px(1, 0) - param.cy) / param.fy, 1);
  }

  inline Eigen::Vector2d Cam2Px(const Eigen::Vector3d p_cam) {
    return Eigen::Vector2d(p_cam(0, 0) * param.fx / p_cam(2, 0) + param.cx,
                           p_cam(1, 0) * param.fy / p_cam(2, 0) + param.cy);
  }

  inline bool Inside(const Eigen::Vector2d &pt) {
    return pt(0, 0) >= param.boarder && pt(1, 0) >= param.boarder &&
           pt(0, 0) + param.boarder < param.width &&
           pt(1, 0) + param.boarder <= param.height;
  }

  inline double BilinearInterpolated(const cv::Mat &img,
                                     const Eigen::Vector2d &pt) {
    uchar *d = &img.data[static_cast<int>(pt(1, 0)) * img.step +
                         static_cast<int>(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));

    return ((1 - xx) * (1 - yy) * static_cast<double>(d[0]) +
            xx * (1 - yy) * static_cast<double>(d[1]) +
            (1 - xx) * yy * static_cast<double>(d[img.step]) +
            xx * yy * static_cast<double>(d[img.step + 1])) /
           255.0;
  }

  double NCC(const cv::Mat &ref, const cv::Mat &curr,
             const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr);

  void showEpipolarMatch(const cv::Mat &ref, const cv::Mat &curr,
                         const Eigen::Vector2d &px_ref,
                         const Eigen::Vector2d &px_curr);

  void showEpipolarLine(const cv::Mat &ref, const cv::Mat &curr,
                        const Eigen::Vector2d &px_ref,
                        const Eigen::Vector2d &px_min_curr,
                        const Eigen::Vector2d &px_max_curr);

  Param param;
};

}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_DEPTHESTIMATOR_DEPTHFILTER_HPP_
