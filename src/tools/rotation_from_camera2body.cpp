/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: rotation_from_camera2body.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-03-05
 *
 *   @Description:
 *
 *******************************************************************************/

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>

int main(int argc, char** argv) {
  Eigen::AngleAxisd rotation_z(
      M_PI / 2, Eigen::Vector3d(0, 0, 1));  // rotate 90 degree along z axies
  Eigen::AngleAxisd rotation_y(
      -M_PI / 2, Eigen::Vector3d(0, 1, 0));  // rotate 90 degree along z axies

  // following is testing

  // in camera
  Eigen::Vector3d vector_body(1, 2, 3);
  Eigen::Vector3d vector_camera = rotation_z * rotation_y * vector_body;

  std::cout << "camera:\n"
            << vector_camera.transpose() << "\nbody\n"
            << vector_body.transpose() << std::endl;

  Eigen::Quaterniond rotat_quatern_cb = rotation_z * rotation_y;

  std::cout << "camera:\n"
            << rotat_quatern_cb * vector_body << "\nbody\n"
            << vector_body.transpose() << std::endl;
  std::cout << "rotat_quan_cb = \n" << rotat_quatern_cb.w()<< std::endl;
  std::cout << "rotat_matrix_cb = \n" << rotat_quatern_cb.matrix() << std::endl;

  return 0;
}
