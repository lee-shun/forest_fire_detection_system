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

  std::cout << "rotat_quan_cb.w.x.y.z = \n"
            << rotat_quatern_cb.w() << std::endl
            << rotat_quatern_cb.x() << std::endl
            << rotat_quatern_cb.y() << std::endl
            << rotat_quatern_cb.z() << std::endl;

  std::cout << "rotat_matrix_cb = \n" << rotat_quatern_cb.matrix() << std::endl;

  // inverse
  Eigen::Matrix3d rotat_matrix_bc = rotat_quatern_cb.matrix().inverse();
  Eigen::Quaterniond rotat_quatern_bc(rotat_matrix_bc);

  std::cout << "rotat_quan_bc.w.x.y.z = \n"
            << rotat_quatern_bc.w() << std::endl
            << rotat_quatern_bc.x() << std::endl
            << rotat_quatern_bc.y() << std::endl
            << rotat_quatern_bc.z() << std::endl;

  return 0;
}
