/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: MathLib.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-25
 *
 *   @Description:
 *
 ******************************************************************************/

#ifndef INCLUDE_TOOLS_MATHLIB_HPP_
#define INCLUDE_TOOLS_MATHLIB_HPP_

#include <cmath>
#include <common/CommonTypes.hpp>

namespace FFDS {
namespace TOOLS {

const double EARTH_R = 6378137.0;
const double CONSTANTS_ONE_G = 9.80665;

static bool isEqualf(const float a, const float b) {
  return (fabs(a - b) <= 1e-6);
}

static bool isEquald(const double a, const double b) {
  return (fabs(a - b) <= 1e-15);
}

template <typename T>
bool isFinite(const T a) {
  return ((fabs(a) > 0.02) && (fabs(a) < 1000));
}

template <typename T>
T Constrain(const T val, const T min, const T max) {
  return (val < min) ? min : ((val > max) ? max : val);
}

template <typename T>
T Max(const T a, const T b) {
  return (a > b) ? a : b;
}

template <typename T>
T Min(const T a, const T b) {
  return (a < b) ? a : b;
}

template <typename T>
void Quaternion2Euler(const T quat[4], T euler[3]) {
  euler[0] = atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                   1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]));
  euler[1] = asin(2.0 * (quat[2] * quat[0] - quat[3] * quat[1]));
  euler[2] = atan2(2.0 * (quat[3] * quat[0] + quat[1] * quat[2]),
                   -1.0 + 2.0 * (quat[0] * quat[0] + quat[1] * quat[1]));
}

template <typename T>
void Euler2Quaternion(const T euler[3], T quat[4]) {
  T cosPhi_2 = cos(static_cast<T>(euler[0]) / 2.0);

  T sinPhi_2 = sin(static_cast<T>(euler[0]) / 2.0);

  T cosTheta_2 = cos(static_cast<T>(euler[1]) / 2.0);

  T sinTheta_2 = sin(static_cast<T>(euler[1]) / 2.0);

  T cosPsi_2 = cos(static_cast<T>(euler[2]) / 2.0);

  T sinPsi_2 = sin(static_cast<T>(euler[2]) / 2.0);

  quat[0] = static_cast<T>(cosPhi_2 * cosTheta_2 * cosPsi_2 +
                           sinPhi_2 * sinTheta_2 * sinPsi_2);

  quat[1] = static_cast<T>(sinPhi_2 * cosTheta_2 * cosPsi_2 -
                           cosPhi_2 * sinTheta_2 * sinPsi_2);

  quat[2] = static_cast<T>(cosPhi_2 * sinTheta_2 * cosPsi_2 +
                           sinPhi_2 * cosTheta_2 * sinPsi_2);

  quat[3] = static_cast<T>(cosPhi_2 * cosTheta_2 * sinPsi_2 -
                           sinPhi_2 * sinTheta_2 * cosPsi_2);
}

template <typename T>
void MatrixByVector3(T vector_res[3], const T rotmax[3][3],
                     const T vector_b[3]) {
  vector_res[0] = rotmax[0][0] * vector_b[0] + rotmax[0][1] * vector_b[1] +
                  rotmax[0][2] * vector_b[2];

  vector_res[1] = rotmax[1][0] * vector_b[0] + rotmax[1][1] * vector_b[1] +
                  rotmax[1][2] * vector_b[2];

  vector_res[2] = rotmax[2][0] * vector_b[0] + rotmax[2][1] * vector_b[1] +
                  rotmax[2][2] * vector_b[2];
}

/**
 * create rotation matrix for the quaternion
 */
template <typename T>
void Quat2Rotmax(const T q[4], T R[3][3]) {
  T aSq = q[0] * q[0];
  T bSq = q[1] * q[1];
  T cSq = q[2] * q[2];
  T dSq = q[3] * q[3];
  R[0][0] = aSq + bSq - cSq - dSq;
  R[0][1] = 2.0f * (q[1] * q[2] - q[0] * q[3]);
  R[0][2] = 2.0f * (q[0] * q[2] + q[1] * q[3]);
  R[1][0] = 2.0f * (q[1] * q[2] + q[0] * q[3]);
  R[1][1] = aSq - bSq + cSq - dSq;
  R[1][2] = 2.0f * (q[2] * q[3] - q[0] * q[1]);
  R[2][0] = 2.0f * (q[1] * q[3] - q[0] * q[2]);
  R[2][1] = 2.0f * (q[0] * q[1] + q[2] * q[3]);
  R[2][2] = aSq - bSq - cSq + dSq;
}

template <typename T>
T Rad2Deg(const T rad) {
  return rad * 180 / M_PI;
}

template <typename T>
T Deg2Rad(const T deg) {
  return deg * M_PI / 180;
}

/**
 * ref_in_deg: lon, lat, alt
 * relative_pos: NED
 * */
template <typename T>
COMMON::GPSPosition<T> Meter2LatLongAlt(
    const COMMON::GPSPosition<T> ref_in_deg,
    const COMMON::LocalPosition<T>& relative_pos) {
  COMMON::GPSPosition<T> res;
  if (relative_pos.x == 0 && relative_pos.y == 0) {
    res.lon = ref_in_deg.lon;
    res.lat = ref_in_deg.lat;
  } else {
    T local_radius = cos(Deg2Rad(ref_in_deg.lat)) * EARTH_R;
    res.lon = ref_in_deg.lon + relative_pos.y / local_radius;
    res.lat = ref_in_deg.lat + relative_pos.x / EARTH_R;
  }

  /* Height */
  res.alt = ref_in_deg.alt + relative_pos.z;
  return res;
}

/**
 * ref_in_deg: lat, lon, alt
 * relative_pos: NED
 * FIXME: need to change the order...
 * */
template <typename T>
void Meter2LatLongAlt(const T ref[3], const COMMON::LocalPosition<T>& local_pos,
                      T result[3]) {
  if (local_pos.x == 0 && local_pos.y == 0) {
    result[0] = ref[0];
    result[1] = ref[1];
  } else {
    T local_radius = cos(Deg2Rad(ref[0])) * EARTH_R;
    /* 得到的是lat，x是北向位置，所以在大圆上 */
    result[0] = ref[0] + Rad2Deg(local_pos.x / EARTH_R);
    /* 得到的是long，在维度圆上 */
    result[1] = ref[1] + Rad2Deg(local_pos.y / local_radius);
  }
  /* 高度 */
  result[2] = ref[2] + local_pos.z;
}

/**
 * a: reference point in deg: lat, lon, alt
 * m: NED
 * FIXME: need to change the order...
 * */
template <typename T>
void LatLong2Meter(const T a_pos[2], const T b_pos[2], T m[2]) {
  T lat1 = a_pos[0];
  T lon1 = a_pos[1];

  T lat2 = b_pos[0];
  T lon2 = b_pos[1];

  /* 涉及到ned是向北增加，且纬度向北也增加 */
  T n_distance = Deg2Rad(lat2 - lat1) * EARTH_R;

  T r_at_ref1 = cos(Deg2Rad(lat1)) * EARTH_R;

  /* 涉及到ned是向东增加，但是经度向东减少 */
  T e_distance = Deg2Rad(lon2 - lon1) * r_at_ref1;

  m[0] = n_distance;
  m[1] = e_distance;
}

}  // namespace TOOLS
}  // namespace FFDS

#endif  // INCLUDE_TOOLS_MATHLIB_HPP_
