/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: PIDController.hpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-10-31
 *
 *   @Description:
 *
 ******************************************************************************/

#ifndef INCLUDE_MODULES_BASICCONTROLLER_PIDCONTROLLER_HPP_
#define INCLUDE_MODULES_BASICCONTROLLER_PIDCONTROLLER_HPP_

#include <iostream>
#include <tools/MathLib.hpp>
#include <tools/SystemLib.hpp>

#include "ControllerBase.hpp"

namespace FFDS {
namespace MODULES {

class PIDController : public ControllerBase {
 private:
  float current_time{0.0};
  float last_time{0.0};
  float _dt{0.0};
  const float _dt_default{0.2};
  const float _dt_max{0.1};
  const float _dt_min{0.01};

  float last_input{0.0};
  bool use_integ{false};
  bool use_diff{false};

  float integ{0.0};
  const float integ_max{500.0};
  const float integ_min{-500.0};

  float kp{0.0};
  float ki{0.0};
  float kd{0.0};

  float ele_p{0};
  float ele_i{0};
  float ele_d{0};

  void cal_dt();

 public:
  PIDController(const float in_kp, const float in_ki, const float in_kd,
                bool useInteg, bool useDiff)
      : kp(in_kp),
        ki(in_ki),
        kd(in_kd),
        use_integ(useInteg),
        use_diff(useDiff) {}

  ~PIDController() {}

  void reset() override;
  void ctrl(const float in) override;
  float getOutput() override;

  inline float getInteg() { return integ; }
};

inline void PIDController::cal_dt() {
  _dt = current_time - last_time;
  _dt = TOOLS::Constrain(_dt, _dt_min, _dt_max);
}

inline void PIDController::reset() {
  current_time = 0;
  last_time = 0;
  _dt = 0;
  integ = 0;
}

inline float PIDController::getOutput() { return output; }

inline void PIDController::ctrl(const float in) {
  current_time = TOOLS::getSysTime() / 1000.0;
  input = in;

  cal_dt();

  ele_p = input * kp;

  ele_d = kd * (input - last_input) / _dt;

  if (integ >= integ_max) {
    if (input < 0) {
      integ = integ + input;
    }
  } else if (integ <= integ_min) {
    if (input > 0) {
      integ = integ + input;
    }
  } else {
    integ = integ + input;
  }

  ele_i = ki * integ;

  if (!use_integ) {
    ele_i = 0.0;
  }

  if (!use_diff) {
    ele_d = 0.0;
  }

  last_time = current_time;
  last_input = input;

  /* calculate the output */
  output = ele_p + ele_i + ele_d;
}
}  // namespace MODULES
}  // namespace FFDS

#endif  // INCLUDE_MODULES_BASICCONTROLLER_PIDCONTROLLER_HPP_
