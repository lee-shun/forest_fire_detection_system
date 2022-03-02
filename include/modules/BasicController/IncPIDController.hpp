/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: IncPIDController.hpp
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

#ifndef INCLUDE_MODULES_BASICCONTROLLER_INCPIDCONTROLLER_HPP_
#define INCLUDE_MODULES_BASICCONTROLLER_INCPIDCONTROLLER_HPP_

#include "ControllerBase.hpp"

namespace FFDS {
namespace MODULES {

class IncPIDController : public ControllerBase {
 public:
  IncPIDController(float p, float i, float d) : Kp(p), Ki(i), Kd(d) {}
  ~IncPIDController() {}

  void reset() override;
  void ctrl(const float in) override;
  float getOutput() override;

  float getIncOutput();
  void setPrevOutput(const float prev);

 private:
  const float Kp;
  const float Ki;
  const float Kd;

  float input{0.0};
  float prev_input{0.0};
  float prev2_input{0.0};

  float increment{0.0};
  float output{0.0};
  float prev_output{0.0};

  void updateInput();
};

inline float IncPIDController::getIncOutput() { return increment; }

/**
 * @Input:
 * @Output:
 * @Description: 用于第一次进入时与其他控制方式的衔接
 */
inline void IncPIDController::setPrevOutput(const float prev) {
  prev_output = prev;
}

inline float IncPIDController::getOutput() {
  output = prev_output + increment;
  prev_output = output;

  return output;
}

inline void IncPIDController::reset() {
  prev_input = 0.0;
  prev2_input = 0.0;
  output = 0.0;
}

inline void IncPIDController::updateInput() {
  prev2_input = prev_input;
  prev_input = input;
}

inline void IncPIDController::ctrl(const float in) {
  input = in;
  float param_p = Kp * (input - prev_input);
  float param_i = Ki * input;
  float param_d = Kd * (input - 2 * prev_input + prev2_input);
  increment = param_p + param_i + param_d;

  updateInput();
}

}  // namespace MODULES

}  // namespace FFDS

#endif  // INCLUDE_MODULES_BASICCONTROLLER_INCPIDCONTROLLER_HPP_
