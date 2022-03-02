/*******************************************************************************
 *
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: ControllerBase.hpp
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

#ifndef INCLUDE_MODULES_BASICCONTROLLER_CONTROLLERBASE_HPP_
#define INCLUDE_MODULES_BASICCONTROLLER_CONTROLLERBASE_HPP_

namespace FFDS {
namespace MODULES {

class ControllerBase {
 public:
  virtual void reset() = 0;
  virtual void ctrl(const float in) = 0;
  virtual float getOutput() = 0;

 protected:
  float input{0.0};
  float output{0.0};
};
}  // namespace MODULES
}  // namespace FFDS
#endif  // INCLUDE_MODULES_BASICCONTROLLER_CONTROLLERBASE_HPP_
