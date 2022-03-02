/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_PrintControl.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-01-06
 *
 *   @Description:
 *
 *******************************************************************************/

#include <tools/PrintControl/PrintCtrlMacro.h>
int main(int argc, char** argv) {
  PRINT_ERROR("hello, %d", 25);
  PRINT_WARN("hello, %d", 25);
  PRINT_INFO("hello, %d", 25);
  PRINT_ENTRY("hello, %d", 25);
  PRINT_DEBUG("hello, %d", 25);

  FPRINT_ERROR("lee.txt", "hello, %d", 25);
  FPRINT_WARN("lee.txt", "hello, %d", 25);
  FPRINT_INFO("lee.txt", "hello, %d", 25);
  FPRINT_ENTRY("lee.txt", "hello, %d", 25);
  FPRINT_DEBUG("lee.txt", "hello, %d", 25);
  return 0;
}
