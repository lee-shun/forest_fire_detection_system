/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_FileWritter.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-01-07
 *
 *   @Description:
 *
 *******************************************************************************/

#include <tools/PrintControl/FileWritter.hpp>
#include <tools/SystemLib.hpp>
int main(int argc, char** argv) {
  FFDS::TOOLS::FileWritter writter("text.txt", 8);
  writter.open();
  writter.erase();
  writter.write("time", "data1", "data2", "data3");
  for (int i = 0; i < 10; i++) {
    std::int32_t time = FFDS::TOOLS::getSysTime();
    PRINT_INFO("time:%d:", time);
    writter.write(FFDS::TOOLS::getSysTime(), i + 1, i + 2, i + 3);
    sleep(1);
  }
  /* Use the deconstructor~ */
  /* writter.close(); */

  return 0;
}
