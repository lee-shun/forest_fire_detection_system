/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: FileWritter.hpp
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

#ifndef INCLUDE_TOOLS_PRINTCONTROL_FILEWRITTER_HPP_
#define INCLUDE_TOOLS_PRINTCONTROL_FILEWRITTER_HPP_

#include <tools/PrintControl/PrintCtrlMacro.h>

#include <iostream>
#include <string>
#include <tools/SystemLib.hpp>

namespace FFDS {
namespace TOOLS {
class FileWritter {
 public:
  FileWritter(const std::string file, const int number_precision)
      : fileName(file), num_precision(number_precision) {
    oufile.flags(std::ios::fixed);
    oufile.precision(num_precision);
  }
  ~FileWritter() {
    close();
    PRINT_INFO("file: %s is closed", fileName.c_str());
  }

  void open() { oufile.open(fileName.c_str(), std::ios::app | std::ios::out); }
  void close() { oufile.close(); }
  void setDelimiter(const std::string del) { delimiter = del; }
  void erase() {
    /* NOTE: fstream has a constructor same as open, can open file when the
     * NOTE: instance!
     * */
    std::ofstream fileErase(fileName.c_str(), std::ios::out | std::ios::trunc);
    fileErase.close();
  }
  void new_open() {
    erase();
    open();
  }

  template <typename T, typename... Ts>
  void write(T arg1, Ts... arg_left) {
    oufile << arg1 << delimiter;
    write(arg_left...);
  }

 private:
  std::string fileName;
  std::string delimiter{","};
  std::fstream oufile;
  int num_precision;

  void write() { oufile << std::endl; }
};
}  // namespace TOOLS
}  // namespace FFDS

#endif  // INCLUDE_TOOLS_PRINTCONTROL_FILEWRITTER_HPP_
