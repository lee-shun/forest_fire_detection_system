/*******************************************************************************
 *   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: test_FileCreationRemove.cpp
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2022-03-20
 *
 *   @Description:
 *
 *******************************************************************************/

#include "tools/SystemLib.hpp"
int main(int argc, char** argv) {
  std::string dir_name = "/home/ls/new_dir/2";
  std::string file_name = dir_name + "/new_file.txt";

  bool rmdir_success = FFDS::TOOLS::shellRm(dir_name);
  PRINT_INFO("rmdir_success: %d", rmdir_success);
  FFDS::TOOLS::removeFile(file_name);

  bool mkdir_success = FFDS::TOOLS::shellMkdir(dir_name);
  PRINT_INFO("mkdir_success: %d", mkdir_success);
  std::ofstream outfile;
  outfile.open(file_name);
  outfile.close();

  return 0;
}
