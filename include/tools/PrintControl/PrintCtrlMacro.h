/*******************************************************************************
 *   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
 *
 *   @Filename: print_ctrl_macro.h
 *
 *   @Author: Shun Li
 *
 *   @Email: 2015097272@qq.com
 *
 *   @Date: 2021-11-10
 *
 *   @Description: macros that print logs to console and file
 *
 *******************************************************************************/

/* NOTE: namespace can not control macros in Cpp! */

#ifndef INCLUDE_TOOLS_PRINTCONTROL_PRINTCTRLMACRO_H_
#define INCLUDE_TOOLS_PRINTCONTROL_PRINTCTRLMACRO_H_

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define PCM_NONE 0
#define PCM_ERROR 1
#define PCM_WARN 2
#define PCM_INFO 3
#define PCM_ENTRY 4
#define PCM_DEBUG 5

#define PCM_PRINT_LEVEL PCM_DEBUG

#define PCM_FILENAME(x) strrchr(x, '/') ? strrchr(x, '/') + 1 : x

/**
 * console output
 * */

/* color control */
#define PCM_COLOR(color, msg) "\033[0;1;" #color "m" msg "\033[0m"
#define PCM_RED 31
#define PCM_GREEN 32
#define PCM_YELLOW 33
#define PCM_PURPLE 34
#define PCM_PINK 35
#define PCM_BLUE 36

#define PRINT_PURE(level, ...)              \
  do {                                      \
    if (level <= PCM_PRINT_LEVEL) {         \
      printf("[" #level "]>>" __VA_ARGS__); \
      printf("\n");                         \
    }                                       \
  } while (0);

#define PRINT(color, level, ...)                                      \
  do {                                                                \
    if (level <= PCM_PRINT_LEVEL) {                                   \
      printf(PCM_COLOR(color, "[" #level "]"));                       \
      printf(PCM_COLOR(36, " %s:%d (in %s) "), PCM_FILENAME(__FILE__), \
             __LINE__, __FUNCTION__);                                 \
      printf(__VA_ARGS__);                                            \
      printf("\n");                                                   \
    }                                                                 \
  } while (0);

#define PRINT_ERROR(...)                 \
  do {                                   \
    PRINT(31, PCM_ERROR, ##__VA_ARGS__); \
  } while (0);

#define PRINT_WARN(...)                 \
  do {                                  \
    PRINT(33, PCM_WARN, ##__VA_ARGS__); \
  } while (0);

#define PRINT_INFO(...)                 \
  do {                                  \
    PRINT(32, PCM_INFO, ##__VA_ARGS__); \
  } while (0);

#define PRINT_ENTRY(...)                 \
  do {                                   \
    PRINT(34, PCM_ENTRY, ##__VA_ARGS__); \
  } while (0);

#define PRINT_DEBUG(...)                 \
  do {                                   \
    PRINT(35, PCM_DEBUG, ##__VA_ARGS__); \
  } while (0);

/**
 * file output
 * */
#define FPRINT(level, file_name, ...)                            \
  do {                                                           \
    FILE *file_fp = NULL;                                        \
    file_fp = fopen(file_name, "a");                             \
                                                                 \
    if (level <= PCM_PRINT_LEVEL) {                              \
      if (file_fp != NULL) {                                     \
        fprintf(file_fp, "[" #level "] [%s:%d|in %s], ",         \
                PCM_FILENAME(__FILE__), __LINE__, __FUNCTION__); \
        fprintf(file_fp, __VA_ARGS__);                           \
        fprintf(file_fp, "\n");                                  \
      } else {                                                   \
        PRINT_ERROR("Can not open file!");                       \
      }                                                          \
      fclose(file_fp);                                           \
    } else {                                                     \
      fclose(file_fp);                                           \
    }                                                            \
  } while (0);

#define FPRINT_ERROR(file_name, ...)             \
  do {                                           \
    FPRINT(PCM_ERROR, file_name, ##__VA_ARGS__); \
  } while (0);

#define FPRINT_WARN(file_name, ...)             \
  do {                                          \
    FPRINT(PCM_WARN, file_name, ##__VA_ARGS__); \
  } while (0);

#define FPRINT_INFO(file_name, ...)             \
  do {                                          \
    FPRINT(PCM_INFO, file_name, ##__VA_ARGS__); \
  } while (0);

#define FPRINT_ENTRY(file_name, ...)             \
  do {                                           \
    FPRINT(PCM_ENTRY, file_name, ##__VA_ARGS__); \
  } while (0);

#define FPRINT_DEBUG(file_name, ...)             \
  do {                                           \
    FPRINT(PCM_DEBUG, file_name, ##__VA_ARGS__); \
  } while (0);

#endif  //  INCLUDE_TOOLS_PRINTCONTROL_PRINTCTRLMACRO_H_
