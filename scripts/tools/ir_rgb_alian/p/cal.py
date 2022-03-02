#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

#------------------------------------------------------------------------------
#
#   Copyright (C) 2021 Concordia NAVlab. All rights reserved.
#
#   @Filename: cal.py
#
#   @Author: Shun Li
#
#   @Date: 2021-11-24
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
#------------------------------------------------------------------------------

import math
import cv2

# got_IR_pos = [(420, 326), (407, 400), (489, 399), (473, 326)]
# got_RGB_pos = [(386, 365), (372, 432), (447, 432), (433, 365)]

got_IR_pos = [(407,273),(407,386),(523,386),(524,271)]
got_RGB_pos = [(390,312),(389,414),(499,414),(499,311)]


def cal_square(pos):
    d = -(pos[0][1] + pos[3][1]) / 2 + (pos[1][1] + pos[2][1]) / 2
    upper = pos[0][0] - pos[3][0]
    lower = pos[1][0] + pos[2][0]

    return (upper + lower) * d / 2


def cal_scale_offset(IR_pos, RGB_pos):
    ir_square = cal_square(IR_pos)
    rgb_square = cal_square(RGB_pos)
    print("ir_square: ", ir_square)
    print("rgb_square: ", rgb_square)

    scale = math.sqrt(rgb_square / ir_square)
    print("scale:", scale)

    scaled_ir_pos = []
    offset = []
    for i in range(len(IR_pos)):

        ir_ele = IR_pos[i]
        rgb_ele = RGB_pos[i]

        scaled_ir_pos_ele = (ir_ele[0] * scale, ir_ele[1] * scale)
        scaled_ir_pos.append(scaled_ir_pos_ele)

        offset_ele = ((rgb_ele[0] - scaled_ir_pos_ele[0]),
                      (rgb_ele[1] - scaled_ir_pos_ele[1]))
        offset.append(offset_ele)

    print("scaled_ir_pos: ", scaled_ir_pos)
    print("offset", offset)

    sum_off_x = 0
    sum_off_y = 0
    for off_ele in offset:
        sum_off_x += off_ele[0]
        sum_off_y += off_ele[1]

    aver_off_x = sum_off_x / len(offset)
    aver_off_y = sum_off_y / len(offset)

    print("aver_off_x:", aver_off_x)
    print("aver_off_y:", aver_off_y)

    return scale, round(aver_off_x), round(aver_off_y)


if __name__ == '__main__':
    scale, off_x, off_y = cal_scale_offset(got_IR_pos, got_RGB_pos)
    rgb_img = cv2.imread("../3.5m_1/RGB_3.5.jpg")
    ir_img = cv2.imread("../3.5m_1/IR_3.5.jpg")
    print(ir_img.shape)

    ir_scale_width = round(ir_img.shape[1] * scale)
    ir_scale_height = round(ir_img.shape[0] * scale)

    ir_resized_img = cv2.resize(ir_img, (ir_scale_width, ir_scale_height))
    cv2.imshow("ir_resized_img", ir_resized_img)
    cv2.waitKey(3)

    for i in range(round(ir_resized_img.shape[1]/2.1), ir_resized_img.shape[1]):
        for j in range(ir_resized_img.shape[0]):
            index_x = i + off_x
            index_y = j + off_y
            if (index_x >= rgb_img.shape[1] or index_y >= rgb_img.shape[0]):
                pass
            else:
                rgb_img[index_y][index_x] = ir_resized_img[j][
                    i] * 0.9 + rgb_img[index_y][index_x] * 0.1

    cv2.imwrite("aligned.jpg", rgb_img)
    cv2.imshow("alianed img", rgb_img)
    cv2.waitKey(0)
