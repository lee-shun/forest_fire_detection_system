#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: align_the_rgb_ir.py
#
#   @Author: Shun Li
#
#   @Date: 2022-01-09
#
#   @Email: 2015097272@qq.com
#
#   @Description:
#
# ------------------------------------------------------------------------------

import numpy as np
h = np.loadtxt("e_vector_last.txt", delimiter=',')
R = np.reshape(h[:9], [3,3])
print(R)

t = np.reshape(h[9:], [3,1])

# print(R)
# print(t)

def transpose_in_rgb(ir_point, R, t):
    rgb_point = np.empty([2,1], dtype=np.float)

    x_1 = ir_point[0,0]
    y_1 = ir_point[1,0]

    rgb_point[0,0] = (R[0,0]*x_1+R[0,1]*y_1+R[0,2]+t[0,0])/(R[2,0]*x_1+R[2,1]*y_1+R[2,2]+t[2,0])
    rgb_point[1,0] = (R[1,0]*x_1+R[1,1]*y_1+R[1,2]+t[1,0])/(R[2,0]*x_1+R[2,1]*y_1+R[2,2]+t[2,0])

    return rgb_point

'''
ir: (241,270),(382,270),(530,364),(588,116),(787,77),(802,361)
rgb: (174,324),(301,324),(446,407),(489,184),(678,141),(688,399)
'''
ir_test = np.array([241, 270, 1],dtype = np.float).reshape([3,1])
rgb_raw = np.matmul(R, ir_test)+t
rgb_test = rgb_raw/rgb_raw[2, 0]
print(rgb_test)

test_rgb = transpose_in_rgb(np.array([241,270]).reshape([2,1]), R, t)
print(test_rgb)


