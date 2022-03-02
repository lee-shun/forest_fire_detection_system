#!/usr/bin/env python3
# -*- coding: utf-8 -*- #

# ------------------------------------------------------------------------------
#
#   Copyright (C) 2022 Concordia NAVlab. All rights reserved.
#
#   @Filename: calculate_paramter.py
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

# to calculate the A^TA eigenvalue and eigenvector.

import numpy as np

np.set_printoptions(suppress=True)


def creat_A_i(ir_point, rgb_point):
    x_1 = ir_point[0]
    y_1 = ir_point[1]

    x_2 = rgb_point[0]
    y_2 = rgb_point[1]

    A_i = np.empty([2, 12], dtype=float)

    # index 0    1     2     3     4     5     6     7     8     9    10   11
    # h = [R_11, R_12, R_13, R_21, R_22, R_23, R_31, R_32, R_33, t_1, t_2, t_3]

    # first row
    A_i[0, 0] = -x_1  # R_11
    A_i[0, 1] = -y_1  # R_12
    A_i[0, 2] = -1  # R_13
    A_i[0, 3] = 0  # R_21
    A_i[0, 4] = 0  # R_22
    A_i[0, 5] = 0  # R_23
    A_i[0, 6] = x_2 * x_1  # R_31
    A_i[0, 7] = x_2 * y_1  # R_32
    A_i[0, 8] = x_2  # R_33
    A_i[0, 9] = -1  # t_1
    A_i[0, 10] = 0  # t_2
    A_i[0, 11] = x_2  # t_3

    # second row
    A_i[1, 0] = 0  # R_11
    A_i[1, 1] = 0  # R_12
    A_i[1, 2] = 0  # R_13
    A_i[1, 3] = -x_1  # R_21
    A_i[1, 4] = -y_1  # R_22
    A_i[1, 5] = -1  # R_23
    A_i[1, 6] = x_1 * y_2  # R_31
    A_i[1, 7] = y_1 * y_2  # R_32
    A_i[1, 8] = y_2  # R_33
    A_i[1, 9] = 0  # t_1
    A_i[1, 10] = -1  # t_2
    A_i[1, 11] = y_2  # t_3

    return A_i


def creat_A(ir_list, rgb_list):
    A = np.empty([12, 12], dtype=float)
    for i in range(len(ir_list)):
        A_i = creat_A_i(ir_list[i], rgb_list[i])
        A[i * 2:(i + 1) * 2, :] = A_i

    return A


'''
1---
ir: (241,270),(382,270),(530,364),(588,116),(787,77),(802,361)
rgb: (174,324),(301,324),(446,407),(489,184),(678,141),(688,399)

2---
ir: (238,275),(375,274),(530,361),(580,118),(782,76),(804,359)
rgb: (173,324),(299,323),(444,405),(489,182),(676,141),(686,397)
'''

if __name__ == '__main__':

    ir_list = [(241, 270), (382, 270), (530, 364), (588, 116), (787, 77),
               (802, 361)]
    rgb_list = [(174, 324), (301, 324), (446, 407), (489, 184), (678, 141),
                (688, 399)]

    A = creat_A(ir_list, rgb_list)
    np.savetxt("A.txt", A, delimiter=',')

    ATA = np.matmul(A.T, A)
    e_value, e_vector = np.linalg.eig(ATA)
    np.savetxt("e_value.txt", e_value, delimiter=',')
    np.savetxt("e_vector.txt", e_vector, delimiter=',')
    np.savetxt("e_vector_last.txt", e_vector[:,-1], delimiter=',')
    
    print(np.matmul(A, e_vector[:,-1]))

    U, D, V = np.linalg.svd(A)
    np.savetxt("SVD_V.txt", V, delimiter=',')
    np.savetxt("SVD_V_last.txt", V[-1,], delimiter=',')
    print(np.matmul(A, V[-1,]))
