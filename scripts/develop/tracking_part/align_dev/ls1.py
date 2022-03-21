import numpy as np
from scipy.optimize import leastsq

def func1(params, x, y):
    R11, R12, R13, t1 = params[0], params[1], params[2], params[3]
    residual = y - (R11 * x[0] + R12 * x[1] + R13 * 1 + t1 * 1)
    return residual

x = np.random.random((2, 1))

params = [0, 0, 0, 0]
result = leastsq(func1, params, (x, y))
R11, R12, R13, t1 = params[0][0], params[0][1], params[0][2], params[0][3]
yfit = R11 * x[0] + R12* x[1] + R13 + t1