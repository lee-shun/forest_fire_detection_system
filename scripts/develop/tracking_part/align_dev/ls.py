'''
problem description
In RGB camera, the pixel coordinates could be written as [u_3, v_3, 1];
In IR camera, the pixel coordinates cold be writen as [u_1, v_1, 1], also the depth could be captured as [z];
In the trasfer function, there are 12 parameters unknown, and there might be zeros.
Therefore, least squares are assumed to be applied for estimation regression.

R.shape() = [3, 3], t.shape() = [3, 1]
q_3.shape() = [3, 1], q_1.shape() = [3, 1], z_3.shape() = [1, 1]
q_3 = R * q_1 + t * 1/z_3 
'''
import numpy as np
from scipy.optimize import leastsq

dataset = 100
# for i in range(dataset):
q_ir_2d = np.random.random((2, 1))
q_irt = np.vstack((q_ir_2d, np.ones((2, 1))))
# q_ir_t = np.squeeze(q_ir_t)
q_rgb_2d = np.random.random((2, 1))
q_rgb = np.vstack((q_rgb_2d, 1))
print('q_irt', q_irt)
print('q_irt.shape', q_irt.shape)
# i += 1
R_init = np.random.rand(3, 4)
np.dot(R_init, q_irt)
print("dot pass!")

Errorfunc = lambda R, q_irt, q_rgb: np.dot(R, q_irt) - q_rgb

R_init = np.random.rand(3, 4)
print('R_init.shape', R_init.shape)

a = np.dot(R_init, q_irt)
print(a.shape)

Rt_final, success = leastsq(Errorfunc, R_init, args = (q_irt, q_rgb))
print('final params', Rt_final)




    
