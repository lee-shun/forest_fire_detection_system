import numpy as np
#  kalman_filter.py
def predict(self, mean, covariance):
    """Run Kalman filter prediction step.
    
    Parameters
    ----------
    mean: ndarray, the 8 dimensional mean vector of the object state at the previous time step.
    covariance: ndarray, the 8x8 dimensional covariance matrix of the object state at the previous time step.
 
    Returns
    -------
    (ndarray, ndarray), the mean vector and covariance matrix of the predicted state. 
     Unobserved velocities are initialized to 0 mean.
    """
    std_pos = [
        self._std_weight_position * mean[3],
        self._std_weight_position * mean[3],
        1e-2,
        self._std_weight_position * mean[3]]
    std_vel = [
        self._std_weight_velocity * mean[3],
        self._std_weight_velocity * mean[3],
        1e-5,
        self._std_weight_velocity * mean[3]]
    
    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))  # 初始化噪声矩阵Q
    mean = np.dot(self._motion_mat, mean)  # x' = Fx
    covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov  # P' = FPF(T) + Q
 
    return mean, covariance