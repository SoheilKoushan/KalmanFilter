# Kalman Filter

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

np.random.seed(42)

'''
1. We must simulate a ground truth 2D trajectory and noisy position measurements
2. Derive and implement a discrete kalman filter for a constant velocity
3. Evaluate accuracy versus a baseline (like a moving average using RMSE)
4. Validate statistical assumptions using innovation or residual tests
5. Stretch to handle missing measurements and occasional outliers robustly

Learning goals:
1. Discrete time kinematics
2. state space modelling
3. Matrix calculus lite (Building A, B, H, Q, and R)
4. Probabalistic reasoning (gaussians, covariance, innovations)
5. Practical tuning and diagnostics
'''

@dataclass
class SimConfig:
    dt: float = 0.1
    n_steps: int = 300
    q: float = 0.5 # process noise spectral density
    sigma_meas: float = 2.0 # measurement noise std (pixel/meters)
    miss_rate: float = 0.05 # probability a measurement is missing
    outlier_rate: float = 0.03 # probability a measurement is an outlier
    outlier_std: float = 15.0 # outlier scale

def simulate_truth(cfg: SimConfig):
    t = np.arange(cfg.n_steps) * cfg.dt

    # smooth-ish 2D trajectory with gentle turns
    x = 0.5 * t**2 + 10*np.sin(0.3*t)
    y = 0.3 * t**2 + 8*np.cos(0.27*t + 0.8)

    vx = np.gradient(x, cfg.dt) # take the numerical derivative of x wrt time, where time steps are evenly spaced by dt.
    vy = np.gradient(y, cfg.dt)

    X = np.vstack([x, vx, y, vy]).T # shape (n, 4)
    return t, X

def simulate_measurements(X, cfg: SimConfig):
    n = X.shape[0]
    z = X[:, [0, 2]] # true positions (x, y)
    meas = z + np.random.normal(0, cfg.sigma_meas, size=z.shape)

    # random missing
    '''
    mask is a boolean array (T/F values) used to mark which measurements exist and which don't.
    Put NaN at the positions where mask was False.
    '''
    mask = np.random.rand(n) > cfg.miss_rate
    meas[~mask] = np.nan

    # random outliers
    outliers = np.random.rand(n) < cfg.outlier_rate
    noise_out = np.random.normal(0, cfg.outlier_std, size=(n, 2))
    meas[outliers & mask] += noise_out[outliers & mask]

    return meas, mask