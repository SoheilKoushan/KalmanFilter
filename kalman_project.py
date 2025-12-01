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
4. Probabilistic reasoning (Gaussians, covariance, innovations)
5. Practical tuning and diagnostics


Steps:
1) Simulation utilities
2) Kalman Filter
3) Baseline: moving average
4) Experiment
5) Metrics
6) Consistency via NIS (innovations)
7) Visualizations
'''

# -----------------------
# 1) Simulation utilities
# -----------------------

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

    # take the numerical derivative of x wrt time, where time steps are evenly spaced by dt.
    vx = np.gradient(x, cfg.dt) 
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

# -----------------------
# 2) Kalman Filter
# -----------------------

@dataclass
class KFConfig:
    dt: float # time step between consecutive states
    q: float # process noise intensity
    sigma_meas: float # sd of the measurement noise

def kf_matrices(cfg: KFConfig):
    '''
    This function takes that config and builds the four core matrices of a linear KF.
    A: state transition matrix
    Q: process noise covariance
    H: observation matrix
    R: measurement noise covariance

    state vector: x_k = [x vx y vy]
    for constant velocity with step dt:
    x_k+1 = x_k + v.dt
    v_k+1 =
    '''
    dt = cfg.dt
    A = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1]
    ], dtype=float)

    # Process noise fort constant-velocity (block diag of 1D form)
    q = cfg.q
    Q1 = q * np.array([[dt**4/4, dt**3/2],
                       [dt**3/2, dt**2]])
    Q = np.zeros((4, 4))
    Q[0:2, 0:2] = Q1
    Q[2:4, 2:4] = Q1

    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ], dtype=float)

    R = (cfg.sigma_meas**2) * np.eye(2)
    return A, Q, H, R

class KalmanFilterCV2D:
    def __init__(self, kf_cfg: KFConfig):
        self.A, self.Q, self.H, self.R = kf_matrices(kf_cfg)
        self.I = np.eye(4)
        self.x = None
        self.P = None

    def init(self, z0, sigma0_pos=0.5, sigma0_vel=5.0):
        # initialize from first measurement (or zero if missing)
        if np.any(np.isnan(z0)):
            self.x = np.array([0, 0, 0, 0], dtype=float)
        else:
            self.x = np.array(z0[0], 0, z0[1], 0, dtype=float)
        self.P = np.diag([sigma0_pos**2, sigma0_vel**2,
                          sigma0_pos**2, sigma0_vel**2])

    def predict(self):
        # @ is matrix multiplication
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q
        self.x, self.P = x_pred, P_pred
        return x_pred, P_pred

    def update(self, z):
        if np.any(np.isnan(z)):
            # no measurement - skip update
            return self.x, self.P, None, None
        # outlier gating with Mahalanobis distance (simple robustification)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T @ self.R
        d2 = y.T @ np.linalg.inv(S) @ y # NIS (chi^2 with 2 dof)

        # if too large, treat as missing (gate)
        # Threshold ~ 9.21 for Chi^2 with dof=2 at 99% (tunable)
        if d2 > 9.21:
            return self.x, self.P, y, S
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K@y
        self.P = (self.I - K@self.H) @ self.P
        return self.x, self.P, y, S

# -----------------------
# 3) Baseline: moving average
# -----------------------
def moving_average_positions(meas, window=5):
    out = np.copy(meas)
    for d in range(2):
        series = meas[:, d]
        # forward-fill NaNs for baseline simplicity
        valid = np.where(~np.isnan(series))[0]
        if len(valid) == 0:
            continue
        last = series[valid[0]]
        filled = []
        j = 0
        for i in range(len(series)):
            if j < len(valid) and i == valid[j]:
                last = series[i]
                j += 1
            filled.append(last)
        arr = np.array(filled, dtype=float)
        # moving average (casual)
        kernel = np.ones(window) / window
        arr_ma = np.convolve(arr, kernel, mode='same')
        out[:, d] = arr_ma
    return out

# -----------------------
# 4) Experiment
# -----------------------
cfg = SimConfig()
t, X_true = simulate_truth(cfg)
Z_meas, mask = simulate_measurements(X_true, cfg)

kf_cfg = KFConfig(dt=cfg.df, q=cfg.q, sigma_meas=cfg.sigma_meas)
kf = KalmanFilterCV2D(kf_cfg)
kf.init(Z_meas[0])

X_est, P_est, innov, S_list = [], [], [], []

for k in range(len(t)):
    kf.predict()
    xk, Pk, yk, Sk = kf.update(Z_meas[k])
    X_est.append(kf.x.copy())
    P_est.append(kf.P.copy())
    innov.append(yk if yk is not None else np.array([np.nan, np.nan]))
    S_list.append(Sk)

X_est = np.array(X_est) # (n, 4)
innov = np.array(innov) # (n, 2)

# Baaseline
Z_ma = moving_average_positions(Z_meas, window=7)

