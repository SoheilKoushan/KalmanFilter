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
