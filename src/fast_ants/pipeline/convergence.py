"""Shared ITK-style host convergence; original NumFOCUS/Apache-2.0 rules."""
from collections import deque
import numpy as np
class WindowConvergence:
    """ITK's degree-1 scattered B-spline profile, not adjacent loss differences.

    Total energy includes all samples since reset. The fitter uses float
    basis/weight arrays even when energy samples are double. Its window is
    specialized to two linear control points, preserving the source formula.
    """

    def __init__(self, window=10):
        if int(window) != window or window < 2:
            raise ValueError("Convergence window must be an integer >= 2")
        self.window = int(window)
        self.values = deque(maxlen=self.window)
        self.total_energy = 0.0

    def add(self, value):
        if not np.isfinite(value):
            raise ValueError("Convergence requires finite energy values")
        self.values.append(float(value))
        self.total_energy += abs(float(value))
        if len(self.values) < self.window:
            return np.finfo(np.float64).max
        if self.total_energy == 0:
            return 0.0
        omega = np.zeros(2, np.float32)
        delta = np.zeros(2, np.float64)
        epsilon = np.float32(.1*np.float32(.001))
        for i, energy in enumerate(self.values):
            point = np.float32(i)/np.float32(self.window-1)
            if abs(point-np.float32(1)) <= epsilon:
                point = np.float32(1)-epsilon
            weights = np.array([np.float32(1-abs(float(point))),
                                np.float32(1-abs(float(np.float32(point-np.float32(1)))))], np.float32)
            squared_sum = np.float32(weights[0]*weights[0]+weights[1]*weights[1])
            for j in range(2):
                omega[j] = np.float32(omega[j]+weights[j]*weights[j])
                factor = np.float32(weights[j]*weights[j]*weights[j]/squared_sum)
                delta[j] += energy/self.total_energy*float(factor)
        return -float(delta[1]/float(omega[1])-delta[0]/float(omega[0]))

