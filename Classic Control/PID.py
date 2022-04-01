import numpy as np

class PD():
    def __init__(self, dynamics):
        self.g = dynamics.g
        self.m = dynamics.m
        self.kp = np.array([4, 4, 8])
        self.kv = np.array([3, 3, 6])
        self.e3 = np.array([0,0,1])  

        # Hardcode reference acceleration to be 0 for PID control purposes
        self.ad = np.zeros(3)

    def solve_PD(self, xt, preview):
        # Compute position and velocity errors
        err_x = xt[3:] -preview[3:]
        err_v = xt[0:3]-preview[0:3]

        # Compute feedback control input
        Fff = self.m*(self.g*self.e3 + self.ad)
        Fpd = -self.kp*err_x - self.kv*err_v
        self.F_command = Fff+Fpd