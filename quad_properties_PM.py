#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 02:38:02 2021

@author: raduefb
"""
import numpy as np
import scipy.linalg

# Create quadrotor class
class quad_class_PM:
    def __init__(self, mass, A, B, C, Q, Qf, R, dR, uL, uU):
        self.mass = mass # Mass (kg)
        self.A = A # xdot = Ax + Bu + C
        self.B = B # xdot = Ax + Bu + C
        self.C = C # xdot = Ax + Bu + C
        self.Q = Q # see J_k below
        self.Qf = Qf # see Jf below
        self.R = R # see Jinput below
        self.dR = dR # see Jdinput below        
        self.uL = uL # uL < u < uU
        self.uU = uU
        self.g = 9.81 # (m/s^2)


# Create multiple quadrotor objects
def falcon_white(freq = 100.):
    m = 0.75 # (kg)
    g = 9.81 # (m/s^2)
    
    # xdot = Ax + Bu + C (linear system description)
    A = np.zeros((6,6))
    A[3,0] = 1; A[4,1] = 1; A[5,2] = 1
    B = np.zeros((6,3))
    B[0,0] = 1/m; B[1,1] = 1/m; B[2,2] = 1/m
    C = np.array([0, 0, -g, 0, 0, 0])
    
    # J_k = (x_k - x_k_ref).T * Q * (x_k - x_k_ref) (stage or running cost matrix on states)
    # First three terms are vx,vy,vz, last three are x,y,z
    Q = 1 * np.diag(np.array([3,3,6,4,4,8]))
    # Q = np.diag([20., 20., 20., 20., 20., 10])

    Ak = np.eye(6) + A*(1.0/freq)
    Bk = B*(1./freq)

    R = 2 * np.diag(np.array([1,1,1]))
    Qf = scipy.linalg.solve_discrete_are(Ak, Bk, Q, R)
    print(Qf)

    # Jf = (x_N - x_N_ref).T * Qf * (x_N - x_N_ref) (terminal cost on states)
    # Qf = 280 * np.diag(np.array([0.23,0.23,2,8,8,12]))
    # Qf = np.array([[ 788.68377391,    0.        ,    0.        ,  682.47613546,           0.        ,    0.        ],
    #    [   0.        ,  788.68377391,    0.        ,    0.        ,         682.47613546,    0.        ],
    #    [   0.        ,    0.        ,  829.42578737,    0.        ,           0.        ,  966.01088581],
    #    [ 682.47613546,    0.        ,    0.        , 4642.48411586,           0.        ,    0.        ],
    #    [   0.        ,  682.47613546,    0.        ,    0.        ,        4642.48411586,    0.        ],
    #    [   0.        ,    0.        ,  966.01088581,    0.        ,           0.        , 6908.87321504]])

    # Jinput = (u_k).T * R * (u_k) (stage cost on inputs)

    # Jdinput = (u_k+1 - u_k).T * dR * (u_k+1 - u_k) (stage cost on input change)
    dR = 0.5 * np.diag(np.array([5,5,4]))
    # u constraints
    uU = np.array([10, 10, 20])
    uL = np.array([-10, -10.0, -9.81*0.75])
    return m, A, B, C, Q, Qf, R, dR, uL, uU