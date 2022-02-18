#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 16:13:00 2022

@author: raduefb
"""
import numpy as np


def circular_traj(k, freq, r, omega, cruise_height):
    # k - time steps; freq - frequncy (Hz)
    # Create storage and set Z- location to 0
    x = np.zeros((6,int(np.size(k))))
    
    # X - and Y - velocities are parametrized
    x[0,:] = - (omega*r) * np.sin((k/freq) * omega) 
    x[1,:] =   (omega*r) * np.cos((k/freq) * omega)
    # X - and Y - location are parametrized
    x[3,:] =  r * np.cos((k/freq) * omega) 
    x[4,:] =  r * np.sin((k/freq) * omega) 
    # Z - location is constant and equal to a height
    x[5,:] = cruise_height * np.ones(int(np.size(k)))

    return x

def set_point(k, freq, setpoint):
    # k - time step; freq - frequncy (Hz)
    # Create storage for both velocities and locations
    # Set velocities to 0
    x = np.zeros((6,int(np.size(k))))
    # Set X- setpoint, Y- setpoint and Z- setpoint
    x[3,:] = setpoint[0] * np.ones(int(np.size(k)))
    x[4,:] = setpoint[1] * np.ones(int(np.size(k)))
    x[5,:] = setpoint[2] * np.ones(int(np.size(k)))

    return x