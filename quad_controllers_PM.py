#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 06:37:23 2021

@author: raduefb
"""
import numpy as np
import pyomo.environ as pyo
import mosek
from quad_trajectory_PM import set_point, circular_traj, liniar_traj
from quad_properties_PM import *

def quad_CFTOC(des_traj, ref_params, quadrotor, N_CFTOC, k_cur, freq, x0 , u0):
    from pyomo.opt import SolverStatus, TerminationCondition
    # des_traj: 0 for set_point, 1: for circular_traj
    # quadrotor: quad object of type quad_classs
    # N_CFTOC: finite time horizon
    # k_cur: starting time step
    # freq: frequency controller is run at
    # x0: initial condition for states
    # u0: last input (to ensure continuity)

    
    # Initialize optimization problem
    model = pyo.ConcreteModel()
    model.N = N_CFTOC
    model.TS = 1/freq
    model.nx = np.size(quadrotor.A, 0)
    model.nu = np.size(quadrotor.B, 1)
    
    # Length of finite optimization problem
    model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )  
    model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
    model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True )
    
    # Create pyomo objects for the linear system description
    model.A = quadrotor.A
    model.B = quadrotor.B
    model.C = quadrotor.C
    
    # Create pyomo objects for the cost matrices
    model.Q = quadrotor.Q
    model.Qf = quadrotor.Qf
    model.R = quadrotor.R
    model.dR = quadrotor.dR
    
    # Create state and input variables trajectory
    model.x = pyo.Var(model.xIDX, model.tIDX)
    model.u = pyo.Var(model.uIDX, model.tIDX)
    
    # Import tracking trajectory for current time and horizon
    stages = np.linspace(k_cur, k_cur+N_CFTOC-1, N_CFTOC) 
    
    if (des_traj == 1):
        ref_traj = circular_traj(stages, freq, ref_params)
    elif (des_traj == 0):
        ref_traj = set_point(stages, freq, ref_params)
    else:
        ref_traj = liniar_traj(stages, freq, ref_params)

    # Objective
    def total_cost(model):
        costX = 0.0
        costU = 0.0
        costdU = 0.0
        costTerminal = 0.0
        for t in model.tIDX:
            # STAGE COST ON STATES
            for i in model.xIDX:
                for j in model.xIDX:
                    if t < model.N-1:
                        costX += (model.x[i, t] - ref_traj[i,t]) * model.Q[i, j] * (model.x[j, t] - ref_traj[j,t])

            for i in model.uIDX:
                for j in model.uIDX:
                    # STAGE COST ON INPUTS
                    if t < model.N:
                        costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
                    
                # STAGE COST ON RATE OF INPUT CHANGE
                if t < model.N-1:
                    costdU += model.dR[i,i] * (model.u[i, t+1] - model.u[i, t])**2   
                    
        # Update stage cost on rate of input change to incorporate first input      
        for i in list(range(0,3)):
            costdU += model.dR[i,i] * (model.u[i, 0] - u0[i])**2  
        
        # TERMINAL COST ON STATES
        for i in model.xIDX:
            for j in model.xIDX:
                costTerminal += (model.x[i, model.N-1] - ref_traj[i,model.N-1]) * model.Qf[i, j] * (model.x[j, model.N-1] - ref_traj[j,model.N-1])
        
        return costX + costU + costdU + costTerminal
    model.cost = pyo.Objective(rule = total_cost, sense = pyo.minimize)
    
    # System Constraints
    def linear_system_description(model, i, t):
        return  model.x[i, t+1] - (model.x [i, t] + model.TS * (sum(model.A[i, j] * model.x[j, t] for j in model.xIDX)
                + sum(model.B[i, j] * model.u[j, t] for j in model.uIDX) + model.C[i])) == 0 if t < model.N else pyo.Constraint.Skip
    model.equality_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=linear_system_description)
    
    # Initial Conditions Constraints
    model.initial_constraints = pyo.Constraint(model.xIDX, rule=lambda model,i: model.x[i,0]==x0[i])
    
    # Input Constraints
    model.input_constraints1 = pyo.Constraint(model.uIDX, model.tIDX, rule=lambda model,i,t: model.u[i,t]<=quadrotor.uU[i])
    model.input_constraints2 = pyo.Constraint(model.uIDX, model.tIDX, rule=lambda model,i,t: model.u[i,t]>=quadrotor.uL[i])

    # Initialize MOSEK solver and solve optimization problem
    solver = pyo.SolverFactory("mosek")
    results = solver.solve(model)
    
    # Check if solver found a feasible, bounded, optimal solution
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        feas = True
        xOpt = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
        uOpt = np.asarray([model.u[:,t]() for t in model.tIDX]).T
        print("MPC problem ", k_cur, " solved!")
    else:
        feas = False
        xOpt = 999
        uOpt = 999

      
    return feas, xOpt, uOpt
