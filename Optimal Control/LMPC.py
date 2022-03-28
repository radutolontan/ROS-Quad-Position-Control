
from pickle import TRUE
import numpy as np
from numpy import True_, linalg as la
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pdb
import copy
import itertools

class LMPC(object):

    def __init__(self, cftoc, N_LMPC, MPC_LapTime):
		# Initialization
        self.t_index = 0
        self.LapTime = [MPC_LapTime]
        self.cftoc = cftoc
        self.SS    = []
        self.uSS   = []
        self.Qfun  = []
        self.N_LMPC = N_LMPC
        self.Q = cftoc.Q
        self.R = cftoc.R
        self.dR = cftoc.dR
        self.it    = 0
        self.xPred = []

        # SS subset parameters
        self.p = 4 # build SS subset from last 'p' iterations only
        self.n = 40 # return 'n' closest terms (WARNING: SS_future_n returns 'n' states for each 'p' iteration)
        self.delta = 0.2 # used to define state constraints (as max deviation delta)
        self.delta_goal = 0.05 # used to define nearness to goal required for optimal time cost
        
        # Vector which holds 1s for predicted states (xPred) over the finish line and 0s otherwise
        self.pred_over_finish = 0 

        # Array which holds closest point on reference trajectory to current state
        self.closest_pt_traj = 0

        # Array which contains the x,y,z coordinates of the goal (will be imported as appropriate to the trajectory)
        self.goal_pt = 0

    def addTrajectory(self, x, u):

        # Set/Reset Time Index
        self.t_index = 0

        # If LMPC is running
        if not(self.SS==[]) :
            # Store completion time
            self.LapTime.append(self.t_index+1)

		# Add the feasible trajectory x and the associated input sequence u to the safe set
        self.SS.append(np.array(copy.copy(x)).T)
        self.uSS.append(np.array(copy.copy(u)).T)

		# Compute and store the cost associated with the feasible trajectory
        cost = self.computeCost_rollout(x, u)
        self.Qfun.append(np.array(cost))

		# Augment iteration counter and print the cost of the trajectories stored in the safe set
        self.it = self.it + 1

        # Reset values
        self.xPred = []

        #print("Trajectory added to the Safe Set. Current Iteration: ", self.it)
        #print("Performance stored trajectories: \n", [self.Qfun[i][0] for i in range(0, self.it)])

    def addtoPrevSS(self, x):
        """
        at iteration j add the current point to SS, and Qfun of the previous iteration
        Arguments:
            x: current state
        """
        self.SS[self.it - 1]  = np.hstack((self.SS[self.it - 1], np.reshape(x,(6,1))))
        self.Qfun[self.it - 1] = np.hstack((self.Qfun[self.it - 1], self.Qfun[self.it - 1][-1]-1))
        
    def computeCost_rollout(self, x, u):
        """compute roll-out cost
        Arguments:
            x: closed-loop trajectory
            u: applied inputs
        """
        x = np.array(x)
        Cost = np.empty((x.shape[0]))
        for jj in range(0, x.shape[0]):
            Cost[jj] = x.shape[0] - jj - 1

        return Cost

    def SS_local_N(self, xt):
        # Initialize storage as non-empty to allow for "append" to be used
        xt = np.reshape(xt,(6, 1))
        SS_subset = xt
        Qfun_subset = [1]
        curr_it = 0

        # If enough data exists to take a subset of "p" iterations
        if (self.it > self.p):
            curr_it = self.it - self.p

        # If more than p iterations exist, create local subset using last p iterations
        # ,otherwise use all existing iterations
        for jj in range(np.min((self.it, self.p))):   
            # Select SS and Qfun associated with desired iteration
            SS_it = self.SS[curr_it]
            Qfun_it = self.Qfun[curr_it]      
            
            slack = 0
            # When a new iteration is starting, the MinNorm might pick up the last states of the 
            # previous iteration if slack is not used
            if self.t_index<=10 :
                slack = 40

            # Return index of column w. smallest squared error 2-norm of POSITION vector ([3:6])
            MinNorm = np.argmin(np.linalg.norm(SS_it[3:6,0:self.LapTime[curr_it]-slack] - xt[3:6], 1, axis=0))     
            
            # Compute indices being returned
            if (MinNorm - self.n/2 >= 0):
                indexSSandQfun = range(-int(self.n/2) + MinNorm, int(self.n/2) + MinNorm + 1)
            else:
                indexSSandQfun = range(MinNorm, MinNorm + int(self.n))

            # Select corresponding SS subset
            SS_subset  = np.hstack((SS_subset,SS_it[:, indexSSandQfun]))

            # Select corresponing Q_fun subset and adjust terms after crossing finish line
            if self.xPred == []: # used on first CFTOC of new lap when xPred does not exist yet
                Qfun_subset = np.hstack((Qfun_subset, Qfun_it[indexSSandQfun]))
            elif (np.any(self.pred_over_finish) == False): # used for all states that generate predictions before finish line
                Qfun_subset = np.hstack((Qfun_subset, Qfun_it[indexSSandQfun]))
            elif curr_it < self.it - 1:
                Qfun_subset = np.hstack((Qfun_subset, Qfun_it[indexSSandQfun] + Qfun_it[0]))
            else:
                predCurrLap = self.N_LMPC - sum(self.pred_over_finish)
                currLapTime = self.t_index
                Qfun_subset = np.hstack((Qfun_subset, Qfun_it[indexSSandQfun] + currLapTime + predCurrLap))

            # Increase current iteration being considered
            curr_it += 1

        # Trim first column/element used only for stacking purposes
        SS_subset = SS_subset[:,1:]
        Qfun_subset =  Qfun_subset[1:]

        # Put Qfun_subset into appropriate shape needed by CFTOC solver
        Qfun_subset = np.reshape(Qfun_subset, (1, len(Qfun_subset)))

        return SS_subset, Qfun_subset

    def solve(self, xt, ut):
			
        # Build local SS subset used for LMPC 
        SS_selected, Qfun_selected = self.SS_local_N(xt)
        
		# Solve the CFTOC 
        self.cftoc.solve_LMPC(xt, ut, self.delta, self.delta_goal, SS_selected, Qfun_selected, self.closest_pt_traj, self.goal_pt)

		# Update predicted trajectory
        self.xPred= self.cftoc.x_pred
        self.uPred= self.cftoc.u_pred


