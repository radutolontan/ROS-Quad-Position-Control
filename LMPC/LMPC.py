
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
    
    def addTrajectory(self, x, u):

        # Set/Reset Time Index
        self.t_index = 0

        # If LMPC is running
        if self.SS!=[]:
            # Store completion time
            self.LapTime.append(self.t_index+1)

		# Add the feasible trajectory x and the associated input sequence u to the safe set
        self.SS.append(np.array(copy.copy(x)).T)
        self.uSS.append(np.array(copy.copy(u)).T)

		# Compute and store the cost associated with the feasible trajectory
        cost = self.computeCost_rollout(x, u)
        self.Qfun.append(np.array(cost))

		# Initialize zVector
        self.zt = np.array(x[self.cftoc.N])

		# Augment iteration counter and print the cost of the trajectories stored in the safe set
        self.it = self.it + 1

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
        
    def computeCost(self, x, u):
        # Import reference trajectory and compute error
        ref_traj = self.cftoc.get_reftraj(0, len(x)-1)
        err = np.array(x) - ref_traj.T
        
		# Compute the cost in a DP like strategy: start from the last point x[len(x)-1] and move backwards
        for i in range(0,len(x)):
            idx = len(x)-1 - i
            if i == 0:
                cost = [np.dot(np.dot(err[idx],self.Q),err[idx])]
            else:
                cost.append(np.dot(np.dot(err[idx],self.Q),err[idx]) + # Cost on states
                            np.dot(np.dot(u[idx],self.R),u[idx]) + # Cost on inputs
                            np.dot(np.dot(u[idx]-u[idx-1],self.dR),u[idx]-u[idx-1]) + # Cost on rate of change on states
                            cost[-1]) # Previous cost
		
		# Finally flip the cost to have correct order
        return np.flip(cost).tolist()

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

    def crossedFinish(self, states):
        """
        Function takes in a vector of states (so a matrix) and returns a T/F vector indicating which
        column in the matrix corrresponds to a state beyond the finish line. 

        states: veector of states (initialized as xPred)
        """
        # Initialize storage
        crossLinevec = np.zeros(states.shape[1])
        
        # For the first time CFTOC is solved in an iteration, return zeros
        if states==[]:
            return crossLinevec
        # Otherwise check to see if state in each column has passed the f_line
        else:
            for jj in range(states.shape[1]):
                # Time index (t_index) is used to ensure the states at the beginning are not picked up
                if all((states[1,jj]>0, states[4,jj] >= 0.0, self.t_index > 300)) == True:
                    crossLinevec[jj] = 1
            return crossLinevec

    def SS_future_N(self, xt, completion_time):
        """
        Return subset of safe set and corresponding value function for next 'n' closest terms
        , after discarding the last 'p' iterations

        Parameters
        ----------
        SS_full : full safe set array
        Qfun_full : full value function array
        xt : current state
        
        Returns
        -------
        SS_subset : subset of safe set closest to xt
        Qfun_subset : subset of added value function
        """
        # Import full SS
        SS_full = self.SS_vector
        Qfun_full = self.Qfun_vector[0]  
        
        # If enough data exists to take a subset of "p" iterations
        if (self.it > self.p):
            # Discard all iterations other then last "p"
            SS_full = SS_full[:, -sum(completion_time[-self.p:]):]
            Qfun_full = Qfun_full[-sum(completion_time[-self.p:]):]    
            # Keep list of desired iterations only
            completion_time = completion_time[-self.p:]   
        else:
            return(self.SS_closest_N(xt,completion_time))

        xt = np.reshape(xt,(6, 1))
        SS_subset = xt
        Qfun_subset = [1]  

        for it in range(np.min((self.it, self.p))):
            # Select SS associated with current iteration
            SS_it = SS_full[:, 0:completion_time[it]]
            Qfun_it = Qfun_full[0:completion_time[it]]

            # Remove current interation from SS_full and Qfun_full
            SS_full = SS_full[:, completion_time[it]:]
            Qfun_full = Qfun_full[completion_time[it]:]        
            
            # Return index of column w. smallest squared error 2-norm
            index = np.argmin(np.linalg.norm(SS_it - xt, axis=0))     

            if (index + self.n <= SS_it.shape[1]):
                # Append first n columns after index to SS_subset
                SS_subset = np.hstack((SS_subset, SS_it[:, index:index + self.n]))  
                Qfun_subset = np.hstack((Qfun_subset, Qfun_it[index:index + self.n]))      
            else:
                # Append as many columns as available from 'it' iteration
                SS_subset = np.hstack((SS_subset, SS_it[:, index:SS_it.shape[1]]))
                Qfun_subset = np.hstack((Qfun_subset, Qfun_it[index:len(Qfun_it)]))
                # Append the remaining columns from 'it+1' iteration
                SS_subset = np.hstack((SS_subset, SS_full[:, 0:index + self.n - SS_it.shape[1]]))    
                Qfun_subset = np.hstack((Qfun_subset, Qfun_full[0:index + self.n - len(Qfun_it)]))    
            
            # Remove first column/element used only for stacking purposes
            SS_subset = SS_subset[:,1:]
            Qfun_subset =  Qfun_subset[1:]

            # Return Qfun_subset to apropriate shape
            Qfun_subset = np.reshape(Qfun_subset, (1, len(Qfun_subset)))

        return SS_subset, Qfun_subset

    def SS_closest_N(self, xt, completion_time):
        """
        Return subset of safe set and corresponding value function closest to 
        current state, after discarding all but the last 'p' iterations

        Parameters
        ----------
        xt : current state
        completion_time : vector which holds number of CFTOC iterations in each solution
        
        Returns
        -------
        SS_subset : subset of safe set closest to xt
        Qfun_subset : subset of added value function

        """
        # Import full SS
        SS_full = self.SS_vector
        Qfun_full = self.Qfun_vector[0]

        if (self.it > self.p):
            # Discard all iterations other then last "p" if enough data exists
            SS_full = SS_full[:,-sum(completion_time[-self.p:]):]
            Qfun_full = Qfun_full[-sum(completion_time[-self.p:]):]

        err_2_norm = []
        n = SS_full.shape[1]
        
        # Compute 2-norm of error for each entry of safeset
        for i in range(n):
            col = SS_full[:,i]
            err_2_norm.append(abs(la.norm(col - xt)))
            
        # Sort first n_points elements of err_2_norm
        res = sorted(range(len(err_2_norm)), key = lambda sub: err_2_norm[sub])[:self.n]
        
        # Append first n_points elements of SS and Qfun to subset entities
        SS_subset = np.array([SS_full[:,res[0]]]).T
        Qfun_subset = np.array([Qfun_full[res[0]]])
        for i in range(1,len(res)):
            SS_subset = np.concatenate((SS_subset, np.array([SS_full[:,res[i]]]).T), axis=1)
            Qfun_subset = np.concatenate((Qfun_subset, np.array([Qfun_full[res[i]]])), axis=0)
         
        # Put Qfun_subset into appropriate shape
        Qfun_subset = np.reshape(Qfun_subset, (1, self.n))
        
        return SS_subset, Qfun_subset

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
            elif (np.any(self.crossedFinish(self.xPred)) == False): # used for all states that generate predictions before finish line
                Qfun_subset = np.hstack((Qfun_subset, Qfun_it[indexSSandQfun]))
            elif curr_it < self.it - 1:
                Qfun_subset = np.hstack((Qfun_subset, Qfun_it[indexSSandQfun] + Qfun_it[0]))
            else:
                predCurrLap = self.N_LMPC - sum(self.crossedFinish(self.xPred))
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
        self.cftoc.solve(xt, ut, self.t_index, SS_selected, Qfun_selected)

        # Troubleshoot Plotting
        if (self.t_index>=540):
            plt.plot(SS_selected[3,:],SS_selected[4,:],'magenta')
            plt.plot(self.xPred[3,:],self.xPred[4,:],'green')
            #plt.plot(self.SS[0][3,:],self.SS[0][4,:],'red')
            plt.legend(['SS_Selected','Previous xPred'])
            plt.show()


		# Update predicted trajectory
        self.xPred= self.cftoc.x_pred
        self.uPred= self.cftoc.u_pred


