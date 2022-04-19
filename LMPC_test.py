"""
Run this script to test MPC and LMPC position controllers in Python.
"""
import sys
import numpy as np
import pdb
import matplotlib.pyplot as plt
import copy
import pickle
import time as tm

sys.path.append('Optimal Control')
sys.path.append('Classic Control')
sys.path.append('Trajectory')

from CFTOC_pyomo import CFTOC
from PID import PD
from LMPC import LMPC
from Trajectory import Trajectory


class point_mass_dynamics:
    def __init__(self):
        # System properties
        self.m = 0.75  
        self.g = 9.81
    
    	# Define continous-time dynamics
        # xdot = A*x + B*u + C
        self.A = np.zeros((6,6))
        self.A[3,0] = 1; self.A[4,1] = 1; self.A[5,2] = 1
        self.B = np.zeros((6,3))
        self.B[0,0] = 1/self.m; self.B[1,1] = 1/self.m; self.B[2,2] = 1/self.m
        self.C = np.array([0, 0, -self.g, 0, 0, 0])
    
        # Define discrete-time dynamics
        # x_{k+1} = Ak* x_k + B_k*u_k + C_k
        self.freq = 100
        self.A_k = (np.eye(6) + self.A/self.freq)
        self.B_k = self.B/self.freq
        self.C_k = self.C/self.freq

class cost_matrices(object):
    def __init__(self, dynamics):
        import scipy.linalg 
        # J_k = (x_k - x_k_ref).T * Q * (x_k - x_k_ref) (stage or running cost matrix on states)
        self.Q = np.diag([3., 3., 6., 4., 4., 8.])
        # Jinput = (u_k).T * R * (u_k) (stage cost on inputs)
        self.R = 2 * np.diag(np.array([1,1,0.0002]))
        # Jf = (x_N - x_N_ref).T * Qf * (x_N - x_N_ref) (terminal cost on states)
        self.Qf = scipy.linalg.solve_discrete_are(dynamics.A_k, dynamics.B_k, 
                                                  self.Q, self.R)
        # Jdinput = (u_k+1 - u_k).T * dR * (u_k+1 - u_k) (stage cost on input change)
        self.dR = 0.2 * np.diag(np.array([16,16,2]))

def main():
    
    # Define linear dynamics
    dynamics = point_mass_dynamics()
 
    # Define optimal control costs
    costs = cost_matrices(dynamics)

    # Select desired reference trajectory (0 - circular)(1 - setPoint)(2 - customTrajectory)
    trajectory = Trajectory(dynamics.freq, 2)
    	
	# Initial Condition
    x0 = np.array([0,0,0,0,0,0.6])

    # Initial conditions on inputs are set to allow smooth input changes
    u0 = np.array([0,0,dynamics.m*dynamics.g]).T
    
	# Initialize MPC object
    N_CFTOC = 7
    CFTOC_MPC  = CFTOC(N_CFTOC, dynamics, costs)

    # Initialize PID object
    PD_controller = PD(dynamics)
    
    # =========================================================================
	# ===============  CONSTRUCT FEASIBLE SOLUTION USING PID  =================
	# =========================================================================
    print("Starting PID...")
    xcl_PID = [x0]
    ucl_PID = []
    xt           = x0
    ut           = u0
    PID_Time   = 0

    while True:
        # Read system state
        xt = xcl_PID[PID_Time]

        # Compute trajectory preview
        preview = trajectory.get_reftraj(PID_Time, 1)

        # Solve PID problem
        PD_controller.solve_PD(xt, np.reshape(preview,9)) 

        # Read PID input
        ut = PD_controller.F_command
        ucl_PID.append(ut)

        # Apply input to system
        xcl_PID.append(CFTOC_MPC.model(xcl_PID[PID_Time], ut))
        PID_Time += 1

        # Stop running when finish line is crossed 
        if trajectory.crossedFinish(np.reshape((xcl_PID)[-1],(6,1)), PID_Time) == True:
            break

        print("PID #", PID_Time-1)

    # PLOT PID RESULTS
    x_array = np.array(xcl_PID)
    plot_trajectories(x_array)

    print("PID Terminated!")

	# =========================================================================
	# ===============  CONSTRUCT FEASIBLE SOLUTION USING MPC  =================
	# =========================================================================
    print("Starting MPC...")

    xcl_feasible = [x0]
    ucl_feasible = []
    xt           = x0
    ut           = u0
    MPC_Time   = 0
    
	# Solve CTOFC for MPC
    while True:
        # Read system state
        xt = xcl_feasible[MPC_Time] 
        
        # Compute trajectory preview
        preview = trajectory.get_reftraj(MPC_Time, N_CFTOC)[0:6] # No need for acceleration here [6:9]

        # Solve MPC CFTOC problem
        CFTOC_MPC.solve_MPC(xt, ut, preview) 
        
		# Read input
        ut = CFTOC_MPC.u_pred[:,0]
        ucl_feasible.append(ut)
        
        # Run system dynamics
        xcl_feasible.append(CFTOC_MPC.model(xcl_feasible[MPC_Time], ut))
        MPC_Time += 1
        
        print("CFTOC #", MPC_Time-1)
        
        # TROUBLESHOOT
        if MPC_Time==1300:
            plot_trajectories(np.array(xcl_feasible))

        # Stop running when finish line is crossed from below
        if trajectory.crossedFinish(np.reshape((xcl_feasible)[-1],(6,1)), MPC_Time) == True:
            break
    
    # PLOT MPC RESULTS
    x_array = np.array(xcl_feasible)
    plot_trajectories(x_array)
    
    print("MPC Terminated!")

	# =========================================================================
	# =============================   Run LMPC   ==============================
	# =========================================================================
    
	# Initialize LMPC objects
    N_LMPC = 9 # horizon length
    CFTOC_LMPC = CFTOC(N_LMPC, dynamics, costs) # CFTOC solved by LMPC
    lmpc = LMPC(CFTOC_LMPC, N_LMPC, MPC_Time+1) # Initialize the LMPC
    lmpc.addTrajectory(xcl_feasible, ucl_feasible) # Add feasible trajectory to the safe set
    lmpc.goal_pt = trajectory.get_goalpoint() # Import coordinates of goal

    totalIterations = 10 # Number of iterations to perform

    print("Starting LMPC...")
    
    # Run LMPC
    for it in range (0,totalIterations):
        # Set initial conditions to the final states from the previous iteration
        xcl = [lmpc.SS[it][:,-1]] 
        ucl =[]
        ut = lmpc.uSS[it][:,-1]
        
		# Solve CFTOC for LMPC
        while True:
			# Read measurement
            xt = xcl[lmpc.t_index] 

            # Check if any predicted states are over the finish line
            lmpc.closest_pt_traj = trajectory.get_closestpoint(xt)
            if (lmpc.xPred != []):
                lmpc.pred_over_finish = trajectory.crossedFinish(lmpc.xPred, lmpc.t_index)

			# Solve CFTOC
            lmpc.solve(xt, ut) 

			# Read and apply optimal input to the system
            ut = lmpc.uPred[:,0]
            ucl.append(ut)
            xcl.append(lmpc.cftoc.model(xcl[lmpc.t_index], ut))

            # Add new state to past iteration SS to enable loop-around 
            lmpc.addtoPrevSS(xcl[-1])
            lmpc.t_index += 1

            print("LMPC ",it+1,"|",lmpc.t_index-1)
                
            # Quit when finish line is reached
            if trajectory.crossedFinish(np.reshape(xcl[-1],(6,1)), lmpc.t_index) == True:
                del xcl[0]
                x_LMPC = np.array(xcl)
                plot_trajectories(x_LMPC)
                break

        print("LMPC Lap #", it+1 , " complete!")

		# Add trajectory to update the safe set and value function
        lmpc.addTrajectory(xcl, ucl)

    print("Lap Times: ",lmpc.LapTime)   
	# Save the lmpc object
    #filename = 'lmpc_object.pkl'
    #filehandler = open(filename, 'wb')
    #pickle.dump(lmpc, filehandler)
   
    #plot_trajectories(x_array)
    
    #print("Completion times: ", completion_time)

    
def plot_trajectories(x):
    
    # 2D X-Y plot
    fig = plt.figure(1, figsize=(18,6))
    plt.subplot(1,3,1)
    plt.plot(  x[:,3]  , x[:,4],'blue')
    plt.scatter(x[0,3] , x[0,4], s =15, c='red')
    plt.title('X vs. Y')

    # 2D X-Z plot
    plt.subplot(1,3,2)
    plt.plot(  x[:,3]  , x[:,5],'blue')
    plt.scatter(x[0,3] , x[0,5], s =15, c='red')
    plt.title('X vs. Z')

    # 2D Y-Z plot
    plt.subplot(1,3,3)
    plt.plot(  x[:,4]  , x[:,5],'blue')
    plt.scatter(x[0,4] , x[0,5], s =15, c='red')
    plt.title('Y vs. Z')
    plt.legend(['Reference Trajectory','Open Loop (CFTOC)','Closed Loop (MPC)','Initial Condition (x0)'])
    plt.show()

    # 2D X-T plot
    fig = plt.figure(2, figsize=(18,6))
    # hor = np.zeros((N+1))
    # for i in range((N+1)):
    #     hor[i] = i
    plt.subplot(2,3,1)
    plt.plot( x[:,3],'blue')
    plt.title('X vs. T')

    # 2D X-T plot
    fig = plt.figure(2, figsize=(18,6))
    # hor = np.zeros((N+1))
    # for i in range((N+1)):
    #     hor[i] = i
    plt.subplot(2,3,2)
    plt.plot( x[:,4],'blue')
    plt.title('Y vs. T')
        
    # 2D X-T plot
    fig = plt.figure(2, figsize=(18,6))
    # hor = np.zeros((N+1))
    # for i in range((N+1)):
    #     hor[i] = i
    plt.subplot(2,3,3)
    plt.plot( x[:,5],'blue')
    plt.title('Z vs. T')
    plt.show()    

if __name__== "__main__":
    main()

