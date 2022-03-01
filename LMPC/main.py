import numpy as np
import pdb
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import pickle
import time as tm

end_time = []

from CFTOC_pyomo import CFTOC
from LMPC import LMPC

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
    
    global end_time
    
    # Define linear dynamics
    dynamics = point_mass_dynamics()
 
    # Define optimal control costs
    costs = cost_matrices(dynamics)

    # Select desired reference trajectory (0 - circular)(1 - setPoint)
    traj = 0
    	
	# Initial Condition
    x0 = np.array([0,0,0,0.68,0.03,0.8])
    
    # Initial conditions on inputs are set to allow smooth input changes
    u0 = np.array([0,0,dynamics.m*dynamics.g]).T
    
	# Initialize MPC object
    N_CFTOC = 7
    CFTOC_MPC  = CFTOC(N_CFTOC, traj, dynamics, costs)
    
	# =========================================================================
	# ===============  CONSTRUCT FEASIBLE SOLUTION USING MPC  =================
	# =========================================================================
    print("Starting MPC...")

    xcl_feasible = [x0]
    ucl_feasible = []
    xt           = x0
    ut           = u0
    time_index   = 0
    
	# Solve CTOFC for MPC
    while True:
        xt = xcl_feasible[time_index] # Read system state
        
        start_time = tm.time()
        CFTOC_MPC.solve(xt, ut, time_index, verbose = False) # Solve CFTOC
        end_time.append(tm.time() - start_time)
        
		# Read input
        ut = CFTOC_MPC.u_pred[:,0]
        ucl_feasible.append(ut)
        
        # Run system dynamics
        xcl_feasible.append(CFTOC_MPC.model(xcl_feasible[time_index], ut))
        time_index += 1
        
        print("CFTOC #", time_index-1)
        
        # Stop running when finish line is crossed from below
        if (xt[1]>0) & (abs(xt[4]) <= 0.028):
            break
            
    print("AVG: ", np.mean(end_time))
    print("MAX: ", np.max(end_time))
    x_array = np.array(xcl_feasible)
    plot_trajectories(x_array)

    
    # Store iteration time
    completion_time = [time_index]
    
    print("MPC Terminated!")

	# =========================================================================
	# =============================   Run LMPC   ==============================
	# =========================================================================
    
	# Initialize LMPC objects
    N_LMPC = 7 # horizon length
    CFTOC_LMPC = CFTOC(N_LMPC, traj, dynamics, costs) # CFTOC solved by LMPC
    lmpc = LMPC(CFTOC_LMPC, N_LMPC, CVX=True) # Initialize the LMPC (decide if you wanna use the CVX hull)
    lmpc.addTrajectory(xcl_feasible, ucl_feasible) # Add feasible trajectory to the safe set
	
    totalIterations = 4 # Number of iterations to perform

    print("Starting LMPC...")
    
    # Run LMPC
    for it in range (0,totalIterations):
        # Reset initial conditions and storage at each iteration
        xcl = [x0] 
        ucl =[]
        ut = u0
        time_index = 0
        
		# Solve CFTOC for LMPC
        while True:
			# Read measurement
            xt = xcl[time_index] 

			# Solve CFTOC
            lmpc.solve(xt, ut, time_index, verbose = False) 
			# Read optimal input
            ut = lmpc.uPred[:,0]

			# Apply optimal input to the system
            ucl.append(ut)
            xcl.append(lmpc.cftoc.model(xcl[time_index], ut))
            time_index += 1

            print("LMPC ",it+1,"|",time_index-1)

            # troubleshootin
            #if (time_index==500):
            #    x_LMPC = np.array(xcl)
            #    plot_trajectories(x_LMPC)
                
            # Quit when finish line is reached
            if (xt[1]>0) & (abs(xt[4]) <= 0.028):
                x_LMPC = np.array(xcl)
                plot_trajectories(x_LMPC)
                break

        print("LMPC Lap #", it+1 , " complete!")

		# Add trajectory to update the safe set and value function
        lmpc.addTrajectory(xcl, ucl)
        
        
        # FOR PLOTTING AND TROUBLESHOOITNG
        #x_LMPC = np.array(xcl)
        #plot_trajectories(x_LMPC)
        
        # Store completion time
        completion_time.append(time_index)

    
	# =====================================================================================
    
	# ====================================================================================
	# Compute optimal solution by solving a FTOCP with long horizon
	# ====================================================================================
    """
    N = 1000 # Set a very long horizon to fake infinite time optimal control problem
    CFTOC_Full = CFTOC(N, traj, dynamics, costs)
    CFTOC_Full.solve(x0,0)
    xOpt = CFTOC_Full.xPred
    uOpt = CFTOC_Full.uPred
    """
    #costOpt = lmpc.computeCost(xOpt.T.tolist(), uOpt.T.tolist())
    #print("Optimal cost is: ", costOpt[0])
	# Store optimal solution in the lmpc object
    #lmpc.optCost = costOpt[0]
    #lmpc.xOpt    = xOpt

	# Save the lmpc object
    #filename = 'lmpc_object.pkl'
    #filehandler = open(filename, 'wb')
    #pickle.dump(lmpc, filehandler)
    #plt.plot(x_array[:,3], x_array[:,4],'green')
    #plt.plot(xOpt[:,3], xOpt[:,4],'magenta')
    plot_trajectories(x_array)
    
    print("Completion times: ", completion_time)

    
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