#!/usr/bin/env python
# license removed for brevity

# Import ROS-specific packages
import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

# Import controllers and trajectories
import sys
sys.path.append('Optimal Control')
sys.path.append('Classic Control')
sys.path.append('Trajectory')
from CFTOC_pyomo import CFTOC
from LMPC import LMPC
from PID import PD
from Trajectory import Trajectory

# Import Python packages
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import copy
import threading
import pickle
import time
import pdb

# Define quadrotor dynamics and optimal control costs
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
    
        # Define discrete-time dynamics and operating frequency
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
        # [ORIGINAL] self.R = 2 * np.diag(np.array([1,1,0.0002]))
        self.R = 2 * np.diag(np.array([1,1,0.03]))
        # Jf = (x_N - x_N_ref).T * Qf * (x_N - x_N_ref) (terminal cost on states)
        self.Qf = scipy.linalg.solve_discrete_are(dynamics.A_k, dynamics.B_k, 
                                                  self.Q, self.R)
        # Jdinput = (u_k+1 - u_k).T * dR * (u_k+1 - u_k) (stage cost on input change)
        # [ORIGINAL] self.dR = 0.2 * np.diag(np.array([16,16,2]))
        self.dR = 0.5 * np.diag(np.array([5,5,0.05]))

# Define linear dynamics
dynamics = point_mass_dynamics()

# Define optimal control costs
costs = cost_matrices(dynamics)

# Define odometry messages, callback functions and publisher
pub = []; rate = []
odom_stored_msg = Odometry()
def odom_callback(msg):
	global odom_stored_msg
	odom_stored_msg = copy.deepcopy(msg)

# Select desired reference trajectory for main navigation (0 - circular)(1 - setPoint)(2 - CUSTOM)
trajectory = Trajectory(dynamics.freq, 2)

# Initial conditions on inputs are set to allow smooth input changes
F_vec = np.array([0,0,dynamics.m*dynamics.g]).T
F_z_offset_MPC = 4.7
F_z_offset_PID = 1.2

# Initialize Landing Sight
L_point = []

# Initialize MPC object
N_CFTOC = 7
CFTOC_MPC  = CFTOC(N_CFTOC, dynamics, costs)

# Initialize PID object
PD_controller = PD(dynamics)

# Initialize LMPC objects
N_LMPC = 9 # horizon length
CFTOC_LMPC = CFTOC(N_LMPC, dynamics, costs) # CFTOC solved by LMPC

# Intiate time storage for PID and MPC
PID_Time = 0; MPC_Time = 0
xcl_MPC = []; ucl_MPC = []; xcl_PID = [] ; ucl_PID = []

# initialize flight mode
flight_mode = 0

def get_flight_mode():
    global flight_mode
    flight_mode = int(input("1 (T/O) ; 2 (Train) ; 3 (LMPC Navi.) ; 4 (Land) ; 5 (Exit) ... "))

def send_command(v_offset):
    global dynamics, F_vec, pub, rate

    # Send thrust command to quadcopter
    thrust = TwistStamped()
    thrust.twist.linear.x = F_vec[0]
    thrust.twist.linear.y = F_vec[1]
    thrust.twist.linear.z = F_vec[2] + v_offset
    thrust.twist.angular.x = 0.0
    thrust.twist.angular.y = 0.0
    thrust.twist.angular.z = 0.0 
    #rospy.loginfo(thrust)
    pub.publish(thrust)
    rate.sleep() 

def hold_position():
    global odom_stored_msg, dynamics, costs, CFTOC_MPC, F_vec, flight_mode, F_z_offset_MPC

    # Copy initial flight mode
    initial_mode = copy.deepcopy(flight_mode)

    # Hold current position
    hold_loc = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                         odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

    ########################################
    # hold_loc = np.array([0, 0, 0, 0.8, 0, 1])
    ########################################

    # Define hold trajectory (type: set point)
    hold_traj = Trajectory(dynamics.freq, 1)
    hold_time = 0

    while (flight_mode == initial_mode) & (not rospy.is_shutdown()):
        # Read quadcopter state
        xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                       odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

        # Compute trajectory preview
        preview = hold_traj.get_reftraj(hold_time, CFTOC_MPC.N, hold_loc[3:6])[0:6] # No need for acceleration here [6:9]

        # Solve MPC CFTOC problem
        CFTOC_MPC.solve_MPC(xt, F_vec, preview) 

        # Read first optimal input and send to drone
        F_vec = CFTOC_MPC.u_pred[:,0]
        send_command(F_z_offset_MPC)

def takeoff_PD():
    global odom_stored_msg, dynamics, costs, F_vec, L_point, PD_controller, F_z_offset_PID
    
    # Initialize input and time
    T_O_point = np.array([0, 0, 0, 0, 0, 0.6])

    # Define takeoff trajectory (type: set point)
    takeoff_traj = Trajectory(dynamics.freq, 1)

    # Store takeoff position as home for landing purposes
    L_point = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                        odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

    while not rospy.is_shutdown():
        # Read quadcopter state
        xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                       odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

        # Compute trajectory preview
        preview = takeoff_traj.get_reftraj(0, 1, T_O_point[3:6])

        # Solve MPC CFTOC problem
        PD_controller.solve_PD(xt, np.reshape(preview,9)) 

        # Read first optimal input and send to drone
        F_vec = PD_controller.F_command
        send_command(F_z_offset_PID)

        # End manouver when desired point is captured
        if (la.norm(xt - T_O_point) <= 0.05):
            #F = ut
            break

def takeoff():
    global odom_stored_msg, dynamics, costs, CFTOC_MPC, F_vec, L_point, F_z_offset_MPC
    
    # Initialize input and time
    T_O_point = np.array([0, 0, 0, 0, 0, 0.6])
    TO_Time = 0

    # Define takeoff trajectory (type: set point)
    takeoff_traj = Trajectory(dynamics.freq, 1)

    # Store takeoff position as home for landing purposes
    L_point = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                        odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

    while not rospy.is_shutdown():
        # Read quadcopter state
        xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                       odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

        # Compute trajectory preview
        preview = takeoff_traj.get_reftraj(TO_Time, CFTOC_MPC.N, T_O_point[3:6])[0:6] # No need for acceleration here [6:9]

        # Solve MPC CFTOC problem
        CFTOC_MPC.solve_MPC(xt, F_vec, preview) 

        # Read first optimal input and send to drone
        F_vec = CFTOC_MPC.u_pred[:,0]
        send_command(F_z_offset_MPC)

        # Increase counter 
        TO_Time += 1

        # End manouver when desired point is captured
        if (la.norm(xt - T_O_point) <= 0.02):
            #F = ut
            CFTOC_MPC.xPred = []
            CFTOC_MPC.uPred = []
            break

def land_PD():
    global odom_stored_msg, dynamics, costs, F_vec, L_point, PD_controller, F_z_offset_PID

    # Define takeoff trajectory (type: set point)
    landing_traj = Trajectory(dynamics.freq, 1)

    while not rospy.is_shutdown():
        # Read quadcopter state
        xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                       odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

        # Compute trajectory preview
        preview = landing_traj.get_reftraj(0, 1, L_point[3:6])
        # Solve MPC CFTOC problem
        PD_controller.solve_PD(xt, np.reshape(preview,9)) 

        # Read first optimal input and send to drone
        F_vec = PD_controller.F_command
        send_command(F_z_offset_PID)

        # Turn off motors when close to home
        if (la.norm(xt - L_point) <= 0.12):
            F_vec = np.zeros(3)
            send_command(0)
            break

def land():
    global odom_stored_msg, dynamics, costs, CFTOC_MPC, F_vec, L_point, F_z_offset_MPC
    
    # Initialize input and time
    L_Time = 0

    # Define takeoff trajectory (type: set point)
    landing_traj = Trajectory(dynamics.freq, 1)

    while not rospy.is_shutdown():
        # Read quadcopter state
        xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                       odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

        # Compute trajectory preview
        preview = landing_traj.get_reftraj(L_Time, CFTOC_MPC.N, L_point[3:6])[0:6] # No need for acceleration here [6:9]

        # Solve MPC CFTOC problem
        CFTOC_MPC.solve_MPC(xt, F_vec, preview) 

        # Read first optimal input and send to drone
        F_vec = CFTOC_MPC.u_pred[:,0]
        send_command(F_z_offset_MPC)

        # Increase counter 
        L_Time += 1

        # Turn off motors when close to home
        if (la.norm(xt - L_point) <= 0.05):
            F_vec = np.zeros(3)
            send_command(0)
            break

def train():
    global dynamics, costs
    global CFTOC_MPC, PD_controller, F_z_offset_MPC, F_z_offset_PID
    global odom_stored_msg, F_vec, trajectory
    global xcl_MPC, ucl_MPC, xcl_PID, ucl_PID
    global PID_Time, MPC_Time

    # ========================================================
    # ===================== TRAIN w. MPC =====================
    # ========================================================
    
    while True:
        # Read system state
        xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                       odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])
        xcl_MPC.append(np.reshape(xt,(6,1)))

        # Compute trajectory preview
        preview = trajectory.get_reftraj(MPC_Time, CFTOC_MPC.N)[0:6] # No need to carry ref. acceleration [6:9]

        # Solve MPC CFTOC problem
        CFTOC_MPC.solve_MPC(xt, F_vec, preview) 
        
		# Read first optimal input and send to drone
        F_vec = CFTOC_MPC.u_pred[:,0]
        ucl_MPC.append(np.reshape(F_vec,(3,1)))
        send_command(F_z_offset_MPC)
        
        MPC_Time += 1 
        # Stop running when finish line is crossed from below
        if trajectory.crossedFinish(np.reshape((xcl_MPC)[-1],(6,1)), MPC_Time) == True:
            break

    print("MPC Training Complete")

    # ========================================================
    # ===================== TRAIN w. PID =====================
    # ========================================================

    while True:
        # Read system state
        xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                       odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])
        xcl_PID.append(np.reshape(xt,(6,1)))

        # Compute trajectory preview
        preview = trajectory.get_reftraj(PID_Time, 1)

        # Solve MPC CFTOC problem
        PD_controller.solve_PD(xt, np.reshape(preview,6)) 
        
		# Read first optimal input and send to drone
        F_vec = PD_controller.F_command
        send_command(F_z_offset_PID)
        ucl_PID.append(np.reshape(F_vec,(3,1)))
        
        PID_Time += 1 
        # Stop running when finish line is crossed from below
        if trajectory.crossedFinish(np.reshape((xcl_PID)[-1],(6,1)), PID_Time) == True:
            PID_Time += 1
            break

    print("PID Training Complete")

def navigate_LMPC():
    global CFTOC_LMPC, trajectory
    global xcl_MPC, ucl_MPC, xcl_PID, ucl_PID
    global PID_Time, MPC_Time

    lmpc = LMPC(CFTOC_LMPC, CFTOC_LMPC.N, PID_Time) # Initialize the LMPC
    lmpc.addTrajectory(xcl_PID, ucl_PID) # Add feasible trajectory to the safe set
    lmpc.goal_pt = trajectory.get_goalpoint() # Import coordinates of goal

    totalIterations = 4 # Number of iterations to perform

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
            xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                           odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])
      
            # Check if any predicted states are over the finish line
            lmpc.closest_pt_traj = trajectory.get_closestpoint(xt)
            if (lmpc.xPred != []):
                lmpc.pred_over_finish = trajectory.crossedFinish(lmpc.xPred, lmpc.t_index)

			# Solve CFTOC
            lmpc.solve(xt, ut) 

			# Read and apply optimal input to the system
            ut = lmpc.uPred[:,0]
            ucl.append(ut)
            xcl.append(xt)

            # Add new state to past iteration SS to enable loop-around 
            lmpc.addtoPrevSS(xcl[-1])
            lmpc.t_index += 1
                
            # Quit when finish line is reached
            if trajectory.crossedFinish(np.reshape(xcl[-1],(6,1)), lmpc.t_index) == True:
                del xcl[0]
                break

        print("LMPC Lap #", it+1 , " complete!")

		# Add trajectory to update the safe set and value function
        lmpc.addTrajectory(xcl, ucl)

def main():
    global flight_mode, odom_stored_msg, pub, rate, CFTOC_MPC
    global xcl_PID

    # Setup odometry subscriber
    odom_sub = rospy.Subscriber("/white_falcon/odometry/mocap", Odometry, odom_callback)
    pub = rospy.Publisher('/white_falcon/thrust_force', TwistStamped, queue_size=10)
    rate = rospy.Rate(dynamics.freq) # operating frequency set inside dynamics
    
    # Prompt user to initiate flight
    get_flight_mode()

    for jj in range(0,4):

        if (int(flight_mode) == 1):
            #========= TAKE OFF =========
            print("TAKEOFF INITIATED...")
            try:
                takeoff_PD()
                print("TAKEOFF SUCCESSFULL! Starting Hold...")
            except:
                print("TAKEOFF ABORTED. LANDING...")
                land_PD()

        elif (int(flight_mode) == 2):
            #========== TRAIN ===========            
            print("TRAINING INITIATED...")
            try:
                train()
                print("TRAINING SUCCESSFULL! Starting Hold...")
            except:
                print("TRAINING ABORTED. LANDING...")
                land_PD()

        elif (int(flight_mode) == 3):
            #=========== LMPC ===========            
            print("LMPC NAVIGATION INITIATED...")
            try:
                navigate_LMPC()
                print("LMPC SUCCESSFULL! Starting Hold...")
            except:
                print("LMPC ABORTED. LANDING...")
                land_PD()

        elif (int(flight_mode) == 4):
            #========= LANDING =========
            land_PD()
            print("LANDING SUCCESSFULL!")
            break 

        else:
            #========== EXIT ===========  
            break   
    
        # Create Multithreaded Hold Process
        hold_thread = threading.Thread(target=hold_position)
        change_mode_thread = threading.Thread(target=get_flight_mode)
        hold_thread.start()
        change_mode_thread.start()
        change_mode_thread.join()
        CFTOC_MPC.xPred = []
        CFTOC_MPC.uPred = []

    xcl_PID = np.array(xcl_PID)
    print("SHAPE: ", xcl_PID[:,:,0].shape)
    plot_trajectories(xcl_PID[:,:,0].T)
            
# =============================================================================
# ==============================    PLOTTING     ==============================
# =============================================================================

def plot_trajectories(x):
    
    # 2D X-Y plot
    fig = plt.figure(1, figsize=(18,6))
    plt.subplot(1,3,1)
    plt.plot(  x[3,:]  , x[4,:],'blue')
    plt.scatter(x[3,0] , x[4,0], s =15, c='red')
    plt.title('X vs. Y')

    # 2D X-Z plot
    plt.subplot(1,3,2)
    plt.plot(  x[3,:]  , x[5,:],'blue')
    plt.scatter(x[3,0] , x[5,0], s =15, c='red')
    plt.title('X vs. Z')

    # 2D Y-Z plot
    plt.subplot(1,3,3)
    plt.plot(  x[4,:]  , x[5,:],'blue')
    plt.scatter(x[4,0] , x[5,0], s =15, c='red')
    plt.title('Y vs. Z')
    plt.legend(['Reference Trajectory','Open Loop (CFTOC)','Closed Loop (MPC)','Initial Condition (x0)'])
    plt.show()

    # 2D X-T plot
    fig = plt.figure(2, figsize=(18,6))
    # hor = np.zeros((N+1))
    # for i in range((N+1)):
    #     hor[i] = i
    plt.subplot(2,3,1)
    plt.plot( x[3,:],'blue')
    plt.title('X vs. T')

    # 2D X-T plot
    fig = plt.figure(2, figsize=(18,6))
    # hor = np.zeros((N+1))
    # for i in range((N+1)):
    #     hor[i] = i
    plt.subplot(2,3,2)
    plt.plot( x[4,:],'blue')
    plt.title('Y vs. T')
        
    # 2D X-T plot
    fig = plt.figure(2, figsize=(18,6))
    # hor = np.zeros((N+1))
    # for i in range((N+1)):
    #     hor[i] = i
    plt.subplot(2,3,3)
    plt.plot( x[5,:],'blue')
    plt.title('Z vs. T')
    plt.show()    
    
    
# =============================================================================    
# =============================================================================
# ================================    RUN     =================================
# =============================================================================
# =============================================================================
    
if __name__ == '__main__':
    try:
        rospy.init_node('offboard_node')
        main()  
        #plot_trajectories()
        
    except rospy.ROSInterruptException:
        pass    
 
    
        


