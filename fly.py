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

# Select desired reference trajectory for main navigation (0 - circular)(1 - setPoint)
trajectory = Trajectory(dynamics.freq, 0)

# Initial conditions on inputs are set to allow smooth input changes
F_vec = np.array([0,0,dynamics.m*dynamics.g]).T

# Initialize Landing Sight
L_point = []

# Initialize MPC object
N_CFTOC = 7
CFTOC_MPC  = CFTOC(N_CFTOC, dynamics, costs)

# Initialize PID object
PD_controller = PD(dynamics)

# initialize flight mode
flight_mode = 0

def get_flight_mode():
    global flight_mode
    flight_mode = int(input("1 (T/O) ; 2 (Train) ; 3 (LMPC Navi.) ; 4 (Land) ; 5 (Exit) ... "))

def send_command():
    global dynamics, F_vec, pub, rate

    # Vertical force offset
    v_offset = 4.7# (7.25 for hardware with cables)

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
    global odom_stored_msg, dynamics, costs, CFTOC_MPC, F_vec, flight_mode

    # Copy initial flight mode
    initial_mode = copy.deepcopy(flight_mode)

    # Hold current position
    hold_loc = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                         odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])
     
    # Define hold trajectory (type: set point)
    hold_traj = Trajectory(dynamics.freq, 1)
    hold_time = 0

    while (flight_mode == initial_mode) & (not rospy.is_shutdown()):
        # Read quadcopter state
        xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                       odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

        # Compute trajectory preview
        preview = hold_traj.get_reftraj(hold_time, CFTOC_MPC.N, hold_loc[3:6])

        # Solve MPC CFTOC problem
        CFTOC_MPC.solve_MPC(xt, F_vec, preview) 

        # Read first optimal input and send to drone
        F_vec = CFTOC_MPC.u_pred[:,0]
        send_command()

def takeoff_PD():
    global odom_stored_msg, dynamics, costs, F_vec, L_point, PD_controller
    
    # Initialize input and time
    T_O_point = np.array([0, 0, 0, 0, 0, 1])

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
        preview = takeoff_traj.get_reftraj(0, 0, T_O_point[3:6])
        # Solve MPC CFTOC problem
        PD_controller.solve_PD(xt, np.reshape(preview,6)) 

        # Read first optimal input and send to drone
        F_vec = PD_controller.F_command
        send_command()

        # End manouver when desired point is captured
        if (la.norm(xt - T_O_point) <= 0.05):
            #F = ut
            break

def takeoff():
    global odom_stored_msg, dynamics, costs, CFTOC_MPC, F_vec, L_point
    
    # Initialize input and time
    T_O_point = np.array([0, 0, 0, 0, 0, 1])
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
        preview = takeoff_traj.get_reftraj(TO_Time, CFTOC_MPC.N, T_O_point[3:6])

        # Solve MPC CFTOC problem
        CFTOC_MPC.solve_MPC(xt, F_vec, preview) 

        # Read first optimal input and send to drone
        F_vec = CFTOC_MPC.u_pred[:,0]
        send_command()

        # Increase counter 
        TO_Time += 1

        # End manouver when desired point is captured
        if (la.norm(xt - T_O_point) <= 0.02):
            #F = ut
            CFTOC_MPC.xPred = []
            CFTOC_MPC.uPred = []
            break

def land_PD():
    global odom_stored_msg, dynamics, costs, F_vec, L_point, PD_controller

    # Define takeoff trajectory (type: set point)
    landing_traj = Trajectory(dynamics.freq, 1)

    while not rospy.is_shutdown():
        # Read quadcopter state
        xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                       odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

        # Compute trajectory preview
        preview = landing_traj.get_reftraj(0, 0, L_point[3:6])
        # Solve MPC CFTOC problem
        PD_controller.solve_PD(xt, np.reshape(preview,6)) 

        # Read first optimal input and send to drone
        F_vec = PD_controller.F_command
        send_command()

        # Turn off motors when close to home
        if (la.norm(xt - L_point) <= 0.05):
            F_vec = np.zeros(3)
            send_command()
            break

def land():
    global odom_stored_msg, dynamics, costs, CFTOC_MPC, F_vec, L_point
    
    # Initialize input and time
    L_Time = 0

    # Define takeoff trajectory (type: set point)
    landing_traj = Trajectory(dynamics.freq, 1)

    while not rospy.is_shutdown():
        # Read quadcopter state
        xt = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                       odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])

        # Compute trajectory preview
        preview = landing_traj.get_reftraj(L_Time, CFTOC_MPC.N, L_point[3:6])

        # Solve MPC CFTOC problem
        CFTOC_MPC.solve_MPC(xt, F_vec, preview) 

        # Read first optimal input and send to drone
        F_vec = CFTOC_MPC.u_pred[:,0]
        send_command()

        # Increase counter 
        L_Time += 1

        # Turn off motors when close to home
        if (la.norm(xt - L_point) <= 0.05):
            F_vec = np.zeros(3)
            send_command()
            break

def main():
    global flight_mode, odom_stored_msg, pub, rate, CFTOC_MPC

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
                takeoff()
                print("TAKEOFF SUCCESSFULL! Starting Hold...")
            except:
                print("TAKEOFF ABORTED. LANDING...")
                land()

        #elif (int(flight_mode) == 2):
            # TO DO TRAINING

        #elif (int(flight_mode) == 3):
            # TO DO LMPC NAVIGATION

        elif (int(flight_mode) == 4):
            #========= LANDING =========
            land()
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
            
# =============================================================================
# ==============================    PLOTTING     ==============================
# =============================================================================

def plot_trajectories():
    global x
    # Remove first column used for stacking purposes
    x = x[:,1:]
    
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
 
    
        


