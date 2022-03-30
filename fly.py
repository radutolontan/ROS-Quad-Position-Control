#!/usr/bin/env python
# license removed for brevity

# Import ROS-specific packages
import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

# Import controllers and trajectories
import sys
sys.path.append('Optimal Control')
#sys.path.append('Classic Control')
sys.path.append('Trajectory')
from CFTOC_pyomo import CFTOC
from LMPC import LMPC
#from PID import PID
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
        self.R = 2 * np.diag(np.array([1,1,0.05]))
        # Jf = (x_N - x_N_ref).T * Qf * (x_N - x_N_ref) (terminal cost on states)
        self.Qf = scipy.linalg.solve_discrete_are(dynamics.A_k, dynamics.B_k, 
                                                  self.Q, self.R)
        # Jdinput = (u_k+1 - u_k).T * dR * (u_k+1 - u_k) (stage cost on input change)
        # [ORIGINAL] self.dR = 0.2 * np.diag(np.array([16,16,2])) # 0.2
        self.dR = 0.5 * np.diag(np.array([5,5,2]))

# Define odometry messages and callbacks
odom_stored_msg = Odometry()
def odom_callback(msg):
	global odom_stored_msg
	odom_stored_msg = copy.deepcopy(msg)

# Define linear dynamics
dynamics = point_mass_dynamics()

# Define optimal control costs
costs = cost_matrices(dynamics)

# Select desired reference trajectory for main navigation (0 - circular)(1 - setPoint)
trajectory = Trajectory(dynamics.freq, 0)

# Initial conditions on inputs are set to allow smooth input changes
F_vec = np.array([0,0,dynamics.m*dynamics.g]).T

# Initialize MPC object
N_CFTOC = 7
CFTOC_MPC  = CFTOC(N_CFTOC, dynamics, costs)

# initialize flight mode
flight_mode = 0

# Select desired quadrotor type and create its object
#m, A, B, C, Q, Qf, R, dR, uL, uU = falcon_white()
#quadrotor = quad_class_PM(m, A, B, C, Q, Qf, R, dR, uL, uU)

#i = 0
#g = 9.81
#kp = np.array([4, 4, 8])
#kv = np.array([3, 3, 6])
#e3 = np.array([0,0,1])
#freq = 100
#N_MPC = 20
#F = np.array([0,0,15])
#N = 300
#x = np.zeros((6,1))




def set_point_PD():
    # k - time steps; freq - frequncy (Hz)
    # Create storage
    x = np.zeros(3)
    v = np.zeros(3)
    a = np.zeros(3)
    # Setpoints
    x_des = 0.0
    y_des = 0.0
    z_des = 1.5
    # Position
    x[0] =  x_des
    x[1] =  y_des
    x[2] =  z_des
    # Velocity
    v[0] = 0
    v[1] = 0
    v[2] = 0
    # Acceleration
    a[0] = 0
    a[1] = 0
    a[2] = 0   
    return x, v, a


def circular_traj_pd(time):
    # k - time steps; freq - frequncy (Hz)
    # Create storage
    
    x = np.zeros((3,int(np.size(time))))
    v = np.zeros((3,int(np.size(time))))
    a = np.zeros((3,int(np.size(time))))
    # Set radius of circlular trajectory
    r = 1 # (m)
    # Set the cruise altitude
    cruise_height = 0.5 # (m)
    # Set velocity tangent to trajectory
    lin_velocity = 0.8 # (m/s)
    
    # Position
    x[0,:] =  r * np.cos(time * lin_velocity / r)
    x[1,:] =  r * np.sin(time * lin_velocity / r)
    x[2,:] =  cruise_height
    # Velocity
    v[0,:] = - r * (lin_velocity / r) * np.sin(time * lin_velocity / r)
    v[1,:] =   r * (lin_velocity / r) * np.cos(time* lin_velocity / r)
    v[2,:] = 0
    # Acceleration
    a[0,:] = - r * (lin_velocity / r)**2 * np.cos(time * lin_velocity / r)
    a[1,:] = - r * (lin_velocity / r)**2 * np.sin(time * lin_velocity / r)
    a[2,:] = 0
        
    return x, v, a
# =============================================================================
# ============================    MPC CONTROL     =============================
# =============================================================================    
    
def mpc_control(des_traj, ref_properties):
    global odom_stored_msg, F, i, quadrotor, freq, N_MPC, x
    # states

    XQ = np.array([[odom_stored_msg.twist.twist.linear.x],
                   [odom_stored_msg.twist.twist.linear.y],
                   [odom_stored_msg.twist.twist.linear.z],
                   [odom_stored_msg.pose.pose.position.x], 
                   [odom_stored_msg.pose.pose.position.y],
                   [odom_stored_msg.pose.pose.position.z]])
    x = np.hstack((x,XQ))
    XQ = np.reshape(XQ, 6)
    # circular_traj
    feas, xMPC, uMPC = quad_CFTOC(des_traj, ref_properties, quadrotor, N_MPC, i, freq, XQ, F)
    print('POS: ', x[3:6,i])
    # control
    i += 1
    F = uMPC[:,0]
    #print('force is :',F)   

def get_flight_mode():
    global flight_mode
    flight_mode = int(input("1 (T/O) ; 2 (Train) ; 3 (LMPC Navi.) ; 4 (Land) ... "))

def send_command():
    global dynamics, F_vec

    # Declare publisher
    pub = rospy.Publisher('/white_falcon/thrust_force', TwistStamped, queue_size=10)
    rate = rospy.Rate(dynamics.freq) # operating frequency set inside dynamics

    # Vertical force offset
    v_offset = 5.25 # (7.25 for hardware with cables)

    # Send thrust command to quadcopter
    thrust = TwistStamped()
    thrust.twist.linear.x = F_vec[0]
    thrust.twist.linear.y = F_vec[1]
    thrust.twist.linear.z = F_vec[2] + v_offset
    thrust.twist.angular.x = 0.0
    thrust.twist.angular.y = 0.0
    thrust.twist.angular.z = 0.0 
    rospy.loginfo(thrust)
    pub.publish(thrust)
    rate.sleep() 

def hold_position():
    global odom_stored_msg, dynamics, costs, CFTOC_MPC, F_vec

    # Hold current position
    hold_loc = np.array([odom_stored_msg.twist.twist.linear.x, odom_stored_msg.twist.twist.linear.y, odom_stored_msg.twist.twist.linear.z,
                         odom_stored_msg.pose.pose.position.x, odom_stored_msg.pose.pose.position.y, odom_stored_msg.pose.pose.position.z])
     
    # Define hold trajectory (type: set point)
    hold_traj = Trajectory(dynamics.freq, 1)
    hold_time = 0

    while not rospy.is_shutdown():
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

def takeoff():
    global odom_stored_msg, dynamics, costs, CFTOC_MPC, F_vec
    
    # Initialize input and time
    T_O_point = np.array([0, 0, 0, 0, 0, 0.8])
    TO_Time = 0
    ut = np.array([0,0,dynamics.m*dynamics.g]).T

    # Define takeoff trajectory (type: set point)
    takeoff_traj = Trajectory(dynamics.freq, 1)

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
            break

def main():
    global flight_mode, odom_stored_msg

    # Setup odometry subscriber
    odom_sub = rospy.Subscriber("/white_falcon/odometry/mocap", Odometry, odom_callback)
    
    # Prompt user to initiate flight
    get_flight_mode()

    for jj in range(0,4):

        if (int(flight_mode) == 1):
            # Take off
            takeoff()
            print("TAKEOFF COMPLETED. Starting Hold...")

            # Create Multithreaded Hold Process
            hold_thread = threading.Thread(target=hold_position)
            change_mode_thread = threading.Thread(target=get_flight_mode)
            hold_thread.start()
            change_mode_thread.start()
            change_mode_thread.join()
            

        #elif (int(flight_mode) == 2):
            # TO DO TRAINING

        #elif (int(flight_mode) == 3):
            # TO DO LMPC NAVIGATION

        #else:
            # TO DO LAND
            
"""
    # ----------------------------- NAVIGATE ----------------------------------
    # Set desired trajectory to 1 for circular navigation
    i = 0
    des_traj = 1
    while (i<N) & (not rospy.is_shutdown()):
        # rad: 0.3 , omega: 4 , r_actual: 1.4
        # rad: 0.2 , omega: 4 , r_actual: 0.95
        # arg#1 - radius ; arg#2 - omega ; arg#2 - ZPos
        mpc_control(des_traj, np.array([0.2,4,1.5]))
        thrust = TwistStamped()
        thrust.twist.linear.x = F[0]
        thrust.twist.linear.y = F[1]
        thrust.twist.linear.z = F[2] + 7.25
        thrust.twist.angular.x = 0.0
        thrust.twist.angular.y = 0.0
        thrust.twist.angular.z = 0.0 
        rospy.loginfo(thrust)
        pub.publish(thrust)
        rate.sleep() 
    
    # -------------------------- LANDING -------------------------------------- 
    des_traj = 0
    while (x[5,i-1] > 0.62) & (not rospy.is_shutdown()):
        # arg#1 - XPos ; arg#2 - YPos ; arg#2 - ZPos
        mpc_control(des_traj, np.array([0.1,0.15,0.6]))
        thrust = TwistStamped()
        thrust.twist.linear.x = F[0]
        thrust.twist.linear.y = F[1]
        thrust.twist.linear.z = F[2] + 7.25
        thrust.twist.angular.x = 0.0
        thrust.twist.angular.y = 0.0
        thrust.twist.angular.z = 0.0 
        rospy.loginfo(thrust)
        pub.publish(thrust)
        rate.sleep() 
        
    # -------------------------- POWER OFF ------------------------------------
    sleep(0.05)
    thrust = TwistStamped()
    thrust.twist.linear.x = 0
    thrust.twist.linear.y = 0
    thrust.twist.linear.z = 0
    thrust.twist.angular.x = 0.0
    thrust.twist.angular.y = 0.0
    thrust.twist.angular.z = 0.0 
    rospy.loginfo(thrust)
    pub.publish(thrust)
    rate.sleep() 
"""

# =============================================================================
# ============================    PID CONTROL     =============================
# =============================================================================
        
def main_PID():
    odom_sub = rospy.Subscriber("/white_falcon/odometry/mocap", Odometry, odom_callback)
    pub = rospy.Publisher('/white_falcon/thrust_force', TwistStamped, queue_size=10)
    rate = rospy.Rate(freq) # 10hz
    start_time = rospy.get_time()
    while  (i<N) & (not rospy.is_shutdown()):
        current_time = rospy.get_time() - start_time
        pd_control(current_time)
        thrust = TwistStamped()
        thrust.twist.linear.x = F[0]
        thrust.twist.linear.y = F[1]
        thrust.twist.linear.z = F[2]
        thrust.twist.angular.x = 0.0
        thrust.twist.angular.y = 0.0
        thrust.twist.angular.z = 0.0
        rospy.loginfo(thrust)
        pub.publish(thrust)
        rate.sleep() 	
 
        
def pd_control(time):
    global odom_stored_msg, F, m, g, e3, i, x
    # states
    xQ = np.array([odom_stored_msg.pose.pose.position.x, 
                   odom_stored_msg.pose.pose.position.y,
                   odom_stored_msg.pose.pose.position.z])
    vQ = np.array([odom_stored_msg.twist.twist.linear.x,
                   odom_stored_msg.twist.twist.linear.y,
                   odom_stored_msg.twist.twist.linear.z])
    #XQ = np.hstack((vQ,xQ))
    #x = np.hstack((x,XQ.T))
    #x[0:3,i] = vQ
    #x[3:6,i] = xQ

    print("xQ", xQ)
    print("vQ", vQ)
    # setpoint 
    xd, vd, ad = set_point_PD()
    # circular_traj
    #xd, vd, ad = circular_traj_pd(time)
    # x[3:6,i] = xQ.T[0]
    # control
    i += 1
    err_x = xQ-xd
    err_v = vQ-vd
    print("xd", xd)
    print("vd", vd)

    print("err_x", err_x)
    print("err_v", err_v)

    Fff = m*(g*e3 + ad)
    Fpd = -kp*err_x - kv*err_v
    F = Fff+Fpd

    print("Fff", Fff)
    print("Fpd", Fpd)

    print('force is :',F)
    
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
        plot_trajectories()
        
    except rospy.ROSInterruptException:
        pass    
 
    
        


