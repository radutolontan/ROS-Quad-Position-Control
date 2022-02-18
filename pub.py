#!/usr/bin/env python
# license removed for brevity

import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy as np
import copy
from quad_controllers_PM import quad_CFTOC
import matplotlib.pyplot as plt
from quad_properties_PM import *
# from quad_dynamics_PM import quad_dyn_PM
from quad_trajectory_PM import circular_traj, corner_traj, set_point

# Select desired quadrotor type and create its object
m,A, B, C, Q, Qf, R, dR, uL, uU = falcon_white()
quadrotor = quad_class_PM(m, A, B, C, Q, Qf, R, dR, uL, uU)

i = 0
g = 9.81
kp = np.array([4, 4, 8])
kv = np.array([3, 3, 6])
e3 = np.array([0,0,1])
freq = 100
N_MPC = 20
F = np.array([0,0,15])
N = 1000
x = np.zeros((6,N+2))
x_act = []


odom_stored_msg = Odometry()

def odom_callback(msg):
	global odom_stored_msg
	odom_stored_msg = copy.deepcopy(msg)

def set_point_PD():
    # k - time steps; freq - frequncy (Hz)
    # Create storage
    x = np.zeros(3)
    v = np.zeros(3)
    a = np.zeros(3)
    # Setpoints
    x_des = 0
    y_des = 0
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
    xQ = np.array([odom_stored_msg.twist.twist.linear.x,
                   odom_stored_msg.twist.twist.linear.y,
                   odom_stored_msg.twist.twist.linear.z,
                   odom_stored_msg.pose.pose.position.x, 
                   odom_stored_msg.pose.pose.position.y,
                   odom_stored_msg.pose.pose.position.z])
    x[:,i] = xQ
    # circular_traj
    feas, xMPC, uMPC = quad_CFTOC(des_traj, ref_properties, quadrotor, N_MPC, i, freq, xQ, F)
    print('POS: ', x[3:6,i])
    # control
    i += 1
    F = uMPC[:,0]
    #print('force is :',F)   
    

def main():
    global odom_stored_msg, F, i, quadrotor, freq, N_MPC, x
    odom_sub = rospy.Subscriber("/white_falcon/odometry/mocap", Odometry, odom_callback)
    pub = rospy.Publisher('/white_falcon/thrust_force', TwistStamped, queue_size=10)
    rate = rospy.Rate(freq) # 100hz
    
    # ---------------------------- TAKE OFF -----------------------------------
    # Set desired trajectory to 0 for vertical-up setpoint
    des_traj = 0
    # Exit execution when cruise height is reached
    while (x[5,i-1] < 1.48) & (not rospy.is_shutdown()):
        # arg#1 - XPos ; arg#2 - YPos ; arg#2 - ZPos
        mpc_control(des_traj, np.array([0,0,1.5]))
        thrust = TwistStamped()
        thrust.twist.linear.x = F[0]
        thrust.twist.linear.y = F[1]
        thrust.twist.linear.z = F[2] + 5.25
        thrust.twist.angular.x = 0.0
        thrust.twist.angular.y = 0.0
        thrust.twist.angular.z = 0.0 
        rospy.loginfo(thrust)
        pub.publish(thrust)
        rate.sleep() 
    
    # Change 10.48 back to 1.48
    # Change while statement back to (x[5,i-1]) < 10.48)
    # Change des_traj back to 0/1
    # Change radius-center in circular_traj
    # Change r/2 in trajectory
        
    # ----------------------------- NAVIGATE ----------------------------------
    # Set desired trajectory to 1 for circular navigation
    
    i = 0
    des_traj = 1
    while (not rospy.is_shutdown()):
        # arg#1 - radius ; arg#2 - V_tan ; arg#2 - ZPos
        mpc_control(des_traj, np.array([1,1,1.5]))
        thrust = TwistStamped()
        thrust.twist.linear.x = F[0]
        thrust.twist.linear.y = F[1]
        thrust.twist.linear.z = F[2] + 5.25
        thrust.twist.angular.x = 0.0
        thrust.twist.angular.y = 0.0
        thrust.twist.angular.z = 0.0 
        rospy.loginfo(thrust)
        pub.publish(thrust)
        rate.sleep() 
    
    # -------------------------- POWER OFF ------------------------------------    
    # thrust = TwistStamped()
    #while (np.linalg.norm(x[:,i] - np.array([0,0,0.53,0,0,0])) > 0.01) and (not rospy.is_shutdown()):
    # thrust.twist.linear.x = 0
    # thrust.twist.linear.y = 0
    # thrust.twist.linear.z = 0
    # thrust.twist.angular.x = 0.0
    # thrust.twist.angular.y = 0.0
    # thrust.twist.angular.z = 0.0 
    # rospy.loginfo(thrust)
    # pub.publish(thrust)
    # rate.sleep() 
        
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
    global odom_stored_msg, F, m, g, e3, i 
    # states
    xQ = np.array([odom_stored_msg.pose.pose.position.x, 
                   odom_stored_msg.pose.pose.position.y,
                   odom_stored_msg.pose.pose.position.z])
    vQ = np.array([odom_stored_msg.twist.twist.linear.x,
                   odom_stored_msg.twist.twist.linear.y,
                   odom_stored_msg.twist.twist.linear.z])
    x[0:3,i] = vQ
    x[3:6,i] = xQ

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
 
    
        


