import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('Trajectory')

traj = np.load("para8.npy").T

print("Traj size 0 - ",np.size(traj,0))
print("Traj size 1 - ",np.size(traj,1))



def plot_trajectories(x):
    
    # 2D X-Y plot
    fig = plt.figure(1, figsize=(18,6))
    plt.subplot(1,3,1)
    plt.plot(x[3,:]  , x[4,:],'blue')
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

traj = traj[:,0:]

traj = np.hstack((traj,traj[:,20:]))

plot_trajectories(traj)