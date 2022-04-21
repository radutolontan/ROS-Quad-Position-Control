"""
This class is used to create reference trajectories for all types of position controllers
"""
import numpy as np
import numpy.linalg as la
import sys

class Trajectory(object):

    def __init__(self, freq, traj_type):
        # Trajectory types: 0 - circular; 1 - setpoint; 2 - custom trajectory
        self.traj_type = traj_type 
        self.freq = freq
        if (self.traj_type==2):
            self.custom_traj = np.load("para9.npy").T
            # Select non-overlapping section
            self.custom_traj = self.custom_traj[:,0:1715] # para8 - 1715 # para7 - 1220 # para9 - 1715
            # Double the trajectory for accurate predictions close to the finish line
            self.custom_traj = np.hstack((self.custom_traj,self.custom_traj[:,20:]))

    # ====================================================================================
    # ============================== TYPES OF TRAJECTORIES ===============================
    # ====================================================================================

    def circular_traj(self, k):
         # k - time steps; freq - frequncy (Hz)
        r = 0.7
        omega = 1.2
        cruise_height = 1.5
        # Create storage and set Z- location to 0
        x = np.zeros((9,int(np.size(k))))
        
        # X - and Y - velocities are parametrized
        x[0,:] = - (omega*r) * np.sin((k/self.freq) * omega) 
        x[1,:] =   (omega*r) * np.cos((k/self.freq) * omega)
        # Z - velocity is zero
        # X - and Y - locations are parametrized
        x[3,:] =  r * np.cos((k/self.freq) * omega) 
        x[4,:] =  r * np.sin((k/self.freq) * omega) 
        # Z - location is constant and equal to a height
        x[5,:] = cruise_height * np.ones(int(np.size(k)))
        # X - and Y - accelerations are parametrized
        x[6,:] = - ((omega**2)*r) * np.cos((k/self.freq) * omega) 
        x[7,:] = - ((omega**2)*r) * np.sin((k/self.freq) * omega) 
        # Z - acceleration is zero
        return x

    def set_point(self, k, setpoint):
        # k - time steps
        # Create storage for both velocities and locations
        # Set velocities to 0
        x = np.zeros((9,int(np.size(k))))

        # Set X- setpoint, Y- setpoint and Z- setpoint
        x[3,:] = setpoint[0] * np.ones(int(np.size(k)))
        x[4,:] = setpoint[1] * np.ones(int(np.size(k)))
        x[5,:] = setpoint[2] * np.ones(int(np.size(k)))
        return x

    def custom(self, k):
        # k - time steps
        x = self.custom_traj[:,k.astype(int)]
        return x

    # ====================================================================================
    # ============================== TRAJECTORY PROCESSING ===============================
    # ====================================================================================
    
    def top_bot_line(self, a, b, x1, y1):
        # Function takes in equation of line (y=a*x+b) and a point (x1,y1), and returns one if
        # point is above line and 0 if it is below the line
        ytemp = a*x1+b

        return 0 if ytemp > y1 else 1

    def crossedFinish(self, states, t_index):
        """
        Method takes in a vector of states (so a matrix) and returns a T/F vector indicating which
        column in the matrix corrresponds to a state beyond the finish line. 

        states: vector of states (initialized as xPred)
        t_index: current time index
        """
        # Initialize storage
        crossLinevec = np.zeros(states.shape[1])
        
        # For the first time CFTOC is solved in an iteration, return zeros
        if states==[]:
            return crossLinevec
        # Otherwise check to see if state in each column has passed the f_line
        else:
            for jj in range(states.shape[1]):

                ############### CIRCULAR TRACK TERMINATION CONDITION ###############
                if (self.traj_type == 0):
                    # Time index (t_index) is used to ensure the states at the beginning are not picked up
                    if all((states[1,jj] > 0.0, states[4,jj] >= 0.0, t_index > 300)) == True:
                        crossLinevec[jj] = 1

                ################# CUSTOM TRACK TERMINATION CONDITION ###############
                else:
                    # Line can only be crossed from Q3 into Q1
                    if np.all((self.top_bot_line(-0.5,0  ,states[3,jj],states[4,jj]) == 1,
                               self.top_bot_line(-0.5,0.1,states[3,jj],states[4,jj]) == 0,
                               states[1,jj] > 0, # Positive y-velocity
                               t_index > 300)) == True:
                        crossLinevec[jj] = 1

            return crossLinevec

    def get_closestpoint(self, xt):
        # Import full trajectory
        full_traj = self.get_reftraj(0,9000)

        # Reshape state into proper shape
        xt = np.reshape(xt,(6,1))

        # Find the point on the trajectory closest to the current state
        MinNorm = np.argmin(la.norm(full_traj[3:6,:] - xt[3:6], 1, axis=0))   

        return full_traj[:,MinNorm] 

    def get_goalpoint(self):
        # Import start/end point
        goal_coordinates = self.get_reftraj(0,0) 

        return np.reshape(goal_coordinates,(6,1)) 
 
    def get_reftraj(self, t_index, horizon, setpoint):
        """   (MPC Position Controller )
        This method returns the reference trajectory preview for 'horizon' timesteps given:
			- t_index: the current time index
            - horizon: number of elements in finite horizon
		""" 
        # Define next N time steps from current time index
        time_steps = np.linspace(t_index, t_index + horizon - 1, horizon) 
        
        # CIRCULAR TRAJECTORY
        if (self.traj_type == 0):
            ref = self.circular_traj(time_steps)
        
        # SETPOINT
        elif (self.traj_type == 1):
            ref = self.set_point(time_steps, setpoint)

        # SPECIAL TRAJECTORY
        else:
            ref = self.custom(time_steps) 
        
        return ref