"""
This class is used to create reference trajectories for all types of position controllers
"""
import numpy as np

class Trajectory(object):

    def __init__(self, freq, traj_type):
        # Trajectory types: 0 - circular; 1 - setpoint
        self.traj_type = traj_type 
        self.freq = freq

    # ====================================================================================
    # ============================== TYPES OF TRAJECTORIES ===============================
    # ====================================================================================

    def circular_traj(self, k):
         # k - time steps; freq - frequncy (Hz)
        r = 0.7
        omega = 1.2
        cruise_height = 0.8
        # Create storage and set Z- location to 0
        x = np.zeros((6,int(np.size(k))))
        
        # X - and Y - velocities are parametrized
        x[0,:] = - (omega*r) * np.sin((k/self.freq) * omega) 
        x[1,:] =   (omega*r) * np.cos((k/self.freq) * omega)
        # X - and Y - location are parametrized
        x[3,:] =  r * np.cos((k/self.freq) * omega) 
        x[4,:] =  r * np.sin((k/self.freq) * omega) 
        # Z - location is constant and equal to a height
        x[5,:] = cruise_height * np.ones(int(np.size(k)))

        return x

    def set_point(self, k, setpoint):
        # k - time steps; freq - frequncy (Hz)
        # Create storage for both velocities and locations
        # Set velocities to 0
        x = np.zeros((6,int(np.size(k))))
        # Set X- setpoint, Y- setpoint and Z- setpoint
        x[3,:] = setpoint[0] * np.ones(int(np.size(k)))
        x[4,:] = setpoint[1] * np.ones(int(np.size(k)))
        x[5,:] = setpoint[2] * np.ones(int(np.size(k)))

        return x

    # ====================================================================================
    # ============================== TRAJECTORY PROCESSING ===============================
    # ====================================================================================
    
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
                # Time index (t_index) is used to ensure the states at the beginning are not picked up
                if all((states[1,jj]>0, states[4,jj] >= 0.0, t_index > 300)) == True:
                    crossLinevec[jj] = 1
            return crossLinevec

    def get_closestpoint(self, xt):
        # Import full trajectory
        full_traj = self.get_reftraj(0,1000)

        # Reshape state into proper shape
        xt = np.reshape(xt,(6,1))

        # Find the point on the trajectory closest to the current state
        MinNorm = np.argmin(la.norm(full_traj[3:6,:] - xt[3:6], 1, axis=0))   

        return full_traj[:,MinNorm+1]  

    def get_reftraj(self, t_index, horizon, setpoint=0):
        """   (MPC Position Controller )
        This method returns the reference trajectory preview for 'horizon' timesteps given:
			- t_index: the current time index
            - horizon: number of elements in finite horizon
		""" 
        # Define next N time steps from current time index
        time_steps = np.linspace(t_index, t_index+horizon, horizon+1) 
        
        # CIRCULAR TRAJECTORY
        if (self.traj_type == 0):
            ref = self.circular_traj(time_steps)
        
        # SETPOINT
        elif (self.traj_type == 1):
            ref = self.set_point(time_steps, setpoint)
                
        return ref