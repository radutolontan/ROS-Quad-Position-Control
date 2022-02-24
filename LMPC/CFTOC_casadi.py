import numpy as np
import pdb 
import scipy
import time as tm
import casadi as csd 
import mosek
from trajectories import circular_traj, set_point


class CFTOC(object):
    """ Constrained Finite Time Optimal Control (CFTOC)
	Methods:
		- solve: solves the CFTOC problem given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t
	"""
    def __init__(self, N, trajectory_type, dynamics, costs):
		# System constraints
        self.F_min = [-10,-10, 0]
        self.F_max = [ 10, 10,20]
        
		# System Dynamics (x_{k+1} = Ax_k + Bu_k + C)
        self.A = dynamics.A_k
        self.B = dynamics.B_k 
        self.C = dynamics.C_k
        self.n = self.A.shape[1]
        self.d = self.B.shape[1]

		# Stage Cost (h(x,u) = (x-x_ref)^TQ(x-x_ref) +u^TRu +(u_{k+1}-u_k)^tdR(u_{k+1}-u_k))
        self.Q = costs.Q
        self.R = costs.R
        self.dR = costs.dR
        
        # Terminal Cost (MPC only)
        self.Qf = costs.Qf
        
		# Initialize Predicted Trajectory
        self.xPred = []
        self.uPred = []
        
        # Selected trajectory 
        self.trajectory_type = trajectory_type
        
        # Selected running frequency
        self.freq = dynamics.freq
        self.N = N
        
    def get_reftraj(self, time):
        """This method returns the reference trajectory preview for N timesteps given:
			- time: the current time index
            - type_traj: 0 for circular trajectory
		""" 
        # Define next N time steps from current time index
        time_steps = np.linspace(time, time+self.N, self.N+1) 
        
        # CIRCULAR TRAJECTORY
        if (self.trajectory_type == 0):
            frequency = self.freq
            radius = 0.7 # (m)
            omega = 0.9 # (rad/s)
            height = 0.8 # (m)
            ref = circular_traj(time_steps, frequency, radius, omega, height)
        
        # SETPOINT
        elif (self.trajectory_type == 1):
            frequency = self.freq
            setpoint = np.array ([0.1,0.1,0.1]) # (m,m,m)
            ref = set_point(time_steps, frequency, setpoint)
                
        return ref
        
        
    def solve(self, x0, u0, time_index, verbose = False, SS = None, Qfun = None, CVX = None):
        """This method solves an CFTOC problem given:
			- x0: initial state condition
            - u0: previously applied input
            - time_index: current time index, used to create reference trajectory preview
			- SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
			- Qfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
		""" 
        
        # Initialize CASADI problem
        costX = 0.0
        costU = 0.0
        costdU = 0.0
        costTerminal = 0.0
        optim_prob = csd.Opti()
        
        # Generate Trajectory Preview
        preview = self.get_reftraj(time_index)
        
    	# =====================================================================
        # =======================  DECISION VARIABLES  ========================
        # =====================================================================
        xvar = optim_prob.variable(self.n, self.N+1)
        uvar = optim_prob.variable(self.d, self.N)
        
		# (LMPC ONLY) initialize lambda
        if SS is not None:
            lambVar = optim_prob.variable(SS.shape[1])

    	# =====================================================================
        # ===========================  CONSTRAINTS  ===========================
        # =====================================================================

		# Initial Constraints
        optim_prob.subject_to(xvar[:, 0] == x0[:])
        
		# (LMPC ONLY) initialize safe set constraints
        if SS is not None:
            # Positive lambda
            optim_prob.subject_to(lambVar >= np.zeros(lambVar.shape[0]))
            # Sum lambda to 1
            optim_prob.subject_to(csd.mtimes(np.ones((1,lambVar.shape[0])), lambVar) == 1)
            # Final state in safe set
            optim_prob.subject_to(xvar[:,self.N] == csd.mtimes(SS, lambVar))


        for t in range(0, self.N):
            
            # System Constrains
            optim_prob.subject_to(xvar[:,t+1] == csd.mtimes(self.A, xvar[:,t]) 
                                  + csd.mtimes(self.B, uvar[:,t]) + self.C)
			
            for i in range(0, self.d):                     
            
                # Input Constraints
                optim_prob.subject_to(self.F_min[i] <= uvar[i, t])
                optim_prob.subject_to(uvar[i, t] <= self.F_max[i])
                
            # =================================================================
            # ==============================  COST  ===========================
            # =================================================================
            
            if t < self.N-1:
                # State Stage (Running) Cost
                # J_k = (x_k - x_k_ref).T * Q * (x_k - x_k_ref)
                costX += csd.mtimes((xvar[:,t] - preview[:,t]).T, 
                                    csd.mtimes(self.Q, (xvar[:,t] - preview[:,t])))
                
                # Stage Cost on Input Rate of Change
                # Jdinput = (u_{k+1} - u_k).T * dR * (u_{k+1} - u_k)
                costdU += csd.mtimes((uvar[:, t+1] - uvar[:, t]).T,
                                     csd.mtimes(self.dR, (uvar[:, t+1] - uvar[:, t])))

            # Input Stage (Runnning) Cost
            # Jinput = (u_k).T * R * (u_k) (stage cost on inputs)
            costU += csd.mtimes(uvar[:,t].T, csd.mtimes(self.R, uvar[:,t]))
            
        # Update stage cost on rate of input change to incorporate first input      
        costdU += csd.mtimes((uvar[:,0] - u0[:]).T, 
                             csd.mtimes(self.dR, (uvar[:,0] - u0[:])))
         
        if SS is not None:
            # (LMPC Only) Value Function
            costTerminal += csd.mtimes(Qfun, lambVar.T)
        else:
            # State Final (Terminal) Cost
            # Jf = (x_N - x_N_ref).T * Qf * (x_N - x_N_ref)
            costTerminal += csd.mtimes((xvar[:,self.N-1] - preview[:,self.N-1]).T, 
                                       csd.mtimes(self.Qf, (xvar[:,self.N-1] - preview[:,self.N-1])))
             
        # =====================================================================
        # =============================  SOLVER  ==============================
        # =====================================================================
        
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        optim_prob.minimize(costX + costU + costdU + costTerminal)
        optim_prob.solver('ipopt', option)
        
        # Solve the Finite Time Optimal Control Problem
        sol = optim_prob.solve()
        
        # Store the open-loop predicted trajectory
        self.x_pred = sol.value(xvar)
        self.u_pred = sol.value(uvar)
        
        if SS is not None:
            self.lamb  = sol.value(lambVar)
        


    def model(self, x, u):
        """This method returns the time evolution of states i.e x{k+1} = Ax_k + Bu_k +C
			- x: state at previous time step (x_k)
            - u: computed input at previous time step (u_k)
		""" 
        return (np.dot(self.A,x) + np.squeeze(np.dot(self.B,u)) + self.C).tolist()





	

