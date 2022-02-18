import numpy as np
import pdb 
import scipy
from cvxpy import Variable, quad_form, norm, Problem, Minimize, MOSEK

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

		# Stage Cost (h(x,u) = (x-x_ref)^TQ(x-x_ref) +u^TRu)
        self.Q = costs.Q
        self.R = costs.R
        
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
        time_steps = np.linspace(time, time+self.N-1, self.N) 
        
        # CIRCULAR TRAJECTORY
        if (self.trajectory_type == 0):
            frequency = self.freq
            radius = 1.5 # (m)
            omega = 0.4 # (rad/s)
            height = 0.75 # (m)
            ref = circular_traj(time_steps, frequency, radius, omega, height)
        
        # SETPOINT
        elif (self.trajectory_type == 1):
            frequency = self.freq
            setpoint = np.array ([0,0,0.5]) # (m,m,m)
            ref = set_point(time_steps, frequency, setpoint)
                
        return ref
        
        
    def solve(self, x0, time_index, verbose = False, SS = None, Qfun = None, CVX = None):
        """This method solves an CFTOC problem given:
			- x0: initial condition
            - time_index: current time index, used to create reference trajectory preview
			- SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
			- Qfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
		""" 
    	# =====================================================================
        # =======================  DECISION VARIABLES  ========================
        # =====================================================================
        x = Variable((self.n, self.N+1))
        u = Variable((self.d, self.N))

		# lambdaVar multipliers if SS not empty --> used for terminal constraint
        if SS is not None:
            if CVX == True:
                lambVar = Variable((SS.shape[1], 1), boolean=False) # Initialize vector of variables
            else:
                lambVar = Variable((SS.shape[1], 1), boolean=True) # Initialize vector of variables

    	# =====================================================================
        # ===========================  CONSTRAINTS  ===========================
        # =====================================================================

		# Initial Constraints
        constr = [x[:,0] == x0[:]]
        
        # System Constrains
        for i in range(0, self.N):
            constr += [x[:,i+1] == self.A@x[:,i] + self.B@u[:,i] + self.C,]
					
        # Input Constraints
        for i in range(0, self.N):
            constr +=  [u[:,i] >= self.F_min[:],
						u[:,i] <=  self.F_max[:],]

		# Terminal Constraint if SS not empty --> enforce the terminal constraint
        if SS is not None:
            constr += [SS * lambVar[:,0] == x[:,self.N], # Terminal state \in ConvHull(SS)
						np.ones((1, SS.shape[1])) * lambVar[:,0] == 1, # Multiplies \lambda sum to 1
						lambVar >= 0] # Multiplier are positive definite
        
        # =====================================================================
        # ==============================  COST  ===============================
        # =====================================================================
        
        # Generate Trajectory Preview
        preview = self.get_reftraj(time_index)
        
		# Stage Cost (State, Input)
        cost = 0
        for i in range(0, self.N):
            # Running cost h(x,u) = (x-preview)^TQ(x-preview) + u^TRu
            cost += (1/2) * quad_form(x[:,i] - preview[:,i], self.Q) + norm(self.R @ u[:,i])**2
     		# cost += norm(self.Q**0.5*x[:,i])**2 + norm(self.R**0.5*u[:,i])**2

		# Terminal cost (Value function (LMPC only), Qf (MPC only))
        if SS is not None:
            cost += Qfun[0,:] * lambVar[:,0]  # It terminal cost is given by interpolation using \lambda
        else:
            cost += quad_form(x[:,self.N] - preview[:,self.N-1], self.Qf) # For MPC

        # =====================================================================
        # =============================  SOLVER  ==============================
        # =====================================================================
        
		# Solve the Finite Time Optimal Control Problem
        problem = Problem(Minimize(cost), constr)
        if CVX == True:
            problem.solve(verbose=verbose, solver=MOSEK) # I find that ECOS is better please use it when solving QPs
        else:
            problem.solve(verbose=verbose, solver=MOSEK)


		# Store the open-loop predicted trajectory
        self.xPred = x.value
        self.uPred = u.value	
        if SS is not None:
            self.lamb  = lambVar.value


    def model(self, x, u):
        """This method returns the time evolution of states i.e x{k+1} = Ax_k + Bu_k +C
			- x: state at previous time step (x_k)
            - u: computed input at previous time step (u_k)
		""" 
        return (np.dot(self.A,x) + np.squeeze(np.dot(self.B,u)) + self.C).tolist()





	

