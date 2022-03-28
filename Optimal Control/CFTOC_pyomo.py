import numpy as np
import numpy.linalg as la
import pdb 
import scipy
import time as tm
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import mosek

class CFTOC(object):
    """ Constrained Finite Time Optimal Control (CFTOC)
	Methods:
		- solve_MPC: solves the CFTOC problem for MPC using LQR costs
        - solve_LMPC: solves CFTOC problem for LMPC using SS, Qfunction, and sigmoid cost
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t
	"""
    def __init__(self, N, dynamics, costs):
		# System constraints
        self.F_min = [-10,-10, 0]
        self.F_max = [ 10, 10,20]
        
		# System Dynamics 
        self.A = dynamics.A
        self.B = dynamics.B 
        self.C = dynamics.C
        self.A_k = dynamics.A_k
        self.B_k = dynamics.B_k
        self.C_k = dynamics.C_k
        self.n = self.A.shape[1]
        self.d = self.B.shape[1]

		# Stage Cost (h(x,u) = (x-x_ref)^TQ(x-x_ref) + u^TRu + (u_{k+1}-u_k)^tdR(u_{k+1}-u_k))
        self.Q = costs.Q
        self.R = costs.R
        self.dR = costs.dR
        
        # Terminal Cost (MPC only)
        self.Qf = costs.Qf
        
		# Initialize Predicted Trajectory
        self.xPred = []
        self.uPred = []
        
        # Selected running frequency and horizon
        self.freq = dynamics.freq
        self.N = N

    def solve_MPC(self, x0, u0, preview):
        """This method solves an MPC CFTOC problem given:
			- x0: initial state condition
            - u0: previously applied input
            - preview: reference trajectory preview
		""" 
        # Initialize optimization problem
        model = pyo.ConcreteModel()
        model.N = self.N
        model.TS = 1/self.freq
        model.nx = np.size(self.A, 0)
        model.nu = np.size(self.B, 1)
        
        # Length of finite optimization problem
        model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )  
        model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
        model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True )
  
        # Create pyomo objects for the linear system description
        model.A = self.A
        model.B = self.B
        model.C = self.C
        
        # Create pyomo objects for the cost matrices
        model.Q = self.Q
        model.Qf = self.Qf
        model.R = self.R
        model.dR = self.dR
        
    	# =====================================================================
        # =======================  DECISION VARIABLES  ========================
        # =====================================================================
        model.x = pyo.Var(model.xIDX, model.tIDX)
        model.u = pyo.Var(model.uIDX, model.tIDX)
            
    	# =====================================================================
        # ===========================  CONSTRAINTS  ===========================
        # =====================================================================

		# Initial Condition Constraints
        model.initial_conditions = pyo.Constraint(model.xIDX, 
                                                  rule=lambda model,i: model.x[i,0]==x0[i])
            
        # System Dynamics Constraints
        def LTI_system(model, i, t):
            return  model.x[i, t+1] - (model.x [i, t] + model.TS * (sum(model.A[i, j] * model.x[j, t] for j in model.xIDX) 
                                                         + sum(model.B[i, j] * model.u[j, t] for j in model.uIDX) + model.C[i])) == 0 if t < model.N else pyo.Constraint.Skip
        model.system_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=LTI_system)
 

        # Input Constraints
        model.input_constraints1 = pyo.Constraint(model.uIDX, model.tIDX, 
                                                  rule=lambda model,i,t: model.u[i,t]<=self.F_max[i])
        model.input_constraints2 = pyo.Constraint(model.uIDX, model.tIDX, 
                                                  rule=lambda model,i,t: model.u[i,t]>=self.F_min[i])
               
        # =====================================================================
        # ==============================  COST  ===============================
        # =====================================================================
        
        def objective(model):
            # Initialize costs with zero
            costX = 0.0
            costU = 0.0
            costdU = 0.0
            costTerminal = 0.0
            
            for t in model.tIDX:
                # STAGE COST ON STATES
                # J_k = (x_k - x_k_ref).T * Q * (x_k - x_k_ref)
                for i in model.xIDX:
                    for j in model.xIDX:
                        if t < model.N-1:
                            costX += (model.x[i, t] - preview[i,t]) * model.Q[i, j] * (model.x[j, t] - preview[j,t])
                
                for i in model.uIDX:
                    for j in model.uIDX:
                        if t < model.N:
                            # STAGE COST ON INPUTS
                            # Jinput = (u_k).T * R * (u_k) (stage cost on inputs)
                            costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
                            
                    # STAGE COST ON RATE OF INPUT CHANGE
                    # Jdinput = (u_{k+1} - u_k).T * dR * (u_{k+1} - u_k)
                    if t < model.N-1:
                        costdU += model.dR[i,i] * (model.u[i, t+1] - model.u[i, t])**2   
                    
            # Update stage cost on rate of input change to incorporate first input      
            for i in model.uIDX:
                costdU += model.dR[i,i] * (model.u[i, 0] - u0[i])**2  

            else:
                # TERMINAL COST ON STATES
                # Jf = (x_N - x_N_ref).T * Qf * (x_N - x_N_ref)
                for i in model.xIDX:
                    for j in model.xIDX:
                        costTerminal += (model.x[i, model.N-1] - preview[i,model.N-1]) * model.Qf[i, j] * (model.x[j, model.N-1] - preview[j,model.N-1])
                
            return costX + costU + costdU + costTerminal
        
        model.cost = pyo.Objective(rule = objective, sense = pyo.minimize)

        # =====================================================================
        # =============================  SOLVER  ==============================
        # =====================================================================
        
        # Initialize MOSEK solver and solve optimization problem
        solver = pyo.SolverFactory('mosek')
        results = solver.solve(model)
        
        # Check if solver found a feasible, bounded, optimal solution
        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            self.x_pred = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
            self.u_pred = np.asarray([model.u[:,t]() for t in model.tIDX]).T
    
        else:
            self.x_pred = 999
            self.u_pred = 999

    def solve_LMPC(self, x0, u0, max_dev, slack_goal, SS, Qfun, x_Q, x_G):
        """
        This method solves an LMPC CFTOC problem given:
			- x0: initial state condition
            - u0: previously applied input
            - max_dev: maximum allowable deviation from trajectory
            - slack_goal: used to define nearness to goal for optimal time cost
			- SS: contains a set of state and the terminal constraint is ConvHull(SS)
			- Qfun: cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
            - x_Q: closest point on trajectory coordinates in cartesian
            - x_G: goal state coordinates in cartesian
        """ 
        # Initialize optimization problem
        model = pyo.ConcreteModel()
        model.N = self.N
        model.TS = 1/self.freq
        model.nx = np.size(self.A, 0)
        model.nu = np.size(self.B, 1)
        model.SS_size = SS.shape[1]

        # Define Big M method constant
        big_M = 10**6
        
        # Length of finite optimization problem
        model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )  
        model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
        model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True )
        model.lIDX = pyo.Set( initialize= range(model.SS_size), ordered=True )  
  
        # Create pyomo objects for the linear system description
        model.A = self.A
        model.B = self.B
        model.C = self.C
        
        # Create pyomo objects for the cost matrices
        model.Q = self.Q
        model.Qf = self.Qf
        model.R = self.R
        model.dR = self.dR

        # LMPC Objects
        model.SS = SS
        model.Qfun = Qfun
        
    	# =====================================================================
        # =======================  DECISION VARIABLES  ========================
        # =====================================================================

        model.x = pyo.Var(model.xIDX, model.tIDX)
        model.u = pyo.Var(model.uIDX, model.tIDX)
        model.lambVar = pyo.Var(model.lIDX)
        model.bVar = pyo.Var(model.tIDX, within = pyo.Binary)
        
    	# =====================================================================
        # ===========================  CONSTRAINTS  ===========================
        # =====================================================================

		# Initial Condition Constraints
        model.initial_conditions = pyo.Constraint(model.xIDX, 
                                                  rule=lambda model,i: model.x[i,0]==x0[i])
            
        # System Dynamics Constraints
        def LTI_system(model, i, t):
            return  model.x[i, t+1] - (model.x [i, t] + model.TS * (sum(model.A[i, j] * model.x[j, t] for j in model.xIDX) 
                                                         + sum(model.B[i, j] * model.u[j, t] for j in model.uIDX) + model.C[i])) == 0 if t < model.N else pyo.Constraint.Skip
        model.system_constraints = pyo.Constraint(model.xIDX, model.tIDX, rule=LTI_system)
 

        # Input Constraints
        model.input_constraints1 = pyo.Constraint(model.uIDX, model.tIDX, 
                                                  rule=lambda model,i,t: model.u[i,t]<=self.F_max[i])
        model.input_constraints2 = pyo.Constraint(model.uIDX, model.tIDX, 
                                                  rule=lambda model,i,t: model.u[i,t]>=self.F_min[i])

        # State Corridor Constraints
        model.corridor_constraints1 = pyo.Constraint(range(3,6), model.tIDX, 
                                                  rule=lambda model,i,t: model.x[i,t]<=x_Q[i]+max_dev)
        model.corridor_constraints2 = pyo.Constraint(range(3,6), model.tIDX, 
                                                  rule=lambda model,i,t: model.x[i,t]>=x_Q[i]-max_dev)

        # Big M Constraints to Enable Optimal Time (force bVar to 0 close to goal and 1 otherwise)
        model.Big_M_constraints1 = pyo.Constraint(range(3,6), model.tIDX, 
                                                  rule=lambda model,i,t: x_G[i] - model.x[i,t] - model.bVar[t] * big_M <= slack_goal)
        model.Big_M_constraints2 = pyo.Constraint(range(3,6), model.tIDX, 
                                                  rule=lambda model,i,t: - x_G[i] + model.x[i,t] - model.bVar[t] * big_M <= slack_goal)
        
        # SS Constraints 1 - Positive lambda
        model.lambVar_positive = pyo.Constraint(model.lIDX, 
                                                rule=lambda model,i: model.lambVar[i]>=0)
        
        # SS Constraints 2 - Sum lambda to 1
        model.lambVar_sum = pyo.Constraint(expr=sum(model.lambVar[i] for i in model.lIDX) == 1 )

        # SS Constraints 3 - Final state inside SS            
        def enforce_SS(model, i):
            return sum(model.SS[i,j] * model.lambVar[j] for j in model.lIDX) == model.x[i, model.N-1]
        model.lambVar_inSS = pyo.Constraint(model.xIDX, rule=enforce_SS)     
                
        # =====================================================================
        # ==============================  COST  ===============================
        # =====================================================================
        
        def objective(model):
            # Initialize costs with zero
            costX = 0.0
            costU = 0.0
            costdU = 0.0
            costTerminal = 0.0
            
            for t in model.tIDX:
                # BINARY COST FOR OPTIMAL TIME
                # h_k = 1 if (-slack_goal < x_G - x_i < slack_goal); 0 else
                costX += 1 * model.bVar[t]
                
                for i in model.uIDX:
                    for j in model.uIDX:
                        if t < model.N:
                            # STAGE COST ON INPUTS
                            # Jinput = (u_k).T * R * (u_k) (stage cost on inputs)
                            costU += model.u[i, t] * model.R[i, j] * model.u[j, t]
                            
                    # STAGE COST ON RATE OF INPUT CHANGE
                    # Jdinput = (u_{k+1} - u_k).T * dR * (u_{k+1} - u_k)
                    if t < model.N-1:
                        costdU += model.dR[i,i] * (model.u[i, t+1] - model.u[i, t])**2   
                    
            # Update stage cost on rate of input change to incorporate first input      
            for i in model.uIDX:
                costdU += model.dR[i,i] * (model.u[i, 0] - u0[i])**2  
        
            # LMPC Value Function
            for l in model.lIDX:
                costTerminal += model.lambVar[l] * model.Qfun[0][l]
  
            return costX + costU + costdU + costTerminal
        model.cost = pyo.Objective(rule = objective, sense = pyo.minimize)
    
        # =====================================================================
        # =============================  SOLVER  ==============================
        # =====================================================================
        
        # Initialize MOSEK solver and solve optimization problem
        solver = pyo.SolverFactory('mosek')
        results = solver.solve(model)
        
        # Check if solver found a feasible, bounded, optimal solution
        if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
            self.x_pred = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX]).T
            self.u_pred = np.asarray([model.u[:,t]() for t in model.tIDX]).T
            self.lamb  = np.asarray([model.lambVar[t]() for t in model.lIDX]).T
        else:
            self.x_pred = 999
            self.u_pred = 999
    
    def model(self, x, u):
        """This method returns the time evolution of states i.e x{k+1} = Ax_k + Bu_k +C
			- x: state at previous time step (x_k)
            - u: computed input at previous time step (u_k)
		""" 
        return (np.dot(self.A_k,x) + np.squeeze(np.dot(self.B_k,u)) + self.C_k).tolist()





	



