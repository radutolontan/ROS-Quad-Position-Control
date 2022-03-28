# Add subfolders to path
import sys
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import mosek
sys.path.append('Optimal Control')
sys.path.append('Classic Control')

# Import classes and functions
from CFTOC_pyomo import CFTOC
from LMPC import LMPC

a = True
b = True

print(a*1+b*1)

