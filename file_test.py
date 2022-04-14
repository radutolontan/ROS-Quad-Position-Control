# Add subfolders to path
import time
import sys
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import mosek

import threading
sys.path.append('Optimal Control')
sys.path.append('Classic Control')

# Import classes and functions
from CFTOC_pyomo import CFTOC
from LMPC import LMPC

a = 0
r = 9

def adder():
    global a
    while True:
        a += 1
        time.sleep(0.02)

def printamiu():
    global r
    r = input("Please input a number, sir: ")
    



hold_thread = threading.Thread(target=adder)
change_mode_thread = threading.Thread(target=printamiu)
hold_thread.start()
change_mode_thread.start()
change_mode_thread.join()

print("a: ",a)
print("r: ",r)
