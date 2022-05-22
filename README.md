This repository was created at UC Berkeley for the "Self-Training Quadcopter Control and Navigation" 2021-2022 MEng capstone project. It contains classical and optimization-based position controllers and is meant for Robot Operating System (ROS) Gazebo simulations and hardware experiments.

# Dependencies
- [qrotor_firmware](https://github.com/HybridRobotics/qrotor_firmware): created by the Hybrid Robotics research group (PRIVATE) 
- [PYOMO](http://www.pyomo.org/): optimization interpreter
- [MOSEK](https://www.mosek.com/): optimization QP solver

# Operation

## Trajectory Generation
Run Trajectory/trajectorytest.py and edit parameters to change maximum velocity along track, and scaling factor of track. This will generate a .para file, which you can visualize by running Trajectory/traj_test.py. To import the track into the position controller, create a new method inside Trajectory/Trajectory.py, selecting the appropriate .para file and trimming the reference position matrix such that there is no overlap.

## Python Simulation
You can visualize the performance of the controllers using point mass dynamics in Python. To do so, run LMPC_test.py. You are able to tune the optimal control cost matrices and PID gains in the class deffinitions at the begining of the script. The code draws on classes and methods inside the Optimal Control and Classic Control folders, and as such you need to navigate to the parent folder first, before running LMPC_test.py to enable system paths to work properly.

## Gazebo Simulation and Hardware Experiments
The script written for hardware simulations is fly.py. Ensure that all ROS infrastructure operates properly, that the controller is set in offboard mode. Finally, you can test the controller by running the command : "rosrun qrotor_firmware fly.py"
