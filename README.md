This repository was created at UC Berkeley for the "Self-Training Quadcopter Control and Navigation" 2021-2022 MEng capstone project. It contains classical and optimization-based position controllers and is meant for Robot Operating System (ROS) Gazebo simulations and hardware experiments.

## Dependencies
- [qrotor_firmware](https://github.com/HybridRobotics/qrotor_firmware): created by the Hybrid Robotics research group (PRIVATE) 
- [PYOMO](http://www.pyomo.org/): optimization interpreter
- [MOSEK](https://www.mosek.com/): optimization QP solver

## Operation

# Trajectory Generation
Run Trajectory/trajectorytest.py and edit parameters to change maximum velocity along track, and scaling factor of track. This will generate a .para file, which you can visualize by running Trajectory/traj_test.py. To import the track into the position controller, create a new method inside Trajectory/Trajectory.py, selecting the appropriate .para file and trimming the reference position matrix such that there is no overlap.

