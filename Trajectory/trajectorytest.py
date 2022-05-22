import numpy as np
from PathPlanning import RRTStar, Map
from TrajGen import trajGenerator, Helix_waypoints, Circle_waypoints
np.random.seed(7)

# 3D boxes   lx, ly, lz, hx, hy, hz
obstacles = [
             [45, 35, 0, 55, 60, 60],
             [-5, 65, 0, 30, 70, 100],
             [70, 50, 0, 80, 80, 100]]

# limits on map dimensions
bounds = np.array([0,150])
# create map with obstacles
mapobs = Map(obstacles, bounds, dim = 3)

#plan a path from start to goal
start1 = np.array([0,0,20])
goal1 = np.array([90,80,60])

rrt1 = RRTStar(start = start1, goal = goal1,
              Map = mapobs, max_iter = 200,
              goal_sample_rate = 0.1)

waypoints1, min_cost1 = rrt1.plan()

start2 = np.array([90,50,50])
goal2 = np.array([30,0,30])

rrt2 = RRTStar(start = start2, goal = goal2,
              Map = mapobs, max_iter = 200,
              goal_sample_rate = 0.1)
waypoints2, min_cost2 = rrt2.plan()
#plot the waypoints and obstacles
rrt1.draw_scene(waypoints1, ax= None)
rrt2.draw_scene(waypoints2, ax= None)
#Generate trajectory through waypoints
waypoints = np.vstack((waypoints1, waypoints2))
waypoints = np.insert(waypoints, len(waypoints), values=waypoints[0], axis=0) # add the start1 to form a circle
waypoints = np.insert(waypoints, len(waypoints), values=waypoints[1], axis=0) # add the second point in waypoints1 to make the start point smoother
waypoints = waypoints * 0.03 #scaling the tragectory to whatever we want
traj = trajGenerator(waypoints, max_vel = 0.75, gamma = 1e6)

#initialise simulation with given controller and trajectory
Tmax = traj.TS[-1]

t = np.arange(0,Tmax + 0.01,0.01)
pose = [] #record the position
vele = [] #record the velocity
accl = [] #record the acceleration
for i in t:
   p,v,a =traj.get_des_state(i)
   pose.append(p)
   vele.append(v)
   accl.append(a)
pose = np.array(pose)#position matrix
vele = np.array(vele)#velocity matrix
accl = np.array(accl)#acc matrix
para = np.hstack((vele, pose, accl))
para_delete = para[:858]
# np.savetxt("para.txt", np.array(para), delimiter =", ")
np.save('para10.npy',np.array(para))
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.gca(projection='3d')
x = pose[:,0]
y = pose[:,1]
z = pose[:,2]
ax.plot(x, y, z, label='parametric curve')
ax.legend()
plt.show()