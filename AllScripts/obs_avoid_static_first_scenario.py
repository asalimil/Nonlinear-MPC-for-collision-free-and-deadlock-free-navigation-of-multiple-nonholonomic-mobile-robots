#! /usr/bin/env python
import rospy
import numpy as np
import math
import numpy.matlib
from numpy import linalg as LA
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
# from sensor_msgs.msg import LaserScan
# import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize
from casadi import *

#=== ODOMETRY
def callback_odom(odom):
    global Xr
    xr = odom.pose.pose.position.x
    yr = odom.pose.pose.position.y
    qz = odom.pose.pose.orientation.z
    qw = odom.pose.pose.orientation.w
    th = 2*np.arcsin(qz)
    # thr = modify(th)
    thr = th
    Xr = np.array([[xr], [yr], [thr]])
    # Xr = np.array([[yr], [xr], [2*np.arcsin(qz)]])


#=== LIDAR
def callback_lidar(lidar):
    global Scan, Dmin, Dmin_idx, numRays
    Scan = list(lidar.ranges)
    for i in range(len(Scan)):
        if Scan[i] == np.Inf:
           Scan[i] = 3.5
    Dmin = np.amin(Scan)
    Dmin_idx = np.argmin(Scan)
    numRays = len(Scan)


#=== HORIZON SHIFT
def shift(T, t0, u):
    # st = x0
    con = np.transpose(u[0:1,0:])
    # f_value = f(st, con)
    # st = st + (T*f_value)
    # x0 = st.full()
    t0 = t0 + T
    ushape = np.shape(u)
    u0 = np.concatenate(( u[1:ushape[0],0:],  u[ushape[0]-1:ushape[0],0:]), axis=0)
    return t0, u0

#=== NODE
rospy.init_node('move_node')


#=== INITIALIZATION
T = 0.05                                                                        # [s]
N = 2                                                                           # prediction horizon
# rob_diam = 0.1

v_max = +0.22; v_min = -v_max
omega_max = +2.84; omega_min = -omega_max

x = SX.sym('x'); y = SX.sym('y'); theta = SX.sym('theta')
states = np.array([[x], [y], [theta]]); n_states = len(states)
# print("STATES: ", states)

v = SX.sym('v'); omega = SX.sym('omega')
controls = np.array([[v],[omega]]); n_controls = len(controls)
rhs = np.array([[v*np.cos(theta)],[v*np.sin(theta)],[omega]])                   # system r.h.s
# print("RHS: ", rhs)
rhs2 = np.array([[v*np.cos(theta)],[v*np.sin(theta)],[omega]])

f = Function('f',[states,controls],[rhs])                                       # nonlinear mapping function f(x,u)
# print("Function :", f)

U = SX.sym('U',n_controls,N);                                                   # Decision variables (controls)
P = SX.sym('P',n_states + n_states)                                             # parameters (which include the initial state and the reference state)
# print("U: ", U)
# print("P: ", P)

X = SX.sym('X',n_states,(N+1));
# print("X: ", X)                                                               # A vector that represents the states over the optimization problem

numRays = 24
H = np.eye(numRays)

A = SX.sym('Dinv', numRays, N)

obj = 0                                                                         # Objective function
                                                                                # Constraints vector
Q = np.zeros((3,3)); Q[0,0] = 1; Q[1,1] = 5; Q[2,2] = 0.1                           # weighing matrices (states)
R = np.zeros((2,2)); R[0,0] = 0.5; R[1,1] = 0.05                                  # weighing matrices (controls)

# print("Q: ", Q)
# print("R: ", R)

st  = X[:,0]                                                                    # initial state
# print("ST: ", st)

# scan_max = SX.sym('scan_max')
# scan_max = 3.5

g = st-P[0:3]                                                                   # initial condition constraints
# print("g0: ", g)
# rospy.spin()

# f2 = Function('f2',[states,controls],[rhs2])
# g2 = A[:,0]



# Xr = np.array([[0.0], [0.0], [0.0]])

# nRays = SX.sym('numRays', 1, 1)
# numRays = 24
# This weight matrix should be designed later
# print("L: ", L)
# distance to the nearest obstacle at each direction for N prediction horizon
# Scan = np.ones((numRays,1))*scan_max
# print("D: ", D)
# At what local angle the rays connects the robot to PO
# Beta0 = np.array(list(range(1,numRays+1)))*(2*math.pi)/numRays                  # SX.sym('Beta0', numRays, 1)
# Beta = SX.sym('Beta', numRays, N)
# print("Beta0: ", Beta0)
# PO = SX.sym('PO', 2*numRays, N)                             # PO: Point Obstacle, contains position of points on boundary of obstacles detected by lidar rays (2* means x and y position)
# print("PO: ", PO)
#
# ePO = np.zeros((2*numRays, 1))                         # E (unite vectors) of PO (2*numRays, N) unit vectors along the direction of lidar rays (ex, ey) for each lidar reading for N prediction horizon
# ePO = SX.sym('ePO', 2*numRays, N)
# print("Epo: ", Epo)
#
# Dist = SX.sym('Dist', numRays, N)
"""
#=== UPDATE CURRENT POINTS on OBSTACLES DETECTED by ROBOT & CALCULATE DISTANCE MATRIX in CURRENT LIDAR READINGS
Rotz_c = np.array([[np.cos(X[2,0]), -np.sin(X[2,0])], [np.sin(X[2,0]), np.cos(X[2,0])]])
for ray in range(numRays):
    # unit vector in local frame along with lidar rays
    ePO[2*ray:2*(ray+1),0] = np.array([[np.cos(Beta[ray,0])], [np.sin(Beta[ray,0])]])
    # points on obstacles by robot along with lidar rays in global frame
    PO[2*ray:2*(ray+1),0] =  X[0:2,0] + mtimes(Dist[ray,0],mtimes(Rotz_c,ePO[2*ray:2*(ray+1),0]))
    Dist[ray,0] = norm_1(PO[2*ray:2*(ray+1),0] - X[0:2,0])
    Dinv[ray,0] = scan_max/Dist[ray,0]                                          # scan_max (3.5 m) used for normalization



# print("Dinv0: ", Dinv[:,0])
# rospy.spin()
PO_c =  PO[:,0]
#=== CALCULATE H MATRIX (DISTANCE) FOR ALL POs DETECTED by ROBOT
beta = Beta[:,0]
for k in range(1,N):
    for ray in range(numRays):
        # PO in global frame at each prediction horizon k
        Beta[ray,k] = beta[ray,0] - U[1,k]*T
        ePO[2*ray:2*(ray+1),k] = np.array([[np.cos(Beta[ray,k])], [np.sin(Beta[ray,k])]])
        PO[2*ray:2*(ray+1),k] = X[0:2,k] - PO_c[2*ray:2*(ray+1),0]
        Dist[ray,k] = norm_1(PO[2*ray:2*(ray+1),0] - X[0:2,k])
        Dinv[ray,k] = scan_max/Dist[ray,k]


    beta = Beta[ray,k]
"""


#
# print("Dinv: ", Dinv)
# rospy.spin()
#=== UPDATE CONSTRAINTS
for k in range(N):
    st = X[:,k];  con = U[:,k]; a = A[:,k]
    # obj = obj  +  mtimes((st-P[3:6]).T,mtimes(Q,(st-P[3:6]))) + mtimes(con.T, mtimes(R, con))               # calculate obj
    # rospy.spin()
    obj = obj + mtimes((st-P[3:6]).T,mtimes(Q,(st-P[3:6]))) + mtimes(con.T, mtimes(R, con)) + mtimes(a.T, mtimes(H, a))               # calculate obj
    #---
    st_next = X[:,k+1]
    f_value = f(st,con)
    st_next_euler = st + (T*f_value)
    g = vertcat(g, st_next-st_next_euler)                                               # compute constraints
    #---
    a_next = A[:,k+1]
    f2_value = f2(a,st,st_next,con)
    a_next_euler = f2_value
    g2 = vertcat(g2, a_next-a_next_euler)                                               # compute constraints
    #---



# print("OBJ: ", obj)
# rospy.spin()

# Make the decision variable one column  vector
# OPT_variables = vertcat(reshape(X, 3*(N+1),1), reshape(U, 2*N, 1))
OPT_variables = vertcat(reshape(X, 3*(N+1),1), reshape(U, 2*N, 1), reshape(Dinv, 24*(N), 1), reshape(Beta, 24*(N), 1))
print("OPT: ", OPT_variables)

nlp_prob = {'f':obj, 'x':OPT_variables, 'g':g, 'p':P}
print("f: ", nlp_prob['f'])

opts = {'print_time': 0, 'ipopt':{'max_iter':2000, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6}}

# print("nlp_prob: ", nlp_prob)
# print("opts: ", opts)
solver = nlpsol('solver','ipopt', nlp_prob, opts)
rospy.spin()

args = {'lbg': np.zeros((1,3*(N+1))), 'ubg': np.zeros((1,3*(N+1))), \
'lbx': np.concatenate((np.matlib.repmat(np.array([[-10],[-10],[-np.Inf]]),N+1,1),np.matlib.repmat(np.array([[v_min],[omega_min]]),N,1)), axis=0), \
'ubx':np.concatenate((np.matlib.repmat(np.array([[+10],[+10],[+np.Inf]]),N+1,1),np.matlib.repmat(np.array([[v_max],[omega_max]]),N,1)), axis=0)}

rospy.spin()

#
sim_tim = 1000
t0 = 0;
x0 = np.array([[0.0], [0.0], [0.0]])                                                             # initial condition.

# print("x0: ", x0)
# xs = np.array([[+1.0], [+1.5], [0.0]])                                                        # Reference posture.

xx = np.zeros((n_states, int(sim_tim/T)))
# print("xx: ", xx[:,0:1])
xx[:,0:1] = x0                                                                    # xx contains the history of states
t = np.zeros(int(sim_tim/T))
# print("t: ", np.shape(t))
t[0] = t0

u0 = np.zeros((n_controls,N));                                                  # two control inputs for each robot
# print("u0: ", u0)
X0 = np.transpose(repmat(x0,1,N+1))                                             # initialization of the states decision variables
# print("X0", X0)
                                                                                # Maximum simulation time
mpciter = 0
xx1 = np.zeros((N+1,n_states,int(sim_tim/T)))
u_cl = np.zeros((int(sim_tim/T),n_controls))

# The main simulaton loop... it works as long as the error is greater than 10^-6 and the number of mpc steps is less than its maximum valueself.
goal_idx = 1
ng = 10

#=== Start MPC
while (not rospy.is_shutdown()):
    # subscriber odometry
    sub1 = rospy.Subscriber('/odom', Odometry, callback_odom)
    # subscriber scan lidar data
    sub2 = rospy.Subscriber('/scan', LaserScan, callback_lidar)
    #
    print("OBJ: ", obj)
    rospy.spin()
    #
    Dist = Scan
    # rospy.spin()
    # print("args: ", args)

    #=== Goal points
    if goal_idx == 1:
        xs = np.array([[1.0], [+0.5], [0.0]]) # reference pose
    elif goal_idx == 2:
        xs = np.array([[+0.0], [0.75], [-1.57]]) # reference pose
    elif goal_idx == 3:
        xs = np.array([[-0.5], [+0.5], [3.14]]) # reference pose
    elif goal_idx == 4:
        xs = np.array([[-0.5], [-0.75], [+0.785]]) # reference pose
    elif goal_idx == 5:
        xs = np.array([[0.75], [-0.75], [-0.785]]) # reference pose
    elif goal_idx == 6:
        xs = np.array([[0.0], [0.0], [0.0]]) # reference pose

    args['p'] = np.concatenate((x0, xs), axis=0)                                # set the values of the parameters vector
    # print("lbg: ", args['lbg'])
    # rospy.spin()
    # print("args.p: ", args['p'])

    # initial value of the optimization variables
    args['x0']  = np.concatenate((reshape(np.transpose(X0),3*(N+1),1), reshape(np.transpose(u0),2*N,1)), axis=0)
    # print("args: ", args['x0'])
    # print("args: ", args)

    sol = solver(x0=args['x0'], p=args['p'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'])

    solu = sol['x'][3*(N+1):]; solu_full = np.transpose(solu.full())
    u = np.transpose(reshape(solu_full, 2,N))                                    # get controls only from the solution
    # print("u: ", u)

    solx = sol['x'][0:3*(N+1)]; solx_full = np.transpose(solx.full())
    xx1[0:,0:3,mpciter] = np.transpose(reshape(solx_full, 3,N+1))                               # get solution TRAJECTORY
    # print("xx1: ", xx1[:,0:3,mpciter])

    u_cl[mpciter,0:] = u[0:1,0:]
    # print("u_cl: ", u_cl[mpciter,0:])

    t[mpciter] = t0

    # Apply the control and shift the solution
    t0, u0 = shift(T, t0, u)
    x0 = Xr
    #
    xx[0:,mpciter+1:mpciter+2] = x0
    # print("xx: ", xx)

    solX0 = sol['x'][0:3*(N+1)]; solX0_full = np.transpose(solX0.full())
    X0 = np.transpose(reshape(solX0_full, 3,N+1))                                # get solution TRAJECTORY

    # print("u: ", u)
    # Shift trajectory to initialize the next step
    X0 = np.concatenate((X0[1:,0:n_states+1], X0[N-1:N,0:n_states+1]), axis=0)

    # Move Robot
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    move = Twist()
    print(np.round(time.clock(),2), np.round(u_cl[mpciter,0]), np.round(u_cl[mpciter,1]), np.round(Xr[0],2), np.round(Xr[1],2), np.round(Xr[2],2))
    move.linear.x = u_cl[mpciter,0]                                                     # apply first optimal linear velocity
    move.angular.z = u_cl[mpciter,1]                                                    # apply first optimal angular velocity
    pub.publish(move)
    time.sleep(T)

    ne = LA.norm(Xr-xs)

    # Stop Condition
    if ne < 0.1:
        goal_idx = goal_idx + 1
        if goal_idx > ng:
            pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
            move = Twist()
            move.linear.x = 0; move.angular.z = 0
            pub.publish(move)
            print("Robot has arrived to GOAL point!")
            # rospy.spin()


    # print("mpciter, error: ", mpciter, LA.norm(x0-xs))
    mpciter = mpciter + 1

#=== Stop Robot
move.linear.x = 0.0                                                     # apply first optimal linear velocity
move.angular.z = 0.0                                                    # apply first optimal angular velocity
pub.publish(move)
print("THE END ...!")
