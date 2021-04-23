#! /usr/bin/env python
import rospy
import numpy as np
import math
import numpy.matlib
from numpy import linalg as LA
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import LaserScan
# import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize
from casadi import *

# Odometry
def callback_odom_first(odom):
    global Xr1, Xr1_init
    x1_init = Xr1_init[0,0]; y1_init = Xr1_init[1,0]; th1_init = Xr1_init[2,0];
    z1 = 0.0
    xr1 = odom.pose.pose.position.x
    yr1 = odom.pose.pose.position.y
    qz1 = odom.pose.pose.orientation.z
    qw1 = odom.pose.pose.orientation.w
    th1 = 2*np.arcsin(qz1)
    P1_local = np.array([[xr1], [yr1], [z1]])# Xr2 in local frame
    phi1 = th1 + th1_init
    R_g1 = np.array([[np.cos(th1_init), -np.sin(th1_init), 0.0],
                     [np.sin(th1_init), +np.cos(th1_init), 0.0],
                     [0.0,               0.0,              1.0]])
    P1 = np.matmul(R_g1,P1_local) + np.array([[x1_init], [y1_init], [z1]])
    # print("P1:", P1)
    # P1 position of robot 1 in global frame, R_g1: rotation of frame robot 1 in global frame
    Xr1 = np.array([[P1[0,0]], [P1[1,0]], [phi1]]) # pose of robot in global frame
    # print("Xr1: ", Xr1)
    # Xr = np.array([[yr], [xr], [2*np.arcsin(qz)]])

def callback_odom_second(odom):
    global Xr2, Xr2_init
    x2_init = Xr2_init[0,0]; y2_init = Xr2_init[1,0]; th2_init = Xr2_init[2,0];
    z2 = 0.0
    xr2 = odom.pose.pose.position.x
    yr2 = odom.pose.pose.position.y
    qz2 = odom.pose.pose.orientation.z
    qw2 = odom.pose.pose.orientation.w
    th2 = 2*np.arcsin(qz2)
    P2_local = np.array([[xr2], [yr2], [z2]]) # Xr2 in local frame
    phi2 = th2 + th2_init
    R_g2 = np.array([[np.cos(th2_init), -np.sin(th2_init), 0.0],
                     [np.sin(th2_init), +np.cos(th2_init), 0.0],
		             [0.0,          0.0,           1.0]])
    P2 = np.matmul(R_g2,P2_local) + np.array([[x2_init], [y2_init], [z2]])
    # print("P2:", P2)
    # P2 Position in global frame, R_g2: rotation of frame robot 2 in global frame
    Xr2 = np.array([[P2[0,0]], [P2[1,0]], [phi2]]) # pose of robot in global frame
    # print("Xr2: ", Xr2)
    # Xr = np.array([[yr], [xr], [2*np.arcsin(qz)]])


# Modify theta
def modify(th):
    if th >= -math.pi and th < 0:
        # th_modified = th + 2*math.pi
        th_modified = th
    else:
        th_modified = th
    return th_modified


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

#===============================================================================
# Succesful scenarios:
# 1:
# xs = np.array([[+2.0], [+2.0], [0.785], [-2.0], [-2.0], [3.926]])
# Xr1 = np.array([[-1.0], [-1.0], [0.785]])
# Xr2 = np.array([[+1.0], [+1.0], [3.926]])
# T = 0.01, N = 50, m = 2, M = m*(m-1)/2, dmin = 0.25
# 2:
#
#===============================================================================#
# node initialization
rospy.init_node('move_node')

# subscriber odometry
sub1 = rospy.Subscriber('/tb3_1/odom', Odometry, callback_odom_first)
sub2 = rospy.Subscriber('/tb3_2/odom', Odometry, callback_odom_second)


#
T = 0.05                                                                        # [s]
N = 70
m = 2
M = m*(m-1)/2                                                                         # prediction horizon
dmin = 0.15                                                                     # dmin^2
# rob_diam = 0.1

v_max = +0.22; v_min = -v_max
omega_max = +2.84; omega_min = -omega_max

x1 = SX.sym('x1'); y1 = SX.sym('y1'); theta1 = SX.sym('theta1')
x2 = SX.sym('x2'); y2 = SX.sym('y2'); theta2 = SX.sym('theta2')
#-->
# d12 = SX.sym('d12')
# d12 = (x1-x2)**2 + (y1-y2)**2
# states = np.array([[x1], [y1], [theta1], [x2], [y2], [theta2]]); n_states = len(states)
states = np.array([[x1], [y1], [theta1], [x2], [y2], [theta2]]); n_states = len(states)
print("states: ", states)

v1 = SX.sym('v1'); omega1 = SX.sym('omega1')
v2 = SX.sym('v2'); omega2 = SX.sym('omega2')
controls = np.array([[v1],[omega1], [v2], [omega2]]); n_controls = len(controls)
rhs = np.array([[v1*np.cos(theta1)],[v1*np.sin(theta1)],[omega1], [v2*np.cos(theta2)],[v2*np.sin(theta2)],[omega2]])                   # system r.h.s
# print("rhs: ", rhs)

f = Function('f',[states, controls],[rhs])                                       # nonlinear mapping function f(x,u)
# print("Function :", f)


U = SX.sym('U', n_controls, N);                                                   # Decision variables (controls)
P = SX.sym('P', n_states + n_states)                                             # parameters (which include the initial state and the reference state)
# print("U: ", U)
# print("P: ", P)

X = SX.sym('X',n_states,(N+1));
# A vector that represents the states over the optimization problem.
# print("X: ", X)


obj = 0                                                                        # Objective function
                                                                        # constraints vector
Q = np.zeros((6,6)); Q[0,0] = 1; Q[1,1] = 5; Q[2,2] = 0.1; Q[3,3] = 1; Q[4,4] = 5; Q[5,5] = 0.1   #0.1                           # weighing matrices (states)
R = np.zeros((4,4)); R[0,0] = 0.5; R[1,1] = 0.05; R[2,2] = 0.5; R[3,3] = 0.05       #0.05                                  # weighing matrices (controls)

#print("Q: ", Q)
#print("R: ", R)

#--->
# st  = X[:,0]                                                                    # initial state
st  = X[:,0]                                                                    # initial state
# print("st: ", st)

# d12 = Xr1[0]-Xr2[3])**2 + (Xr1[1]-Xr2[4])**2

g = vertcat(st-P[0:6], np.array([3.5]))                                                                   # initial condition constraints
# g = st-P[0:6]
# print("g: ", np.shape(g))

for k in range(N):
    #--->
    # st = X[:,k];  con = U[:,k]
    st = X[0:,k];  con = U[:,k]

    #--->
    d12 = (st[0]-st[3])**2 + (st[1]-st[4])**2
    #print("st: ", st); print("con: ", con)
    print("d12: ", d12)

    #--->
    # obj = obj  +  mtimes((st-P[6:]).T,mtimes(Q,(st-P[6:]))) + mtimes(con.T, mtimes(R, con))               # calculate obj
    obj = obj  +  mtimes((st-P[6:]).T,mtimes(Q,(st-P[6:]))) + mtimes(con.T, mtimes(R, con))               # calculate obj
    #print("Obj: ", obj)
    #--->
    # st_next = X[:,k+1];
    st_next = X[0:,k+1]

    #print("st_next: ", st_next)
    f_value = f(st,con);
    #print("f_value: ", f_value)
    st_next_euler = st + (T*f_value)
    #print("st_next_euler: ", st_next_euler)
    # g = vertcat(g, st_next-st_next_euler)                                               # compute constraints
    g = vertcat(g, vertcat(st_next-st_next_euler, d12))                                               # compute constraints

print("g: ", g)


# make the decision variable one column  vector
#--->
# OPT_variables = vertcat(reshape(X, 6*(N+1),1), reshape(U, 4*N, 1))
OPT_variables = vertcat(reshape(X, 6*(N+1),1), reshape(U, 4*N, 1))
# print("OPT: ", OPT_variables)

nlp_prob = {'f':obj,   'x':OPT_variables,   'g':g,   'p':P}
print("NLP: ", nlp_prob)

opts = {'print_time': 0, 'ipopt':{'max_iter':2000, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6}}
solver = nlpsol('solver','ipopt', nlp_prob, opts)


args = {'lbg': np.matlib.repmat(horzcat(np.zeros((1,6)), np.array([dmin*dmin])), 1, N+1), 'ubg': np.matlib.repmat(horzcat(np.zeros((1,6)), np.array([np.Inf])), 1, N+1), \
'lbx': np.concatenate((np.matlib.repmat(np.array([[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf]]),N+1,1),np.matlib.repmat(np.array([[v_min],[omega_min],[v_min],[omega_min]]),N,1)), axis=0), \
'ubx': np.concatenate((np.matlib.repmat(np.array([[+10],[+10],[+np.Inf],[+10],[+10],[+np.Inf]]),N+1,1),np.matlib.repmat(np.array([[v_max],[omega_max],[v_max],[omega_max]]),N,1)), axis=0)}
print("args: ", args)


print("lbg size: ", np.shape(args['lbg']))
print("ubg size: ", np.shape(args['ubg']))

sim_tim = 1000
t0 = 0;
x0 = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
Xr1_init = np.array([[-0.7112], [-0.7112], [+0.785]])
Xr2_init = np.array([[+0.7112], [+0.7112], [-2.356]])
Xr1 = np.array([[0.0], [0.0], [0.0]])
Xr2 = np.array([[0.0], [0.0], [0.0]])                                                            # initial condition.
#
print("Xr1: ", Xr1)
print("Xr2: ", Xr2)

#print("x0: ", x0)

# print("x0: ", x0)
xs = np.array([[+0.7112], [+0.7112], [0.785], [-0.7112], [-0.7112], [-2.356]])                                                        # Reference posture.

xx = np.zeros((n_states, int(sim_tim/T)))
# print("xx: ", xx[:,0:1])

xx[:,0:1] = x0                                                                    # xx contains the history of states
t = np.zeros(int(sim_tim/T))
# print("t: ", np.shape(t))
t[0] = t0

u0 = np.zeros((n_controls,N));                                                             # two control inputs for each robot
# print("u0: ", u0)
X0 = np.transpose(repmat(x0,1,N+1))                                                         # initialization of the states decision variables
print("X0", X0)

# Maximum simulation time
# Start MPC
mpciter = 0
xx1 = np.zeros((N+1,n_states,int(sim_tim/T)))
u_cl = np.zeros((int(sim_tim/T),n_controls))

#---
# the main simulaton loop... it works as long as the error is greater
# than 10^-6 and the number of mpc steps is less than its maximum
# value.
# main_loop = tic;

# Main Loop
while (LA.norm(x0-xs) > 5e-2) and (not rospy.is_shutdown()):
 # and (mpciter < sim_tim / T)

    args['p'] = np.concatenate((x0, xs), axis=0)                                # set the values of the parameters vector
    # print("args.p: ", args['p'])

    # initial value of the optimization variables
    args['x0']  = np.concatenate((reshape(np.transpose(X0),6*(N+1),1), reshape(np.transpose(u0),4*N,1)), axis=0)
    # print("args: ", args['x0'])
    # print("args: ", args)
    # print("lbx size: ", np.shape(args['lbx']))
    # print("ubx size: ", np.shape(args['ubx']))
    # print("p size: ", np.shape(args['p']))
    # print("x0 size: ", np.shape(args['x0']))


    sol = solver(x0=args['x0'], p=args['p'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'])
    # print("sol: ", sol['x'])


    solu = sol['x'][6*(N+1):]; solu_full = np.transpose(solu.full())
    u = np.transpose(reshape(solu_full, 4,N))                                    # get controls only from the solution
    # print("u: ", u)

    solx = sol['x'][0:6*(N+1)]; solx_full = np.transpose(solx.full())
    xx1[0:,0:6,mpciter] = np.transpose(reshape(solx_full, 6,N+1))                               # get solution TRAJECTORY
    # print("xx1: ", xx1[:,0:3,mpciter])

    u_cl[mpciter,0:] = u[0:1,0:]
    # print("u_cl: ", u_cl[mpciter,0:])

    t[mpciter] = t0

    # Apply the control and shift the solution
    t0, u0 = shift(T, t0, u)
    x0 = np.concatenate((Xr1, Xr2), axis=0)
    #
    xx[0:,mpciter+1:mpciter+2] = x0
    # print("xx: ", xx)

    solX0 = sol['x'][0:6*(N+1)]; solX0_full = np.transpose(solX0.full())
    X0 = np.transpose(reshape(solX0_full, 6, N+1))                                # get solution TRAJECTORY

    # print("u: ", u)
    # Shift trajectory to initialize the next step
    X0 = np.concatenate((X0[1:,0:n_states+1], X0[N-1:N,0:n_states+1]), axis=0)

    # Move Robot
    pub1 = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=10)
    move1 = Twist()
    pub2 = rospy.Publisher('/tb3_2/cmd_vel', Twist, queue_size=10)
    move2 = Twist()

    # print(time.clock(), u_cl[mpciter,0], u_cl[mpciter,1], u_cl[mpciter,2], u_cl[mpciter,3], Xr1, Xr2)
    print(time.clock(), u_cl[mpciter,0], u_cl[mpciter,1], u_cl[mpciter,2], u_cl[mpciter,3])
    move1.linear.x = u_cl[mpciter,0]                                                     # apply first optimal linear velocity
    move1.angular.z = u_cl[mpciter,1]                                                    # apply first optimal angular velocity
    move2.linear.x = u_cl[mpciter,2]                                                     # apply first optimal linear velocity
    move2.angular.z = u_cl[mpciter,3]                                                    # apply first optimal angular velocity
    pub1.publish(move1)
    pub2.publish(move2)
    # time.sleep(T)

    # print("mpciter, error: ", mpciter, LA.norm(x0-xs))
    mpciter = mpciter + 1

# Stop Robot
move1.linear.x = 0.0                                                     # apply first optimal linear velocity
move1.angular.z = 0.0                                                    # apply first optimal angular velocity
move2.linear.x = 0.0                                                     # apply first optimal linear velocity
move2.angular.z = 0.0
pub1.publish(move1)
pub2.publish(move2)
print("THE END ...!")
