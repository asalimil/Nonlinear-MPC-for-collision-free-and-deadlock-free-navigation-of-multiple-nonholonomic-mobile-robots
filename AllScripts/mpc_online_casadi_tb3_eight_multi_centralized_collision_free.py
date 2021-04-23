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


#===============================================================================
def callback_odom_first(odom):
    global Xr1
    xr1 = odom.pose.pose.position.x
    yr1 = odom.pose.pose.position.y
    qz1 = odom.pose.pose.orientation.z
    qw1 = odom.pose.pose.orientation.w
    th1 = 2*np.arcsin(qz1)
    thr1 = modify(th1)
    Xr1 = np.array([[xr1], [yr1], [thr1]])

def callback_odom_second(odom):
    global Xr2
    xr2 = odom.pose.pose.position.x
    yr2 = odom.pose.pose.position.y
    qz2 = odom.pose.pose.orientation.z
    qw2 = odom.pose.pose.orientation.w
    th2 = 2*np.arcsin(qz2)
    thr2 = modify(th2)
    Xr2 = np.array([[xr2], [yr2], [thr2]])

def callback_odom_third(odom):
    global Xr3
    xr3 = odom.pose.pose.position.x
    yr3 = odom.pose.pose.position.y
    qz3 = odom.pose.pose.orientation.z
    qw3 = odom.pose.pose.orientation.w
    th3 = 2*np.arcsin(qz3)
    thr3 = modify(th3)
    Xr3 = np.array([[xr3], [yr3], [thr3]])

def callback_odom_fourth(odom):
    global Xr4
    xr4 = odom.pose.pose.position.x
    yr4 = odom.pose.pose.position.y
    qz4 = odom.pose.pose.orientation.z
    qw4 = odom.pose.pose.orientation.w
    th4 = 2*np.arcsin(qz4)
    thr4 = modify(th4)
    Xr4 = np.array([[xr4], [yr4], [thr4]])

def callback_odom_fifth(odom):
    global Xr5
    xr5 = odom.pose.pose.position.x
    yr5 = odom.pose.pose.position.y
    qz5 = odom.pose.pose.orientation.z
    qw5 = odom.pose.pose.orientation.w
    th5 = 2*np.arcsin(qz5)
    thr5 = modify(th5)
    Xr5 = np.array([[xr5], [yr5], [thr5]])

def callback_odom_sixth(odom):
    global Xr6
    xr6 = odom.pose.pose.position.x
    yr6 = odom.pose.pose.position.y
    qz6 = odom.pose.pose.orientation.z
    qw6 = odom.pose.pose.orientation.w
    th6 = 2*np.arcsin(qz6)
    thr6 = modify(th6)
    Xr6 = np.array([[xr6], [yr6], [thr6]])

def callback_odom_seventh(odom):
    global Xr7
    xr7 = odom.pose.pose.position.x
    yr7 = odom.pose.pose.position.y
    qz7 = odom.pose.pose.orientation.z
    qw7 = odom.pose.pose.orientation.w
    th7 = 2*np.arcsin(qz7)
    thr7 = modify(th7)
    Xr7 = np.array([[xr7], [yr7], [thr7]])

def callback_odom_eighth(odom):
    global Xr8
    xr8 = odom.pose.pose.position.x
    yr8 = odom.pose.pose.position.y
    qz8 = odom.pose.pose.orientation.z
    qw8 = odom.pose.pose.orientation.w
    th8 = 2*np.arcsin(qz8)
    thr8 = modify(th8)
    Xr8 = np.array([[xr8], [yr8], [thr8]])

#===============================================================================

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
sub3 = rospy.Subscriber('/tb3_3/odom', Odometry, callback_odom_third)
sub4 = rospy.Subscriber('/tb3_4/odom', Odometry, callback_odom_fourth)
sub5 = rospy.Subscriber('/tb3_5/odom', Odometry, callback_odom_fifth)
sub6 = rospy.Subscriber('/tb3_6/odom', Odometry, callback_odom_sixth)
sub7 = rospy.Subscriber('/tb3_7/odom', Odometry, callback_odom_seventh)
sub8 = rospy.Subscriber('/tb3_8/odom', Odometry, callback_odom_eighth)


#==============================================================================#
# T = 0.02                                                                        # [s]
# N = 50
T = 0.02
N = 5
m = 8
M = m*(m-1)/2                                                                      # prediction horizon
dmin = 0.25                                                                     # dmin^2
# rob_diam = 0.1

v_max = +0.22; v_min = -v_max
omega_max = +2.84; omega_min = -omega_max

x1 = SX.sym('x1'); y1 = SX.sym('y1'); theta1 = SX.sym('theta1')
x2 = SX.sym('x2'); y2 = SX.sym('y2'); theta2 = SX.sym('theta2')
x3 = SX.sym('x3'); y3 = SX.sym('y3'); theta3 = SX.sym('theta3')
x4 = SX.sym('x4'); y4 = SX.sym('y4'); theta4 = SX.sym('theta4')
x5 = SX.sym('x5'); y5 = SX.sym('y5'); theta5 = SX.sym('theta5')
x6 = SX.sym('x6'); y6 = SX.sym('y6'); theta6 = SX.sym('theta6')
x7 = SX.sym('x7'); y7 = SX.sym('y7'); theta7 = SX.sym('theta7')
x8 = SX.sym('x8'); y8 = SX.sym('y8'); theta8 = SX.sym('theta8')

#-->

# d12 = SX.sym('d12')
# d12 = (x1-x2)**2 + (y1-y2)**2
# states = np.array([[x1], [y1], [theta1], [x2], [y2], [theta2]]); n_states = len(states)
states = np.array([[x1], [y1], [theta1], [x2], [y2], [theta2], [x3], [y3], [theta3], [x4], [y4], [theta4], [x5], [y5], [theta5] , [x6], [y6], [theta6] , [x7], [y7], [theta7] , [x8], [y8], [theta8]]); n_states = len(states)
# print("states: ", states)

v1 = SX.sym('v1'); omega1 = SX.sym('omega1')
v2 = SX.sym('v2'); omega2 = SX.sym('omega2')
v3 = SX.sym('v3'); omega3 = SX.sym('omega3')
v4 = SX.sym('v4'); omega4 = SX.sym('omega4')
v5 = SX.sym('v5'); omega5 = SX.sym('omega5')
v6 = SX.sym('v6'); omega6 = SX.sym('omega6')
v7 = SX.sym('v7'); omega7 = SX.sym('omega7')
v8 = SX.sym('v8'); omega8 = SX.sym('omega8')



controls = np.array([[v1],[omega1], [v2], [omega2], [v3], [omega3], [v4], [omega4], [v5], [omega5] , [v6], [omega6], [v7], [omega7], [v8], [omega8]]); n_controls = len(controls)
rhs = np.array([[v1*np.cos(theta1)],[v1*np.sin(theta1)],[omega1], [v2*np.cos(theta2)],[v2*np.sin(theta2)],[omega2], [v3*np.cos(theta3)],[v3*np.sin(theta3)],[omega3], [v4*np.cos(theta4)],[v4*np.sin(theta4)],[omega4], [v5*np.cos(theta5)],[v5*np.sin(theta5)],[omega5], \
[v6*np.cos(theta6)],[v6*np.sin(theta6)],[omega6],[v7*np.cos(theta7)],[v7*np.sin(theta7)],[omega7],[v8*np.cos(theta8)],[v8*np.sin(theta8)],[omega8]])                   # system r.h.s
# print("rhs: ", rhs)

f = Function('f',[states, controls],[rhs])                                       # nonlinear mapping function f(x,u)
# print("Function :", f)

U = SX.sym('U', n_controls, N);                                                   # Decision variables (controls)
P = SX.sym('P', 2*n_states)                                             # parameters (which include the initial state and the reference state)
# print("U: ", U)
# print("P: ", P)

X = SX.sym('X',n_states,(N+1));
# A vector that represents the states over the optimization problem.
# print("X: ", X)


obj = 0                                                                        # Objective function
                                                                        # constraints vector
Q = np.zeros((n_states,n_states))
Q[0,0] = 1; Q[1,1] = 5; Q[2,2] = 0.1;
Q[3,3] = 1; Q[4,4] = 5; Q[5,5] = 0.1   #0.1                           # weighing matrices (states)
Q[6,6] = 1; Q[7,7] = 5; Q[8,8] = 0.1;
Q[9,9] = 1; Q[10,10] = 5; Q[11,11] = 0.1   #0.1                           # weighing matrices (states)
Q[12,12] = 1; Q[13,13] = 5; Q[14,14] = 0.1;
Q[15,15] = 1; Q[16,16] = 5; Q[17,17] = 0.1;
Q[18,18] = 1; Q[19,19] = 5; Q[20,20] = 0.1;
Q[21,21] = 1; Q[22,22] = 5; Q[23,23] = 0.1;


R = np.zeros((n_controls,n_controls))
R[0,0] = 0.5; R[1,1] = 0.05;
R[2,2] = 0.5; R[3,3] = 0.05       #0.05                                  # weighing matrices (controls)
R[4,4] = 0.5; R[5,5] = 0.05;
R[6,6] = 0.5; R[7,7] = 0.05       #0.05
R[8,8] = 0.5; R[9,9] = 0.05
R[10,10] = 0.5; R[11,11] = 0.05
R[12,12] = 0.5; R[13,13] = 0.05
R[14,14] = 0.5; R[15,15] = 0.05




#print("Q: ", Q)
#print("R: ", R)

#--->
# st  = X[:,0]                                                                    # initial state
st  = X[:,0]                                                                    # initial state
# print("st: ", st)
# print("P: ", P)


g = vertcat(st-P[0:n_states], np.matlib.repmat(np.array([3.5]), M, 1))                                                                   # initial condition constraints
# g = st-P[0:6]
# print("g: ", np.shape(g))

for k in range(N):
    #--->
    # st = X[:,k];  con = U[:,k]
    st = X[0:,k];  con = U[:,k]

    #--->
    d12 = (st[0]-st[3])**2 + (st[1]-st[4])**2
    d13 = (st[0]-st[6])**2 + (st[1]-st[7])**2
    d14 = (st[0]-st[9])**2 + (st[1]-st[10])**2
    d15 = (st[0]-st[12])**2 + (st[1]-st[13])**2
    d16 = (st[0]-st[15])**2 + (st[1]-st[16])**2
    d17 = (st[0]-st[18])**2 + (st[1]-st[19])**2
    d18 = (st[0]-st[21])**2 + (st[1]-st[22])**2
    #
    d23 = (st[3]-st[6])**2 + (st[4]-st[7])**2
    d24 = (st[3]-st[9])**2 + (st[4]-st[10])**2
    d25 = (st[3]-st[12])**2 + (st[4]-st[13])**2
    d26 = (st[3]-st[15])**2 + (st[4]-st[16])**2
    d27 = (st[3]-st[18])**2 + (st[4]-st[19])**2
    d28 = (st[3]-st[21])**2 + (st[4]-st[22])**2
    #
    d34 = (st[6]-st[9])**2 + (st[7]-st[10])**2
    d35 = (st[6]-st[12])**2 + (st[7]-st[13])**2
    d36 = (st[6]-st[15])**2 + (st[7]-st[16])**2
    d37 = (st[6]-st[18])**2 + (st[7]-st[19])**2
    d38 = (st[6]-st[21])**2 + (st[7]-st[22])**2
    #
    d45 = (st[9]-st[12])**2 + (st[10]-st[13])**2
    d46 = (st[9]-st[15])**2 + (st[10]-st[16])**2
    d47 = (st[9]-st[18])**2 + (st[10]-st[19])**2
    d48 = (st[9]-st[21])**2 + (st[10]-st[22])**2
    #
    d56 = (st[12]-st[15])**2 + (st[13]-st[16])**2
    d57 = (st[12]-st[18])**2 + (st[13]-st[19])**2
    d58 = (st[12]-st[21])**2 + (st[13]-st[22])**2
    #
    d67 = (st[15]-st[18])**2 + (st[16]-st[19])**2
    d68 = (st[15]-st[21])**2 + (st[16]-st[22])**2
    #
    d78 = (st[18]-st[21])**2 + (st[19]-st[22])**2
    #print("st: ", st); print("con: ", con)
    # print("d12: ", d12)

    # print("st: ", st)
    # print("P[]: ", P[n_states:])
    #--->
    # obj = obj  +  mtimes((st-P[6:]).T,mtimes(Q,(st-P[6:]))) + mtimes(con.T, mtimes(R, con))               # calculate obj
    obj = obj  +  mtimes((st-P[n_states:]).T,mtimes(Q,(st-P[n_states:]))) + mtimes(con.T, mtimes(R, con))               # calculate obj
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
    g = vertcat(g, vertcat(vertcat(vertcat(\
    vertcat(vertcat(vertcat(vertcat(vertcat( \
    vertcat(vertcat(vertcat(vertcat(vertcat(vertcat(vertcat(vertcat(vertcat(vertcat(\
    vertcat(vertcat(vertcat(vertcat(vertcat(vertcat(vertcat(vertcat(vertcat(st_next-st_next_euler, d12), d13), \
    d14), d15), d16), d17), d18), d23), d24), d25), d26), d27), d28), \
    d34), d35), d36), d37), d38), d45), d46), d47), d48), \
    d56), d57), d58), \
    d67), d68), \
    d78)
     # compute constraints

# print("g: ", g)

# make the decision variable one column  vector
#--->
# OPT_variables = vertcat(reshape(X, 6*(N+1),1), reshape(U, 4*N, 1))
OPT_variables = vertcat(reshape(X, n_states*(N+1),1), reshape(U, n_controls*N, 1))
# print("OPT: ", OPT_variables)

nlp_prob = {'f':obj,  'x':OPT_variables,   'g':g,   'p':P}
# print("NLP: ", nlp_prob)

opts = {'print_time': 0, 'ipopt':{'max_iter':2000, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6}}
solver = nlpsol('solver','ipopt', nlp_prob, opts)


args = {'lbg': np.matlib.repmat(horzcat(np.zeros((1,n_states)), np.matlib.repmat(np.array([dmin*dmin]), 1, M)), 1, N+1), \
'ubg': np.matlib.repmat(horzcat(np.zeros((1,n_states)), np.matlib.repmat(np.array([np.Inf]), 1, M)), 1, N+1), \
'lbx': np.concatenate((np.matlib.repmat(np.array([[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf],[-10],[-10],[-np.Inf]]),N+1,1),np.matlib.repmat(np.array([[v_min],[omega_min],[v_min],[omega_min], [v_min],[omega_min], [v_min],[omega_min], [v_min],[omega_min],[v_min],[omega_min],[v_min],[omega_min],[v_min],[omega_min]]),N,1)), axis=0), \
'ubx': np.concatenate((np.matlib.repmat(np.array([[+10],[+10],[+np.Inf],[+10],[+10],[+np.Inf], [+10],[+10],[+np.Inf], [+10],[+10],[+np.Inf], [+10],[+10],[+np.Inf],[+10],[+10],[+np.Inf],[+10],[+10],[+np.Inf],[+10],[+10],[+np.Inf]]),N+1,1),np.matlib.repmat(np.array([[v_max],[omega_max],[v_max],[omega_max],[v_max],[omega_max],[v_max],[omega_max],[v_max],[omega_max],[v_max],[omega_max],[v_max],[omega_max],[v_max],[omega_max]]),N,1)), axis=0)}
# print("args: ", args)


# print("lbg size: ", np.shape(args['lbg']))
# print("ubg size: ", np.shape(args['ubg']))

sim_tim = 1000
t0 = 0;
x0 = np.array([[-1.0], [+1.0], [-0.785], [+1.0], [+1.0], [-2.356], [+1.0], [-1.0], [+2.356], [-1.0], [-1.0], [0.785], [0.0], [0.0], [0.0], [+1.0], [+1.0], [-2.356], [+1.0], [+1.0], [-2.356], [+1.0], [+1.0], [-2.356]])
Xr1 = np.array([[0.866], [+0.5], [-2.618]])
Xr2 = np.array([[+0.5], [+0.866], [-2.094]])                                                            # initial condition.
Xr3 = np.array([[-0.5], [+0.866], [-1.047]])
Xr4 = np.array([[-0.866], [0.5], [-0.523]])                                                            # initial condition.
Xr5 = np.array([[-0.866], [-0.5], [0.523]])
Xr6 = np.array([[-0.5], [-0.866], [1.047]])
Xr7 = np.array([[+0.5], [-0.866], [2.094]])                                                            # initial condition.
Xr8 = np.array([[0.866], [-0.5], [+2.618]])


#
# print("Xr1: ", Xr1)
# print("Xr2: ", Xr2)

#print("x0: ", x0)

# print("x0: ", x0)
# Reference point
xs = np.array([[-0.866], [-0.5], [-2.618], [-0.5], [-0.866], [-2.094], [+0.5], [-0.866], [-1.047],\
 [0.866], [-0.5], [-0.523], [0.866], [0.5], [+0.523] ,\
 [+0.5], [+0.866], [+1.047], [-0.5], [+0.866], [2.094], [-0.866], [+0.5], [2.618]])                                                        # Reference posture.

xx = np.zeros((n_states, int(sim_tim/T)))
# print("xx: ", xx[:,0:1])

xx[:,0:1] = x0                                                                    # xx contains the history of states
t = np.zeros(int(sim_tim/T))
# print("t: ", np.shape(t))
t[0] = t0

u0 = np.zeros((n_controls,N));                                                             # two control inputs for each robot
# print("u0: ", u0)
X0 = np.transpose(repmat(x0,1,N+1))                                                         # initialization of the states decision variables
# print("X0", X0)

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
while (LA.norm(x0-xs) > 1e-1) and (not rospy.is_shutdown()):
 # and (mpciter < sim_tim / T)

    args['p'] = np.concatenate((x0, xs), axis=0)                                # set the values of the parameters vector
    # print("args.p: ", args['p'])

    # initial value of the optimization variables
    args['x0']  = np.concatenate((reshape(np.transpose(X0),n_states*(N+1),1), reshape(np.transpose(u0),n_controls*N,1)), axis=0)
    # print("args: ", args['x0'])
    # print("args: ", args)
    # print("lbx size: ", np.shape(args['lbx']))
    # print("ubx size: ", np.shape(args['ubx']))
    # print("p size: ", np.shape(args['p']))
    # print("x0 size: ", np.shape(args['x0']))


    sol = solver(x0=args['x0'], p=args['p'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'])
    # print("sol: ", sol['x'])


    solu = sol['x'][n_states*(N+1):]; solu_full = np.transpose(solu.full())
    u = np.transpose(reshape(solu_full, n_controls,N))                                    # get controls only from the solution
    # print("u: ", u)

    solx = sol['x'][0:n_states*(N+1)]; solx_full = np.transpose(solx.full())
    xx1[0:,0:n_states,mpciter] = np.transpose(reshape(solx_full, n_states,N+1))                               # get solution TRAJECTORY
    # print("xx1: ", xx1[:,0:3,mpciter])

    u_cl[mpciter,0:] = u[0:1,0:]
    # print("u_cl: ", u_cl[mpciter,0:])

    t[mpciter] = t0

    # Apply the control and shift the solution
    t0, u0 = shift(T, t0, u)
    x0 = np.concatenate((np.concatenate((np.concatenate((np.concatenate((np.concatenate((\
    np.concatenate((np.concatenate(\
    (Xr1, Xr2), axis=0), Xr3), axis=0), Xr4), axis=0), Xr5), axis=0), Xr6), axis=0)\
    , Xr7), axis=0), Xr8), axis=0)

    # print("x0: ", x0)
    #
    xx[0:,mpciter+1:mpciter+2] = x0
    # print("xx: ", xx)

    solX0 = sol['x'][0:n_states*(N+1)]; solX0_full = np.transpose(solX0.full())
    X0 = np.transpose(reshape(solX0_full, n_states, N+1))                                # get solution TRAJECTORY

    # print("u: ", u)
    # Shift trajectory to initialize the next step
    X0 = np.concatenate((X0[1:,0:n_states+1], X0[N-1:N,0:n_states+1]), axis=0)

    # Move Robot
    pub1 = rospy.Publisher('/tb3_1/cmd_vel', Twist, queue_size=10)
    move1 = Twist()
    pub2 = rospy.Publisher('/tb3_2/cmd_vel', Twist, queue_size=10)
    move2 = Twist()
    pub3 = rospy.Publisher('/tb3_3/cmd_vel', Twist, queue_size=10)
    move3 = Twist()
    pub4 = rospy.Publisher('/tb3_4/cmd_vel', Twist, queue_size=10)
    move4 = Twist()
    pub5 = rospy.Publisher('/tb3_5/cmd_vel', Twist, queue_size=10)
    move5 = Twist()
    pub6 = rospy.Publisher('/tb3_6/cmd_vel', Twist, queue_size=10)
    move6 = Twist()
    pub7 = rospy.Publisher('/tb3_7/cmd_vel', Twist, queue_size=10)
    move7 = Twist()
    pub8 = rospy.Publisher('/tb3_8/cmd_vel', Twist, queue_size=10)
    move8 = Twist()


    print("v1, w1, v2, w2, v3, w3, v4, w4, v5, w5, v6, w6, v7, w7, v8, w8: ", u_cl[mpciter,0], u_cl[mpciter,1], u_cl[mpciter,2], u_cl[mpciter,3], u_cl[mpciter,4], u_cl[mpciter,5], u_cl[mpciter,6], u_cl[mpciter,7], u_cl[mpciter,8], u_cl[mpciter,9],u_cl[mpciter,10], u_cl[mpciter,11], u_cl[mpciter,12], u_cl[mpciter,13], u_cl[mpciter,14], u_cl[mpciter,15])
    move1.linear.x = u_cl[mpciter,0]                                                     # apply first optimal linear velocity
    move1.angular.z = u_cl[mpciter,1]                                                    # apply first optimal angular velocity
    move2.linear.x = u_cl[mpciter,2]                                                     # apply first optimal linear velocity
    move2.angular.z = u_cl[mpciter,3]                                                    # apply first optimal angular velocity
    move3.linear.x = u_cl[mpciter,4]                                                     # apply first optimal linear velocity
    move3.angular.z = u_cl[mpciter,5]                                                    # apply first optimal angular velocity
    move4.linear.x = u_cl[mpciter,6]                                                     # apply first optimal linear velocity
    move4.angular.z = u_cl[mpciter,7]                                                    # apply first optimal angular velocity
    move5.linear.x = u_cl[mpciter,8]                                                     # apply first optimal linear velocity
    move5.angular.z = u_cl[mpciter,9]                                                    # apply first optimal angular velocity
    move6.linear.x = u_cl[mpciter,10]                                                     # apply first optimal linear velocity
    move6.angular.z = u_cl[mpciter,11]                                                    # apply first optimal angular velocity
    move7.linear.x = u_cl[mpciter,12]                                                     # apply first optimal linear velocity
    move7.angular.z = u_cl[mpciter,13]                                                    # apply first optimal angular velocity
    move8.linear.x = u_cl[mpciter,14]                                                     # apply first optimal linear velocity
    move8.angular.z = u_cl[mpciter,15]                                                    # apply first optimal angular velocity
    #
    pub1.publish(move1)
    pub2.publish(move2)
    pub3.publish(move3)
    pub4.publish(move4)
    pub5.publish(move5)
    pub6.publish(move6)
    pub7.publish(move7)
    pub8.publish(move8)
    time.sleep(T)

    # print("mpciter, error: ", mpciter, LA.norm(x0-xs))
    mpciter = mpciter + 1

# Stop Robot
move1.linear.x = 0.0                                                     # apply first optimal linear velocity
move1.angular.z = 0.0                                                    # apply first optimal angular velocity
move2.linear.x = 0.0                                                     # apply first optimal linear velocity
move2.angular.z = 0.0
move3.linear.x = 0.0                                                     # apply first optimal linear velocity
move3.angular.z = 0.0
move4.linear.x = 0.0                                                     # apply first optimal linear velocity
move4.angular.z = 0.0
move5.linear.x = 0.0                                                     # apply first optimal linear velocity
move5.angular.z = 0.0
move6.linear.x = 0.0                                                     # apply first optimal linear velocity
move6.angular.z = 0.0
move7.linear.x = 0.0                                                     # apply first optimal linear velocity
move7.angular.z = 0.0
move8.linear.x = 0.0                                                     # apply first optimal linear velocity
move8.angular.z = 0.0
#
pub1.publish(move1)
pub2.publish(move2)
pub3.publish(move3)
pub4.publish(move4)
pub5.publish(move5)
pub6.publish(move6)
pub7.publish(move7)
pub8.publish(move8)
print("THE END ...!")
