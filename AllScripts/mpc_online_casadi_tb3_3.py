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

def callback_odom(odom):
    global Xr
    xr = odom.pose.pose.position.x
    yr = odom.pose.pose.position.y
    qz = odom.pose.pose.orientation.z
    qw = odom.pose.pose.orientation.w
    th = 2*np.arcsin(qz)
    thr = modify(th)
    Xr = np.array([[xr], [yr], [thr]])
    # Xr = np.array([[yr], [xr], [2*np.arcsin(qz)]])

def modify(th):
    if th >= -math.pi and th < 0:
        th_modified = th + 2*math.pi
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


# node initialization
rospy.init_node('move_node3')

# subscriber odometry
sub1 = rospy.Subscriber('/tb3_3/odom', Odometry, callback_odom)


#
T = 0.01                                                                        # [s]
N = 200                                                                          # prediction horizon
# rob_diam = 0.1

v_max = +0.22; v_min = -v_max
omega_max = +2.84; omega_min = -omega_max

x = SX.sym('x'); y = SX.sym('y'); theta = SX.sym('theta')
states = np.array([[x], [y], [theta]]); n_states = len(states)
#print("states: ", states)

v = SX.sym('v'); omega = SX.sym('omega')
controls = np.array([[v],[omega]]); n_controls = len(controls)
rhs = np.array([[v*np.cos(theta)],[v*np.sin(theta)],[omega]])                   # system r.h.s
#print("rhs: ", rhs)

f = Function('f',[states,controls],[rhs])                                       # nonlinear mapping function f(x,u)
#print("Function :", f)

U = SX.sym('U',n_controls,N);                                                   # Decision variables (controls)
P = SX.sym('P',n_states + n_states)                                             # parameters (which include the initial state and the reference state)
#print("U: ", U)
#print("P: ", P)

X = SX.sym('X',n_states,(N+1));
# A vector that represents the states over the optimization problem.
#print("X: ", X)


obj = 0                                                                        # Objective function
                                                                        # constraints vector
Q = np.zeros((3,3)); Q[0,0] = 1; Q[1,1] = 5; Q[2,2] = 0.1                           # weighing matrices (states)
R = np.zeros((2,2)); R[0,0] = 0.5; R[1,1] = 0.05                                  # weighing matrices (controls)

#print("Q: ", Q)
#print("R: ", R)


st  = X[:,0]                                                                    # initial state
#print("st: ", st)

g = st-P[0:3]                                                                   # initial condition constraints
#print("g: ", g)


for k in range(N):
    st = X[:,k];  con = U[:,k]
    #print("st: ", st); print("con: ", con)
    #print("R: ", R)

    obj = obj  +  mtimes((st-P[3:6]).T,mtimes(Q,(st-P[3:6]))) + mtimes(con.T, mtimes(R, con))               # calculate obj
    #print("Obj: ", obj)
    st_next = X[:,k+1];
    #print("st_next: ", st_next)
    f_value = f(st,con);
    #print("f_value: ", f_value)
    st_next_euler = st + (T*f_value)
    #print("st_next_euler: ", st_next_euler)
    g = vertcat(g, st_next-st_next_euler)                                               # compute constraints

# print("g: ", g)

# make the decision variable one column  vector
OPT_variables = vertcat(reshape(X, 3*(N+1),1), reshape(U, 2*N, 1))
# print("OPT: ", OPT_variables)
nlp_prob = {'f':obj, 'x':OPT_variables, 'g':g, 'p':P}
# print("NLP: ", nlp_prob)

opts = {'print_time': 0, 'ipopt':{'max_iter':2000, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6}}
solver = nlpsol('solver','ipopt', nlp_prob, opts)



args = {'lbg': np.zeros((1,3*(N+1))), 'ubg': np.zeros((1,3*(N+1))), \
'lbx': np.concatenate((np.matlib.repmat(np.array([[-10],[-10],[-np.Inf]]),N+1,1),np.matlib.repmat(np.array([[v_min],[omega_min]]),N,1)), axis=0), \
'ubx':np.concatenate((np.matlib.repmat(np.array([[+10],[+10],[+np.Inf]]),N+1,1),np.matlib.repmat(np.array([[v_max],[omega_max]]),N,1)), axis=0)}
# print("args: ", args)


sim_tim = 1000
t0 = 0;
x0 = np.array([[0.0], [0.0], [0.0]])                                                             # initial condition.
x0 = Xr

# print("x0: ", x0)
xs = np.array([[0.0], [-3.0], [5.497]])                                                        # Reference posture.

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
    args['x0']  = np.concatenate((reshape(np.transpose(X0),3*(N+1),1), reshape(np.transpose(u0),2*N,1)), axis=0)
    # print("args: ", args['x0'])
    # print("args: ", args)

    sol = solver(x0=args['x0'], p=args['p'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'])
    # print("sol: ", sol['x'])

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
    pub = rospy.Publisher('/tb3_3/cmd_vel', Twist, queue_size=10)
    move = Twist()
    print("v, w: ", u_cl[mpciter,0], u_cl[mpciter,1])
    move.linear.x = u_cl[mpciter,0]                                                     # apply first optimal linear velocity
    move.angular.z = u_cl[mpciter,1]                                                    # apply first optimal angular velocity
    pub.publish(move)
    time.sleep(T)

    # print("mpciter, error: ", mpciter, LA.norm(x0-xs))
    mpciter = mpciter + 1

# Stop Robot
move.linear.x = 0.0                                                     # apply first optimal linear velocity
move.angular.z = 0.0                                                    # apply first optimal angular velocity
pub.publish(move)
print("THE END ...!")

'''
# main_loop_time = toc(main_loop);
ss_error = LA.norm(x0-xs)
print("Steady State Error: ", ss_error)
print("Closed-Loop Control: ", u_cl)
# average_mpc_time = main_loop_time/(mpciter+1)
# Draw_MPC_point_stabilization_v1 (t,xx,xx1,u_cl,xs,N,rob_diam)


pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
move = Twist()


# tt = 0
# tc0 = time.clock()
# tc = 0.0
i = 0
# print("time: ", tc0)

while (i < np.shape(u_cl)[0]) and (not rospy.is_shutdown()):
    print("v, w: ", u_cl[i,0], u_cl[i,1])
    move.linear.x = u_cl[i,0]                                                     # apply first optimal linear velocity
    move.angular.z = u_cl[i,1]                                                    # apply first optimal angular velocity
    # t_new = tt + T                                                    # current time of program
    #
    # print("time: ", tc)
    # while (tc < t_new):
    #    pub.publish(move)
    #    tc = time.clock() - tc0
    #
    pub.publish(move)
    time.sleep(T)
    i = i + 1
    # tt = t_new


# Stop Robot
move.linear.x = 0.0                                                     # apply first optimal linear velocity
move.angular.z = 0.0                                                    # apply first optimal angular velocity
pub.publish(move)
print("THE END ...!")
'''
