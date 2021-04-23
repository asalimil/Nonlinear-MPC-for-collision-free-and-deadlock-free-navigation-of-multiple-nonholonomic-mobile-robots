#! /usr/bin/env python
import rospy
import numpy as np
import math
import numpy.matlib
from numpy import linalg as LA
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import time
from scipy.integrate import odeint
from scipy.optimize import minimize
from casadi import *

#=== Odometry
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
    global Scan   # , Dmin, Dmin_idx, numRays
    Scan = list(lidar.ranges)
    # Dmin = np.amin(Scan)
    # Dmin_idx = np.argmin(Scan)
    numRays = len(Scan)


#=== Shift horizon
def shift(T, t0, u):
    con = np.transpose(u[0:1,0:])
    t0 = t0 + T
    ushape = np.shape(u)
    u0 = np.concatenate(( u[1:ushape[0],0:],  u[ushape[0]-1:ushape[0],0:]), axis=0)
    return t0, u0


#=== Node initialization
rospy.init_node('move_node')

#=== Parameters initialization
T = 0.05                                                                        # [s]
N = 100                                                                          # prediction horizon

#=== Lower/Upper bounds
robot_radius = 0.2
x_min = -np.Inf; x_max = np.Inf; y_min = -np.Inf; y_max = +np.Inf; theta_min = -np.Inf; theta_max = np.Inf
v_max = +0.22; v_min = -v_max
omega_max = +2.84; omega_min = -omega_max

#=== States
x = SX.sym('x'); y = SX.sym('y'); theta = SX.sym('theta')
states = np.array([[x], [y], [theta]]); n_states = len(states)
#print("states: ", states)

#=== Control inputs
v = SX.sym('v'); omega = SX.sym('omega')
controls = np.array([[v],[omega]]); n_controls = len(controls)
rhs_f = np.array([[v*np.cos(theta)],[v*np.sin(theta)],[omega]])                   # system r.h.s
# print("rhs_f: ", rhs_f, np.shape(rhs_f))

#=== Function: states updates
f = Function('f',[states,controls],[rhs_f])                                       # nonlinear mapping function f(x,u)
# print("f :", np.shape(f))

#=== Symbolic parameters definitions
U = SX.sym('U',n_controls,N);                                                   # Decision variables (controls)
# P = SX.sym('P',n_states + n_states)                                             # parameters (which include the initial state and the reference state)
P = SX.sym('P',n_states + n_states + numRays + numRays)                                   # parameters (which include the initial state and the reference state)
#print("U: ", U)
#print("P: ", P)

X = SX.sym('X',n_states,(N+1));
# A vector that represents the states over the optimization problem.
#print("X: ", X)

#------------------------------
Xr = np.array([[0.0], [0.0], [0.0]])
numRays = 10
D = SX.sym('D',numRays,N+1)
# Independent Variable (acts as P for X)
angRay = SX.sym('angRay',numRays,1)                                                 # Ray includes directions
#------------------------------

#=== objective initial value
obj = 0                                                                         # Objective function
                                                                                # constraints vector
#=== Weights
Q = np.zeros((3,3)); Q[0,0] = 1; Q[1,1] = 5; Q[2,2] = 0.1                           # weighing matrices (states)
# Q = np.zeros((3,3)); Q[0,0] = 10; Q[1,1] = 50; Q[2,2] = 1                           # weighing matrices (states)
R = np.zeros((2,2)); R[0,0] = 0.5; R[1,1] = 0.05                                  # weighing matrices (controls)

#print("Q: ", Q)
#print("R: ", R)

st  = X[:,0]                                                                    # initial state
gX = st-P[0:n_states]                                                                  # initial condition constraints
#print("g: ", g)

#------------------------------------
st_d = D[:,0]
gD = st_d - P[2*n_states:2*n_states+numRays]
#
pObs = SX.sym('pObs',2,numRays)                                                       # contains position of points detected by lidar on obatcles at current measurement
Rz = np.array([[np.cos(X[2,0]), -np.sin(X[2,0])], [np.sin(X[2,0]), np.cos(X[2,0])]])
#
for m in range(numRays):
    e = np.array([[np.cos(angRay[m,0])],[np.sin(angRay[m,0])]])
    pObs[:,m] = mtimes(Rz, mtimes(D[m,0],e)) + np.array([[X[0,0]],[X[1,0]]])

#------------------------------------


for k in range(N):
    st = X[:,k];  con = U[:,k]     # ; d = D[:,k]
    #print("st: ", st); print("con: ", con)
    #print("R: ", R)
    obj = obj  +  mtimes((st-P[3:6]).T,mtimes(Q,(st-P[3:6]))) + mtimes(con.T, mtimes(R, con))               # calculate obj
    #print("Obj: ", obj)
    st_next = X[:,k+1];
    #print("st_next: ", st_next)
    f_value = f(st,con);
    # print("f_value: ", np.shape(f_value))
    st_next_euler = st + (T*f_value)
    #print("st_next_euler: ", st_next_euler)
    gX = vertcat(gX, st_next-st_next_euler)                                               # compute constraints
    #
    #------------------------------------
    for m in range(numRays):
        d_next = D[m,k+1]
        rhs_z = np.array([[X[0,k+1] - pObs[0,m]],[X[1,k+1] - pObs[1,m]]])
        z = Function('z', [X[:,k+1]], [rhs_z])
        z_value = z(st_next)
        d_next_euler = norm_1(z_value)
        gD = vertcat(gD, d_next-d_next_euler)                                               # compute constraints
        # print("gD: ", gD, np.shape(gD))
        # rospy.spin()
    #------------------------------------



# print("shape gX: ", np.shape(gX))
# print("shape gD: ", np.shape(gD))
g = vertcat(gX, gD)
# print("g: ", g)
# rospy.spin()

# make the decision variable one column  vector
# OPT_variables = vertcat(reshape(X, 3*(N+1),1), reshape(U, 2*N, 1))
OPT_variables = vertcat(vertcat(vertcat(reshape(X,3*(N+1),1),reshape(U,2*N,1)),reshape(D,numRays*(N+1),1)))
# print("OPT: ", OPT_variables)
nlp_prob = {'f':obj, 'x':OPT_variables, 'g':g, 'p':P}
# print("NLP: ", nlp_prob)
#
opts = {'print_time': 0, 'ipopt':{'max_iter':2000, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6}}
solver = nlpsol('solver','ipopt', nlp_prob, opts)
# rospy.spin()

# args = {'lbg': np.zeros((1,3*(N+1))), 'ubg': np.zeros((1,3*(N+1))), \
# 'lbx': np.concatenate((np.matlib.repmat(np.array([[-10],[-10],[-np.Inf]]),N+1,1),np.matlib.repmat(np.array([[v_min],[omega_min]]),N,1)), axis=0), \
# 'ubx':np.concatenate((np.matlib.repmat(np.array([[+10],[+10],[+np.Inf]]),N+1,1),np.matlib.repmat(np.array([[v_max],[omega_max]]),N,1)), axis=0)}
# print("args: ", args)
lowerbound_eqConst = np.zeros((1,n_states*(N+1)+numRays*(N+1)))
upperbound_eqConst = np.zeros((1,n_states*(N+1)+numRays*(N+1)))
#
lowerbound_X = np.matlib.repmat(np.array([[x_min],[y_min],[theta_min]]),N+1,1)
lowerbound_U = np.matlib.repmat(np.array([[v_min],[omega_min]]),N,1)
lowerbound_D = np.matlib.repmat(np.ones((numRays,1))*robot_radius,N+1,1)
# lowerbound_Beta = np.ones((numRays,1))*(-1*math.pi)
#
upperbound_X = np.matlib.repmat(np.array([[x_max],[y_max],[theta_max]]),N+1,1)
upperbound_U = np.matlib.repmat(np.array([[v_max],[omega_max]]),N,1)
upperbound_D = np.matlib.repmat(np.ones((numRays,1))*np.Inf,N+1,1)
# upperbound_Beta = np.ones((numRays,1))*(+1*math.pi)
#
args = {'lbg': lowerbound_eqConst, 'ubg': upperbound_eqConst, \
'lbx': np.concatenate((np.concatenate((lowerbound_X,lowerbound_U),axis=0),lowerbound_D),axis=0),\
'ubx': np.concatenate((np.concatenate((upperbound_X,upperbound_U),axis=0),upperbound_D),axis=0)}


#
sim_tim = 1000
t0 = 0;
x0 = np.array([[0.0], [0.0], [0.0]])                                                             # initial condition.
#----------------------------------
d0 = 3.5*np.ones((numRays,1))
#----------------------------------

# print("x0: ", x0)
# xs = np.array([[+1.0], [+1.5], [0.0]])                                                        # Reference posture.

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
goal_idx = 1
ng = 10

# rospy.spin()
#----------------------------------
Beta0 = np.zeros((numRays,1))
for m in range(numRays):
    Beta0[m,0] = m*math.pi/numRays


scan_max = 3.5
Scan = scan_max*np.ones((numRays,1))
D0 = np.transpose(repmat(d0,1,N+1))

# Independent variable
indepVar = vertcat(P, Beta)
#----------------------------------

# Main Loop
while (not rospy.is_shutdown()):
    #=== Odometry & Lidar subscribers
    sub1 = rospy.Subscriber('/odom', Odometry, callback_odom)
    sub2 = rospy.Subscriber('/scan', LaserScan, callback_lidar)
    #===
 # and (mpciter < sim_tim / T)
    # Goal points
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

    # Independent Variables CASADI
    args['p'] = np.concatenate((x0, xs), axis=0)                                # set the values of the parameters vector
    # print("args.p: ", args['p'])

    # initial value of the optimization variables
    # args['x0']  = np.concatenate((reshape(np.transpose(X0),3*(N+1),1), reshape(np.transpose(u0),2*N,1)), axis=0)
    #--------------------------------------------
    X0_reSh = reshape(np.transpose(X0),n_states*(N+1),1)
    u0_reSh = reshape(np.transpose(u0),n_controls*N,1)
    D0_reSh = reshape(np.transpose(D0),numRays*(N+1),1)
    Beta0_reSh = Beta0 # reshape(np.transpose(Beta0),numRays*(N+1),1)
    args['x0']  = \
    np.concatenate((\
    np.concatenate((\
    np.concatenate((X0_reSh,u0_reSh),axis=0),D0_reSh),axis=0),Beta0_reSh),axis=0)
    #--------------------------------------------

    # print("x0: ", args['x0'])
    # print("p: ", args['p'])
    # print("lbx: ", args['lbx'])
    # print("ubx: ", args['ubx'])
    # print("lbg: ", args['lbg'])
    # print("ubg: ", args['ubg'])
    # print("args: ", np.shape(args['x0']))
    # rospy.spin()

    sol = solver(x0=args['x0'], p=args['p'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'])

    # print("sol: ", sol['x'])
    # print("sol: ", np.shape(sol['x']))
    # rospy.spin()

    solu = sol['x'][n_states*(N+1):n_states*(N+1)+n_controls*N]        # extract optimal u*
    #-----------------------------------------------------------------
    # solu = sol['x'][3*(N+1)+n_controls*N+numRays*(N+1)+numRays:]
    #-----------------------------------------------------------------
    solu_full = np.transpose(solu.full())
    # print("solu: ", sol)
    # rospy.spin()

    u = np.transpose(reshape(solu_full, n_controls,N))                            # get controls only from the solution
    print("u: ", u)

    solx = sol['x'][0:3*(N+1)]
    solx_full = np.transpose(solx.full())
    xx1[0:,0:3,mpciter] = np.transpose(reshape(solx_full, 3,N+1))                               # get solution TRAJECTORY
    # print("xx1: ", xx1[:,0:3,mpciter])

    u_cl[mpciter,0:] = u[0:1,0:]
    # print("u_cl: ", u_cl[mpciter,0:])

    t[mpciter] = t0

    # Apply the control and shift the solution
    t0, u0 = shift(T, t0, u)
    # x0 = Xr
    #-----------------
    X0 = np.concatenate((X0[1:,0:n_states+1], X0[N-1:N,0:n_states+1]), axis=0)
    #-----------------
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
    print(time.clock(), u_cl[mpciter,0], u_cl[mpciter,1], Xr[0], Xr[1], Xr[2])
    move.linear.x = u_cl[mpciter,0]                                                     # apply first optimal linear velocity
    move.angular.z = u_cl[mpciter,1]                                                    # apply first optimal angular velocity
    # pub.publish(move)
    time.sleep(T)
    print(time.clock(), u_cl[mpciter,0], u_cl[mpciter,1], Xr[0], Xr[1], Xr[2])
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
    #--------------------
    D0 = np.transpose(repmat(Scan,1,N+1))
    #--------------------


# Stop Robot
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
move = Twist()
move.linear.x = 0.0                                                     # apply first optimal linear velocity
move.angular.z = 0.0                                                    # apply first optimal angular velocity
pub.publish(move)
print("THE END ...!")
