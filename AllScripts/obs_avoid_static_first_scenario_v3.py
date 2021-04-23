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
    thr = th
    Xr = np.array([[xr], [yr], [thr]])


"""
#=== LIDAR
def callback_lidar(lidar):
    global Scan   # , Dmin, Dmin_idx, numRays
    scan_max = 3.5
    Scan = list(lidar.ranges)
    for i in range(len(Scan)):
        if Scan[i] == np.Inf:
           Scan[i] = 3.5

    # Dmin = np.amin(Scan)
    # Dmin_idx = np.argmin(Scan)
    numRays = len(Scan)
"""

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
T = 0.075                                                                        # [s]
N = 125                                                                          # prediction horizon
Xr = np.array([[0.0], [0.0], [0.0]])
numRays = 10
scan_max = 3.5
Scan = scan_max*np.ones((numRays,1))
goal_idx = 1
ng = 1

#=== Lower/Upper bounds
robot_radius = 0.2
x_min = -10; x_max = 10; y_min = -10; y_max = +10; theta_min = -np.Inf; theta_max = np.Inf
d_min = robot_radius; d_max = np.Inf
#
v_max = +0.15; v_min = -v_max
omega_max = +2.0; omega_min = -omega_max

#===
x = SX.sym('x'); y = SX.sym('y'); theta = SX.sym('theta')
statesX = np.array([[x], [y], [theta]])
n_statesX = len(statesX)
states = statesX
d = SX.sym('d',numRays,1)
for m in range(numRays):
    states = np.concatenate( (states, np.array([[d[m]]])), axis=0)


n_states = len(states)

#=== Control inputs
v = SX.sym('v'); omega = SX.sym('omega')
controls = np.array([[v],[omega]]); n_controls = len(controls)

#=== Function: states updates
rhs_f = np.array([[v*np.cos(theta)],[v*np.sin(theta)],[omega]])                   # system r.h.s
f = Function('f',[statesX,controls],[rhs_f])                                       # nonlinear mapping function f(x,u)

#=== Symbolic parameters definitions
U = SX.sym('U',n_controls, N)                                                  # Decision variables (controls)
# P >> x0, xs, D0, angRays
P = SX.sym('P',n_statesX + n_statesX + numRays + numRays)                                # parameters (which include the initial state and the reference state)
X = SX.sym('X',n_states, (N+1))
B = SX.sym('B',numRays,1)                                                 # Ray includes directions

#=== objective initial value
obj = 0                                                                         # Objective function

#=== Constraints                                                                                # constraints vector
st_x = X[0:n_statesX,0]                                                                    # initial state
st_d = X[n_statesX:,0]
gx = st_x - P[0:n_statesX]                                                                  # initial condition constraints
gd = st_d - P[2*n_statesX:2*n_statesX+numRays]                                                                  # initial condition constraints
B = P[2*n_statesX+numRays:2*n_statesX+numRays+numRays]

pObs = SX.sym('pObs',2,numRays)                                                       # contains position of points detected by lidar on obatcles at current measurement
Rz = np.array([[np.cos(X[2,0]), -np.sin(X[2,0])], [np.sin(X[2,0]), np.cos(X[2,0])]])
for m in range(numRays):
    e = np.array([[np.cos(B[m,0])],[np.sin(B[m,0])]])
    pObs[:,m] = mtimes(Rz, mtimes(X[n_statesX+m,0],e)) + np.array([[X[0,0]],[X[1,0]]])


Q = np.zeros((3,3)); Q[0,0] = 1; Q[1,1] = 5; Q[2,2] = 0.1                       # weighing matrices (states)
R = np.zeros((2,2)); R[0,0] = 0.5; R[1,1] = 0.05                                # weighing matrices (controls)

for k in range(N):
    st_x = X[0:n_statesX,k]; con = U[:,k]
    # st_d = X[n_statesX:,k]
    obj = obj  +  mtimes((st_x-P[3:6]).T,mtimes(Q,(st_x-P[3:6]))) + mtimes(con.T, mtimes(R, con))               # calculate obj
    st_next = X[0:n_statesX,k+1];
    f_value = f(st_x,con);
    st_next_euler = st_x + (T*f_value)
    gx = vertcat(gx, st_next-st_next_euler)                                               # compute constraints
    #
    for m in range(numRays):
        d_next = X[n_statesX+m,k+1]
        rhs_z = np.array([[X[0,k+1] - pObs[0,m]],[X[1,k+1] - pObs[1,m]]])
        z = Function('z', [X[0:n_statesX,k+1]], [rhs_z])
        z_value = z(st_next)
        d_next_euler = norm_1(z_value)
        gd = vertcat(gd, d_next-d_next_euler)                                               # compute constraints


g = vertcat(gx, gd)
# print("g: ", g)
# rospy.spin()


# Make the decision variable one column  vector
OPT_variables = vertcat(vertcat(vertcat(reshape(X,n_states*(N+1),1),reshape(U,n_controls*N,1))))
nlp_prob = {'f':obj, 'x':OPT_variables, 'g':g, 'p':P}
opts = {'print_time': 0, 'ipopt':{'max_iter':2000, 'print_level':0, 'acceptable_tol':1e-8, 'acceptable_obj_change_tol':1e-6}}
solver = nlpsol('solver','ipopt', nlp_prob, opts)

lowerbound_eqConst = np.zeros((1,n_states*(N+1)))
upperbound_eqConst = np.zeros((1,n_states*(N+1)))
#
lowerbound_X = np.concatenate((\
np.matlib.repmat(np.array([[x_min],[y_min],[theta_min]]),N+1,1), \
np.matlib.repmat(np.ones((numRays,1))*d_min,N+1,1)), axis=0)
#
lowerbound_U = np.matlib.repmat(np.array([[v_min],[omega_min]]),N,1)
#
upperbound_X = np.concatenate((\
np.matlib.repmat(np.array([[x_max],[y_max],[theta_max]]),N+1,1), \
np.matlib.repmat(np.ones((numRays,1))*d_max,N+1,1)), axis=0)
#
upperbound_U = np.matlib.repmat(np.array([[v_max],[omega_max]]),N,1)

args = {'lbg': lowerbound_eqConst, 'ubg': upperbound_eqConst, \
'lbx': np.concatenate((lowerbound_X,lowerbound_U),axis=0),\
'ubx': np.concatenate((upperbound_X,upperbound_U),axis=0)}

#=== MPC horizon parameters and initializations
sim_tim = 1000
t0 = 0;
x0 = np.concatenate( \
(np.array([[0.0], [0.0], [0.0]]), scan_max*np.ones((numRays,1))), axis=0)       # initial condition.

xx = np.zeros((n_states, int(sim_tim/T)))
xx[:,0:1] = x0                                                                  # xx contains the history of states


t = np.zeros(int(sim_tim/T))                                                    # Maximum simulation time
t[0] = t0

u0 = np.zeros((n_controls,N));                                                  # two control inputs for each robot
X0 = np.transpose(repmat(x0,1,N+1))                                             # initialization of the states decision variables

#=== Start MPC
mpciter = 0
xx1 = np.zeros((N+1,n_states,int(sim_tim/T)))
u_cl = np.zeros((int(sim_tim/T),n_controls))

# B0 is angular increments of the lidar's rays (define: 2p = 2*math.pi) e.g., [0*2p/m, 1*2p/m, ..., (m-1)*2p/m]
B0 = np.zeros((numRays,1))
for m in range(numRays):
    B0[m,0] = m*(2*math.pi)/numRays


#=== Main Loop
# The main simulaton loop... it works as long as the error is greater than 10^-6 and the number of mpc steps is less than its maximum value.
while (not rospy.is_shutdown() and goal_idx <= ng):

    #=== Odometry & Lidar subscribers
    sub1 = rospy.Subscriber('/odom', Odometry, callback_odom)
    # sub2 = rospy.Subscriber('/scan', LaserScan, callback_lidar)

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

    #=== Independent Variables CASADI
    # set the values of the parameters vector
    args['p'] = np.concatenate((\
    np.concatenate((\
    np.concatenate((\
    x0[0:n_statesX], xs), axis=0), \
    x0[n_statesX:n_statesX+numRays]),axis=0), \
    B0), axis=0)

    #=== Initial value of the optimization variables
    X0_reSh = reshape(np.transpose(X0),n_states*(N+1),1)
    u0_reSh = reshape(np.transpose(u0),n_controls*N,1)
    args['x0'] = np.concatenate((X0_reSh,u0_reSh),axis=0)

    #=== Solve NMPC problem
    sol = solver(x0=args['x0'], p=args['p'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'])

    solu = sol['x'][n_states*(N+1):n_states*(N+1)+n_controls*N]        # extract optimal u*
    solu_full = np.transpose(solu.full())
    u = np.transpose(reshape(solu_full, n_controls,N))                            # get controls only from the solution

    solx = sol['x'][0:n_states*(N+1)]
    solx_full = np.transpose(solx.full())
    xx1[0:,0:n_states,mpciter] = np.transpose(reshape(solx_full, n_states,N+1))                               # get solution TRAJECTORY

    u_cl[mpciter,0:] = u[0:1,0:]                                                # Take only the first optimal control solution
    t[mpciter] = t0

    #=== Apply the control and shift the solution
    t0, u0 = shift(T, t0, u)
    # x0[n_statesX:n_statesX+numRays] = Scan # np.reshape(Scan, (-1,1))
    # x0[0:n_statesX] = Xr
    #****************************** CHECK THIS LINE *************************************
    X0 = np.concatenate((X0[1:,0:n_states+1], X0[N-1:N,0:n_states+1]), axis=0)
    xx[0:,mpciter+1:mpciter+2] = x0

    solX0 = sol['x'][0:n_states*(N+1)]
    solX0_full = np.transpose(solX0.full())
    X0 = np.transpose(reshape(solX0_full, n_states,N+1))                                # get solution TRAJECTORY

    #=== Shift trajectory to initialize the next step
    X0 = np.concatenate((X0[1:,0:n_states+1], X0[N-1:N,0:n_states+1]), axis=0)

    #=== Move Robot
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    move = Twist()
    # print(time.clock(), u_cl[mpciter,0], u_cl[mpciter,1], Xr[0], Xr[1], Xr[2])
    move.linear.x = u_cl[mpciter,0]                                                     # apply first optimal linear velocity
    move.angular.z = u_cl[mpciter,1]                                                    # apply first optimal angular velocity
    pub.publish(move)
    # time.sleep(T)
    print(time.clock(), u_cl[mpciter,0], u_cl[mpciter,1], Xr[0], Xr[1], Xr[2])
    ne = LA.norm(Xr-xs)

    #=== Stop Condition
    if ne < 0.1:
        goal_idx = goal_idx + 1
        if goal_idx > ng:
            pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
            move = Twist()
            move.linear.x = 0; move.angular.z = 0
            pub.publish(move)
            print("Robot has arrived to GOAL point!")


    mpciter = mpciter + 1
    x0[n_statesX:n_statesX+numRays] = Scan # np.reshape(Scan, (-1,1))
    x0[0:n_statesX] = Xr
    # print("Scan: ", Scan)


#=== Stop Robot
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
move = Twist()
move.linear.x = 0.0                                                     # apply first optimal linear velocity
move.angular.z = 0.0                                                    # apply first optimal angular velocity
pub.publish(move)
print("THE END ...!")
