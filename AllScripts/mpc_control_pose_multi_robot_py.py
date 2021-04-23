#! /usr/bin/env python
import rospy
import numpy as np
import math
from numpy import linalg as LA
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import LaserScan
import time
from scipy.integrate import odeint
from scipy.optimize import minimize

# Odometry
def callback_odom(odom):
    global Xr
    xr = odom.pose.pose.position.x
    yr = odom.pose.pose.position.y
    qz = odom.pose.pose.orientation.z
    qw = odom.pose.pose.orientation.w
    Xr = np.array([xr, yr, 2*np.arcsin(qz)])

# Objective Function
def objective(OptVars):
    global ns, nc, N, Xref, Xr, Vref, Nc, Dt
    L = np.zeros(N)                                                             # array of losses
    #
    Xp = OptVars[0:ns*N];                                                       # predicted pose of robot
    Xp[0:ns] = Xr                                                               # inialize predicted Xp with Xr
    U = OptVars[ns*N:ns*N+nc*N]                                                 # optimal control laws
    #---
    for k in range(N):
        if k < Nc:
            vk = U[k*nc]; wk = U[k*nc+nc-1]
            uk = np.array([vk, wk])
        uref = np.array([np.power(Vref[0],2) + np.power(Vref[1],2), Vref[2]])
        L[k] = loss(Xp[k*ns:(k+1)*ns],Xref,uk,uref)                             # calculation of loss value for each future step
    #---
    J = np.sum(L)
    return J

# Loss function
def loss(Xp,Xref,uk,uref):
    # quadratic obj. func.
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, 0.1]])
    R = np.array([[0.5, 0.0],[0.0, 0.05]])
    # print("e: ", Xk-Xref)
    loss_value = np.matmul(Xp-Xref,Q.dot(np.transpose(Xp-Xref))) + np.matmul(uk-uref,R.dot(np.transpose(uk-uref)))
    return loss_value

# Equality constraints
def equality_constraint(OptVars):
    global ns, nc, N, Xr, Nc, Dt
    #
    Xp = OptVars[0:ns*N];                                                       # predicted pose of robot
    Xp[0:ns] = Xr                                                               # inialize predicted Xp with Xr
    Xk = Xr                                                                     # initialize actual pose of robot by current pose of robot
    #
    U = OptVars[ns*N:ns*N+nc*N]                                                 # optimal control laws
    #
    eq_cons = []
    #---
    for k in range(N-1):
        if k < Nc:
            vk = U[k*nc]; wk = U[k*nc+nc-1]
            uk = np.array([vk, wk])
        # actual new pose of robot
        f = rhs(Xk,uk)                                                           # right-hand side calculation
        Xnew = Xk + f*Dt                                                        # X(k_1) = X0 + f(X(0),u(0))*Dt + f(X(1),u(1))*Dt + ... + f(X(k),u(k))*Dt
        #
        con = Xnew - Xp[(k+1)*ns:(k+2)*ns]                                      # constraints of actual new pose and predicted new pose
        eq_cons = np.append(eq_cons, con)                                       # fxu := f(x(k),u(k))
        Xk = Xnew
    #---
    return eq_cons

# Right-hand side
def rhs(Xk,uk):
    f = np.array([uk[0]*np.cos(Xk[2]), uk[0]*np.sin(Xk[2]), uk[1]])
    return f

# node initialization
rospy.init_node('move_node')

# subscriber odometry
sub1 = rospy.Subscriber('/odom', Odometry, callback_odom)


# Initializations
ne = np.Inf
nc = 2                                                                          # number of control inputs
ns = 3                                                                          # number of state variables
N = 5                                                                           # number of horizon steps >> N = 3 works well for Dt = 0.5
Nc = 2                                                                          # number of control horizon
Dt = 0.5
# Ng = 2 # number of goal points
# g = 1 # goal points' index
# ng = 2
Xr = np.array([0.0, 0.0, 0.0]) # pose of robot
OptVars0 = np.zeros(ns*N+nc*Nc) # all optimization variables X(0), X(1), ... ,X(N-1), U(0), U(1), ..., U(N-1)
# Pose control >> Velocities of target is constant
Vref = np.array([0.0, 0.0, 0.0]) #  reference velocity
Xref = np.array([1.0, -2.0, 1.57]) # reference pose

# print('OptVars =' + str(OptVars0))
# show initial objective
# print('Initial SSE Objective: ' + str(objective(OptVars0)))

# Min/Max of Pose and Velocity
xmin = -3.0; ymin = -3.0; thmin = -3.14
xmax = +3.0; ymax = +3.0; thmax = +3.14
# v \in (-0.22, +0.22) w \in (-2.84, +2.84)
vmin = -0.1; wmin = -0.5
vmax = +0.1; wmax = +0.5

# Bounds on Pose and Velocity
X_bnds = ((xmin,xmax),(ymin,ymax),(thmin,thmax))*N
V_bnds = ((vmin,vmax),(wmin,wmax))*Nc
bnds = X_bnds + V_bnds


pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
move = Twist()
# Main Loop
while not rospy.is_shutdown():

    '''
    # Goal points
    if g == 1:
        Xref = np.array([1.0, -2.0, 0.0]) # reference pose
    elif g == 2:
        Xref = np.array([0.0, 0.0, 0.0]) # reference pose
    '''

    # ****** MPC controller ******
    # equality constraints definition
    cons = {'type': 'eq', 'fun': equality_constraint}

    # minimization step
    # solution = minimize(objective,OptVars0,method='SLSQP',bounds=bnds,constraints=cons)
    solution = minimize(objective,OptVars0,method='SLSQP',bounds=bnds,constraints=cons)
    # solution = minimize(objective,OptVars0,method='SLSQP',bounds=bnds,constraints=cons,options={'maxiter': 10})
    opt = solution.x  # x is defined inside the optimizer
    mode = solution.status # >> 0 means successful!
    OptVars0 = opt

    # optimal variables extraction
    lin_vel = opt[ns*N]
    ang_vel = opt[ns*N+nc-1]

    # apply first optimal (v0, w0)s
    move.linear.x = lin_vel
    move.angular.z = ang_vel
    # print("v:",lin_vel, "w: ",ang_vel, "mode 0?:", mode)
    # Optimal Control Variables (Nc := N)
    # print("v: ", lin_vel, "w: ", ang_vel)
    pub.publish(move)
    # rospy.sleep(0.1)

    # error calculation
    # rate = rospy.Rate(2)
    ne = LA.norm(Xr-Xref)
    print("norm E: ", ne)

    # stop condition
    if ne < 0.075:
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        move = Twist()
        move.linear.x = 0; move.angular.z = 0
        pub.publish(move)
        print("Robot has arrived to GOAL point!")
        rospy.spin()

    rospy.Rate(2)

