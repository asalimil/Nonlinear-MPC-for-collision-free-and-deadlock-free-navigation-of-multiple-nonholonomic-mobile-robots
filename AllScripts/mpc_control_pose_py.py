#! /usr/bin/env python
import rospy
import numpy as np
import math
from numpy import linalg as LA
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import LaserScan
# import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize

def callback_odom(odom):
    global Xr
    xr = odom.pose.pose.position.x
    yr = odom.pose.pose.position.y
    qz = odom.pose.pose.orientation.z
    qw = odom.pose.pose.orientation.w
    Xr = np.array([xr, yr, 2*np.arcsin(qz)])


def objective(OptVars):
    # print("u: ", u)
    global ns, nc, N, Xref, Xr, Vref
    # Objective Function
    # J = sum(loss) from k=0 to k=N-1
    L = np.zeros(N) # returen array of zeros
    Xu = OptVars[0:ns*N]; U = OptVars[ns*N:ns*N+nc*N]
    X0 = Xr # pose in global frame
    # Xk_1 = X0 + f(X(0),u(0))*Dt + f(X(1),u(1))*Dt + ... + f(X(k),u(k))*Dt
    F = np.zeros(3)
    Dt = 0.5
    for k in range(N):
        vk = U[k*nc]; wk = U[k*nc+nc-1]
        uk = np.array([vk, wk])
        Xk = Xu[k*ns:k*ns+ns]
        f = rhs(Xk,uk) # f(x(k),u(k))
        F = F + f
        Xk_1 = X0 + F*Dt
        uref = np.array([np.power(Vref[0],2) + np.power(Vref[1],2), Vref[2]])
        L[k] = loss(Xk_1,Xref,uk,uref)
    #
    J = np.sum(L)
    # print("J: ", J)
    return J

def loss(Xk_1,Xref,uk,uref):
    # quadratic obj. func.
    Q = np.array([[1.0, 0.0, 0.0],[0.0, 5.0, 0.0],[0.0, 0.0, 0.1]])
    R = np.array([[0.5, 0.0],[0.0, 0.05]])
    # print("e: ", Xk-Xref)
    loss_value = np.matmul(Xk_1-Xref,Q.dot(np.transpose(Xk_1-Xref))) + np.matmul(uk-uref,R.dot(np.transpose(uk-uref)))
    return loss_value

def equality_constraint(OptVars):
    global ns, nc, N, Xr
    Dt = 0.5
    # equality const
    # u_resh = np.reshape(u, (n,N))
    # Xk = Xr # pose in global frame
    Xu = OptVars[0:ns*N]; U = OptVars[ns*N:ns*N+nc*N]
    X0 = Xr; F = np.zeros(3)
    eq_cons = []
    for k in range(N):
        vk = U[k*nc]; wk = U[k*nc+nc-1]
        uk = np.array([vk, wk])
        #
        Xk = Xu[k*ns:k*ns+ns]
        f = rhs(Xk,uk) # f(x(k),u(k))
        F = F + f
        # Xk_1 = X0 + f(X(0),u(0))*Dt + f(X(1),u(1))*Dt + ... + f(X(k),u(k))*Dt
        Xk_1 = X0 + F*Dt
        con = Xk_1 - Xk - f*Dt
        eq_cons = np.append(eq_cons, con) # fxu := f(x(k),u(k))
        # Xk = Xk_1
    # print("eq_const: ",eq_cons)
    return eq_cons

# Right-hand side
def rhs(Xk,uk):
    f = np.array([uk[0]*np.cos(Xk[2]), uk[0]*np.sin(Xk[2]), uk[1]])
    return f

# node initialization
rospy.init_node('move_node')

# subscriber odometry
sub1 = rospy.Subscriber('/odom', Odometry, callback_odom)


# goal points
Xr = np.array([0.0, 0.0, 0.0]) # pose of robot in local coordinate frame
Xref = np.array([+1.0, +1.5, 0.0]) # reference pose
Vref = np.array([0.0, 0.0, 0.0]) #  reference velocity

ne = np.Inf
nc = 2 # number of control inputs
ns = 3 # number of state variables
N = 3 # number of horizon steps >> N = 3 works well for Dt = 0.5
Nc = 1 # number of control horizon
# u0 = np.zeros(n*N) # return array of zeros
OptVars0 = np.zeros(ns*N+nc*N) # all optimization variables X(0), X(1), ... ,X(N-1), U(0), U(1), ..., U(N-1)
# print('OptVars =' + str(OptVars0))
# show initial objective
# print('Initial SSE Objective: ' + str(objective(OptVars0)))

# Min/Max of Pose and Velocity
xmin = -3.0; ymin = -3.0; thmin = -3.14
xmax = +3.0; ymax = +3.0; thmax = +3.14
# v \in (-0.22, +0.22) w \in (-2.84, +2.84)
vmin = -0.15; wmin = -1.0
vmax = +0.15; wmax = +1.0

# Bounds on Pose and Velocity
X_bnds = ((xmin,xmax),(ymin,ymax),(thmin,thmax))*N
V_bnds = ((vmin,vmax),(wmin,wmax))*N
bnds = X_bnds + V_bnds

pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
move = Twist()
# Main Loop
while not rospy.is_shutdown():

    # ****** MPC controller ******
    cons = {'type': 'eq', 'fun': equality_constraint}
    # solution = minimize(objective,OptVars0,method='SLSQP',bounds=bnds,constraints=cons)
    # solution = minimize(objective,OptVars0,method='SLSQP',bounds=bnds,constraints=cons)
    solution = minimize(objective,OptVars0,method='SLSQP',bounds=bnds,constraints=cons,options={'maxiter': 10})
    opt = solution.x  # x is defined inside the optimizer
    mode = solution.status # >> 0 means successful!
    OptVars0 = opt

    # Optimal Control Variables (Nc = 1)
    lin_vel = opt[ns*N]
    ang_vel = opt[ns*N+nc-1]
    #
    move.linear.x = lin_vel
    move.angular.z = ang_vel
    # print("v:",lin_vel, "w: ",ang_vel, "mode 0?:", mode)

    # Optimal Control Variables (Nc := N)
    # print("v: ", lin_vel, "w: ", ang_vel)
    pub.publish(move)
    # rospy.sleep(0.1)

    '''
    # Move Robot
    for n in range(Nc):
        # Control Command
        lin_vel = opt[ns*N+n*nc]
        ang_vel = opt[ns*N+1+n*nc]
        move.linear.x = lin_vel
        move.angular.z = ang_vel
        # print("v: ", lin_vel, "w: ", ang_vel)
        pub.publish(move)
        rospy.sleep(0.5)
    '''

    # rate = rospy.Rate(2)
    ne = LA.norm(Xr-Xref)
    print("norm E: ", ne)

    # Stop Condition
    if ne < 0.075:
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        move = Twist()
        move.linear.x = 0; move.angular.z = 0
        pub.publish(move)
        print("Robot has arrived to GOAL point!")
        rospy.spin()

    rospy.Rate(2)
