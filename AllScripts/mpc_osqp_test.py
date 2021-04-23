#! /usr/bin/env python
import rospy
import osqp
import numpy as np
import scipy as sp
from scipy import sparse
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
    # Xr = np.array([xr, yr, 2*np.arcsin(qz)])
    Xr = np.array([xr, yr, 2*np.arcsin(qz)])

def gamma(w,Ts):
    if w == 0:
        out = Ts/2
    else:
        out = math.sin((Ts/2)*w)/w
    return out

# node initialization
rospy.init_node('move_node')

# subscriber odometry
sub1 = rospy.Subscriber('/odom', Odometry, callback_odom)


# Constraints
u0 = np.array([0.0, 0.0])
umin = np.array([-0.22, -1.0]) - u0
umax = np.array([+0.22, +1.0]) - u0
xmin = np.array([-np.Inf,-np.Inf,-2*math.pi])
xmax = np.array([+np.Inf,+np.Inf,+2*math.pi])

# Initial and reference states
Xr = np.array([0.0, 0.0, 0.0])
xr = np.array([1.0, 1.0, 0*math.pi/2])

# Prediction horizon
N = 100

# Objective function
Q = sparse.diags([1.0, 5.0, 0.1])
QN = Q
R = sparse.diags([0.5, 0.05])

nx = 3; nu = 2

# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
               np.zeros(N*nu)])


# Simulate in closed loop
# nsim = 15
ne = np.Inf
ctrl = u0
Ts = 0.01
# for i in range(nsim):


while any((not rospy.is_shutdown(), not ne < 0.05)):

    # Discrete time model of a quadcopter
    Ad = sparse.csc_matrix([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
    ])


    w = gamma(ctrl[1],Ts)
    Bd = sparse.csc_matrix([
    [2*w*math.cos(Xr[2]), Ts/2],
    [2*w*math.sin(Xr[2]), Ts/2],
    [0.0,                 Ts]
    ])

    '''
    Bd = sparse.csc_matrix([
    [math.cos(Xr[2]), 0.0],
    [math.sin(Xr[2]), 0.0],
    [0.0,             1.0]
    ])
    '''
    [nx, nu] = Bd.shape

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - linear dynamics
    Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
    Aeq = sparse.hstack([Ax, Bu])
    leq = np.hstack([-Xr, np.zeros(N*nx)])
    ueq = leq
    # - input and state constraints
    Aineq = sparse.eye((N+1)*nx + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq], format='csc')
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u, warm_start=True)

    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    # x0 = Ad.dot(x0) + Bd.dot(ctrl)*Ts

    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    move = Twist()

    # apply first optimal (v0, w0)s
    move.linear.x = ctrl[0]
    move.angular.z = ctrl[1]
    pub.publish(move)

    rate = rospy.Rate(10)
    rate.sleep()

    # Update initial state
    l[:nx] = -Xr
    u[:nx] = -Xr
    prob.update(l=l, u=u)
    print("control :", ctrl)


pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
move = Twist()
move.linear.x = 0; move.angular.z = 0
pub.publish(move)
print("Robot has arrived to GOAL point!")
rospy.spin()

'''
Ad = sparse.csc_matrix([
  [1.,      0.,     0., 0., 0., 0., 0.1,     0.,     0.,  0.,     0.,     0.    ],
  [0.,      1.,     0., 0., 0., 0., 0.,      0.1,    0.,  0.,     0.,     0.    ],
  [0.,      0.,     1., 0., 0., 0., 0.,      0.,     0.1, 0.,     0.,     0.    ],
  [0.0488,  0.,     0., 1., 0., 0., 0.0016,  0.,     0.,  0.0992, 0.,     0.    ],
  [0.,     -0.0488, 0., 0., 1., 0., 0.,     -0.0016, 0.,  0.,     0.0992, 0.    ],
  [0.,      0.,     0., 0., 0., 1., 0.,      0.,     0.,  0.,     0.,     0.0992],
  [0.,      0.,     0., 0., 0., 0., 1.,      0.,     0.,  0.,     0.,     0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      1.,     0.,  0.,     0.,     0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      0.,     1.,  0.,     0.,     0.    ],
  [0.9734,  0.,     0., 0., 0., 0., 0.0488,  0.,     0.,  0.9846, 0.,     0.    ],
  [0.,     -0.9734, 0., 0., 0., 0., 0.,     -0.0488, 0.,  0.,     0.9846, 0.    ],
  [0.,      0.,     0., 0., 0., 0., 0.,      0.,     0.,  0.,     0.,     0.9846]
])
Bd = sparse.csc_matrix([
  [0.,      -0.0726,  0.,     0.0726],
  [-0.0726,  0.,      0.0726, 0.    ],
  [-0.0152,  0.0152, -0.0152, 0.0152],
  [-0.,     -0.0006, -0.,     0.0006],
  [0.0006,   0.,     -0.0006, 0.0000],
  [0.0106,   0.0106,  0.0106, 0.0106],
  [0,       -1.4512,  0.,     1.4512],
  [-1.4512,  0.,      1.4512, 0.    ],
  [-0.3049,  0.3049, -0.3049, 0.3049],
  [-0.,     -0.0236,  0.,     0.0236],
  [0.0236,   0.,     -0.0236, 0.    ],
  [0.2107,   0.2107,  0.2107, 0.2107]])
[nx, nu] = Bd.shape


# Constraints
u0 = 10.5916
umin = np.array([9.6, 9.6, 9.6, 9.6]) - u0
umax = np.array([13., 13., 13., 13.]) - u0
xmin = np.array([-np.pi/6,-np.pi/6,-np.inf,-np.inf,-np.inf,-1.,
                 -np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
xmax = np.array([ np.pi/6, np.pi/6, np.inf, np.inf, np.inf, np.inf,
                  np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

# Objective function
Q = sparse.diags([0., 0., 10., 10., 10., 10., 0., 0., 0., 5., 5., 5.])
QN = Q
R = 0.1*sparse.eye(4)

# Initial and reference states
x0 = np.zeros(12)
xr = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

# Prediction horizon
N = 10

# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
# - linear objective
q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
               np.zeros(N*nu)])
# - linear dynamics
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-x0, np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

# Create an OSQP object
prob = osqp.OSQP()

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

# Simulate in closed loop
nsim = 15
for i in range(nsim):
    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')

    # Apply first control input to the plant
    ctrl = res.x[-N*nu:-(N-1)*nu]
    x0 = Ad.dot(x0) + Bd.dot(ctrl)

    # Update initial state
    l[:nx] = -x0
    u[:nx] = -x0
    prob.update(l=l, u=u)
'''
