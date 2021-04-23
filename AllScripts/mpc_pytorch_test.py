#! /usr/bin/env python
'''
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
'''
import rospy
import numpy as np
import numpy as np
import torch
#
from torch.autograd import Variable
from mpc import mpc
from mpc.mpc import QuadCost, LinDx

# node initialization
rospy.init_node('move_node')

'''
import math
from numpy import linalg as LA
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import LaserScan
import time
from scipy.integrate import odeint
from scipy.optimize import minimize

torch.manual_seed(0)

n_batch, n_state, n_ctrl, T = 2, 3, 4, 5
n_sc = n_state + n_ctrl

# Randomly initialize a PSD quadratic cost and linear dynamics.
C = torch.randn(T*n_batch, n_sc, n_sc)
C = torch.bmm(C, C.transpose(1, 2)).view(T, n_batch, n_sc, n_sc)
c = torch.randn(T, n_batch, n_sc)

alpha = 0.2
R = (torch.eye(n_state)+alpha*torch.randn(n_state, n_state)).repeat(T, n_batch, 1, 1)
S = torch.randn(T, n_batch, n_state, n_ctrl)
F = torch.cat((R, S), dim=3)

# The initial state.
x_init = torch.randn(n_batch, n_state)

# The upper and lower control bounds.
u_lower = -torch.rand(T, n_batch, n_ctrl)
u_upper = torch.rand(T, n_batch, n_ctrl)

x_lqr, u_lqr, objs_lqr = mpc.MPC(
    n_state=n_state,
    n_ctrl=n_ctrl,
    T=T,
    u_lower=u_lower, 
    u_upper=u_upper,
    lqr_iter=20,
    verbose=1,
    backprop=False,
    exit_unconverged=False,
)(x_init, QuadCost(C, c), LinDx(F))
'''
