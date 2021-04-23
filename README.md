# Nonlinear-MPC-for-collision-free-and-deadlock-free-navigation-of-multiple-nonholonomic-mobile-robots
This repository contains Python codes for simulations and experimental implementations associated with the following paper:

@article{lafmejani2021nonlinear,
  title={Nonlinear MPC for collision-free and deadlock-free navigation of multiple nonholonomic mobile robots},
  author={Lafmejani, Amir Salimi and Berman, Spring},
  journal={Robotics and Autonomous Systems},
  volume={141},
  pages={103774},
  year={2021},
  publisher={Elsevier}
}

In this project, we present an online nonlinear Model Predictive Control (MPC) method for collision-free, deadlock-free navigation by multiple autonomous nonholonomic Wheeled Mobile Robots (WMRs). Our proposed method solves a nonlinear constrained optimization problem at each time step over a specified horizon to compute a sequence of optimal control inputs that drive the robots to target poses along collision-free trajectories, where the robots’ future states are predicted according to a unicycle kinematic model. To reduce the computational complexity of the optimization problem, we formulate it without stabilizing terminal constraints or terminal costs. We describe a computationally efficient approach to programming and solving the optimization problem, using open-source software tools for fast nonlinear optimization and applying the multiple-shooting method. We also provide rigorous proofs of the feasibility of the optimization problem and the stability of the proposed method. To validate the performance of our MPC method, we implement it in both 3D robot simulations and experiments with real nonholonomic WMRs for different multi-robot navigation scenarios with up to six robots. In all scenarios, the robots successfully navigate to their goal poses without colliding with one another or becoming trapped in a deadlock.
