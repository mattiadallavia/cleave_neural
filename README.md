# CLEAVE Neural

As part of the CLEAVE research project, we design, implement, and evaluate a suite of 3 increasingly sophisticated networked controllers:
a PID controller, a MP controller, and neural network controller.
The purpose of the project is to investigate the performance of networked control systems in edge computing applications.
The control's objective is to balance an inverted pendulum in presence of network delay and packet loss.
The investigation is carried out via real-time software simulation of the control system.
Our results indicate that:
(1) the PID controller achieves balance of the pendulum up to a high level of noise, however it is not resilient to network disruptions;
(2) the MP controller withstands network disruptions by anticipating the system's evolution and transmitting future actuations ahead of time, but its computation time is inadequate for a real-time implementation;
(3) the neural network controller is a promising compromise of the former, with favourable control characteristics and quicker computation time, howbeit further effort is to be undertaken to refine its training and integration.
