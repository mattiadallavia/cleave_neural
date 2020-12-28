#  Copyright (c) 2020 KTH Royal Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# model predictive controller

import math
import numpy
import time
from typing import Mapping
from gekko import GEKKO
import sys
from cleave.api.controller import Controller
from cleave.api.util import PhyPropMapping

class ControllerMP_SEQ(Controller):
    def __init__(self,
                 reference: float,
                 actuation_bound: float,
                 actuation_noise_var: float,
                 y_bound: int,
                 cart_mass: int,
                 pend_mass: int,
                 prediction_horizon: int,
                 delta_t: float,
                 datafile,
                 ):
        super(ControllerMP_SEQ, self).__init__()

        self._r = reference
        self._u_bound = actuation_bound
        self._u_noise_var = actuation_noise_var
        self._y_bound = y_bound
        self._dat = datafile
        self._horizon = prediction_horizon
        self._cart_mass = cart_mass
        self._pend_mass = pend_mass
        self._delta_t = delta_t

        self._t_init = 0
        self._t_begin = 0
        self._t_end = 0
        self._e_prev = 0
        self._e_int = 0
        self._u_prev = 0 
        self._x_prev = 0
        self._theta_prev = 0
        self._v_prev = 0
        self._omega_prev = 0



    def process(self, sensor_values: PhyPropMapping) -> PhyPropMapping:

        # time keeping (stored in nanoseconds)
        if self._t_init == 0:
            self._t_init = self._t_begin = self._t_end = time.time_ns() # first iteration time
        self._t_begin = time.time_ns()
        t_elapsed = self._t_begin - self._t_init    # time since start of run
        t_period = self._t_begin - self._t_end      # time it took to sample and receive new value


        model = GEKKO(remote=False) #Should be GEKKO(remote=False), otherwise calls APmonitor server
        model.options.IMODE = 6     # Tells GEKKO that this is an MP controller
        m1 = model.Const(value=self._cart_mass)
        m2 = model.Const(value=self._pend_mass)
        l = model.Const(value=self._pend_length)
        g = model.Const(value=-9.82)


        # Prediction horizon
        model.time = numpy.linspace(0, 20, self._horizon)                  
        final = numpy.ones(len(model.time))
        i = 0

        # This loop sets what values of the end sequence that should be optimized for
        # end_loc param can be used to generate new values.
        while i < len(final):
            if i <=end_loc:
                final[i] = 0
            i += 1

        final = model.Param(value=final)

        # Read all sensor values
        # TODO implement package loss detection logic
        if #PACKAGE
            x_r = sensor_values['position']     # [m]
            theta_r = sensor_values['angle']    # [m/s]
            v_r = sensor_values['speed']        # [rad/s]
            omega_r = sensor_values['ang_vel']  # [rad/s]
            deg = numpy.degrees(theta_r)        # Degree version of theta_r, for visualization purposes.
        else:
            #[x_r, theta_r, v_r, omega_r] = controller.plant_predictor()
            # deg = numpy.degrees(theta_r) 


        # used in the model update version
        self._x_prev = x_r
        self._theta_prev = theta_r
        self._v_prev = v_r
        self._omega_prev = omega_r

        # Errors
        e = self._r - theta_r                                # current angle error [rad]
        e_deg = self._r - deg                                # current angle error [deg]
        e_der = (e - self._e_prev) / (t_period / 1000000000) # error discrete derivative
        self._e_int += e * (t_period / 1000000000)           # error discrete integral
        self._e_prev = e

        # State variables
        try:
            u = model.Var(value=self._u_prev, lb=-self._u_bound, ub=self._u_bound)  #Actuation force [N]
            x = model.Var(value=x_r, lb = -self._y_bound, ub=self._y_bound)         #Cart position [m]
            theta = model.Var(value=theta_r)                                        #Pendulum angle [rad]
            v = model.Var(value=v_r)                                                #Cart speed [m/s]
            omega = model.Var(value=omega_r)                                        #Angular velocity [rad/s]
        except KeyError:
            print(sensor_values)
            raise

        while abs(deg) > 360:
            if deg > 0: deg -= 360
            else: deg += 360      
        if deg > 180: deg -= 360
        elif deg < -180: deg += 360

        # Intermediate, GEKKO type of variable. Nothing to take note of 
        eps = model.Intermediate(m2/(m1+m2))

        # State-space model, see project report for thorough description
        model.Equation(x.dt() == v)
        model.Equation(v.dt() == -1/m1*v -m2/m1*theta*g + u/m1)
        model.Equation(theta.dt() == omega)
        model.Equation(omega.dt() == -1/(m1*l)*v -(m1+m2)*g/(m1*l)*theta +u/(m1+l))

        # Model cost function
        model.Obj((final*(theta**2)) + final*0.01*(omega**2))
    
        try:
            model.solve(disp=False)
        except:
            pass

        u_seq = numpy.zeros(self._horizon) # Numpy list
        n_seq = numpy.zeros(self._horizon) # Numpy list
        i = 1

        while i < len(u_seq):
            n = numpy.random.normal(0, math.sqrt(self._u_noise_var))  # Individual noise p
            u_k = -u.value[i]*1.4
            n_seq[i] += n
            u_seq[i] += u_k + n    
            i += 1
        
        # Convert to lists for easier management on plant end
        n_seq = n_seq.tolist()
        u_seq = u_seq.tolist()

        self._u_prev = u_seq[1]

        self._t_end = time.time_ns()
        t_iter = self._t_end - self._t_begin

        # screen output
        print('\r' +
              't = {:03.0f} s, '.format(t_elapsed / 1000000000) +
              'y = {:+07.2f} m '.format(y_r) +
              'angle = {:+07.2f} deg, '.format(numpy.degrees(theta_r)) +
              'err = {:+07.2f} deg, '.format(e_deg) +
              'f = {:+06.2f} N'.format(self._u_prev),
              end='\n')

        # data file output
        self._dat.write('{:f}\t'.format(t_elapsed / 1000000000) + # elapsed time (s)
                        '{:f}\t'.format(t_period / 1000000000) + # sampling period (s)
                        '{:f}\t'.format(t_iter / 1000000000) + # runtime of solver (s)
                        '{:f}\t'.format(theta_r) + # angle (rad)
                        '{:f}\t'.format(omega_r) + # angle rate (rad/s)
                        '{:f}\t'.format(y_r) + # position (m)
                        '{:f}\t'.format(v_r) + # position rate (m/s)
                        '{:f}\t'.format(e_der) + # angle error (rad)
                        '{:f}\t'.format(self._e_int) + # angle error integral (rad*s)
                        '{:f}\t'.format(e_der) + # angle error derivative (rad/s)
                        '{:f}\t'.format(-u.value[1]) + # controller actuation force (N)
                        '{:f}\t'.format(n_seq[1) + # actuation force noise (N)
                        '{:f}\n'.format(u_seq[1]) # total force on the cart (N)
                        )
        return {'force': u_seq}

    def plant_predictor(self):

        m1 = self._cart_mass
        m2 = self._pend_mass
        l = self._pend_length
        g = -9.82

        x = numpy.array([self._x_prev; self._v_prev; self._theta_prev; self._omega_prev])
        u = numpy.array([0; self._u_prev/m1; 0, self._u_prev]/(m1+l))
        A = numpy.array([0, 1, 0, 0; 0, -1/m1, -m2/m1*g, 0; 0, 0, 0, 1; 0, -1/(m1*l), -(m1+m2)*g/(m1*l), ])

        x_dot = A.dot(x) + u

        x_next = x_dot*self._delta_t

        x_next = x_next[0] 
        v_next = x_next[1] 
        theta_next = x_next[2]
        omega_next = x_next[3] 

        next_ls = [x_next, theta_next, v_next, omega_next]

        return next_ls
        