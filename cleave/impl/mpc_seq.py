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

        self._t_init = 0
        self._t_begin = 0
        self._t_end = 0
        self._e_prev = 0
        self._e_int = 0
        self._u_prev = 0  

    def process(self, sensor_values: PhyPropMapping) -> PhyPropMapping:

        # time keeping (stored in nanoseconds)
        if self._t_init == 0:
            self._t_init = self._t_begin = self._t_end = time.time_ns() # first iteration time
        self._t_begin = time.time_ns()
        t_elapsed = self._t_begin - self._t_init # time since start of run
        t_period = self._t_begin - self._t_end # time it took to sample and receive new value


        model = GEKKO(remote=False) #computes values locally, should be GEKKO(remote=False)
        model.options.IMODE = 6 #MPC
        m1 = model.Const(value=self._cart_mass)
        m2 = model.Const(value=self._pend_mass)


        # Prediction horizon
        model.time = numpy.linspace(0, 0.5, self._horizon)
        end_loc = int(self._horizon*0.5)                    # PARAM
        final = numpy.zeros(len(model.time))
        i = 0

        while i < len(final):
            if i >=end_loc:
                final[i] = 1
            i += 1
        final = model.Param(value=final)

        # Read all sensor values
        y_r = sensor_values['position']
        theta_r = sensor_values['angle']
        v_r = sensor_values['speed']
        omega_r = sensor_values['ang_vel']

        deg = numpy.degrees(theta_r)

        while abs(deg) > 360:
            if deg > 0: deg -= 360
            else: deg += 360      
        if deg > 180: deg -= 360
        elif deg < -180: deg += 360


        # Errors
        e = self._r - theta_r #current angle error [rad]
        e_deg = self._r - deg
        e_der = (e - self._e_prev) / (t_period / 1000000000) # error discrete derivative
        self._e_int += e * (t_period / 1000000000) # error discrete integral
        self._e_prev = e

        # State variables
        try:
            u = model.Var(value=self._u_prev, lb=-self._u_bound, ub=self._u_bound)
            y = model.Var(value=y_r, lb = -self._y_bound, ub=self._y_bound)
            theta = model.Var(value=numpy.radians(deg))
            v = model.Var(value=v_r)
            omega = model.Var(value=omega_r)
        except KeyError:
            print(sensor_values)
            raise

        # Intermediate
        eps = model.Intermediate(m2/(m1+m2))

        # State-space model
        model.Equation(y.dt() == v)
        model.Equation(v.dt() == -eps*theta + u)
        model.Equation(theta.dt() == omega)
        model.Equation(omega.dt() == theta -u)

        # Objectives
        model.Obj(final*(theta**2))
        #if abs(y_r) < 0.2:
        #    model.Obj(0.0000001*final*(y**2))  #Remove for better solver that does not take position into account

        model.fix(theta,pos=end_loc,val=0.0)
        try:
            model.solve(disp=False)
        except:
            pass

        u_seq = numpy.zeros(self._horizon)
        n_seq = numpy.zeros(self._horizon)
        i = 0

        while i < len(u_seq):
            n = numpy.random.normal(0, math.sqrt(self._u_noise_var))
            u_k = -u.value[i]*1.4
            n_seq[i] += n
            u_seq[i] += u_k + n
            i += 1
        
        n_seq = n_seq.tolist()
        u_seq = u_seq.tolist()

        self._u_prev = u_seq[2]

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
                        '{:f}\t'.format(-u.value[2]) + # controller actuation force (N)
                        '{:f}\t'.format(n_seq[2]) + # actuation force noise (N)
                        '{:f}\n'.format(u_seq[2]) # total force on the cart (N)
                        )
        return {'force': u_seq}

