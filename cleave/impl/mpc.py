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

# proportional-derivative controller

import math
import numpy
import time
from typing import Mapping
from gekko import GEKKO
import sys
from cleave.api.controller import Controller
from cleave.api.util import PhyPropMapping

class ControllerMP(Controller):
    def __init__(self,
                 reference: float,
                 actuation_bound: float,
                 actuation_noise_var: float,
                 y_bound: int,
                 cart_mass: int,
                 pend_mass: int,
                 prediction_horizon: int,
                 datafile,
                 datafile2
                 ):
        super(ControllerMP, self).__init__()

        self._r = reference
        self._u_bound = actuation_bound
        self._u_noise_var = actuation_noise_var
        self._y_bound = y_bound
        self._dat = datafile
        self._dat2 = datafile2
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
        model.time = numpy.linspace(0, 1, self._horizon)
        end_loc = int(self._horizon*0.8)                    # PARAM
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
        model.Obj(10*final*(theta**2)) 
        model.fix(theta,pos=end_loc,val=0.0) 

        model.solve(disp=False)

        # actuation noise
        n = numpy.random.normal(0, math.sqrt(self._u_noise_var))
        u_k = -u.value[1]*1.8
        u_r = u_k + n
        self._u_prev = u_r

        self._t_end = time.time_ns()
        t_iter = self._t_end - self._t_begin

        # screen output
        print('\r' +
              't = {:03.0f} s, '.format(t_elapsed / 1000000000) +
              'angle = {:+07.2f} deg, '.format(numpy.degrees(theta_r)) +
              'err = {:+07.2f} deg, '.format(e_deg) +
              'f = {:+06.2f} N'.format(u_r),
              end='\n')

        # data file output
        self._dat.write('{:f}\t'.format(t_elapsed / 1000000000) + # elapsed time (s)
                        '{:f}\t'.format(t_period / 1000000000) + # sampling period (s)
                        '{:f}\t'.format(t_iter / 1000000000) + # runtime of solver (s)
                        '{:f}\t'.format(theta_r) + # angle (rad)
                        '{:f}\t'.format(omega_r) + # angle rate (rad/s)
                        '{:f}\t'.format(y_r) + # position (m)
                        '{:f}\t'.format(v_r) + # position rate (m/s)
                        '{:f}\t'.format(e_deg) + # angle error (rad) TODO should be e_der
                        '{:f}\t'.format(self._e_int) + # angle error integral (rad*s)
                        '{:f}\t'.format(e_der) + # angle error derivative (rad/s)
                        '{:f}\t'.format(u_k) + # controller actuation force (N)
                        '{:f}\t'.format(n) + # actuation force noise (N)
                        '{:f}\n'.format(u_r) # total force on the cart (N)
                        )
        
        ls = []
        for i in u.value:
            ls.append(i)
        ls = str(ls)
        self._dat2.write('{:s}\n'.format(ls))    

        return {'force': u_r}

