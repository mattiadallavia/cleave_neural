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

from ..base.backend.controller import Controller
from ..base.util import PhyPropType

class ControllerMP(Controller):
    def __init__(self,
                 reference: float,
                 actuation_bound: float,
                 y_bound: int,
                 actuation_noise_var: float,
                 cart_mass: int,
                 pend_mass: int,
                 prediction_horizon: int,
                 # TODO: A_matrix
                 # TODO: B_matrix
                 # TODO: C_matrix
                 # TODO: x_vector
                 # TODO: u_vector
                 # TODO: x_constraint_matrix
                 # TODO: u_constraint_matrix
                 # TODO: objectives_matrix
                 datafile,
                 datafile2
                 ):
        super(ControllerMP, self).__init__()

        self._r = reference
        self._u_bound = actuation_bound
        self._y_bound = y_bound
        self._u_noise_var = actuation_noise_var
        self._t_begin = time.time_ns()
        self._dat = datafile
        self._dat2 = datafile2
        self._horizon = prediction_horizon
        self._cart_mass = cart_mass
        self._pend_mass = pend_mass
        self._time_end = 0

        self._t_curr = 0
        self._t_prev = 0
        self._e_prev = 0
        self._e_int = 0
        self._u_prev = 0

        
    def stateSpace(self):
         # TODO
         # Takes the x and u constraints and .Var() objets
         #
         # Takes the A, B & C matrices and creates .Equation() objects

         # returns: x vector

         return

    def process(self, sensor_values: Mapping[str, PhyPropType]) \
            -> Mapping[str, PhyPropType]:

        """
        Currently not a method general enough 
        """ 

        # TODO
        # Ideal way of operating for process() should be
        # 1. Keep PH and time keeping
        # 2. Call stateSpace() to create the model
        # 3.  ???

        # time keeping (stored in nanoseconds)
        self._t_prev = self._t_curr
        self._t_curr = time.time_ns()
        t_elapsed = self._t_curr - self._t_begin
        t_delta = self._t_curr - self._t_prev

        model = GEKKO(remote=False) #computes values locally, should be GEKKO(remote=False)
        model.options.IMODE = 6 #MPC
        m1 = model.Const(value=self._cart_mass)
        m2 = model.Const(value=self._pend_mass)


        # Prediction horizon
        model.time = numpy.linspace(0, 1, self._horizon)

        final = numpy.zeros(len(model.time))
        final[-1] = 1

        final = model.Param(value=final)
        
        end_loc = int(self._horizon*0.9)

        # State variables
        try:
            u = model.Var(value=self._u_prev, lb=-self._u_bound, ub=self._u_bound)
            y = model.Var(value=sensor_values['position'], lb = -self._y_bound, ub=self._y_bound)
            v = model.Var(value=sensor_values['speed'])
            theta = model.Var(value=sensor_values['angle'])
            q = model.Var(value=sensor_values['ang_vel'])
        except KeyError:
            print(sensor_values)
            raise

        # Intermediate
        eps = model.Intermediate(m2/(m1+m2))

        # State-space model
        model.Equation(y.dt() == v)
        model.Equation(v.dt() == -eps*theta + u)
        model.Equation(theta.dt() == q)
        model.Equation(q.dt() == theta -u)
        

        # Objectives
        model.Obj(final*3*theta**2)
        model.Obj(final*y**2)
        model.Obj(final*0.1*v**2)
        model.Obj(final*0.1*q**2)
        
        #model.fix(y,pos=end_loc,val=0.0)
        #model.fix(theta,pos=end_loc,val=0.0)
        #model.fix(v,pos=end_loc,val=0.0)
        #model.fix(q,pos=end_loc,val=0.0)

        model.solve(disp=False)

        # Errors
        e = self._r - y.value[0] #current angle error [rad]
        e_der = (e - self._e_prev) / (t_delta / 1000000000) # error discrete derivative
        self._e_int += e * (t_delta / 1000000000) # error discrete integral
        self._e_prev = e


        # actuation noise
        n = numpy.random.normal(0, math.sqrt(self._u_noise_var))
        u_meas = u.value[0] + n
        self._u_prev = u_meas

        self._time_end = time.time_ns()
        runtime = self._time_end - self._t_curr


        # screen output
        print('\r' +
              't = {:03.0f} s, '.format(t_elapsed / 1000000000) +
              'angle = {:+07.2f} deg, '.format(numpy.degrees(theta.value[0])) +
              'err = {:+0.4f}, '.format(e) +
              'f = {:+06.2f} N'.format(u_meas),
              end='')

        # data file output
        self._dat.write('{:.0f}\t'.format(runtime / 1000000) + # runtime of solver
                        '{:.0f}\t'.format(t_elapsed / 1000000) + # elapsed time (ms)
                        '{:.0f}\t'.format(t_delta / 1000000) + # sampling period (ms)
                        '{:f}\t'.format(theta.value[0]) + # angle (rad)
                        '{:f}\t'.format(q.value[0]) + # angle rate (rad/s)
                        '{:f}\t'.format(y.value[0]) + # position (m)
                        '{:f}\t'.format(v.value[0]) + # position rate (m/s)
                        '{:f}\t'.format(e) + # angle error (rad)
                        '{:f}\t'.format(self._e_int) + # angle error integral (rad*s)
                        '{:f}\t'.format(e_der) + # angle error derivative (rad/s)
                        '{:f}\t'.format(u.value[0]) + # controller actuation force (N)
                        '{:f}\t'.format(n) + # actuation force noise (N)
                        '{:f}\n'.format(u_meas) # total force on the cart (N)
                        )

        # write swequence of u values
        for i in len(u):
            self.dat2.write('{:f}\t'+format(u.value[i]))

        return {'force': u_meas}
