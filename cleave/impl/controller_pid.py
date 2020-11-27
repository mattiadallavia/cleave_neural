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

# PID controller
# https://w3.cs.jmu.edu/spragunr/CS354_F17/handouts/pid.pdf

import math
import numpy
import time
from typing import Mapping

from ..core.backend.controller import Controller
from ..core.util import PhyPropType

def bound(low, high, value):
    return max(low, min(high, value))

class ControllerPID(Controller):
    def __init__(self,
                 reference: float,
                 actuation_bound: float,
                 actuation_noise_var: float,
                 gain_p: float,
                 gain_i: float,
                 gain_d: float,
                 datafile
                 ):
        super(ControllerPID, self).__init__()

        self._r = reference
        self._u_bound = actuation_bound
        self._u_noise_var = actuation_noise_var
        self._k_p = gain_p
        self._k_i = gain_i
        self._k_d = gain_d
        self._t_init = 0
        self._t_begin = 0
        self._e_prev = 0
        self._e_int = 0
        self._dat = datafile

    def process(self, sensor_values: Mapping[str, PhyPropType]) \
            -> Mapping[str, PhyPropType]:

        # timekeeping
        # stored in nanoseconds
        if self._t_init == 0: self._t_init = self._t_begin = time.time_ns() # first iteration time
        t_prev = self._t_begin
        self._t_begin = time.time_ns()
        t_elapsed = self._t_begin - self._t_init
        t_period = self._t_begin - t_prev

        # measurement
        try:
            y = sensor_values['angle']
            y_rate = sensor_values['ang_vel']
            z = sensor_values['position']
            z_rate = sensor_values['speed']
        except KeyError:
            print(sensor_values)
            raise

        # control
        e = self._r - y # error
        e_der = (e - self._e_prev) / (t_period / 1000000000) # error discrete derivative
        self._e_int += e * (t_period / 1000000000) # error discrete integral
        self._e_prev = e

        u = self._k_p * e + self._k_i * self._e_int + self._k_d * e_der # command

        u = bound(-self._u_bound, self._u_bound, u)

        # actuation noise
        n = numpy.random.normal(0, math.sqrt(self._u_noise_var))
        v = u + n

        # timekeeping
        t_end = time.time_ns()
        t_iter = t_end - self._t_begin

        # screen output
        print('\r' +
              't = {:03.0f} s, '.format(t_elapsed / 1000000000) +
              'angle = {:+07.2f} deg, '.format(numpy.degrees(y)) +
              'err = {:+0.4f}, '.format(e) +
              'f = {:+06.2f} N'.format(u),
              end='')

        # data file output
        self._dat.write('{:.0f}\t'.format(t_elapsed / 1000000) + # elapsed time (ms)
                        '{:.0f}\t'.format(t_period / 1000000) + # sampling period (ms)
                        '{:.0f}\t'.format(t_iter / 1000000) + # execution time of the iteration (ms)
                        '{:f}\t'.format(y) + # angle (rad)
                        '{:f}\t'.format(y_rate) + # angle rate (rad/s)
                        '{:f}\t'.format(z) + # position (m)
                        '{:f}\t'.format(z_rate) + # position rate (m/s)
                        '{:f}\t'.format(e) + # angle error (rad)
                        '{:f}\t'.format(self._e_int) + # angle error integral (rad*s)
                        '{:f}\t'.format(e_der) + # angle error derivative (rad/s)
                        '{:f}\t'.format(u) + # controller actuation force (N)
                        '{:f}\t'.format(n) + # actuation force noise (N)
                        '{:f}\n'.format(v) # total force on the cart (N)
                        )

        return {'force': v}
