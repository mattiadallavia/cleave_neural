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

from ..base.backend.controller import Controller
from ..base.util import PhyPropType

class ControllerPID(Controller):
    def __init__(self):
        super(ControllerPID, self).__init__()
        self._t_begin = time.time_ns()
        self._t_curr = 0
        self._t_prev = 0
        self._e_prev = 0
        self._e_int = 0
        self._dat = open('data/controller_pid.dat', 'w')

    def process(self, sensor_values: Mapping[str, PhyPropType]) \
            -> Mapping[str, PhyPropType]:

        # timekeeping
        # stored in nanoseconds
        self._t_prev = self._t_curr
        self._t_curr = time.time_ns()
        t_elapsed = self._t_curr - self._t_begin
        t_delta = self._t_curr - self._t_prev

        # measurement
        try:
            y = sensor_values['angle']
        except KeyError:
            print(sensor_values)
            raise

        # control
        r = 0 # setpoint
        k_p = 100 # proportional gain
        k_i = 0 # integral gain
        k_d = 3 # derivative gain

        e = r - y # error
        e_der = (e - self._e_prev) / (t_delta / 1000000000) # error discrete derivative
        self._e_int += e * (t_delta / 1000000000) # error discrete integral
        self._e_prev = e

        u = k_p * e + k_i * self._e_int + k_d * e_der # command

        # screen output
        print('\r' +
              't = {:06.0f} ms, '.format(t_elapsed / 1000000) +
              'angle = {:+07.2f} deg, '.format(numpy.degrees(y)) +
              'err = {:+0.4f}, '.format(e) +
              'f = {:+06.2f} N'.format(u),
              end='')

        # data file output
        self._dat.write('{:.0f}\t'.format(t_elapsed / 1000000) +
                        '{:f}\t'.format(numpy.degrees(y)) +
                        '{:f}\t'.format(e) +
                        '{:f}\n'.format(u))

        return {'force': u}
