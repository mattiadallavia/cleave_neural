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

# proportional controller
import math
import numpy
import time
from typing import Mapping

from ..base.backend.controller import Controller
from ..base.util import PhyPropType

class ControllerP(Controller):
    def __init__(self):
        super(ControllerP, self).__init__()
        self._t_begin = time.time_ns()
        self._dat = open('controller_p.dat', 'w')

    def process(self, sensor_values: Mapping[str, PhyPropType]) \
            -> Mapping[str, PhyPropType]:

        try:
            y = sensor_values['angle']
        except KeyError:
            print(sensor_values)
            raise

        r = 0 # setpoint
        k_p = 100 # proportional gain

        # control
        e = r - y # error
        u = k_p * e # command

        # screen output
        print('\r' +
              't = {:06.0f} ms, '.format((time.time_ns() - self._t_begin)/1000000) +
              'angle = {:+07.2f} deg, '.format(numpy.degrees(y)) +
              'err = {:+0.4f}, '.format(e) +
              'f = {:+06.2f} N'.format(u),
              end='')

        # data file output
        self._dat.write('{:.0f}\t'.format((time.time_ns() - self._t_begin)/1000000) +
                        '{:f}\t'.format(numpy.degrees(y)) +
                        '{:f}\t'.format(e) +
                        '{:f}\n'.format(u))

        return {'force': u}
