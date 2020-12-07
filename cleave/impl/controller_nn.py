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

# Neural network controller

import math
import numpy
import time
from typing import Mapping
from tensorflow import keras

from ..core.backend.controller import Controller
from ..core.util import PhyPropType

class ControllerNN(Controller):
    def __init__(self,
                 datafile
                 ):
        super(ControllerNN, self).__init__()

        self._t_init = 0
        self._t_begin = 0
        self._dat = datafile
        self._model = keras.models.load_model('model_3')
        self._y_series = numpy.zeros(10)
        self._y_rate_series = numpy.zeros(10)

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

        # data processing
        self._y_series = numpy.delete(self._y_series, 0)
        self._y_series = numpy.append(self._y_series, y)
        self._y_rate_series = numpy.delete(self._y_rate_series, 0)
        self._y_rate_series = numpy.append(self._y_rate_series, y_rate)
        y_blend = numpy.append(self._y_series, self._y_rate_series)
        y_horiz = numpy.reshape(y_blend, (1, 20))

        # control
        u = self._model.predict(x = y_horiz)[0][0]

        # timekeeping
        t_end = time.time_ns()
        t_iter = t_end - self._t_begin

        # screen output
        print('\r' +
              't = {:03.0f} s, '.format(t_elapsed / 1000000000) +
              'angle = {:+07.2f} deg, '.format(numpy.degrees(y)) +
              'f = {:+06.2f} N'.format(u),
              end='')

        # data file output
        self._dat.write('{:f}\t'.format(t_elapsed / 1000000000) + # elapsed time (s)
                        '{:f}\t'.format(t_period / 1000000000) + # sampling period (s)
                        '{:f}\t'.format(t_iter / 1000000000) + # execution time of the iteration (s)
                        '{:f}\t'.format(y) + # angle (rad)
                        '{:f}\t'.format(y_rate) + # angle rate (rad/s)
                        '{:f}\t'.format(z) + # position (m)
                        '{:f}\t'.format(z_rate) + # position rate (m/s)
                        '{:f}\n'.format(u) # controller actuation force (N)
                        )

        self._dat.flush()

        return {'force': u}
