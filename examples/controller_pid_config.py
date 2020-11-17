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

# example config for a controller for an inverted pendulum plant
from cleave.impl import ControllerPID

port = 50000

controller = ControllerPID(reference = 0,
                           actuation_bound = 25,
                           gain_p = 300,
                           gain_i = 0,
                           gain_d = 30,
                           datafile = open('build/controller_pid.dat', 'w')
                           );
