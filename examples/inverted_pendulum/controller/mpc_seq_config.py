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
from cleave.impl import ControllerMP_SEQ

port = 50000

controller = ControllerMP_SEQ(reference = 0, # [rad]
                          actuation_bound = 25, # [N]
                          actuation_noise_var = 0.0005, # [N^2]
                          y_bound = 100, # [m]
                          cart_mass = 0.5, # [kg]
                          pend_mass = 0.2, # [kg]
                          prediction_horizon = 50,
                          delta_t = 0.1 # [s]
                          datafile = open('data/controller_mp_seq.dat', 'w')
                           )