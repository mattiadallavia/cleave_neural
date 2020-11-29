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

# Example config file for an inverted pendulum plant

from cleave.base.client import SimpleConstantActuator, SimpleSensor
from cleave.base.network import UDPControllerInterface
from cleave.impl import PlantPendulum

# class PrintSink(Sink):
#     def sink(self, values: Mapping) -> None:
#         print(values)
#
#
# @make_sink
# def fn_sink(values: Mapping) -> None:
#     print(f'Function sink! {values}')


host = 'localhost'
port = 50000

state = PlantPendulum(upd_freq_hz=6) # should be 200
controller_interface = UDPControllerInterface

sensors = [
    SimpleSensor('position', 3), # should be 100
    SimpleSensor('speed', 3), # should be 100
    SimpleSensor('angle', 3), # should be 100
    SimpleSensor('ang_vel', 3),# should be 100
]

actuators = [
    SimpleConstantActuator('force', start_value=0)
]
