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

import math
from typing import Mapping

import numpy as np
import pymunk
from pymunk.vec2d import Vec2d

from ..base.backend.controller import Controller
from ..base.client import ActuatorVariable, SensorVariable, State
from ..base.util import PhyPropType

# gravity constants
G_CONST = Vec2d(0, -9.8)

class _SortVec2D(Vec2d):
    def __gt__(self, other):
        if self.x != other.x:
            return self.x > other.x
        else:
            return self.y > other.y

    def __lt__(self, other):
        if self.x != other.x:
            return self.x < other.x
        else:
            return self.y < other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __ge__(self, other):
        return self > other or self == other

    def __le__(self, other):
        return self < other or self == other

class PlantPendulum(State):
    def __init__(self,
                 upd_freq_hz: int,
                 ground_friction: float = 0.1,
                 cart_mass: float = 0.5,
                 cart_dims: Vec2d = Vec2d(0.3, 0.2),
                 pend_com: float = 0.6,
                 pend_width: float = 0.1,
                 pend_mass: float = 0.2,
                 pend_moment: float = 0.001,  # TODO: calculate with pymunk?
                 ):
        super(PlantPendulum, self) \
            .__init__(update_freq_hz=upd_freq_hz)
        # set up state

        # space
        self._space = pymunk.Space(threaded=True)
        self._space.gravity = G_CONST

        # populate space
        # ground
        filt = pymunk.ShapeFilter(group=1)
        self._ground = pymunk.Segment(self._space.static_body,
                                      (-4, -0.1),
                                      (4, -0.1),
                                      0.1)  # TODO remove magic numbers

        self._ground.friction = ground_friction
        self._ground.filter = filt
        self._space.add(self._ground)

        # cart
        cart_moment = pymunk.moment_for_box(cart_mass, cart_dims)
        self._cart_body = pymunk.Body(mass=cart_mass, moment=cart_moment)
        self._cart_body.position = (0.0, cart_dims.y / 2)
        self._cart_shape = pymunk.Poly.create_box(self._cart_body, cart_dims)
        self._cart_shape.friction = ground_friction
        self._space.add(self._cart_body, self._cart_shape)

        # pendulum arm and mass
        pend_dims = (pend_width, pend_com * 2)
        self._pend_body = pymunk.Body(mass=pend_mass, moment=pend_moment)
        self._pend_body.position = \
            (self._cart_body.position.x,
             self._cart_body.position.y + (cart_dims.y / 2) + pend_com)
        self._pend_shape = pymunk.Poly.create_box(self._pend_body, pend_dims)
        self._pend_shape.filter = filt
        self._space.add(self._pend_body, self._pend_shape)

        # joint
        _joint_pos = self._cart_body.position + Vec2d(0, cart_dims.y / 2)
        joint = pymunk.constraint.PivotJoint(self._cart_body, self._pend_body,
                                             _joint_pos)
        joint.collide_bodies = False
        self._space.add(joint)

        # actuated and sensor variables
        self.force = ActuatorVariable(0.0)
        self.position = SensorVariable(self._cart_body.position.x)
        self.speed = SensorVariable(self._cart_body.velocity.x)
        self.angle = SensorVariable(self._pend_body.angle)
        self.ang_vel = SensorVariable(self._pend_body.angular_velocity)

    def advance(self) -> None:
        # apply actuation
        force = self.force

        self._cart_body.apply_force_at_local_point(Vec2d(force, 0.0),
                                                   Vec2d(0, 0))

        # advance the world state
        # delta T is received as nanoseconds, turn into seconds
        deltaT = self.get_delta_t()
        self._space.step(deltaT)

        angle_deg = np.degrees(self._pend_body.angle)

        print('\r' +
              'pos = {:+0.3f} m, '.format(self._cart_body.position[0]) +
              'angle = {:+07.2f} deg, '.format(angle_deg) +
              'f = {:+06.2f} N, '.format(force) +
              'delta_t = {:05.0f} us'.format(deltaT*1000000),
              end='')

        # setup new world state
        self.position = self._cart_body.position.x
        self.speed = self._cart_body.velocity.x
        self.angle = self._pend_body.angle
        self.ang_vel = self._pend_body.angular_velocity
