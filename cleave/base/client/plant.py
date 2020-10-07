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

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections import Mapping
from threading import RLock

from twisted.internet import threads
from twisted.internet.posixbase import PosixReactorBase
from twisted.python.failure import Failure

from .actuator import Actuator, ActuatorArray
from .sensor import NoSensorUpdate, Sensor, SensorArray
from .state import State
from ..network.client import BaseControllerInterface
from ...base.util import PhyPropType, nanos2seconds, seconds2nanos


class PlantBuilderWarning(Warning):
    pass


class EmulationWarning(Warning):
    pass


class Plant(ABC):
    """
    Interface for all plants.
    """

    @abstractmethod
    def execute(self):
        """
        Executes this plant. Depending on implementation, this method may or
        may not be asynchronous.

        Returns
        -------

        """
        pass

    @property
    @abstractmethod
    def update_freq_hz(self) -> int:
        """
        The update frequency of this plant in Hz. Depending on
        implementation, accessing this property may or may not be thread-safe.

        Returns
        -------
        int
            The update frequency of the plant in Hertz.

        """
        pass

    @property
    @abstractmethod
    def plant_state(self) -> State:
        """
        The State object associated with this plant. Depending on
        implementation, accessing this property may or may not be thread-safe.

        Returns
        -------
        State
            The plant State.
        """
        pass


class _BasePlant(Plant):
    def __init__(self,
                 reactor: PosixReactorBase,
                 update_freq: int,
                 state: State,
                 sensor_array: SensorArray,
                 actuator_array: ActuatorArray,
                 control_interface: BaseControllerInterface):
        self._reactor = reactor
        self._freq = update_freq
        self._state = state
        self._sensors = sensor_array
        self._actuators = actuator_array
        self._cycles = 0
        self._control = control_interface

        self._lock = RLock()

    @property
    def update_freq_hz(self) -> int:
        return self._freq

    @property
    def plant_state(self) -> State:
        return self._state

    def _emu_step(self):
        # 1. get raw actuation inputs
        # 2. process actuation inputs
        # 3. advance state
        # 4. process sensor outputs
        # 5. send sensor outputs

        act = self._control.get_actuator_values()
        proc_act = self._actuators.process_actuation_inputs(act)
        with self._lock:
            # this is always called from a separate thread so add some
            # thread-safety just in case
            self._state.actuate(proc_act)
            self._state.advance()
            sensor_samples = self._state.get_state()
            self._cycles += 1

            # this will only send sensor updates if we actually have any,
            # since otherwise it raises an exception which will be catched in
            # the callback
            proc_sens = self._sensors.process_plant_state(sensor_samples)
            self._control.put_sensor_values(proc_sens)

    def _timestep(self, target_dt_ns: int):
        ti = time.monotonic_ns()

        if not self._control.is_ready():
            # controller not ready, wait a bit
            self._reactor.callLater(0.01, self._timestep, target_dt_ns)
            return

        def no_samples_to_send(failure: Failure):
            failure.trap(NoSensorUpdate)
            # no sensor data to send, ignore
            return

        def send_step_results(sensor_samples: Mapping[str, PhyPropType]):
            self._control.put_sensor_values(sensor_samples)

        def reschedule_step_callback(*args, **kwargs):
            dt = nanos2seconds(target_dt_ns - (time.monotonic_ns() - ti))
            if dt >= 0:
                self._reactor.callLater(dt, self._timestep, target_dt_ns)
            else:
                warnings.warn(
                    'Emulation step took longer than allotted time slot!',
                    EmulationWarning)
                self._reactor.callLater(0, self._timestep, target_dt_ns)

        # instead of using the default deferToThread method
        # this way we can pass the reactor and don't have to trust the
        # function to use the default one.
        threads.deferToThreadPool(self._reactor,
                                  self._reactor.getThreadPool(),
                                  self._emu_step) \
            .addCallback(send_step_results) \
            .addErrback(no_samples_to_send) \
            .addCallback(reschedule_step_callback)

        # threads \
        #     .deferToThread(self._emu_step) \
        #     .addCallback(reschedule_step_callback)

    def execute(self):
        target_dt_ns = seconds2nanos(1.0 / self._freq)
        self._control.register_with_reactor(self._reactor)

        self._reactor.callWhenRunning(self._timestep, target_dt_ns)

        self._reactor.suggestThreadPoolSize(3)  # input, output and processing
        self._reactor.run()


# noinspection PyAttributeOutsideInit
class PlantBuilder:
    """
    Builder for plant objects.

    This class is not meant to be instantiated by users --- a library
    singleton is provided as cleave.client.builder.
    """

    def reset(self) -> None:
        """
        Resets this builder, removing all previously added sensors,
        actuators, as well as detaching the plant state and comm client.

        Returns
        -------

        """
        self._sensors = []
        self._actuators = []
        # self._comm_client = None
        self._controller = None
        self._plant_state = None

    def __init__(self, reactor: PosixReactorBase):
        self._reactor = reactor
        self.reset()

    def attach_sensor(self, sensor: Sensor) -> None:
        """
        Attaches a sensor to the plant under construction.

        Parameters
        ----------
        sensor
            A Sensor instance to be attached to the target plant.


        Returns
        -------

        """
        self._sensors.append(sensor)

    def attach_actuator(self, actuator: Actuator) -> None:
        """
        Attaches an actuator to the plant under construction.

        Parameters
        ----------
        actuator
            An Actuator instance to be attached to the target plant.

        Returns
        -------

        """
        self._actuators.append(actuator)

    def set_controller(self, controller: BaseControllerInterface) -> None:
        if self._controller is not None:
            warnings.warn(
                'Replacing already set controller for plant.',
                PlantBuilderWarning
            )

        self._controller = controller

    def set_plant_state(self, plant_state: State) -> None:
        """
        Sets the State that will govern the evolution of the plant.

        Note that any previously assigned State will be overwritten by this
        operation.

        Parameters
        ----------
        plant_state
            A State instance to assign to the plant.

        Returns
        -------

        """
        if self._plant_state is not None:
            warnings.warn(
                'Replacing already set State for plant.',
                PlantBuilderWarning
            )

        self._plant_state = plant_state

    def build(self) -> Plant:
        """
        Builds a Plant instance and returns it. The actual subtype of this
        plant will depend on the previously provided parameters.

        Returns
        -------
        Plant
            A Plant instance.

        """

        # TODO: raise error if missing parameters OR instantiate different
        #  types of plants?
        try:
            return _BasePlant(
                reactor=self._reactor,
                update_freq=self._plant_state.update_frequency,
                state=self._plant_state,
                sensor_array=SensorArray(
                    plant_freq=self._plant_state.update_frequency,
                    sensors=self._sensors),
                actuator_array=ActuatorArray(actuators=self._actuators),
                control_interface=self._controller
            )
        finally:
            self.reset()