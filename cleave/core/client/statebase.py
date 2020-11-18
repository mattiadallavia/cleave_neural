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
import abc
from abc import ABC
from copy import copy
from typing import Any, Set, SupportsBytes, SupportsFloat, SupportsInt, \
    Type, final

from cleave.api.util import PhyPropMapping


class BaseSemanticVariable(SupportsFloat,
                           SupportsInt,
                           SupportsBytes,
                           abc.ABC):
    """
    Base class for semantically significant variables in a State.
    """

    def __init__(self, value: Any, record: bool = True):
        self._value = value
        self._record = record

    def get_value(self) -> Any:
        return self._value

    def set_value(self, value: Any):
        self._value = value

    def get_type(self) -> Type:
        return type(self._value)

    @property
    def record(self) -> bool:
        return self._record

    def __float__(self) -> float:
        return float(self._value)

    def __int__(self) -> int:
        return int(self._value)

    def __bytes__(self) -> bytes:
        return bytes(self._value)


class StateBase(ABC):
    """
    Defines the underlying structure for the State interface without
    having to resort to hacks.
    """

    def __init__(self):
        super(StateBase, self).__init__()
        self._sensor_vars = set()
        self._actuator_vars = set()
        self._controller_params = set()
        self._record_vars = set()

    @final
    def get_sensed_prop_names(self) -> Set[str]:
        """
        Returns
        -------
        Set
            Set containing the identifiers of the sensed variables.
        """
        return copy(self._sensor_vars)

    @final
    def get_actuated_prop_names(self) -> Set[str]:
        """
        Returns
        -------
        Set
            Set containing the identifiers of the actuated variables.
        """
        return copy(self._actuator_vars)

    @final
    def get_record_variables(self) -> PhyPropMapping:
        """
        Returns
        -------
        PhyPropMapping
            A mapping containing the values of the recorded variables in
            this state.
        """
        return {var: getattr(self, var, None) for var in self._record_vars}

    @final
    def get_controller_parameters(self) -> PhyPropMapping:
        """
        Returns
        -------
        PhyPropMapping
            A mapping from strings to values containing the initialization
            parameters for the controller associated with this physical
            simulation.
        """

        return {var: getattr(self, var) for var in self._controller_params}
