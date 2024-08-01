import copy
from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import TypeVar

from torch import Tensor
from torch.nn import Module

from pfhedge.instruments import BaseDerivative

T = TypeVar("T", bound="Feature")


class Feature(ABC):
    

    name: str
    derivative: BaseDerivative
    hedger: Optional[Module]

    def __init__(self) -> None:
        self.register_hedger(None)

    @abstractmethod
    def get(self, time_step: Optional[int]) -> Tensor:
        pass
        

    def of(self: T, derivative: BaseDerivative, hedger: Optional[Module] = None) -> T:
        
        output = copy.copy(self)
        output.register_derivative(derivative)
        output.register_hedger(hedger)
        return output

    def register_derivative(self, derivative: BaseDerivative) -> None:
        setattr(self, "derivative", derivative)

    def register_hedger(self, hedger: Optional[Module]) -> None:
        setattr(self, "hedger", hedger)

    def _get_name(self) -> str:
        return self.__class__.__name__

    def is_state_dependent(self) -> bool:
        # If a feature uses the state of a hedger, it is state dependent.
        return getattr(self, "hedger") is not None

    def __str__(self):
        return self.name

    # TODO(simaki) Remove later
    def __getitem__(self, time_step: Optional[int]) -> Tensor:
        # raise DeprecationWarning("Use `<feature>.get(time_step)` instead")
        return self.get(time_step)


class StateIndependentFeature(Feature):
    # Features that does not use the state of the hedger.

    derivative: BaseDerivative
    hedger: None

    def of(
        self: "StateIndependentFeature",
        derivative: BaseDerivative,
        hedger: Optional[Module] = None,
    ) -> "StateIndependentFeature":
        return super().of(derivative=derivative, hedger=None)
