import copy
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module

from pfhedge.instruments import BaseDerivative

from ._base import Feature
from ._getter import get_feature

T = TypeVar("T", bound="FeatureList")
TM = TypeVar("TM", bound="ModuleOutput")


class FeatureList(Feature):
    

    def __init__(self, features: List[Union[str, Feature]]):
        self.features = list(map(get_feature, features))

    def __len__(self) -> int:
        return len(self.features)

    def get(self, time_step: Optional[int]) -> Tensor:
        # Return size: (N, T, F)
        return torch.cat([f.get(time_step) for f in self.features], dim=-1)

    def __str__(self) -> str:
        return str(list(map(str, self.features)))

    def __repr__(self) -> str:
        return str(self)

    def of(self: T, derivative: BaseDerivative, hedger: Optional[Module] = None) -> T:
        output = copy.copy(self)
        output.features = [f.of(derivative, hedger) for f in self.features]
        return output

    def is_state_dependent(self) -> bool:
        return any(map(lambda f: f.is_state_dependent(), self.features))


class ModuleOutput(Feature, Module):

    module: Module
    inputs: FeatureList

    def __init__(self, module: Module, inputs: List[Union[str, Feature]]) -> None:
        super(Module, self).__init__()
        super(Feature, self).__init__()

        self.add_module("module", module)
        self.inputs = FeatureList(inputs)

    def extra_repr(self) -> str:
        return "inputs=" + str(self.inputs)

    def forward(self, input: Tensor) -> Tensor:
        return self.module(input)

    def get(self, time_step: Optional[int]) -> Tensor:
        return self(self.inputs.get(time_step))

    def of(self: TM, derivative: BaseDerivative, hedger: Optional[Module] = None) -> TM:
        self.inputs = self.inputs.of(derivative, hedger)
        return self

    def is_state_dependent(self) -> bool:
        return self.inputs.is_state_dependent()
