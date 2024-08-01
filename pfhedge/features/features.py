from typing import List
from typing import Optional
from typing import Type

import torch
from torch import Tensor
from torch.nn import Module

from pfhedge._utils.str import _format_float
from pfhedge.instruments.derivative.base import BaseDerivative
from pfhedge.instruments.derivative.base import OptionMixin

from ._base import Feature
from ._base import StateIndependentFeature
from ._getter import FeatureFactory


# for mypy only
class OptionType(BaseDerivative, OptionMixin):
    pass


class Moneyness(StateIndependentFeature):
    

    derivative: OptionType

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "log_moneyness" if self.log else "moneyness"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        return self.derivative.moneyness(time_step, log=self.log).unsqueeze(-1)


class LogMoneyness(Moneyness):

    derivative: OptionType

    def __init__(self) -> None:
        super().__init__(log=True)


class TimeToMaturity(StateIndependentFeature):
    

    derivative: OptionType
    name = "time_to_maturity"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        return self.derivative.time_to_maturity(time_step).unsqueeze(-1)


class ExpiryTime(TimeToMaturity):
    

    def __str__(self) -> str:
        return "expiry_time"


class UnderlierSpot(StateIndependentFeature):
    

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "underlier_log_spot" if self.log else "underlier_spot"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        output = self.derivative.ul().spot[:, index].unsqueeze(-1)
        if self.log:
            output.log_()
        return output


class UnderlierLogSpot(UnderlierSpot):
    

    def __init__(self):
        super().__init__(log=True)


class Spot(StateIndependentFeature):
    

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "log_spot" if self.log else "spot"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        output = self.derivative.spot[:, index].unsqueeze(-1)
        if self.log:
            output.log_()
        return output


class Volatility(StateIndependentFeature):
    

    name = "volatility"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        return self.derivative.ul().volatility[:, index].unsqueeze(-1)


class Variance(StateIndependentFeature):
    

    name = "variance"

    def get(self, time_step: Optional[int]) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        return self.derivative.ul().variance[:, index].unsqueeze(-1)


class PrevHedge(Feature):
    

    hedger: Module
    name = "prev_hedge"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        if time_step is None:
            raise ValueError("time_step for prev_output should be specified")
        return self.hedger.get_buffer("prev_output")


class Barrier(StateIndependentFeature):
    

    def __init__(self, threshold: float, up: bool = True) -> None:
        super().__init__()
        self.threshold = threshold
        self.up = up

    def __repr__(self) -> str:
        params = [_format_float(self.threshold), "up=" + str(self.up)]
        return self._get_name() + "(" + ", ".join(params) + ")"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        spot = self.derivative.ul().spot
        if time_step is None:
            if self.up:
                max = spot.cummax(-1).values
                output = (max >= self.threshold).to(spot.dtype)
            else:
                min = spot.cummin(-1).values
                output = (min <= self.threshold).to(spot.dtype)
        else:
            if self.up:
                max = spot[..., : time_step + 1].max(-1, keepdim=True).values
                output = (max >= self.threshold).to(spot.dtype)
            else:
                min = spot[..., : time_step + 1].min(-1, keepdim=True).values
                output = (min <= self.threshold).to(self.derivative.ul().spot.dtype)
        return output.unsqueeze(-1)


class Zeros(StateIndependentFeature):
    

    name = "zeros"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if time_step is not None else ...
        return torch.zeros_like(self.derivative.ul().spot[..., index]).unsqueeze(-1)


class Ones(StateIndependentFeature):
    

    name = "ones"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if time_step is not None else ...
        return torch.ones_like(self.derivative.ul().spot[..., index]).unsqueeze(-1)


class Empty(StateIndependentFeature):
    

    name = "empty"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if time_step is not None else ...
        return torch.empty_like(self.derivative.ul().spot[..., index]).unsqueeze(-1)


class MaxMoneyness(StateIndependentFeature):
    

    derivative: OptionType

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "max_log_moneyness" if self.log else "max_moneyness"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        return self.derivative.max_moneyness(time_step, log=self.log).unsqueeze(-1)


class MaxLogMoneyness(MaxMoneyness):
    

    derivative: OptionType

    def __init__(self) -> None:
        super().__init__(log=True)


FEATURES: List[Type[Feature]] = [
    Empty,
    ExpiryTime,
    TimeToMaturity,
    LogMoneyness,
    MaxLogMoneyness,
    MaxMoneyness,
    Moneyness,
    PrevHedge,
    Variance,
    Volatility,
    Zeros,
    Spot,
    UnderlierSpot,
]

for cls in FEATURES:
    FeatureFactory().register_feature(str(cls()), cls)
