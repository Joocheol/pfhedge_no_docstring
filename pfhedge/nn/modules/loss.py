from abc import ABC
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from pfhedge._utils.bisect import bisect
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar

from ..functional import entropic_risk_measure
from ..functional import exp_utility
from ..functional import expected_shortfall
from ..functional import isoelastic_utility
from ..functional import quadratic_cvar


class HedgeLoss(Module, ABC):
    

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        pass
        

    def cash(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        
        pl = input - target
        return bisect(self, self(pl), pl.min(), pl.max())


class EntropicRiskMeasure(HedgeLoss):


    def __init__(self, a: float = 1.0) -> None:
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")

        super().__init__()
        self.a = a

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return entropic_risk_measure(input - target, a=self.a)

    def cash(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -self(input - target)


class EntropicLoss(HedgeLoss):

    def __init__(self, a: float = 1.0) -> None:
        if not a > 0:
            raise ValueError("Risk aversion coefficient should be positive.")

        super().__init__()
        self.a = a

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -exp_utility(input - target, a=self.a).mean(0)

    def cash(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -(-exp_utility(input - target, a=self.a).mean(0)).log() / self.a


class IsoelasticLoss(HedgeLoss):

    def __init__(self, a: float) -> None:
        if not 0 < a <= 1:
            raise ValueError(
                "Relative risk aversion coefficient should satisfy 0 < a <= 1."
            )

        super().__init__()
        self.a = a

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -isoelastic_utility(input - target, a=self.a).mean(0)


class ExpectedShortfall(HedgeLoss):

    def __init__(self, p: float = 0.1):
        if not 0 < p <= 1:
            raise ValueError("The quantile level should satisfy 0 < p <= 1.")

        super().__init__()
        self.p = p

    def extra_repr(self) -> str:
        return str(self.p)

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return expected_shortfall(input - target, p=self.p, dim=0)

    def cash(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -self(input - target)


class QuadraticCVaR(HedgeLoss):
      # NOQA

    def __init__(self, lam: float = 10.0):
        if not lam >= 1.0:
            raise ValueError("The lam should satisfy lam >= 1.")

        super().__init__()
        self.lam = lam

    def extra_repr(self) -> str:
        return str(self.lam)

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return quadratic_cvar(input - target, lam=self.lam, dim=0)

    def cash(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return -self(input - target)


class OCE(HedgeLoss):

    def __init__(self, utility: Callable[[Tensor], Tensor]) -> None:
        super().__init__()

        self.utility = utility
        self.w = Parameter(torch.tensor(0.0))

    def extra_repr(self) -> str:
        w = float(self.w.item())
        return self.utility.__name__ + ", w=" + _format_float(w)

    def forward(self, input: Tensor, target: TensorOrScalar = 0.0) -> Tensor:
        return self.w - self.utility(input - target + self.w).mean(0)
