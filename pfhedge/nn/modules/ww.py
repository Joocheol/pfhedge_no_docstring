from typing import List

from torch import Tensor
from torch.nn import Module

from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import ww_width

from ...instruments import Derivative
from .bs.black_scholes import BlackScholes


class WhalleyWilmott(Module):

    def __init__(self, derivative: Derivative, a: float = 1.0) -> None:
        super().__init__()
        self.derivative = derivative
        self.a = a

        self.bs = BlackScholes(derivative)

    def inputs(self) -> List[str]:
        
        return self.bs.inputs() + ["prev_hedge"]

    def extra_repr(self) -> str:
        return "a=" + _format_float(self.a) if self.a != 1 else ""

    def forward(self, input: Tensor) -> Tensor:
        prev_hedge = input[..., [-1]]

        delta = self.bs(input[..., :-1])
        width = self.width(input[..., :-1])
        min = delta - width
        max = delta + width

        return prev_hedge.clamp(min=min, max=max)

    def width(self, input: Tensor) -> Tensor:
        cost = self.derivative.underlier.cost
        spot = self.derivative.strike * input[..., [0]].exp()
        gamma = self.bs.gamma(*(input[..., [i]] for i in range(input.size(-1))))
        # width = (cost * (3 / 2) * gamma.square() * spot / self.a).pow(1 / 3)

        return ww_width(gamma=gamma, spot=spot, cost=cost, a=self.a)
