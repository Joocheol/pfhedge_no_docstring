from typing import Optional

from torch import Tensor
from torch.nn import Module

from pfhedge._utils.str import _format_float

from ..functional import clamp
from ..functional import leaky_clamp


class LeakyClamp(Module):

    def __init__(self, clamped_slope: float = 0.01, inverted_output: str = "mean"):
        super().__init__()
        self.clamped_slope = clamped_slope
        self.inverted_output = inverted_output

    def extra_repr(self) -> str:
        return "clamped_slope=" + _format_float(self.clamped_slope)

    def forward(
        self, input: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
    ) -> Tensor:
        return leaky_clamp(input, min=min, max=max, clamped_slope=self.clamped_slope)


class Clamp(Module):

    def forward(
        self, input: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
    ) -> Tensor:
        return clamp(input, min=min, max=max)
