from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import realized_variance

from ..primary.base import BasePrimary
from .base import BaseDerivative


class VarianceSwap(BaseDerivative):

    def __init__(
        self,
        underlier: BasePrimary,
        strike: float = 0.04,
        maturity: float = 20 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.register_underlier("underlier", underlier)
        self.strike = strike
        self.maturity = maturity

        # TODO(simaki): Remove later. Deprecated for > v0.12.3
        if dtype is not None or device is not None:
            self.to(dtype=dtype, device=device)
            raise DeprecationWarning(
                "Specifying device and dtype when constructing a Derivative is deprecated."
                "Specify them in the constructor of the underlier instead."
            )

    def extra_repr(self) -> str:
        return ", ".join(
            (
                "strike=" + _format_float(self.strike),
                "maturity=" + _format_float(self.maturity),
            )
        )

    def payoff_fn(self) -> Tensor:
        return realized_variance(self.ul().spot, dt=self.ul().dt) - self.strike


# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(VarianceSwap, "simulate", BaseDerivative.simulate)
_set_attr_and_docstring(VarianceSwap, "to", BaseDerivative.to)
_set_attr_and_docstring(VarianceSwap, "ul", BaseDerivative.ul)
_set_attr_and_docstring(VarianceSwap, "list", BaseDerivative.list)
_set_docstring(VarianceSwap, "payoff", BaseDerivative.payoff)
