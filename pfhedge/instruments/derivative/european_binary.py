from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import european_binary_payoff

from ..primary.base import BasePrimary
from .base import BaseDerivative
from .base import OptionMixin


class EuropeanBinaryOption(BaseDerivative, OptionMixin):

    def __init__(
        self,
        underlier: BasePrimary,
        call: bool = True,
        strike: float = 1.0,
        maturity: float = 20 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.register_underlier("underlier", underlier)
        self.call = call
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
        params = []
        if not self.call:
            params.append("call=" + str(self.call))
        params.append("strike=" + _format_float(self.strike))
        params.append("maturity=" + _format_float(self.maturity))
        return ", ".join(params)

    def payoff_fn(self) -> Tensor:
        return european_binary_payoff(
            self.ul().spot, call=self.call, strike=self.strike
        )


# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(EuropeanBinaryOption, "simulate", BaseDerivative.simulate)
_set_attr_and_docstring(EuropeanBinaryOption, "to", BaseDerivative.to)
_set_attr_and_docstring(EuropeanBinaryOption, "ul", BaseDerivative.ul)
_set_attr_and_docstring(EuropeanBinaryOption, "list", BaseDerivative.list)
_set_docstring(EuropeanBinaryOption, "payoff", BaseDerivative.payoff)
_set_attr_and_docstring(EuropeanBinaryOption, "moneyness", OptionMixin.moneyness)
_set_attr_and_docstring(
    EuropeanBinaryOption, "log_moneyness", OptionMixin.log_moneyness
)
_set_attr_and_docstring(
    EuropeanBinaryOption, "time_to_maturity", OptionMixin.time_to_maturity
)
