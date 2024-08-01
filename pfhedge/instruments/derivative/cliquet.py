from math import floor

from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import european_forward_start_payoff

from ..primary.base import BasePrimary
from .base import BaseDerivative


class EuropeanForwardStartOption(BaseDerivative):

    def __init__(
        self,
        underlier: BasePrimary,
        strike: float = 1.0,
        maturity: float = 20 / 250,
        start: float = 10 / 250,
    ) -> None:
        super().__init__()
        self.register_underlier("underlier", underlier)
        self.strike = strike
        self.maturity = maturity
        self.start = start

    def extra_repr(self) -> str:
        params = [
            "strike=" + _format_float(self.strike),
            "maturity=" + _format_float(self.maturity),
            "start=" + _format_float(self.start),
        ]
        return ", ".join(params)

    def _start_index(self) -> int:
        return floor(self.start / self.ul().dt)

    def payoff_fn(self) -> Tensor:
        return european_forward_start_payoff(
            self.ul().spot, strike=self.strike, start_index=self._start_index()
        )


# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(EuropeanForwardStartOption, "simulate", BaseDerivative.simulate)
_set_attr_and_docstring(EuropeanForwardStartOption, "to", BaseDerivative.to)
_set_attr_and_docstring(EuropeanForwardStartOption, "ul", BaseDerivative.ul)
_set_attr_and_docstring(EuropeanForwardStartOption, "list", BaseDerivative.list)
_set_docstring(EuropeanForwardStartOption, "payoff", BaseDerivative.payoff)
