from typing import List
from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.bisect import find_implied_volatility
from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.instruments import EuropeanBinaryOption
from pfhedge.nn.functional import bs_european_binary_delta
from pfhedge.nn.functional import bs_european_binary_gamma
from pfhedge.nn.functional import bs_european_binary_price
from pfhedge.nn.functional import bs_european_binary_theta
from pfhedge.nn.functional import bs_european_binary_vega

from ._base import BSModuleMixin
from ._base import acquire_params_from_derivative_0
from ._base import acquire_params_from_derivative_1
from .black_scholes import BlackScholesModuleFactory


class BSEuropeanBinaryOption(BSModuleMixin):
    

    def __init__(
        self,
        call: bool = True,
        strike: float = 1.0,
        derivative: Optional[EuropeanBinaryOption] = None,
    ) -> None:
        super().__init__()
        self.call = call
        self.strike = strike
        self.derivative = derivative

    @classmethod
    def from_derivative(cls, derivative):
        
        return cls(
            call=derivative.call, strike=derivative.strike, derivative=derivative
        )

    def extra_repr(self) -> str:
        params = []
        if not self.call:
            params.append("call=" + str(self.call))
        params.append("strike=" + _format_float(self.strike))
        return ", ".join(params)

    def inputs(self) -> List[str]:
        return ["log_moneyness", "time_to_maturity", "volatility"]

    def price(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        
        (
            log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_1(
            self.derivative, log_moneyness, time_to_maturity, volatility
        )
        return bs_european_binary_price(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
        )

    @torch.enable_grad()
    def delta(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        
        (
            log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_1(
            self.derivative, log_moneyness, time_to_maturity, volatility
        )
        return bs_european_binary_delta(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
            strike=self.strike,
        )

    def gamma(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        
        (
            log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_1(
            self.derivative, log_moneyness, time_to_maturity, volatility
        )
        return bs_european_binary_gamma(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
            strike=self.strike,
        )

    def vega(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        
        (
            log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_1(
            self.derivative, log_moneyness, time_to_maturity, volatility
        )
        return bs_european_binary_vega(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
            strike=self.strike,
        )

    def theta(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        
        (
            log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_1(
            self.derivative, log_moneyness, time_to_maturity, volatility
        )
        return bs_european_binary_theta(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
            strike=self.strike,
        )

    def implied_volatility(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        price: Optional[Tensor] = None,
        precision: float = 1e-6,
    ) -> Tensor:
        
        (log_moneyness, time_to_maturity) = acquire_params_from_derivative_0(
            self.derivative, log_moneyness, time_to_maturity
        )
        if price is None:
            raise ValueError(
                "price is required in this method. None is set only for compatibility to the previous versions."
            )
        return find_implied_volatility(
            self.price,
            price=price,
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            precision=precision,
        )


factory = BlackScholesModuleFactory()
factory.register_module("EuropeanBinaryOption", BSEuropeanBinaryOption)

# Assign docstrings so they appear in Sphinx documentation
_set_docstring(BSEuropeanBinaryOption, "inputs", BSModuleMixin.inputs)
_set_attr_and_docstring(BSEuropeanBinaryOption, "forward", BSModuleMixin.forward)
