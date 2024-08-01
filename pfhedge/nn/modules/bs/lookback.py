from typing import List
from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.bisect import find_implied_volatility
from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.instruments import LookbackOption
from pfhedge.nn.functional import bs_lookback_price

from ._base import BSModuleMixin
from ._base import acquire_params_from_derivative_0
from ._base import acquire_params_from_derivative_2
from .black_scholes import BlackScholesModuleFactory


class BSLookbackOption(BSModuleMixin):
    

    def __init__(
        self,
        call: bool = True,
        strike: float = 1.0,
        derivative: Optional[LookbackOption] = None,
    ) -> None:
        if not call:
            raise ValueError(
                f"{self.__class__.__name__} for a put option is not yet supported."
            )

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
        params.append("strike=" + _format_float(self.strike))
        return ", ".join(params)

    def inputs(self) -> List[str]:
        return ["log_moneyness", "max_log_moneyness", "time_to_maturity", "volatility"]

    def price(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        (
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_2(
            self.derivative,
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        )
        return bs_lookback_price(
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            strike=self.strike,
        )

    @torch.enable_grad()
    def delta(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
        create_graph: bool = False,
    ) -> Tensor:
        
        (
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_2(
            self.derivative,
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        )
        return super().delta(
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            create_graph=create_graph,
            strike=self.strike,
        )

    @torch.enable_grad()
    def gamma(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        
        (
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_2(
            self.derivative,
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        )
        return super().gamma(
            strike=self.strike,
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
        )

    @torch.enable_grad()
    def vega(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        
        (
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_2(
            self.derivative,
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        )
        return super().vega(
            strike=self.strike,
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
        )

    @torch.enable_grad()
    def theta(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        
        (
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_2(
            self.derivative,
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        )
        return super().theta(
            strike=self.strike,
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
        )

    def implied_volatility(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        price: Optional[Tensor] = None,
        precision: float = 1e-6,
    ) -> Tensor:
        
        # price seems optional in typing, but it isn't. It is set for the compatibility to the previous versions.
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
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            precision=precision,
        )


factory = BlackScholesModuleFactory()
factory.register_module("LookbackOption", BSLookbackOption)

# Assign docstrings so they appear in Sphinx documentation
_set_docstring(BSLookbackOption, "inputs", BSModuleMixin.inputs)
_set_attr_and_docstring(BSLookbackOption, "forward", BSModuleMixin.forward)
