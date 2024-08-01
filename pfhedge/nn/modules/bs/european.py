from typing import Optional

from torch import Tensor

from pfhedge._utils.bisect import find_implied_volatility
from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.str import _format_float
from pfhedge.instruments import EuropeanOption
from pfhedge.nn.functional import bs_european_delta
from pfhedge.nn.functional import bs_european_gamma
from pfhedge.nn.functional import bs_european_price
from pfhedge.nn.functional import bs_european_theta
from pfhedge.nn.functional import bs_european_vega

from ._base import BSModuleMixin
from ._base import acquire_params_from_derivative_0
from ._base import acquire_params_from_derivative_1
from .black_scholes import BlackScholesModuleFactory


class BSEuropeanOption(BSModuleMixin):
    

    def __init__(
        self,
        call: bool = True,
        strike: float = 1.0,
        derivative: Optional[EuropeanOption] = None,
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
        return bs_european_delta(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
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
        return bs_european_gamma(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
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
        return bs_european_vega(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
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
        return bs_european_theta(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            strike=self.strike,
        )

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
        return bs_european_price(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            strike=self.strike,
            call=self.call,
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
factory.register_module("EuropeanOption", BSEuropeanOption)

# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(BSEuropeanOption, "inputs", BSModuleMixin.inputs)
_set_attr_and_docstring(BSEuropeanOption, "forward", BSModuleMixin.forward)
