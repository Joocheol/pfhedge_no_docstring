from inspect import signature
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import no_type_check

from torch import Tensor
from torch.nn import Module

import pfhedge.autogreek as autogreek
from pfhedge.instruments import AmericanBinaryOption
from pfhedge.instruments import EuropeanBinaryOption
from pfhedge.instruments import EuropeanOption
from pfhedge.instruments import LookbackOption


class BSModuleMixin(Module):
    

    def forward(self, input: Tensor) -> Tensor:
        
        return self.delta(*(input[..., [i]] for i in range(input.size(-1))))

    @no_type_check
    def price(self, *args, **kwargs) -> Tensor:
        pass
        
    @no_type_check
    def delta(self, **kwargs) -> Tensor:
        
        return autogreek.delta(self.price, **kwargs)

    @no_type_check
    def gamma(self, **kwargs) -> Tensor:
        
        return autogreek.gamma(self.price, **kwargs)

    @no_type_check
    def vega(self, **kwargs) -> Tensor:
        
        return autogreek.vega(self.price, **kwargs)

    @no_type_check
    def theta(self, **kwargs) -> Tensor:
        
        return autogreek.theta(self.price, **kwargs)

    def inputs(self) -> List[str]:
        
        return list(signature(self.delta).parameters.keys())


def acquire_params_from_derivative_0(
    derivative: Optional[
        Union[
            EuropeanOption, EuropeanBinaryOption, AmericanBinaryOption, LookbackOption
        ]
    ],
    log_moneyness: Optional[Tensor] = None,
    time_to_maturity: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if log_moneyness is None:
        if derivative is None:
            raise ValueError(
                "log_moneyness is required if derivative is not set at this initialization."
            )
        log_moneyness = derivative.log_moneyness()
    if time_to_maturity is None:
        if derivative is None:
            raise ValueError(
                "time_to_maturity is required if derivative is not set at this initialization."
            )
        time_to_maturity = derivative.time_to_maturity()
    return log_moneyness, time_to_maturity


def acquire_params_from_derivative_1(
    derivative: Optional[
        Union[
            EuropeanOption, EuropeanBinaryOption, AmericanBinaryOption, LookbackOption
        ]
    ],
    log_moneyness: Optional[Tensor] = None,
    time_to_maturity: Optional[Tensor] = None,
    volatility: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    log_moneyness, time_to_maturity = acquire_params_from_derivative_0(
        derivative=derivative,
        log_moneyness=log_moneyness,
        time_to_maturity=time_to_maturity,
    )
    if volatility is None:
        if derivative is None:
            raise ValueError(
                "time_to_maturity is required if derivative is not set at this initialization."
            )
        if derivative.ul().volatility is None:
            raise AttributeError(
                "please simulate first and check if volatility exists in the derivative's underlier."
            )
        volatility = derivative.ul().volatility
    return log_moneyness, time_to_maturity, volatility


def acquire_params_from_derivative_2(
    derivative: Optional[Union[AmericanBinaryOption, LookbackOption]],
    log_moneyness: Optional[Tensor] = None,
    max_log_moneyness: Optional[Tensor] = None,
    time_to_maturity: Optional[Tensor] = None,
    volatility: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    log_moneyness, time_to_maturity, volatility = acquire_params_from_derivative_1(
        derivative=derivative,
        log_moneyness=log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
    )
    if max_log_moneyness is None:
        if derivative is None:
            raise ValueError(
                "max_log_moneyness is required if derivative is not set at this initialization."
            )
        max_log_moneyness = derivative.max_log_moneyness()

    return log_moneyness, max_log_moneyness, time_to_maturity, volatility
