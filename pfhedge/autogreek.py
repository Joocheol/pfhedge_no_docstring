from inspect import signature
from typing import Any
from typing import Callable

import torch
from torch import Tensor

from ._utils.parse import parse_spot
from ._utils.parse import parse_time_to_maturity
from ._utils.parse import parse_volatility


def delta(
    pricer: Callable[..., Tensor], *, create_graph: bool = False, **params: Any
) -> Tensor:
    
    spot = parse_spot(**params).requires_grad_()
    params["spot"] = spot
    if "strike" in params:
        params["moneyness"] = spot / params["strike"]
        params["log_moneyness"] = (spot / params["strike"]).log()

    # Delete parameters that are not in the signature of pricer to avoid
    # TypeError: <pricer> got an unexpected keyword argument '<parameter>'
    for parameter in list(params.keys()):
        if parameter not in signature(pricer).parameters.keys():
            del params[parameter]

    price = pricer(**params)
    return torch.autograd.grad(
        price,
        inputs=spot,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]


def gamma(
    pricer: Callable[..., Tensor], *, create_graph: bool = False, **params: Any
) -> Tensor:
    
    spot = parse_spot(**params).requires_grad_()
    params["spot"] = spot
    if "strike" in params:
        params["moneyness"] = spot / params["strike"]
        params["log_moneyness"] = (spot / params["strike"]).log()

    tensor_delta = delta(pricer, create_graph=True, **params).requires_grad_()
    return torch.autograd.grad(
        tensor_delta,
        inputs=spot,
        grad_outputs=torch.ones_like(tensor_delta),
        create_graph=create_graph,
    )[0]


def gamma_from_delta(
    fn: Callable[..., Tensor], *, create_graph: bool = False, **params: Any
) -> Tensor:
    
    return delta(pricer=fn, create_graph=create_graph, **params)


def vega(
    pricer: Callable[..., Tensor], *, create_graph: bool = False, **params: Any
) -> Tensor:
    
    volatility = parse_volatility(**params).requires_grad_()
    params["volatility"] = volatility
    params["variance"] = volatility.square()

    # Delete parameters that are not in the signature of pricer to avoid
    # TypeError: <pricer> got an unexpected keyword argument '<parameter>'
    for parameter in list(params.keys()):
        if parameter not in signature(pricer).parameters.keys():
            del params[parameter]

    price = pricer(**params)
    return torch.autograd.grad(
        price,
        inputs=volatility,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]


def theta(
    pricer: Callable[..., Tensor], *, create_graph: bool = False, **params: Any
) -> Tensor:
    
    time_to_maturity = parse_time_to_maturity(**params).requires_grad_()
    params["time_to_maturity"] = time_to_maturity

    # Delete parameters that are not in the signature of pricer to avoid
    # TypeError: <pricer> got an unexpected keyword argument '<parameter>'
    for parameter in list(params.keys()):
        if parameter not in signature(pricer).parameters.keys():
            del params[parameter]

    price = pricer(**params)
    # Note: usually theta is calculated reversely (\partial{S}/\partial{T} = \partial{S}/\partial{-time_to_maturity})
    return -torch.autograd.grad(
        price,
        inputs=time_to_maturity,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
    )[0]
