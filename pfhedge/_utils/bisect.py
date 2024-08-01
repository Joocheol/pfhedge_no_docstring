from typing import Any
from typing import Callable
from typing import Union

import torch
from torch import Tensor


def bisect(
    fn: Callable[[Tensor], Tensor],
    target: Tensor,
    lower: Union[float, Tensor],
    upper: Union[float, Tensor],
    precision: float = 1e-6,
    max_iter: int = 100000,
) -> Tensor:
    
    lower, upper = map(torch.as_tensor, (lower, upper))

    if not (lower < upper).all():
        raise ValueError("condition lower < upper should be satisfied.")

    if (fn(lower) > fn(upper)).all():
        # If fn is a decreasing function
        def mf(inputs: Tensor) -> Tensor:
            return -fn(inputs)

        return bisect(mf, -target, lower, upper, precision=precision, max_iter=max_iter)

    n_iter = 0
    while torch.max(upper - lower) > precision:
        n_iter += 1
        if n_iter > max_iter:
            raise RuntimeError(f"Aborting since iteration exceeds max_iter={max_iter}.")

        m = (lower + upper) / 2
        output = fn(m)
        lower = lower.where(output >= target, m)
        upper = upper.where(output < target, m)

    return upper


def find_implied_volatility(
    pricer: Callable,
    price: Tensor,
    lower: float = 0.001,
    upper: float = 1.000,
    precision: float = 1e-6,
    max_iter: int = 100,
    **params: Any,
) -> Tensor:
    

    def fn(volatility: Tensor) -> Tensor:
        return pricer(volatility=volatility, **params)

    return bisect(
        fn,
        price,
        torch.as_tensor(lower).to(price),
        torch.as_tensor(upper).to(price),
        precision=precision,
        max_iter=max_iter,
    )
