from typing import Any
from typing import Callable

import torch
from torch import Tensor
from torch.testing import assert_close


def assert_monotone(
    fn: Callable[[Tensor], Tensor],
    x1: Tensor,
    x2: Tensor,
    increasing: bool = False,
    allow_equal: bool = False,
) -> None:
    
    assert not increasing, "not supported"
    assert not allow_equal, "not supported"
    assert (x1 > x2).all(), "x1 > x2 must be satisfied"

    assert (fn(x1) < fn(x2)).all()


def assert_convex(
    fn: Callable[[Tensor], Tensor], x1: Tensor, x2: Tensor, alpha: float
) -> None:
    
    y = fn(alpha * x1 + (1 - alpha) * x2)
    y1 = fn(x1)
    y2 = fn(x2)
    assert y <= alpha * y1 + (1 - alpha) * y2


def assert_cash_invariant(
    fn: Callable[[Tensor], Tensor], x: Tensor, c: float, **kwargs: Any
) -> None:
    
    assert_close(fn(x + c), fn(x) - c, **kwargs)


def assert_cash_equivalent(
    fn: Callable[[Tensor], Tensor], x: Tensor, c: float, **kwargs: Any
) -> None:
    
    assert_close(fn(x), fn(torch.full_like(x, c)), **kwargs)
