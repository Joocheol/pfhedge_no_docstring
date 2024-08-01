from math import ceil
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_rough_bergomi

from .base import BasePrimary


class RoughBergomiStock(BasePrimary):

    spot: Tensor
    variance: Tensor

    def __init__(
        self,
        alpha: float = -0.4,
        rho: float = -0.9,
        eta: float = 1.9,
        xi: float = 0.04,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.rho = rho
        self.eta = eta
        self.xi = xi
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0, self.xi)

    @property
    def volatility(self) -> Tensor:
        
        return self.get_buffer("variance").clamp(min=0.0).sqrt()

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        
        if init_state is None:
            init_state = self.default_init_state

        output = generate_rough_bergomi(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            init_state=init_state,
            alpha=self.alpha,
            rho=self.rho,
            eta=self.eta,
            xi=self.xi,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", output.spot)
        self.register_buffer("variance", output.variance)

    def extra_repr(self) -> str:
        params = [
            "alpha=" + _format_float(self.alpha),
            "rho=" + _format_float(self.rho),
            "eta=" + _format_float(self.eta),
            "xi=" + _format_float(self.xi),
        ]
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(RoughBergomiStock, "default_init_state", BasePrimary.default_init_state)
_set_attr_and_docstring(RoughBergomiStock, "to", BasePrimary.to)
