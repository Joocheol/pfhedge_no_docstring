from math import ceil
from typing import Optional
from typing import Tuple
from typing import cast

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_geometric_brownian

from .base import BasePrimary


class BrownianStock(BasePrimary):

    def __init__(
        self,
        sigma: float = 0.2,
        mu: float = 0.0,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.sigma = sigma
        self.mu = mu
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0,)

    @property
    def volatility(self) -> Tensor:
        
        return torch.full_like(self.get_buffer("spot"), self.sigma)

    @property
    def variance(self) -> Tensor:
        
        return torch.full_like(self.get_buffer("spot"), self.sigma ** 2)

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar]] = None,
    ) -> None:
        
        if init_state is None:
            init_state = cast(Tuple[float], self.default_init_state)

        spot = generate_geometric_brownian(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            init_state=init_state,
            sigma=self.sigma,
            mu=self.mu,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", spot)

    def extra_repr(self) -> str:
        params = ["sigma=" + _format_float(self.sigma)]
        if self.mu != 0.0:
            params.append("mu=" + _format_float(self.mu))
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(BrownianStock, "default_init_state", BasePrimary.default_init_state)
_set_attr_and_docstring(BrownianStock, "to", BasePrimary.to)
