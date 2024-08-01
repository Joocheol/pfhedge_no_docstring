from math import ceil
from typing import Callable
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_merton_jump

from .base import BasePrimary


class MertonJumpStock(BasePrimary):

    spot: Tensor

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.2,
        jump_per_year: float = 68,
        jump_mean: float = 0.0,
        jump_std: float = 0.01,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        engine: Callable[..., Tensor] = torch.randn,
    ) -> None:
        super().__init__()

        self.mu = mu
        self.sigma = sigma
        self.jump_per_year = jump_per_year
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.cost = cost
        self.dt = dt
        self.engine = engine

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
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        
        if init_state is None:
            init_state = self.default_init_state

        output = generate_merton_jump(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            init_state=init_state,
            sigma=self.sigma,
            mu=self.mu,
            jump_per_year=self.jump_per_year,
            jump_mean=self.jump_mean,
            jump_std=self.jump_std,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
            engine=self.engine,
        )

        self.register_buffer("spot", output)

    def extra_repr(self) -> str:
        params = [
            "mu=" + _format_float(self.mu),
            "sigma=" + _format_float(self.sigma),
            "jump_per_year=" + _format_float(self.jump_per_year),
            "jump_mean=" + _format_float(self.jump_mean),
            "jump_std=" + _format_float(self.jump_std),
        ]
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(MertonJumpStock, "default_init_state", BasePrimary.default_init_state)
_set_attr_and_docstring(MertonJumpStock, "to", BasePrimary.to)
