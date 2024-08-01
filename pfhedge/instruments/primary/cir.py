from math import ceil
from typing import Optional
from typing import Tuple

import torch

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_cir

from .base import BasePrimary


class CIRRate(BasePrimary):

    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 0.04,
        sigma: float = 0.2,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (self.theta,)

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        
        if init_state is None:
            init_state = self.default_init_state

        spot = generate_cir(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            init_state=init_state,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", spot)

    def extra_repr(self) -> str:
        params = [
            "kappa=" + _format_float(self.kappa),
            "theta=" + _format_float(self.theta),
            "sigma=" + _format_float(self.sigma),
        ]
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(CIRRate, "default_init_state", BasePrimary.default_init_state)
_set_attr_and_docstring(CIRRate, "to", BasePrimary.to)
