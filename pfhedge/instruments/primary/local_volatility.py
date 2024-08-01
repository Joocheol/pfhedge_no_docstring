from math import ceil
from typing import Optional
from typing import Tuple
from typing import cast

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import LocalVolatilityFunction
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_local_volatility_process

from .base import BasePrimary


class LocalVolatilityStock(BasePrimary):

    spot: Tensor
    volatility: Tensor

    def __init__(
        self,
        sigma_fn: LocalVolatilityFunction,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.sigma_fn = sigma_fn
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0,)

    @property
    def variance(self) -> Tensor:
        
        return self.volatility.square()

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar]] = None,
    ) -> None:
        
        if init_state is None:
            init_state = cast(Tuple[float], self.default_init_state)

        output = generate_local_volatility_process(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            sigma_fn=self.sigma_fn,
            init_state=init_state,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", output.spot)
        self.register_buffer("volatility", output.volatility)

    def extra_repr(self) -> str:
        params = []
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(
    LocalVolatilityStock, "default_init_state", BasePrimary.default_init_state
)
_set_attr_and_docstring(LocalVolatilityStock, "to", BasePrimary.to)
