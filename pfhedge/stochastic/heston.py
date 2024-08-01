from collections import namedtuple
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from pfhedge._utils.str import _addindent
from pfhedge._utils.typing import TensorOrScalar

from ._utils import cast_state
from .cir import generate_cir


class SpotVarianceTuple(namedtuple("SpotVarianceTuple", ["spot", "variance"])):

    __module__ = "pfhedge.stochastic"

    def __repr__(self) -> str:
        items_str_list = []
        for field, tensor in self._asdict().items():
            items_str_list.append(field + "=\n" + str(tensor))
        items_str = _addindent("\n".join(items_str_list), 2)
        return self.__class__.__name__ + "(\n" + items_str + "\n)"

    @property
    def volatility(self) -> Tensor:
        return self.variance.clamp(min=0.0).sqrt()


def generate_heston(
    n_paths: int,
    n_steps: int,
    init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    kappa: float = 1.0,
    theta: float = 0.04,
    sigma: float = 0.2,
    rho: float = -0.7,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> SpotVarianceTuple:
    
    if init_state is None:
        init_state = (1.0, theta)

    init_state = cast_state(init_state, dtype=dtype, device=device)

    GAMMA1 = 0.5
    GAMMA2 = 0.5

    variance = generate_cir(
        n_paths=n_paths,
        n_steps=n_steps,
        init_state=init_state[1:],
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dtype=dtype,
        device=device,
    )

    log_spot = torch.empty_like(variance)
    log_spot[:, 0] = init_state[0].log()
    randn = torch.randn_like(variance)

    for i_step in range(n_steps - 1):
        # Compute log S(t + 1): Eq(33)
        k0 = -rho * kappa * theta * dt / sigma
        k1 = GAMMA1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
        k2 = GAMMA2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
        k3 = GAMMA1 * dt * (1 - rho ** 2)
        k4 = GAMMA2 * dt * (1 - rho ** 2)
        v0 = variance[:, i_step]
        v1 = variance[:, i_step + 1]
        log_spot[:, i_step + 1] = (
            log_spot[:, i_step]
            + k0
            + k1 * v0
            + k2 * v1
            + (k3 * v0 + k4 * v1).sqrt() * randn[:, i_step]
        )

    return SpotVarianceTuple(log_spot.exp(), variance)
