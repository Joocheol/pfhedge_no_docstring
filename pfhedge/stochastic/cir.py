from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar

from ._utils import cast_state


def _get_epsilon(dtype: Optional[torch.dtype]) -> float:
    return torch.finfo(dtype).tiny if dtype else torch.finfo().tiny


def generate_cir(
    n_paths: int,
    n_steps: int,
    init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    kappa: TensorOrScalar = 1.0,
    theta: TensorOrScalar = 0.04,
    sigma: TensorOrScalar = 0.2,
    dt: TensorOrScalar = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    
    if init_state is None:
        init_state = (theta,)

    init_state = cast_state(init_state, dtype=dtype, device=device)

    # PSI_CRIT in [1.0, 2.0]. See section 3.2.3
    PSI_CRIT = 1.5
    # Prevent zero division
    EPSILON = _get_epsilon(dtype)

    output = torch.empty(*(n_paths, n_steps), dtype=dtype, device=device)  # type: ignore
    output[:, 0] = init_state[0]

    randn = torch.randn_like(output)
    rand = torch.rand_like(output)

    # Cast to Tensor with desired dtype and device
    kappa, theta, sigma, dt = map(torch.as_tensor, (kappa, theta, sigma, dt))
    kappa, theta, sigma, dt = map(lambda t: t.to(output), (kappa, theta, sigma, dt))

    for i_step in range(n_steps - 1):
        v = output[:, i_step]

        # Compute m, s, psi: Eq(17,18)
        exp = (-kappa * dt).exp()
        m = theta + (v - theta) * exp
        s2 = v * (sigma ** 2) * exp * (1 - exp) / kappa + theta * (sigma ** 2) * (
            (1 - exp).square()
        ) / (2 * kappa)
        psi = s2 / m.square().clamp(min=EPSILON)

        # Compute V(t + dt) where psi <= PSI_CRIT: Eq(23, 27, 28)
        b = ((2 / psi) - 1 + (2 / psi).sqrt() * (2 / psi - 1).sqrt()).sqrt()
        a = m / (1 + b.square())
        next_0 = a * (b + randn[:, i_step]).square()

        # Compute V(t + dt) where psi > PSI_CRIT: Eq(25)
        u = rand[:, i_step]
        p = (psi - 1) / (psi + 1)
        beta = (1 - p) / m.clamp(min=EPSILON)
        pinv = ((1 - p) / (1 - u).clamp(min=EPSILON)).log() / beta
        next_1 = torch.where(u > p, pinv, torch.zeros_like(u))

        output[:, i_step + 1] = torch.where(psi <= PSI_CRIT, next_0, next_1)

    return output
