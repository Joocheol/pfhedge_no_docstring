from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar

from ._utils import cast_state


def generate_vasicek(
    n_paths: int,
    n_steps: int,
    init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    kappa: TensorOrScalar = 1.0,
    theta: TensorOrScalar = 0.04,
    sigma: TensorOrScalar = 0.04,
    dt: TensorOrScalar = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:

    if init_state is None:
        init_state = (theta,)

    init_state = cast_state(init_state, dtype, device)

    if init_state[0] != 0:
        new_init_state = (init_state[0] - theta,)
        return theta + generate_vasicek(
            n_paths=n_paths,
            n_steps=n_steps,
            init_state=new_init_state,
            kappa=kappa,
            theta=0.0,
            sigma=sigma,
            dt=dt,
            dtype=dtype,
            device=device,
        )

    output = torch.empty(*(n_paths, n_steps), dtype=dtype, device=device)  # type: ignore
    output[:, 0] = init_state[0]

    # Cast to Tensor with desired dtype and device
    kappa, theta, sigma, dt = map(torch.as_tensor, (kappa, theta, sigma, dt))
    kappa, theta, sigma, dt = map(lambda t: t.to(output), (kappa, theta, sigma, dt))

    randn = torch.randn_like(output)

    # Compute \mu: Equation (3.3)
    mu = (-kappa * dt).exp()
    for i_step in range(n_steps - 1):
        # Compute \sigma_X: Equation (3.4)
        vola = sigma * ((1 - mu.square()) / 2 / kappa).sqrt()
        output[:, i_step + 1] = mu * output[:, i_step] + vola * randn[:, i_step]

    return output
