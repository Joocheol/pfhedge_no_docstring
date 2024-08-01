from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar

from ._utils import cast_state


def generate_brownian(
    n_paths: int,
    n_steps: int,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (0.0,),
    sigma: float = 0.2,
    mu: float = 0.0,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    engine: Callable[..., Tensor] = torch.randn,
) -> Tensor:
    
    init_state = cast_state(init_state, dtype=dtype, device=device)

    init_value = init_state[0]
    # randn = torch.randn((n_paths, n_steps), dtype=dtype, device=device)
    randn = engine(*(n_paths, n_steps), dtype=dtype, device=device)
    randn[:, 0] = 0.0
    drift = mu * dt * torch.arange(n_steps).to(randn)
    brown = randn.new_tensor(dt).sqrt() * randn.cumsum(1)
    return drift + sigma * brown + init_value


def generate_geometric_brownian(
    n_paths: int,
    n_steps: int,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (1.0,),
    sigma: float = 0.2,
    mu: float = 0.0,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    engine: Callable[..., Tensor] = torch.randn,
) -> Tensor:
    
    init_state = cast_state(init_state, dtype=dtype, device=device)

    brownian = generate_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        init_state=(0.0,),
        sigma=sigma,
        mu=mu,
        dt=dt,
        dtype=dtype,
        device=device,
        engine=engine,
    )
    t = dt * torch.arange(n_steps).to(brownian).unsqueeze(0)
    return init_state[0] * (brownian - (sigma ** 2) * t / 2).exp()
