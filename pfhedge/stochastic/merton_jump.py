import math
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic._utils import cast_state


def generate_merton_jump(
    n_paths: int,
    n_steps: int,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (1.0,),
    mu: float = 0.0,
    sigma: float = 0.2,
    jump_per_year: float = 68.2,
    jump_mean: float = 0.0,
    jump_std: float = 0.02,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    engine: Callable[..., Tensor] = torch.randn,
) -> Tensor:
    
    # https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python
    init_state = cast_state(init_state, dtype=dtype, device=device)

    poisson = torch.distributions.poisson.Poisson(rate=jump_per_year * dt)
    n_jumps = poisson.sample((n_paths, n_steps - 1)).to(dtype=dtype, device=device)

    # Eq. (3) in https://www.imes.boj.or.jp/research/papers/japanese/kk22-b1-3.pdf
    jump = (
        jump_mean * n_jumps
        + engine(*(n_paths, n_steps - 1), dtype=dtype, device=device)
        * jump_std
        * n_jumps.sqrt()
    )
    jump = torch.cat(
        [torch.zeros((n_paths, 1), dtype=dtype, device=device), jump], dim=1
    )

    randn = engine(*(n_paths, n_steps), dtype=dtype, device=device)
    randn[:, 0] = 0.0
    drift = (
        (mu - (sigma ** 2) / 2 - jump_per_year * (jump_mean + jump_std ** 2 / 2))
        * dt
        * torch.arange(n_steps).to(randn)
    )
    brown = randn.cumsum(1) * math.sqrt(dt)
    return init_state[0] * (drift + sigma * brown + jump.cumsum(1)).exp()
