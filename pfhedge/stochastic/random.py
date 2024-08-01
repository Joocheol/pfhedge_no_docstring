from typing import Optional

import torch
from torch import Tensor

from .engine import RandnSobolBoxMuller


def randn_antithetic(
    *size: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    dim: int = 0,
    shuffle: bool = True,
) -> Tensor:
    
    if dim != 0:
        raise ValueError("dim != 0 is not supported.")

    size_list = list(size)
    size_half = [-(-size_list[0] // 2)] + size_list[1:]
    randn = torch.randn(*size_half, dtype=dtype, device=device)  # type: ignore

    output = torch.cat((randn, -randn), dim=0)

    if shuffle:
        output = output[torch.randperm(output.size(dim))]

    output = output[: size_list[0]]

    return output


def randn_sobol_boxmuller(
    *size: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    scramble: bool = True,
    seed: Optional[int] = None,
) -> Tensor:
    
    engine = RandnSobolBoxMuller(scramble=scramble, seed=seed)
    return engine(*size, dtype=dtype, device=device)
