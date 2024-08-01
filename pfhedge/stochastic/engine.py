from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from pfhedge.nn.functional import box_muller


def _get_numel(size: Tuple[int, ...]) -> int:
    out = 1
    for dim in size:
        out *= dim
    return out


class RandnSobolBoxMuller:
    

    def __init__(self, scramble: bool = False, seed: Optional[int] = None):
        self.scramble = scramble
        self.seed = seed

    def __call__(
        self,
        *size: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        numel = _get_numel(tuple(size))
        output = self._generate_1d(numel, dtype=dtype, device=device)
        output.resize_(*size)
        return output

    def _generate_1d(
        self,
        n: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        engine = SobolEngine(2, scramble=self.scramble, seed=self.seed)
        rand = engine.draw(n // 2 + 1).to(dtype=dtype, device=device)
        z0, z1 = box_muller(rand[:, 0], rand[:, 1])
        return torch.cat((z0, z1), dim=0)[:n]
