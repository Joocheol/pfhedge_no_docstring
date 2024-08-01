from collections import namedtuple
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.str import _addindent
from pfhedge._utils.typing import LocalVolatilityFunction
from pfhedge._utils.typing import TensorOrScalar

from ._utils import cast_state


class LocalVolatilityTuple(namedtuple("LocalVolatilityTuple", ["spot", "volatility"])):

    __module__ = "pfhedge.stochastic"

    def __repr__(self) -> str:
        items_str_list = []
        for field, tensor in self._asdict().items():

            items_str_list.append(field + "=\n" + str(tensor))
        items_str = _addindent("\n".join(items_str_list), 2)
        return self.__class__.__name__ + "(\n" + items_str + "\n)"

    @property
    def variance(self) -> Tensor:
        return self.volatility.square()


def generate_local_volatility_process(
    n_paths: int,
    n_steps: int,
    sigma_fn: LocalVolatilityFunction,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (1.0,),
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> LocalVolatilityTuple:
    
    init_state = cast_state(init_state, dtype=dtype, device=device)

    spot = torch.empty(*(n_paths, n_steps), dtype=dtype, device=device)  # type: ignore
    spot[:, 0] = init_state[0]
    volatility = torch.empty_like(spot)

    time = dt * torch.arange(n_steps).to(spot)
    dw = torch.randn_like(spot) * torch.as_tensor(dt).sqrt()

    for i_step in range(n_steps):
        sigma = sigma_fn(time[i_step], spot[:, i_step])
        volatility[:, i_step] = sigma
        if i_step != n_steps - 1:
            spot[:, i_step + 1] = spot[:, i_step] * (1 + sigma * dw[:, i_step])

    return LocalVolatilityTuple(spot, volatility)
