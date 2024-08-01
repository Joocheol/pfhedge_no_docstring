from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar


def cast_state(
    state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, ...]:
    
    if isinstance(state, (Tensor, float, int)):
        state_tuple: Tuple[TensorOrScalar, ...] = (state,)
    else:
        state_tuple = state

    # Cast to init_state: Tuple[Tensor, ...] with desired dtype and device
    state_tensor_tuple: Tuple[Tensor, ...] = tuple(map(torch.as_tensor, state_tuple))
    state_tensor_tuple = tuple(map(lambda t: t.to(device, dtype), state_tensor_tuple))

    return state_tensor_tuple
