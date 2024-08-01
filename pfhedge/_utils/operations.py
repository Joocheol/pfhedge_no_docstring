from typing import Any
from typing import Callable

import torch
from torch import Tensor


def ensemble_mean(
    function: Callable[..., Tensor], n_times: int = 1, *args: Any, **kwargs: Any
) -> Tensor:
    
    if n_times == 1:
        return function(*args, **kwargs)
    else:
        stack = torch.stack([function(*args, **kwargs) for _ in range(n_times)])
        return stack.mean(dim=0)
