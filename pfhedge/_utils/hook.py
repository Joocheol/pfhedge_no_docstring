from typing import Any
from typing import Optional
from typing import Tuple

from torch import Tensor
from torch.nn import Module


def save_prev_output(
    module: Module, input: Tuple[Any, ...], output: Optional[Tensor]
) -> None:
    
    module.register_buffer("prev_output", output, persistent=False)
