import torch


def _addindent(string: str, n_spaces: int = 2) -> str:
    return "\n".join(" " * n_spaces + line for line in string.split("\n"))


def _format_float(value: float) -> str:
    
    # format a float following PyTorch printoptions
    # see `torch.set_printoptions` for details
    tensor = torch.tensor([value])
    return torch._tensor_str._Formatter(tensor).format(value)
