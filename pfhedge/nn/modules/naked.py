from torch import Tensor
from torch.nn import Module


class Naked(Module):
    
    def __init__(self, out_features: int = 1):
        super().__init__()
        self.out_features = out_features

    def forward(self, input: Tensor) -> Tensor:
        return input.new_zeros(input.size()[:-1] + (self.out_features,))
