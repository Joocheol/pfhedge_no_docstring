from torch import Tensor
from torch.nn import Module

from pfhedge._utils.typing import TensorOrScalar
from pfhedge.nn.functional import svi_variance


class SVIVariance(Module):

    def __init__(
        self,
        a: TensorOrScalar,
        b: TensorOrScalar,
        rho: TensorOrScalar,
        m: TensorOrScalar,
        sigma: TensorOrScalar,
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma

    def forward(self, input: Tensor) -> Tensor:
        return svi_variance(
            input, a=self.a, b=self.b, rho=self.rho, m=self.m, sigma=self.sigma
        )

    def extra_repr(self) -> str:
        params = (
            f"a={self.a}",
            f"b={self.b}",
            f"rho={self.rho}",
            f"m={self.m}",
            f"sigma={self.sigma}",
        )
        return ", ".join(params)
