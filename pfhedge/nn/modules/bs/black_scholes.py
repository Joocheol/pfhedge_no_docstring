from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Type

from torch import Tensor
from torch.nn import Module

from pfhedge.instruments import Derivative


class BlackScholesModuleFactory:

    _modules: Dict[str, Type[Module]]

    # singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls._instance._modules = OrderedDict()
        return cls._instance

    def register_module(self, name: str, cls: Type[Module]) -> None:
        self._modules[name] = cls

    def named_modules(self) -> Iterator[Tuple[str, Type[Module]]]:
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

    def names(self) -> Iterator[str]:
        for name, _ in self.named_modules():
            yield name

    def features(self) -> Iterator[Type[Module]]:
        for _, module in self.named_modules():
            yield module

    def get_class(self, name: str) -> Type[Module]:
        return self._modules[name]

    def get_class_from_derivative(self, derivative: Derivative) -> Type[Module]:
        return self.get_class(derivative.__class__.__name__).from_derivative(derivative)  # type: ignore


class BlackScholes(Module):
    

    inputs: Callable[..., List[str]]  # inputs(self) -> List[str]
    price: Callable[..., Tensor]  # price(self, ...) -> Tensor
    delta: Callable[..., Tensor]  # delta(self, ...) -> Tensor
    gamma: Callable[..., Tensor]  # gamma(self, ...) -> Tensor
    vega: Callable[..., Tensor]  # vega(self, ...) -> Tensor
    theta: Callable[..., Tensor]  # theta(self, ...) -> Tensor

    def __new__(cls, derivative):
        return BlackScholesModuleFactory().get_class_from_derivative(derivative)
