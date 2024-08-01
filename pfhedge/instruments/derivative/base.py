from abc import abstractmethod
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import TypeVar

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _addindent
from pfhedge._utils.typing import TensorOrScalar

from ..base import BaseInstrument
from ..primary.base import BasePrimary

T = TypeVar("T", bound="BaseDerivative")
Clause = Callable[[T, Tensor], Tensor]


class BaseDerivative(BaseInstrument):
    

    underlier: BasePrimary
    cost: float
    maturity: float
    pricer: Optional[Callable[[Any], Tensor]]
    _clauses: Dict[str, Clause]
    _underliers: Dict[str, BasePrimary]

    def __init__(self) -> None:
        super().__init__()
        self.pricer = None
        self.cost = 0.0
        self._clauses = OrderedDict()
        self._underliers = OrderedDict()

    @property
    def dtype(self) -> Optional[torch.dtype]:
        if len(list(self.underliers())) == 1:
            return self.ul(0).dtype
        else:
            raise AttributeError(
                "dtype is not well-defined for a derivative with multiple underliers"
            )

    @property
    def device(self) -> Optional[torch.device]:
        if len(list(self.underliers())) == 1:
            return self.ul(0).device
        else:
            raise AttributeError(
                "device is not well-defined for a derivative with multiple underliers"
            )

    def simulate(
        self, n_paths: int = 1, init_state: Optional[Tuple[TensorOrScalar, ...]] = None
    ) -> None:
        
        for underlier in self.underliers():
            underlier.simulate(
                n_paths=n_paths, time_horizon=self.maturity, init_state=init_state
            )

    def ul(self, index: int = 0) -> BasePrimary:
        
        return list(self.underliers())[index]

    def to(self: T, *args: Any, **kwargs: Any) -> T:
        for underlier in self.underliers():
            underlier.to(*args, **kwargs)
        return self

    @abstractmethod
    def payoff_fn(self) -> Tensor:
        pass
        

    def payoff(self) -> Tensor:
        
        payoff = self.payoff_fn()
        for clause in self.clauses():
            payoff = clause(self, payoff)
        return payoff

    def list(self: T, pricer: Callable[[T], Tensor], cost: float = 0.0) -> None:
        
        self.pricer = pricer
        self.cost = cost

    def delist(self: T) -> None:
        
        self.pricer = None
        self.cost = 0.0

    @property
    def is_listed(self) -> bool:
        return self.pricer is not None

    def add_clause(self, name: str, clause: Clause) -> None:
        
        if not isinstance(name, (str, bytes)):
            raise TypeError(
                f"clause name should be a string. Got {torch.typename(name)}"
            )
        elif hasattr(self, name) and name not in self._clauses:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise KeyError(f'clause name cannot contain ".", got: {name}')
        elif name == "":
            raise KeyError('clause name cannot be empty string ""')

        if not hasattr(self, "_clauses"):
            raise AttributeError(
                "cannot assign clause before BaseDerivative.__init__() call"
            )

        self._clauses[name] = clause

    def named_clauses(self) -> Iterator[Tuple[str, Clause]]:
        if hasattr(self, "_clauses"):
            for name, clause in self._clauses.items():
                yield name, clause

    def clauses(self) -> Iterator[Clause]:
        for _, clause in self.named_clauses():
            yield clause

    def register_underlier(self, name: str, underlier: BasePrimary) -> None:
        if not isinstance(name, (str, bytes)):
            raise TypeError(f"name should be a string. Got {torch.typename(name)}")
        elif hasattr(self, name) and name not in self._underliers:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise KeyError(f'name cannot contain ".", got: {name}')
        elif name == "":
            raise KeyError('name cannot be empty string ""')

        if not hasattr(self, "_underliers"):
            raise AttributeError(
                "cannot assign underlier before BaseDerivative.__init__() call"
            )

        self._underliers[name] = underlier

    def named_underliers(self) -> Iterator[Tuple[str, BasePrimary]]:
        if hasattr(self, "_underliers"):
            for name, underlier in self._underliers.items():
                yield name, underlier

    def underliers(self) -> Iterator[BasePrimary]:
        for _, underlier in self.named_underliers():
            yield underlier

    def get_underlier(self, name: str) -> BasePrimary:
        if "_underliers" in self.__dict__:
            if name in self._underliers:
                return self._underliers[name]
        raise AttributeError(self._get_name() + " has no attribute " + name)

    def __getattr__(self, name: str) -> BasePrimary:
        return self.get_underlier(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, BasePrimary):
            self.register_underlier(name, value)
        super().__setattr__(name, value)

    @property
    def spot(self) -> Tensor:
        
        if self.pricer is None:
            raise ValueError("self is not listed.")
        return self.pricer(self)

    def __repr__(self) -> str:
        params_str = ""
        if self.extra_repr() != "":
            params_str += self.extra_repr() + "\n"
        if self._clauses:
            params_str += "clauses=" + repr(list(self._clauses.keys())) + "\n"
        params_str += "(underlier): " + repr(self.ul())
        if params_str != "":
            params_str = "\n" + _addindent(params_str) + "\n"
        return self._get_name() + "(" + params_str + ")"


class Derivative(BaseDerivative):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        raise DeprecationWarning(
            "Derivative is deprecated. Use BaseDerivative instead."
        )


class OptionMixin:
    

    underlier: BasePrimary
    strike: float
    maturity: float

    def moneyness(self, time_step: Optional[int] = None, log: bool = False) -> Tensor:
        
        index = ... if time_step is None else [time_step]
        output = self.underlier.spot[..., index] / self.strike
        if log:
            output = output.log()
        return output

    def log_moneyness(self, time_step: Optional[int] = None) -> Tensor:
        return self.moneyness(time_step=time_step, log=True)

    def time_to_maturity(self, time_step: Optional[int] = None) -> Tensor:
        
        n_paths, n_steps = self.underlier.spot.size()
        if time_step is None:
            # Time passed from the beginning
            t = torch.arange(n_steps).to(self.underlier.spot) * self.underlier.dt
            return (t[-1] - t).unsqueeze(0).expand(n_paths, -1)
        else:
            time = n_steps - (time_step % n_steps) - 1
            t = torch.tensor([[time]]).to(self.underlier.spot) * self.underlier.dt
            return t.expand(n_paths, -1)

    def max_moneyness(
        self, time_step: Optional[int] = None, log: bool = False
    ) -> Tensor:
        
        moneyness = self.moneyness(None, log=log)
        if time_step is None:
            return moneyness.cummax(dim=-1).values
        else:
            return moneyness[..., : time_step + 1].max(dim=-1, keepdim=True).values

    def max_log_moneyness(self, time_step: Optional[int] = None) -> Tensor:
        
        return self.max_moneyness(time_step, log=True)


class BaseOption(BaseDerivative, OptionMixin):
    

    def __init__(self):
        super().__init__()
        raise DeprecationWarning(
            "BaseOption is deprecated. Inherit `BaseDerivative` and `OptionMixin` instead."
        )


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(BaseDerivative, "to", BaseInstrument.to)
_set_attr_and_docstring(BaseDerivative, "cpu", BaseInstrument.cpu)
_set_attr_and_docstring(BaseDerivative, "cuda", BaseInstrument.cuda)
_set_attr_and_docstring(BaseDerivative, "double", BaseInstrument.double)
_set_attr_and_docstring(BaseDerivative, "float", BaseInstrument.float)
_set_attr_and_docstring(BaseDerivative, "half", BaseInstrument.half)
