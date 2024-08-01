from abc import abstractmethod
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import no_type_check

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.typing import TensorOrScalar

from ..base import BaseInstrument

T = TypeVar("T", bound="BasePrimary")


class BasePrimary(BaseInstrument):
    

    dt: float
    cost: float
    _buffers: Dict[str, Tensor]
    dtype: Optional[torch.dtype]
    device: Optional[torch.device]

    def __init__(self) -> None:
        super().__init__()
        self._buffers = OrderedDict()

    @property
    def default_init_state(self) -> Tuple[TensorOrScalar, ...]:
        
        return ()

    # TODO(simaki): Remove @no_type_check once BrownianStock and HestonStock get
    #   unified signatures.
    @abstractmethod
    @no_type_check
    def simulate(
        self,
        n_paths: int,
        time_horizon: float,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        pass
        

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        
        # Implementation here refers to torch.nn.Module.register_buffer.
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before __init__() call")
        elif not isinstance(name, (str, bytes)):
            raise TypeError(
                "buffer name should be a string. " "Got {}".format(torch.typename(name))
            )
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "cannot assign '{}' object to buffer '{}' "
                "(torch Tensor or None required)".format(torch.typename(tensor), name)
            )
        else:
            if isinstance(tensor, Tensor):
                tensor = tensor.to(self.device, self.dtype)
            self._buffers[name] = tensor

    def named_buffers(self) -> Iterator[Tuple[str, Tensor]]:
        
        for name, buffer in self._buffers.items():
            if buffer is not None:
                yield name, buffer

    def buffers(self) -> Iterator[Tensor]:
        
        for _, buffer in self.named_buffers():
            yield buffer

    def get_buffer(self, name: str) -> Tensor:
        
        if "_buffers" in self.__dict__:
            if name in self._buffers:
                return self._buffers[name]
        raise AttributeError(self._get_name() + " has no buffer named " + name)

    def __getattr__(self, name: str) -> Tensor:
        return self.get_buffer(name)

    @property
    def spot(self) -> Tensor:
        name = "spot"
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        raise AttributeError(
            f"'{self._get_name()}' object has no attribute '{name}'. "
            "Asset may not be simulated."
        )

    @property
    def is_listed(self) -> bool:
        return True

    def to(self: T, *args: Any, **kwargs: Any) -> T:
        device, dtype, *_ = self._parse_to(*args, **kwargs)

        if dtype is not None and not dtype.is_floating_point:
            raise TypeError(
                f"to() only accepts floating point dtypes, but got desired dtype={dtype}"
            )

        if not hasattr(self, "dtype") or dtype is not None:
            self.dtype = dtype
        if not hasattr(self, "device") or device is not None:
            self.device = device

        for name, buffer in self.named_buffers():
            self.register_buffer(name, buffer.to(device, dtype))

        return self

    @staticmethod
    def _parse_to(
        *args: Any, **kwargs: Any
    ) -> Union[
        Tuple[torch.device, torch.dtype],
        Tuple[torch.device, torch.dtype, bool, torch.memory_format],
    ]:
        # Can be called as:
        #   to(device=None, dtype=None)
        #   to(tensor)
        #   to(instrument)
        # and return a tuple (device, dtype, ...)
        if len(args) > 0 and isinstance(args[0], BaseInstrument):
            instrument = args[0]
            return getattr(instrument, "device"), getattr(instrument, "dtype")
        elif "instrument" in kwargs:
            instrument = kwargs["instrument"]
            return getattr(instrument, "device"), getattr(instrument, "dtype")
        else:
            return torch._C._nn._parse_to(*args, **kwargs)

    def __repr__(self) -> str:
        extra_repr = self.extra_repr()
        dinfo = ", ".join(self._dinfo())
        main_str = self._get_name() + "("
        if extra_repr and dinfo:
            extra_repr += ", "
        main_str += extra_repr + dinfo + ")"
        return main_str


class Primary(BasePrimary):
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        raise DeprecationWarning("Primary is deprecated. Use BasePrimary instead.")


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(BasePrimary, "to", BaseInstrument.to)
_set_attr_and_docstring(BasePrimary, "cpu", BaseInstrument.cpu)
_set_attr_and_docstring(BasePrimary, "cuda", BaseInstrument.cuda)
_set_attr_and_docstring(BasePrimary, "double", BaseInstrument.double)
_set_attr_and_docstring(BasePrimary, "float", BaseInstrument.float)
_set_attr_and_docstring(BasePrimary, "half", BaseInstrument.half)
