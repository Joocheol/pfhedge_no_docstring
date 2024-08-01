from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Optional
from typing import TypeVar
from typing import no_type_check

import torch
from torch import Tensor

T = TypeVar("T", bound="BaseInstrument")


class BaseInstrument(ABC):
    

    cost: float

    @property
    @abstractmethod
    def spot(self) -> Tensor:
        pass
        
    @property
    @abstractmethod
    def is_listed(self) -> bool:
        pass
        
    @abstractmethod
    @no_type_check
    def simulate(self, n_paths: int, time_horizon: float, **kwargs) -> None:
        pass
        

    @abstractmethod
    def to(self: T, *args: Any, **kwargs: Any) -> T:
        pass
        

    def cpu(self: T) -> T:
        
        return self.to(torch.device("cpu"))

    def cuda(self: T, device: Optional[int] = None) -> T:
        
        return self.to(torch.device(f"cuda:{device}" if device is not None else "cuda"))

    def double(self: T) -> T:
        
        return self.to(torch.float64)

    def float(self: T) -> T:
        
        return self.to(torch.float32)

    def half(self: T) -> T:
        
        return self.to(torch.float16)

    def float64(self: T) -> T:
        
        return self.double()

    def float32(self: T) -> T:
        
        return self.float()

    def float16(self: T) -> T:
        
        return self.half()

    def bfloat16(self: T) -> T:
        
        return self.to(torch.bfloat16)

    def extra_repr(self) -> str:
        
        return ""

    def _get_name(self) -> str:
        return self.__class__.__name__

    def _dinfo(self) -> List[str]:
        # Returns list of strings that tell ``dtype`` and ``device`` of self.
        # Intended to be used in :func:`__repr__`.
        # If ``dtype`` (``device``) is the one specified in default type,
        # ``dinfo`` will not have the information of it.
        # Implementation here refers to the function _str_intern in
        # pytorch/_tensor_str.py.
        dinfo = []

        dtype = getattr(self, "dtype", None)
        if dtype is not None:
            if dtype != torch.get_default_dtype():
                dinfo.append("dtype=" + str(dtype))

        # A general logic here is we only print device when it doesn't match
        # the device specified in default tensor type.
        device = getattr(self, "device", None)
        if device is not None:
            if device.type != torch._C._get_default_device() or (
                device.type == "cuda" and torch.cuda.current_device() != device.index
            ):
                dinfo.append("device='" + str(device) + "'")

        return dinfo


class Instrument(BaseInstrument):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)  # type: ignore
        raise DeprecationWarning(
            "Instrument is deprecated. Use BaseInstrument instead."
        )
