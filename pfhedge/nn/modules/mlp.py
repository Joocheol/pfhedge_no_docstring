from copy import deepcopy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential


class MultiLayerPerceptron(Sequential):

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: int = 1,
        n_layers: int = 4,
        n_units: Union[int, Sequence[int]] = 32,
        activation: Module = ReLU(),
        out_activation: Module = Identity(),
    ):
        n_units = (n_units,) * n_layers if isinstance(n_units, int) else n_units

        layers: List[Module] = []
        for i in range(n_layers):
            if i == 0:
                if in_features is None:
                    layers.append(LazyLinear(n_units[0]))
                else:
                    layers.append(Linear(in_features, n_units[0]))
            else:
                layers.append(Linear(n_units[i - 1], n_units[i]))
            layers.append(deepcopy(activation))
        layers.append(Linear(n_units[-1], out_features))
        layers.append(deepcopy(out_activation))

        super().__init__(*layers)
