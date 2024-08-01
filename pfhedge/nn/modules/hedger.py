from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.optim import Optimizer

# error: Skipping analyzing "tqdm": found module but no type hints or library stubs
from tqdm import tqdm  # type: ignore

from pfhedge._utils.hook import save_prev_output
from pfhedge._utils.lazy import has_lazy
from pfhedge._utils.operations import ensemble_mean
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.features import FeatureList
from pfhedge.features._base import Feature
from pfhedge.instruments.base import BaseInstrument
from pfhedge.instruments.derivative.base import BaseDerivative
from pfhedge.nn.functional import pl

from .loss import EntropicRiskMeasure
from .loss import HedgeLoss


class Hedger(Module):
    
    inputs: FeatureList

    def __init__(
        self,
        model: Module,
        inputs: List[Union[str, Feature]],
        criterion: HedgeLoss = EntropicRiskMeasure(),
    ) -> None:
        super().__init__()

        self.model = model
        self.inputs = FeatureList(inputs)
        self.criterion = criterion

        self.register_forward_hook(save_prev_output)

    def forward(self, input: Tensor) -> Tensor:      
        return self.model(input)

    def extra_repr(self) -> str:
        return "inputs=" + str(self.inputs)

    def get_input(self, derivative: BaseDerivative, time_step: Optional[int]) -> Tensor:     
        return self.inputs.of(derivative=derivative).get(time_step)

    def _get_hedge(
        self, derivative: BaseDerivative, hedge: Optional[List[BaseInstrument]]
    ) -> List[BaseInstrument]:
        if hedge is None:
            hedge = list(derivative.underliers())
        return hedge

    def compute_hedge(
        self, derivative: BaseDerivative, hedge: Optional[List[BaseInstrument]] = None
    ) -> Tensor:
        
        inputs = self.inputs.of(derivative, self)
        hedge = self._get_hedge(derivative, hedge)

        # Check that the spot prices of the hedges have the same sizes
        if not all(h.spot.size() == hedge[0].spot.size() for h in hedge):
            raise ValueError("The spot prices of the hedges must have the same size")

        (n_paths, n_steps), n_hedges = hedge[0].spot.size(), len(hedge)
        if inputs.is_state_dependent():
            zeros = hedge[0].spot.new_zeros((n_paths, 1, n_hedges))
            save_prev_output(self, input=(), output=zeros)
            outputs = []
            for time_step in range(n_steps - 1):
                input = inputs.get(time_step)  # (N, T=1, F)
                outputs.append(self(input))  # (N, T=1, H)
            outputs.append(outputs[-1])
            output = torch.cat(outputs, dim=-2)  # (N, T, H)
        else:
            # If all features are state-independent, compute the output at all
            # time steps at once, which would be faster.
            input = inputs.get(None)  # (N, T, F)
            output = self(input)  # (N, T, H)
            # This maintains consistency with the previous implementations.
            # In previous implementation for loop is computed for 0...T-2 and
            # the last time step is not included.
            output[..., -1, :] = output[..., -2, :]

        output = output.transpose(-1, -2)  # (N, H, T)

        return output

    def compute_portfolio(
        self, derivative: BaseDerivative, hedge: Optional[List[BaseInstrument]] = None
    ) -> Tensor:
    
        hedge = self._get_hedge(derivative, hedge)

        spot = torch.stack([h.spot for h in hedge], dim=1)
        unit = self.compute_hedge(derivative, hedge=hedge)
        cost = [h.cost for h in hedge]

        return pl(spot=spot, unit=unit, cost=cost)

    def compute_pl(
        self, derivative: BaseDerivative, hedge: Optional[List[BaseInstrument]] = None
    ) -> Tensor:
        
        hedge = self._get_hedge(derivative, hedge)

        spot = torch.stack([h.spot for h in hedge], dim=1)
        unit = self.compute_hedge(derivative, hedge=hedge)
        cost = [h.cost for h in hedge]

        return pl(spot=spot, unit=unit, cost=cost, payoff=derivative.payoff())

    def compute_pnl(
        self,
        derivative: BaseDerivative,
        hedge: Optional[List[BaseInstrument]] = None,
        n_paths: int = 1000,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> Tensor:
        
        # TODO(simaki): Raise DeprecationWarning later
        derivative.simulate(n_paths=n_paths, init_state=init_state)
        return self.compute_pl(derivative=derivative, hedge=hedge)

    def compute_loss(
        self,
        derivative: BaseDerivative,
        hedge: Optional[List[BaseInstrument]] = None,
        n_paths: int = 1000,
        n_times: int = 1,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
        enable_grad: bool = True,
    ) -> Tensor:
        
        with torch.set_grad_enabled(enable_grad):

            def _get_loss():
                derivative.simulate(n_paths=n_paths, init_state=init_state)
                portfolio = self.compute_portfolio(derivative, hedge=hedge)
                return self.criterion(portfolio, derivative.payoff())

            mean_loss = ensemble_mean(_get_loss, n_times=n_times)

        return mean_loss

    def _configure_optimizer(
        self,
        derivative: BaseDerivative,
        optimizer: Union[Optimizer, Callable[..., Optimizer]],
    ) -> Optimizer:
        if not isinstance(optimizer, Optimizer):
            if has_lazy(self):
                # Run a placeholder forward to initialize lazy parameters
                derivative.simulate(n_paths=1)
                _ = self.compute_pl(derivative)
            # If we use `if issubclass(optimizer, Optimizer)` here, mypy thinks that
            # optimizer is Optimizer rather than its subclass (e.g. Adam)
            # and complains that the required parameter default is missing.
            if Optimizer in getattr(optimizer, "__mro__", []):
                optimizer = optimizer(self.model.parameters())
            else:
                raise TypeError("optimizer is not an Optimizer type")
        return optimizer

    def fit(
        self,
        derivative: BaseDerivative,
        hedge: Optional[List[BaseInstrument]] = None,
        n_epochs: int = 100,
        n_paths: int = 1000,
        n_times: int = 1,
        optimizer: Union[Optimizer, Callable[..., Optimizer]] = Adam,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
        verbose: bool = True,
        validation: bool = True,
        tqdm_kwargs: dict = {},
    ) -> Optional[List[float]]:
        
        optimizer = self._configure_optimizer(derivative, optimizer)

        def compute_loss(**kwargs: Any) -> Tensor:
            return self.compute_loss(
                derivative,
                hedge=hedge,
                n_paths=n_paths,
                init_state=init_state,
                **kwargs,
            )

        history = []
        progress = tqdm(range(n_epochs), disable=not verbose, **tqdm_kwargs)
        for _ in progress:
            # Compute training loss and backpropagate
            self.train()
            optimizer.zero_grad()
            loss = compute_loss()
            loss.backward()
            optimizer.step()

            # Compute validation loss
            if validation:
                self.eval()
                loss = compute_loss(n_times=n_times, enable_grad=False)
                history.append(loss.item())

                progress.desc = "Loss=" + _format_float(float(loss.item()))

        return history if validation else None

    def price(
        self,
        derivative: BaseDerivative,
        hedge: Optional[List[BaseInstrument]] = None,
        n_paths: int = 1000,
        n_times: int = 1,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
        enable_grad: bool = False,
    ) -> Tensor:
        
        with torch.set_grad_enabled(enable_grad):

            def _get_price():
                derivative.simulate(n_paths=n_paths, init_state=init_state)
                portfolio = self.compute_portfolio(derivative, hedge)
                # Negative because selling
                return -self.criterion.cash(portfolio, target=derivative.payoff())

            mean_price = ensemble_mean(_get_price, n_times=n_times)

        return mean_price
