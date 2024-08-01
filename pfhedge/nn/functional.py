import math
from math import ceil
from math import pi as kPI
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as fn
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions.utils import broadcast_all

from pfhedge import autogreek
from pfhedge._utils.bisect import bisect
from pfhedge._utils.typing import TensorOrScalar


def european_payoff(input: Tensor, call: bool = True, strike: float = 1.0) -> Tensor:
    
    if call:
        return fn.relu(input[..., -1] - strike)
    else:
        return fn.relu(strike - input[..., -1])


def lookback_payoff(input: Tensor, call: bool = True, strike: float = 1.0) -> Tensor:
    
    if call:
        return fn.relu(input.max(dim=-1).values - strike)
    else:
        return fn.relu(strike - input.min(dim=-1).values)


def american_binary_payoff(
    input: Tensor, call: bool = True, strike: float = 1.0
) -> Tensor:
    
    if call:
        return (input.max(dim=-1).values >= strike).to(input)
    else:
        return (input.min(dim=-1).values <= strike).to(input)


def european_binary_payoff(
    input: Tensor, call: bool = True, strike: float = 1.0
) -> Tensor:
    
    if call:
        return (input[..., -1] >= strike).to(input)
    else:
        return (input[..., -1] <= strike).to(input)


def european_forward_start_payoff(
    input: Tensor, strike: float = 1.0, start_index: int = 0, end_index: int = -1
) -> Tensor:
    
    return fn.relu(input[..., end_index] / input[..., start_index] - strike)


def exp_utility(input: Tensor, a: float = 1.0) -> Tensor:

    return -(-a * input).exp()


def isoelastic_utility(input: Tensor, a: float) -> Tensor:

    if a == 1.0:
        return input.log()
    else:
        return input.pow(1.0 - a)


def entropic_risk_measure(input: Tensor, a: float = 1.0) -> Tensor:
    
    return (torch.logsumexp(-input * a, dim=0) - math.log(input.size(0))) / a


def topp(
    input: Tensor, p: float, dim: Optional[int] = None, largest: bool = True
) -> "torch.return_types.return_types.topk":  # type: ignore
    # ToDo(masanorihirano): in torch 1.9.0 or some versions (before 1.13.0), this type and alternatives do not exist)
    
    if dim is None:
        return input.topk(ceil(p * input.numel()), largest=largest)
    else:
        return input.topk(ceil(p * input.size(dim)), dim=dim, largest=largest)


def expected_shortfall(input: Tensor, p: float, dim: Optional[int] = None) -> Tensor:
    
    if dim is None:
        return -topp(input, p=p, largest=False).values.mean()
    else:
        return -topp(input, p=p, largest=False, dim=dim).values.mean(dim=dim)


def _min_values(input: Tensor, dim: Optional[int] = None) -> Tensor:
    return input.min() if dim is None else input.min(dim=dim).values


def _max_values(input: Tensor, dim: Optional[int] = None) -> Tensor:
    return input.max() if dim is None else input.max(dim=dim).values


def value_at_risk(input: Tensor, p: float, dim: Optional[int] = None) -> Tensor:
      # NOQA
    n = input.numel() if dim is None else input.size(dim)

    if p <= 1 / n:
        output = _min_values(input, dim=dim)
    elif p > 1 - 1 / n:
        output = _max_values(input, dim=dim)
    else:
        q = (p - (1 / n)) / (1 - (1 / n))
        output = input.quantile(q, dim=dim)

    return output


def quadratic_cvar(input: Tensor, lam: float, dim: Optional[int] = None) -> Tensor:
      # NOQA
    if dim is None:
        return quadratic_cvar(input.flatten(), lam, 0)

    output_target = torch.as_tensor(1 / (2 * lam))
    base = input.mean(dim=dim, keepdim=True)
    input = input - base

    def fn_target(omega: Tensor) -> Tensor:
        return fn.relu(-omega - input).mean(dim=dim, keepdim=True)

    lower = torch.amin(-input, dim=dim, keepdim=True) - 1e-8
    upper = torch.amax(-input, dim=dim, keepdim=True) + 1e-8

    precision = 1e-6 * 10 ** int(math.log10((upper - lower).amax()))

    omega = bisect(
        fn=fn_target,
        target=output_target,
        lower=lower,
        upper=upper,
        precision=precision,
    )
    return (
        omega
        + lam * fn.relu(-omega - input).square().mean(dim=dim, keepdim=True)
        - base
    ).squeeze(dim)


def leaky_clamp(
    input: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    clamped_slope: float = 0.01,
    inverted_output: str = "mean",
) -> Tensor:

    x = input

    if min is not None:
        min = torch.as_tensor(min).to(x)
        x = x.maximum(min + clamped_slope * (x - min))

    if max is not None:
        max = torch.as_tensor(max).to(x)
        x = x.minimum(max + clamped_slope * (x - max))

    if min is not None and max is not None:
        if inverted_output == "mean":
            y = (min + max) / 2
        elif inverted_output == "max":
            y = max
        else:
            raise ValueError("inverted_output must be 'mean' or 'max'.")
        x = x.where(min <= max, y)

    return x


def clamp(
    input: Tensor,
    min: Optional[Tensor] = None,
    max: Optional[Tensor] = None,
    inverted_output: str = "mean",
) -> Tensor:

    if inverted_output == "mean":
        output = leaky_clamp(input, min, max, clamped_slope=0.0, inverted_output="mean")
    elif inverted_output == "max":
        output = torch.clamp(input, min, max)
    else:
        raise ValueError("inverted_output must be 'mean' or 'max'.")
    return output


def realized_variance(input: Tensor, dt: TensorOrScalar) -> Tensor:

    return input.log().diff(dim=-1).square().mean(dim=-1) / dt


def realized_volatility(input: Tensor, dt: Union[Tensor, float]) -> Tensor:
    
    return realized_variance(input, dt=dt).sqrt()


def pl(
    spot: Tensor,
    unit: Tensor,
    cost: Optional[List[float]] = None,
    payoff: Optional[Tensor] = None,
    deduct_first_cost: bool = True,
    deduct_final_cost: bool = False,
) -> Tensor:

    # TODO(simaki): Support deduct_final_cost=True
    assert not deduct_final_cost, "not supported"

    if spot.size() != unit.size():
        raise RuntimeError(f"unmatched sizes: spot {spot.size()}, unit {unit.size()}")
    if payoff is not None:
        if payoff.dim() != 1 or spot.size(0) != payoff.size(0):
            raise RuntimeError(
                f"unmatched sizes: spot {spot.size()}, payoff {payoff.size()}"
            )

    output = unit[..., :-1].mul(spot.diff(dim=-1)).sum(dim=(-2, -1))

    if payoff is not None:
        output -= payoff

    if cost is not None:
        c = torch.tensor(cost).to(spot).unsqueeze(0).unsqueeze(-1)
        output -= (spot[..., 1:] * unit.diff(dim=-1).abs() * c).sum(dim=(-2, -1))
        if deduct_first_cost:
            output -= (spot[..., [0]] * unit[..., [0]].abs() * c).sum(dim=(-2, -1))

    return output


def terminal_value(
    spot: Tensor,
    unit: Tensor,
    cost: Optional[List[float]] = None,
    payoff: Optional[Tensor] = None,
    deduct_first_cost: bool = True,
) -> Tensor:
    
    return pl(
        spot=spot,
        unit=unit,
        cost=cost,
        payoff=payoff,
        deduct_first_cost=deduct_first_cost,
    )


def ncdf(input: Tensor) -> Tensor:

    return Normal(0.0, 1.0).cdf(input)


def npdf(input: Tensor) -> Tensor:

    return Normal(0.0, 1.0).log_prob(input).exp()


def d1(
    log_moneyness: TensorOrScalar,
    time_to_maturity: TensorOrScalar,
    volatility: TensorOrScalar,
) -> Tensor:

    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    if not (t >= 0).all():
        raise ValueError("all elements in time_to_maturity have to be non-negative")
    if not (v >= 0).all():
        raise ValueError("all elements in volatility have to be non-negative")
    variance = v * t.sqrt()
    output = s / variance + variance / 2
    # TODO(simaki): Replace zeros_like with 0.0 once https://github.com/pytorch/pytorch/pull/62084 is merged
    return output.where((s != 0).logical_or(variance != 0), torch.zeros_like(output))


def d2(
    log_moneyness: TensorOrScalar,
    time_to_maturity: TensorOrScalar,
    volatility: TensorOrScalar,
) -> Tensor:

    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    if not (t >= 0).all():
        raise ValueError("all elements in time_to_maturity have to be non-negative")
    if not (v >= 0).all():
        raise ValueError("all elements in volatility have to be non-negative")
    variance = v * t.sqrt()
    output = s / variance - variance / 2
    # TODO(simaki): Replace zeros_like with 0.0 once https://github.com/pytorch/pytorch/pull/62084 is merged
    return output.where((s != 0).logical_or(variance != 0), torch.zeros_like(output))


def ww_width(
    gamma: Tensor, spot: Tensor, cost: TensorOrScalar, a: TensorOrScalar = 1.0
) -> Tensor:

    return (cost * (3 / 2) * gamma.square() * spot / a).pow(1 / 3)


def svi_variance(
    input: TensorOrScalar,
    a: TensorOrScalar,
    b: TensorOrScalar,
    rho: TensorOrScalar,
    m: TensorOrScalar,
    sigma: TensorOrScalar,
) -> Tensor:

    k_m = torch.as_tensor(input - m)  # k - m
    return a + b * (rho * k_m + (k_m.square() + sigma ** 2).sqrt())


def bilerp(
    input1: Tensor,
    input2: Tensor,
    input3: Tensor,
    input4: Tensor,
    weight1: TensorOrScalar,
    weight2: TensorOrScalar,
) -> Tensor:

    lerp1 = torch.lerp(input1, input2, weight1)
    lerp2 = torch.lerp(input3, input4, weight1)
    return torch.lerp(lerp1, lerp2, weight2)


def _bs_theta_gamma_relation(gamma: Tensor, spot: Tensor, volatility: Tensor) -> Tensor:
    # theta = -(1/2) * vola^2 * spot^2 * gamma
    # by Black-Scholes formula
    return -gamma * volatility.square() * spot.square() / 2


def _bs_vega_gamma_relation(
    gamma: Tensor, spot: Tensor, time_to_maturity: Tensor, volatility: Tensor
) -> Tensor:
    # vega = vola * spot^2 * time * gamma
    # in Black-Scholes model
    # See Chapter 5 Appendix A, Bergomi "Stochastic volatility modeling"
    return gamma * volatility * spot.square() * time_to_maturity


def bs_european_price(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar = 1.0,
    call: bool = True,
) -> Tensor:
    
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

    spot = s.exp() * strike
    price = spot * ncdf(d1(s, t, v)) - strike * ncdf(d2(s, t, v))
    price = price + strike * (1 - s.exp()) if not call else price  # put-call parity

    return price


def bs_european_delta(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
) -> Tensor:
    
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

    delta = ncdf(d1(s, t, v))
    delta = delta - 1 if not call else delta  # put-call parity

    return delta


def bs_european_gamma(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = strike * s.exp()
    numerator = npdf(d1(s, t, v))
    denominator = spot * v * t.sqrt()
    output = numerator / denominator
    return torch.where(
        (numerator == 0).logical_and(denominator == 0), torch.zeros_like(output), output
    )


def bs_european_vega(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    price = strike * s.exp()
    return npdf(d1(s, t, v)) * price * t.sqrt()


def bs_european_theta(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    price = strike * s.exp()
    numerator = -npdf(d1(s, t, v)) * price * v
    denominator = 2 * t.sqrt()
    output = numerator / denominator
    return torch.where(
        (numerator == 0).logical_and(denominator == 0), torch.zeros_like(output), output
    )


def bs_european_binary_price(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
) -> Tensor:
    
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

    price = ncdf(d2(s, t, v))
    price = 1.0 - price if not call else price  # put-call parity

    return price


def bs_european_binary_delta(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

    spot = s.exp() * strike

    numerator = npdf(d2(s, t, v))
    denominator = spot * v * t.sqrt()
    delta = numerator / denominator
    delta = torch.where(
        (numerator == 0).logical_and(denominator == 0), torch.zeros_like(delta), delta
    )
    delta = -delta if not call else delta  # put-call parity

    return delta


def bs_european_binary_gamma(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = s.exp() * strike

    d2_tensor = d2(s, t, v)
    w = volatility * time_to_maturity.square()

    gamma = -npdf(d2_tensor).div(w * spot.square()) * (1 + d2_tensor.div(w))

    gamma = -gamma if not call else gamma  # put-call parity

    return gamma


def bs_european_binary_vega(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    
    gamma = bs_european_binary_gamma(
        log_moneyness=log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        call=call,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_vega_gamma_relation(
        gamma, spot=spot, time_to_maturity=time_to_maturity, volatility=volatility
    )


def bs_european_binary_theta(
    log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    call: bool = True,
    strike: TensorOrScalar = 1.0,
) -> Tensor:
    
    gamma = bs_european_binary_gamma(
        log_moneyness=log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        call=call,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_theta_gamma_relation(gamma, spot=spot, volatility=volatility)


def bs_american_binary_price(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
) -> Tensor:
    
    # This formula is derived using the results in Section 7.3.3 of Shreve's book.
    # Price is I_2 - I_4 where the interval of integration is [k --> -inf, b].
    # By this substitution we get N([log(S(0) / K) + ...] / sigma T) --> 1.

    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    p = ncdf(d2(s, t, v)) + s.exp() * ncdf(d1(s, t, v))

    return p.where(max_log_moneyness < 0, torch.ones_like(p))


def bs_american_binary_delta(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = s.exp() * strike

    d1_tensor = d1(s, t, v)
    d2_tensor = d2(s, t, v)
    w = v * t.sqrt()

    # ToDo: fix 0/0 issue
    p = (
        npdf(d2_tensor).div(spot * w)
        + ncdf(d1_tensor).div(strike)
        + npdf(d1_tensor).div(strike * w)
    )
    return p.where(max_log_moneyness < 0, torch.zeros_like(p))


def bs_american_binary_gamma(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
    spot = s.exp() * strike

    d1_tensor = d1(s, t, v)
    d2_tensor = d2(s, t, v)
    w = v * t.sqrt()

    p = (
        -npdf(d2_tensor).div(spot.square() * w)
        - d2_tensor * npdf(d2_tensor).div(spot.square() * w.square())
        + npdf(d1_tensor).div(spot * strike * w)
        - d1_tensor * npdf(d1_tensor).div(spot * strike * w.square())
    )
    return p.where(max_log_moneyness < 0, torch.zeros_like(p))


def bs_american_binary_vega(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    gamma = bs_american_binary_gamma(
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_vega_gamma_relation(
        gamma, spot=spot, time_to_maturity=time_to_maturity, volatility=volatility
    )


def bs_american_binary_theta(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    gamma = bs_american_binary_gamma(
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_theta_gamma_relation(gamma, spot=spot, volatility=volatility)


def bs_lookback_price(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    s, m, t, v = map(
        torch.as_tensor,
        (log_moneyness, max_log_moneyness, time_to_maturity, volatility),
    )

    spot = s.exp() * strike
    max = m.exp() * strike
    d1_value = d1(s, t, v)
    d2_value = d2(s, t, v)
    m1 = d1(s - m, t, v)  # d' in the paper
    m2 = d2(s - m, t, v)

    # when max < strike
    price_0 = spot * (
        ncdf(d1_value) + v * t.sqrt() * (d1_value * ncdf(d1_value) + npdf(d1_value))
    ) - strike * ncdf(d2_value)
    # when max >= strike
    price_1 = (
        spot * (ncdf(m1) + v * t.sqrt() * (m1 * ncdf(m1) + npdf(m1)))
        - strike
        + max * (1 - ncdf(m2))
    )

    return torch.where(max < strike, price_0, price_1)


def bs_lookback_delta(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    # TODO(simaki): Calculate analytically
    return autogreek.delta(
        bs_lookback_price,
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )


def bs_lookback_gamma(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    # TODO(simaki): Calculate analytically
    return autogreek.gamma(
        bs_lookback_price,
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )


def bs_lookback_vega(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    gamma = bs_lookback_gamma(
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_vega_gamma_relation(
        gamma, spot=spot, time_to_maturity=time_to_maturity, volatility=volatility
    )


def bs_lookback_theta(
    log_moneyness: Tensor,
    max_log_moneyness: Tensor,
    time_to_maturity: Tensor,
    volatility: Tensor,
    strike: TensorOrScalar,
) -> Tensor:
    
    gamma = bs_lookback_gamma(
        log_moneyness=log_moneyness,
        max_log_moneyness=max_log_moneyness,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        strike=strike,
    )
    spot = log_moneyness.exp() * strike
    return _bs_theta_gamma_relation(gamma, spot=spot, volatility=volatility)


def box_muller(
    input1: Tensor, input2: Tensor, epsilon: float = 1e-10
) -> Tuple[Tensor, Tensor]:

    radius = (-2 * input1.clamp(min=epsilon).log()).sqrt()
    angle = 2 * kPI * input2
    output1 = radius * angle.cos()
    output2 = radius * angle.sin()
    return output1, output2
