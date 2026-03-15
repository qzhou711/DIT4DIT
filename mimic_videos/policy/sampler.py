"""ODESampler — standalone ODE integration utilities.

Separating the ODE solver from the policy class lets you swap integration
methods (Euler, RK4, DDIM-style) without touching inference logic.

Currently provided solvers
--------------------------
- :class:`EulerSampler` — fixed-step Euler method (simplest, fastest).
- :class:`MidpointSampler` — 2nd-order Runge-Kutta for lower truncation error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch


VelocityFn = Callable[[torch.Tensor, float], torch.Tensor]
"""Type alias: ``(x_t, tau_float) -> velocity``"""


class ODESampler(ABC):
    """Abstract ODE sampler interface."""

    @abstractmethod
    def integrate(
        self,
        velocity_fn: VelocityFn,
        x_init: torch.Tensor,
        num_steps: int,
        tau_start: float = 1.0,
        tau_end: float = 0.0,
    ) -> torch.Tensor:
        """Integrate the ODE from *tau_start* to *tau_end*.

        Args:
            velocity_fn: Callable returning the flow velocity at ``(x_t, τ)``.
            x_init: Initial state at *tau_start*, ``[B, ...]``.
            num_steps: Number of integration steps.
            tau_start: Starting flow time (pure noise, default 1.0).
            tau_end: Final flow time (clean data, default 0.0).

        Returns:
            Estimated state at *tau_end*, same shape as *x_init*.
        """


class EulerSampler(ODESampler):
    """Fixed-step Euler integration of ``dx/dτ = v(x_τ, τ)``."""

    def integrate(
        self,
        velocity_fn: VelocityFn,
        x_init: torch.Tensor,
        num_steps: int,
        tau_start: float = 1.0,
        tau_end: float = 0.0,
    ) -> torch.Tensor:
        dt = (tau_end - tau_start) / num_steps
        x = x_init
        tau = tau_start
        for _ in range(num_steps):
            v = velocity_fn(x, tau)
            x = x + v * dt
            tau += dt
        return x


class MidpointSampler(ODESampler):
    """Explicit midpoint (RK2) integration — lower truncation error than Euler.

    Uses two velocity evaluations per step::

        k1 = v(x_t,     τ)
        k2 = v(x_t + k1*dt/2,  τ + dt/2)
        x_{t+1} = x_t + k2 * dt
    """

    def integrate(
        self,
        velocity_fn: VelocityFn,
        x_init: torch.Tensor,
        num_steps: int,
        tau_start: float = 1.0,
        tau_end: float = 0.0,
    ) -> torch.Tensor:
        dt = (tau_end - tau_start) / num_steps
        x = x_init
        tau = tau_start
        for _ in range(num_steps):
            k1 = velocity_fn(x, tau)
            k2 = velocity_fn(x + k1 * (dt / 2.0), tau + dt / 2.0)
            x = x + k2 * dt
            tau += dt
        return x


def build_sampler(method: str = "euler") -> ODESampler:
    """Instantiate a sampler by *method* name.

    Args:
        method: ``"euler"`` or ``"midpoint"``.

    Returns:
        :class:`ODESampler` instance.

    Raises:
        ValueError: For unknown method names.
    """
    methods = {"euler": EulerSampler, "midpoint": MidpointSampler}
    if method not in methods:
        raise ValueError(f"Unknown ODE method '{method}'. Choose from {list(methods)}")
    return methods[method]()
