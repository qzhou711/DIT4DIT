"""Flow-matching noise schedules for video and action denoising.

Each schedule defines *how* flow times tau ∈ [0, 1] are sampled during
training, and also provides the shared interpolation / velocity / ODE
utilities used by both the backbone and the decoder.

Key design difference from the original:
  - Schedules are standalone classes registered in the Registry.
  - Action schedule correctly implements ``T_a(τ) ∝ √τ - 0.001``
    (from the paper) via acceptance-rejection sampling — fixing a bug
    in the original implementation where power=0.999 produced a near-
    uniform distribution instead.

Usage::

    from mimic_videos.core.noise_schedule import build_schedule

    vid_sch  = build_schedule("logit_normal")
    act_sch  = build_schedule("sqrt_minus_eps")

    tau_v = vid_sch.sample(batch_size, device)
    tau_a = act_sch.sample(batch_size, device)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .registry import Registry


# ------------------------------------------------------------------ #
# Abstract base
# ------------------------------------------------------------------ #


class FlowSchedule(ABC):
    """Base class for flow-matching timestep schedules."""

    @abstractmethod
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample *batch_size* timesteps τ ∈ (0, 1).

        Args:
            batch_size: Number of samples.
            device: Target device.

        Returns:
            Tensor ``[B]`` with values in ``(0, 1)``.
        """

    # ---------------------------------------------------------------- #
    # Shared flow-matching utilities (same for all schedules)
    # ---------------------------------------------------------------- #

    @staticmethod
    def interpolate(
        x0: torch.Tensor, eps: torch.Tensor, tau: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation: ``x_τ = (1 - τ) * x0 + τ * eps``.

        Args:
            x0: Clean data, ``[B, ...]``.
            eps: Standard Gaussian noise, same shape as *x0*.
            tau: Flow times, ``[B]`` (broadcast automatically).

        Returns:
            Noisy sample ``x_τ``, same shape as *x0*.
        """
        while tau.ndim < x0.ndim:
            tau = tau.unsqueeze(-1)
        return (1.0 - tau) * x0 + tau * eps

    @staticmethod
    def velocity_target(x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Optimal velocity target for linear flow: ``v = eps - x0``."""
        return eps - x0

    @staticmethod
    def mse_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Masked MSE between *pred* and *target* velocity fields.

        Args:
            pred:   Predicted velocity, ``[B, ...]``.
            target: Ground-truth velocity, ``[B, ...]``.
            mask:   Optional boolean/float mask; loss computed only where > 0.

        Returns:
            Scalar loss value.
        """
        diff = pred - target
        if mask is not None:
            diff = diff * mask
            return (diff ** 2).sum() / mask.sum().clamp(min=1.0)
        return F.mse_loss(pred, target)

    @staticmethod
    def euler_integrate(
        velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
        x_init: torch.Tensor,
        num_steps: int,
        tau_start: float = 1.0,
        tau_end: float = 0.0,
    ) -> torch.Tensor:
        """Fixed-step Euler ODE integration of ``dx/dτ = v(x_τ, τ)``.

        Integrates from ``tau_start`` (typically 1 = pure noise) to
        ``tau_end`` (typically 0 = clean data).

        Args:
            velocity_fn: ``(x_t, tau_scalar) -> velocity``.
            x_init: Initial state at *tau_start*, ``[B, ...]``.
            num_steps: Number of Euler steps.
            tau_start: Starting flow time (default 1.0).
            tau_end: Ending flow time (default 0.0).

        Returns:
            Estimated clean sample at *tau_end*, same shape as *x_init*.
        """
        dt = (tau_end - tau_start) / num_steps
        x = x_init
        tau = tau_start
        for _ in range(num_steps):
            v = velocity_fn(x, tau)
            x = x + v * dt
            tau = tau + dt
        return x


# ------------------------------------------------------------------ #
# Concrete schedule implementations
# ------------------------------------------------------------------ #


@Registry.schedule("logit_normal")
class LogitNormalSchedule(FlowSchedule):
    """Logit-normal distribution for video flow times.

    ``τ = σ(z)`` where ``z ~ N(mean, std)``.  Concentrates samples
    around 0.5, providing balanced coverage of the denoising trajectory.

    This matches the video pretraining schedule in the mimic-video paper.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(batch_size, device=device) * self.std + self.mean
        return torch.sigmoid(z)


@Registry.schedule("sqrt_minus_eps")
class SqrtMinusEpsSchedule(FlowSchedule):
    """Action flow schedule: ``T_a(τ) ∝ √τ - ε`` with ``ε = 0.001``.

    This is the distribution specified in the mimic-video paper (Sec. V-F
    and Algorithm 2).  The unnormalized density is ``p(τ) ∝ √τ - 0.001``,
    which evaluates to zero at τ ≈ 10⁻⁶ and grows toward 1.

    Implemented via acceptance-rejection sampling using a ``Uniform(0,1)``
    proposal and the normalised acceptance ratio.

    Note:
        The original open-source implementation used ``U^(1/0.999) ≈ Uniform``,
        which does *not* match the paper.  This class corrects that.
    """

    EPS: float = 0.001

    def sample(
        self, batch_size: int, device: torch.device, max_attempts: int = 20
    ) -> torch.Tensor:
        # Compute the maximum of p(τ) = √τ - ε on [0, 1] at τ = 1
        p_max = 1.0 - self.EPS  # √1 - 0.001 = 0.999

        collected: list[torch.Tensor] = []
        remaining = batch_size

        for _ in range(max_attempts):
            u = torch.rand(remaining * 4, device=device)          # candidates
            p_u = u.sqrt() - self.EPS                              # unnormalized density
            accept_prob = (p_u / p_max).clamp(min=0.0)            # ratio (never > 1)
            accept_mask = torch.rand_like(u) < accept_prob         # acceptance test
            accepted = u[accept_mask][:remaining]
            collected.append(accepted)
            remaining -= accepted.numel()
            if remaining <= 0:
                break

        result = torch.cat(collected, dim=0)[:batch_size]
        # Safety fallback (extremely rare): fill any missing slots with uniform
        if result.numel() < batch_size:
            extra = torch.rand(batch_size - result.numel(), device=device)
            result = torch.cat([result, extra], dim=0)
        return result.clamp(min=1e-6, max=1.0 - 1e-6)


@Registry.schedule("uniform")
class UniformSchedule(FlowSchedule):
    """Uniform distribution over ``(0, 1)`` — simple baseline schedule."""

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device).clamp(min=1e-6, max=1.0 - 1e-6)


@Registry.schedule("power_law")
class PowerLawSchedule(FlowSchedule):
    """Power-law schedule: ``τ = U^(1/power)`` where ``U ~ Uniform(0, 1)``.

    Args:
        power: Exponent controlling the skew.
            - ``power > 1``: biases toward larger τ (noisier samples).
            - ``power < 1``: biases toward smaller τ (cleaner samples).
            - ``power = 1``: equivalent to Uniform.
    """

    def __init__(self, power: float = 2.0) -> None:
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        self.power = power

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        u = torch.rand(batch_size, device=device).clamp(min=1e-6)
        return u.pow(1.0 / self.power).clamp(max=1.0 - 1e-6)


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #


def build_schedule(name: str, **kwargs) -> FlowSchedule:
    """Instantiate a registered schedule by *name*.

    Args:
        name: Registered schedule name (e.g. ``"logit_normal"``).
        **kwargs: Constructor keyword arguments forwarded to the schedule.

    Returns:
        Configured :class:`FlowSchedule` instance.

    Raises:
        KeyError: If *name* is not registered.

    Example::

        schedule = build_schedule("power_law", power=3.0)
        tau = schedule.sample(8, torch.device("cuda"))
    """
    cls = Registry.get_schedule(name)
    return cls(**kwargs)
