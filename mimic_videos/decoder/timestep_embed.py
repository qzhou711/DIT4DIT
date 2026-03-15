"""Timestep embedding modules for the action decoder.

Provides a bilinear-affine embedding that jointly encodes the video
flow time ``τ_v`` and the action flow time ``τ_a`` into a single
conditioning vector consumed by each DiT block via AdaLN-Zero.

Keeping this in a separate module makes it easy to swap alternative
embedding strategies (e.g., learnt positional, FiLM, cross-attention)
without touching the main decoder architecture.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Classic sinusoidal positional embedding for a scalar in ``[0, 1]``.

    Args:
        dim: Output embedding dimension.
        max_period: Controls the frequency range of the sinusoids.
    """

    def __init__(self, dim: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of scalars.

        Args:
            x: ``[B]`` float values in ``[0, 1]``.

        Returns:
            ``[B, dim]`` sinusoidal embeddings.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=x.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = x[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class JointTimestepEmbedding(nn.Module):
    """Bilinear-affine joint embedding of ``(τ_v, τ_a)``.

    Encodes each flow time independently with sinusoidal embeddings +
    MLP, then fuses them via element-wise multiplication (bilinear)
    followed by a learned affine transform.

    This design allows the action decoder to be simultaneously aware
    of *both* denoising states, which is essential for the coupled video-
    action generation scheme.

    Args:
        cond_dim: Dimension of the output conditioning vector.
        sinusoidal_dim: Intermediate sinusoidal embedding dimension.
    """

    def __init__(self, cond_dim: int = 512, sinusoidal_dim: int = 256) -> None:
        super().__init__()
        self.cond_dim = cond_dim

        self.sin_v = SinusoidalEmbedding(sinusoidal_dim)
        self.sin_a = SinusoidalEmbedding(sinusoidal_dim)

        self.proj_v = nn.Sequential(
            nn.Linear(sinusoidal_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.proj_a = nn.Sequential(
            nn.Linear(sinusoidal_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Post-bilinear affine — zero-init so the gate starts at identity
        self.fuse = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, tau_v: torch.Tensor, tau_a: torch.Tensor) -> torch.Tensor:
        """Compute the joint timestep conditioning vector.

        Args:
            tau_v: Video flow time ``[B]``.
            tau_a: Action flow time ``[B]``.

        Returns:
            Conditioning vector ``[B, cond_dim]``.
        """
        h_v = self.proj_v(self.sin_v(tau_v))   # [B, cond_dim]
        h_a = self.proj_a(self.sin_a(tau_a))   # [B, cond_dim]
        return self.fuse(h_v * h_a)             # bilinear → affine → [B, cond_dim]
