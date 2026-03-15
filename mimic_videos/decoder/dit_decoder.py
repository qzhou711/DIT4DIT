"""DiTActionDecoder — lightweight Diffusion Transformer for action denoising.

Architecture
------------
Given noisy actions, proprioception, and video features from the backbone,
the decoder predicts the flow-matching velocity field ``v = ε - a₀``.

Each :class:`DecoderBlock` applies, in order:
  1. **Cross-attention** to video backbone features (AdaLN-Zero conditioned)
  2. **Self-attention** over the action sequence (AdaLN-Zero conditioned)
  3. **Feed-forward MLP** with GELU (AdaLN-Zero conditioned)

The sequence is::

    [ proprio_token | a_1 | a_2 | ... | a_T ]

where ``proprio_token`` is either the projected robot state or a learned
masking token (durante training with ``proprio_mask_prob > 0``).

Registration
------------
Registered as ``"dit_action_decoder"`` in the global Registry so it can
be constructed from YAML config.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mimic_videos.core.registry import Registry
from .timestep_embed import JointTimestepEmbedding


# ------------------------------------------------------------------ #
# AdaLN-Zero modulation block
# ------------------------------------------------------------------ #


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with Zero initialisation.

    Produces (shift, scale, gate) from a conditioning vector; the linear
    projection is zero-initialised so each block starts as an identity
    transform at the beginning of training.

    Args:
        dim: Feature dimension to normalise.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim),
        )
        # Zero-init — gates start at 0, outputs start at 0
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalise *x* and compute the residual gate.

        Args:
            x: ``[B, L, D]``
            cond: ``[B, D]`` conditioning vector.

        Returns:
            Tuple ``(modulated_x, gate)`` where gate is ``[B, 1, D]``.
        """
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)  # each [B, D]
        x_norm = self.norm(x)
        x_mod = x_norm * (1.0 + scale[:, None]) + shift[:, None]
        return x_mod, gate[:, None]   # [B, L, D], [B, 1, D]


# ------------------------------------------------------------------ #
# Single decoder transformer block
# ------------------------------------------------------------------ #


class DecoderBlock(nn.Module):
    """One transformer block of the action decoder.

    Args:
        dim: Action sequence hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden dimension multiplier.
        context_dim: Dimension of the video backbone features (K, V sources).
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        context_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        # 1. Cross-attention to video context
        self.adaLN_cross = AdaLNZero(dim)
        self.cross_q = nn.Linear(dim, dim)
        self.cross_k = nn.Linear(context_dim, dim)
        self.cross_v = nn.Linear(context_dim, dim)
        self.cross_out = nn.Linear(dim, dim)

        # 2. Self-attention over action sequence
        self.adaLN_self = AdaLNZero(dim)
        self.self_q = nn.Linear(dim, dim)
        self.self_k = nn.Linear(dim, dim)
        self.self_v = nn.Linear(dim, dim)
        self.self_out = nn.Linear(dim, dim)

        # 3. Feed-forward MLP (GELU)
        self.adaLN_ff = AdaLNZero(dim)
        ff_dim = dim * mlp_ratio
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def _mha(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Shared multi-head attention kernel (uses Flash Attention if available)."""
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)
        out = F.scaled_dot_product_attention(q, k, v)
        return rearrange(out, "b h l d -> b l (h d)")

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Action token sequence, ``[B, L, dim]``.
            context: Video backbone features, ``[B, S, context_dim]``.
            cond: Joint timestep conditioning, ``[B, dim]``.

        Returns:
            Updated sequence, ``[B, L, dim]``.
        """
        # --- Cross-attention to video features ---
        x_mod, gate = self.adaLN_cross(x, cond)
        cross_out = self.cross_out(
            self._mha(
                self.cross_q(x_mod),
                self.cross_k(context),
                self.cross_v(context),
            )
        )
        x = x + gate * cross_out

        # --- Self-attention over action sequence ---
        x_mod, gate = self.adaLN_self(x, cond)
        self_out = self.self_out(
            self._mha(
                self.self_q(x_mod),
                self.self_k(x_mod),
                self.self_v(x_mod),
            )
        )
        x = x + gate * self_out

        # --- Feed-forward MLP ---
        x_mod, gate = self.adaLN_ff(x, cond)
        x = x + gate * self.ff(x_mod)

        return x


# ------------------------------------------------------------------ #
# Full decoder
# ------------------------------------------------------------------ #


@Registry.decoder("dit_action_decoder")
class DiTActionDecoder(nn.Module):
    """Diffusion Transformer for action chunk denoising.

    Args:
        action_dim: Dimension of each action vector.
        proprio_dim: Dimension of the proprioception state.
        hidden_dim: Transformer hidden dimension.
        num_layers: Number of :class:`DecoderBlock` layers.
        num_heads: Attention heads per block.
        mlp_ratio: Feed-forward expansion ratio.
        context_dim: Video feature dimension (backbone hidden dim).
        chunk_size: Number of action steps in a chunk.
        proprio_mask_prob: Probability of replacing proprio with the
            learned mask token during training.
    """

    def __init__(
        self,
        action_dim: int = 16,
        proprio_dim: int = 16,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        context_dim: int = 2048,
        chunk_size: int = 16,
        proprio_mask_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.proprio_mask_prob = proprio_mask_prob

        # Input projections
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learned proprioception mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Absolute learned positional embeddings: [1 proprio + chunk_size actions]
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + chunk_size, hidden_dim) * 0.02
        )

        # Joint timestep conditioning
        self.timestep_embed = JointTimestepEmbedding(
            cond_dim=hidden_dim, sinusoidal_dim=256
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                context_dim=context_dim,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm + zero-init output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        proprio: torch.Tensor,
        video_features: torch.Tensor,
        tau_a: torch.Tensor,
        tau_v: torch.Tensor,
        apply_proprio_mask: bool = False,
    ) -> torch.Tensor:
        """Predict the velocity field for action denoising.

        Args:
            noisy_actions: ``[B, chunk_size, action_dim]``.
            proprio: Current robot state, ``[B, proprio_dim]``.
            video_features: Backbone features, ``[B, S, context_dim]``.
            tau_a: Action flow time ``[B]``.
            tau_v: Video flow time ``[B]``.
            apply_proprio_mask: If ``True`` and ``proprio_mask_prob > 0``,
                randomly replace proprio tokens with the mask token.

        Returns:
            Predicted velocity, ``[B, chunk_size, action_dim]``.
        """
        B = noisy_actions.shape[0]

        # Project actions and proprio
        a_tokens = self.action_proj(noisy_actions)         # [B, T, D]
        s_token = self.proprio_proj(proprio.unsqueeze(1))  # [B, 1, D]

        # Optional proprio masking (dropout-style augmentation)
        if apply_proprio_mask and self.proprio_mask_prob > 0.0:
            mask = torch.rand(B, 1, 1, device=proprio.device) < self.proprio_mask_prob
            s_token = torch.where(mask, self.mask_token.expand(B, -1, -1), s_token)

        # Build sequence: [proprio | action_1 | ... | action_T]
        x = torch.cat([s_token, a_tokens], dim=1)          # [B, 1+T, D]
        x = x + self.pos_embed[:, : x.shape[1]]

        # Compute conditioning from both flow times
        cond = self.timestep_embed(tau_v, tau_a)            # [B, D]

        # Transformer pass
        for block in self.blocks:
            x = block(x, video_features, cond)

        x = self.final_norm(x)

        # Slice out action tokens (skip proprio), project to action space
        velocity = self.output_head(x[:, 1:])               # [B, T, action_dim]
        return velocity
