"""FeatureExtractor — decoupled hidden-state hook and pooling module.

Separating hook management and spatial pooling from CosmosBackbone
lets you independently configure *which layer(s)* to tap and *how*
to aggregate the resulting token sequence before passing it downstream.

Design notes
------------
- Supports registering hooks on **multiple** transformer layers simultaneously
  (the original code only supported a single layer).
- Pooling mode is set explicitly at construction time — never implicitly
  falling back to a default that silently contradicts the config.
- ``PoolingMode.SPATIAL_MEAN`` (5 temporal tokens) and
  ``PoolingMode.ALL_TOKENS`` (~6 000 tokens) are both available.
  The paper uses ALL_TOKENS.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class PoolingMode(str, Enum):
    """How to compress spatial/temporal tokens from the transformer."""

    SPATIAL_MEAN = "spatial_mean"
    """Average over spatial patches → ``[B, T_lat, D]`` (5 tokens for a 5-frame sequence)."""

    ALL_TOKENS = "all_tokens"
    """Return the full flattened sequence → ``[B, T_lat * H' * W', D]``."""


class FeatureExtractor(nn.Module):
    """Manages forward hooks on a ViT/DiT transformer and aggregates tokens.

    Args:
        transformer: The transformer module to hook into.
        layer_indices: Which transformer block indices to hook.
            Activations from all hooks are stored separately under
            ``captured[layer_idx]``.
        pooling_mode: How to aggregate spatial tokens before cross-attention.
            Must be set *explicitly* — there is no implicit default.
        num_latent_frames: Expected number of latent time frames ``T``.
            Required when ``pooling_mode=SPATIAL_MEAN`` to compute
            ``H' * W' = T*H'*W' / T``.
    """

    def __init__(
        self,
        transformer: nn.Module,
        layer_indices: List[int],
        pooling_mode: PoolingMode,
        num_latent_frames: int,
    ) -> None:
        super().__init__()
        self.layer_indices = sorted(layer_indices)
        self.pooling_mode = pooling_mode
        self.num_latent_frames = num_latent_frames

        # Raw captured tensors from the most recent forward pass.
        # Keys are layer indices.
        self._captured: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        self._attach_hooks(transformer)

    # ---------------------------------------------------------------- #
    # Hook management
    # ---------------------------------------------------------------- #

    def _attach_hooks(self, transformer: nn.Module) -> None:
        """Register a forward hook on each requested layer."""
        self._detach_hooks()

        # Resolve the list of transformer blocks robustly
        blocks = self._resolve_blocks(transformer)

        for layer_idx in self.layer_indices:
            if layer_idx >= len(blocks):
                raise IndexError(
                    f"Requested layer {layer_idx} but transformer has "
                    f"only {len(blocks)} blocks."
                )
            block = blocks[layer_idx]

            def _make_hook(idx: int):
                def _hook(
                    _module: nn.Module, _inp: tuple, output: torch.Tensor
                ) -> None:
                    # output: [B, T*H'*W', D]
                    self._captured[idx] = output

                return _hook

            handle = block.register_forward_hook(_make_hook(layer_idx))
            self._hooks.append(handle)

    def _detach_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @staticmethod
    def _resolve_blocks(transformer: nn.Module) -> nn.ModuleList:
        """Find the transformer block list regardless of PEFT wrapping."""
        candidates = [
            # LoRA via PEFT — base_model.model has the raw model
            lambda m: m.base_model.model.transformer_blocks,
            lambda m: m.transformer_blocks,
        ]
        for fn in candidates:
            try:
                blocks = fn(transformer)
                if blocks is not None:
                    return blocks
            except AttributeError:
                continue
        raise AttributeError(
            "Cannot locate transformer_blocks in the provided model. "
            "Make sure the model has a .transformer_blocks attribute."
        )

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def clear(self) -> None:
        """Discard all cached activations (call before each forward pass)."""
        self._captured.clear()

    def get_raw(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return the raw captured activation for *layer_idx*, or ``None``."""
        return self._captured.get(layer_idx)

    def get_primary(self) -> torch.Tensor:
        """Return pooled features from the *last* (highest-index) hook layer.

        This is the default conditioning signal for the action decoder.

        Returns:
            Pooled features — shape depends on ``pooling_mode``:
            - ``SPATIAL_MEAN``  → ``[B, T_lat, D]``
            - ``ALL_TOKENS``    → ``[B, T_lat * H'W', D]``

        Raises:
            RuntimeError: If no activation has been captured yet.
        """
        primary_idx = self.layer_indices[-1]
        raw = self._captured.get(primary_idx)
        if raw is None:
            raise RuntimeError(
                "No hidden states captured. "
                "Ensure the backbone ran a forward pass before calling get_primary()."
            )
        return self.pool(raw)

    def get_all_layers(self) -> Dict[int, torch.Tensor]:
        """Return pooled features for *every* hooked layer.

        Returns:
            Dict mapping layer index → pooled tensor.
        """
        return {idx: self.pool(v) for idx, v in self._captured.items()}

    def pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply spatial aggregation according to ``pooling_mode``.

        Args:
            hidden: Raw token sequence, ``[B, T*H'*W', D]``.

        Returns:
            Aggregated features; shape determined by ``pooling_mode``.
        """
        if self.pooling_mode is PoolingMode.ALL_TOKENS:
            return hidden  # [B, T*H'*W', D]

        # SPATIAL_MEAN: average spatial patches per temporal frame
        B, THW, D = hidden.shape
        T = self.num_latent_frames
        HW = THW // T
        # Reshape → [B, T, HW, D] → mean over HW → [B, T, D]
        return hidden.view(B, T, HW, D).mean(dim=2)
