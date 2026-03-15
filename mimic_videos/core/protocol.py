"""RobotPolicy Protocol and shared data containers.

Defines the canonical interface that all policy implementations must
satisfy, enabling drop-in swaps between different backbone/decoder
combinations without changing downstream code.

Typed containers ``Observation`` and ``ActionChunk`` keep tensor shapes
explicit throughout the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable

import torch


# ------------------------------------------------------------------ #
# Data containers
# ------------------------------------------------------------------ #


@dataclass
class Observation:
    """A single robot observation at time *t*.

    Attributes:
        video: Concatenated camera frames, shape ``[T, C, H, W]`` in ``[-1, 1]``.
            *T* is the observation horizon (e.g. 17 pixel frames).
        proprio: Proprioceptive state vector, shape ``[proprio_dim]``.
        camera_names: Ordered list of camera identifiers used to build *video*.
        language_embedding: Optional precomputed T5 embedding,
            shape ``[seq_len, text_dim]``.
        extras: Arbitrary auxiliary data keyed by string.
    """

    video: torch.Tensor
    proprio: torch.Tensor
    camera_names: list[str] = field(default_factory=list)
    language_embedding: Optional[torch.Tensor] = None
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> "Observation":
        """Move all tensors to *device*, returning a new Observation."""
        return Observation(
            video=self.video.to(device),
            proprio=self.proprio.to(device),
            camera_names=self.camera_names,
            language_embedding=(
                self.language_embedding.to(device)
                if self.language_embedding is not None
                else None
            ),
            extras={k: v.to(device) for k, v in self.extras.items()},
        )


@dataclass
class ActionChunk:
    """A predicted chunk of future robot actions.

    Attributes:
        actions: Predicted actions, shape ``[chunk_size, action_dim]``.
        tau_v: Video noise level used during prediction (inference hyperparameter).
    """

    actions: torch.Tensor
    tau_v: float = 1.0

    @property
    def chunk_size(self) -> int:
        return self.actions.shape[0]

    @property
    def action_dim(self) -> int:
        return self.actions.shape[1]


# ------------------------------------------------------------------ #
# RobotPolicy Protocol
# ------------------------------------------------------------------ #


@runtime_checkable
class RobotPolicy(Protocol):
    """Structural Protocol defining the interface for all robot policies.

    Any class implementing ``predict`` and ``reset`` satisfies this
    protocol, regardless of inheritance.  This allows mixing policies from
    different frameworks in evaluation loops.

    Example::

        policy: RobotPolicy = WorldActionPolicy(...)
        assert isinstance(policy, RobotPolicy)   # True — runtime check

        chunk = policy.predict(obs)
        policy.reset()   # clear episode state between rollouts
    """

    def predict(self, obs: Observation) -> ActionChunk:
        """Predict the next action chunk from *obs*.

        Args:
            obs: Current robot observation.

        Returns:
            Predicted action chunk (denormalized, ready to execute).
        """
        ...

    def reset(self) -> None:
        """Reset any episode-level state (e.g. cached latents, buffers).

        Call this between rollout episodes during evaluation.
        """
        ...
