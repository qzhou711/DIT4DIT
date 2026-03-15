"""Training stage strategies — Stage1Strategy and Stage2Strategy.

Using the Strategy pattern means the unified :class:`~mimic_videos.engine.trainer.Trainer`
doesn't need to know whether it's finetuning the backbone or training the decoder:
it just calls ``stage.forward_step(batch)`` and ``stage.configure_optimizer()``.

This is the main structural difference versus the original which had two
largely copy-pasted trainer classes (``Stage1Trainer`` / ``Stage2Trainer``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class StageConfig:
    """Shared training hyperparameters for one training stage."""

    lr: float = 1e-4
    warmup_steps: int = 1000
    total_steps: int = 27000
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 64
    lr_schedule: str = "constant"   # "constant" | "linear_decay"
    dtype: str = "bfloat16"
    output_dir: str = "checkpoints/"
    log_every: int = 10
    save_every: int = 1000
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Action noise schedule name (used by Stage 2)
    action_noise_schedule: str = "sqrt_minus_eps"
    # Video noise schedule name
    video_noise_schedule: str = "logit_normal"


class TrainingStage(ABC):
    """Abstract training stage strategy.

    Implementations define *which parameters* are trained and *how* a
    single gradient step is computed.  The :class:`Trainer` handles
    everything else: optimiser, logging, checkpointing, gradient scaling.
    """

    @abstractmethod
    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return parameters to pass to the optimiser."""

    @abstractmethod
    def forward_step(self, batch: dict, device: str) -> torch.Tensor:
        """Compute the flow-matching loss for *batch* and return it.

        Implementations must NOT call ``.backward()`` — that is the Trainer's
        responsibility (to support gradient accumulation, AMP, etc.).

        Args:
            batch: Dict with keys ``"video"``, ``"proprio"``, ``"actions"``,
                   and optionally ``"t5_embedding"``.
            device: Target device string.

        Returns:
            Scalar loss tensor (already divided by gradient accumulation steps).
        """

    def after_step(self, step: int) -> None:
        """Optional hook called after each *effective* optimiser step."""


class Stage1Strategy(TrainingStage):
    """LoRA fine-tuning of the Cosmos video backbone.

    The backbone learns to predict future video frames conditioned on past
    frames, acting as a world model prior for downstream action decoding.

    Args:
        backbone: :class:`~mimic_videos.backbone.CosmosBackbone` instance.
        video_schedule: Instantiated video flow schedule.
        num_cond_frames: Number of conditioning latent frames.
        precomputed_t5: Optional text embedding ``[1, seq_len, D]``.
    """

    def __init__(
        self,
        backbone,
        video_schedule,
        num_cond_frames: int = 2,
        precomputed_t5: Optional[torch.Tensor] = None,
    ) -> None:
        self.backbone = backbone
        self.video_schedule = video_schedule
        self.num_cond_frames = num_cond_frames
        self.precomputed_t5 = precomputed_t5

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.transformer.parameters() if p.requires_grad]

    def forward_step(self, batch: dict, device: str) -> torch.Tensor:
        from mimic_videos.core.noise_schedule import FlowSchedule

        video = batch["video"].to(device)         # [B, T, C, H, W]
        B, T, C, H, W = video.shape

        # VAE encode: permute to [B, C, T, H, W]
        z_clean = self.backbone.encode_frames(video.permute(0, 2, 1, 3, 4))
        z_cond = z_clean[:, :, : self.num_cond_frames]
        z_future = z_clean[:, :, self.num_cond_frames:]
        z_future = z_future.to(device)

        # Sample video flow time τ_v
        tau_v = self.video_schedule.sample(B, device=device)

        # Add noise to future latents
        eps = torch.randn_like(z_future)
        z_noisy = self.video_schedule.interpolate(z_future, eps, tau_v.view(B, 1, 1, 1, 1))

        # Text conditioning
        t5_emb = batch.get("t5_embedding", self.precomputed_t5)
        if t5_emb is None:
            raise ValueError("No T5 embedding available for Stage 1 training.")
        t5_emb = t5_emb.to(device)
        if t5_emb.ndim == 2:
            t5_emb = t5_emb.unsqueeze(0).expand(B, -1, -1)

        # Forward transformer
        raw_out, _ = self.backbone.forward_transformer(z_noisy, z_cond, tau_v, t5_emb)

        # Velocity target for prediction frames only
        v_target = self.video_schedule.velocity_target(z_future, eps)
        v_pred = raw_out[:, :, self.num_cond_frames:]

        return self.video_schedule.mse_loss(v_pred, v_target)


class Stage2Strategy(TrainingStage):
    """Action decoder training with a frozen video backbone.

    Uses *both* a video and action flow schedule.  The backbone provides
    conditioning hidden states at a randomly sampled video noise level τ_v,
    and the decoder is trained to denoise action chunks at randomly sampled
    action noise levels τ_a.

    Args:
        backbone: Frozen :class:`~mimic_videos.backbone.CosmosBackbone`.
        decoder: :class:`~mimic_videos.decoder.DiTActionDecoder` to train.
        video_schedule: Video flow schedule (for τ_v sampling).
        action_schedule: Action flow schedule (for τ_a sampling, corrected).
        num_cond_frames: Number of conditioning latent frames.
        precomputed_t5: Optional text embedding ``[1, seq_len, D]``.
    """

    def __init__(
        self,
        backbone,
        decoder,
        video_schedule,
        action_schedule,
        num_cond_frames: int = 2,
        precomputed_t5: Optional[torch.Tensor] = None,
    ) -> None:
        self.backbone = backbone
        self.decoder = decoder
        self.video_schedule = video_schedule
        self.action_schedule = action_schedule
        self.num_cond_frames = num_cond_frames
        self.precomputed_t5 = precomputed_t5

        # Ensure backbone is frozen
        self.backbone.freeze()

    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.decoder.parameters())

    def forward_step(self, batch: dict, device: str) -> torch.Tensor:
        video = batch["video"].to(device)
        actions = batch["actions"].to(device)
        proprio = batch["proprio"].to(device)
        B = video.shape[0]

        # VAE encode (no_grad — backbone is frozen)
        with torch.no_grad():
            z_clean = self.backbone.encode_frames(video.permute(0, 2, 1, 3, 4))
            z_cond = z_clean[:, :, : self.num_cond_frames]
            z_future = z_clean[:, :, self.num_cond_frames:]

            # Sample τ_v and partially denoise future latents
            tau_v = self.video_schedule.sample(B, device=device)
            eps_v = torch.randn_like(z_future)
            z_noisy = self.video_schedule.interpolate(z_future, eps_v, tau_v.view(B, 1, 1, 1, 1))

            t5_emb = batch.get("t5_embedding", self.precomputed_t5)
            if t5_emb is None:
                raise ValueError("No T5 embedding available for Stage 2 training.")
            t5_emb = t5_emb.to(device)
            if t5_emb.ndim == 2:
                t5_emb = t5_emb.unsqueeze(0).expand(B, -1, -1)

            # Run backbone to capture hidden states
            self.backbone.forward_transformer(z_noisy, z_cond, tau_v, t5_emb)
            # Explicitly use ALL_TOKENS pooling (no silent defaults)
            video_feats = self.backbone.features.get_primary().float()  # [B, S, D]

        # Sample τ_a and add noise to actions
        tau_a = self.action_schedule.sample(B, device=device)
        eps_a = torch.randn_like(actions)
        a_noisy = self.action_schedule.interpolate(actions, eps_a, tau_a.view(B, 1, 1))

        # Predict velocity field (with random proprio masking)
        v_pred = self.decoder(
            a_noisy, proprio, video_feats.detach(), tau_a, tau_v,
            apply_proprio_mask=True,
        )
        v_target = self.action_schedule.velocity_target(actions, eps_a)

        return self.action_schedule.mse_loss(v_pred, v_target)
