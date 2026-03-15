"""WorldActionPolicy — implements the RobotPolicy Protocol.

This is the top-level inference object that ties together:
- :class:`~mimic_videos.backbone.CosmosBackbone` (video world model)
- :class:`~mimic_videos.decoder.DiTActionDecoder` (action denoising)
- :class:`~mimic_videos.policy.sampler.ODESampler` (ODE integration)

It implements the :class:`~mimic_videos.core.protocol.RobotPolicy` Protocol so it
can be used interchangeably with any other conforming policy in eval loops.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from mimic_videos.core.protocol import ActionChunk, Observation, RobotPolicy
from mimic_videos.backbone import CosmosBackbone
from mimic_videos.decoder import DiTActionDecoder
from .sampler import ODESampler, EulerSampler


class WorldActionPolicy(nn.Module):
    """Inference-time policy combining the video backbone with the action decoder.

    Implements the two-step inference algorithm from the paper:
    1. Encode past video frames; optionally partially denoise future frames up
       to ``tau_v``.
    2. Fully denoise an action chunk (τ: 1 → 0) conditioned on the resulting
       intermediate backbone representations.

    Args:
        backbone: Pre-loaded and fine-tuned :class:`CosmosBackbone`.
        decoder: Trained :class:`DiTActionDecoder`.
        tau_v: Video noise level at inference time.  ``1.0`` (pure noise)
            requires only *one* backbone forward pass — the fastest setting.
        video_denoise_steps: Steps used to partially denoise future frames.
            Ignored when ``tau_v >= 1.0``.
        action_denoise_steps: ODE steps for action denoising (default 10).
        sampler: ODE integration method (default: Euler).
        action_mean / action_std: Normalisation constants for de-normalising
            raw decoder output back to robot units.
        num_cond_frames: Number of conditioned (past) latent frames.
        num_pred_frames: Number of predicted (future) latent frames.
        device: Compute device.
    """

    def __init__(
        self,
        backbone: CosmosBackbone,
        decoder: DiTActionDecoder,
        tau_v: float = 1.0,
        video_denoise_steps: int = 0,
        action_denoise_steps: int = 10,
        sampler: Optional[ODESampler] = None,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
        t5_embedding: Optional[torch.Tensor] = None,
        num_cond_frames: int = 2,
        num_pred_frames: int = 3,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.tau_v = tau_v
        self.video_denoise_steps = video_denoise_steps
        self.action_denoise_steps = action_denoise_steps
        self.sampler: ODESampler = sampler or EulerSampler()
        self.num_cond_frames = num_cond_frames
        self.num_pred_frames = num_pred_frames
        self.t5_embedding = t5_embedding
        self.device = device

        if action_mean is not None:
            self.register_buffer("action_mean", action_mean)
            self.register_buffer("action_std", action_std)
        else:
            self.action_mean = None
            self.action_std = None

    # ---------------------------------------------------------------- #
    # RobotPolicy Protocol
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def predict(self, obs: Observation) -> ActionChunk:
        """Predict the next action chunk from *obs*.

        Satisfies :class:`~mimic_videos.core.protocol.RobotPolicy`.

        Args:
            obs: Current observation (video + proprio + optional text embed).

        Returns:
            :class:`~mimic_videos.core.protocol.ActionChunk` (denormalized).
        """
        obs = obs.to(self.device)
        t5_emb = obs.language_embedding or self.t5_embedding
        if t5_emb is None:
            raise ValueError("No language embedding provided in Observation or at construction.")
        t5_emb = t5_emb.to(self.device)

        # 1. VAE encode — video: [1, T, C, H, W] → [1, C, T_lat, H_lat, W_lat]
        B = 1
        video_bcthw = obs.video.unsqueeze(0).permute(0, 2, 1, 3, 4)
        self.backbone.restore_auxiliary(self.device)
        z_all = self.backbone.encode_frames(video_bcthw)
        self.backbone.offload_auxiliary("cpu")

        z_cond = z_all[:, :, : self.num_cond_frames]
        _, C_lat, _, H_lat, W_lat = z_all.shape
        T_lat_total = z_all.shape[2]

        # 2. Prepare future latents
        if self.tau_v >= 1.0:
            z_future = torch.randn(B, C_lat, self.num_pred_frames, H_lat, W_lat, device=self.device)
            current_tau_v = torch.ones(B, device=self.device)
        elif self.video_denoise_steps > 0:
            z_future = torch.randn(B, C_lat, self.num_pred_frames, H_lat, W_lat, device=self.device)
            dt = (self.tau_v - 1.0) / self.video_denoise_steps
            tau = 1.0
            for _ in range(self.video_denoise_steps):
                tau_t = torch.full((B,), tau, device=self.device)
                raw, _ = self.backbone.forward_transformer(z_future, z_cond, tau_t, t5_emb)
                z_future = z_future + raw[:, :, self.num_cond_frames:] * dt
                tau += dt
            current_tau_v = torch.full((B,), self.tau_v, device=self.device)
        else:
            z_future = z_all[:, :, self.num_cond_frames:]
            current_tau_v = torch.zeros(B, device=self.device)

        # 3. Backbone forward to extract features
        self.backbone.forward_transformer(z_future, z_cond, current_tau_v, t5_emb.to(self.backbone.dtype))
        video_feats = self.backbone.features.get_primary().float()  # [1, S, D]

        # 4. Denoise action chunk from τ=1 → 0
        a_noise = torch.randn(B, self.decoder.chunk_size, self.decoder.action_dim, device=self.device)

        def _vel_fn(a_t: torch.Tensor, tau: float) -> torch.Tensor:
            tau_a = torch.full((B,), tau, device=self.device)
            return self.decoder(a_t, obs.proprio.unsqueeze(0), video_feats, tau_a, current_tau_v, apply_proprio_mask=False)

        a_clean = self.sampler.integrate(_vel_fn, a_noise, self.action_denoise_steps)

        # 5. Denormalise
        a_clean = self._denormalise(a_clean.squeeze(0))

        return ActionChunk(actions=a_clean, tau_v=self.tau_v)

    def reset(self) -> None:
        """Clear any cached episode state."""
        self.backbone.features.clear()

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #

    def _denormalise(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_mean is None:
            return actions
        return actions * self.action_std.to(actions.device) + self.action_mean.to(actions.device)
