"""CosmosBackbone — wraps the Cosmos-Predict2 video diffusion model.

Responsibilities
----------------
- Loads the Cosmos2VideoToWorldPipeline, extracts its components
  (transformer, VAE, T5 text encoder).
- Applies LoRA to the transformer using the PEFT library.
- Delegates hidden-state extraction to a companion :class:`FeatureExtractor`.
- Exposes encode / decode / forward_transformer primitives used by
  both training stages and inference.

Registration
------------
Registered in the global :class:`~mimic_videos.core.registry.Registry` under the
name ``"cosmos_lora"`` so it can be instantiated from YAML config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from mimic_videos.core.registry import Registry
from .feature_extractor import FeatureExtractor, PoolingMode


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation of the Cosmos transformer.

    Attributes:
        rank: LoRA rank (r).
        alpha: LoRA alpha (scaling factor = alpha / rank).
        target_modules: Transformer submodule names to inject LoRA into.
        dropout: LoRA dropout probability during training.
    """

    rank: int = 16
    alpha: int = 16
    target_modules: List[str] = field(
        default_factory=lambda: [
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
            "ff.net.0.proj", "ff.net.2",
        ]
    )
    dropout: float = 0.0


@Registry.backbone("cosmos_lora")
class CosmosBackbone(nn.Module):
    """Video diffusion backbone based on Cosmos-Predict2 with LoRA.

    Args:
        model_id: HuggingFace model ID for Cosmos-Predict2.
        lora: LoRA configuration.
        feature_layer_indices: Transformer block indices where hidden
            states are extracted for the action decoder.
        pooling_mode: How to aggregate spatial tokens — passed directly
            to :class:`FeatureExtractor` (no silent defaults).
        num_latent_frames: Total latent frames fed to the transformer
            (conditioning + prediction). Used by the feature extractor.
        dtype: Mixed-precision dtype for the transformer.
        device: Primary device.
    """

    def __init__(
        self,
        model_id: str = "nvidia/Cosmos-Predict2-2B-Video2World",
        lora: Optional[LoRAConfig] = None,
        feature_layer_indices: List[int] = None,
        pooling_mode: PoolingMode = PoolingMode.ALL_TOKENS,
        num_latent_frames: int = 5,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.model_id = model_id
        self.dtype = dtype
        self._device = device
        self.num_latent_frames = num_latent_frames

        lora = lora or LoRAConfig()
        feature_layer_indices = feature_layer_indices or [19]

        # -------- Load pipeline -------- #
        self._load_components(model_id, dtype)

        # -------- Derived dimensions -------- #
        self.hidden_dim: int = (
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )
        self.patch_size: int = self.transformer.config.patch_size

        # -------- Apply LoRA -------- #
        self._apply_lora(lora)

        # -------- Feature extractor (decoupled) -------- #
        self.features = FeatureExtractor(
            transformer=self.transformer,
            layer_indices=feature_layer_indices,
            pooling_mode=pooling_mode,
            num_latent_frames=num_latent_frames,
        )

        # -------- VAE normalization constants -------- #
        self._register_vae_norm_buffers()

    # ---------------------------------------------------------------- #
    # Loading helpers
    # ---------------------------------------------------------------- #

    def _load_components(self, model_id: str, dtype: torch.dtype) -> None:
        """Load the Cosmos pipeline and detach its sub-components."""
        import diffusers.pipelines.cosmos.pipeline_cosmos2_video2world as _mod

        # Bypass the gated safety checker
        _mod.CosmosSafetyChecker = lambda *a, **kw: None

        from diffusers import Cosmos2VideoToWorldPipeline

        pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None
        )
        # Cosmos imports re-disable grad globally — restore it
        torch.set_grad_enabled(True)

        self.transformer = pipeline.transformer
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.scheduler = pipeline.scheduler
        self.vae_scale_factor_temporal = pipeline.vae_scale_factor_temporal
        self.vae_scale_factor_spatial = pipeline.vae_scale_factor_spatial
        del pipeline

    def _apply_lora(self, cfg: LoRAConfig) -> None:
        from peft import LoraConfig as PeftLoraConfig, get_peft_model

        peft_cfg = PeftLoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.dropout,
            bias="none",
        )
        self.transformer = get_peft_model(self.transformer, peft_cfg)

        # Re-attach feature extractor hooks after PEFT wrapping
        if hasattr(self, "features"):
            self.features._attach_hooks(self.transformer)

    def _register_vae_norm_buffers(self) -> None:
        z_dim = self.vae.config.z_dim
        self.register_buffer(
            "latents_mean",
            torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1),
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1),
        )
        self.sigma_data: float = self.scheduler.config.sigma_data

    # ---------------------------------------------------------------- #
    # VAE encode / decode
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def encode_frames(self, pixel_frames: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames ``[B, C, T, H, W]`` to normalised latents.

        Args:
            pixel_frames: ``[B, C, T, H, W]`` in ``[-1, 1]``.

        Returns:
            Normalised latents ``[B, z_dim, T_lat, H_lat, W_lat]``.
        """
        vae_device = next(self.vae.parameters()).device
        frames = pixel_frames.to(vae_device, self.vae.dtype)
        posterior = self.vae.encode(frames).latent_dist
        latents = posterior.mode().float()
        return (
            (latents - self.latents_mean.to(latents.device))
            / self.latents_std.to(latents.device)
            * self.sigma_data
        )

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalised latents back to pixel frames.

        Returns:
            ``[B, C, T, H, W]`` in ``[-1, 1]``.
        """
        vae_device = next(self.vae.parameters()).device
        latents = (
            latents * self.latents_std.to(latents.device) / self.sigma_data
            + self.latents_mean.to(latents.device)
        )
        latents = latents.to(vae_device, self.vae.dtype)
        return self.vae.decode(latents, return_dict=False)[0]

    # ---------------------------------------------------------------- #
    # Text encoding
    # ---------------------------------------------------------------- #

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Return T5 embeddings for *prompt*, shape ``[1, seq_len, D]``."""
        device = next(self.text_encoder.parameters()).device
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        return self.text_encoder(
            input_ids=tokens.input_ids.to(device),
            attention_mask=tokens.attention_mask.to(device),
        )[0]

    # ---------------------------------------------------------------- #
    # Transformer forward
    # ---------------------------------------------------------------- #

    def forward_transformer(
        self,
        z_noisy: torch.Tensor,
        z_cond: torch.Tensor,
        tau_v: torch.Tensor,
        text_embeds: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the Cosmos DiT for one flow-matching step.

        Constructs the per-frame timestep tensor and condition mask expected
        by the Cosmos API, runs the transformer, and applies the Cosmos
        preconditioning (c_skip, c_out).

        Args:
            z_noisy: Noisy prediction latents ``[B, C, T_pred, H_lat, W_lat]``.
            z_cond: Clean conditioning latents ``[B, C, T_cond, H_lat, W_lat]``.
            tau_v: Video flow time ``[B]`` in ``[0, 1]``.
            text_embeds: T5 embeddings ``[B, seq_len, D]``.
            condition_mask: Optional ``[B, 1, T_total, H, W]`` (built automatically
                if not supplied).

        Returns:
            Tuple of:
            - ``raw_output``: raw transformer output (velocity) ``[B, C, T_total, H, W]``.
            - ``denoised``: preconditioning-applied output for sampling ``[B, C, T_total, H, W]``.
        """
        B, C, T_cond, H, W = z_cond.shape
        T_pred = z_noisy.shape[2]
        T_total = T_cond + T_pred
        device = z_noisy.device

        # --- Build per-frame timesteps ---
        sigma_cond = 0.0001
        t_cond_scalar = sigma_cond / (sigma_cond + 1.0)
        timesteps = torch.zeros(B, 1, T_total, 1, 1, device=device, dtype=z_noisy.dtype)
        timesteps[:, :, :T_cond] = t_cond_scalar
        for b in range(B):
            t_val = tau_v[b].item() if tau_v.ndim > 0 else tau_v.item()
            timesteps[b, :, T_cond:] = t_val

        # --- Scale inputs (Cosmos preconditioning: c_in = 1 - t) ---
        c_in_cond = 1.0 - t_cond_scalar
        c_in_pred = (1.0 - tau_v).view(B, 1, 1, 1, 1)

        z_in = torch.cat(
            [z_cond * c_in_cond, z_noisy * c_in_pred], dim=2
        )  # [B, C, T_total, H, W]

        # --- Condition mask (1 = conditioning frame, 0 = prediction frame) ---
        if condition_mask is None:
            condition_mask = torch.zeros(B, 1, T_total, H, W, device=device, dtype=z_noisy.dtype)
            condition_mask[:, :, :T_cond] = 1.0

        padding_mask = torch.zeros(1, 1, H, W, device=device, dtype=z_noisy.dtype)

        # --- Clear cached states before the forward pass ---
        self.features.clear()

        raw_output = self.transformer(
            hidden_states=z_in.to(self.dtype),
            timestep=timesteps.to(self.dtype),
            encoder_hidden_states=text_embeds.to(self.dtype),
            condition_mask=condition_mask.to(self.dtype),
            padding_mask=padding_mask.to(self.dtype),
            return_dict=False,
        )[0]

        # --- Apply Cosmos output preconditioning: c_skip * input + c_out * net_out ---
        denoised = torch.zeros_like(z_in, dtype=torch.float32)

        c_skip_cond = 1.0 - t_cond_scalar
        c_out_cond = -t_cond_scalar
        denoised[:, :, :T_cond] = (
            c_skip_cond * z_in[:, :, :T_cond].float()
            + c_out_cond * raw_output[:, :, :T_cond].float()
        )
        # Replace conditioning frames with original clean latents
        denoised[:, :, :T_cond] = z_cond.float()

        c_skip_pred = (1.0 - tau_v).view(B, 1, 1, 1, 1)
        c_out_pred = -tau_v.view(B, 1, 1, 1, 1)
        denoised[:, :, T_cond:] = (
            c_skip_pred * z_noisy.float()
            + c_out_pred * raw_output[:, :, T_cond:].float()
        )

        return raw_output.float(), denoised

    # ---------------------------------------------------------------- #
    # Freeze / serialise helpers
    # ---------------------------------------------------------------- #

    def freeze(self) -> None:
        """Freeze all parameters (used in Stage 2 training)."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_lora(self) -> None:
        """Unfreeze only the LoRA adapter weights."""
        for name, p in self.transformer.named_parameters():
            if "lora_" in name:
                p.requires_grad = True

    def save_adapter(self, path: str) -> None:
        """Persist LoRA adapter weights to *path*."""
        self.transformer.save_pretrained(path)

    def load_adapter(self, path: str) -> None:
        """Load LoRA adapter weights from *path* and re-attach hooks."""
        from peft import PeftModel

        if hasattr(self.transformer, "load_adapter"):
            self.transformer.load_adapter(path, adapter_name="default")
        else:
            self.transformer = PeftModel.from_pretrained(self.transformer, path)

        # Re-attach hooks after potential re-wrapping
        self.features._attach_hooks(self.transformer)

    def offload_auxiliary(self, device: str = "cpu") -> None:
        """Move VAE and text encoder to *device* to free GPU memory."""
        self.vae.to(device)
        if self.text_encoder is not None:
            self.text_encoder.to(device)

    def restore_auxiliary(self, device: str = "cuda") -> None:
        """Move VAE and text encoder back to *device*."""
        self.vae.to(device)
        if self.text_encoder is not None:
            self.text_encoder.to(device)
