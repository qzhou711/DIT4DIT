"""
Profile Cosmos Video DiT: self-attention vs cross-attention vs FFN time ratio.

Uses CUDA events on each transformer block's attn1, attn2, and ff (diffusers CosmosTransformerBlock).
Helps decide whether to prioritize attention kernels (flash/sdpa) vs FFN (width, fusion).

Example:
    python scripts/profile_video_dit_attn_ffn.py --suite libero_object \\
        --stage1_checkpoint checkpoints/libero_object/stage1/step_8000 \\
        --cosmos_model_id checkpoints/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/<hash> \\
        --warmup 3 --repeats 20

    # No suite: uses default DataConfig latent sizes
    python scripts/profile_video_dit_attn_ffn.py --repeats 10
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import statistics
import time

import torch


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_inner_transformer(backbone: torch.nn.Module):
    """Unwrap PEFT to CosmosTransformer3DModel."""
    t = backbone.transformer
    if hasattr(t, "module"):
        t = t.module
    if hasattr(t, "base_model"):
        t = t.base_model.model
    return t


@dataclass
class HookState:
    """Mutable bucket for one forward pass (reset before each forward)."""

    attn1_ms: float = 0.0
    attn2_ms: float = 0.0
    ff_ms: float = 0.0


def _register_block_hooks(
    transformer: torch.nn.Module,
    state: HookState,
) -> Callable[[], None]:
    """Register pre/post hooks on every block's attn1, attn2, ff. Returns remove_all()."""
    blocks = transformer.transformer_blocks
    _starts: Dict[int, torch.cuda.Event] = {}
    handles: List = []

    def pre_hook(module, inp):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        _starts[id(module)] = ev

    def make_post(add_to: str):
        def post(module, inp, out):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            ms = _starts[id(module)].elapsed_time(end)
            if add_to == "attn1":
                state.attn1_ms += ms
            elif add_to == "attn2":
                state.attn2_ms += ms
            elif add_to == "ff":
                state.ff_ms += ms

        return post

    for block in blocks:
        for name, key in (("attn1", "attn1"), ("attn2", "attn2"), ("ff", "ff")):
            mod = getattr(block, name, None)
            if mod is None:
                raise RuntimeError(f"Block missing {name}: {type(block)}")
            handles.append(mod.register_forward_pre_hook(pre_hook))
            handles.append(mod.register_forward_hook(make_post(key)))

    def remove_all():
        for h in handles:
            h.remove()

    return remove_all


def parse_args():
    from configs.config import LIBERO_SUITES

    p = argparse.ArgumentParser(
        description="Profile Video DiT: attn1 vs attn2 vs FFN (Cosmos transformer blocks)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--suite", type=str, default=None, choices=list(LIBERO_SUITES.keys()))
    p.add_argument("--cosmos_model_id", default="nvidia/Cosmos-Predict2-2B-Video2World")
    p.add_argument("--stage1_checkpoint", default=None)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for meaningful submodule timing.")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    from configs.config import DataConfig, ModelConfig, get_suite_data_config
    from mimic_video.models.video_backbone import CosmosVideoBackbone

    if args.suite:
        dcfg = get_suite_data_config(args.suite)
    else:
        dcfg = DataConfig()
    mcfg = ModelConfig()

    print("Loading CosmosVideoBackbone …")
    backbone = CosmosVideoBackbone(
        model_id=args.cosmos_model_id,
        lora_rank=mcfg.lora_rank,
        lora_alpha=mcfg.lora_alpha,
        lora_target_modules=mcfg.lora_target_modules,
        hidden_state_layer=mcfg.hidden_state_layer,
        dtype=dtype,
        device=str(device),
    )
    stage1 = args.stage1_checkpoint
    if stage1 is None and args.suite:
        stage1 = f"checkpoints/{args.suite}/stage1/final"
    if stage1 and os.path.isdir(stage1):
        print(f"  Loading LoRA: {stage1}")
        backbone.load_lora(stage1)

    backbone.to(device)
    backbone.eval()
    backbone.freeze_for_stage2()

    inner = get_inner_transformer(backbone)
    n_blocks = len(inner.transformer_blocks)
    print(f"  Transformer blocks: {n_blocks} (CosmosTransformerBlock: attn1=self, attn2=cross, ff=FeedForward)")

    B = 1
    with torch.no_grad():
        t5_emb = backbone.encode_text("a person doing an action")
        t5_emb = t5_emb.to(device=device, dtype=dtype)

    C_lat = backbone.vae.config.z_dim if hasattr(backbone.vae, "config") else 16
    T_cond = dcfg.num_cond_latent_frames
    T_pred = dcfg.num_pred_latent_frames
    vae_sf_sp = getattr(backbone, "vae_scale_factor_spatial", 8)
    H_lat = dcfg.camera_height // vae_sf_sp // 2
    W_lat = dcfg.camera_width // vae_sf_sp // 2

    z_cond = torch.randn(B, C_lat, T_cond, H_lat, W_lat, device=device, dtype=dtype)
    z_noisy = torch.randn(B, C_lat, T_pred, H_lat, W_lat, device=device, dtype=dtype)
    tau_v = torch.ones(B, device=device, dtype=dtype)

    backbone.offload_vae_and_text_encoder("cpu")

    # --- hooks: one mutable state per forward ---
    hook_state = HookState()
    remove_hooks = _register_block_hooks(inner, hook_state)

    rows_attn1: List[float] = []
    rows_attn2: List[float] = []
    rows_ff: List[float] = []
    rows_total: List[float] = []

    def one_forward():
        hook_state.attn1_ms = 0.0
        hook_state.attn2_ms = 0.0
        hook_state.ff_ms = 0.0
        cuda_sync()
        t0 = time.perf_counter()
        backbone.forward_transformer(
            z_noisy=z_noisy,
            z_cond=z_cond,
            tau_v=tau_v,
            encoder_hidden_states=t5_emb,
        )
        cuda_sync()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return wall_ms

    print(f"\nWarmup {args.warmup} …")
    with torch.no_grad():
        for _ in range(args.warmup):
            one_forward()

    print(f"Timed repeats: {args.repeats}\n")
    with torch.no_grad():
        for _ in range(args.repeats):
            wall_ms = one_forward()
            rows_attn1.append(hook_state.attn1_ms)
            rows_attn2.append(hook_state.attn2_ms)
            rows_ff.append(hook_state.ff_ms)
            rows_total.append(wall_ms)

    remove_hooks()

    def mean(xs):
        return statistics.mean(xs) if xs else 0.0

    a1, a2, ff = mean(rows_attn1), mean(rows_attn2), mean(rows_ff)
    att_sum = a1 + a2
    core = att_sum + ff
    tot = mean(rows_total)

    print("  (All ms: mean over repeats; attn* = sum over all transformer blocks per forward)\n")
    print(f"  {'Component':<28s}  {'mean_ms':>10s}  {'share_of_attn+ff':>18s}")
    print("  " + "-" * 62)
    print(f"  {'Self-attn (attn1)':<28s}  {a1:10.2f}  {a1 / core * 100:16.1f}%")
    print(f"  {'Cross-attn (attn2)':<28s}  {a2:10.2f}  {a2 / core * 100:16.1f}%")
    print(f"  {'FFN (ff)':<28s}  {ff:10.2f}  {ff / core * 100:16.1f}%")
    print("  " + "-" * 62)
    print(f"  {'attn1+attn2+ff (hooked)':<28s}  {core:10.2f}  {'100.0%':>18s}")
    print(f"  {'forward_transformer (wall)':<28s}  {tot:10.2f}  {'(incl. patch/time/rope/norm/out)':>18s}")

    print("\n  --- Ratios (for acceleration direction) ---")
    if ff > 1e-6:
        print(f"  (attn1 + attn2) / FFN     = {att_sum / ff:.3f}")
    if a1 > 1e-6:
        print(f"  cross-attn / self-attn    = {a2 / a1:.3f}")
    if att_sum > 1e-6:
        print(f"  FFN / (all attention)     = {ff / att_sum:.3f}")
    rest = tot - core
    if tot > 1e-6:
        print(f"  unhooked overhead / wall   ≈ {rest / tot * 100:.1f}%  ({rest:.2f} ms)")

    print(
        "\n  Interpretation: if (attn1+attn2)/FFN >> 1, attention-dominated → flash/sdpa/sparsity; "
        "if FFN/(attn) high → consider FFN fusion, width, or fewer layers."
    )


if __name__ == "__main__":
    main()
