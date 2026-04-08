"""
Latency Benchmark for mimic-video Inference Pipeline
=====================================================

测量推理管线各阶段墙钟时间（perf_counter + CUDA sync）：
  - 默认端到端：每个 repeat 含 VAE 编码（像素→latent）→ Video DiT → h_video 池化 → Action DiT（含单步前向与 Euler ODE）。
  - dry_run：不加载权重，用随机张量代理各算子。

──────────────────────────────────────────────────────────────────────────────
运行模式
──────────────────────────────────────────────────────────────────────────────
  --dry_run
      不加载 Cosmos / decoder；仅随机张量形状与轻量代理网络，用于无 GPU/无权重时快速自测。

  真实模型（不加 --dry_run）
      加载 CosmosVideoBackbone（可选 Stage-1 LoRA）与 ActionDecoderDiT（若存在则加载
      <stage2>/action_decoder.pt，否则随机初始化）。
      文本条件：无 --use_dataset 时用 backbone.encode_text；有 --use_dataset 时用预计算 T5。

  --suite <name>
      name ∈ libero_spatial | libero_object | libero_goal | libero_10。
      自动设置 DataConfig（repo_id、precomputed 等）及默认 checkpoint 路径：
        Stage-1: checkpoints/<suite>/stage1/final
        Stage-2: checkpoints/<suite>/stage2/final
        预计算:  precomputed/<suite>/
      未指定 --suite 时等价于默认 DataConfig，Stage 默认路径为 checkpoints/stage1|stage2/final。

  VAE 是否计入「每次 repeat」计时（端到端）
      默认：计入。每个 repeat 先 encode_video，再 forward_transformer。
      --exclude_vae_from_timing：不计入每次 repeat。
        · 有 --use_dataset：仅首次 VAE 编码，之后固定 latent 测 DiT+decoder；并打印一次参考耗时。
        · 无 --use_dataset：用随机 latent，不跑 VAE。

  --use_dataset
      从 LeRobot 取一条训练样本；需 dcfg.repo_id（通常配合 --suite）。
      需 precomputed_dir 下 t5_embeddings.pt 或 t5_embedding.pt；建议有 action_stats.pt。
      --dataset_index：样本在数据集中的下标（默认 0）。

──────────────────────────────────────────────────────────────────────────────
命令示例
──────────────────────────────────────────────────────────────────────────────
  # 代理张量基准（无需权重）
  python scripts/benchmark_latency.py --dry_run

  # 全模型 + 默认端到端（随机 RGB 视频，每 repeat 含 VAE）
  python scripts/benchmark_latency.py --device cuda --warmup 3 --repeats 10

  # 指定 suite 与本地 Cosmos 快照（与训练一致时常用）
  python scripts/benchmark_latency.py --suite libero_object \\
      --cosmos_model_id /path/to/Cosmos-Predict2-2B-Video2World/snapshots/<hash>

  # 真实数据一条 + 预计算 T5（每 repeat 含 VAE，端到端）
  python scripts/benchmark_latency.py --suite libero_object --use_dataset --dataset_index 0

  # 只测 DiT+池化+Action ODE，不含每轮 VAE（数据集仅编码一次）
  python scripts/benchmark_latency.py --suite libero_object --use_dataset --exclude_vae_from_timing

  # 对比 h_video 池化方式（分两次跑）
  python scripts/benchmark_latency.py --pool_mode mean  --repeats 10
  python scripts/benchmark_latency.py --pool_mode none  --repeats 10


python scripts/benchmark_latency.py \
  --suite libero_object \
  --stage1_checkpoint checkpoints/libero_object/stage1/step_8000 \
  --stage2_checkpoint checkpoints/libero_object/stage2/step_26000 \
  --cosmos_model_id checkpoints/models--nvidia--Cosmos-Predict2-2B-Video2World/snapshots/f50c09f5d8ab133a90cac3f4886a6471e9ba3f18 \
  --use_dataset \
  --pool_mode none \
  --dataset_index 0 \
  --device cuda \
  --warmup 3 \
  --repeats 10



──────────────────────────────────────────────────────────────────────────────
全部参数说明
──────────────────────────────────────────────────────────────────────────────
  --device              运行设备：cuda | cpu（无 CUDA 时自动退回 cpu）[默认: cuda]
  --dtype               bf16 | fp32 [默认: bf16]
  --batch_size          批次大小 B [默认: 1]
  --warmup              预热轮数（不计入统计）[默认: 3]
  --repeats             正式计时的重复次数 [默认: 10]
  --action_steps        Action 流匹配 Euler 步数 [默认: 10]
  --pool_mode           Video hidden 池化供 Action DiT：mean | none [默认: mean]
  --dry_run             启用代理基准，不加载真实模型
  --cosmos_model_id     HuggingFace ID 或本地 Cosmos 目录 [默认: nvidia/Cosmos-Predict2-2B-Video2World]
  --suite               LIBERO 套件名，见上文（可选）
  --stage1_checkpoint   Stage-1 LoRA 目录；默认见 --suite 说明（可选）
  --stage2_checkpoint   Stage-2 目录，需含 action_decoder.pt；默认见 --suite（可选）
  --precomputed_dir     覆盖预计算根目录（T5、action_stats 等）；suite 下常为 precomputed/<suite>/
  --use_dataset         使用一条 LeRobot 真实样本 + 预计算 T5
  --dataset_index       --use_dataset 时使用的样本索引 [默认: 0]
  --exclude_vae_from_timing
                        不把 VAE 纳入每次 repeat；行为见上文「VAE 是否计入」

──────────────────────────────────────────────────────────────────────────────
输出表头含义
──────────────────────────────────────────────────────────────────────────────
  默认会打印：VAE encode（若计入）、Video-DiT forward、Action 单步前向、Action ODE（N 步），
  以及 TOTAL：VAE+DiT+pool+ODE 或 DiT+pool+ODE（与是否含 VAE 一致）。
  pool 行参与 TOTAL，与 --pool_mode 一致（mean 或 none）。
"""

import argparse
import sys
import os
import time
import statistics
from typing import Optional

# Ensure project root (parent of scripts/) is on sys.path so that
# `configs` and `mimic_video` packages can be found regardless of CWD.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Context-manager timer that records elapsed seconds."""

    def __enter__(self):
        cuda_sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        cuda_sync()
        self.elapsed = time.perf_counter() - self._start


def stats(times: list) -> dict:
    return {
        "mean_ms":   statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "min_ms":    min(times) * 1000,
        "max_ms":    max(times) * 1000,
        "std_ms":    (statistics.stdev(times) * 1000) if len(times) > 1 else 0.0,
    }


def print_stats(label: str, times: list):
    s = stats(times)
    print(
        f"  {label:<40s}  "
        f"mean={s['mean_ms']:7.2f} ms  "
        f"median={s['median_ms']:7.2f} ms  "
        f"min={s['min_ms']:7.2f} ms  "
        f"max={s['max_ms']:7.2f} ms  "
        f"std={s['std_ms']:6.2f} ms"
    )


def load_stage2_action_decoder(
    stage2_path: str,
    device: torch.device,
    dtype: torch.dtype,
    backbone_hidden_dim: int,
    mcfg,
    dcfg,
):
    """Load ActionDecoderDiT from checkpoints/.../stage2/.../action_decoder.pt if present."""
    from mimic_video.models.action_decoder import ActionDecoderDiT

    decoder_path = os.path.join(stage2_path, "action_decoder.pt")
    if not os.path.isfile(decoder_path):
        print(f"    → No {decoder_path} — using freshly initialized ActionDecoderDiT")
        dec = ActionDecoderDiT(
            action_dim=dcfg.action_dim,
            proprio_dim=dcfg.proprio_dim,
            hidden_dim=mcfg.decoder_hidden_dim,
            num_layers=mcfg.decoder_num_layers,
            num_heads=mcfg.decoder_num_heads,
            mlp_ratio=mcfg.decoder_mlp_ratio,
            backbone_hidden_dim=backbone_hidden_dim,
            action_chunk_size=dcfg.action_chunk_size,
            proprio_mask_prob=0.0,
        ).to(device=device, dtype=dtype)
        dec.eval()
        return dec, False

    print(f"    → Loading action decoder from {decoder_path}")
    state_dict = torch.load(decoder_path, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    inferred_hidden_dim = state_dict["blocks.0.self_attn_q.weight"].shape[0]
    inferred_layers = 1 + max(
        int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("blocks.")
    )
    inferred_heads = inferred_hidden_dim // 64

    dec = ActionDecoderDiT(
        action_dim=dcfg.action_dim,
        proprio_dim=dcfg.proprio_dim,
        hidden_dim=inferred_hidden_dim,
        num_layers=inferred_layers,
        num_heads=inferred_heads,
        mlp_ratio=mcfg.decoder_mlp_ratio,
        backbone_hidden_dim=backbone_hidden_dim,
        action_chunk_size=dcfg.action_chunk_size,
        proprio_mask_prob=0.0,
    )
    dec.load_state_dict(state_dict)
    dec.to(device=device, dtype=dtype)
    dec.eval()
    print(f"       (inferred dim={inferred_hidden_dim}, layers={inferred_layers}, heads={inferred_heads})")
    return dec, True


def _sample_to_batch_tensors(sample: dict, B: int, device: torch.device, dtype: torch.dtype):
    """Turn one MimicVideoDataset item into batched tensors."""
    video = sample["video"]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if B > 1:
        video = video.expand(B, -1, -1, -1, -1)

    proprio = sample["proprio"]
    if proprio.ndim == 1:
        proprio = proprio.unsqueeze(0)
    if B > 1:
        proprio = proprio.expand(B, -1)
    proprio = proprio.to(device=device, dtype=dtype)

    if "t5_embedding" not in sample:
        raise ValueError(
            "Dataset sample has no t5_embedding. Run precompute_embeddings.py and set --precomputed_dir."
        )
    t5 = sample["t5_embedding"]
    if t5.ndim == 2:
        t5 = t5.unsqueeze(0)
    if B > 1:
        t5 = t5.expand(B, -1, -1)
    t5 = t5.to(device=device, dtype=dtype)
    return video, proprio, t5


# ─────────────────────────────────────────────────────────────────────────────
# Dry-run benchmark (no model loading — synthetic random tensors)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_dry_run(args):
    """Use random tensors that mimic real tensor shapes; no model weights."""
    print("\n" + "=" * 70)
    print("DRY-RUN MODE — using random tensors (no model loaded)")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    B               = args.batch_size
    C_lat           = 16          # Cosmos VAE latent channels
    T_lat           = 5           # latent frames (2 cond + 3 pred)
    H_lat, W_lat    = 16, 32      # spatial latent dims for 256×256 with 2-cam concat
    backbone_hidden = 2048        # Cosmos 2B hidden dim
    patch_h, patch_w = 1, 1      # effective patch after patchify (already factored into H_lat*W_lat)
    action_dim      = 7
    action_chunk    = 16
    proprio_dim     = 8
    decoder_hidden  = 512
    decoder_layers  = 8
    decoder_heads   = 8

    # Simulated raw hidden states [B, T*H'*W', backbone_hidden]
    THW  = T_lat * H_lat * W_lat  # ~12 000 for 5×30×80
    h_raw = torch.randn(B, THW, backbone_hidden, device=device, dtype=dtype)

    print(f"\n  Tensor shapes used:")
    print(f"    h_video (raw) :  {list(h_raw.shape)}  ({THW} tokens)")
    print(f"    Action chunk  :  [{B}, {action_chunk}, {action_dim}]")

    # ── 1. Simulate Video DiT forward (just a Linear + LayerNorm as proxy)
    proxy_linear = torch.nn.Linear(backbone_hidden, backbone_hidden).to(device=device, dtype=dtype)

    times_vdit = []
    for i in range(args.warmup + args.repeats):
        x_in = torch.randn_like(h_raw)
        with Timer() as t:
            _ = proxy_linear(x_in)
        if i >= args.warmup:
            times_vdit.append(t.elapsed)

    # ── 2. h_video pooling: spatial mean pool → [B, T_lat, D]
    times_pool_mean = []
    times_pool_none = []

    for i in range(args.warmup + args.repeats):
        h = torch.randn_like(h_raw)
        # mean pool
        with Timer() as t:
            h_bthw = h.view(B, T_lat, H_lat * W_lat, backbone_hidden)
            _ = h_bthw.mean(dim=2)  # [B, T, D]
        if i >= args.warmup:
            times_pool_mean.append(t.elapsed)

        # none (identity — just a view)
        with Timer() as t:
            _ = h.clone()  # simulate copy / no-op pass-through
        if i >= args.warmup:
            times_pool_none.append(t.elapsed)

    # ── 3. Action DiT forward (proxy: single MHA + linear)
    import torch.nn as nn
    proxy_attn  = nn.MultiheadAttention(decoder_hidden, decoder_heads, batch_first=True).to(device=device, dtype=dtype)
    proxy_proj  = nn.Linear(decoder_hidden, action_dim).to(device=device, dtype=dtype)

    times_action = []
    for i in range(args.warmup + args.repeats):
        a = torch.randn(B, action_chunk, decoder_hidden, device=device, dtype=dtype)
        with Timer() as t:
            for _ in range(args.action_steps):
                a, _ = proxy_attn(a, a, a)
            _ = proxy_proj(a)
        if i >= args.warmup:
            times_action.append(t.elapsed)

    # ── Results
    print(f"\n  Results (warmup={args.warmup}, repeats={args.repeats}):\n")
    print_stats("Video-DiT forward (proxy linear)",   times_vdit)
    print_stats("h_video pool → mean [B,T,D]",        times_pool_mean)
    print_stats("h_video pool → none (identity)",      times_pool_none)
    print_stats(f"Action-DiT ({args.action_steps} Euler steps, proxy)", times_action)


# ─────────────────────────────────────────────────────────────────────────────
# Full benchmark with real model weights
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_real(args, dcfg, mcfg, stage1_path: Optional[str], stage2_path: str):
    print("\n" + "=" * 70)
    print("REAL MODEL MODE — loading Cosmos + ActionDecoderDiT")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # ── Load backbone ──────────────────────────────────────────────────────
    print("\n[1/2] Loading CosmosVideoBackbone …")
    from mimic_video.models.video_backbone import CosmosVideoBackbone

    backbone = CosmosVideoBackbone(
        model_id            = args.cosmos_model_id,
        lora_rank           = mcfg.lora_rank,
        lora_alpha          = mcfg.lora_alpha,
        lora_target_modules = mcfg.lora_target_modules,
        hidden_state_layer  = mcfg.hidden_state_layer,
        dtype               = dtype,
        device              = str(device),
    )

    if stage1_path and os.path.isdir(stage1_path):
        print(f"    → Loading LoRA from {stage1_path}")
        backbone.load_lora(stage1_path)
    elif stage1_path:
        print(f"    → WARNING: Stage-1 path not found: {stage1_path} (using base backbone)")

    backbone.to(device)
    backbone.eval()
    backbone.freeze_for_stage2()  # freeze for inference; keeps hooks active

    from mimic_video.models.flow_matching import FlowMatchingScheduler

    fm = FlowMatchingScheduler()
    B = args.batch_size
    include_vae = not args.exclude_vae_from_timing
    video_bcthw = None
    eps_v = None
    tau_v = None
    vae_encode_ms = None  # informational: one-shot encode when exclude_vae + dataset
    times_vae = []
    data_source = "synthetic latents + on-the-fly T5"

    # ── Inputs: real dataset sample or synthetic pixels / latents ───────────
    if args.use_dataset:
        from mimic_video.data.dataset import MimicVideoDataset

        print("\n  Loading MimicVideoDataset for --use_dataset …")
        train_episodes = list(range(dcfg.train_episodes))
        stats_path = os.path.join(dcfg.precomputed_dir, "action_stats.pt")
        action_stats = None
        if os.path.isfile(stats_path):
            action_stats = torch.load(stats_path, map_location="cpu", weights_only=True)

        dataset = MimicVideoDataset(
            repo_id=dcfg.repo_id,
            camera_names=dcfg.camera_names,
            state_keys=dcfg.state_keys,
            action_keys=dcfg.action_keys,
            num_pixel_frames=dcfg.num_pixel_frames,
            action_chunk_size=dcfg.action_chunk_size,
            action_dim=dcfg.action_dim,
            proprio_dim=dcfg.proprio_dim,
            target_height=dcfg.camera_height,
            target_width=dcfg.camera_width,
            episode_indices=train_episodes,
            precomputed_dir=dcfg.precomputed_dir,
            action_norm_type=dcfg.action_norm_type,
            fps=dcfg.fps,
            require_action_chunk=True,
            allow_partial_action_chunk=True,
        )
        if action_stats is not None:
            dataset.action_mean = action_stats.get("mean")
            dataset.action_std = action_stats.get("std")
            dataset.action_min = action_stats.get("min")
            dataset.action_max = action_stats.get("max")

        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty — check repo_id / episode filters.")

        idx = min(max(0, args.dataset_index), len(dataset) - 1)
        sample = dataset[idx]
        video, proprio, t5_emb = _sample_to_batch_tensors(sample, B, device, dtype)
        data_source = f"dataset idx={idx} ({dcfg.repo_id}) + precomputed T5"
        video_bcthw = video.permute(0, 2, 1, 3, 4)
    else:
        with torch.no_grad():
            t5_emb = backbone.encode_text("a person doing an action")
            t5_emb = t5_emb.to(device=device, dtype=dtype)
            if B > 1 and t5_emb.shape[0] == 1:
                t5_emb = t5_emb.expand(B, -1, -1)

        proprio = torch.randn(B, dcfg.proprio_dim, device=device, dtype=dtype)
        if include_vae:
            video_bcthw = torch.randn(
                B,
                3,
                dcfg.num_pixel_frames,
                dcfg.camera_height,
                dcfg.camera_width,
                device=device,
                dtype=torch.float32,
            )
            video_bcthw = torch.tanh(video_bcthw * 0.5)
            data_source = "synthetic RGB video + on-the-fly T5"

    # ── Latents: VAE each repeat (default) vs random latents / one-shot encode ─
    if include_vae:
        if video_bcthw is None:
            raise RuntimeError("include_vae requires pixel video (use --use_dataset or enable VAE path).")
        backbone.move_vae_to(device)
        with torch.no_grad():
            z_0 = backbone.encode_video(video_bcthw)
        z_cond = z_0[:, :, : dcfg.num_cond_latent_frames]
        z_pred = z_0[:, :, dcfg.num_cond_latent_frames :]
        eps_v = torch.randn_like(z_pred)
        tau_v = torch.ones(B, device=z_pred.device, dtype=z_pred.dtype)
        z_noisy = fm.interpolate(z_pred, eps_v, tau_v)
        T_lat = z_0.shape[2]
        H_lat = z_0.shape[3]
        W_lat = z_0.shape[4]
        if args.use_dataset:
            print(f"\n  Real batch: VAE latent shape {tuple(z_0.shape)}  (VAE timed each repeat → end-to-end)")
        else:
            print(f"\n  Synthetic video → VAE latent shape {tuple(z_0.shape)}  (VAE timed each repeat → end-to-end)")
        if backbone.text_encoder is not None:
            backbone.text_encoder.to("cpu")
    elif args.use_dataset:
        backbone.move_vae_to(device)
        with torch.no_grad():
            cuda_sync()
            t0 = time.perf_counter()
            z_0 = backbone.encode_video(video_bcthw)
            cuda_sync()
            vae_encode_ms = (time.perf_counter() - t0) * 1000.0
        z_cond = z_0[:, :, : dcfg.num_cond_latent_frames]
        z_pred = z_0[:, :, dcfg.num_cond_latent_frames :]
        eps_v = torch.randn_like(z_pred)
        tau_v = torch.ones(B, device=z_pred.device, dtype=z_pred.dtype)
        z_noisy = fm.interpolate(z_pred, eps_v, tau_v)
        T_lat = z_0.shape[2]
        H_lat = z_0.shape[3]
        W_lat = z_0.shape[4]
        print(f"\n  Real batch: VAE latent shape {tuple(z_0.shape)}  (one-time encode: {vae_encode_ms:.2f} ms, not in repeat loop)")
        backbone.offload_vae_and_text_encoder("cpu")
    else:
        C_lat = backbone.vae.config.z_dim if hasattr(backbone.vae, "config") else 16
        T_lat = dcfg.num_latent_frames
        T_cond = dcfg.num_cond_latent_frames
        T_pred = dcfg.num_pred_latent_frames
        vae_sf_sp = getattr(backbone, "vae_scale_factor_spatial", 8)
        H_lat = dcfg.camera_height // vae_sf_sp // 2
        W_lat = dcfg.camera_width // vae_sf_sp // 2
        print(f"\n  Inferred latent spatial size: {H_lat}×{W_lat}  (from VAE scale_factor={vae_sf_sp})")

        z_cond = torch.randn(B, C_lat, T_cond, H_lat, W_lat, device=device, dtype=dtype)
        z_noisy = torch.randn(B, C_lat, T_pred, H_lat, W_lat, device=device, dtype=dtype)
        tau_v = torch.ones(B, device=device, dtype=dtype)
        backbone.offload_vae_and_text_encoder("cpu")

    # ── Load action decoder (Stage-2 checkpoint if available) ─────────────
    print("[2/2] Building ActionDecoderDiT …")
    action_decoder, loaded_s2 = load_stage2_action_decoder(
        stage2_path, device, dtype, backbone.hidden_dim, mcfg, dcfg,
    )

    a_noise = torch.randn(B, dcfg.action_chunk_size, dcfg.action_dim, device=device, dtype=dtype)

    # Video timestep for backbone may be float32 (latent dtype); decoder weights use --dtype (bf16/fp32).
    tau_v_action = tau_v.to(device=device, dtype=dtype)

    pool_mode = args.pool_mode

    def _latents_from_vae():
        """Re-encode pixels; reuse fixed eps_v for z_noisy (same as training at tau_v=1)."""
        with torch.no_grad():
            z_0 = backbone.encode_video(video_bcthw)
        z_c = z_0[:, :, : dcfg.num_cond_latent_frames]
        z_p = z_0[:, :, dcfg.num_cond_latent_frames :]
        z_n = fm.interpolate(z_p, eps_v, tau_v)
        return z_c, z_n

    # ── Warmup ────────────────────────────────────────────────────────────
    print(f"\n  Warming up ({args.warmup} iterations) …")
    with torch.no_grad():
        for _ in range(args.warmup):
            backbone.clear_hidden_states_cache()
            if include_vae:
                z_cond, z_noisy = _latents_from_vae()
            backbone.forward_transformer(
                z_noisy=z_noisy, z_cond=z_cond,
                tau_v=tau_v, encoder_hidden_states=t5_emb,
            )
            h_raw = backbone.get_captured_hidden_states()
            backbone.pool_hidden_states(h_raw.float(), T_lat, mode=pool_mode)

    # ── BENCHMARK 1: VAE (optional) + Video DiT forward ───────────────────
    print(f"\n  Benchmarking ({args.repeats} repeats) …")
    times_vdit = []
    with torch.no_grad():
        for _ in range(args.repeats):
            backbone.clear_hidden_states_cache()
            if include_vae:
                cuda_sync()
                t_vae0 = time.perf_counter()
                z_cond, z_noisy = _latents_from_vae()
                cuda_sync()
                times_vae.append(time.perf_counter() - t_vae0)
                with Timer() as t:
                    backbone.forward_transformer(
                        z_noisy=z_noisy, z_cond=z_cond,
                        tau_v=tau_v, encoder_hidden_states=t5_emb,
                    )
                times_vdit.append(t.elapsed)
            else:
                with Timer() as t:
                    backbone.forward_transformer(
                        z_noisy=z_noisy, z_cond=z_cond,
                        tau_v=tau_v, encoder_hidden_states=t5_emb,
                    )
                times_vdit.append(t.elapsed)
    # Capture hidden states for pool benchmark
    h_raw_cached = backbone.get_captured_hidden_states().float().detach()

    # ── BENCHMARK 2: h_video pooling ────────────────────────────────────
    times_pool_mean = []
    times_pool_none = []

    with torch.no_grad():
        for _ in range(args.repeats):
            h = h_raw_cached.clone()
            with Timer() as t:
                _ = backbone.pool_hidden_states(h, T_lat, mode="mean")
            times_pool_mean.append(t.elapsed)

        for _ in range(args.repeats):
            h = h_raw_cached.clone()
            with Timer() as t:
                _ = backbone.pool_hidden_states(h, T_lat, mode="none")
            times_pool_none.append(t.elapsed)

    # ── BENCHMARK 3: Action DiT (full ODE denoising) ────────────────────
    # Get pooled h_video for action decoder input
    h_pooled_mean = backbone.pool_hidden_states(h_raw_cached.clone(), T_lat, mode="mean")
    h_pooled_none = backbone.pool_hidden_states(h_raw_cached.clone(), T_lat, mode="none")

    h_for_action = h_pooled_mean if pool_mode == "mean" else h_pooled_none
    h_for_action = h_for_action.to(device=device, dtype=dtype)

    # Single action decoder forward
    times_action_fwd = []
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = action_decoder(
                noisy_actions=a_noise, proprio=proprio,
                h_video=h_for_action, tau_a=tau_v_action, tau_v=tau_v_action,
                t5_embedding=t5_emb, training=False,
            )
        for _ in range(args.repeats):
            with Timer() as t:
                _ = action_decoder(
                    noisy_actions=a_noise, proprio=proprio,
                    h_video=h_for_action, tau_a=tau_v_action, tau_v=tau_v_action,
                    t5_embedding=t5_emb, training=False,
                )
            times_action_fwd.append(t.elapsed)

    # Full ODE denoising (N Euler steps)
    times_action_ode = []
    dt = -1.0 / args.action_steps

    with torch.no_grad():
        for _ in range(args.repeats):
            a_t = a_noise.clone()
            tau  = 1.0
            tau_tensor = torch.ones(B, device=device, dtype=dtype)
            with Timer() as t:
                for _ in range(args.action_steps):
                    tau_tensor.fill_(tau)
                    v = action_decoder(
                        noisy_actions=a_t, proprio=proprio,
                        h_video=h_for_action, tau_a=tau_tensor, tau_v=tau_v_action,
                        t5_embedding=t5_emb, training=False,
                    )
                    a_t = a_t + v * dt
                    tau += dt
            times_action_ode.append(t.elapsed)

    # ── Print Results ─────────────────────────────────────────────────────
    THW = h_raw_cached.shape[1]
    print(f"\n  Config: B={B}, latent={T_lat}×{H_lat}×{W_lat}, h_tokens={THW}, "
          f"pool_mode={pool_mode}, action_steps={args.action_steps}")
    print(f"  data={data_source}")
    print(f"  VAE in timed repeats (end-to-end): {include_vae}")
    print(f"  stage2_checkpoint={'loaded' if loaded_s2 else 'not found (random init)'}  path={stage2_path}")
    if vae_encode_ms is not None:
        print(f"  VAE one-shot encode (reference only, --exclude_vae_from_timing): {vae_encode_ms:.2f} ms")
    print(f"  dtype={args.dtype}, device={device}\n")

    print(f"  {'Stage':<45s}  {'mean':>8s}  {'median':>8s}  {'min':>8s}  {'max':>8s}  {'std':>7s}")
    print("  " + "-" * 90)

    def row(label, times):
        s = stats(times)
        print(f"  {label:<45s}  "
              f"{s['mean_ms']:>7.2f}ms  "
              f"{s['median_ms']:>7.2f}ms  "
              f"{s['min_ms']:>7.2f}ms  "
              f"{s['max_ms']:>7.2f}ms  "
              f"{s['std_ms']:>6.2f}ms")

    n = 1
    if times_vae:
        row(f"{n}. VAE encode (pixels → latents)", times_vae)
        n += 1
    row(f"{n}. Video-DiT forward (h_video raw)", times_vdit)
    n += 1
    row(f"{n}. Action-DiT single forward", times_action_fwd)
    n += 1
    row(f"{n}. Action-DiT ODE ({args.action_steps} Euler steps)", times_action_ode)

    pool_mean_t = statistics.mean(times_pool_mean) if pool_mode == "mean" else statistics.mean(times_pool_none)
    total_mean = pool_mean_t + statistics.mean(times_action_ode) + statistics.mean(times_vdit)
    if times_vae:
        total_mean += statistics.mean(times_vae)
    print("  " + "-" * 90)
    total_label = "  → TOTAL (VAE + DiT + pool + ODE)" if times_vae else "  → TOTAL (DiT + pool + ODE)"
    print(f"  {total_label:<45s}  {total_mean*1000:>7.2f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    from configs.config import LIBERO_SUITES

    p = argparse.ArgumentParser(description="Latency benchmark for mimic-video pipeline")
    p.add_argument("--device",           default="cuda",  help="cuda / cpu")
    p.add_argument("--dtype",            default="bf16",  choices=["bf16", "fp32"])
    p.add_argument("--batch_size",       type=int, default=1)
    p.add_argument("--warmup",           type=int, default=3,  help="warmup iterations (not timed)")
    p.add_argument("--repeats",          type=int, default=10, help="timed iterations")
    p.add_argument("--action_steps",     type=int, default=10, help="Euler ODE steps for action")
    p.add_argument("--pool_mode",        default="mean",  choices=["mean", "none"],
                   help="h_video pooling mode for action decoder input")
    p.add_argument("--dry_run",          action="store_true",
                   help="Use random proxy tensors, skip model loading")
    p.add_argument("--cosmos_model_id",  default="nvidia/Cosmos-Predict2-2B-Video2World")
    p.add_argument("--suite",            type=str, default=None, choices=list(LIBERO_SUITES.keys()),
                   help="LIBERO suite: sets DataConfig and default checkpoint / precomputed dirs")
    p.add_argument("--stage1_checkpoint", default=None,
                   help="Stage-1 LoRA dir (default: checkpoints/<suite>/stage1/final or checkpoints/stage1/final)")
    p.add_argument("--stage2_checkpoint", default=None,
                   help="Stage-2 dir with action_decoder.pt (default: checkpoints/<suite>/stage2/final or checkpoints/stage2/final)")
    p.add_argument("--precomputed_dir",   default=None,
                   help="Override precomputed T5 / action_stats (default from DataConfig or suite)")
    p.add_argument("--use_dataset",       action="store_true",
                   help="Use one real training sample: VAE-encode latents + precomputed T5 (requires LeRobot data)")
    p.add_argument("--dataset_index",     type=int, default=0,
                   help="Index into MimicVideoDataset when --use_dataset")
    p.add_argument("--exclude_vae_from_timing", action="store_true",
                   help="Time only DiT+decoder: VAE encode once (dataset) or use random latents (no --use_dataset)")
    return p.parse_args()


if __name__ == "__main__":
    from configs.config import DataConfig, ModelConfig, get_suite_data_config

    args = parse_args()

    print(f"\nmimic-video Latency Benchmark")
    print(f"  device={args.device}  dtype={args.dtype}  batch_size={args.batch_size}")
    print(f"  warmup={args.warmup}  repeats={args.repeats}  action_ode_steps={args.action_steps}")
    print(f"  pool_mode={args.pool_mode}  dry_run={args.dry_run}")
    print(f"  suite={args.suite}  use_dataset={args.use_dataset}  exclude_vae_from_timing={args.exclude_vae_from_timing}")

    if args.dry_run:
        benchmark_dry_run(args)
    else:
        if args.suite:
            dcfg = get_suite_data_config(args.suite)
        else:
            dcfg = DataConfig()
        if args.precomputed_dir:
            dcfg.precomputed_dir = args.precomputed_dir

        mcfg = ModelConfig()

        if args.use_dataset and not dcfg.repo_id:
            raise SystemExit(
                "--use_dataset needs a dataset repo_id. Pass e.g. --suite libero_object "
                "or configure repo_id in DataConfig."
            )

        stage1_path = args.stage1_checkpoint
        if stage1_path is None:
            stage1_path = (
                f"checkpoints/{args.suite}/stage1/final" if args.suite else "checkpoints/stage1/final"
            )

        stage2_path = args.stage2_checkpoint
        if stage2_path is None:
            stage2_path = (
                f"checkpoints/{args.suite}/stage2/final" if args.suite else "checkpoints/stage2/final"
            )

        benchmark_real(args, dcfg, mcfg, stage1_path, stage2_path)
