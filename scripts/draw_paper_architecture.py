#!/usr/bin/env python3
"""Generate publication-quality architecture figures for mimic-video (DIT4DIT).

Outputs:
  figures/mimic_video_architecture.pdf
  figures/mimic_video_architecture.png  (300 dpi)
  figures/mimic_video_architecture.svg

Run from repo root:
  python scripts/draw_paper_architecture.py
  python scripts/draw_paper_architecture.py --suite libero_object
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "figures"),
        help="Directory for PDF/PNG/SVG",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Optional: libero_* suite for DataConfig (camera size, etc.)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from configs.config import DataConfig, ModelConfig, get_suite_data_config

    if args.suite:
        dcfg = get_suite_data_config(args.suite)
    else:
        dcfg = DataConfig()
    mcfg = ModelConfig()

    H_px, W_px = dcfg.camera_height, dcfg.camera_width
    T_pix = dcfg.num_pixel_frames
    T_lat = dcfg.num_latent_frames
    T_cond, T_pred = dcfg.num_cond_latent_frames, dcfg.num_pred_latent_frames
    z_c = mcfg.vae_latent_channels
    adim = dcfg.action_dim
    pdim = dcfg.proprio_dim
    chunk = dcfg.action_chunk_size
    Dh = mcfg.backbone_hidden_dim
    n_dec = mcfg.decoder_num_layers
    n_heads = mcfg.decoder_num_heads
    d_dec = mcfg.decoder_hidden_dim
    layer_k = mcfg.hidden_state_layer
    n_blks = 28  # Cosmos Predict2 2B Video2World
    lora_r = mcfg.lora_rank

    # Approximate token count after pool=none: T_lat * (H/8/2)^2 * 2 for 2-cam concat — document as "~"
    h_tok = T_lat * (H_px // 16) * (W_px // 16)  # rough grid after patch 2x2 spatial

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib import patheffects as pe

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Nimbus Sans"],
            "font.size": 9,
            "axes.linewidth": 0.8,
        }
    )

    # Colors (print-safe, colorblind-friendly-ish)
    C_IN = "#ECEFF1"
    E_IN = "#37474F"
    C_S1 = "#BBDEFB"
    E_S1 = "#0D47A1"
    C_S2 = "#FFCDD2"
    E_S2 = "#B71C1C"
    C_OUT = "#C8E6C9"
    E_OUT = "#1B5E20"
    C_FM = "#FFF3E0"
    E_FM = "#E65100"
    BG = "#FFFFFF"

    def rounded_box(ax, x, y, w, h, text, fc, ec, fs=8, fw="normal", alpha=1.0, lw=1.2):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            alpha=alpha,
            zorder=2,
        )
        ax.add_patch(p)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight=fw,
            color=ec,
            zorder=3,
            linespacing=1.35,
        )

    def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.3, rad=0.0):
        arr = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
        ax.add_patch(arr)

    def banner(ax, x, y, w, h, title, subtitle, fc, ec, fs=8):
        rounded_box(ax, x, y, w, h, f"{title}\n{subtitle}", fc, ec, fs=fs, fw="bold")

    fig = plt.figure(figsize=(14.5, 10.2), facecolor=BG)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.15, 1.0], hspace=0.22, left=0.04, right=0.96, top=0.93, bottom=0.06)

    # ─── Panel A: Inference ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_facecolor(BG)

    ax1.text(
        0.5,
        1.02,
        "(a) Inference: world model (video DiT) → action decoder (flow matching)",
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#1a1a1a",
    )

    # Section backgrounds
    ax1.add_patch(
        FancyBboxPatch(
            (0.02, 0.08),
            0.44,
            0.88,
            boxstyle="round,pad=0.015",
            facecolor="#E3F2FD",
            edgecolor=E_S1,
            linewidth=1.0,
            linestyle="--",
            alpha=0.35,
            zorder=0,
        )
    )
    ax1.text(0.035, 0.93, "Stage-1  Cosmos backbone (frozen after training)", fontsize=8, color=E_S1, fontweight="bold")

    ax1.add_patch(
        FancyBboxPatch(
            (0.48, 0.08),
            0.50,
            0.88,
            boxstyle="round,pad=0.015",
            facecolor="#FFEBEE",
            edgecolor=E_S2,
            linewidth=1.0,
            linestyle="--",
            alpha=0.35,
            zorder=0,
        )
    )
    ax1.text(0.495, 0.93, "Stage-2  ActionDecoderDiT", fontsize=8, color=E_S2, fontweight="bold")

    # Left inputs
    rounded_box(
        ax1,
        0.03,
        0.72,
        0.12,
        0.12,
        f"Multi-cam RGB\n[{T_pix} frames]\n≈[{H_px}×{W_px}]",
        C_IN,
        E_IN,
        fs=7.5,
    )
    rounded_box(
        ax1,
        0.03,
        0.52,
        0.12,
        0.12,
        "Task text\n(precomp. T5)",
        C_IN,
        E_IN,
        fs=7.5,
    )

    banner(ax1, 0.17, 0.72, 0.14, 0.12, "VAE encoder", "[frozen]", C_S1, E_S1)
    ax1.text(0.24, 0.68, f"z  [{z_c}×{T_lat}×…]", ha="center", fontsize=7, color=E_S1)

    banner(ax1, 0.17, 0.52, 0.14, 0.12, "T5 → c_txt", "[frozen]", C_S1, E_S1)

    # Cosmos DiT big box
    rounded_box(
        ax1,
        0.17,
        0.22,
        0.27,
        0.38,
        f"Cosmos-2B Video DiT\n{n_blks} × TransformerBlock\n"
        f"(self-attn · cross-attn · FFN)\n"
        f"+ LoRA (r={lora_r}) on attn & FFN\n"
        f"Hidden state at layer {layer_k}\n"
        f"→ h_raw  [B, N, {Dh}]",
        C_S1,
        E_S1,
        fs=7.5,
        alpha=0.95,
    )

    banner(ax1, 0.17, 0.10, 0.27, 0.09, "Spatial pool (mean / none)", f"→ h_video  [B, {T_lat}, {Dh}] or [B, N, {Dh}]", "#E1F5FE", E_S1)

    arrow(ax1, 0.09, 0.78, 0.17, 0.78, E_S1)
    arrow(ax1, 0.09, 0.58, 0.17, 0.58, E_S1)
    arrow(ax1, 0.24, 0.72, 0.24, 0.60, E_S1)
    arrow(ax1, 0.24, 0.52, 0.24, 0.60, E_S1)
    arrow(ax1, 0.31, 0.52, 0.31, 0.60, E_S1)

    # z + text into DiT (conceptual merge at DiT)
    ax1.annotate(
        "",
        xy=(0.24, 0.60),
        xytext=(0.24, 0.72),
        arrowprops=dict(arrowstyle="-|>", color=E_S1, lw=1.2),
    )
    ax1.annotate(
        "",
        xy=(0.31, 0.60),
        xytext=(0.31, 0.52),
        arrowprops=dict(arrowstyle="-|>", color=E_S1, lw=1.2),
    )

    arrow(ax1, 0.24, 0.22, 0.24, 0.19, E_S1)

    # Bridge to Stage 2
    arrow(ax1, 0.44, 0.14, 0.50, 0.14, E_S2, lw=1.8)
    ax1.text(0.46, 0.16, "h_video", fontsize=7, color=E_S2, fontweight="bold")

    # Stage 2 inputs (bottom left of red zone)
    rounded_box(
        ax1,
        0.50,
        0.72,
        0.11,
        0.10,
        f"Proprio\n[B, {pdim}]",
        C_IN,
        E_IN,
        fs=7.5,
    )
    rounded_box(
        ax1,
        0.50,
        0.58,
        0.11,
        0.10,
        f"Noisy actions\n[B,{chunk},{adim}]",
        C_FM,
        E_FM,
        fs=7.5,
    )
    rounded_box(
        ax1,
        0.50,
        0.44,
        0.11,
        0.10,
        "τ_a, τ_v\n[0,1]",
        C_FM,
        E_FM,
        fs=7.5,
    )

    banner(
        ax1,
        0.63,
        0.58,
        0.32,
        0.24,
        "ActionDecoderDiT",
        f"{n_dec} blocks × (AdaLN + Self-Attn + Cross-Attn to video + FFN)\n"
        f"d_model={d_dec}, heads={n_heads}  ·  optional T5 token → concat to h_video\n"
        f"Bilinear τ_v ⊗ τ_a conditioning",
        C_S2,
        E_S2,
        fs=7.5,
    )

    banner(
        ax1,
        0.63,
        0.36,
        0.32,
        0.12,
        "Zero-init linear",
        "velocity v(a, τ)  [B, chunk, action_dim]",
        C_S2,
        E_S2,
        fs=7.5,
    )

    banner(
        ax1,
        0.63,
        0.10,
        0.32,
        0.12,
        "Euler ODE (flow matching)",
        f"{10} steps  τ: 1→0  (default)",
        C_FM,
        E_FM,
        fs=7.5,
    )

    rounded_box(ax1, 0.88, 0.42, 0.10, 0.14, f"Actions\n[B,{chunk},{adim}]\n(denorm.)", C_OUT, E_OUT, fs=8, fw="bold")

    # Arrows stage 2
    arrow(ax1, 0.61, 0.77, 0.63, 0.77, E_S2)
    arrow(ax1, 0.61, 0.63, 0.63, 0.68, E_S2)
    arrow(ax1, 0.61, 0.49, 0.63, 0.59, E_S2)
    arrow(ax1, 0.50, 0.14, 0.63, 0.30, E_S2, rad=0.08)
    arrow(ax1, 0.79, 0.36, 0.79, 0.22, E_S2)
    arrow(ax1, 0.79, 0.10, 0.88, 0.49, E_S2, rad=0.05)

    # h_video into decoder (from pool)
    arrow(ax1, 0.44, 0.14, 0.63, 0.62, "#6A1B9A", lw=1.5)
    ax1.text(0.52, 0.40, "cross-attn\nKV", fontsize=6.5, color="#6A1B9A")

    # ─── Panel B: Training ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_facecolor(BG)

    ax2.text(
        0.5,
        1.05,
        "(b) Training (two stages)",
        transform=ax2.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#1a1a1a",
    )

    rounded_box(
        ax2,
        0.03,
        0.18,
        0.42,
        0.72,
        "Stage-1\n\n"
        "Train LoRA on Cosmos Video DiT\n"
        "VAE & T5 frozen · flow-matching\n"
        "objective on video latents\n\n"
        "← dataset: RGB windows + T5",
        C_S1,
        E_S1,
        fs=8.5,
        fw="bold",
        alpha=0.9,
    )

    rounded_box(
        ax2,
        0.53,
        0.18,
        0.42,
        0.72,
        "Stage-2\n\n"
        "Cosmos + LoRA frozen\n"
        "Train ActionDecoderDiT only\n"
        "Flow matching on actions\n"
        "conditioned on h_video (+T5)\n\n"
        "← dataset: video + proprio + actions",
        C_S2,
        E_S2,
        fs=8.5,
        fw="bold",
        alpha=0.9,
    )

    arrow(ax2, 0.45, 0.54, 0.53, 0.54, "#424242", lw=2.0)

    ax2.text(
        0.5,
        0.06,
        f"Config snapshot: latent T={T_lat} ({T_cond} cond + {T_pred} pred)  ·  "
        f"cam {H_px}×{W_px}  ·  hook layer k={layer_k}  ·  "
        f"~N≈{h_tok} tokens if pool=none (order-of-magnitude)",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#555",
        style="italic",
    )

    fig.text(
        0.5,
        0.98,
        "Mimic-Video / DIT4DIT  —  Architecture Overview",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#111",
    )

    base = os.path.join(args.output_dir, "mimic_video_architecture")
    for ext in ("pdf", "png", "svg"):
        p = f"{base}.{ext}"
        dpi = 300 if ext == "png" else None
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=BG, edgecolor="none")
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
