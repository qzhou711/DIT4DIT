"""Unified training entry point.

Usage::

    # Stage 1 — video backbone LoRA finetuning
    python tools/train.py --stage 1 --config conf/train/stage1.yaml

    # Stage 2 — action decoder training
    python tools/train.py --stage 2 --config conf/train/stage2.yaml \
        --backbone_checkpoint checkpoints/stage1/final
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_yaml(path: str) -> dict:
    try:
        from omegaconf import OmegaConf
        return dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))
    except ImportError:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)


def build_backbone(cfg: dict, device: str):
    from mimic_videos.backbone import CosmosBackbone, FeatureExtractor, PoolingMode
    from mimic_videos.backbone.cosmos_backbone import LoRAConfig

    m_cfg = cfg.get("model", {})
    lora_cfg = m_cfg.get("lora", {})
    lora = LoRAConfig(
        rank=lora_cfg.get("rank", 16),
        alpha=lora_cfg.get("alpha", 16),
        target_modules=lora_cfg.get("target_modules", None) or LoRAConfig().target_modules,
    )
    pooling_mode = PoolingMode(m_cfg.get("pooling_mode", "all_tokens"))
    return CosmosBackbone(
        model_id=m_cfg.get("model_id", "nvidia/Cosmos-Predict2-2B-Video2World"),
        lora=lora,
        feature_layer_indices=m_cfg.get("feature_layers", [19]),
        pooling_mode=pooling_mode,
        num_latent_frames=m_cfg.get("num_latent_frames", 5),
        dtype=torch.bfloat16,
        device=device,
    )


def build_decoder(cfg: dict):
    from mimic_videos.decoder import DiTActionDecoder

    d_cfg = cfg.get("model", {}).get("decoder", {})
    return DiTActionDecoder(
        action_dim=d_cfg.get("action_dim", 16),
        proprio_dim=d_cfg.get("proprio_dim", 16),
        hidden_dim=d_cfg.get("hidden_dim", 512),
        num_layers=d_cfg.get("num_layers", 8),
        num_heads=d_cfg.get("num_heads", 8),
        context_dim=d_cfg.get("context_dim", 2048),
        chunk_size=d_cfg.get("chunk_size", 16),
        proprio_mask_prob=d_cfg.get("proprio_mask_prob", 0.1),
    )


def build_schedules(cfg: dict):
    from mimic_videos.core.noise_schedule import build_schedule

    t_cfg = cfg.get("training", {}).get("noise_schedule", {})
    vid_sch = build_schedule(t_cfg.get("video", "logit_normal"))
    act_sch = build_schedule(t_cfg.get("action", "sqrt_minus_eps"))
    return vid_sch, act_sch


def build_datamodule(cfg: dict, precomputed_dir: str):
    from mimic_videos.data import VideoActionDataModule, DataConfig

    d_cfg = cfg.get("data", {})
    data_config = DataConfig(
        repo_id=d_cfg.get("repo_id", DataConfig.repo_id),
        num_train_episodes=d_cfg.get("num_train_episodes", 400),
    )
    t_cfg = cfg.get("training", {})
    return VideoActionDataModule(
        cfg=data_config,
        micro_batch_size=t_cfg.get("micro_batch_size", 4),
        num_workers=t_cfg.get("num_workers", 4),
        precomputed_dir=precomputed_dir,
    )


def run_stage1(args, cfg):
    from mimic_videos.engine import Stage1Strategy, StageConfig, Trainer

    print("=== Stage 1: Video backbone LoRA finetuning ===")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    backbone = build_backbone(cfg, args.device)
    backbone.transformer.to(args.device)
    backbone.offload_auxiliary("cpu")

    if args.compile:
        print("Compiling backbone transformer...")
        backbone.transformer = torch.compile(backbone.transformer)

    dm = build_datamodule(cfg, args.precomputed_dir)
    dm.setup()

    vid_sch, _ = build_schedules(cfg)
    stage = Stage1Strategy(
        backbone=backbone,
        video_schedule=vid_sch,
        num_cond_frames=cfg.get("data", {}).get("num_cond_frames", 2),
        precomputed_t5=None,  # Loaded per-batch from dataset
    )

    t_cfg = cfg.get("training", {})
    trainer = Trainer(
        stage=stage,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader() if t_cfg.get("validate", False) else None,
        lr=t_cfg.get("lr", 1.778e-4),
        warmup_steps=t_cfg.get("warmup_steps", 1000),
        total_steps=t_cfg.get("total_steps", 27000),
        weight_decay=t_cfg.get("weight_decay", 0.0),
        grad_clip=t_cfg.get("grad_clip", 1.0),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 64),
        lr_schedule=t_cfg.get("lr_schedule", "constant"),
        dtype=t_cfg.get("dtype", "bfloat16"),
        output_dir=t_cfg.get("output_dir", "checkpoints/"),
        log_every=t_cfg.get("log_every", 10),
        save_every=t_cfg.get("save_every", 1000),
        device=args.device,
        stage_name="stage1",
        wandb_project=t_cfg.get("wandb_project"),
        wandb_run_name=t_cfg.get("wandb_run_name"),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


def run_stage2(args, cfg):
    from mimic_videos.engine import Stage2Strategy, Trainer

    print("=== Stage 2: Action decoder training ===")

    backbone = build_backbone(cfg, args.device)
    backbone.offload_auxiliary("cpu")

    if args.backbone_checkpoint:
        print(f"Loading backbone from {args.backbone_checkpoint}")
        backbone.load_adapter(args.backbone_checkpoint)

    backbone.transformer.to(args.device).eval()
    backbone.freeze()

    decoder = build_decoder(cfg)
    decoder.to(args.device)

    dm = build_datamodule(cfg, args.precomputed_dir)
    dm.setup()

    vid_sch, act_sch = build_schedules(cfg)
    stage = Stage2Strategy(
        backbone=backbone,
        decoder=decoder,
        video_schedule=vid_sch,
        action_schedule=act_sch,
        num_cond_frames=cfg.get("data", {}).get("num_cond_frames", 2),
    )

    t_cfg = cfg.get("training", {})
    trainer = Trainer(
        stage=stage,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader() if t_cfg.get("validate", False) else None,
        lr=t_cfg.get("lr", 1e-4),
        warmup_steps=t_cfg.get("warmup_steps", 1000),
        total_steps=t_cfg.get("total_steps", 26000),
        weight_decay=t_cfg.get("weight_decay", 0.0),
        grad_clip=t_cfg.get("grad_clip", 1.0),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 8),
        lr_schedule=t_cfg.get("lr_schedule", "linear_decay"),
        dtype=t_cfg.get("dtype", "bfloat16"),
        output_dir=t_cfg.get("output_dir", "checkpoints/"),
        log_every=t_cfg.get("log_every", 10),
        save_every=t_cfg.get("save_every", 1000),
        device=args.device,
        stage_name="stage2",
        wandb_project=t_cfg.get("wandb_project"),
        wandb_run_name=t_cfg.get("wandb_run_name"),
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="VAP — Video-Action Policy trainer")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="Training stage (1=backbone, 2=decoder)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precomputed_dir", type=str, default="precomputed/")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--backbone_checkpoint", type=str, default=None,
                        help="[Stage 2] Path to Stage 1 backbone checkpoint")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the backbone transformer")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)

    if args.stage == 1:
        run_stage1(args, cfg)
    else:
        run_stage2(args, cfg)


if __name__ == "__main__":
    main()
