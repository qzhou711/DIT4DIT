# Video-Action Policy (VAP)

> Generalist robot control via video diffusion model priors.

VAP implements the [mimic-video](https://arxiv.org/abs/2512.15692) framework
with a refactored, extensible codebase featuring YAML-driven configuration,
a model registry, and corrected paper implementations.

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Precompute T5 embeddings
python tools/precompute.py --output_dir precomputed/

# 2. Stage 1 — video backbone LoRA finetuning
python tools/train.py --stage 1 --config conf/train/stage1.yaml

# 3. Stage 2 — action decoder training  
python tools/train.py --stage 2 \
    --config conf/train/stage2.yaml \
    --backbone_checkpoint checkpoints/stage1/final

# 4. Unit tests
pytest tests/ -v
```

## Architecture

```
vap/                            # Core package
├── core/
│   ├── registry.py             # @Registry.backbone / @Registry.decoder decorators
│   ├── protocol.py             # RobotPolicy Protocol + Observation / ActionChunk types
│   └── noise_schedule.py       # Flow schedules (corrected τ_a: SqrtMinusEpsSchedule)
├── backbone/
│   ├── cosmos_backbone.py      # CosmosBackbone (Cosmos-Predict2 + LoRA)
│   └── feature_extractor.py    # FeatureExtractor (multi-layer hooks, PoolingMode enum)
├── decoder/
│   ├── timestep_embed.py       # JointTimestepEmbedding (τ_v × τ_a bilinear)
│   └── dit_decoder.py          # DiTActionDecoder (cross-attn + self-attn + FFN)
├── policy/
│   ├── sampler.py              # EulerSampler + MidpointSampler
│   └── world_action_policy.py  # WorldActionPolicy implements RobotPolicy Protocol
├── data/
│   ├── video_transforms.py     # ConcatCameras + NormalizeToUnitRange (callables)
│   ├── episode_dataset.py      # EpisodeDataset (injected transforms)
│   └── datamodule.py           # VideoActionDataModule (train/val split)
└── engine/
    ├── stage.py                # Stage1Strategy + Stage2Strategy (Strategy pattern)
    └── trainer.py              # Unified Trainer (LambdaLR, validation, JSON ckpts)
conf/train/
├── stage1.yaml                 # Stage 1 hyperparameters
└── stage2.yaml                 # Stage 2 hyperparameters
tools/
└── train.py                    # python tools/train.py --stage 1/2 --config ...
tests/
├── test_noise_schedule.py
└── test_dit_decoder.py
```

## Key Design Decisions

| Aspect | Choice |
|--------|--------|
| Config | YAML + OmegaConf (swappable, version-controlled) |
| Models | `@Registry.backbone/decoder` decorators for plugin extensibility |
| Inference | `RobotPolicy` Protocol — any conforming class usable in eval loops |
| Stages | Strategy pattern — single `Trainer` instead of two copy-pasted classes |
| Transforms | Callable objects — composable with `torchvision.transforms.Compose` |
| ODE | `EulerSampler` + `MidpointSampler` — injected, swappable |
| τ_a schedule | `SqrtMinusEpsSchedule` — correct paper distribution (acceptance-rejection) |
| Pooling | `PoolingMode` enum — explicit, no silent defaults |
