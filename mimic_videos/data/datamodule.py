"""VideoActionDataModule — DataLoader factory in the LightningDataModule style.

Decouples dataset construction, train/val splitting, and DataLoader
configuration from the training loop.  This keeps trainer code free of
dataset-specific arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from .episode_dataset import EpisodeDataset


@dataclass
class DataConfig:
    """Full dataset and loader configuration.

    Designed to be constructed directly from a YAML file::

        from omegaconf import OmegaConf
        cfg = OmegaConf.structured(DataConfig)
    """

    repo_id: str = "pierre818191/UnitreeBagClose"
    camera_names: List[str] = field(default_factory=lambda: [
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "observation.images.cam_high",
        "observation.images.cam_low",
    ])
    state_keys: List[str] = field(default_factory=lambda: [
        "observation.state"
    ])
    action_keys: List[str] = field(default_factory=lambda: [
        "action"
    ])
    num_video_frames: int = 17
    action_chunk_size: int = 16
    action_dim: int = 16
    proprio_dim: int = 16
    target_height: int = 480
    target_width: int = 640
    fps: int = 30
    num_train_episodes: int = 400
    val_split: float = 0.1   # fraction of episodes held out for validation


class VideoActionDataModule:
    """Constructs train and validation :class:`EpisodeDataset` instances.

    Args:
        cfg: :class:`DataConfig` instance.
        micro_batch_size: Batch size for each DataLoader step.
        num_workers: DataLoader worker count.
        precomputed_dir: Optional path to precomputed T5 embedding files.
    """

    def __init__(
        self,
        cfg: DataConfig,
        micro_batch_size: int = 4,
        num_workers: int = 4,
        precomputed_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.micro_batch_size = micro_batch_size
        self.num_workers = num_workers
        self.precomputed_dir = precomputed_dir

        self._train_dataset: Optional[EpisodeDataset] = None
        self._val_dataset: Optional[EpisodeDataset] = None

    def setup(self) -> None:
        """Build train and validation datasets."""
        import os

        t5_emb = None
        if self.precomputed_dir:
            t5_path = os.path.join(self.precomputed_dir, "t5_embedding.pt")
            if os.path.exists(t5_path):
                t5_emb = torch.load(t5_path, map_location="cpu", weights_only=True)

        all_eps = list(range(self.cfg.num_train_episodes))
        n_val = max(1, int(len(all_eps) * self.cfg.val_split))
        val_eps = all_eps[-n_val:]
        train_eps = all_eps[:-n_val]

        shared_kwargs = dict(
            repo_id=self.cfg.repo_id,
            camera_names=self.cfg.camera_names,
            state_keys=self.cfg.state_keys,
            action_keys=self.cfg.action_keys,
            num_video_frames=self.cfg.num_video_frames,
            action_chunk_size=self.cfg.action_chunk_size,
            action_dim=self.cfg.action_dim,
            proprio_dim=self.cfg.proprio_dim,
            fps=self.cfg.fps,
            precomputed_t5=t5_emb,
        )

        self._train_dataset = EpisodeDataset(
            episode_indices=train_eps, **shared_kwargs
        )
        self._val_dataset = EpisodeDataset(
            episode_indices=val_eps, **shared_kwargs
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader().")
        return DataLoader(
            self._train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before val_dataloader().")
        return DataLoader(
            self._val_dataset,
            batch_size=self.micro_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    @property
    def train_size(self) -> int:
        if self._train_dataset is None:
            raise RuntimeError("Call setup() first.")
        return len(self._train_dataset)

    @property
    def val_size(self) -> int:
        if self._val_dataset is None:
            raise RuntimeError("Call setup() first.")
        return len(self._val_dataset)
