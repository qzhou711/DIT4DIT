"""EpisodeDataset — LeRobot-backed dataset for video-action training.

Key differences from the original MimicVideoDataset
-----------------------------------------------------
- Multi-camera concatenation is done via the :class:`~mimic_videos.data.video_transforms.ConcatCameras`
  callable transform rather than a standalone function, making augmentation
  composition straightforward.
- Transform pipeline is injected at construction time, so augmentations can be
  swapped without subclassing.
- ``build_index()`` returns the index table as an inspectable attribute
  (``valid_frames``) to help with debugging and reproducibility logging.
- Action normalisation uses explicit ``(mean, std)`` buffers passed in; no
  implicit stats computation inside the dataset.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from .video_transforms import ConcatCameras, NormalizeToUnitRange


class EpisodeDataset(Dataset):
    """Robot episode dataset backed by a LeRobot repository.

    Args:
        repo_id: HuggingFace dataset repo identifier.
        camera_names: Ordered list of camera observation keys.
        state_keys: Keys forming the proprioceptive state vector.
        action_keys: Keys forming the action vector.
        num_video_frames: Pixel frames per sample (observation horizon).
        action_chunk_size: Number of future actions per sample.
        action_dim: Dimensionality of each action vector (truncated if needed).
        proprio_dim: Dimensionality of the state vector (truncated if needed).
        episode_indices: If given, restrict to these episode indices.
        video_transform: Callable applied to the stacked camera views
            ``List[T×C×H×W] → T×C×H×W``.  Defaults to :class:`ConcatCameras`
            with a 2-column 480×640 grid.
        action_mean / action_std: Pre-computed normalisation constants.
            Pass ``None`` to skip normalisation.
        fps: Dataset recording frame rate (used to build delta timestamps).
        precomputed_t5: Optional pre-loaded T5 text embedding tensor.
    """

    def __init__(
        self,
        repo_id: str,
        camera_names: List[str],
        state_keys: List[str],
        action_keys: List[str],
        num_video_frames: int = 17,
        action_chunk_size: int = 16,
        action_dim: int = 16,
        proprio_dim: int = 16,
        episode_indices: Optional[List[int]] = None,
        video_transform: Optional[Callable] = None,
        action_mean: Optional[torch.Tensor] = None,
        action_std: Optional[torch.Tensor] = None,
        fps: int = 30,
        precomputed_t5: Optional[torch.Tensor] = None,
    ) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.camera_names = camera_names
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.num_video_frames = num_video_frames
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.precomputed_t5 = precomputed_t5

        # Default transform: 2-column 480×640 grid + normalise to [-1, 1]
        self.video_transform: Callable = video_transform or _default_video_transform(fps)

        # Normalisation (may be None)
        self.action_mean = action_mean
        self.action_std = action_std

        # Build LeRobot dataset with per-camera delta timestamps
        delta_timestamps = {
            cam: [i / fps for i in range(num_video_frames)] for cam in camera_names
        }
        self._lerobot = LeRobotDataset(repo_id=repo_id, delta_timestamps=delta_timestamps)

        # Build valid-frame index (public attribute for inspection)
        self.valid_frames: List[int] = self._build_index(episode_indices)

    # ---------------------------------------------------------------- #
    # Index construction
    # ---------------------------------------------------------------- #

    def _build_index(self, episode_indices: Optional[List[int]]) -> List[int]:
        """Return global frame indices where a full sample can be extracted."""
        min_len = self.num_video_frames + self.action_chunk_size
        valid: List[int] = []

        for ep in self._lerobot.meta.episodes:
            if episode_indices is not None and ep["episode_index"] not in episode_indices:
                continue
            ep_len = ep["dataset_to_index"] - ep["dataset_from_index"]
            if ep_len < min_len:
                continue
            valid.extend(
                ep["dataset_from_index"] + offset
                for offset in range(ep_len - min_len + 1)
            )
        return valid

    # ---------------------------------------------------------------- #
    # Dataset interface
    # ---------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.valid_frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx = self.valid_frames[idx]

        # Load camera frames via LeRobot (torchcodec-accelerated)
        sample = self._lerobot[global_idx]
        camera_views = []
        for cam in self.camera_names:
            frames = sample[cam]
            if frames.ndim == 3:  # Single frame edge case
                frames = frames.unsqueeze(0)
            camera_views.append(frames)

        video = self.video_transform(camera_views)  # [T, C, H, W]

        # Load state & action by direct index (faster than delta_timestamps)
        proprio, actions = self._load_state_and_actions(global_idx)
        actions = self._normalise(actions)

        result: Dict[str, torch.Tensor] = {
            "video": video,
            "proprio": proprio,
            "actions": actions,
        }
        if self.precomputed_t5 is not None:
            result["t5_embedding"] = self.precomputed_t5
        return result

    def _load_state_and_actions(
        self, global_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load proprioception vector and action chunk by direct row access."""
        def _parse_row(row: dict, keys: List[str], dim: int) -> torch.Tensor:
            parts = []
            for k in keys:
                v = row[k]
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, dtype=torch.float32)
                parts.append(v.float().flatten())
            return torch.cat(parts)[:dim]

        row = self._lerobot.hf_dataset[global_idx]
        proprio = _parse_row(row, self.state_keys, self.proprio_dim)

        action_rows = [
            _parse_row(self._lerobot.hf_dataset[global_idx + i], self.action_keys, self.action_dim)
            for i in range(1, self.action_chunk_size + 1)
        ]
        actions = torch.stack(action_rows)  # [chunk_size, action_dim]
        return proprio, actions

    def _normalise(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_mean is None:
            return actions
        dev = actions.device
        return (actions - self.action_mean.to(dev)) / (self.action_std.to(dev) + 1e-8)

    def denormalise(self, actions: torch.Tensor) -> torch.Tensor:
        """Invert normalisation (for evaluation / visualisation)."""
        if self.action_mean is None:
            return actions
        dev = actions.device
        return actions * (self.action_std.to(dev) + 1e-8) + self.action_mean.to(dev)


# ------------------------------------------------------------------ #
# Private helpers
# ------------------------------------------------------------------ #


def _default_video_transform(fps: int = 30) -> Callable:
    """Build the default (non-augmenting) video transform pipeline."""
    concat = ConcatCameras(target_height=480, target_width=640, cols=2)
    normalise = NormalizeToUnitRange()

    def _transform(views):
        grid = concat(views)
        return normalise(grid)

    return _transform
