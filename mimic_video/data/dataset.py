"""MimicVideoDataset wrapping LeRobot for the mimic-video pipeline."""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from mimic_video.data.transforms import concat_cameras, normalize_to_neg1_pos1


class MimicVideoDataset(Dataset):
    """Dataset for mimic-video training.

    Uses delta_timestamps only for camera video frames (which need multi-frame decode).
    State and action data are loaded by manually indexing consecutive frames,
    avoiding the slow delta_timestamps lookup for tabular data.
    """

    def __init__(
        self,
        repo_id: str,
        camera_names: list,
        state_keys: List[str],
        action_keys: List[str],
        num_pixel_frames: int = 17,
        action_chunk_size: int = 16,
        action_dim: int = 16,
        proprio_dim: int = 16,
        target_height: int = 480,
        target_width: int = 640,
        episode_indices: Optional[list] = None,
        precomputed_dir: Optional[str] = None,
        action_stats: Optional[Dict[str, torch.Tensor]] = None,
        action_norm_type: str = "min-max",
        fps: int = 10,
        require_action_chunk: bool = True,
        allow_partial_action_chunk: bool = False,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.repo_id = repo_id
        self.camera_names = camera_names
        self.state_keys = state_keys
        self.action_keys = action_keys
        self.num_pixel_frames = num_pixel_frames
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.target_height = target_height
        self.target_width = target_width
        self.precomputed_dir = precomputed_dir
        self.action_norm_type = action_norm_type
        self.fps = fps
        self.require_action_chunk = require_action_chunk
        self.allow_partial_action_chunk = allow_partial_action_chunk
        self._index_to_episode_end = {}

        # Only use delta_timestamps for cameras (video decode needs it)
        delta_timestamps = {}
        frame_deltas = [i / fps for i in range(num_pixel_frames)]
        for cam_name in camera_names:
            delta_timestamps[cam_name] = frame_deltas

        self.lerobot_dataset = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=delta_timestamps,
        )

        self._build_valid_indices(episode_indices)

        if action_stats is not None:
            self.action_mean = action_stats.get("mean", None)
            self.action_std = action_stats.get("std", None)
            self.action_min = action_stats.get("min", None)
            self.action_max = action_stats.get("max", None)
        else:
            self.action_mean = None
            self.action_std = None
            self.action_min = None
            self.action_max = None

        self.t5_embedding = None       # single-task fallback [1, seq_len, dim]
        self.t5_embeddings = None       # multi-task dict {task_index: [1, seq_len, dim]}

        # Try multi-task first, then single-task
        if precomputed_dir:
            multi_path = os.path.join(precomputed_dir, "t5_embeddings.pt")
            single_path = os.path.join(precomputed_dir, "t5_embedding.pt")
            if os.path.exists(multi_path):
                self.t5_embeddings = torch.load(multi_path, map_location="cpu", weights_only=True)
            elif os.path.exists(single_path):
                self.t5_embedding = torch.load(single_path, map_location="cpu", weights_only=True)

        # Build episode_index → task_index mapping for multi-task datasets
        self.episode_to_task = {}
        self._build_episode_task_map()

    def _build_episode_task_map(self):
        """Build a mapping from episode_index to task_index using dataset metadata."""
        try:
            hf = self.lerobot_dataset.hf_dataset
            if "task_index" in hf.column_names and "episode_index" in hf.column_names:
                # Sample the first row of each episode to get its task_index
                episodes = self.lerobot_dataset.meta.episodes
                for ep in episodes:
                    ep_idx = ep["episode_index"]
                    first_frame = ep["dataset_from_index"]
                    row = hf[first_frame]
                    t_idx = row["task_index"]
                    if isinstance(t_idx, torch.Tensor):
                        t_idx = t_idx.item()
                    self.episode_to_task[ep_idx] = int(t_idx)
        except Exception:
            pass  # Not a multi-task dataset, no mapping needed

    def _get_task_index_for_global_idx(self, global_idx: int) -> int:
        """Get the task_index for a given global frame index."""
        if not self.episode_to_task:
            return 0
        row = self.lerobot_dataset.hf_dataset[global_idx]
        t_idx = row.get("task_index", 0)
        if isinstance(t_idx, torch.Tensor):
            t_idx = t_idx.item()
        return int(t_idx)

    def _build_valid_indices(self, episode_indices: Optional[list] = None):
        self.valid_indices = []
        self._index_to_episode_end = {}
        episodes = self.lerobot_dataset.meta.episodes

        for i in range(len(episodes)):
            ep = episodes[i]
            ep_idx = ep["episode_index"]
            if episode_indices is not None and ep_idx not in episode_indices:
                continue

            ep_start = ep["dataset_from_index"]
            ep_end = ep["dataset_to_index"]
            ep_len = ep_end - ep_start

            if not self.require_action_chunk:
                min_frames_needed = self.num_pixel_frames
            elif self.allow_partial_action_chunk:
                # Need full video window plus at least one future action.
                min_frames_needed = self.num_pixel_frames + 1
            else:
                min_frames_needed = self.num_pixel_frames + self.action_chunk_size
            if ep_len < min_frames_needed:
                continue

            for frame_offset in range(ep_len - min_frames_needed + 1):
                global_idx = ep_start + frame_offset
                self.valid_indices.append(global_idx)
                self._index_to_episode_end[global_idx] = ep_end

    def __len__(self) -> int:
        return len(self.valid_indices)

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        device = actions.device
        if self.action_norm_type == "min-max":
            if self.action_min is None or self.action_max is None:
                return actions
            # Normalize to [-1, 1]
            # First map to [0, 1]
            a_min = self.action_min.to(device)
            a_max = self.action_max.to(device)
            scaled = (actions - a_min) / (a_max - a_min + 1e-4) # 1e-4 prevents div by 0
            # Map to [-1, 1]
            return scaled * 2.0 - 1.0
        elif self.action_norm_type == "mean-std":
            if self.action_mean is None or self.action_std is None:
                return actions
            return (actions - self.action_mean.to(device)) / self.action_std.to(device)
        else:
            return actions

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        device = actions.device
        if self.action_norm_type == "min-max":
            if self.action_min is None or self.action_max is None:
                return actions
            a_min = self.action_min.to(device)
            a_max = self.action_max.to(device)
            # Reverse map from [-1, 1] to [0, 1]
            unscaled = (actions + 1.0) / 2.0
            # Map to original range
            return unscaled * (a_max - a_min + 1e-4) + a_min
        elif self.action_norm_type == "mean-std":
            if self.action_mean is None or self.action_std is None:
                return actions
            return actions * self.action_std.to(device) + self.action_mean.to(device)
        else:
            return actions

    def _get_proprio(self, global_idx: int) -> torch.Tensor:
        """Load proprio state at current frame."""
        # State at current frame
        row = self.lerobot_dataset.hf_dataset[global_idx]
        proprio_parts = []
        for sk in self.state_keys:
            val = row[sk]
            if not isinstance(val, torch.Tensor):
                val = torch.tensor(val, dtype=torch.float32)
            proprio_parts.append(val.float().flatten())
        proprio = torch.cat(proprio_parts)[:self.proprio_dim]
        return proprio

    def _get_action_chunk(self, global_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load action chunk and mask by directly indexing consecutive frames.

        Returns:
            actions: [action_chunk_size, action_dim]
            action_mask: [action_chunk_size, 1], 1 for valid targets, 0 for padded tail.
        """
        if global_idx not in self._index_to_episode_end:
            raise KeyError(f"global_idx={global_idx} not found in episode boundary map")
        ep_end = self._index_to_episode_end[global_idx]
        available_future = max(0, ep_end - global_idx - 1)
        valid_len = min(self.action_chunk_size, available_future)
        if valid_len <= 0:
            raise ValueError(
                f"No future actions available for global_idx={global_idx}; "
                "increase filtering constraints."
            )

        # Action chunk: gather from consecutive frames
        action_rows = []
        for offset in range(1, valid_len + 1):
            a_row = self.lerobot_dataset.hf_dataset[global_idx + offset]
            parts = []
            for ak in self.action_keys:
                val = a_row[ak]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32)
                parts.append(val.float().flatten())
            action_rows.append(torch.cat(parts))
        actions = torch.stack(action_rows)[:, :self.action_dim]  # [valid_len, action_dim]

        if valid_len < self.action_chunk_size:
            if self.allow_partial_action_chunk:
                # Pad with the last valid action; the mask ensures padded tail has zero loss.
                pad_count = self.action_chunk_size - valid_len
                pad_rows = actions[-1:].repeat(pad_count, 1)
                actions = torch.cat([actions, pad_rows], dim=0)
            else:
                raise ValueError(
                    f"Insufficient future actions at global_idx={global_idx}: "
                    f"valid_len={valid_len}, expected={self.action_chunk_size}"
                )

        action_mask = torch.zeros(self.action_chunk_size, 1, dtype=torch.float32)
        action_mask[:valid_len] = 1.0
        return actions, action_mask

    def compute_action_stats(self, max_samples: int = 10000) -> Dict[str, torch.Tensor]:
        """Compute mean and standard deviation of actions from the dataset."""
        if not self.require_action_chunk:
            raise ValueError("compute_action_stats requires require_action_chunk=True")
        num_samples = min(len(self.valid_indices), max_samples)
        indices = np.random.choice(len(self.valid_indices), num_samples, replace=False)
        
        all_actions = []
        for idx in indices:
            global_idx = self.valid_indices[idx]
            actions, action_mask = self._get_action_chunk(global_idx)
            valid = action_mask.squeeze(-1) > 0
            if valid.any():
                all_actions.append(actions[valid])

        if not all_actions:
            raise ValueError("No valid action rows collected for action stats.")
        all_actions = torch.cat(all_actions, dim=0)
        
        return {
            "mean": all_actions.mean(dim=0),
            "std": all_actions.std(dim=0).clamp(min=1e-4),
            "min": all_actions.min(dim=0)[0],
            "max": all_actions.max(dim=0)[0],
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        global_idx = self.valid_indices[idx]

        # Get video frames via lerobot (uses torchcodec for fast video decode)
        sample = self.lerobot_dataset[global_idx]

        camera_frames = []
        for cam_name in self.camera_names:
            frames = sample[cam_name]
            if frames.ndim == 3:
                frames = frames.unsqueeze(0)
            camera_frames.append(frames)

        video = concat_cameras(camera_frames, self.target_height, self.target_width)
        video = normalize_to_neg1_pos1(video)

        result = {
            "video": video,
            "proprio": self._get_proprio(global_idx),
        }

        # Stage-2 requires action chunk; Stage-1 can disable this to keep episode tail samples.
        if self.require_action_chunk:
            actions, action_mask = self._get_action_chunk(global_idx)
            result["actions"] = self.normalize_actions(actions)
            result["action_mask"] = action_mask

        # Multi-task: return per-sample T5 embedding based on task_index
        if self.t5_embeddings is not None:
            task_idx = self._get_task_index_for_global_idx(global_idx)
            if task_idx in self.t5_embeddings:
                result["t5_embedding"] = self.t5_embeddings[task_idx]
            else:
                # Fallback to first available embedding
                first_key = sorted(self.t5_embeddings.keys())[0]
                result["t5_embedding"] = self.t5_embeddings[first_key]
        elif self.t5_embedding is not None:
            result["t5_embedding"] = self.t5_embedding

        return result
