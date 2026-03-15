"""Video transform utilities for the data pipeline.

Extends the original with a more explicit API: transforms are callable
objects so they can be composed with ``torchvision.transforms.Compose``
and individually unit-tested.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


class ConcatCameras:
    """Stack multiple camera views into a 2×N grid.

    Views are tiled left-to-right, top-to-bottom (row-major order).
    All input views must share the same ``(C, H, W)`` per-frame dimensions.

    Args:
        target_height: Final grid height in pixels.
        target_width: Final grid width in pixels.
        cols: Number of columns in the tile grid.  Views are wrapped
            automatically if ``len(views) > cols``.
    """

    def __init__(self, target_height: int = 480, target_width: int = 640, cols: int = 2) -> None:
        self.target_height = target_height
        self.target_width = target_width
        self.cols = cols

    def __call__(self, views: List[torch.Tensor]) -> torch.Tensor:
        """Merge camera views.

        Args:
            views: List of ``[T, C, H_i, W_i]`` tensors (one per camera).

        Returns:
            Tiled tensor ``[T, C, target_height, target_width]``.
        """
        n = len(views)
        T, C, _, _ = views[0].shape

        # Resize each view to the per-cell size
        rows = (n + self.cols - 1) // self.cols
        cell_h = self.target_height // rows
        cell_w = self.target_width // self.cols

        resized = []
        for v in views:
            v_flat = v.flatten(0, 0).float()  # [T, C, H, W]
            v_r = F.interpolate(v_flat, size=(cell_h, cell_w), mode="bilinear", align_corners=False)
            resized.append(v_r)  # [T, C, cell_h, cell_w]

        # Pad to full rows × cols grid
        while len(resized) < rows * self.cols:
            resized.append(torch.zeros_like(resized[0]))

        row_tensors = []
        for r in range(rows):
            row = torch.cat(resized[r * self.cols: (r + 1) * self.cols], dim=-1)  # [T, C, cell_h, W]
            row_tensors.append(row)
        grid = torch.cat(row_tensors, dim=-2)  # [T, C, H, W]

        # Final resize to exact target resolution
        grid = F.interpolate(grid.float(), size=(self.target_height, self.target_width),
                             mode="bilinear", align_corners=False)
        return grid  # [T, C, target_height, target_width]


class NormalizeToUnitRange:
    """Rescale pixel values from ``[0, 255]`` (uint8) or ``[0, 1]`` (float) to ``[-1, 1]``."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return x * 2.0 - 1.0


class RandomHorizontalFlipVideo:
    """Random horizontal flip applied consistently across all frames.

    Args:
        p: Probability of flipping.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """Apply or skip flipping.

        Args:
            video: ``[T, C, H, W]``.

        Returns:
            Optionally flipped tensor, same shape.
        """
        if torch.rand(1).item() < self.p:
            return video.flip(-1)
        return video
