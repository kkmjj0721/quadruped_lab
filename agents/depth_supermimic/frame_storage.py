# Copyright (c) 2026-2027 zh
"""内容：引用CPU原始深度帧，按请求行和时刻生成有界的FP32设备批次。"""

from __future__ import annotations

import torch
from torch import Tensor


class HostDepthFrames:
    """按行保留原始FP16帧引用，不分配整段前缀的设备像素张量。"""

    def __init__(self, rows: tuple[tuple[Tensor | None, ...], ...], image_shape):
        self.rows = tuple(tuple(row) for row in rows)
        self.image_shape = tuple(image_shape)
        self.lengths = tuple(len(row) for row in self.rows)
        self.shape = (len(self.rows), max(self.lengths, default=0), *self.image_shape)

    def index_rows(self, index):
        """保留切片、整型或布尔行索引的顺序及重复项，像素仍引用同一原始帧。"""
        if isinstance(index, slice):
            rows = self.rows[index]
        else:
            indices = index.tolist() if isinstance(index, Tensor) else index
            if isinstance(indices, int):
                indices = (indices,)
            if indices and isinstance(indices[0], bool):
                indices = [row for row, selected in enumerate(indices) if selected]
            rows = tuple(self.rows[row] for row in indices)
        return HostDepthFrames(rows, self.image_shape)

    def gather(self, rows, times, device):
        """仅搬运实际请求帧；空批次和显式None沿声明尺寸生成，不截断梯度。"""
        frames = [self.rows[row][time] for row, time in zip(rows, times, strict=True)]
        if not frames:
            return torch.empty((0, *self.image_shape), device=device, dtype=torch.float32)
        zero = torch.zeros(self.image_shape, dtype=torch.float16)
        pixels = torch.stack([zero if frame is None else frame for frame in frames])
        # 先传FP16再在设备上扩展，避免CPU转换和双倍像素传输。
        return pixels.to(device=device).float()
