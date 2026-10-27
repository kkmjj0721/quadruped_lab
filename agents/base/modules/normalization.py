# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2020 Preferred Networks, Inc.
# Copyright (c) 2026-2027 zh
"""
内容：
    维护单设备观测的在线均值与总体方差。
"""

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """统计更新与前向归一化分离，保证每轮 PPO 使用固定的观测变换。"""

    def __init__(self, shape: int | tuple[int, ...] | list[int], eps: float = 1e-2, until: int | None = None) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if until is not None and until < 0:
            raise ValueError("until must be non-negative.")
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """根据已保存的统计量归一化，不隐式更新统计量。"""
        return (x - self._mean) / (self._std + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """用双精度合并批量统计，非法输入不会改变现有状态。"""
        if not self.training or (self.until is not None and self.count >= self.until):
            return
        if x.ndim != self._mean.ndim or x.shape[1:] != self._mean.shape[1:]:
            raise ValueError("Normalizer input shape does not match the configured feature shape.")
        if not torch.isfinite(x).all():
            raise FloatingPointError("Normalizer input contains non-finite values.")
        if x.shape[0] == 0:
            return
        variance_x, mean_x = torch.var_mean(x.double(), dim=0, unbiased=False, keepdim=True)
        next_count = self.count + x.shape[0]
        rate = x.shape[0] / next_count.double()
        delta = mean_x - self._mean.double()
        mean = self._mean.double() + rate * delta
        variance = (1 - rate) * self._var.double() + rate * variance_x + rate * (1 - rate) * delta.square()
        mean = mean.to(self._mean.dtype)
        variance = variance.to(self._var.dtype)
        std = variance.sqrt()
        if not (torch.isfinite(mean).all() and torch.isfinite(variance).all() and torch.isfinite(std).all()):
            raise FloatingPointError("Normalizer candidate is invalid; running statistics were not changed.")
        self.count.copy_(next_count)
        self._mean.copy_(mean)
        self._var.copy_(variance)
        self._std.copy_(std)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """恢复到观测原始尺度。"""
        return y * (self._std + self.eps) + self._mean
