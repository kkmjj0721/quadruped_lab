# Copyright (c) 2026-2027 zh
"""内容：以独立设备统计归一化真实样本，前向不隐式更新统计。"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

class NormStatistics(NamedTuple):
    """设备上的批次总体统计；finite仅覆盖参与累计的样本。"""

    count: Tensor
    mean: Tensor
    variance: Tensor
    finite: Tensor


class RunningNorm(nn.Module):
    """按参考实现合并总体方差；前向固定统计，不隐式更新。"""

    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def prepare_update(self, values: Tensor, mask: Tensor | None = None) -> NormStatistics:
        """固定形状归约，不读取设备条件，也不修改已有统计。"""
        values = values.reshape(-1, self.mean.numel())
        if mask is not None and (mask.shape != (values.shape[0],) or mask.dtype != torch.bool
                                 or mask.device != values.device):
            raise ValueError("Normalization mask must be boolean [N] on the values device")
        if values.shape[0] == 0:
            return NormStatistics(self.count.new_zeros(()), torch.zeros_like(self.mean), torch.zeros_like(self.var),
                                  torch.ones((), dtype=torch.bool, device=values.device))
        if mask is None:
            variance, mean = torch.var_mean(values, dim=0, correction=0)
            return NormStatistics(self.count.new_full((), values.shape[0]), mean, variance, torch.isfinite(values).all())
        count = mask.sum().to(self.count.dtype)
        safe = torch.where(mask[:, None], values, 0)
        mean = safe.sum(dim=0) / count.clamp_min(1)
        centered = torch.where(mask[:, None], values - mean, 0)
        variance = centered.square().sum(dim=0) / count.clamp_min(1)
        return NormStatistics(count, mean, variance, torch.isfinite(safe).all())

    @torch.no_grad()
    def apply_update(self, statistics: NormStatistics):
        """调用方已统一检查finite；原位提交且空批次不改变任何统计。"""
        count, mean, variance, _ = statistics
        total = self.count + count
        difference = mean - self.mean
        updated_var = (self.var * self.count + variance * count + difference.square() * self.count * count / total) / total
        updated_mean = self.mean + difference * count / total
        active = count > 0
        self.var.copy_(torch.where(active, updated_var, self.var))
        self.mean.copy_(torch.where(active, updated_mean, self.mean))
        self.count.copy_(torch.where(active, total, self.count))

    @torch.no_grad()
    def update(self, values: Tensor, mask: Tensor | None = None):
        """独立调用入口保留即时有限性检查；采集使用准备和提交接口合并检查。"""
        statistics = self.prepare_update(values, mask)
        if not statistics.finite:
            raise FloatingPointError("Normalization data contains non-finite values")
        self.apply_update(statistics)

    def forward(self, values: Tensor) -> Tensor:
        return ((values - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-10.0, 10.0)

    def unnormalize(self, values: Tensor) -> Tensor:
        return values * (self.var.sqrt() + 1e-6) + self.mean


