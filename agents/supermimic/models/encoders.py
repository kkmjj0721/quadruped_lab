# Copyright (c) 2026-2027 zh
"""
内容：
    提供教师表征编码器、网络装配及显式更新的观测统计。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def make_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int, *, final_elu: bool = False) -> nn.Sequential:
    """构造隐藏层使用ELU的前馈网络，不添加动作裁剪。"""
    widths = (input_dim, *hidden_dims, output_dim)
    if any(width <= 0 for width in widths):
        raise ValueError("Network dimensions must be positive")
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(left, right))
        if index < len(widths) - 2 or final_elu:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class RunningMoments(nn.Module):
    """前向只读取统计；支持当前帧和历史帧广播同一变换。"""

    def __init__(self, width: int, epsilon: float = 1e-5):
        super().__init__()
        if width <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Normalizer width and epsilon must be positive")
        self.width = width
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("variance", torch.ones(width))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.width:
            raise ValueError("Normalizer feature width does not match")
        return (features - self.mean) * (self.variance + self.epsilon).rsqrt()

    @torch.no_grad()
    def update(self, features: Tensor) -> None:
        """合并批量总体方差，先检查候选统计再写入缓冲区。"""
        if features.ndim != 2 or features.shape[1] != self.width:
            raise ValueError("Normalizer update requires a matrix of frames")
        if features.device != self.mean.device:
            raise ValueError("Normalizer and real observations must share a device")
        if not torch.isfinite(features).all():
            raise FloatingPointError("Real observations contain non-finite values")
        if not self.training or features.shape[0] == 0:
            return
        batch_variance, batch_mean = torch.var_mean(features.double(), dim=0, correction=0)
        total = self.count + features.shape[0]
        fraction = features.shape[0] / total.double()
        difference = batch_mean - self.mean.double()
        candidate_mean = self.mean.double() + fraction * difference
        candidate_variance = ((1 - fraction) * self.variance.double() + fraction * batch_variance
                              + fraction * (1 - fraction) * difference.square())
        candidate_mean = candidate_mean.to(self.mean.dtype)
        candidate_variance = candidate_variance.to(self.variance.dtype)
        if not (torch.isfinite(candidate_mean).all() and torch.isfinite(candidate_variance).all()):
            raise FloatingPointError("Normalizer candidate statistics are invalid")
        self.mean.copy_(candidate_mean)
        self.variance.copy_(candidate_variance.clamp_min(0))
        self.count.copy_(total)


class TeacherEncoder(nn.Module):
    """从当前特权特征提取L2归一化的确定性表征。"""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Sequence[int] = (512, 256)):
        super().__init__()
        self.network = make_mlp(input_dim, hidden_dims, latent_dim)

    def forward(self, privileged: Tensor) -> Tensor:
        return F.normalize(self.network(privileged), dim=-1, eps=1e-8)
