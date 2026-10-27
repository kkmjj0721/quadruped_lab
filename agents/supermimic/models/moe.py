# Copyright (c) 2026-2027 zh
"""
内容：
    用独立门控和八专家密集混合编码本体历史，混合后统一归一化。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .encoders import make_mlp


class StudentMoEEncoder(nn.Module):
    """专家输出为表征；不使用稀疏路由，也不预设专家技能标签。"""

    def __init__(
        self, history_dim: int, latent_dim: int = 32, hidden_dims: Sequence[int] = (512, 256, 256),
        num_experts: int = 8,
    ):
        super().__init__()
        if num_experts != 8:
            raise ValueError("The SuperMimic CTS student requires exactly eight experts")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("Student hidden dimensions must be positive and non-empty")
        self.history_dim = history_dim
        self.latent_dim = latent_dim
        self.num_experts = num_experts
        self.expert_width = hidden_dims[-1]
        self.backbone = make_mlp(history_dim, hidden_dims[:-1], num_experts * self.expert_width, final_elu=True)
        self.projection = nn.Conv1d(num_experts * self.expert_width, num_experts * latent_dim, 1, groups=num_experts)
        self.gate = make_mlp(history_dim, hidden_dims[:-1], num_experts)

    def _flatten(self, history: Tensor) -> Tensor:
        if history.ndim not in (2, 3):
            raise ValueError("History must have shape [B,H,O] or [B,H*O]")
        flattened = history.flatten(1)
        if flattened.shape[1] != self.history_dim:
            raise ValueError("History width does not match the student encoder")
        return flattened

    def expert_outputs(self, history: Tensor) -> Tensor:
        """返回[B,8,L]的原始专家结果，用于数值对拍和表征诊断。"""
        features = self.backbone(self._flatten(history)).unsqueeze(-1)
        return self.projection(features).squeeze(-1).reshape(-1, self.num_experts, self.latent_dim)

    def routing_weights(self, history: Tensor) -> Tensor:
        return self.gate(self._flatten(history)).softmax(dim=-1)

    def forward(self, history: Tensor) -> tuple[Tensor, Tensor]:
        weights = self.routing_weights(history)
        mixture = (weights.unsqueeze(-1) * self.expert_outputs(history)).sum(dim=1)
        return F.normalize(mixture, dim=-1, eps=1e-8), weights
