# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026-2027 zh
"""
内容：
    提供状态无关标准差的对角高斯策略分布。
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal


class GaussianDistribution(nn.Module):
    """支持直接或对数标准差参数，沿动作维度求和概率和熵。"""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        super().__init__()
        if output_dim <= 0 or not math.isfinite(init_std) or init_std <= 0:
            raise ValueError("output_dim and init_std must be positive and finite.")
        if len(std_range) != 2 or not all(math.isfinite(value) for value in std_range):
            raise ValueError("std_range must contain two finite values.")
        if not 0 < std_range[0] <= std_range[1]:
            raise ValueError("std_range must satisfy 0 < lower <= upper.")
        if std_type not in ("scalar", "log"):
            raise ValueError("std_type must be 'scalar' or 'log'.")
        self.output_dim = output_dim
        self.std_type = std_type
        self.std_range = (max(std_range[0], 1e-6), std_range[1])
        if self.std_range[0] > self.std_range[1]:
            raise ValueError("std_range upper bound must be at least 1e-6.")
        self.log_std_range = tuple(math.log(value) for value in self.std_range)
        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim), requires_grad=learn_std)
        else:
            self.log_std_param = nn.Parameter(torch.full((output_dim,), math.log(init_std)), requires_grad=learn_std)
        self._distribution: Normal | None = None

    def __getstate__(self) -> dict:
        """复制模型时丢弃瞬时分布，避免复制最近一次前向的计算图。"""
        state = super().__getstate__()
        state["_distribution"] = None
        return state

    def update(self, mlp_output: torch.Tensor) -> None:
        """以网络输出为均值，更新当前批量分布。"""
        if self.std_type == "scalar":
            std = self.std_param.clamp(*self.std_range)
        else:
            std = self.log_std_param.clamp(*self.log_std_range).exp()
        self._distribution = Normal(mlp_output, std, validate_args=False)

    def _current(self) -> Normal:
        """在访问采样结果前检查分布是否已初始化。"""
        if self._distribution is None:
            raise RuntimeError("Call update() before accessing the action distribution.")
        return self._distribution

    def sample(self) -> torch.Tensor:
        """从当前策略分布采样动作。"""
        return self._current().sample()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """确定性动作直接使用网络均值。"""
        return mlp_output

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._current().mean

    @property
    def std(self) -> torch.Tensor:
        return self._current().stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self._current().entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean, self.std

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """计算各动作维度的联合对数概率。"""
        return self._current().log_prob(outputs).sum(dim=-1)

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """计算 KL(旧策略 || 新策略)。"""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old = Normal(old_mean, old_std, validate_args=False)
        new = Normal(new_mean, new_std, validate_args=False)
        return torch.distributions.kl_divergence(old, new).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """该分布沿用前馈网络的默认权重初始化。"""
        pass
