# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026-2027 zh
"""
内容：
    将观测组拼接为前馈 actor 或 critic 的输入。
"""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from tensordict import TensorDict

from ..modules import EmpiricalNormalization, GaussianDistribution, MLP


class MLPModel(nn.Module):
    """支持可选观测归一化和高斯动作分布的前馈模型。"""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if obs_set not in obs_groups or not obs_groups[obs_set]:
            raise ValueError(f"Observation set '{obs_set}' must contain at least one group.")
        self.obs_groups = list(obs_groups[obs_set])
        self.obs_dim = 0
        for group in self.obs_groups:
            if group not in obs:
                raise ValueError(f"Observation group '{group}' was not found.")
            if obs[group].ndim != 2 or not obs[group].is_floating_point():
                raise ValueError(f"MLP observation group '{group}' must be a floating-point tensor of shape [N, D].")
            self.obs_dim += obs[group].shape[-1]
        self.obs_normalization = obs_normalization
        self.obs_normalizer = EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
        self.distribution: GaussianDistribution | None = None
        if distribution_cfg is not None:
            dist_cfg = deepcopy(distribution_cfg)
            class_name = dist_cfg.pop("class_name", "GaussianDistribution")
            allowed = (
                "GaussianDistribution", "agents.base.modules:GaussianDistribution",
                "agents.base.modules.distribution:GaussianDistribution",
            )
            if class_name not in allowed and class_name is not GaussianDistribution:
                raise ValueError(f"Unsupported distribution class: {class_name}.")
            self.distribution = GaussianDistribution(output_dim, **dist_cfg)
        self.mlp = MLP(self.obs_dim, output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(self, obs: TensorDict, stochastic_output: bool = False) -> torch.Tensor:
        """默认返回网络输出；策略显式请求随机输出时更新分布并采样。"""
        output = self.mlp(self.get_latent(obs))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def get_latent(self, obs: TensorDict) -> torch.Tensor:
        """按配置顺序拼接并归一化观测。"""
        return self.obs_normalizer(torch.cat([obs[group] for group in self.obs_groups], dim=-1))

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """前馈网络没有需要重置的回合状态。"""
        pass

    def _output_distribution(self) -> GaussianDistribution:
        if self.distribution is None:
            raise RuntimeError("This model has no output distribution.")
        return self.distribution

    @property
    def output_mean(self) -> torch.Tensor:
        return self._output_distribution().mean

    @property
    def output_std(self) -> torch.Tensor:
        return self._output_distribution().std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self._output_distribution().entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self._output_distribution().params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """计算动作在最近一次随机前向的策略分布下的对数概率。"""
        return self._output_distribution().log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """计算两组策略参数之间的 KL 散度。"""
        return self._output_distribution().kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        """从整轮采样更新归一化统计。"""
        if self.obs_normalization:
            self.obs_normalizer.update(torch.cat([obs[group] for group in self.obs_groups], dim=-1))
