# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026-2027 zh
"""
内容：
    保存前馈 PPO 的整轮采样并生成完整覆盖样本的小批量。
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

import torch
from tensordict import TensorDict


class RolloutStorage:
    """按 [采样步, 环境, 特征] 布局存放 PPO 更新数据。"""

    class Transition:
        """在采样过程中逐步填充的一次状态转移。"""

        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor | None = None
            self.distribution_params: tuple[torch.Tensor, ...] | None = None

        def clear(self) -> None:
            """清除本次转移的引用，准备下一次采样。"""
            self.__init__()

    @dataclass
    class Batch:
        """PPO 优化所需的一组样本。"""

        observations: TensorDict
        actions: torch.Tensor
        values: torch.Tensor
        advantages: torch.Tensor
        returns: torch.Tensor
        old_actions_log_prob: torch.Tensor
        old_distribution_params: tuple[torch.Tensor, ...]

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        if training_type != "rl":
            raise ValueError("RolloutStorage only supports training_type='rl'.")
        if num_envs <= 0 or num_transitions_per_env <= 0:
            raise ValueError("num_envs and num_transitions_per_env must be positive.")
        if tuple(obs.batch_size) != (num_envs,):
            raise ValueError("Observation batch size must equal [num_envs].")
        if len(actions_shape) != 1 or actions_shape[0] <= 0:
            raise ValueError("actions_shape must contain one positive action dimension.")
        self.training_type = training_type
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.actions_shape = tuple(actions_shape)
        self.observations = TensorDict(
            {
                key: torch.zeros(num_transitions_per_env, *value.shape, dtype=value.dtype, device=device)
                for key, value in obs.items()
            },
            batch_size=[num_transitions_per_env, num_envs],
            device=device,
        )
        scalar_shape = (num_transitions_per_env, num_envs, 1)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.rewards = torch.zeros(*scalar_shape, device=device)
        self.dones = torch.zeros(*scalar_shape, device=device, dtype=torch.bool)
        self.values = torch.zeros(*scalar_shape, device=device)
        self.actions_log_prob = torch.zeros(*scalar_shape, device=device)
        self.returns = torch.zeros(*scalar_shape, device=device)
        self.advantages = torch.zeros(*scalar_shape, device=device)
        self.distribution_params: tuple[torch.Tensor, ...] | None = None
        self.step = 0

    @torch.no_grad()
    def add_transition(self, transition: Transition) -> None:
        """复制转移数据，避免环境随后复用张量影响已保存样本。"""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow; call clear() before adding more transitions.")
        required = ("observations", "actions", "rewards", "dones", "values", "actions_log_prob", "distribution_params")
        if any(getattr(transition, name) is None for name in required):
            raise ValueError("Transition must contain all PPO fields.")
        if not transition.distribution_params:
            raise ValueError("Transition distribution_params must not be empty.")
        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(self.num_transitions_per_env, *param.shape, dtype=param.dtype, device=self.device)
                for param in transition.distribution_params
            )
        if len(self.distribution_params) != len(transition.distribution_params):
            raise ValueError("Distribution parameter count changed within the rollout.")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.reshape(self.num_envs, 1))
        self.dones[self.step].copy_(transition.dones.reshape(self.num_envs, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.reshape(self.num_envs, 1))
        for saved, param in zip(self.distribution_params, transition.distribution_params):
            saved[self.step].copy_(param)
        self.step += 1

    def clear(self) -> None:
        """仅重置写入位置，下一轮会覆盖旧数据。"""
        self.step = 0

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[Batch, None, None]:
        """每轮重新打乱样本，不能整除时将余数分配至前几个批次。"""
        if self.step != self.num_transitions_per_env or self.distribution_params is None:
            raise RuntimeError("A complete rollout is required before generating mini-batches.")
        batch_size = self.num_envs * self.num_transitions_per_env
        if not 1 <= num_mini_batches <= batch_size or num_epochs <= 0:
            raise ValueError("num_mini_batches must be in [1, batch_size] and num_epochs must be positive.")
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_distribution_params = tuple(param.flatten(0, 1) for param in self.distribution_params)
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for batch_indices in torch.tensor_split(indices, num_mini_batches):
                yield self.Batch(
                    observations=observations[batch_indices],
                    actions=actions[batch_indices],
                    values=values[batch_indices],
                    advantages=advantages[batch_indices],
                    returns=returns[batch_indices],
                    old_actions_log_prob=old_actions_log_prob[batch_indices],
                    old_distribution_params=tuple(param[batch_indices] for param in old_distribution_params),
                )
