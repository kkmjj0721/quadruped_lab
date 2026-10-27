# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026-2027 zh
"""
内容：
    定义基础 PPO 所需的向量环境结构协议。
"""

from __future__ import annotations

from typing import Protocol

import torch
from tensordict import TensorDict


class VecEnv(Protocol):
    """环境负责终止实例的重置，并返回相同批量大小的下一步观测。"""

    num_envs: int
    num_actions: int
    device: torch.device | str
    max_episode_length: int | torch.Tensor
    episode_length_buf: torch.Tensor

    def get_observations(self) -> TensorDict:
        """返回批量为 [num_envs]、各观测组为 [num_envs, D] 的观测。"""
        ...

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """接收 [N, A] 动作，返回观测、[N] 奖励、[N] 终止和附加信息。

        extras 的可选 time_outs 字段为 [N] 超时掩码，用于价值自举补偿。
        """
        ...
