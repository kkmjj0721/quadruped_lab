# Copyright (c) 2026-2027 zh
"""
内容：
    定义混合专家世界模型训练的状态、维度及真实转移协议。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields

import torch
from torch import Tensor


@dataclass(frozen=True)
class AlgorithmSpec:
    """维度之外，semantic_id必须标识坐标系、布局、动作及任务语义。"""

    state_dim: int
    delta_dim: int
    action_dim: int
    observation_dim: int
    privileged_dim: int
    context_dim: int
    memory_dim: int
    command_dim: int
    semantic_id: str
    history_length: int = 5
    model_id: str = "supermimic_moe_v1"

    def __post_init__(self):
        positive = (self.state_dim, self.delta_dim, self.action_dim,
                    self.observation_dim, self.privileged_dim, self.history_length)
        if any(value <= 0 for value in positive) or min(self.context_dim, self.memory_dim, self.command_dim) < 0:
            raise ValueError("Invalid algorithm dimensions")
        if not self.semantic_id or self.model_id != "supermimic_moe_v1":
            raise ValueError("Unsupported or missing algorithm semantic identifier")

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass
class TaskState:
    """末轴为特征；通常为[B,D]，窗口目标为[B,T,D]。"""

    physical: Tensor
    memory: Tensor
    command: Tensor
    context: Tensor

    def map(self, function) -> TaskState:
        return TaskState(**{field.name: function(getattr(self, field.name)) for field in fields(self)})

    def index(self, index) -> TaskState:
        return self.map(lambda tensor: tensor[index])

    def detached_clone(self) -> TaskState:
        return self.map(lambda tensor: tensor.detach().clone())


@dataclass
class EnvStep:
    """adapter先复制final_state，再reset；next_state为下一决策态。"""

    final_state: TaskState
    next_state: TaskState
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    context_version: Tensor


def append_history(history: Tensor, observation: Tensor) -> Tensor:
    """无原地修改地推进历史，保留想象中的跨步梯度。"""
    return torch.cat((history[:, 1:], observation.unsqueeze(1)), dim=1)
