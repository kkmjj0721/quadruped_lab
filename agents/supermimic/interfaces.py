# Copyright (c) 2026-2027 zh
"""
内容：
    定义与机器人无关的状态、真实转移、连续窗口及可微任务协议。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class AlgorithmSpec:
    """维度之外，semantic_id必须标识坐标系、布局、动作及任务语义。"""

    state_dim: int
    feature_dim: int
    delta_dim: int
    action_dim: int
    observation_dim: int
    privileged_dim: int
    context_dim: int
    memory_dim: int
    command_dim: int
    semantic_id: str
    history_length: int = 5
    latent_dim: int = 32
    model_id: str = "supermimic_cts_moe_v1"

    def __post_init__(self):
        positive = (self.state_dim, self.feature_dim, self.delta_dim, self.action_dim,
                    self.observation_dim, self.privileged_dim, self.history_length, self.latent_dim)
        if any(value <= 0 for value in positive) or min(self.context_dim, self.memory_dim, self.command_dim) < 0:
            raise ValueError("Invalid algorithm dimensions")
        if not self.semantic_id or self.model_id not in ("supermimic_cts_moe_v1", "supermimic_single_moe_v1"):
            raise ValueError("Unsupported or missing algorithm semantic identifier")

    @property
    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass
class TaskState:
    """末轴为特征；通常为[B,D]，窗口目标为[T,B,D]。"""

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
class PolicyInput:
    """branch=True为教师；历史包括当前已加噪帧，按旧到新排列。"""

    state: TaskState
    observation: Tensor
    history: Tensor
    privileged: Tensor
    branch: Tensor

    def index(self, index) -> PolicyInput:
        return PolicyInput(self.state.index(index), self.observation[index], self.history[index],
                           self.privileged[index], self.branch[index])

    def detached_clone(self) -> PolicyInput:
        return PolicyInput(self.state.detached_clone(), self.observation.detach().clone(),
                           self.history.detach().clone(), self.privileged.detach().clone(), self.branch.clone())


@dataclass
class Transition:
    """固定环境行的一个真实控制步；next必须在仿真reset前复制。"""

    current: PolicyInput
    next: PolicyInput
    action: Tensor
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    episode_id: Tensor
    step_id: Tensor
    context_version: Tensor


@dataclass
class ReplayWindow:
    """时间优先，终止末条有效，右侧补齐条目不参与监督。"""

    initial: PolicyInput
    actions: Tensor
    next_states: TaskState
    rewards: Tensor
    terminated: Tensor
    truncated: Tensor
    valid: Tensor


@dataclass
class EnvStep:
    """adapter先复制final_state，再reset；next_state为下一决策态。"""

    final_state: TaskState
    next_state: TaskState
    reward: Tensor
    terminated: Tensor
    truncated: Tensor
    context_version: Tensor


class TaskModel(Protocol):
    """实现必须保持Tensor输入导数，且不得引用仿真器可变缓存。"""

    spec: AlgorithmSpec

    def dynamics_features(self, state: TaskState) -> Tensor:
        """返回规范化物理特征；context由模型单独拼接。"""
        ...

    def integrate(self, state: TaskState, action: Tensor, delta: Tensor) -> Tensor:
        """将预测delta按任务物理积分规则转换为下一physical。"""
        ...

    def advance(self, state: TaskState, action: Tensor, next_physical: Tensor) -> TaskState:
        """解析推进记忆和目标；外生命令采样在模型之外完成。"""
        ...

    def observe(self, state: TaskState, *, noisy: bool = False) -> Tensor:
        """仅对新生成的当前帧加噪，不重复污染历史。"""
        ...

    def privileged(self, state: TaskState) -> Tensor:
        ...

    def reward(self, state: TaskState, action: Tensor, next_state: TaskState, failure: Tensor) -> Tensor:
        """使用转移起点命令计算该步奖励；失败量为实数概率。"""
        ...

    def state_error(self, predicted: TaskState, target: TaskState) -> Tensor:
        """返回[B]归一化误差；各物理分组及尺度由任务定义。"""
        ...


class CollectionEnv(Protocol):
    """真实采集adapter；所有输出必须与算法同设备、float32。"""

    num_envs: int
    device: str | torch.device

    def reset(self) -> TaskState:
        ...

    def step(self, actions: Tensor) -> EnvStep:
        ...


def append_history(history: Tensor, observation: Tensor) -> Tensor:
    """无原地修改地推进历史，保留想象中的跨步梯度。"""
    return torch.cat((history[:, 1:], observation.unsqueeze(1)), dim=1)


def validate_policy_input(data: PolicyInput, spec: AlgorithmSpec) -> None:
    """在采集／存储边界检查协议，不在每层网络中隐式搬运数据。"""
    batch = data.observation.shape[0]
    expected = {
        "physical": spec.state_dim, "memory": spec.memory_dim,
        "command": spec.command_dim, "context": spec.context_dim,
    }
    tensors = []
    for name, width in expected.items():
        tensor = getattr(data.state, name)
        if tensor.shape != (batch, width):
            raise ValueError(f"Invalid state shape for {name}")
        tensors.append(tensor)
    for tensor, shape in ((data.observation, (batch, spec.observation_dim)),
                          (data.history, (batch, spec.history_length, spec.observation_dim)),
                          (data.privileged, (batch, spec.privileged_dim))):
        if tensor.shape != shape:
            raise ValueError("Invalid policy feature shape")
        tensors.append(tensor)
    if data.branch.shape != (batch,) or data.branch.dtype != torch.bool:
        raise ValueError("Branch must be a boolean batch vector")
    if any(tensor.device != data.observation.device or tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("Policy features must use float32 on one device")
    if data.branch.device != data.observation.device:
        raise ValueError("Branch and policy features must use one device")
