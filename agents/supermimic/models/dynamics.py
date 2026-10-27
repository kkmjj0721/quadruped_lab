# Copyright (c) 2026-2027 zh
"""
内容：
    根据任务声明的物理特征学习残差动力学及下一状态失败概率。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from agents.supermimic.interfaces import TaskModel, TaskState


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    """构造带ELU的确定性网络，不引入动作裁剪。"""
    if any(width <= 0 for width in hidden_dims):
        raise ValueError("Hidden dimensions must be positive")
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class DynamicsModel(nn.Module):
    """仅学习物理delta；坐标积分、目标与记忆推进由任务提供。"""

    def __init__(self, task: TaskModel, hidden_dims: tuple[int, ...] = (512, 512, 512)):
        super().__init__()
        self.task = task
        self.spec = task.spec
        spec = self.spec
        self.dynamics = _mlp(spec.feature_dim + spec.action_dim + spec.context_dim, hidden_dims, spec.delta_dim)
        self.termination_head = _mlp(spec.feature_dim + spec.command_dim + spec.context_dim, hidden_dims[-1:], 1)

    def predict(self, state: TaskState, action: Tensor) -> Tensor:
        """输出可微下一物理状态，保留对动作及物理输入的导数。"""
        features = self.task.dynamics_features(state)
        delta = self.dynamics(torch.cat((features, action, state.context), dim=-1))
        return self.task.integrate(state, action, delta)

    def forward(self, state: TaskState, action: Tensor) -> TaskState:
        """闭环想象采用任务解析推进；不读取日志中的未来状态。"""
        return self.task.advance(state, action, self.predict(state, action))

    def supervised_step(self, state: TaskState, action: Tensor, target: TaskState) -> TaskState:
        """推进监督窗口；解析记忆及外生条件取日志，学习状态保持自回归。"""
        return TaskState(self.predict(state, action), target.memory.detach(), target.command.detach(), target.context.detach())

    def supervised_auxiliary_loss(self, state: TaskState, action: Tensor,
                                  predicted: TaskState, target: TaskState) -> Tensor:
        """监督supervised_step生成的辅助状态；基础模型没有附加损失。"""
        del state, action, target
        return predicted.physical.new_zeros(predicted.physical.shape[0])

    def termination_logits(self, next_state: TaskState) -> Tensor:
        """失败头在预测的下一状态求值，不把人工超时当作失败。"""
        features = self.task.dynamics_features(next_state)
        inputs = torch.cat((features, next_state.command, next_state.context), dim=-1)
        return self.termination_head(inputs).squeeze(-1)

    def termination(self, next_state: TaskState) -> Tensor:
        return torch.sigmoid(self.termination_logits(next_state))
