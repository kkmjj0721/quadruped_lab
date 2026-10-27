# Copyright (c) 2026-2027 zh
"""内容：提供盲 MoE actor、特权 critic 和带输入／残差归一化的动力学模型。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..interfaces import TaskState
from ..models.encoders import make_mlp
from ..models.moe import StudentMoEEncoder


class RunningNorm(nn.Module):
    """按参考实现合并总体方差；前向固定统计，不隐式更新。"""

    def __init__(self, width: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, values: Tensor):
        values = values.reshape(-1, self.mean.numel())
        if values.shape[0] == 0:
            return
        if not torch.isfinite(values).all():
            raise FloatingPointError("Normalization data contains non-finite values")
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        count = values.shape[0]
        total = self.count + count
        difference = mean - self.mean
        self.var.copy_((self.var * self.count + variance * count + difference.square() * self.count * count / total) / total)
        self.mean.add_(difference * count / total)
        self.count.copy_(total)

    def forward(self, values: Tensor) -> Tensor:
        return ((values - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-10.0, 10.0)

    def unnormalize(self, values: Tensor) -> Tensor:
        return values * (self.var.sqrt() + 1e-6) + self.mean


class Go2MoEEncoder(StudentMoEEncoder):
    """保持 go2_rl_gym 的默认 L2 epsilon，不改变旧 CTS 编码器。"""

    def forward(self, history: Tensor) -> tuple[Tensor, Tensor]:
        weights = self.routing_weights(history)
        mixed = (weights.unsqueeze(-1) * self.expert_outputs(history)).sum(dim=1)
        return F.normalize(mixed, dim=-1, eps=1e-12), weights


class MoEActor(nn.Module):
    """仅接收本体历史，采用 go2_rl_gym 的 latent 在前、当前帧在后结构。"""

    def __init__(self, spec, config):
        super().__init__()
        self.observation_dim = spec.observation_dim
        self.history_length = spec.history_length
        self.action_dim = spec.action_dim
        self.action_limit = config.action_limit
        self.normalizer = RunningNorm(spec.observation_dim)
        self.encoder = Go2MoEEncoder(spec.observation_dim * spec.history_length, spec.latent_dim, config.moe_hidden)
        self.actor = make_mlp(spec.latent_dim + spec.observation_dim, config.actor_hidden, spec.action_dim)

    def action_and_gates(self, history: Tensor) -> tuple[Tensor, Tensor]:
        frames = history.reshape(-1, self.history_length, self.observation_dim)
        normalized = self.normalizer(frames)
        latent, gates = self.encoder(normalized)
        raw = self.actor(torch.cat((latent, normalized[:, -1]), dim=-1))
        limit = self.action_limit
        return (raw if limit is None else limit * torch.tanh(raw / limit)), gates

    def forward(self, history: Tensor) -> Tensor:
        return self.action_and_gates(history)[0]

    def student_action(self, observation: Tensor, history: Tensor) -> Tensor:
        """兼容已有盲策略播放接口；当前帧已包含在历史末尾。"""
        return self(history)


class Critic(nn.Module):
    def __init__(self, width: int, hidden):
        super().__init__()
        self.normalizer = RunningNorm(width)
        self.net = make_mlp(width, hidden, 1)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(self.normalizer(features)).squeeze(-1)


class WorldModel(nn.Module):
    """任务决定残差坐标；网络只学习记录物理转移，不学习奖励或失败头。"""

    def __init__(self, task, hidden):
        super().__init__()
        self.task = task
        self.spec = task.spec
        layers, width = [], task.model_input_dim
        for hidden_width in hidden:
            layers.extend((nn.Linear(width, hidden_width), nn.SiLU()))
            width = hidden_width
        layers.append(nn.Linear(width, task.spec.delta_dim))
        self.net = nn.Sequential(*layers)
        self.input_norm = RunningNorm(task.model_input_dim)
        self.output_norm = RunningNorm(task.spec.delta_dim)

    @torch.no_grad()
    def update_stats(self, state: TaskState, action: Tensor, target: TaskState):
        self.input_norm.update(self.task.model_input(state, action))
        self.output_norm.update(self.task.model_target(state, target))

    def forward(self, state: TaskState, action: Tensor) -> TaskState:
        delta = self.output_norm.unnormalize(self.net(self.input_norm(self.task.model_input(state, action))))
        physical = self.task.integrate(state, action, delta)
        return self.task.advance(state, action, physical)

    def loss(self, window) -> Tensor:
        current = window.initial
        scale = self.output_norm.var.sqrt() + 1e-6
        total = current.physical.new_zeros(())
        for step in range(window.actions.shape[1]):
            target = window.targets.index((slice(None), step))
            action = window.actions[:, step]
            predicted = self(current, action)
            error = (self.task.model_target(current, predicted) - self.task.model_target(current, target)) / scale
            total = total + (error.square().mean(-1) * window.valid[:, step]).sum()
            # 本协议的记忆只含记录动作与外生计时，按日志恢复命令重采样；物理状态始终回喂预测。
            current = TaskState(predicted.physical, target.memory, target.command, target.context)
        return total / window.valid.sum().clamp_min(1)
