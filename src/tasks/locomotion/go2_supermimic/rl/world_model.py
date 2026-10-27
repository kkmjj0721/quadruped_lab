# Copyright (c) 2026-2027 zh
"""
内容：
    预测Go2接触奖励与持续时间，并保护想象轨迹的静态地形支持域。
"""

import math

import torch
from torch.nn import functional as F

from agents.supermimic import DynamicsModel
from agents.supermimic.interfaces import TaskState
from agents.supermimic.models.dynamics import _mlp

from ..mdp.state_spec import REWARD_AUXILIARY


class Go2DynamicsModel(DynamicsModel):
    """接触奖励来自真实监督；越界想象显式拒绝，不复用真实终态占位。"""

    def __init__(self, task, hidden_dims=(512, 512, 512)):
        super().__init__(task, hidden_dims)
        scales = torch.as_tensor(task.auxiliary_scales, dtype=torch.float32).detach().clone()
        width = REWARD_AUXILIARY.stop - REWARD_AUXILIARY.start
        if scales.shape != (width,) or not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError("Go2 reward auxiliary scales must contain 18 positive finite values")
        self.register_buffer("auxiliary_scales", scales)
        self.auxiliary_loss_weight = float(task.cfg.model.auxiliary_loss_weight)
        if not math.isfinite(self.auxiliary_loss_weight) or self.auxiliary_loss_weight < 0:
            raise ValueError("Go2 auxiliary loss weight must be finite and nonnegative")
        decimation = task.cfg.control.decimation
        if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1:
            raise ValueError("Go2 control decimation must be a positive integer")
        self.register_buffer("bounded_auxiliary_indices", torch.tensor((3, 4, 7, 8, 9), device=scales.device))
        # 原始奖励上界来自定义，不能随模型归一化尺度一起变化。
        self.register_buffer("auxiliary_upper_bounds", scales.new_tensor((1., 1., 1., float(decimation), 1.)))
        spec = self.spec
        input_dim = 2 * spec.feature_dim + spec.memory_dim + spec.command_dim + spec.context_dim + spec.action_dim
        self.auxiliary_head = _mlp(input_dim, hidden_dims[-2:], width)

    def normalized_auxiliary_prediction(self, state, action, predicted):
        """终点仅提供物理／地形特征，禁止将终点奖励和计时标签作为输入。"""
        memory = torch.cat((state.memory[:, :REWARD_AUXILIARY.start],
                            state.memory[:, REWARD_AUXILIARY] / self.auxiliary_scales), dim=-1)
        features = torch.cat((self.task.dynamics_features(state), memory, state.command, state.context,
                              action, self.task.dynamics_features(predicted)), dim=-1)
        logits = self.auxiliary_head(features)
        normalized = F.softplus(logits)
        indices = self.bounded_auxiliary_indices
        bounded = logits.index_select(1, indices).sigmoid() * self.auxiliary_upper_bounds
        bounded = bounded / self.auxiliary_scales.index_select(0, indices)
        return normalized.index_copy(1, indices, bounded)

    def supervised_auxiliary_loss(self, state, action, predicted, target):
        """复用监督推进的预测，不重复调用辅助头或读取终态标签作为输入。"""
        del state, action
        normalized = predicted.memory[:, REWARD_AUXILIARY] / self.auxiliary_scales
        expected = target.memory[:, REWARD_AUXILIARY].detach() / self.auxiliary_scales
        return self.auxiliary_loss_weight * (normalized - expected).square().mean(-1)

    def _with_auxiliary(self, state, action, predicted):
        auxiliary = self.normalized_auxiliary_prediction(state, action, predicted) * self.auxiliary_scales
        memory = torch.cat((predicted.memory[:, :REWARD_AUXILIARY.start], auxiliary), dim=-1)
        return TaskState(predicted.physical, memory, predicted.command, predicted.context)

    def supervised_step(self, state, action, target):
        """辅助记忆随预测闭环；真实失败终态监督不使用想象支持域断言。"""
        return self._with_auxiliary(state, action, super().supervised_step(state, action, target))

    def forward(self, state, action):
        predicted = super().forward(state, action)
        xy = self.task.world_position(predicted)[:, :2]
        valid = self.task.surface.contains(xy, margin=self.task.cfg.terrain.boundary_margin_m)
        if not valid.all():
            raise ValueError("Imagined Go2 state leaves terrain support; shorten the horizon or resample seeds")
        return self._with_auxiliary(state, action, predicted)
