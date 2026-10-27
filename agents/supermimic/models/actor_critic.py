# Copyright (c) 2026-2027 zh
"""
内容：
    装配并发教师与MoE学生、共享动作网络及带分支信息的公共价值网络。
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..interfaces import AlgorithmSpec, PolicyInput
from .encoders import RunningMoments, TeacherEncoder, make_mlp
from .moe import StudentMoEEncoder


class ActorCritic(nn.Module):
    """所有前向保留输入导数，参数冻结由各优化阶段的调用者负责。"""

    def __init__(
        self, spec: AlgorithmSpec, actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128), teacher_hidden_dims: Sequence[int] = (512, 256),
        student_hidden_dims: Sequence[int] = (512, 256, 256), normalize_observations: bool = True,
        action_limit: float | None = None,
    ):
        super().__init__()
        self.spec = spec
        self.normalize_observations = normalize_observations
        if action_limit is not None and (isinstance(action_limit, bool) or not isinstance(action_limit, (int, float))
                                         or not math.isfinite(action_limit) or action_limit <= 0):
            raise ValueError("Action limit must be a positive finite number or None")
        self.action_limit = float(action_limit) if action_limit is not None else None
        self._hidden_dimensions = {
            "actor_hidden_dims": tuple(actor_hidden_dims), "critic_hidden_dims": tuple(critic_hidden_dims),
            "teacher_hidden_dims": tuple(teacher_hidden_dims), "student_hidden_dims": tuple(student_hidden_dims),
        }
        self.observation_normalizer = RunningMoments(spec.observation_dim)
        self.privileged_normalizer = RunningMoments(spec.privileged_dim)
        history_dim = spec.history_length * spec.observation_dim
        self.teacher_encoder = TeacherEncoder(spec.privileged_dim, spec.latent_dim, teacher_hidden_dims)
        self.student_encoder = StudentMoEEncoder(history_dim, spec.latent_dim, student_hidden_dims)
        self.actor = make_mlp(spec.observation_dim + spec.latent_dim, actor_hidden_dims, spec.action_dim)
        self.critic = make_mlp(spec.privileged_dim + history_dim + spec.latent_dim + 2, critic_hidden_dims, 1)

    @property
    def construction_config(self) -> dict:
        """供检查点校验网络结构及归一化语义，返回独立的配置副本。"""
        config = {**self._hidden_dimensions, "normalize_observations": self.normalize_observations}
        if self.action_limit is not None:
            config["action_limit"] = self.action_limit
        return config

    def transform_action(self, action: Tensor) -> Tensor:
        """教师、学生及诊断共用动作域；平滑限制在零点保持单位导数。"""
        return action if self.action_limit is None else self.action_limit * torch.tanh(action / self.action_limit)

    def normalize_observation(self, observation: Tensor) -> Tensor:
        return self.observation_normalizer(observation) if self.normalize_observations else observation

    def normalize_history(self, history: Tensor) -> Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.spec.history_length, self.spec.observation_dim):
            raise ValueError("Policy history must use the configured time-major frame layout")
        return self.normalize_observation(history)

    def normalize_privileged(self, privileged: Tensor) -> Tensor:
        return self.privileged_normalizer(privileged) if self.normalize_observations else privileged

    @torch.no_grad()
    def update_normalization(self, data: PolicyInput) -> None:
        """只统计本次实际当前帧，历史复用该变换而不重复计入样本。"""
        if self.normalize_observations and self.training:
            self.observation_normalizer.update(data.observation)
            self.privileged_normalizer.update(data.privileged)

    def encode(self, data: PolicyInput) -> Tensor:
        """按分支选择编码器；未选择分支的特权值不参与学生计算。"""
        if data.branch.dtype != torch.bool or data.branch.shape != (data.observation.shape[0],):
            raise ValueError("Policy branch must be a boolean vector")
        latent = data.observation.new_zeros((data.observation.shape[0], self.spec.latent_dim))
        teachers = data.branch.nonzero(as_tuple=True)[0]
        students = (~data.branch).nonzero(as_tuple=True)[0]
        if teachers.numel():
            encoded = self.teacher_encoder(self.normalize_privileged(data.privileged[teachers]))
            latent = latent.index_copy(0, teachers, encoded)
        if students.numel():
            encoded, _ = self.student_encoder(self.normalize_history(data.history[students]))
            latent = latent.index_copy(0, students, encoded)
        return latent

    def act(self, data: PolicyInput) -> Tensor:
        latent = self.encode(data)
        return self.transform_action(self.actor(torch.cat((self.normalize_observation(data.observation), latent), dim=-1)))

    def critic_features(self, data: PolicyInput, latent: Tensor | None = None) -> Tensor:
        if latent is None:
            latent = self.encode(data)
        branch = F.one_hot(data.branch.long(), num_classes=2).to(data.observation.dtype)
        return torch.cat((self.normalize_privileged(data.privileged), self.normalize_history(data.history).flatten(1),
                          latent, branch), dim=-1)

    def value(self, data: PolicyInput, latent: Tensor | None = None) -> Tensor:
        return self.critic(self.critic_features(data, latent)).squeeze(-1)

    def student_action(self, observation: Tensor, history: Tensor) -> Tensor:
        """学生推理仅接收实际本体帧和历史，不访问特权或分支。"""
        latent, _ = self.student_encoder(self.normalize_history(history))
        return self.transform_action(self.actor(torch.cat((self.normalize_observation(observation), latent), dim=-1)))
