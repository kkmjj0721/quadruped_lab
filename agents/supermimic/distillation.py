# Copyright (c) 2026-2027 zh
"""
内容：
    在真实学生状态上监督历史表征，并约束八专家的批量平均门控质量。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .interfaces import PolicyInput
from .models.actor_critic import ActorCritic


def representation_loss(student: Tensor, teacher: Tensor, weights: Tensor, balance_weight: float = 0.01):
    """按样本和latent维度共同平均，教师目标在该边界停止梯度。"""
    latent_loss = F.mse_loss(student, teacher.detach())
    balance_loss = (weights.mean(dim=0) - 1 / weights.shape[-1]).square().mean()
    return latent_loss + balance_weight * balance_loss, latent_loss, balance_loss


class StudentDistiller:
    """优化器只持有学生参数；每次使用最新教师重新计算同状态目标。"""

    def __init__(
        self, policy: ActorCritic, learning_rate: float = 1e-3, balance_weight: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        if not all(math.isfinite(value) for value in (learning_rate, balance_weight, max_grad_norm)):
            raise ValueError("Distillation settings must be finite")
        if learning_rate <= 0 or balance_weight < 0 or max_grad_norm <= 0:
            raise ValueError("Invalid distillation settings")
        self.policy = policy
        self.balance_weight = balance_weight
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(policy.student_encoder.parameters(), lr=learning_rate)

    def update(self, data: PolicyInput | None) -> dict[str, float]:
        if data is None or not (~data.branch).any():
            return {"student/skipped": 1.0, "student/samples": 0.0}
        students = data.index(~data.branch).detached_clone()
        with torch.no_grad():
            teacher = self.policy.teacher_encoder(self.policy.normalize_privileged(students.privileged))
        latent, weights = self.policy.student_encoder(self.policy.normalize_history(students.history))
        loss, latent_loss, balance_loss = representation_loss(latent, teacher, weights, self.balance_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Student loss is non-finite")
        with torch.no_grad():
            observation = self.policy.normalize_observation(students.observation)
            teacher_action = self.policy.transform_action(self.policy.actor(torch.cat((observation, teacher), -1)))
            student_action = self.policy.transform_action(self.policy.actor(torch.cat((observation, latent), -1)))
            action_error = F.mse_loss(student_action, teacher_action)
            entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(self.policy.student_encoder.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        metrics = {
            "student/skipped": 0.0, "student/samples": float(latent.shape[0]),
            "student/loss": float(loss.detach()), "student/latent_loss": float(latent_loss.detach()),
            "student/balance_loss": float(balance_loss.detach()), "student/gradient_norm": float(grad_norm),
            "student/action_mse": float(action_error), "student/gate_entropy": float(entropy),
        }
        for index, weight in enumerate(weights.detach().mean(0)):
            metrics[f"student/gate_{index}"] = float(weight)
        return metrics
