# Copyright (c) 2026-2027 zh
"""
内容：
    管理控制、价值及学生监督的互斥优化器和参数冻结范围。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .distillation import StudentDistiller
from .imagination import branch_mean, imagine, lambda_returns


@contextmanager
def frozen_parameters(*modules):
    """只冻结参数，不使用no_grad，异常退出也恢复原requires_grad。"""
    parameters = list({id(parameter): parameter for module in modules for parameter in module.parameters()}.values())
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, previous):
            parameter.requires_grad_(enabled)


@dataclass(frozen=True)
class LearnerConfig:
    learning_rate: float = 2e-4
    student_learning_rate: float = 1e-3
    gamma: float = 0.99
    lam: float = 0.95
    horizon: int = 32
    critic_epochs: int = 5
    critic_minibatches: int = 4
    student_epochs: int = 1
    student_minibatches: int = 4
    balance_weight: float = 0.01
    teacher_weight: float = 0.5
    max_grad_norm: float = 1.0
    actor_grad_clip: float = 0.1
    observation_noise: bool = True
    actor_ramp_updates: int = 0

    def __post_init__(self):
        if isinstance(self.actor_ramp_updates, bool) or not isinstance(self.actor_ramp_updates, int) or self.actor_ramp_updates < 0:
            raise ValueError("Actor ramp updates must be a nonnegative integer")
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Learner settings must be finite")
        if min(self.learning_rate, self.student_learning_rate, self.max_grad_norm, self.actor_grad_clip) <= 0:
            raise ValueError("Learning rates and gradient limits must be positive")
        if min(self.horizon, self.critic_epochs, self.critic_minibatches, self.student_epochs, self.student_minibatches) < 1:
            raise ValueError("Update counts must be positive")
        if not 0 <= self.gamma < 1 or not 0 <= self.lam <= 1 or not 0 <= self.teacher_weight <= 1:
            raise ValueError("Invalid discount or branch weight")
        if self.balance_weight < 0:
            raise ValueError("Balance weight must be nonnegative")


class SuperMimicLearner:
    """世界模型optimizer由WorldModelTrainer独立拥有。"""

    def __init__(self, policy, world_model, config: LearnerConfig | None = None):
        self.policy = policy
        self.world_model = world_model
        self.config = config or LearnerConfig()
        if policy.spec != world_model.spec:
            raise ValueError("Policy and world model specs differ")
        self.control_parameters = list(policy.actor.parameters()) + list(policy.teacher_encoder.parameters())
        self.control_optimizer = torch.optim.Adam(self.control_parameters, lr=self.config.learning_rate, betas=(0.9, 0.95))
        self.control_update_count = 0
        self.value_optimizer = torch.optim.Adam(policy.critic.parameters(), lr=self.config.learning_rate)
        self.distiller = StudentDistiller(policy, learning_rate=self.config.student_learning_rate,
                                         balance_weight=self.config.balance_weight, max_grad_norm=self.config.max_grad_norm)
        groups = [self.control_parameters, list(policy.critic.parameters()),
                  list(policy.student_encoder.parameters()), list(world_model.parameters())]
        identifiers = [id(parameter) for group in groups for parameter in group]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Optimizer parameter sets must be disjoint")

    def update_control(self, seeds):
        """先固定critic监督数据，再执行一次BPTT控制更新。"""
        cfg = self.config
        self.policy.zero_grad(set_to_none=True)
        self.world_model.zero_grad(set_to_none=True)
        with frozen_parameters(self.world_model, self.policy.critic, self.policy.student_encoder):
            rollout = imagine(self.policy, self.world_model, seeds.detached_clone(), cfg.horizon,
                              cfg.gamma, noisy=cfg.observation_noise)
            targets = lambda_returns(rollout, cfg.gamma, cfg.lam)
            features = rollout.critic_features[:-1].detach().clone()
            loss = -branch_mean(rollout.returns, rollout.branch, cfg.teacher_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite control loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.control_parameters, cfg.max_grad_norm, error_if_nonfinite=True)
            for parameter in self.control_parameters:
                if parameter.grad is not None:
                    parameter.grad.clamp_(-cfg.actor_grad_clip, cfg.actor_grad_clip)
            ramp_fraction = min(1.0, (self.control_update_count + 1) / cfg.actor_ramp_updates) if cfg.actor_ramp_updates else 1.0
            control_learning_rate = cfg.learning_rate * ramp_fraction
            for group in self.control_optimizer.param_groups:
                group["lr"] = control_learning_rate
            self.control_optimizer.step()
            self.control_update_count += 1
        metrics = {"control_loss": float(loss.detach()), "control_grad_norm": float(norm),
                   "imagined_steps": float(cfg.horizon * seeds.branch.numel()),
                   "control_learning_rate": control_learning_rate}
        diagnostics = {"survival_h": rollout.survival[-1], "discounted_reward": rollout.discounted_rewards,
                       "bootstrap": rollout.bootstrap}
        metrics.update({f"imagination/{name}": float(branch_mean(value.detach(), rollout.branch, cfg.teacher_weight))
                        for name, value in diagnostics.items()})
        for flag, name in ((True, "teacher"), (False, "student")):
            selected = rollout.branch == flag
            if selected.any():
                metrics[f"{name}_return"] = float(rollout.returns[selected].detach().mean())
        return metrics, (features, targets, rollout.branch.detach().clone())

    def update_critic(self, data):
        """每个小批量保持两组数据，避免按采集比例稀释学生价值损失。"""
        features, targets, branches = data
        features = features.detach().flatten(0, 1)
        targets = targets.detach().flatten()
        branches = branches.unsqueeze(0).expand(data[1].shape).reshape(-1)
        groups = [(branches == flag).nonzero().flatten() for flag in (True, False)]
        groups = [group for group in groups if group.numel()]
        if not groups:
            raise ValueError("Empty critic training data")
        batches = min(self.config.critic_minibatches, min(group.numel() for group in groups))
        total, updates = 0.0, 0
        for _ in range(self.config.critic_epochs):
            partitions = [group[torch.randperm(group.numel(), device=group.device)].tensor_split(batches) for group in groups]
            for batch in range(batches):
                indices = torch.cat([partition[batch] for partition in partitions])
                errors = (self.policy.critic(features[indices]).squeeze(-1) - targets[indices]).square()
                loss = branch_mean(errors, branches[indices], self.config.teacher_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite critic loss")
                self.value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config.max_grad_norm, error_if_nonfinite=True)
                self.value_optimizer.step()
                total += float(loss.detach())
                updates += 1
        return {"critic_loss": total / updates, "critic_updates": float(updates)}

    def update_student(self, real_students):
        if real_students is None or not (~real_students.branch).any():
            return self.distiller.update(None)
        students = real_students.index(~real_students.branch)
        count = students.branch.numel()
        totals, updates = {}, 0
        for _ in range(self.config.student_epochs):
            order = torch.randperm(count, device=students.branch.device)
            for indices in order.tensor_split(min(count, self.config.student_minibatches)):
                result = self.distiller.update(students.index(indices))
                for name, value in result.items():
                    totals[name] = totals.get(name, 0.0) + value
                updates += 1
        return {**{name: value / updates for name, value in totals.items()}, "student_updates": float(updates)}

    def update(self, seeds, real_students=None):
        metrics, critic_data = self.update_control(seeds)
        metrics.update(self.update_critic(critic_data))
        metrics.update(self.update_student(real_students))
        return metrics

    def state_dict(self):
        return {"config": asdict(self.config), "control": self.control_optimizer.state_dict(),
                "value": self.value_optimizer.state_dict(), "student": self.distiller.optimizer.state_dict(),
                "control_updates": self.control_update_count}

    def load_state_dict(self, state):
        saved_config = {**state["config"]}
        saved_config.setdefault("actor_ramp_updates", 0)
        if saved_config != asdict(self.config):
            raise ValueError("Learner configuration mismatch")
        count = state.get("control_updates", 0)
        if (isinstance(count, bool) or not isinstance(count, int) or count < 0
                or (self.config.actor_ramp_updates and "control_updates" not in state)):
            raise ValueError("Invalid or missing control update count in learner checkpoint")
        self.control_optimizer.load_state_dict(state["control"])
        self.value_optimizer.load_state_dict(state["value"])
        self.distiller.optimizer.load_state_dict(state["student"])
        self.control_update_count = count
