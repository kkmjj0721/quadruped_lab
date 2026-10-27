# Copyright (c) 2026-2027 zh
"""
内容：
    展开教师／学生闭环想象，构造软存活回报与固定λ-return。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .interfaces import PolicyInput, append_history


@dataclass
class ImaginedRollout:
    """时间优先数据；features和values包含H+1个节点。"""

    critic_features: Tensor
    values: Tensor
    rewards: Tensor
    failures: Tensor
    survival: Tensor
    returns: Tensor
    branch: Tensor
    discounted_rewards: Tensor | None = None
    bootstrap: Tensor | None = None


def branch_mean(values: Tensor, branch: Tensor, teacher_weight: float = 0.5) -> Tensor:
    """每分支先平均；仅一个分支时使用其均值，空批量报错。"""
    if not 0 <= teacher_weight <= 1 or values.shape[-1] != branch.numel() or branch.numel() == 0:
        raise ValueError("Invalid branch mean inputs")
    teacher = bool(branch.any())
    student = bool((~branch).any())
    if teacher and student:
        return teacher_weight * values[..., branch].mean() + (1 - teacher_weight) * values[..., ~branch].mean()
    return values.mean()


def imagine(policy, world_model, seeds: PolicyInput, horizon: int = 32, gamma: float = 0.99,
            *, noisy: bool = True) -> ImaginedRollout:
    """调用方冻结参数；此处保留状态、历史、latent和末端价值导数。"""
    if horizon < 1 or not 0 <= gamma < 1:
        raise ValueError("Invalid imagination horizon or discount")
    if policy.spec != world_model.spec:
        raise ValueError("Policy and world model specs differ")
    task = world_model.task
    current = seeds
    features, values, rewards, failures = [], [], [], []
    survival = [seeds.observation.new_ones(seeds.observation.shape[0])]
    objective = torch.zeros_like(survival[0])
    for time in range(horizon):
        latent = policy.encode(current)
        features.append(policy.critic_features(current, latent))
        values.append(policy.critic(features[-1]).squeeze(-1))
        # act使用同一当前输入；其内部不改变历史或归一化统计。
        action = policy.act(current)
        next_state = world_model(current.state, action)
        failure = world_model.termination(next_state)
        reward = task.reward(current.state, action, next_state, failure)
        if reward.shape != survival[0].shape or failure.shape != survival[0].shape:
            raise ValueError("Reward and termination must be batch vectors")
        objective = objective + gamma ** time * survival[-1] * reward
        rewards.append(reward)
        failures.append(failure)
        survival.append(survival[-1] * (1 - failure))
        frame = task.observe(next_state, noisy=noisy)
        current = PolicyInput(next_state, frame, append_history(current.history, frame),
                              task.privileged(next_state), seeds.branch)
    features.append(policy.critic_features(current))
    values.append(policy.critic(features[-1]).squeeze(-1))
    bootstrap = gamma ** horizon * survival[-1] * values[-1]
    return ImaginedRollout(torch.stack(features), torch.stack(values), torch.stack(rewards),
                           torch.stack(failures), torch.stack(survival), objective + bootstrap,
                           seeds.branch, objective, bootstrap)


@torch.no_grad()
def lambda_returns(rollout: ImaginedRollout, gamma: float = 0.99, lam: float = 0.95) -> Tensor:
    """同一次rollout的critic快照生成一次固定目标，不在拟合中重算。"""
    if not 0 <= gamma < 1 or not 0 <= lam <= 1:
        raise ValueError("Invalid return discount or lambda")
    targets = torch.empty_like(rollout.rewards)
    tail = rollout.values[-1].detach()
    for time in reversed(range(rollout.rewards.shape[0])):
        tail = rollout.rewards[time] + gamma * (1 - rollout.failures[time]) * (
            (1 - lam) * rollout.values[time + 1] + lam * tail)
        targets[time] = tail
    return targets.detach()
