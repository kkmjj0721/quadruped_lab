# Copyright (c) 2026-2027 zh
"""内容：滚动可微物理与视觉状态，构造存活加权回报和固定价值监督。"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .interfaces import append_history


@dataclass
class ImaginedData:
    objective: Tensor
    rewards: Tensor
    failures: Tensor
    features: Tensor
    values: Tensor


class Imagination:
    """策略端截断视觉特征，模型内部的潜变量递推始终保留计算图。"""

    def __init__(self, task, actor, model, critic, config):
        self.task, self.actor, self.model, self.critic, self.config = task, actor, model, critic, config

    def rollout(self, seed) -> ImaginedData:
        config = self.config
        state, history = seed.state, seed.history
        with torch.no_grad():
            latent = self.actor.encoder(seed.depth)
        survival = torch.ones(history.shape[0], device=history.device)
        objective = torch.zeros_like(survival)
        rewards, failures, features = [], [], []
        for step in range(config.horizon):
            features.append(self.task.privileged(state, latent).detach())
            action = self.actor.act_latent(history, latent)
            predicted = self.model(state, latent, action)
            failure = self.task.soft_termination(predicted)
            reward = self.task.reward(state, action, predicted, failure)
            objective = objective + config.gamma ** step * survival * reward
            survival = survival * (1 - failure)
            rewards.append(reward.detach())
            failures.append(failure.detach())
            state, latent = predicted.state, predicted.latent
            history = append_history(history, self.task.observe(state, noisy=True))
        endpoint = self.task.privileged(state, latent)
        objective = objective + config.gamma ** config.horizon * survival * self.critic(endpoint)
        features.append(endpoint.detach())
        stacked = torch.stack(features, 1)
        # 在critic统计或参数改变之前固定整条轨迹的价值快照。
        with torch.no_grad():
            values = self.critic(stacked)
        return ImaginedData(objective, torch.stack(rewards, 1), torch.stack(failures, 1), stacked, values)

    @torch.no_grad()
    def lambda_returns(self, data: ImaginedData) -> Tensor:
        tail = data.values[:, -1]
        targets = torch.empty_like(data.rewards)
        for step in reversed(range(self.config.horizon)):
            tail = data.rewards[:, step] + self.config.gamma * (1 - data.failures[:, step]) * (
                (1 - self.config.lam) * data.values[:, step + 1] + self.config.lam * tail)
            targets[:, step] = tail
        return targets
