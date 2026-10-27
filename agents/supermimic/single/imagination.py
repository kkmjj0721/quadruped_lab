# Copyright (c) 2026-2027 zh
"""内容：编译可微想象单步，保留末端价值导数并生成固定 lambda-return。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..interfaces import TaskState, append_history


@dataclass
class ImaginedData:
    objective: Tensor
    rewards: Tensor
    failures: Tensor
    features: Tensor
    survival: Tensor
    discounted_reward: Tensor
    bootstrap: Tensor
    gates: Tensor


class Imagination:
    def __init__(self, task, actor, model, critic, config):
        self.task, self.actor, self.model, self.critic, self.config = task, actor, model, critic, config
        self.step = (torch.compile(self._step, backend=config.compile_backend, dynamic=False, fullgraph=True)
                     if config.compile_imagination else self._step)

    def _step(self, state: TaskState, history: Tensor):
        action, gates = self.actor.action_and_gates(history)
        with torch.no_grad():
            features = self.task.privileged(state)
        predicted = self.model(state, action)
        failure = self.task.soft_termination(predicted)
        reward = self.task.reward(state, action, predicted, failure)
        frame = self.task.observe(predicted, noisy=True)
        return predicted, append_history(history, frame), reward, failure, features, gates

    def rollout(self, seed) -> ImaginedData:
        cfg = self.config
        state, history = seed.state, seed.history
        survival = torch.ones(history.shape[0], device=history.device)
        discounted = torch.zeros_like(survival)
        rewards, failures, features, gates = [], [], [], []
        for step in range(cfg.horizon):
            state, history, reward, failure, feature, routing = self.step(state, history)
            discounted = discounted + cfg.gamma ** step * survival * reward
            survival = survival * (1 - failure)
            rewards.append(reward.detach())
            failures.append(failure.detach())
            features.append(feature.detach())
            gates.append(routing.detach())
        endpoint = self.task.privileged(state)
        bootstrap = cfg.gamma ** cfg.horizon * survival * self.critic(endpoint)
        features.append(endpoint.detach())
        return ImaginedData(discounted + bootstrap, torch.stack(rewards, dim=1), torch.stack(failures, dim=1),
                            torch.stack(features, dim=1), survival, discounted, bootstrap, torch.stack(gates, dim=1))

    @torch.no_grad()
    def lambda_returns(self, data: ImaginedData) -> Tensor:
        """固定同一次 rollout 的价值快照，直接使用失败概率避免存活除法下溢。"""
        values = self.critic(data.features)
        targets = torch.empty_like(data.rewards)
        tail = values[:, -1]
        for step in reversed(range(self.config.horizon)):
            tail = data.rewards[:, step] + self.config.gamma * (1 - data.failures[:, step]) * (
                (1 - self.config.lam) * values[:, step + 1] + self.config.lam * tail)
            targets[:, step] = tail
        return targets

