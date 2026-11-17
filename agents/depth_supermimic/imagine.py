# Copyright (c) 2026-2027 zh
"""内容：通过独立物理与视觉模型完成32控制步BPTT，保留视觉刷新和末端价值输入导数。"""

from dataclasses import dataclass

import torch

from .interfaces import append_history
from .compilation import compile_kernel


@dataclass
class ImaginedData:
    objective: torch.Tensor
    rewards: torch.Tensor
    failures: torch.Tensor
    features: torch.Tensor
    survival: torch.Tensor
    discounted_reward: torch.Tensor
    bootstrap: torch.Tensor
    survivals: torch.Tensor


class Imagination:
    def __init__(self, task, actor, model, critic, config, perception):
        self.task, self.actor, self.model, self.critic = task, actor, model, critic
        self.config, self.perception = config, perception
        self.step = (torch.compile(self._step, backend=config.compile_backend, dynamic=False)
                     if config.compile_imagination else self._step)
        self._returns = (compile_kernel(self._lambda_returns, config.compile_backend)
                         if config.compile_imagination else self._lambda_returns)

    def features(self, state, history, tokens, actions, age):
        return torch.cat((self.task.privileged(state), history.flatten(1), tokens.flatten(1),
                          actions.flatten(1), age), -1)

    def _step(self, state, history, tokens, actions, age):
        points = self.perception.points(self.actor.normalized_history(history))
        action = self.actor(history, tokens, points)
        with torch.no_grad():
            features = self.features(state, history, tokens, actions, age)
        predicted = self.model(state, action)
        failure = self.task.soft_termination(predicted)
        reward = self.task.reward(state, action, predicted, failure)
        actions = torch.cat((actions[:, 1:], action[:, None]), 1)
        tokens, age = self.perception.advance(tokens, self.task.physical_features(predicted), actions, age)
        history = append_history(history, self.task.observe(predicted, noisy=True))
        return predicted, history, tokens, actions, age, reward, failure, features

    def rollout(self, seed):
        cfg = self.config
        state, history, tokens, actions, age = seed.state, seed.history, seed.tokens, seed.action_history, seed.age
        survival = torch.ones(history.shape[0], device=history.device)
        discounted = torch.zeros_like(survival)
        rewards, failures, features, survivals = [], [], [], []
        for step in range(cfg.horizon):
            survivals.append(survival)
            state, history, tokens, actions, age, reward, failure, feature = self.step(
                state, history, tokens, actions, age)
            discounted = discounted + cfg.gamma ** step * survival * reward
            survival = survival * (1 - failure)
            rewards.append(reward.detach())
            failures.append(failure.detach())
            features.append(feature.detach())
        endpoint = self.features(state, history, tokens, actions, age)
        bootstrap = cfg.gamma ** cfg.horizon * survival * self.critic(endpoint)
        features.append(endpoint.detach())
        survivals.append(survival)
        return ImaginedData(discounted + bootstrap, torch.stack(rewards, 1), torch.stack(failures, 1),
                            torch.stack(features, 1), survival, discounted, bootstrap,
                            torch.stack(survivals, 1).detach())

    @torch.no_grad()
    def lambda_returns(self, data):
        return self._returns(data)

    def _lambda_returns(self, data):
        values = self.critic(data.features)
        tail, targets = values[:, -1], torch.empty_like(data.rewards)
        for step in reversed(range(self.config.horizon)):
            continuation = data.survivals[:, step + 1] / data.survivals[:, step].clamp_min(1e-6)
            tail = data.rewards[:, step] + self.config.gamma * continuation * (
                (1 - self.config.lam) * values[:, step + 1] + self.config.lam * tail)
            targets[:, step] = tail
        return targets
