# Copyright (c) 2026-2027 zh
"""内容：在视觉潜变量与物理状态的完整时间链上构建 BPTT 回报和固定 λ-return。"""

from dataclasses import dataclass

import torch

from agents.supermimic.compilation import compile_kernel
from agents.supermimic.interfaces import append_history


@dataclass
class ImaginedData:
    objective: torch.Tensor
    rewards: torch.Tensor
    failures: torch.Tensor
    features: torch.Tensor
    survival: torch.Tensor


def lambda_returns(rewards, failures, values, gamma, lam):
    """直接使用参考 Parkour 的 (1-d)，不换成累计存活比值。"""
    tail, targets = values[:, -1], []
    for step in reversed(range(rewards.shape[1])):
        tail = rewards[:, step] + gamma * (1 - failures[:, step]) * ((1 - lam) * values[:, step + 1] + lam * tail)
        targets.append(tail)
    return torch.stack(targets[::-1], 1)


class Imagination:
    def __init__(self, task, actor, model, critic, failure, config, execution):
        self.task, self.actor, self.model, self.critic, self.failure = task, actor, model, critic, failure
        self.config = config
        self.step = compile_kernel(self._step, execution.compile_backend) if execution.compile else self._step

    def _step(self, state, history, latent, hidden, observation_noise, use_failure_head):
        with torch.no_grad():
            hidden = self.actor.remember(latent.detach(), hidden)
        action = self.actor.act_latent(history, hidden).clamp(-20, 20)
        with torch.no_grad():
            features = self.task.privileged(state)
        predicted, _, next_latent = self.model(state, action, latent)
        failure = self.task.soft_termination(predicted)
        reward = self.task.reward(state, action, predicted, failure)
        if use_failure_head:
            learned = self.failure(self.task.privileged(predicted)).sigmoid().clamp_max(self.config.fail_clip)
            failure = 1 - (1 - failure) * (1 - learned)
        frame = self.task.observe(predicted, noisy=False) + observation_noise
        return predicted, append_history(history, frame), next_latent, hidden, reward, failure, features

    def sample_noise(self, batch, prototype):
        """每个时间步一次完整批次采样，微批切分不重排参考随机数消费。"""
        return torch.stack([self.task.add_proprio_noise(prototype.new_zeros(batch, self.task.spec.observation_dim))
                            for _ in range(self.config.horizon)])

    def rollout(self, seed, *, use_failure_head, observation_noise=None):
        cfg, task = self.config, self.task
        state, history = seed.state, seed.history
        if observation_noise is None:
            observation_noise = self.sample_noise(history.shape[0], history)
        with torch.no_grad():
            latent = self.actor.encoder(seed.depth)
        hidden = seed.hidden
        survival = torch.ones(history.shape[0], device=history.device)
        objective = torch.zeros_like(survival)
        rewards, failures, features = [], [], []
        for step in range(cfg.horizon):
            predicted, history, latent, hidden, reward, failure, feature = self.step(
                state, history, latent, hidden, observation_noise[step], use_failure_head)
            objective = objective + cfg.gamma**step * survival * reward
            survival = survival * (1 - failure)
            rewards.append(reward.detach())
            failures.append(failure.detach())
            features.append(feature)
            state = predicted
        endpoint = task.privileged(state)
        objective = objective + cfg.gamma**cfg.horizon * survival * self.critic(endpoint)
        features.append(endpoint.detach())
        return ImaginedData(objective, torch.stack(rewards, 1), torch.stack(failures, 1),
                            torch.stack(features, 1), survival.detach())
