# Copyright (c) 2026-2027 zh
"""内容：完整传播物理状态、视觉循环记忆和本体历史的三十二步想象及固定 λ-return。"""

from dataclasses import dataclass

import torch
from torch import Tensor

from .compilation import compile_kernel
from .tensor_state import append_history


@dataclass
class ImaginedData:
    """完整想象目标保留梯度；价值拟合特征、收益及生存快照已停止梯度。"""

    objective: Tensor
    rewards: Tensor
    failures: Tensor
    features: Tensor
    survival: Tensor
    discounted_reward: Tensor
    bootstrap: Tensor
    survivals: Tensor


class DepthImagination:
    """参数冻结由学习器管理；末端价值与生存权重始终保留输入梯度。"""

    def __init__(self, task, actor, model, critic, config):
        self.task, self.actor, self.model, self.critic, self.config = task, actor, model, critic, config
        self.camera_period = model.camera_period
        self.step = (torch.compile(self._step, backend=config.compile_backend, dynamic=False)
                     if config.compile_imagination else self._step)
        self._returns = (compile_kernel(self._lambda_returns, config.compile_backend)
                         if config.compile_imagination else self._lambda_returns)

    def _step(self, state, history, hidden, latent, phase):
        action = self.actor(history, latent)
        features = self.critic.features(self.task.privileged(state), hidden, phase)
        predicted, feature = self.model(state, action, latent, phase)
        phase = (phase + 1) % self.camera_period
        hidden, latent = self.actor.update_memory(feature, hidden, phase == 0)
        failure = self.task.soft_termination(predicted)
        reward = self.task.reward(state, action, predicted, failure)
        history = append_history(history, self.task.observe(predicted, noisy=True))
        return predicted, history, hidden, latent, phase, reward, failure, features

    def rollout(self, seed, initial_visual=None):
        state, history, phase = seed.state, seed.history, seed.phase
        hidden, latent = self.actor.rebuild(seed) if initial_visual is None else initial_visual
        survival = history.new_ones(history.shape[0])
        discounted = torch.zeros_like(survival)
        rewards, failures, features, survivals = [], [], [], []
        for time in range(self.config.horizon):
            survivals.append(survival)
            state, history, hidden, latent, phase, reward, failure, feature = self.step(
                state, history, hidden, latent, phase)
            discounted = discounted + self.config.gamma ** time * survival * reward
            survival = survival * (1 - failure)
            rewards.append(reward.detach())
            failures.append(failure.detach())
            features.append(feature.detach())
        endpoint = self.critic.features(self.task.privileged(state), hidden, phase)
        bootstrap = self.config.gamma ** self.config.horizon * survival * self.critic.value(endpoint)
        features.append(endpoint.detach())
        survivals.append(survival)
        return ImaginedData(discounted + bootstrap, torch.stack(rewards, 1), torch.stack(failures, 1),
                            torch.stack(features, 1), survival, discounted, bootstrap,
                            torch.stack(survivals, 1).detach())

    @torch.no_grad()
    def lambda_returns(self, data):
        return self._returns(data)

    @torch.no_grad()
    def _lambda_returns(self, data):
        values = self.critic.value(data.features)
        targets = torch.empty_like(data.rewards)
        tail = values[:, -1]
        for time in reversed(range(self.config.horizon)):
            continuation = data.survivals[:, time + 1] / data.survivals[:, time].clamp_min(1e-6)
            tail = data.rewards[:, time] + self.config.gamma * continuation * (
                (1 - self.config.lam) * values[:, time + 1] + self.config.lam * tail)
            targets[:, time] = tail
        return targets
