# Copyright (c) 2026-2027 zh
"""内容：在预测状态上更新视觉观测，沿用 SuperMimic 想象目标与 λ-return。"""

import torch

from ..imagine import ImaginedData, Imagination
from ..interfaces import append_history
from .interfaces import VisualMemory


class VisualImagination(Imagination):
    """相机、感知和离散选点停梯度；动作、物理状态、目标坐标转换保留梯度。"""

    def __init__(self, task, actor, model, critic, perception, planner, config):
        super().__init__(task, actor, model, critic, config)
        self.perception, self.planner = perception, planner

    def prepare(self, state, history, memory):
        with torch.no_grad():
            camera, fresh = self.task.update_camera(state, memory.camera)
            token = memory.token
            if fresh.any():
                indices = fresh.nonzero(as_tuple=True)[0]
                token = token.index_copy(0, indices, self.perception.encode_depth(camera.depth_pair[indices]))
            height_map, hidden = self.perception.step(token, history, memory.hidden)
            candidates = self.planner.predict(history, state.command, height_map).targets
            memory = VisualMemory(camera, token, hidden)
        # 锁存属于任务状态转换；不能因感知停梯度而切断物理状态或 yaw 转换。
        state, targets, valid = self.task.update_visual_targets(state, candidates)
        return state, memory, height_map, targets, valid

    def rollout(self, seed):
        cfg = self.config
        state, history, memory = seed.state, seed.history, seed.visual
        survival = history.new_ones(history.shape[0])
        discounted = torch.zeros_like(survival)
        rewards, failures, features, survivals = [], [], [], []
        for step in range(cfg.horizon):
            state, memory, height_map, targets, valid = self.prepare(state, history, memory)
            features.append(self.task.visual_privileged(state, memory).detach())
            survivals.append(survival.detach())
            action = self.actor(history, height_map, targets, valid)
            predicted = self.model(state, action)
            failure = self.task.soft_termination(predicted)
            reward = self.task.reward(state, action, predicted, failure)
            discounted = discounted + cfg.gamma ** step * survival * reward
            survival = survival * (1 - failure)
            rewards.append(reward.detach())
            failures.append(failure.detach())
            history = append_history(history, self.task.observe(predicted, noisy=True))
            state = predicted
        # 终点与中间时刻遵守同一感知／锁存顺序，避免 critic 使用滞后记忆。
        state, memory, _, _, _ = self.prepare(state, history, memory)
        endpoint = self.task.visual_privileged(state, memory)
        bootstrap = cfg.gamma ** cfg.horizon * survival * self.critic(endpoint)
        features.append(endpoint.detach())
        survivals.append(survival.detach())
        return ImaginedData(discounted + bootstrap, torch.stack(rewards, 1), torch.stack(failures, 1),
                            torch.stack(features, 1), survival, discounted, bootstrap, torch.stack(survivals, 1))
