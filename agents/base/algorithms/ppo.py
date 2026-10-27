# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026-2027 zh
"""
内容：
    从参考仓库 RSL-RL 精简的单设备前馈 PPO，负责采样记录、GAE 和策略更新。
"""

from __future__ import annotations

from copy import deepcopy
from itertools import chain

import torch
from torch import nn
from tensordict import TensorDict

from ..env import VecEnv
from ..models import MLPModel
from ..storage import RolloutStorage
from ..utils import resolve_class, resolve_obs_groups


class PPO:
    """使用裁剪代理目标和独立 actor、critic 的基础 PPO。"""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float | None = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
    ) -> None:
        if actor.distribution is None or critic.distribution is not None:
            raise ValueError("PPO requires a stochastic actor and a deterministic critic.")
        batch_size = storage.num_envs * storage.num_transitions_per_env
        if num_learning_epochs < 1 or not 1 <= num_mini_batches <= batch_size:
            raise ValueError("Learning epochs must be positive and mini-batches must be between 1 and rollout size.")
        if schedule not in ("fixed", "adaptive"):
            raise ValueError("Learning rate schedule must be 'fixed' or 'adaptive'.")
        if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
            raise ValueError("gamma and lam must be between 0 and 1.")
        if learning_rate <= 0 or max_grad_norm <= 0 or clip_param <= 0:
            raise ValueError("Learning rate, gradient norm and clipping range must be positive.")
        if desired_kl is not None and desired_kl <= 0:
            raise ValueError("desired_kl must be positive or None.")
        optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
        if optimizer.lower() not in optimizers:
            raise ValueError(f"Unsupported optimizer: {optimizer!r}.")
        self.device = device
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.optimizer = optimizers[optimizer.lower()](chain(actor.parameters(), critic.parameters()), lr=learning_rate)
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    # ---------- 采样与回报 ----------

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        """采样动作，在环境可能原地更新数据之前保存当前状态。"""
        self.transition.observations = obs.clone()
        actions = self.actor(obs, stochastic_output=True)
        self.transition.actions = actions.detach().clone()
        self.transition.values = self.critic(obs).detach().clone()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(actions).detach().clone()
        self.transition.distribution_params = tuple(p.detach().clone() for p in self.actor.output_distribution_params)
        return actions

    @torch.no_grad()
    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        """保存奖励与终止信息；超时补偿沿用参考实现的当前状态价值。"""
        expected_shape = (self.storage.num_envs,)
        if rewards.shape != expected_shape or dones.shape != expected_shape:
            raise ValueError(f"Rewards and dones must have shape {expected_shape}.")
        self.transition.rewards = rewards.to(self.device).clone()
        self.transition.dones = dones.to(self.device)
        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(self.device)
            if time_outs.shape != expected_shape:
                raise ValueError(f"Timeouts must have shape {expected_shape}.")
            self.transition.rewards += self.gamma * self.transition.values.squeeze(-1) * time_outs
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    @torch.no_grad()
    def compute_returns(self, obs: TensorDict) -> None:
        """计算完整 rollout 的 GAE 和价值回报。"""
        storage = self.storage
        if storage.step != storage.num_transitions_per_env:
            raise RuntimeError("Cannot compute returns before the rollout is complete.")
        last_values = self.critic(obs).detach()
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(storage.num_transitions_per_env)):
            next_values = last_values if step == storage.num_transitions_per_env - 1 else storage.values[step + 1]
            not_terminal = 1.0 - storage.dones[step].float()
            delta = storage.rewards[step] + not_terminal * self.gamma * next_values - storage.values[step]
            advantage = delta + not_terminal * self.gamma * self.lam * advantage
            storage.returns[step] = advantage + storage.values[step]
        storage.advantages.copy_(storage.returns - storage.values)
        if not self.normalize_advantage_per_mini_batch:
            storage.advantages.copy_(self._normalize_advantages(storage.advantages))

    # ---------- 参数更新 ----------

    def update(self) -> dict[str, float]:
        """遍历小批量，更新模型并返回平均损失。"""
        losses = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}
        num_updates = 0
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            advantages = batch.advantages
            if self.normalize_advantage_per_mini_batch:
                advantages = self._normalize_advantages(advantages)
            self.actor(batch.observations, stochastic_output=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations)
            entropy = self.actor.output_entropy
            self._adapt_learning_rate(batch.old_distribution_params, self.actor.output_distribution_params)

            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob.squeeze(-1))
            surrogate = -advantages.squeeze(-1) * ratio
            clipped_surrogate = -advantages.squeeze(-1) * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)
            surrogate_loss = torch.maximum(surrogate, clipped_surrogate).mean()
            if self.use_clipped_value_loss:
                clipped_values = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((values - batch.returns).square(), (clipped_values - batch.returns).square()).mean()
            else:
                value_loss = (values - batch.returns).square().mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            losses["value"] += value_loss.item()
            losses["surrogate"] += surrogate_loss.item()
            losses["entropy"] += entropy.mean().item()
            num_updates += 1

        observations = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(observations)
        self.critic.update_normalization(observations)
        self.storage.clear()
        return {key: value / num_updates for key, value in losses.items()}

    @staticmethod
    def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
        """总体标准差使单样本批次也能得到有限结果。"""
        return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    @torch.no_grad()
    def _adapt_learning_rate(self, old_params: tuple, new_params: tuple) -> None:
        """根据新旧高斯策略的平均 KL 调整学习率。"""
        if self.schedule != "adaptive" or self.desired_kl is None:
            return
        kl = self.actor.get_kl_divergence(old_params, new_params).mean().item()
        if kl > 2 * self.desired_kl:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif 0 < kl < self.desired_kl / 2:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    # ---------- 模式与检查点 ----------

    def train_mode(self) -> None:
        """启用模型训练与归一化统计更新。"""
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """固定归一化统计，供确定性推理使用。"""
        self.actor.eval()
        self.critic.eval()

    def save(self) -> dict:
        """返回基础模型、归一化缓冲和优化器状态。"""
        return {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        """按选项恢复训练状态，返回是否应恢复迭代计数。"""
        options = {"actor": True, "critic": True, "optimizer": True, "iteration": True} if load_cfg is None else load_cfg
        if options.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if options.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if options.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        return options.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """获取确定性前向调用的 actor。"""
        return self.actor

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """从基础训练配置构造组件，保留调用方配置原样。"""
        config = deepcopy(cfg)
        algorithm_class, algorithm_cfg = resolve_class(config.get("algorithm", {}), PPO)
        actor_class, actor_cfg = resolve_class(config.get("actor", {}), MLPModel)
        critic_class, critic_cfg = resolve_class(config.get("critic", {}), MLPModel)
        groups = resolve_obs_groups(obs, config.get("obs_groups"))
        actor_cfg.setdefault("distribution_cfg", {"class_name": "GaussianDistribution"})
        actor = actor_class(obs, groups, "actor", env.num_actions, **actor_cfg)
        critic = critic_class(obs, groups, "critic", 1, **critic_cfg)
        storage = RolloutStorage("rl", env.num_envs, config["num_steps_per_env"], obs, [env.num_actions], device)
        return algorithm_class(actor, critic, storage, device=device, **algorithm_cfg)
