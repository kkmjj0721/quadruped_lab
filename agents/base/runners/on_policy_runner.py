# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026-2027 zh
"""
内容：
    组织单设备 PPO 采样与更新，记录基础指标并保存、恢复检查点。
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import torch
from tensordict import TensorDict

from ..algorithms import PPO
from ..env import VecEnv
from ..models import MLPModel


class OnPolicyRunner:
    """通过向量化环境接口运行基础 PPO 训练。"""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.cfg = deepcopy(train_cfg)
        self.device = device
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg.get("save_interval", 50)
        if self.num_steps_per_env < 1 or self.save_interval < 1:
            raise ValueError("Rollout length and save interval must be positive.")
        obs = self.env.get_observations().to(self.device)
        self.alg = PPO.construct_algorithm(obs, env, self.cfg, device)
        self.current_learning_iteration = 0
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """执行指定次数更新；重复调用时接续已完成的迭代计数。"""
        if num_learning_iterations < 0:
            raise ValueError("Number of learning iterations must be non-negative.")
        if init_at_random_ep_len:
            self.env.episode_length_buf.copy_(
                torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        for _ in range(num_learning_iterations):
            start = time.perf_counter()
            reward_sum = 0.0
            # 使用 no_grad 而非 inference_mode，确保保存的采样可参与后续反向传播。
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    if self.cfg.get("check_for_nan", True):
                        self._check_finite(obs, rewards, dones)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    reward_sum += rewards.mean().item()
                self.alg.compute_returns(obs)
            losses = self.alg.update()
            self.current_learning_iteration += 1
            self._log(losses, reward_sum / self.num_steps_per_env, time.perf_counter() - start)
            if self.log_dir is not None and self.current_learning_iteration % self.save_interval == 0:
                self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
        if self.log_dir is not None and num_learning_iterations > 0 and self.current_learning_iteration % self.save_interval:
            self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")

    def save(self, path: str | Path, infos: dict | None = None) -> None:
        """保存已完成更新计数、模型与优化器。"""
        checkpoint = self.alg.save()
        checkpoint.update(iter=self.current_learning_iteration, infos=infos)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)

    def load(
        self, path: str | Path, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict | None:
        """加载可信的基础 PPO 检查点，默认映射到当前训练设备。"""
        checkpoint = torch.load(path, weights_only=False, map_location=map_location or self.device)
        if self.alg.load(checkpoint, load_cfg, strict):
            self.current_learning_iteration = checkpoint["iter"]
        self.alg.storage.clear()
        self.alg.transition.clear()
        return checkpoint.get("infos")

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """返回固定统计量的确定性策略；跨设备时复制模型以保留训练设备。"""
        self.alg.eval_mode()
        policy = self.alg.get_policy()
        if device is not None and torch.device(device) != torch.device(self.device):
            return deepcopy(policy).to(device)
        return policy

    def _log(self, losses: dict[str, float], reward: float, elapsed: float) -> None:
        """输出英文进度，可选追加一行 JSON 指标。"""
        metrics = {
            "iteration": self.current_learning_iteration,
            "reward_mean": reward,
            "learning_rate": self.alg.learning_rate,
            "elapsed_seconds": elapsed,
            **losses,
        }
        print(f"Iteration {self.current_learning_iteration}: reward={reward:.4f}, value_loss={losses['value']:.4f}")
        if self.log_dir is not None:
            with (self.log_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")

    @staticmethod
    def _check_finite(obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """在异常数据进入 rollout 前终止本轮训练。"""
        for name, value in list(obs.items()) + [("rewards", rewards), ("dones", dones)]:
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Environment returned non-finite values in {name!r}.")
