# Copyright (c) 2026-2027 zh
"""内容：执行真实采集、世界模型拟合、单策略更新、回放归档及参数续训。"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import torch

from ..interfaces import append_history
from ..logging import TrainingLogger
from .config import SingleConfig, SingleExecutionConfig
from .learner import Learner
from .models import Critic, MoEActor, WorldModel
from .replay import Replay, Window, WindowArchive


def _numbers(metrics):
    """在日志边界一次拷贝标量，采样热路径保留设备张量。"""
    keys = [key for key, value in metrics.items() if isinstance(value, torch.Tensor)]
    values = torch.stack([metrics[key].detach().float().reshape(()) for key in keys]).cpu().tolist() if keys else []
    result = {key: float(value) for key, value in metrics.items() if key not in keys}
    result.update(zip(keys, values))
    return result


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    return deepcopy(value)


class SingleRunner:
    """参数续训会重建真实回放和预热，既不恢复仿真状态，也不重置累计迭代。"""

    def __init__(self, env, config: SingleConfig, log_dir=None, control_enabled=True, execution=None):
        self.env, self.task, self.spec = env, env.task, env.task.spec
        self.device = torch.device(env.device)
        self.config = SingleConfig(**asdict(config))
        self.execution = SingleExecutionConfig(**asdict(execution)) if execution is not None else SingleExecutionConfig()
        cfg = self.config
        if not math.isfinite(self.task.dt) or self.task.dt <= 0:
            raise ValueError("Task control timestep must be finite and positive")
        if cfg.explore_frac and cfg.explore_theta * self.task.dt >= 2:
            raise ValueError("OU exploration requires explore_theta * control_dt < 2")
        if cfg.replay_capacity < self.spec.history_length + cfg.wm_window - 1:
            raise ValueError("Replay capacity cannot fit the model window and observation history")
        self.actor = MoEActor(self.spec, cfg).to(self.device)
        self.policy = self.actor
        self.model = WorldModel(self.task, cfg.wm_hidden).to(self.device)
        self.critic = Critic(self.spec.privileged_dim, cfg.critic_hidden).to(self.device)
        self.learner = Learner(self.task, self.actor, self.model, self.critic, cfg, execution=self.execution)
        self.replay = Replay(self.spec, env.num_envs, cfg.replay_capacity, self.device)
        self.archive = WindowArchive(cfg.archive_size)
        self.control_enabled = bool(control_enabled)
        self.iteration = self.real_steps = self.imagined_steps = 0
        self.current = self.history = None
        self.ages = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.ou_noise = torch.zeros(env.num_envs, self.spec.action_dim, device=self.device)
        self.exploration_count = int(env.num_envs * cfg.explore_frac)
        self.warmup_remaining = cfg.warmup_iterations
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.logger = TrainingLogger(self.log_dir, num_envs=env.num_envs, device=self.device)
        self.target_iteration = 0
        self.elapsed = 0.0
        self.session_iterations = 0
        self.episode_returns = torch.zeros(env.num_envs, device=self.device)
        self.episode_lengths = torch.zeros_like(self.ages)
        self.completed_episodes = deque(maxlen=100)
        self._safe_policy = None
        self._collapse_strikes = 0
        self._guard_failures = self._guard_transitions = 0.0
        if self.log_dir is not None:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")
            (self.log_dir / "execution.json").write_text(json.dumps(asdict(self.execution), indent=2) + "\n", encoding="utf-8")

    def configure_logging(self, *, target_iteration, log_interval=1):
        if target_iteration < self.iteration or log_interval < 1:
            raise ValueError("Invalid logging target or interval")
        self.target_iteration = int(target_iteration)
        self.logger.log_interval = int(log_interval)

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _collection_step_metrics(self):
        """任务子类提供设备上的原始／加权指标，算法不硬编码任务奖励。"""
        return {}

    @torch.no_grad()
    def reset_collection(self):
        self.current = self.env.reset()
        frame = self.task.observe(self.current, noisy=True)
        self.history = frame[:, None].repeat(1, self.spec.history_length, 1)
        self.ages.zero_()
        self.ou_noise.zero_()
        self.episode_returns.zero_()
        self.episode_lengths.zero_()
        self.completed_episodes.clear()
        self.replay.clear()
        self.archive.clear()
        self.warmup_remaining = self.config.warmup_iterations

    @torch.no_grad()
    def collect(self, steps):
        if steps < 1:
            raise ValueError("Collection steps must be positive")
        if self.current is None:
            self.reset_collection()
        cfg = self.config
        totals = {}
        reward_sum = torch.zeros((), device=self.device)
        failure_count = torch.zeros((), device=self.device)
        timeout_count = torch.zeros((), device=self.device)
        for _ in range(steps):
            observation = self.history[:, -1]
            self.actor.normalizer.update(observation)
            action = self.actor(self.history)
            noise = cfg.action_noise * torch.randn_like(action)
            if self.exploration_count:
                count = self.exploration_count
                decay = 1 - cfg.explore_theta * self.task.dt
                self.ou_noise[:count].mul_(decay).add_(cfg.explore_noise * math.sqrt(2 * cfg.explore_theta * self.task.dt)
                                                     * torch.randn_like(action[:count]))
                noise[:count] = self.ou_noise[:count]
            action = action + noise
            if cfg.action_limit is not None:
                action = action.clamp(-cfg.action_limit, cfg.action_limit)
            if not torch.isfinite(action).all():
                raise FloatingPointError("Collected actions are non-finite")
            result = self.env.step(action)
            self.replay.add(self.current, observation, action, result, self.ages)
            self.model.update_stats(self.current, action, result.final_state)
            done = result.terminated | result.truncated
            self.episode_returns.add_(result.reward)
            self.episode_lengths.add_(1)
            if done.any():
                completed = torch.stack((self.episode_returns[done], self.episode_lengths[done].float()), dim=-1).cpu().tolist()
                self.completed_episodes.extend(completed)
            self.episode_returns.masked_fill_(done, 0)
            self.episode_lengths.masked_fill_(done, 0)
            frame = self.task.observe(result.next_state, noisy=True)
            history = append_history(self.history, frame)
            self.history = torch.where(done[:, None, None], frame[:, None].expand_as(history), history)
            self.ou_noise.masked_fill_(done[:, None], 0)
            self.ages = torch.where(done, 0, self.ages + 1)
            self.current = result.next_state.detached_clone()
            reward_sum += result.reward.mean()
            failure_count += result.terminated.sum()
            timeout_count += (result.truncated & ~result.terminated).sum()
            for name, value in self._collection_step_metrics().items():
                totals[name] = totals.get(name, 0) + value
            self.real_steps += self.env.num_envs
        metrics = {name: value / steps for name, value in totals.items()}
        metrics.update(real_reward=reward_sum / steps, failures=failure_count, timeouts=timeout_count)
        if self.completed_episodes:
            size = len(self.completed_episodes)
            metrics["episode/return_mean"] = sum(item[0] for item in self.completed_episodes) / size
            metrics["episode/length_steps_mean"] = sum(item[1] for item in self.completed_episodes) / size
            metrics["episode/length_seconds_mean"] = metrics["episode/length_steps_mean"] * self.task.dt
            metrics["episode/window_count"] = float(size)
        return metrics

    def _priority(self, windows):
        if hasattr(self.task, "replay_priority"):
            scores = self.task.replay_priority(windows)
        else:
            scores = (windows.terminated & windows.valid).float().sum(-1)
        if scores.shape != windows.valid.shape[:1] or not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("Replay priority must be a finite nonnegative score per window")
        return scores

    def _model_batch(self):
        cfg = self.config
        prioritized = int(cfg.wm_batch * cfg.wm_prioritized_frac)
        archived = int(cfg.wm_batch * cfg.wm_archive_frac) if self.archive.size else 0
        uniform = cfg.wm_batch - prioritized - archived
        parts = [self.replay.sample_windows(uniform, cfg.wm_window)] if uniform else []
        if prioritized:
            candidates = self.replay.sample_windows(4 * prioritized, cfg.wm_window)
            scores = self._priority(candidates)
            chosen = (scores + 1e-3 * torch.rand_like(scores)).topk(prioritized).indices
            parts.append(candidates.index(chosen))
            worthy = self.task.archive_worthy(candidates) if hasattr(self.task, "archive_worthy") else scores > 0
            if worthy.shape != scores.shape or worthy.dtype != torch.bool:
                raise ValueError("Archive eligibility must be a boolean mask per window")
            keep = worthy & (torch.rand_like(scores) < cfg.archive_keep_prob)
            self.archive.add(candidates.index(keep))
        if archived:
            parts.append(self.archive.sample(archived))
        return Window.concatenate(parts) if len(parts) > 1 else parts[0]

    def _collapse_guard(self, metrics, real_delta):
        self._guard_failures += metrics["failures"]
        self._guard_transitions += real_delta
        cfg = self.config
        if self.iteration % cfg.collapse_interval:
            return
        rate = self._guard_failures / max(self._guard_transitions * self.task.dt / 60, 1e-9)
        self._guard_failures = self._guard_transitions = 0.0
        metrics["guard/falls_per_environment_minute"] = rate
        eligible = self._safe_policy is not None and self.warmup_remaining == 0 and self.learner.actor_updates > 0
        self._collapse_strikes = self._collapse_strikes + 1 if eligible and rate > cfg.collapse_falls else 0
        if self._collapse_strikes < 2:
            return
        for name in ("actor", "critic"):
            getattr(self, name).load_state_dict(self._safe_policy[name])
        # 参考实现清除导致发散的优化器动量；模型优化器及累计更新次数保留。
        self.learner.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate, betas=(.9, .95))
        self.learner.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.learner.lr_multiplier *= .5
        self.learner.ramp_start = self.learner.actor_updates
        self._collapse_strikes = 0
        metrics["guard/rollback"] = 1.0
        print("[WARN] Restored saved actor and critic after repeated physical falls; learning rate reduced", flush=True)

    def learn(self, iterations):
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("Training iterations must be a positive integer")
        output = []
        target = max(self.target_iteration, self.iteration + iterations)
        for _ in range(iterations):
            self._synchronize()
            started = time.perf_counter()
            previous_steps = self.real_steps
            steps = self.config.initial_collection_steps if self.replay.size == 0 else self.config.collect_steps
            metrics = self.collect(steps)
            self._synchronize()
            collection_time = time.perf_counter() - started
            model_time = control_time = 0.0
            self.learner.schedule(self.iteration + 1)
            enabled = False
            if self.replay.ready(self.config.wm_window):
                phase = time.perf_counter()
                model_metrics = {}
                for _ in range(self.config.wm_updates):
                    for key, value in self.learner.fit_model(self._model_batch()).items():
                        model_metrics[key] = model_metrics.get(key, 0) + value
                metrics.update({key: value / self.config.wm_updates for key, value in model_metrics.items()})
                self._synchronize()
                model_time = time.perf_counter() - phase
                if self.warmup_remaining:
                    self.warmup_remaining -= 1
                elif self.control_enabled:
                    enabled = True
                    phase = time.perf_counter()
                    metrics.update(self.learner.improve(self.replay.sample_seeds(self.config.imag_batch)))
                    self.imagined_steps += self.config.horizon * self.config.imag_batch
                    self._synchronize()
                    control_time = time.perf_counter() - phase
            else:
                metrics["model_not_ready"] = 1.0
            self.iteration += 1
            elapsed = time.perf_counter() - started
            self.elapsed += elapsed
            self.session_iterations += 1
            delta = self.real_steps - previous_steps
            metrics.update(iteration=self.iteration, training_target_iteration=target, real_steps=self.real_steps,
                           real_steps_delta=delta, control_enabled=float(enabled), imagined_steps_total=self.imagined_steps,
                           world_model_updates_total=self.learner.model_updates, control_updates_total=self.learner.actor_updates)
            metrics.update({"timing/collection_seconds": collection_time, "timing/world_model_seconds": model_time,
                            "timing/control_seconds": control_time, "timing/iteration_seconds": elapsed,
                            "timing/session_elapsed_seconds": self.elapsed,
                            "timing/eta_seconds": self.elapsed / self.session_iterations * (target - self.iteration),
                            "performance/real_total_fps": delta / max(elapsed, 1e-9),
                            "performance/real_collection_fps": delta / max(collection_time, 1e-9),
                            "replay/archive_size": self.archive.size})
            metrics = _numbers(metrics)
            self._collapse_guard(metrics, delta)
            self.logger.record(metrics)
            output.append(metrics)
        return output

    def _remember_safe_policy(self):
        self._safe_policy = {name: _cpu_tree(getattr(self, name).state_dict()) for name in ("actor", "critic")}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"format": "supermimic_single_moe_v1", "spec": asdict(self.spec), "spec_hash": self.spec.hash,
                "config": asdict(self.config), "control_enabled": self.control_enabled,
                "actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "world_model": self.model.state_dict(),
                "learner": self.learner.state_dict(), "iteration": self.iteration, "real_steps": self.real_steps,
                "imagined_steps": self.imagined_steps, "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(data, temporary)
        os.replace(temporary, path)
        if self._collapse_strikes == 0:
            self._remember_safe_policy()

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        if data.get("format") != "supermimic_single_moe_v1" or data.get("spec_hash") != self.spec.hash:
            raise ValueError("Single-actor checkpoint format or task specification mismatch")
        if data.get("spec") != asdict(self.spec) or data.get("config") != asdict(self.config):
            raise ValueError("Single-actor checkpoint configuration mismatch")
        if data.get("control_enabled") != self.control_enabled:
            raise ValueError("Cannot change control update mode while resuming")
        for name, checkpoint_key in (("actor", "actor"), ("critic", "critic"), ("model", "world_model")):
            state = data[checkpoint_key]
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Checkpoint contains non-finite network tensors")
            if any((value < 0).any() for key, value in state.items() if key.endswith(".var")):
                raise ValueError("Checkpoint contains invalid normalization variance")
            if any((value <= 0).any() for key, value in state.items() if key.endswith(".count")):
                raise ValueError("Checkpoint contains invalid normalization count")
            getattr(self, name).load_state_dict(state)
        self.learner.load_state_dict(data["learner"])
        for name in ("iteration", "real_steps", "imagined_steps"):
            value = data[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Invalid checkpoint training counters")
            setattr(self, name, value)
        self.current = self.history = None
        self.replay.clear()
        self.archive.clear()
        self.warmup_remaining = self.config.warmup_iterations
        self.elapsed = 0.0
        self.session_iterations = 0
        self._collapse_strikes = 0
        self._guard_failures = self._guard_transitions = 0.0
        torch.set_rng_state(data["rng"].cpu())
        if self.device.type == "cuda" and data["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in data["cuda_rng"]])
        self._remember_safe_policy()
        return data
