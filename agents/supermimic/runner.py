# Copyright (c) 2026-2027 zh
"""内容：执行真实采集、世界模型拟合、策略更新、回放归档及参数续训。"""

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

from .interfaces import append_history
from .logging import TrainingLogger
from .buffer import Replay, Window, WindowArchive
from .config import ExecutionConfig, SuperMimicConfig
from .exploration import stratified_collection_mask
from .learner import Learner
from .metrics import ModelMetrics, read_metrics
from .policy import Critic, make_actor
from .timing import PhaseTimer, TrainingTiming
from .world_model import WorldModel


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item) for item in value)
    return deepcopy(value)


def _validate_network_state(state, label="network"):
    """在检查点边界校验网络张量和归一化统计，不加载任何已有模块。"""
    if not isinstance(state, dict) or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                                          for value in state.values()):
        raise ValueError(f"Checkpoint contains invalid or non-finite {label} tensors")
    if any((value < 0).any() for key, value in state.items() if key.endswith(".var")):
        raise ValueError(f"Checkpoint contains invalid {label} normalization variance")
    if any((value <= 0).any() for key, value in state.items() if key.endswith(".count")):
        raise ValueError(f"Checkpoint contains invalid {label} normalization count")


class SuperMimicRunner:
    """参考采集与优化循环；恢复区分重新计时的热启动与保留迭代号的继续训练。"""

    def __init__(self, env, config: SuperMimicConfig, log_dir=None, control_enabled=True, execution=None,
                 collection_groups=None):
        self.env, self.task, self.spec = env, env.task, env.task.spec
        self.device = torch.device(env.device)
        self.config = SuperMimicConfig(**asdict(config))
        self.execution = ExecutionConfig(**asdict(execution)) if execution is not None else ExecutionConfig()
        self.execution.cuda_graph = self.execution.cuda_graph and self.device.type == "cuda"
        self.timing = TrainingTiming(self.device, enabled=self.execution.detailed_timing,
                                     interval=self.execution.timing_interval)
        if self.execution.cuda_graph and not self.execution.fused_adam:
            raise ValueError("CUDA-graph updates require fused Adam; disable cuda_graph to use non-fused Adam")
        cfg = self.config
        if not math.isfinite(self.task.dt) or self.task.dt <= 0:
            raise ValueError("Task control timestep must be finite and positive")
        if cfg.replay_capacity <= cfg.wm_window + 1:
            raise ValueError("Replay capacity must exceed the world-model window by at least two")
        if collection_groups is None:
            collection_groups = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        if (not isinstance(collection_groups, torch.Tensor) or collection_groups.shape != (env.num_envs,)
                or collection_groups.device != self.device
                or collection_groups.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("Collection groups must be an integer tensor with one entry per environment on its device")
        self.deterministic_collection_mask = stratified_collection_mask(collection_groups, cfg.collect_det_frac)
        self.actor = make_actor(self.spec, cfg, self.task.observation_term_dims).to(self.device)
        self.policy = self.actor
        self.model = self._make_world_model().to(self.device)
        self.model.configure_supervised_compile(enabled=self.execution.wm_compile, backend=cfg.compile_backend)
        self.critic = Critic(self.spec.privileged_dim, cfg.critic_hidden).to(self.device)
        self.learner = self._make_learner()
        self.replay = Replay(self.spec, env.num_envs, cfg.replay_capacity, self.device)
        self.archive = WindowArchive(cfg.archive_size)
        self.control_enabled = bool(control_enabled)
        self.iteration = self.real_steps = self.imagined_steps = 0
        self.prefill_real_steps = 0
        self._collect_step_accumulator = 0.0
        self.current = self.history = None
        self.driver = self.expert = None
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.logger = TrainingLogger(self.log_dir, num_envs=env.num_envs, device=self.device)
        self.target_iteration = 0
        self.elapsed = 0.0
        self.session_iterations = 0
        self.episode_returns = torch.zeros(env.num_envs, device=self.device)
        self.episode_lengths = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.completed_episodes = deque(maxlen=100)
        self._safe_policy = None
        self._collapse_strikes = 0
        self._guard_falls = 0.0
        self._guard_reset_steps = 0
        if self.log_dir is not None:
            # 算法配置同步提供TEXT面板，任务入口负责可直接查看的params快照。
            self.logger.add_text("run/configuration", "```json\n" + json.dumps(
                {"config": asdict(cfg), "execution": asdict(self.execution)}, indent=2) + "\n```")

    def _make_world_model(self):
        """默认使用通用状态模型；任务子类可指定其输出统计语义。"""
        return WorldModel(self.task, self.config.wm_hidden)

    def _make_learner(self):
        """初始化与恢复共用执行选择，CPU保持原学习器。"""
        learner_type = Learner
        if self.execution.cuda_graph:
            from .cuda_graph import CudaGraphLearner

            learner_type = CudaGraphLearner
        return learner_type(self.task, self.actor, self.model, self.critic, self.config, execution=self.execution)

    def configure_logging(self, *, target_iteration, log_interval=1):
        if target_iteration < self.iteration or log_interval < 1:
            raise ValueError("Invalid logging target or interval")
        self.target_iteration = int(target_iteration)
        self.logger.log_interval = int(log_interval)

    def _collection_step_metrics(self):
        """任务子类提供设备上的原始／加权指标，算法不硬编码任务奖励。"""
        return {}

    def _override_actions(self, action, history, observation):
        """任务子类可用外部策略接管部分环境的动作；算法只保留调用位置。"""
        return action

    def _driver_scale(self):
        """驱动占比按参考实现从起始值线性衰减到结束值。"""
        start, end = self.config.driver_fade
        return min(1.0, max(0.0, (end - self.iteration) / max(1, end - start)))

    def _reset_drivers(self, done):
        """任务子类在回合结束时清理外部策略状态；默认没有外部策略。"""
        return None

    @torch.no_grad()
    def reset_collection(self):
        self.current = self.env.reset()
        self._collect_step_accumulator = 0.0
        frame = self.task.observe(self.current, noisy=True)
        self.history = frame[:, None].repeat(1, self.spec.history_length, 1)
        self.episode_returns.zero_()
        self.episode_lengths.zero_()
        self.completed_episodes.clear()
        self.replay.clear()
        self.archive.clear()
        if self.driver is not None or self.expert is not None:
            self._reset_drivers(torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device))

    @torch.no_grad()
    def collect(self, steps):
        if steps < 1:
            raise ValueError("Collection steps must be positive")
        if self.current is None:
            self.reset_collection()
        cfg = self.config
        totals = {}
        metric_counts = {}
        reward_sum = torch.zeros((), device=self.device)
        failure_count = torch.zeros((), device=self.device)
        timeout_count = torch.zeros((), device=self.device)
        reset_steps = 0
        for _ in range(steps):
            observation = self.history[:, -1]
            action = self._override_actions(self.actor(self.history), self.history, observation)
            noise = cfg.action_noise * torch.randn_like(action)
            noise.masked_fill_(self.deterministic_collection_mask[:, None], 0)
            action = action + noise
            if not torch.isfinite(action).all():
                raise FloatingPointError("Collected actions are non-finite")
            result = self.env.step(action)
            self.actor.update_statistics(self.history)
            self.replay.add(self.current, self.history, action, result)
            done = result.terminated | result.truncated
            keep = ~done
            if keep.any():
                self.model.update_stats(self.current.index(keep), action[keep], result.final_state.index(keep))
            self.episode_returns.add_(result.reward)
            self.episode_lengths.add_(1)
            if done.any():
                reset_steps += 1
                completed = torch.stack((self.episode_returns[done], self.episode_lengths[done].float()), dim=-1).cpu().tolist()
                self.completed_episodes.extend(completed)
            self.episode_returns.masked_fill_(done, 0)
            self.episode_lengths.masked_fill_(done, 0)
            frame = self.task.observe(result.next_state, noisy=True)
            history = append_history(self.history, frame)
            self.history = torch.where(done[:, None, None], frame[:, None].expand_as(history), history)
            if self.driver is not None or self.expert is not None:
                self._reset_drivers(done)
            self.current = result.next_state.detached_clone()
            reward_sum += result.reward.mean()
            failure_count += result.terminated.sum()
            timeout_count += (result.truncated & ~result.terminated).sum()
            for name, value in self._collection_step_metrics().items():
                totals[name] = totals.get(name, 0) + value
                metric_counts[name] = metric_counts.get(name, 0) + 1
            self.real_steps += self.env.num_envs
        # 回合指标只在结束步出现，按实际出现次数平均，与RSL-RL一致。
        metrics = {name: value / metric_counts[name] for name, value in totals.items()}
        metrics.update(real_reward=reward_sum / steps, failures=failure_count, timeouts=timeout_count,
                       reset_steps=reset_steps)
        metrics["episode/completed"] = failure_count + timeout_count
        if self.completed_episodes:
            size = len(self.completed_episodes)
            returns = [item[0] for item in self.completed_episodes]
            metrics["episode/return_mean"] = sum(returns) / size
            metrics["episode/return_min"] = min(returns)
            metrics["episode/return_max"] = max(returns)
            metrics["episode/return_std"] = math.sqrt(sum((value - metrics["episode/return_mean"]) ** 2
                                                         for value in returns) / size)
            metrics["episode/length_steps_mean"] = sum(item[1] for item in self.completed_episodes) / size
            metrics["episode/length_seconds_mean"] = metrics["episode/length_steps_mean"] * self.task.dt
            metrics["episode/window_count"] = float(size)
        return metrics

    def _priority(self, windows):
        scores = self.task.replay_priority(windows)
        if scores.shape != windows.valid.shape[:1] or not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("Replay priority must be a finite nonnegative score per window")
        return scores

    def _model_batch(self):
        cfg = self.config
        prioritized = int(cfg.wm_batch * cfg.wm_prioritized_frac)
        archived = int(cfg.wm_batch * cfg.wm_archive_frac) if self.archive.size else 0
        uniform = cfg.wm_batch - prioritized - archived
        parts = [self.replay.sample_windows(uniform, cfg.wm_window)] if uniform else []
        candidates = self.replay.sample_windows(4 * max(prioritized, 1), cfg.wm_window)
        scores = self._priority(candidates)
        worthy = self.task.archive_worthy(candidates)
        if worthy.shape != scores.shape or worthy.dtype != torch.bool:
            raise ValueError("Archive eligibility must be a boolean mask per window")
        keep = worthy & (torch.rand_like(scores) < cfg.archive_keep_prob)
        self.archive.add(candidates.index(keep))
        chosen = (scores + 1e-3 * torch.rand_like(scores)).topk(prioritized).indices
        if prioritized:
            parts.append(candidates.index(chosen))
        if archived:
            parts.append(self.archive.sample(archived))
        return Window.concatenate(parts) if len(parts) > 1 else parts[0]

    def _collapse_guard(self, metrics):
        """模型被利用表现为真实摔倒激增：回滚策略并降低学习率。

        MjLab只在发生reset的step输出Episode_Termination；因此分母是发生过reset
        的step数（含纯超时step），不是窗口内所有step数，也不是环境分钟。
        """
        cfg = self.config
        self._guard_falls += metrics["failures"]
        self._guard_reset_steps += metrics["reset_steps"]
        if self.iteration % cfg.collapse_interval:
            return
        falls = self._guard_falls / max(self._guard_reset_steps, 1)
        self._guard_falls, self._guard_reset_steps = 0.0, 0
        metrics["guard/falls_per_reset_step"] = falls
        if self.iteration <= self.learner.actor_frozen_until or falls <= cfg.collapse_falls:
            self._collapse_strikes = 0
            return
        self._collapse_strikes += 1
        if self._collapse_strikes < 2 or self._safe_policy is None:
            return
        for name in ("actor", "critic"):
            getattr(self, name).load_state_dict(self._safe_policy[name])
        self.learner.reset_policy_optimizers()
        self.learner.lr_multiplier *= .5
        self.learner.ramp_start = self.iteration
        self._collapse_strikes = 0
        metrics["guard/rollback"] = 1.0
        print(f"[WARN] Collapse guard: {falls:.1f} falls per reset step, "
              f"restored saved policy and value states; lr scale {self.learner.lr_multiplier:.3g}", flush=True)

    def learn(self, iterations):
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("Training iterations must be a positive integer")
        output = []
        target = max(self.target_iteration, self.iteration + iterations)
        # 预填使用整数批次且不消耗训练累加器；默认8步窗口仍先采10步。
        if self.replay.size <= self.config.wm_window + 1:
            prefill_started = time.perf_counter()
            prefill_previous_steps = self.real_steps
            prefill_falls = torch.zeros((), device=self.device)
            prefill_reset_steps = 0
            while self.replay.size <= self.config.wm_window + 1:
                metrics = self.collect(int(self.config.collect_steps))
                prefill_falls += metrics["failures"]
                prefill_reset_steps += metrics["reset_steps"]
            self._guard_falls += read_metrics({"failures": prefill_falls}, self.device)["failures"]
            self._guard_reset_steps += prefill_reset_steps
            prefill_seconds = time.perf_counter() - prefill_started
            self.elapsed += prefill_seconds
            self.prefill_real_steps += self.real_steps - prefill_previous_steps
            self.timing.record_wall("prefill", prefill_seconds)
        for _ in range(iterations):
            # 来源在真实采集前设置当前迭代号，驱动衰减与学习率使用同一编号。
            self.iteration += 1
            started = time.perf_counter()
            timer = PhaseTimer(self.device)
            previous_steps = self.real_steps
            previous_model_updates = self.learner.model_updates.clone()
            previous_actor_updates = self.learner.actor_updates.clone()
            previous_critic_updates = self.learner.critic_updates.clone()
            control_requested = self.control_enabled and self.iteration > self.learner.policy_start
            frozen = self.iteration <= self.learner.actor_frozen_until
            if self.execution.detailed_timing:
                phase = "actor_frozen" if control_requested and frozen else "training" if control_requested else "model_warmup"
                setup = self.learner.timing_setup_pending(control_enabled=control_requested, freeze_actor=frozen)
                self.timing.begin_iteration(self.iteration, phase, setup=setup)
            # 累加器跨learn调用保留余量，默认依次采2、3步。
            self._collect_step_accumulator += self.config.collect_steps
            collect_steps = int(self._collect_step_accumulator)
            self._collect_step_accumulator -= collect_steps
            with self.timing.section("collection"):
                metrics = self.collect(collect_steps)
            metrics["collection/control_steps"] = collect_steps
            timer.mark("collection")
            self.learner.schedule(self.iteration)
            enabled = False
            model_validity = {}
            policy_metric_names = set()
            model_ready = self.replay.ready(self.config.wm_window)
            if model_ready:
                model_metrics = ModelMetrics()
                for _ in range(self.config.wm_updates):
                    with self.timing.section("wm_sampling"):
                        window = self._model_batch()
                    with self.timing.section("wm_update"):
                        model_metrics.add(self.learner.fit_model(window))
                    del window
                averages, model_validity = model_metrics.mean(self.config.wm_updates)
                metrics.update(averages)
            else:
                metrics["model_not_ready"] = 1.0
            timer.mark("model")
            if model_ready and self.control_enabled and self.iteration > self.learner.policy_start:
                enabled = True
                with self.timing.section("seed_sampling"):
                    seed = self.replay.sample_seeds(self.config.imag_batch)
                if self.execution.detailed_timing:
                    policy_metrics = self.learner.improve(seed, freeze_actor=frozen, timing=self.timing)
                else:
                    policy_metrics = self.learner.improve(seed, freeze_actor=frozen)
                del seed
                policy_metric_names = set(policy_metrics)
                metrics.update(policy_metrics)
                metrics["control/actor_frozen"] = float(frozen)
                self.imagined_steps += self.config.horizon * self.config.imag_batch
            timer.mark("control")
            delta = self.real_steps - previous_steps
            metrics.update(iteration=self.iteration, training_target_iteration=target, real_steps=self.real_steps,
                           real_steps_delta=delta, control_enabled=float(enabled), imagined_steps_total=self.imagined_steps,
                           world_model_updates_total=self.learner.model_updates, control_updates_total=self.learner.actor_updates,
                           critic_updates_total=self.learner.critic_updates,
                           world_model_updates_delta=self.learner.model_updates - previous_model_updates,
                           control_updates_delta=self.learner.actor_updates - previous_actor_updates,
                           critic_updates_delta=self.learner.critic_updates - previous_critic_updates,
                           world_model_learning_rate=self.learner.model_optimizer.param_groups[0]["lr"],
                           critic_learning_rate=self.learner.critic_optimizer.param_groups[0]["lr"],
                           control_learning_rate=self.learner.actor_optimizer.param_groups[0]["lr"],
                           nonfinite_steps=self.learner.nonfinite_steps,
                           **{"replay/steps_per_env": self.replay.size,
                              "replay/transitions": self.replay.size * self.env.num_envs,
                              "replay/fill_fraction": self.replay.size / self.replay.capacity,
                              "replay/archive_size": self.archive.size})
            # 计数、学习率、损失和掩码同次读取；该拷贝也等待此前记录的阶段事件。
            with self.timing.section("metric_read", cuda=False):
                metrics = read_metrics(metrics, self.device, optional=policy_metric_names, validity=model_validity)
            collection_time = timer.seconds("start", "collection")
            model_time = timer.seconds("collection", "model") if model_ready else 0.0
            control_time = timer.seconds("model", "control") if enabled else 0.0
            metrics.update(self.timing.device_metrics())
            metrics.update(self.timing.previous_metrics())
            elapsed = time.perf_counter() - started
            self.elapsed += elapsed
            self.session_iterations += 1
            metrics.update({"timing/collection_seconds": collection_time, "timing/world_model_seconds": model_time,
                            "timing/control_seconds": control_time, "timing/learning_seconds": model_time + control_time,
                            "timing/iteration_seconds": elapsed, "timing/session_elapsed_seconds": self.elapsed,
                            "timing/eta_seconds": self.elapsed / self.session_iterations * (target - self.iteration),
                            "timing/cuda_events": float(self.device.type == "cuda"),
                            "performance/real_total_fps": delta / max(elapsed, 1e-9),
                            "performance/real_collection_fps": delta / max(collection_time, 1e-9)})
            if metrics["episode/completed"]:
                metrics["episode/failure_fraction"] = metrics["failures"] / metrics["episode/completed"]
                metrics["episode/timeout_fraction"] = metrics["timeouts"] / metrics["episode/completed"]
            self._collapse_guard(metrics)
            logging_started = time.perf_counter() if self.execution.detailed_timing else None
            self.logger.record(metrics)
            if self.execution.detailed_timing:
                finished = time.perf_counter()
                self.timing.end_iteration(finished - started, finished - logging_started)
            output.append(metrics)
        return output

    def _remember_safe_policy(self):
        self._safe_policy = {name: _cpu_tree(getattr(self, name).state_dict()) for name in ("actor", "critic")}

    def close(self):
        """关闭事件文件写入器；环境由调用方负责释放。"""
        self.logger.close()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"format": self.spec.model_id, "spec": asdict(self.spec), "spec_hash": self.spec.hash,
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

    def load(self, path, *, continue_run=False, reset_critic=False):
        """按参考热启动网络和同形状优化器；继续训练仅追加恢复累计日志计数。

        预热、回滚缩放、随机数及真实回放不从检查点恢复；策略形状不兼容时明确拒绝。
        网络结构变化时保留当前任务定义，重建相应网络并应用参考重预热排期。
        """
        data = torch.load(path, map_location=self.device, weights_only=False)
        if data.get("format") != self.spec.model_id:
            raise ValueError("SuperMimic checkpoint format or task specification mismatch")
        saved_config = SuperMimicConfig(**data["config"])
        if continue_run and saved_config.collect_det_frac != self.config.collect_det_frac:
            raise ValueError("Deterministic collection fraction changed; use an explicit warm start")
        same = {}
        for name, checkpoint_key in (("actor", "actor"), ("critic", "critic"), ("model", "world_model")):
            state = data[checkpoint_key]
            _validate_network_state(state)
            current = getattr(self, name).state_dict()
            same[name] = set(current) == set(state) and all(current[key].shape == state[key].shape for key in current)
        if not same["actor"]:
            raise ValueError("Actor shape changed: no warm start is defined for this observation or architecture")
        saved_learner = data.get("learner", {})
        counters = {}
        for name in ("iteration", "real_steps", "imagined_steps"):
            value = data.get(name, 0) if continue_run else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Invalid checkpoint training counters")
            counters[name] = value
        self.actor.load_state_dict(data["actor"])
        if same["model"]:
            self.model.load_state_dict(data["world_model"])
        same["critic"] = same["critic"] and not reset_critic
        if same["critic"]:
            self.critic.load_state_dict(data["critic"])
        self.learner = self._make_learner()
        for name in ("actor", "model", "critic"):
            if same[name] and name in saved_learner:
                self.learner.load_optimizer_state_dict(name, saved_learner[name])
            # 该累计计数仅用于本项目日志，不影响来源的学习率和预热排期。
            if continue_run and same[name]:
                setattr(self.learner, name + "_updates", saved_learner.get(name + "_updates", 0))
        if not same["model"]:
            self.learner.policy_start = max(self.learner.policy_start, self.config.wm_rewarmup_iterations)
            self.learner.actor_frozen_until = self.learner.policy_start + self.config.critic_warmup_iterations
            print(f"World model re-initialised (shape changed); policy updates from iteration "
                  f"{self.learner.policy_start}", flush=True)
        if not same["critic"]:
            self.learner.actor_frozen_until = max(self.learner.actor_frozen_until,
                                                 self.learner.policy_start + self.config.critic_warmup_iterations)
            print(f"Critic re-initialised; actor frozen until iteration {self.learner.actor_frozen_until}", flush=True)
        for name, value in counters.items():
            setattr(self, name, value)
        self.current = self.history = None
        # 与指定参考一致，恢复进程重新开始采集相位，物理与回放同时重置。
        self._collect_step_accumulator = 0.0
        self.replay.clear()
        self.archive.clear()
        self.elapsed = 0.0
        self.prefill_real_steps = 0
        self.session_iterations = 0
        self._collapse_strikes = 0
        self._guard_falls, self._guard_reset_steps = 0.0, 0
        self._safe_policy = None
        print(f"Resumed from {path} (iteration {data.get('iteration')})", flush=True)
        if continue_run:
            print(f"Continuing at iteration {self.iteration}", flush=True)
        return data
