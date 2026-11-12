# Copyright (c) 2026-2027 zh
"""内容：执行真实采集、世界模型拟合、策略更新、回放归档及参数续训。"""

from __future__ import annotations

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
from .metrics import EpisodeWindow, MetricWindow, ModelMetrics, read_metrics
from .policy import Critic, Mlp, make_actor
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
        if self.execution.wm_bf16 or self.execution.actor_bf16 or self.execution.critic_bf16:
            if self.device.type != "cuda":
                raise ValueError("BF16 network execution requires a native BF16 CUDA device")
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError("BF16 network execution requires native BF16 support on the selected CUDA device")
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
        self.critic = Critic(self.spec.privileged_dim, cfg.critic_hidden).to(self.device)
        self.model.use_bf16 = self.execution.wm_bf16
        for module, enabled in ((self.actor, self.execution.actor_bf16), (self.critic, self.execution.critic_bf16)):
            for layer in module.modules():
                if isinstance(layer, Mlp):
                    layer.use_bf16 = enabled
        self.model.configure_supervised_compile(enabled=self.execution.wm_compile, backend=cfg.compile_backend)
        self.learner = self._make_learner()
        self.replay = Replay(self.spec, env.num_envs, cfg.replay_capacity, self.device,
                             history_storage=self.execution.replay_history_storage,
                             device_sampling=self.execution.cuda_graph)
        self.archive = WindowArchive(cfg.archive_size, self.device)
        self._configure_sampling()
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
        self.completed_episodes = EpisodeWindow(env.num_envs, self.device)
        self._collection_metric_validity = {}
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

    def _configure_sampling(self):
        """构造和恢复共用执行选择；模式切换时旧回放本就不参与恢复。"""
        self._model_sampling = self._seed_sampling = None
        if self.replay.device_sampling != self.execution.cuda_graph:
            self.replay = Replay(self.spec, self.env.num_envs, self.config.replay_capacity, self.device,
                                 history_storage=self.execution.replay_history_storage,
                                 device_sampling=self.execution.cuda_graph)
        if self.execution.cuda_graph:
            from .cuda_graph import CapturedUpdate

            self._model_sampling = CapturedUpdate(self._sample_model_device, self.device)
            self._seed_sampling = CapturedUpdate(self._sample_seed_device, self.device)

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
        self._collection_metric_validity = {}
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
        reset_steps = torch.zeros((), dtype=torch.long, device=self.device)
        statistics_finite = torch.ones((), dtype=torch.bool, device=self.device)
        for _ in range(steps):
            observation = self.history[:, -1]
            action = self._override_actions(self.actor(self.history), self.history, observation)
            noise = cfg.action_noise * torch.randn_like(action)
            noise.masked_fill_(self.deterministic_collection_mask[:, None], 0)
            action = action + noise
            with self.timing.section("collection_action_check"):
                # 上一步统计异常与当前动作共用一次回读，坏动作仍在仿真调用前阻断。
                if not (torch.isfinite(action).all() & statistics_finite):
                    if not statistics_finite:
                        raise FloatingPointError("Collection normalization data contains non-finite values")
                    raise FloatingPointError("Collected actions are non-finite")
            result = self.env.step(action)
            done = result.terminated | result.truncated
            with self.timing.section("collection_statistics"):
                actor_stats = self.actor.prepare_statistics(self.history)
                model_stats = self.model.prepare_statistics(self.current, action, result.final_state, mask=~done)
                statistics_finite = torch.stack([stats.finite for stats in (*actor_stats, *model_stats)]).all()
                # 任一输入无效时所有统计均不提交；下一动作前或本次collect返回前报告。
                actor_stats = tuple(item._replace(count=torch.where(statistics_finite, item.count, 0))
                                    for item in actor_stats)
                model_stats = tuple(item._replace(count=torch.where(statistics_finite, item.count, 0))
                                    for item in model_stats)
                self.actor.apply_statistics(actor_stats)
                self.model.apply_statistics(model_stats)
            with self.timing.section("replay_history_write"):
                self.replay.add(self.current, self.history, action, result)
            with self.timing.section("collection_episodes"):
                self.episode_returns.add_(result.reward)
                self.episode_lengths.add_(1)
                reset_steps.add_(done.any())
                self.completed_episodes.append(self.episode_returns, self.episode_lengths, done)
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
        with self.timing.section("collection_statistics_check"):
            if not statistics_finite:
                raise FloatingPointError("Collection normalization data contains non-finite values")
        # 回合指标只在结束步出现，按实际出现次数平均，与RSL-RL一致。
        metrics = {name: value / metric_counts[name] for name, value in totals.items()}
        metrics.update(real_reward=reward_sum / steps, failures=failure_count, timeouts=timeout_count,
                       reset_steps=reset_steps)
        metrics["episode/completed"] = failure_count + timeout_count
        episode_metrics, self._collection_metric_validity = self.completed_episodes.metrics(self.task.dt)
        metrics.update(episode_metrics)
        return metrics

    def _priority(self, windows):
        scores = self.task.replay_priority(windows)
        message = "Replay priority must be a finite nonnegative score per window"
        if scores.shape != windows.valid.shape[:1]:
            raise ValueError(message)
        valid = (torch.isfinite(scores) & (scores >= 0)).all()
        if scores.device.type == "cuda":
            # 优先级非法属于不可继续训练的不变量错误；设备断言失败后进程须退出。
            torch._assert_async(valid, message)
        elif not valid:
            raise ValueError("Replay priority must be a finite nonnegative score per window")
        return scores

    def _sample_model_device(self):
        return self._model_batch(device_indices=True)

    def _sample_seed_device(self):
        return self.replay.sample_seeds(self.config.imag_batch, device_indices=True)

    def _model_batch(self, *, device_indices=False):
        cfg = self.config
        prioritized = int(cfg.wm_batch * cfg.wm_prioritized_frac)
        archived = int(cfg.wm_batch * cfg.wm_archive_frac) if self.archive.capacity else 0
        uniform = cfg.wm_batch - prioritized - archived
        parts = [self.replay.sample_windows(uniform, cfg.wm_window, device_indices=device_indices)] if uniform else []
        populated = self.archive.size > 0
        fallback = self.replay.sample_windows(archived, cfg.wm_window, device_indices=device_indices) if archived else None
        candidates = self.replay.sample_windows(4 * max(prioritized, 1), cfg.wm_window, device_indices=device_indices)
        scores = self._priority(candidates)
        worthy = self.task.archive_worthy(candidates)
        if worthy.shape != scores.shape or worthy.dtype != torch.bool:
            raise ValueError("Archive eligibility must be a boolean mask per window")
        keep = worthy & (torch.rand_like(scores) < cfg.archive_keep_prob)
        self.archive.add(candidates, keep)
        chosen = (scores + 1e-3 * torch.rand_like(scores)).topk(prioritized).indices
        if prioritized:
            parts.append(candidates.index(chosen))
        if archived:
            parts.append(self.archive.sample_or(fallback, populated))
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
        metric_window = MetricWindow(self.device, compile_backend=self.config.compile_backend
                                     if self.execution.cuda_graph else None)
        phase_timers = []
        window_started = None
        previous_window = None
        target = max(self.target_iteration, self.iteration + iterations)
        # 预填使用整数批次且不消耗训练累加器；默认8步窗口仍先采10步。
        if self.replay.size <= self.config.wm_window + 1:
            prefill_started = time.perf_counter()
            prefill_previous_steps = self.real_steps
            prefill_falls = torch.zeros((), device=self.device)
            prefill_reset_steps = torch.zeros((), dtype=torch.long, device=self.device)
            while self.replay.size <= self.config.wm_window + 1:
                metrics = self.collect(int(self.config.collect_steps))
                prefill_falls += metrics["failures"]
                prefill_reset_steps += metrics["reset_steps"]
            prefill = read_metrics({"failures": prefill_falls, "reset_steps": prefill_reset_steps}, self.device)
            self._guard_falls += prefill["failures"]
            self._guard_reset_steps += prefill["reset_steps"]
            prefill_seconds = time.perf_counter() - prefill_started
            self.elapsed += prefill_seconds
            self.prefill_real_steps += self.real_steps - prefill_previous_steps
            self.timing.record_wall("prefill", prefill_seconds)
        for local_iteration in range(iterations):
            # 来源在真实采集前设置当前迭代号，驱动衰减与学习率使用同一编号。
            self.iteration += 1
            started = time.perf_counter()
            if window_started is None:
                window_started = started
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
                if self._model_sampling is not None:
                    setup = (setup or self._model_sampling.graph is None
                             or control_requested and self._seed_sampling.graph is None)
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
                model_metrics = ModelMetrics(compile_backend=self.config.compile_backend
                                             if self.execution.cuda_graph else None)
                for _ in range(self.config.wm_updates):
                    with self.timing.section("wm_sampling"):
                        window = self._model_sampling() if self._model_sampling is not None else self._model_batch()
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
                    seed = self._seed_sampling() if self._seed_sampling is not None else self.replay.sample_seeds(self.config.imag_batch)
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
            metric_window.add(metrics, optional=policy_metric_names,
                              validity={**self._collection_metric_validity, **model_validity})
            phase_timers.append((timer, model_ready, enabled))
            flush = (self.session_iterations == 0 or self.iteration % self.execution.metric_interval == 0
                     or self.iteration % self.config.collapse_interval == 0 or local_iteration == iterations - 1)
            self.session_iterations += 1
            if not flush:
                self.timing.end_iteration(time.perf_counter() - started, 0.0, completed=False)
                continue
            # 展示数据仅在窗口边界集中回读，原护栏轮次也强制结束窗口。
            with self.timing.section("metric_read", cuda=False):
                aggregate, validity = metric_window.aggregate()
                metrics = read_metrics(aggregate, self.device, validity=validity)
            collection_time = sum(item.seconds("start", "collection") for item, _, _ in phase_timers)
            model_time = sum(item.seconds("collection", "model") for item, ready, _ in phase_timers if ready)
            control_time = sum(item.seconds("model", "control") for item, _, active in phase_timers if active)
            metrics.update(self.timing.device_metrics())
            if self.execution.metric_interval == 1:
                metrics.update(self.timing.previous_metrics())
            if previous_window is not None:
                metrics.update(previous_window)
            elapsed = time.perf_counter() - window_started
            count = len(metric_window)
            session_elapsed = self.elapsed + elapsed
            metrics.update({"timing/collection_seconds": collection_time / count,
                            "timing/world_model_seconds": model_time / count,
                            "timing/control_seconds": control_time / count,
                            "timing/learning_seconds": (model_time + control_time) / count,
                            "timing/iteration_seconds": elapsed / count,
                            "timing/window_seconds": elapsed,
                            "timing/window_collection_seconds": collection_time,
                            "timing/window_world_model_seconds": model_time,
                            "timing/window_control_seconds": control_time,
                            "timing/session_elapsed_seconds": session_elapsed,
                            "timing/eta_seconds": session_elapsed / self.session_iterations * (target - self.iteration),
                            "timing/cuda_events": float(self.device.type == "cuda"),
                            "performance/real_total_fps": metrics["real_steps_delta"] / max(elapsed, 1e-9),
                            "performance/real_collection_fps": metrics["real_steps_delta"] / max(collection_time, 1e-9)})
            if metrics["episode/completed"]:
                metrics["episode/failure_fraction"] = metrics["failures"] / metrics["episode/completed"]
                metrics["episode/timeout_fraction"] = metrics["timeouts"] / metrics["episode/completed"]
            self._collapse_guard(metrics)
            logging_started = time.perf_counter()
            self.logger.record(metrics)
            finished = time.perf_counter()
            completed_wall = finished - window_started
            self.elapsed += completed_wall
            self.timing.record_wall("metrics_windows", completed_wall)
            self.timing.record_wall("metrics_logging", finished - logging_started)
            self.timing.end_iteration(finished - started, finished - logging_started)
            # 当前日志写完后才知道完整窗口时间，下一窗口记录原起止轮号；末窗保存在summary。
            previous_window = {"timing/previous_window_start_iteration": metrics["metrics/window_start_iteration"],
                               "timing/previous_window_end_iteration": self.iteration,
                               "timing/previous_window_wall_seconds": completed_wall,
                               "timing/previous_window_logging_seconds": finished - logging_started}
            output.append(metrics)
            metric_window.clear()
            phase_timers.clear()
            window_started = None
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
                "config": asdict(self.config), "execution": asdict(self.execution), "control_enabled": self.control_enabled,
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
        # 在写入任何网络参数前核对实际要恢复的优化器，避免跨类型的部分恢复。
        for name, field in (("actor", "actor_optimizer"), ("model", "wm_optimizer")):
            if same[name] and name in saved_learner:
                kind = getattr(self.config, field)
                saved_kind = getattr(saved_config, field)
                state_kind = "muon_adam" if "muon" in saved_learner[name] else "adam"
                if kind != saved_kind or kind != state_kind:
                    raise ValueError(f"{name} optimizer type changed; use matching checkpoint configuration or start a new run")
                if kind == "muon_adam" and self.config.muon_weight_decay != saved_config.muon_weight_decay:
                    raise ValueError(f"{name} Muon weight decay changed; use matching checkpoint configuration or start a new run")
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
        self._configure_sampling()
        self.prefill_real_steps = 0
        self.session_iterations = 0
        self._collapse_strikes = 0
        self._guard_falls, self._guard_reset_steps = 0.0, 0
        self._safe_policy = None
        print(f"Resumed from {path} (iteration {data.get('iteration')})", flush=True)
        if continue_run:
            print(f"Continuing at iteration {self.iteration}", flush=True)
        return data
