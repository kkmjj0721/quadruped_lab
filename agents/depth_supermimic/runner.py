# Copyright (c) 2026-2027 zh
"""内容：交替真实深度采集、八步模型拟合和三十二步想象，维护独立日志及检查点。"""

from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import json
import math
from pathlib import Path
import time

import torch

from agents.supermimic.interfaces import TaskState, append_history
from agents.supermimic.exploration import stratified_collection_mask
from agents.supermimic.timing import PhaseTimer

from .buffer import DepthReplay
from .checkpoints import load_checkpoint, save_checkpoint
from .config import DepthExecutionConfig
from .export import export_policy
from .learner import DepthLearner
from .networks import Critic, DepthActor, FailureHead, LatentDecoder, WorldModel


class DepthSuperMimicRunner:
    def __init__(self, env, config, log_dir=None, execution=None, *, collection_groups=None):
        self.env, self.task, self.spec = env, env.task, env.task.spec
        self.device, self.config = torch.device(env.device), deepcopy(config)
        self.execution = deepcopy(execution) if execution is not None else DepthExecutionConfig()
        if collection_groups is None:
            collection_groups = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        if (not isinstance(collection_groups, torch.Tensor) or collection_groups.shape != (env.num_envs,)
                or collection_groups.device != self.device
                or collection_groups.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
            raise ValueError("Collection groups must be an integer tensor with one entry per environment on its device")
        self.deterministic_collection_mask = stratified_collection_mask(collection_groups, config.collect_det_frac)
        cuda = self.device.type == "cuda"
        self.execution.cuda_graph = self.execution.cuda_graph and cuda
        for field in ("actor_bf16", "encoder_bf16", "wm_bf16", "decoder_bf16", "critic_bf16"):
            setattr(self.execution, field, getattr(self.execution, field) and cuda)
        self.actor = DepthActor(self.spec, config, bf16=self.execution.actor_bf16,
                                encoder_bf16=self.execution.encoder_bf16).to(self.device)
        self.model = WorldModel(self.task, config.wm_hidden, bf16=self.execution.wm_bf16).to(self.device)
        self.decoder = LatentDecoder(self.spec.latent_dim, self.spec.depth_shape[0], bf16=self.execution.decoder_bf16).to(self.device)
        self.critic = Critic(self.spec.privileged_dim, config.critic_hidden, bf16=self.execution.critic_bf16).to(self.device)
        self.failure = FailureHead(self.spec.privileged_dim, bf16=self.execution.critic_bf16).to(self.device)
        self.learner = DepthLearner(self.task, self.actor, self.model, self.decoder, self.critic, self.failure, self.config, self.execution)
        self.replay = DepthReplay(self.task, env.num_envs, config.replay_capacity, self.device,
                                  holdout_envs=config.wm_holdout_envs, device_sampling=self.execution.cuda_graph,
                                  history_storage=self.execution.replay_history_storage)
        self._configure_sampling()
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.writer = None
        if self.log_dir is not None:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(self.log_dir))
            self.writer.add_text("run/algorithm", json.dumps(asdict(config), indent=2))
        self.iteration = self.real_steps = self.prefill_real_steps = 0
        self.elapsed = self.training_seconds = 0.0
        self.nonfinite_steps = 0
        self.current = self.history = self.depth = self.hidden = None
        self._collect_accumulator = 0.0
        self._prefilled = False
        self._metrics, self._metric_counts = {}, {}
        self.last_metrics = {}
        self.episode_step = torch.zeros(env.num_envs, device=self.device, dtype=torch.int32)
        self.episode_return = torch.zeros(env.num_envs, device=self.device)

    def _configure_sampling(self):
        """采样读设备计数的稳定地址；恢复和清空回放时重建采样图。"""
        self._model_sampling = self._seed_sampling = None
        if self.execution.cuda_graph:
            from agents.supermimic.cuda_graph import CapturedUpdate

            self._model_sampling = CapturedUpdate(self._sample_model_device, self.device)
            self._seed_sampling = CapturedUpdate(self._sample_seed_device, self.device)

    def _sample_model_device(self):
        return self.replay.sample_windows(self.config.wm_batch, self.config.wm_window,
            failure_frac=self.config.wm_failure_frac, failure_horizon=self.config.wm_failure_horizon, device_indices=True)

    def _sample_seed_device(self):
        return self.replay.sample_starts(self.config.imag_batch, self.config.horizon,
                                        self.config.command_margin_steps, device_indices=True)

    def sample_model(self):
        """模型更新同步消费本次窗口，下一次采样才可复用图输出存储。"""
        if self._model_sampling is not None:
            return self._model_sampling()
        return self.replay.sample_windows(self.config.wm_batch, self.config.wm_window,
            failure_frac=self.config.wm_failure_frac, failure_horizon=self.config.wm_failure_horizon)

    def sample_seed(self):
        """整段策略更新消费完毕后再重放起点采样图。"""
        if self._seed_sampling is not None:
            return self._seed_sampling()
        return self.replay.sample_starts(self.config.imag_batch, self.config.horizon, self.config.command_margin_steps)

    def _accumulate(self, metrics):
        for name, value in metrics.items():
            self._accumulate_sum_count(name, value, 1)

    def _accumulate_sum_count(self, name, total, count):
        """回合均值按实际结束数加权，无回合结束的控制步不稀释分母。"""
        total = torch.as_tensor(total, device=self.device).detach()
        count = torch.as_tensor(count, device=self.device).detach()
        if name not in self._metrics:
            self._metrics[name], self._metric_counts[name] = total.clone(), count.clone()
        else:
            self._metrics[name].add_(total)
            self._metric_counts[name].add_(count)

    @torch.no_grad()
    def reset_collection(self):
        self.current = self.env.reset().detached_clone()
        self.depth = self.env.depth_stack().clone()
        self.hidden = torch.zeros(self.env.num_envs, self.spec.latent_dim, device=self.device)
        frame = self.env.proprio_frame()
        self.history = frame[:, None].expand(-1, self.spec.history_length, -1).clone()
        self.episode_step.zero_()
        self.episode_return.zero_()
        self.replay.clear()
        self._configure_sampling()
        self._prefilled = False
        self._collect_accumulator = 0.0

    @torch.no_grad()
    def collect(self, steps):
        if self.current is None:
            self.reset_collection()
        rewards = torch.zeros((), device=self.device)
        terminated = torch.zeros((), device=self.device)
        truncated = torch.zeros((), device=self.device)
        for _ in range(steps):
            action, next_hidden = self.actor(self.history, self.depth, self.hidden)
            noise = self.config.action_noise * torch.randn_like(action)
            quiet = self.deterministic_collection_mask
            if self.config.standing_zero_noise:
                quiet = quiet | self.task.standing_mask(self.current)
            noise.masked_fill_(quiet[:, None], 0)
            action = (action + noise).clamp(-20, 20)
            if not torch.isfinite(action).all():
                raise FloatingPointError("Collected actions contain non-finite values")
            result = self.env.step(action)
            executed = result.executed_action
            self.replay.add(self.current, self.depth, executed, result, self.episode_step, history=self.history, hidden=self.hidden)
            self.model.update_statistics(self.current, executed, result.final_state)
            done = result.terminated | result.truncated
            self.episode_return.add_(result.reward)
            rewards += result.reward.mean() / self.task.dt
            terminated += result.terminated.sum()
            truncated += result.truncated.sum()
            self.episode_step.add_(1)
            completed = done.sum()
            self._accumulate({"sim/completed_episodes_per_step": completed})
            self._accumulate_sum_count("sim/completed_episode_steps", (self.episode_step * done).sum(), completed)
            self._accumulate_sum_count("sim/completed_episode_return", (self.episode_return * done).sum(), completed)
            task_metrics = {f"task/{name}": value for name, value in self.env.last_metrics.items()}
            task_metrics.update({f"Reward_Rate/{name[7:]}": value / self.task.dt
                                 for name, value in self.env.last_metrics.items() if name.startswith("reward/")})
            self._accumulate(task_metrics)
            self.episode_return.masked_fill_(done, 0)
            self.episode_step.masked_fill_(done, 0)
            frame = result.next_observation
            history = append_history(self.history, frame)
            self.history = torch.where(done[:, None, None], frame[:, None].expand_as(history), history)
            self.current = result.next_state.detached_clone()
            self.depth = result.next_depth.clone()
            self.hidden = next_hidden.masked_fill(done[:, None], 0)
            self.real_steps += self.env.num_envs
        self._accumulate_sum_count("sim/reward_rate", rewards, steps)
        self._accumulate_sum_count("sim/terminated", terminated, steps)
        self._accumulate_sum_count("sim/truncated", truncated, steps)

    @torch.no_grad()
    def evaluate_model(self, batch=256):
        cfg = self.config
        if self.replay.size < cfg.horizon:
            return {}
        window = self.replay.sample_windows(batch, cfg.horizon, env_range=self.replay.holdout_envs)
        state = window.states.index((slice(None), 0))
        initial = self.actor.encoder(window.depth[:, 0].float() / 255)
        latent, metrics = initial, {}
        for step in range(cfg.horizon):
            state = TaskState(state.physical, state.memory, window.states.command[:, step], window.states.context[:, step])
            state, _, latent = self.model(state, window.actions[:, step], latent)
            if step + 1 in (1, 8, 16, 32):
                target = self.actor.encoder(window.next_depth[:, step].float() / 255)
                scale = target.var(0).mean().clamp_min(1e-6)
                mask = window.valid[:, step]
                for name, value in (("predicted", latent), ("copy", initial)):
                    metrics[f"holdout/latent_{step + 1}_{name}"] = ((value - target).square().mean(-1) * mask).sum() / mask.sum().clamp_min(1) / scale
        return metrics

    def _flush_metrics(self, *, print_log, started, target_iteration, phase_timers, window_started, window_steps):
        names = list(self._metrics)
        if not names:
            return
        totals = torch.stack([self._metrics[name] for name in names])
        counts = torch.stack([self._metric_counts[name] for name in names])
        values, host_counts = torch.stack((totals / counts.clamp_min(1), counts)).cpu().tolist()
        row = {name: value for name, value, count in zip(names, values, host_counts) if count > 0}
        invalid = [name for name, value in row.items() if not math.isfinite(value)]
        self.nonfinite_steps += len(invalid)
        if invalid:
            raise FloatingPointError(f"Training metrics contain non-finite values: {invalid}")
        self.last_metrics = row
        if self.writer is not None:
            for name, value in row.items():
                self.writer.add_scalar(name, value, self.iteration)
            self.writer.add_scalar("perf/real_steps", self.real_steps, self.iteration)
            self.writer.flush()
        if print_log:
            finished = time.perf_counter()
            window_seconds = finished - window_started
            iterations = len(phase_timers)
            collection_seconds = sum(timer.seconds("start", "collection") for timer in phase_timers) / iterations
            learning_seconds = sum(timer.seconds("collection", "learning") for timer in phase_timers) / iterations
            iteration_seconds = window_seconds / iterations
            fields = [
                ("Total steps", str(self.real_steps)),
                ("Steps per second", f"{(self.real_steps - window_steps) / max(window_seconds, 1e-9):.0f}"),
                ("Collection time", f"{collection_seconds:.3f}s"),
                ("Learning time", f"{learning_seconds:.3f}s"),
            ]
            if self.iteration <= self.config.warmup_iterations:
                fields.append(("Training phase", "world-model only"))
            for label, key in (
                ("Mean world model loss", "wm/loss"),
                ("Mean latent prediction error", "latent/err_window"),
                ("Mean depth reconstruction loss", "latent/reconstruction_l1"),
                ("Mean value loss", "critic/loss"),
                ("Mean control loss", "policy/loss"),
            ):
                if key in row:
                    fields.append((label, f"{row[key]:.4f}"))
            if "perf/actor_learning_rate" in row:
                fields.append(("Control learning rate", f"{row['perf/actor_learning_rate']:.2e}"))
            for label, key in (("Mean reward", "sim/completed_episode_return"),
                               ("Mean episode length", "sim/completed_episode_steps")):
                if key in row:
                    fields.append((label, f"{row[key]:.2f}"))
            for label, key in (
                ("Mean reward rate", "sim/reward_rate"),
                ("Metrics/terminations_per_step", "sim/terminated"),
                ("Metrics/timeouts_per_step", "sim/truncated"),
                ("Metrics/terrain_level", "task/terrain_level"),
                ("Metrics/action_rms", "task/action_rms"),
            ):
                if key in row:
                    fields.append((label, f"{row[key]:.4f}"))
            for label, key in (("Metrics/linear_velocity_rmse", "task/tracking_linear_mse"),
                               ("Metrics/yaw_rate_rmse", "task/tracking_angular_mse")):
                if key in row:
                    fields.append((label, f"{math.sqrt(max(0, row[key])):.4f}"))
            fields.extend((key, f"{value:.4f}") for key, value in row.items() if key.startswith("Reward_Rate/"))
            lines = ["#" * 80, f" Learning iteration {self.iteration}/{target_iteration} ".center(80), ""]
            lines.extend(f"{label + ':':>40} {value}" for label, value in fields)
            footer = [
                ("Iteration time", f"{iteration_seconds:.2f}s"),
                ("Time elapsed", str(timedelta(seconds=int(finished - started)))),
                ("ETA", str(timedelta(seconds=int(iteration_seconds * (target_iteration - self.iteration))))),
            ]
            lines.extend(["-" * 80, *(f"{label + ':':>40} {value}" for label, value in footer), ""])
            print("\n".join(lines), flush=True)
        self._metrics.clear()
        self._metric_counts.clear()

    def learn(self, max_iterations, *, save_interval=5000, log_interval=50):
        if max_iterations < self.iteration or save_interval < 1 or log_interval < 1:
            raise ValueError("Training target cannot precede the restored iteration; intervals must be positive")
        if self.current is None:
            self.reset_collection()
        session_start = time.perf_counter()
        log_started = session_start
        if not self._prefilled:
            self.collect(self.config.warmup_steps)
            self.prefill_real_steps += self.config.warmup_steps * self.env.num_envs
            self._prefilled = True
            print(f"[INFO] Replay warmup completed: vector_steps={self.replay.size}, transitions={self.real_steps}", flush=True)
        training_start = time.perf_counter()
        window_started, window_steps = training_start, self.real_steps
        phase_timers = []
        while self.iteration < max_iterations:
            timer = PhaseTimer(self.device)
            # 新渐增排期按Go2从第1轮起算；历史无渐增配置沿用原轮次索引。
            rate = self.learner.schedule(self.iteration + int(self.config.actor_ramp_iterations > 0))
            self._collect_accumulator += self.config.collect_steps
            steps = int(self._collect_accumulator)
            self._collect_accumulator -= steps
            self.collect(steps)
            timer.mark("collection")
            for _ in range(self.config.wm_updates):
                window = self.sample_model()
                self._accumulate(self.learner.fit_model(window))
            if self.iteration >= self.config.warmup_iterations:
                seed = self.sample_seed()
                self._accumulate(self.learner.improve(seed, self.iteration))
            timer.mark("learning")
            phase_timers.append(timer)
            self.iteration += 1
            self._accumulate({"perf/learning_rate": rate,
                              "perf/actor_learning_rate": self.learner.opt_actor.param_groups[0]["lr"]})
            if self.iteration % self.execution.metric_interval == 0 or self.iteration == max_iterations:
                self._accumulate(self.evaluate_model())
                self._flush_metrics(print_log=self.iteration % log_interval == 0 or self.iteration == max_iterations,
                                    started=log_started, target_iteration=max_iterations, phase_timers=phase_timers,
                                    window_started=window_started, window_steps=window_steps)
                phase_timers.clear()
                window_started, window_steps = time.perf_counter(), self.real_steps
            if self.iteration % save_interval == 0 or self.iteration == max_iterations:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self.training_seconds += time.perf_counter() - training_start
                self.elapsed += time.perf_counter() - session_start
                if self.log_dir is not None:
                    self.save(self.log_dir / f"model_{self.iteration}.pt")
                    self.export_policy(self.log_dir / "policy.pt")
                session_start = training_start = time.perf_counter()

    def save(self, path):
        save_checkpoint(self, path)

    def load(self, path):
        return load_checkpoint(self, path)

    def export_policy(self, path, *, onnx=False):
        snapshot = {}
        if self.log_dir is not None and (self.log_dir / "task_config.json").exists():
            snapshot = json.loads((self.log_dir / "task_config.json").read_text(encoding="utf-8"))
        return export_policy(self.actor, path, spec=self.spec, iteration=self.iteration, metadata=snapshot, onnx=onnx)

    def close(self):
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            self.env.close()
