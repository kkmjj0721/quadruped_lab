# Copyright (c) 2026-2027 zh
"""
内容：
    调度真实双组采集、世界模型预热、想象学习及参数续训。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .interfaces import PolicyInput, Transition, append_history, validate_policy_input
from .logging import TrainingLogger
from .metrics import EpisodeStatistics
from .storage.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class RunnerConfig:
    replay_capacity: int = 256
    initial_collection_steps: int = 100
    collection_steps: int = 8
    world_model_updates: int = 4
    model_warmup_updates: int = 200
    window_length: int = 8
    world_model_batch_size: int = 256
    imagination_batch_size: int = 256
    student_batch_size: int = 256
    teacher_fraction: float = 0.75
    action_noise_std: float = 0.1
    enable_control_updates: bool = False

    def __post_init__(self):
        counts = (self.replay_capacity, self.initial_collection_steps, self.collection_steps, self.world_model_updates,
                  self.window_length, self.world_model_batch_size, self.imagination_batch_size, self.student_batch_size)
        if min(counts) < 1 or self.model_warmup_updates < 0:
            raise ValueError("Runner counts must be positive")
        if self.replay_capacity < self.window_length or self.imagination_batch_size < 2:
            raise ValueError("Insufficient replay capacity or imagination batch size")
        if not 0 < self.teacher_fraction < 1 or not math.isfinite(self.action_noise_std) or self.action_noise_std < 0:
            raise ValueError("Invalid teacher fraction or action noise")


class SuperMimicRunner:
    """adapter提供物理语义；启用控制前应由调用方完成模型与梯度验收。"""

    def __init__(self, env, learner, world_model_trainer, config: RunnerConfig | None = None,
                 log_dir: str | Path | None = None, *, control_gate=None, replay_privileged_fn=None):
        self.env = env
        self.learner = learner
        self.policy = learner.policy
        self.world_model = learner.world_model
        self.task = self.world_model.task
        self.spec = self.task.spec
        self.world_model_trainer = world_model_trainer
        self.config = config or RunnerConfig()
        self.control_gate = control_gate
        self.device = next(self.policy.parameters()).device
        if torch.device(env.device) != self.device or next(self.world_model.parameters()).device != self.device:
            raise ValueError("Environment and models must use the same device")
        if world_model_trainer.model is not self.world_model:
            raise ValueError("Trainer and learner must share the world model")
        if env.num_envs < 2:
            raise ValueError("CTS collection requires at least two environments")
        teachers = max(1, min(env.num_envs - 1, round(env.num_envs * self.config.teacher_fraction)))
        self.branch = torch.arange(env.num_envs, device=self.device) < teachers
        self.replay = ReplayBuffer(self.spec, env.num_envs, self.config.replay_capacity, self.device,
                                   privileged_fn=replay_privileged_fn)
        self.episode_id = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.step_id = torch.zeros_like(self.episode_id)
        self.current = None
        self.iteration = 0
        self.real_steps = 0
        self.imagined_steps = 0
        self.world_model_steps = 0
        self.world_model_prediction_steps = 0
        self.control_updates = 0
        self.warmup_remaining = self.config.model_warmup_updates
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.episode_statistics = EpisodeStatistics(env.num_envs, self.device, step_dt=getattr(env, "step_dt", None))
        self.configure_logging()
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            source_hashes = {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in Path(__file__).parent.rglob("*.py")}
            snapshot = {"spec": asdict(self.spec), "spec_hash": self.spec.hash, "runner": asdict(self.config),
                        "policy": self.policy.construction_config,
                        "learner": asdict(self.learner.config), "torch": torch.__version__, "seed": torch.initial_seed(),
                        "source_hashes": source_hashes}
            (self.log_dir / "config.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    def configure_logging(self, *, target_iteration: int | None = None, log_interval: int = 1) -> None:
        """设置本次运行的完整目标；保存检查点的分段learn不重置显示与计时。"""
        if target_iteration is not None and (isinstance(target_iteration, bool) or not isinstance(target_iteration, int)
                                             or target_iteration < self.iteration):
            raise ValueError("Logging target must be an integer at or beyond the current iteration")
        self.logger = TrainingLogger(self.log_dir, num_envs=self.env.num_envs, device=str(self.device), log_interval=log_interval)
        self._log_target_iteration = target_iteration
        self._log_session_start_iteration = self.iteration
        self._log_training_seconds = 0.0
        self._log_completed_iterations = 0

    def _synchronize_logging(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _collection_step_metrics(self) -> dict[str, float]:
        """任务子类提供本次真实步的诊断值，通用算法不依赖任务实现。"""
        return {}

    def _policy_input(self, state, history=None):
        frame = self.task.observe(state, noisy=self.learner.config.observation_noise)
        if history is None:
            history = frame.unsqueeze(1).repeat(1, self.spec.history_length, 1)
        data = PolicyInput(state, frame, history, self.task.privileged(state), self.branch.clone())
        validate_policy_input(data, self.spec)
        return data

    def reset_collection(self):
        """外部reset后清空回放，避免把新的episode接到原连续片段。"""
        self.replay.clear()
        self.episode_statistics.reset()
        self.episode_id.zero_()
        self.step_id.zero_()
        self.current = self._policy_input(self.env.reset()).detached_clone()
        self.warmup_remaining = self.config.model_warmup_updates

    @torch.no_grad()
    def collect(self, steps: int):
        if steps < 1:
            raise ValueError("Collection steps must be positive")
        if self.current is None:
            self.reset_collection()
        self.episode_statistics.begin_collection()
        self.policy.train()
        reward_sum, timeout_values, timeouts, failures = 0.0, 0.0, 0, 0
        teacher_rewards, student_rewards = 0.0, 0.0
        task_totals, task_counts = {}, {}
        for _ in range(steps):
            self.policy.update_normalization(self.current)
            action = self.policy.act(self.current)
            if self.config.action_noise_std:
                action = action + self.config.action_noise_std * torch.randn_like(action)
            if self.policy.action_limit is not None:
                action = action.clamp(-self.policy.action_limit, self.policy.action_limit)
            result = self.env.step(action)
            count = self.env.num_envs
            for tensor, dtype in ((result.reward, torch.float32), (result.terminated, torch.bool),
                                  (result.truncated, torch.bool), (result.context_version, torch.long)):
                if tensor.shape != (count,) or tensor.dtype != dtype or tensor.device != self.device:
                    raise ValueError("Invalid environment transition metadata")
            final = self._policy_input(result.final_state)
            final.history = append_history(self.current.history, final.observation)
            transition = Transition(self.current, final, action, result.reward, result.terminated, result.truncated,
                                    self.episode_id, self.step_id, result.context_version)
            self.replay.add(transition)
            # 失败与超时同时出现时，物理失败优先，不做超时bootstrap。
            timeout = result.truncated & ~result.terminated
            if timeout.any():
                timeout_values += float(self.policy.value(final.index(timeout)).sum())
                timeouts += int(timeout.sum())
            done = result.terminated | result.truncated
            self.episode_statistics.update(result.reward, result.terminated, result.truncated, self.branch)
            next_state = result.next_state.detached_clone()
            observation = final.observation.clone()
            history = final.history.clone()
            if done.any():
                reset_frame = self.task.observe(next_state.index(done), noisy=self.learner.config.observation_noise)
                observation[done] = reset_frame
                history[done] = reset_frame.unsqueeze(1).expand(-1, self.spec.history_length, -1)
            self.current = PolicyInput(next_state, observation, history, self.task.privileged(next_state), self.branch.clone())
            validate_policy_input(self.current, self.spec)
            self.current = self.current.detached_clone()
            self.episode_id += done.long()
            self.step_id = torch.where(done, torch.zeros_like(self.step_id), self.step_id + 1)
            reward_sum += float(result.reward.mean())
            teacher_rewards += float(result.reward[self.branch].mean())
            student_rewards += float(result.reward[~self.branch].mean())
            failures += int(result.terminated.sum())
            for name, value in self._collection_step_metrics().items():
                task_totals[name] = task_totals.get(name, 0.0) + float(value)
                task_counts[name] = task_counts.get(name, 0) + 1
            self.real_steps += count
        return {"real_reward": reward_sum / steps, "real_teacher_reward": teacher_rewards / steps,
                "real_student_reward": student_rewards / steps, "failures": float(failures),
                "timeouts": float(timeouts), "timeout_final_value": timeout_values / max(1, timeouts),
                **self.episode_statistics.metrics(),
                **{name: value / task_counts[name] for name, value in task_totals.items()}}

    def _fit_world_model(self, updates):
        sums = {}
        for _ in range(updates):
            window = self.replay.sample_windows(self.config.world_model_batch_size, self.config.window_length)
            metrics = self.world_model_trainer.update(window)
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + value
            if not metrics.get("skipped", 0):
                self.world_model_steps += 1
                self.world_model_prediction_steps += int(window.valid.sum())
        return {f"wm_{name}": value / updates for name, value in sums.items()} if updates else {}

    def learn(self, iterations: int):
        if iterations < 1:
            raise ValueError("Learning iterations must be positive")
        outputs = []
        target_iteration = max(self._log_target_iteration or 0, self.iteration + iterations)
        for _ in range(iterations):
            self._synchronize_logging()
            iteration_started = time.perf_counter()
            real_before, world_before, control_before = self.real_steps, self.world_model_steps, self.control_updates
            metrics = self.collect(self.config.initial_collection_steps if len(self.replay) == 0 else self.config.collection_steps)
            self._synchronize_logging()
            collection_seconds = time.perf_counter() - iteration_started
            world_seconds, control_seconds, warmup = 0.0, 0.0, 0
            if not self.replay.can_sample(self.config.window_length):
                metrics["model_not_ready"] = 1.0
            else:
                world_started = time.perf_counter()
                warmup = self.warmup_remaining
                metrics.update(self._fit_world_model(self.config.world_model_updates + self.warmup_remaining))
                self._synchronize_logging()
                world_seconds = time.perf_counter() - world_started
                self.warmup_remaining = 0
                allowed = self.config.enable_control_updates and (self.control_gate is None or self.control_gate(metrics))
                if allowed:
                    control_started = time.perf_counter()
                    seeds = self.replay.sample_seeds(self.config.imagination_batch_size, self.config.teacher_fraction)
                    students = self.replay.sample_students(self.config.student_batch_size)
                    metrics.update(self.learner.update(seeds, students))
                    self.control_updates += 1
                    self.imagined_steps += self.learner.config.horizon * seeds.branch.numel()
                    self._synchronize_logging()
                    control_seconds = time.perf_counter() - control_started
                metrics["control_enabled"] = float(allowed)
            self.iteration += 1
            iteration_seconds = time.perf_counter() - iteration_started
            self._log_training_seconds += iteration_seconds
            self._log_completed_iterations += 1
            real_delta = self.real_steps - real_before
            metrics.update(iteration=self.iteration, real_steps=self.real_steps, imagined_steps_total=self.imagined_steps,
                           world_model_updates_total=self.world_model_steps,
                           world_model_prediction_steps=self.world_model_prediction_steps, control_updates_total=self.control_updates)
            metrics.update({
                "training_target_iteration": target_iteration,
                "training_session_start_iteration": self._log_session_start_iteration,
                "real_steps_delta": real_delta,
                "world_model_updates_delta": self.world_model_steps - world_before,
                "world_model_warmup_updates": warmup,
                "control_updates_delta": self.control_updates - control_before,
                "replay/transitions": len(self.replay), "replay/steps_per_env": len(self.replay) / self.env.num_envs,
                "timing/collection_seconds": collection_seconds, "timing/world_model_seconds": world_seconds,
                "timing/control_seconds": control_seconds, "timing/iteration_seconds": iteration_seconds,
                "timing/session_elapsed_seconds": self._log_training_seconds,
                "timing/eta_seconds": self._log_training_seconds / self._log_completed_iterations * max(0, target_iteration - self.iteration),
                "performance/real_collection_fps": real_delta / max(collection_seconds, 1e-9),
                "performance/real_total_fps": real_delta / max(iteration_seconds, 1e-9),
            })
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Non-finite runner metrics")
            self.logger.record(metrics)
            outputs.append(metrics)
        return outputs

    def save(self, path: str | Path):
        """仅保存参数续训；仿真状态与回放不纳入此格式。"""
        checkpoint = {
            "format": "supermimic_parameter_resume_v1", "spec": asdict(self.spec), "spec_hash": self.spec.hash,
            "runner_config": asdict(self.config), "policy_config": self.policy.construction_config,
            "policy": self.policy.state_dict(), "world_model": self.world_model.state_dict(),
            "learner": self.learner.state_dict(), "trainer": self.world_model_trainer.state_dict(), "branch": self.branch.cpu(),
            "counters": {name: getattr(self, name) for name in ("iteration", "real_steps", "imagined_steps", "world_model_steps",
                                                               "world_model_prediction_steps", "control_updates")},
            "rng_torch": torch.get_rng_state(), "rng_python": random.getstate(),
            "rng_cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)

    def load(self, path: str | Path):
        """加载可信本项目检查点；重新reset采集、清回放并预热模型。"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format") != "supermimic_parameter_resume_v1" or checkpoint.get("spec_hash") != self.spec.hash:
            raise ValueError("Incompatible SuperMimic checkpoint")
        if checkpoint.get("spec") != asdict(self.spec) or checkpoint["runner_config"] != asdict(self.config):
            raise ValueError("Checkpoint configuration mismatch")
        if checkpoint.get("policy_config") != self.policy.construction_config:
            raise ValueError("Checkpoint policy construction mismatch")
        if not torch.equal(checkpoint["branch"].to(self.device), self.branch):
            raise ValueError("Checkpoint environment grouping mismatch")
        self.policy.load_state_dict(checkpoint["policy"])
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.learner.load_state_dict(checkpoint["learner"])
        self.world_model_trainer.load_state_dict(checkpoint["trainer"])
        for name, value in checkpoint["counters"].items():
            setattr(self, name, value)
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        random.setstate(checkpoint["rng_python"])
        if self.device.type == "cuda" and checkpoint["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng_cuda"]])
        self.current = None
        self.episode_statistics.reset()
        self.configure_logging(log_interval=self.logger.log_interval)
        self.replay.clear()
        self.warmup_remaining = self.config.model_warmup_updates
        return {"resume_mode": "parameters", "replay_restored": False, "environment_restored": False}

    def get_inference_policy(self):
        """返回只接收盲当前帧和历史的学生策略，历史由调用方管理。"""
        self.policy.eval()
        return self.policy.student_action
