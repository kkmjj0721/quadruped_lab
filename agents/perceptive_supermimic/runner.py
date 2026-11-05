# Copyright (c) 2026-2027 zh
"""内容：装配同步视觉采集、模型拟合、想象更新与显式检查点恢复。"""

from dataclasses import asdict
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch

from .buffer import Replay
from .config import ExecutionConfig, PerceptiveConfig
from .interfaces import append_history, validate_decision
from .learner import Learner
from .policy import Critic, PerceptiveActor
from .vision import DepthDecoder, DepthEncoder
from .world_model import WorldModel


logger = logging.getLogger(__name__)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _numbers(metrics):
    keys = [name for name, value in metrics.items() if isinstance(value, torch.Tensor)]
    values = torch.stack([metrics[name].detach().float().reshape(()) for name in keys]).cpu().tolist() if keys else []
    result = {name: float(value) for name, value in metrics.items() if name not in keys}
    result.update(zip(keys, values))
    if not all(math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Training metrics are non-finite")
    return result


class PerceptiveRunner:
    """仅依赖任务协议；日志接入使用callback，机器人环境和奖励由调用方装配。"""

    def __init__(self, env, config=None, *, execution=None, log_dir=None):
        self.env, self.task, self.spec = env, env.task, env.task.spec
        self.device = torch.device(env.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        _integer(env.num_envs, "Environment count")
        if not math.isfinite(self.task.dt) or self.task.dt <= 0:
            raise ValueError("Task control timestep must be finite and positive")
        self.config = config or PerceptiveConfig()
        self.execution = execution or ExecutionConfig()
        config = self.config
        self.encoder = DepthEncoder(self.spec.depth_shape, config.latent_dim).to(self.device)
        self.decoder = DepthDecoder(self.spec.depth_shape, config.latent_dim).to(self.device)
        self.actor = PerceptiveActor(self.spec, config, self.encoder).to(self.device)
        self.model = WorldModel(self.task, config).to(self.device)
        self.critic = Critic(self.spec.privileged_dim, config.critic_hidden).to(self.device)
        self.learner = Learner(self.task, self.actor, self.model, self.critic, self.decoder, config, self.execution)
        self.replay = Replay(self.spec, env.num_envs, config.replay_capacity, config.image_capacity, self.device)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.iteration = self.real_steps = self.imagined_steps = 0
        self.training_seconds = self._collect_accumulator = 0.0
        self.current = self.history = None
        self._needs_prefill = True

    @torch.no_grad()
    def reset_collection(self):
        decision = self.env.reset()
        validate_decision(decision, self.spec, self.env.num_envs, self.device)
        self.current = decision.detached_clone()
        self.history = decision.observation.proprio[:, None].repeat(1, self.spec.history_length, 1)
        self.replay.clear()
        self._needs_prefill = True

    @torch.no_grad()
    def collect(self, steps):
        _integer(steps, "Collection steps")
        if self.current is None:
            self.reset_collection()
        reward = torch.zeros((), device=self.device)
        failures = torch.zeros_like(reward)
        timeouts = torch.zeros_like(reward)
        for _ in range(steps):
            action = self.actor(self.history, self.current.observation.depth)
            action = (action + self.config.action_noise * torch.randn_like(action)).clamp(-self.config.action_clip, self.config.action_clip)
            result = self.env.step(action)
            # Replay.add是统一的真实转移校验边界，后续只消费已确认的协议。
            self.replay.add(self.current, self.history, action, result)
            self.actor.update_statistics(self.history)
            done = result.terminated | result.truncated
            frame = result.next.observation.proprio
            history = append_history(self.history, frame)
            self.history = torch.where(done[:, None, None], frame[:, None].expand_as(history), history)
            self.current = result.next.detached_clone()
            reward += result.reward.mean()
            failures += result.terminated.sum()
            timeouts += (result.truncated & ~result.terminated).sum()
            self.real_steps += self.env.num_envs
        return {"real_reward": reward / steps, "failures": failures, "timeouts": timeouts}

    def _prepare(self):
        if self.current is None:
            self.reset_collection()
        if self._needs_prefill:
            remaining = max(0, self.config.warmup_steps - self.replay.size)
            if remaining:
                self.collect(remaining)
            self._needs_prefill = False
        if not self.replay.ready(self.config.wm_window):
            raise RuntimeError("Replay has no valid model window; check image capacity and environment frame/context continuity")

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def learn(self, iterations, *, callback=None):
        """执行指定数量的额外迭代；初次采集和恢复预填不计入实际训练时间。"""
        _integer(iterations, "Training iterations")
        self._prepare()
        latest = {}
        for _ in range(iterations):
            self._synchronize()
            started = time.perf_counter()
            self._collect_accumulator += self.config.collect_steps
            steps = math.floor(self._collect_accumulator)
            self._collect_accumulator -= steps
            metrics = self.collect(steps)
            self.learner.schedule(self.iteration)
            model_metrics = {}
            for _ in range(self.config.wm_updates):
                window = self.replay.sample_windows(self.config.wm_batch, self.config.wm_window)
                for name, value in self.learner.fit_model(window).items():
                    model_metrics[name] = model_metrics.get(name, 0) + value / self.config.wm_updates
            metrics.update(model_metrics)
            if self.iteration >= self.config.wm_only_iterations:
                seed = self.replay.sample_seeds(self.config.imag_batch)
                metrics.update(self.learner.improve(seed))
                self.imagined_steps += self.config.imag_batch * self.config.horizon
            self.iteration += 1
            self._synchronize()
            self.training_seconds += time.perf_counter() - started
            metrics.update(iteration=self.iteration, real_steps=self.real_steps, imagined_steps=self.imagined_steps,
                           model_updates=self.learner.model_updates, actor_updates=self.learner.actor_updates,
                           critic_updates=self.learner.critic_updates, training_seconds=self.training_seconds)
            latest = _numbers(metrics)
            if callback is not None:
                callback(latest)
            if self.iteration % self.config.log_interval == 0:
                logger.info("Iteration %d: model_loss=%.6f, real_steps=%d", self.iteration, latest["model_loss"], self.real_steps)
            if self.log_dir is not None and self.iteration % self.config.save_interval == 0:
                self.save(self.log_dir / f"model_{self.iteration}.pt")
        return latest

    def save(self, path):
        """保存训练状态；仿真与回放不持久化，恢复后必须重新采集。"""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": self.spec.model_id, "spec": asdict(self.spec), "spec_hash": self.spec.hash,
                   "config": asdict(self.config), "execution": asdict(self.execution),
                   "actor": self.actor.state_dict(), "model": self.model.state_dict(),
                   "critic": self.critic.state_dict(), "decoder": self.decoder.state_dict(),
                   "optimizers": {name: getattr(self.learner, f"{name}_optimizer").state_dict()
                                  for name in ("model", "actor", "critic")},
                   "counters": {"iteration": self.iteration, "real_steps": self.real_steps,
                                "imagined_steps": self.imagined_steps, "training_seconds": self.training_seconds,
                                "collect_accumulator": self._collect_accumulator,
                                **{f"{name}_updates": getattr(self.learner, f"{name}_updates") for name in ("model", "actor", "critic")}},
                   "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None}
        with TemporaryDirectory(prefix=".perceptive_checkpoint_", dir=destination.parent) as temporary:
            pending = Path(temporary) / "model.pt"
            torch.save(payload, pending)
            pending.replace(destination)
        return destination

    def load(self, path):
        """仅恢复相同任务与算法配置，设备和微批预算允许由当前运行选择。"""
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        if payload["format"] != self.spec.model_id or payload["spec_hash"] != self.spec.hash or payload["spec"] != asdict(self.spec):
            raise ValueError("Checkpoint task, depth or algorithm semantics do not match")
        if payload["config"] != asdict(self.config):
            raise ValueError("Checkpoint algorithm configuration does not match")
        modules = {"actor": self.actor, "model": self.model, "critic": self.critic, "decoder": self.decoder}
        # 在写入任何网络前核对完整权重布局，避免把形状兼容误当成任务兼容。
        for name, module in modules.items():
            current = module.state_dict()
            saved = payload[name]
            if saved.keys() != current.keys() or any(saved[key].shape != value.shape for key, value in current.items()):
                raise ValueError(f"Checkpoint {name} tensor layout does not match")
            if any(not bool(torch.isfinite(value).all()) for value in saved.values()):
                raise ValueError(f"Checkpoint {name} contains non-finite tensors")
        for name, module in modules.items():
            module.load_state_dict(payload[name])
        for name in ("model", "actor", "critic"):
            getattr(self.learner, f"{name}_optimizer").load_state_dict(payload["optimizers"][name])
            setattr(self.learner, f"{name}_updates", payload["counters"][f"{name}_updates"])
        for name in ("iteration", "real_steps", "imagined_steps", "training_seconds"):
            setattr(self, name, payload["counters"][name])
        self._collect_accumulator = payload["counters"]["collect_accumulator"]
        torch.set_rng_state(payload["torch_rng"].cpu())
        if self.device.type == "cuda" and payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state(payload["cuda_rng"].cpu(), self.device)
        self.current = self.history = None
        self._needs_prefill = True
        self.replay.clear()
