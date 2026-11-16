# Copyright (c) 2026-2027 zh
"""内容：串联同步深度采集、完整循环回放、模型拟合、想象更新与参数续训。"""

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import time

import torch

from .execution_config import ExecutionConfig
from .exploration import stratified_collection_mask
from .tensor_state import append_history
from .logging import TrainingLogger
from .metrics import MetricWindow, read_metrics

from .buffer import DepthReplay
from .config import DepthNetworkConfig, DepthSuperMimicConfig
from .export import export_depth_policy
from .learner import DepthLearner
from .policy import DepthActor, DepthCritic
from .vision import DepthDecoder
from .world_model import DepthWorldModel


class DepthSuperMimicRunner:
    """装配视觉训练执行方式，保持完整八步监督和三十二步反向链。"""

    def __init__(self, env, config: DepthSuperMimicConfig, physical_model_factory, *, log_dir=None,
                 execution=None, networks=None, image_capacity=None, control_enabled=True, metric_interval=10):
        self.env, self.task, self.spec = env, env.task, env.task.spec
        self.device = torch.device(env.device)
        self.config = config
        self.networks = networks or DepthNetworkConfig(actor_hidden=config.actor_hidden,
                                                       wm_hidden=config.wm_hidden, critic_hidden=config.critic_hidden)
        self.execution = execution or ExecutionConfig(imag_microbatch=16, critic_microbatch=256,
                                                       wm_compile=False, cuda_graph=False, fused_adam=False)
        if metric_interval < 1:
            raise ValueError("Metric interval must be positive")
        self.metric_interval = metric_interval
        if self.spec.model_id != "supermimic_depth_cnn_gru_v1":
            raise ValueError("Depth runner requires the depth CNN-GRU task model identifier")
        expected = (47, 5, 12, 262, 80, 239)
        actual = (self.spec.observation_dim, self.spec.history_length, self.spec.action_dim,
                  self.spec.privileged_dim, self.spec.delta_dim, self.task.model_input_dim)
        if actual != expected:
            raise ValueError(f"Depth task dimensions differ from the network protocol: {actual}")
        self.actor = DepthActor(self.networks).to(self.device)
        self.decoder = DepthDecoder(self.networks).to(self.device)
        physical_model = physical_model_factory(self.task, config.wm_hidden).to(self.device)
        self.model = DepthWorldModel(physical_model, self.actor, self.networks,
                                     feature_coef=config.feature_coef).to(self.device)
        self.critic = DepthCritic(self.networks).to(self.device)
        self.model.use_bf16 = self.execution.wm_bf16
        self.actor.actor.use_bf16 = self.execution.actor_bf16
        self.critic.net.use_bf16 = self.execution.critic_bf16
        learner_type = DepthLearner
        if self.execution.cuda_graph:
            from .cuda_graph import CudaGraphDepthLearner

            learner_type = CudaGraphDepthLearner
        self.learner = learner_type(self.task, self.actor, self.decoder, self.model, self.critic,
                                    config, self.execution)
        period = self.networks.camera_period
        max_episode = int(round(env.cfg.env.episode_length_s / env.step_dt))
        # 短回合可每控制步重置并立即拍首帧；主环按最坏每slot一帧预算，
        # 另留最长活跃/跨环回合的完整低频前缀，不能只按稳定10Hz估算。
        frame_budget = env.num_envs * (config.replay_capacity + (max_episode + period - 1) // period + 2)
        self.replay = DepthReplay(self.spec, env.num_envs, config.replay_capacity, self.device,
                                   image_capacity=image_capacity or frame_budget,
                                   archive_capacity=config.archive_size)
        self.collection_mask = stratified_collection_mask(env.terrain_curriculum.family_ids, config.collect_det_frac)
        self.control_enabled = control_enabled
        self.iteration = self.real_steps = self.imagined_steps = 0
        self.current = self.history = self.hidden = self.z = None
        self.collect_accumulator = 0.
        self.elapsed = 0.
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.logger = TrainingLogger(self.log_dir, num_envs=env.num_envs, device=self.device)
        self.metrics = MetricWindow(self.device)
        self.target_iteration = 0
        self._safe_policy = None
        self._collapse_strikes = 0
        self._guard_falls = 0.0
        self._guard_reset_steps = 0

    def memory_budget(self):
        """区分CPU原始帧预算与有界GPU暂存，不将像素量冒充训练显存。"""
        pixels = self.networks.image_height * self.networks.image_width
        return {"image_capacity": self.replay.image_capacity,
                "image_storage_upper_bytes": self.replay.image_capacity * pixels * 5,
                "image_storage_device": "cpu",
                "image_storage_accounting": "CPU raw frame pixels in independent row storage; excludes tensor metadata and transfer buffers",
                "image_storage_block_rows": 1,
                "image_transfer_block_rows": self.replay.frame_block_rows,
                "retained_image_pixel_bytes": self.replay.image_count * pixels * 5,
                "gpu_full_episode_pixel_storage": False,
                "gpu_prefix_fp32_batch_upper_bytes": max(self.env.num_envs, self.config.wm_batch, self.config.imag_batch) * pixels * 4,
                "gpu_label_fp32_batch_upper_bytes": self.config.wm_batch * self.config.wm_window * pixels * 4,
                "control_replay_capacity": self.config.replay_capacity,
                "archive_window_capacity": self.replay.archive_capacity,
                "num_envs": self.env.num_envs, "state_precision": "FP32", "perception_precision": "FP32",
                "mlp_bf16": {"world_model": self.execution.wm_bf16, "actor": self.execution.actor_bf16,
                             "critic": self.execution.critic_bf16},
                "compile": {"world_model": self.execution.wm_compile, "imagination": self.config.compile_imagination},
                "cuda_graph": self.execution.cuda_graph,
                "scope": "CPU raw pixel capacity and individual FP32 GPU image batches only; excludes transfer conversion buffers, CPU metadata, model, optimizers, physical replay, rendering, CNN/BPTT activations and CUDA graph pools"}

    @torch.no_grad()
    def reset_collection(self):
        self.current = self.env.reset()
        frame = self.task.observe(self.current, noisy=True)
        self.history = frame[:, None].repeat(1, self.spec.history_length, 1)
        self.replay.clear()
        self.hidden = torch.zeros(self.env.num_envs, self.networks.hidden_dim, device=self.device)
        self.hidden, self.z = self.actor.encode(self.env.depth.noisy, self.hidden, self.env.depth.new_frame)

    @torch.no_grad()
    def collect(self, steps):
        if self.current is None:
            self.reset_collection()
        else:
            # 参数更新后以完整活跃前缀重建，采集态与当前 CNN/GRU 权重一致。
            seed = self.replay.current_seed(self.current, self.history, self.env.depth)
            self.hidden, self.z = self.actor.rebuild(seed)
            del seed
        reward = torch.zeros((), device=self.device)
        failures = torch.zeros((), device=self.device)
        timeouts = torch.zeros((), device=self.device)
        reset_steps = torch.zeros((), dtype=torch.long, device=self.device)
        totals = {}
        for _ in range(steps):
            before_depth = self.env.depth
            action = self.actor(self.history, self.z)
            standing = self.current.command[:, :2].norm(dim=-1) + self.current.command[:, 2].abs()
            quiet = self.collection_mask
            if self.env.cfg.runner.standing_zero_noise:
                quiet = quiet | (standing <= self.env.cfg.reward.stand_still_command_threshold)
            action = action + self.config.action_noise * torch.randn_like(action) * ~quiet[:, None]
            result = self.env.step(action)
            done = result.terminated | result.truncated
            reset_steps += done.any()
            self.actor.update_statistics(self.history)
            self.replay.add(self.current, self.history, action, result, before_depth, self.env.depth)
            frame = self.task.observe(result.next_state, noisy=True)
            history = append_history(self.history, frame)
            self.history = torch.where(done[:, None, None], frame[:, None].expand_as(history), history)
            self.hidden = torch.where(done[:, None], 0., self.hidden)
            self.hidden, self.z = self.actor.encode(self.env.depth.noisy, self.hidden, self.env.depth.new_frame)
            self.current = result.next_state
            reward += result.reward.mean()
            failures += result.terminated.sum()
            timeouts += (result.truncated & ~result.terminated).sum()
            for name, value in self.env.last_metrics.items():
                if name.startswith("reward/"):
                    term = name.removeprefix("reward/")
                    scale = getattr(self.env.cfg.reward.scales, term)
                    totals[f"Reward_Rate/{term}"] = totals.get(f"Reward_Rate/{term}", 0) + scale * value
                elif not name.startswith("episode_reward/"):
                    totals[f"task/{name}"] = totals.get(f"task/{name}", 0) + value
            self.real_steps += self.env.num_envs
        return {"real_reward": reward / steps, "failures": failures, "timeouts": timeouts,
                "reset_steps": reset_steps, "collection/control_steps": steps,
                **{name: value / steps for name, value in totals.items()}}

    def _collapse_guard(self, metrics):
        """沿用Go2的失败数／发生重置的控制步口径，连续两窗异常后恢复安全策略。"""
        self._guard_falls += metrics["failures"]
        self._guard_reset_steps += metrics["reset_steps"]
        if self.iteration % self.config.collapse_interval:
            return
        falls = self._guard_falls / max(self._guard_reset_steps, 1)
        self._guard_falls, self._guard_reset_steps = 0.0, 0
        metrics["guard/falls_per_reset_step"] = falls
        if self.iteration <= self.learner.actor_frozen_until or falls <= self.config.collapse_falls:
            self._collapse_strikes = 0
            return
        self._collapse_strikes += 1
        if self._collapse_strikes < 2 or self._safe_policy is None:
            return
        for name in ("actor", "decoder", "critic"):
            getattr(self, name).load_state_dict(self._safe_policy[name])
        self.learner.reset_policy_optimizers()
        self.learner.lr_multiplier *= .5
        self.learner.ramp_start = self.iteration
        self._collapse_strikes = 0
        metrics["guard/rollback"] = 1.0
        print(f"[WARN] Collapse guard: {falls:.1f} falls per reset step, restored actor, decoder and critic; "
              f"lr scale {self.learner.lr_multiplier:.3g}", flush=True)

    def _policy_snapshot(self):
        """安全快照包含共享感知对应的解码器，保存在CPU并与后续更新隔离。"""
        return {name: {key: value.detach().cpu().clone() for key, value in getattr(self, name).state_dict().items()}
                for name in ("actor", "decoder", "critic")}

    def learn(self, iterations):
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError("Training iterations must be a positive integer")
        records = []
        while not self.replay.ready(self.config.wm_window):
            prefill = self.collect(self.config.wm_window + 2)
            counts = read_metrics({name: prefill[name] for name in ("failures", "reset_steps")}, self.device)
            self._guard_falls += counts["failures"]
            self._guard_reset_steps += counts["reset_steps"]
        for local in range(iterations):
            started = time.perf_counter()
            self.iteration += 1
            self.collect_accumulator += self.config.collect_steps
            steps = int(self.collect_accumulator)
            self.collect_accumulator -= steps
            metrics = self.collect(steps)
            collection_seconds = time.perf_counter() - started
            self.learner.schedule(self.iteration)
            model_metrics = {}
            model_started = time.perf_counter()
            for _ in range(self.config.wm_updates):
                window = self.replay.model_batch(self.task, self.config)
                for name, value in self.learner.fit_model(window).items():
                    if name.startswith("cuda_graph/"):
                        model_metrics[name] = value
                    else:
                        model_metrics[name] = model_metrics.get(name, 0) + value / self.config.wm_updates
                del window
            metrics.update(model_metrics)
            model_seconds = time.perf_counter() - model_started
            control_started = time.perf_counter()
            enabled = self.control_enabled and self.iteration > self.config.warmup_iterations
            if enabled:
                seed = self.replay.sample_seeds(self.config.imag_batch)
                metrics.update(self.learner.improve(seed))
                del seed
                self.imagined_steps += self.config.imag_batch * self.config.horizon
            seconds = time.perf_counter() - started
            self.elapsed += seconds
            metrics.update(iteration=float(self.iteration), real_steps=float(self.real_steps),
                           imagined_steps=float(self.imagined_steps), control_enabled=float(enabled),
                           training_target_iteration=float(self.target_iteration or self.iteration),
                           **{"timing/collection_seconds": collection_seconds,
                              "timing/world_model_seconds": model_seconds,
                              "timing/control_seconds": time.perf_counter() - control_started,
                              "timing/iteration_seconds": seconds, "timing/session_elapsed_seconds": self.elapsed,
                              "performance/real_total_fps": steps * self.env.num_envs / seconds,
                              "replay/image_frames": float(self.replay.image_count),
                              "replay/archive_windows": float(len(self.replay.archive))})
            self.metrics.add(metrics)
            if (self.iteration == 1 or self.iteration % self.metric_interval == 0
                    or self.iteration % self.config.collapse_interval == 0 or local == iterations - 1):
                aggregate, validity = self.metrics.aggregate()
                values = read_metrics(aggregate, self.device, validity=validity)
                values.update(iteration=self.iteration, real_steps=self.real_steps,
                              imagined_steps=self.imagined_steps,
                              training_target_iteration=self.target_iteration or self.iteration)
                self._collapse_guard(values)
                self.logger.record(values)
                records.append(values)
                self.metrics.clear()
        return records

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_policy = self._policy_snapshot() if self._collapse_strikes == 0 else self._safe_policy
        payload = {"format": "depth_cnn_gru_v1", "spec": asdict(self.spec), "network_config": asdict(self.networks),
                   "config": asdict(self.config), "iteration": self.iteration, "real_steps": self.real_steps,
                   "imagined_steps": self.imagined_steps, "learner": self.learner.state_dict(),
                   "curriculum": self.env.curriculum_state_dict(),
                   "guard": {"safe_policy": safe_policy, "strikes": self._collapse_strikes,
                             "falls": self._guard_falls, "reset_steps": self._guard_reset_steps},
                   **{name: getattr(self, name).state_dict() for name in ("actor", "decoder", "model", "critic")}}
        pending = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, pending)
        export_depth_policy(self.actor, path.parent / "policy.pt", metadata=self.deployment_metadata())
        os.replace(pending, path)
        self._safe_policy = safe_policy

    def load(self, path, *, continue_run=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("format") != "depth_cnn_gru_v1" or checkpoint.get("network_config") != asdict(self.networks):
            raise ValueError("Depth checkpoint format or network dimensions differ")
        if checkpoint.get("spec") != asdict(self.spec):
            raise ValueError("Depth checkpoint task, robot or camera semantics differ")
        if continue_run:
            runtime_fields = {"compile_imagination", "compile_backend"}
            saved_algorithm = {name: value for name, value in checkpoint["config"].items() if name not in runtime_fields}
            current_algorithm = {name: value for name, value in asdict(self.config).items() if name not in runtime_fields}
            if saved_algorithm != current_algorithm:
                raise ValueError("Continue-run requires the original algorithm configuration")
        states = [(name, checkpoint[name]) for name in ("actor", "decoder", "model", "critic")]
        if continue_run and checkpoint["guard"]["safe_policy"] is not None:
            states.extend(checkpoint["guard"]["safe_policy"].items())
        for name, state in states:
            expected = getattr(self, name).state_dict()
            if state.keys() != expected.keys() or any(state[key].shape != expected[key].shape for key in expected):
                raise ValueError(f"Checkpoint {name} tensor names or shapes differ")
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError(f"Checkpoint {name} contains non-finite values")
        for name in ("actor", "decoder", "model", "critic"):
            getattr(self, name).load_state_dict(checkpoint[name])
        if continue_run:
            self.learner.load_state_dict(checkpoint["learner"])
            self.iteration, self.real_steps = checkpoint["iteration"], checkpoint["real_steps"]
            self.imagined_steps = checkpoint["imagined_steps"]
            self.env.load_curriculum_state_dict(checkpoint["curriculum"])
            self.env.resume_progress(self.iteration)
            guard = checkpoint["guard"]
            self._safe_policy = ({name: {key: value.detach().cpu().clone() for key, value in state.items()}
                                  for name, state in guard["safe_policy"].items()}
                                 if guard["safe_policy"] is not None else None)
            self._collapse_strikes, self._guard_falls = guard["strikes"], guard["falls"]
            self._guard_reset_steps = guard["reset_steps"]
        else:
            self._safe_policy = None
            self._collapse_strikes = 0
            self._guard_falls, self._guard_reset_steps = 0.0, 0
        self.current = self.history = None
        self.replay.clear()
        return checkpoint

    def deployment_metadata(self):
        model = self.env.raw_env.sim.mj_model
        camera = model.camera(f"robot/{self.env.cfg.sensors.depth.camera_name}")
        camera_id = camera.id
        body_id = int(camera.bodyid[0])
        chain = []
        while body_id:
            body = model.body(body_id)
            chain.append({"name": body.name, "position": body.pos.tolist(), "quaternion_wxyz": body.quat.tolist()})
            body_id = int(body.parentid[0])
        return {"model_id": self.spec.model_id, "iteration": self.iteration, "network": asdict(self.networks),
                "history_order": "frame_major_oldest_first", "hidden_reset": "zeros_before_first_new_frame",
                "camera": self.env.cfg.to_dict()["sensors"]["depth"],
                "camera_calibration": {"position_in_parent": camera.pos.tolist(),
                                       "quaternion_in_parent_wxyz": camera.quat.tolist(), "parent_chain": chain,
                                       "fovy_degrees": float(model.cam_fovy[camera_id]),
                                       "intrinsic": model.cam_intrinsic[camera_id].tolist(),
                                       "sensor_size": model.cam_sensorsize[camera_id].tolist(),
                                       "resolution": model.cam_resolution[camera_id].tolist()},
                "joint_order": list(self.env.robot.joint_names), "step_dt": self.env.step_dt,
                "default_joint_pos": self.task.default_joint_pos.cpu().tolist(),
                "action_scales": self.env.cfg.control.action_scales,
                "control": self.env.cfg.to_dict()["control"]}

    def close(self):
        self.logger.close()
