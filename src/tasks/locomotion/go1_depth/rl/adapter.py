# Copyright (c) 2026-2027 zh
"""内容：从MjLab Go1采集同步深度与重置前转移，提供视觉SuperMimic任务协议。"""

import math

import torch
from mjlab.envs import ManagerBasedRlEnv

from agents.perceptive_supermimic.interfaces import Decision, EnvStep, ModelPrediction, TaskState, VisualObservation
from src.assets.robots.go1_depth.go1_depth_constants import JOINT_NAMES
from src.config.go1_depth.go1_depth_config import Go1LocomotionConfig

from ..env_cfg import make_env_cfg
from ..mdp.depth import process_depth
from ..mdp.task import Go1DepthTask
from ..mdp.terrain import Go1Terrain


class Go1DepthEnv:
    """每控制步对应一张新深度；先复制final，再重置结束行并构造next。"""

    def __init__(self, cfg=None, device="cuda:0", latent_dim=32, *, render_mode=None):
        self.cfg = cfg or Go1LocomotionConfig()
        self.raw_env = ManagerBasedRlEnv(make_env_cfg(self.cfg), device=device, render_mode=render_mode)
        try:
            self.robot = self.raw_env.scene["robot"]
            self.device = self.robot.data.joint_pos.device
            self.num_envs = self.cfg.num_envs
            action_term = self.raw_env.action_manager.get_term("joint_pos")
            if tuple(self.robot.joint_names) != JOINT_NAMES or tuple(action_term.target_names) != JOINT_NAMES:
                raise ValueError("Go1 joint and action order must be FL,FR,RL,RR with hip,thigh,calf")
            self.surface = Go1Terrain(self.raw_env.sim.mj_model, self.cfg, self.device)
            self.task = Go1DepthTask(self.cfg, self.robot.data.default_joint_pos[0], self.robot.data.soft_joint_pos_limits[0],
                                   self.surface, device=self.device, latent_dim=latent_dim)
            self.episodes = torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device)
            self.frames = torch.zeros_like(self.episodes)
            self.steps = torch.zeros_like(self.episodes)
            self.context_versions = torch.zeros_like(self.episodes)
            self.command = torch.zeros(self.num_envs, 3, device=self.device)
            self.previous_action = torch.zeros(self.num_envs, 12, device=self.device)
            self.max_steps = math.ceil(self.cfg.episode_length_s / self.task.dt)
            self.current = None
            self.last_metrics = {}
        except BaseException:
            self.raw_env.close()
            raise

    def _physical(self):
        data = self.robot.data
        return torch.cat((data.root_link_pos_w, data.root_link_quat_w, data.root_link_lin_vel_w,
                          data.root_link_ang_vel_b, data.joint_pos, data.joint_vel), -1).clone()

    def _decision(self):
        state = TaskState(self._physical(), self.previous_action.clone(), self.command.clone(), self.raw_env.scene.env_origins.clone())
        noisy, clean = process_depth(self.raw_env.scene["depth"].data.depth.clone(), self.cfg)
        observation = VisualObservation(self.task.observe(state, noisy=True), noisy, clean, self.episodes.clone(), self.frames.clone())
        return Decision(state, observation, self.context_versions.clone())

    def _reset_rows(self, ids):
        self.raw_env.reset(env_ids=ids)
        self.episodes[ids] += 1
        self.context_versions[ids] += 1
        self.frames[ids] = 0
        self.steps[ids] = 0
        self.previous_action[ids] = 0
        bounds = self.command.new_tensor((self.cfg.command_forward, self.cfg.command_lateral, self.cfg.command_yaw))
        self.command[ids] = bounds[:, 0] + torch.rand(ids.numel(), 3, device=self.device) * (bounds[:, 1] - bounds[:, 0])
        standing = torch.rand(ids.numel(), device=self.device) < self.cfg.standing_probability
        self.command[ids[standing]] = 0

    @torch.no_grad()
    def reset(self):
        self._reset_rows(torch.arange(self.num_envs, device=self.device))
        self.current = self._decision()
        self.last_metrics = {}
        return self.current.detached_clone()

    @torch.no_grad()
    def step(self, action, *, auto_reset=True):
        if self.current is None:
            raise RuntimeError("Reset Go1DepthEnv before stepping")
        if action.shape != (self.num_envs, 12) or action.dtype != torch.float32 or action.device != self.device:
            raise ValueError("Go1 action must be float32 [num_envs,12] on the environment device")
        if not torch.isfinite(action).all() or (action.abs() > self.cfg.action_clip).any():
            raise ValueError("Go1 action must be finite and within the configured clip")
        before = self.current.state
        self.raw_env.step(action)
        self.previous_action.copy_(action)
        self.frames += 1
        self.steps += 1
        final = self._decision()
        if not torch.isfinite(final.state.physical).all():
            raise FloatingPointError("Go1 simulator produced non-finite state")
        terminated = self.task.hard_termination(final.state)
        truncated = self.steps >= self.max_steps
        contacts = self.raw_env.scene["feet_contact"].data.found.reshape(self.num_envs, 4, -1).any(-1).float().clone()
        prediction = ModelPrediction(final.state, action.new_empty(self.num_envs, 0), action.new_zeros(self.num_envs, 4))
        reward = self.task.reward(before, action, prediction, terminated.float())
        terms = self.task.reward_terms(before, action, final.state, terminated.float())
        self.last_metrics = {f"reward/{name}": (self.task.dt * getattr(self.cfg, f"{name}_weight") * value).mean().item()
                             for name, value in terms.items()}
        velocity = self.task.body_linear_velocity(final.state)
        self.last_metrics.update(tracking_linear_rmse=(velocity[:, :2] - before.command[:, :2]).square().mean().sqrt().item(),
                                 tracking_yaw_rmse=(final.state.physical[:, 12] - before.command[:, 2]).square().mean().sqrt().item(),
                                 base_height=self.task.base_height(final.state).mean().item(),
                                 action_saturation=(action.abs() >= self.cfg.action_clip).float().mean().item(),
                                 failure_rate=terminated.float().mean().item())
        done = terminated | truncated
        if auto_reset and done.any():
            ids = done.nonzero().flatten()
            if self.cfg.terrain == "vision":
                terrain = self.raw_env.scene.terrain
                distance = (self.task.world_position(final.state)[ids, :2] - final.state.context[ids, :2]).norm(dim=-1)
                levels = terrain.terrain_levels[ids] + (distance > 2.0).long() - (terminated[ids] & (distance < 1.0)).long()
                terrain.terrain_levels[ids] = levels.clamp(0, self.cfg.terrain_rows - 1)
                terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
            self._reset_rows(ids)
            reset = self._decision()
            # 继续行保留final的原图及噪声，不能在部分reset后重新抽一份观测噪声。
            next_value = final.detached_clone()
            for name in ("physical", "memory", "command", "context"):
                getattr(next_value.state, name)[ids] = getattr(reset.state, name)[ids]
            for name in ("proprio", "depth", "depth_clean", "episode_id", "frame_id"):
                getattr(next_value.observation, name)[ids] = getattr(reset.observation, name)[ids]
            next_value.context_version[ids] = reset.context_version[ids]
            self.current = next_value
        else:
            self.current = final.detached_clone()
        return EnvStep(final.detached_clone(), self.current.detached_clone(), reward.detach().clone(),
                       terminated.clone(), truncated.clone(), contacts)

    def close(self):
        self.raw_env.close()
