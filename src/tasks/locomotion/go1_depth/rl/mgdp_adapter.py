# Copyright (c) 2026-2027 zh
"""内容：按 MGDP 两阶段协议采集 Go1 真实转移、保持视觉并执行命令/地形课程。"""

from __future__ import annotations

from dataclasses import fields
import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply_inverse

from agents.perceptive_supermimic.interfaces import Decision, EnvStep, TaskState, VisualObservation
from src.assets.robots.go1_depth.go1_mgdp_constants import JOINT_NAMES, JOINT_VELOCITY_LIMIT
from src.config.go1_depth.go1_mgdp_config import Go1MGDPConfig
from src.sim.mjwarp_implicit import configure_mjwarp_implicit
from src.terrain.mgdp import MgdpTerrainQuery, generate_mgdp_terrain

from ..mgdp_env_cfg import make_mgdp_env_cfg
from ..mdp import mgdp_state as slots
from ..mdp.mgdp_depth import capture_height_label, process_mgdp_depth
from ..mdp.mgdp_randomization import apply_mgdp_push
from ..mdp.mgdp_task import MGDPTask


class MGDPGo1DepthEnv:
    """50Hz 物理推进与全局 10Hz 相机分离，先保存 final 再选择性重置。"""

    def __init__(self, cfg=None, device="cuda:0", latent_dim=64, *, render_mode=None):
        self.cfg = cfg or Go1MGDPConfig()
        if self.cfg.physics_integrator == "implicit":
            configure_mjwarp_implicit()
        terrain_data = generate_mgdp_terrain(self.cfg.seed, family=self.cfg.family)
        self.raw_env = ManagerBasedRlEnv(make_mgdp_env_cfg(self.cfg, terrain_data), device=device, render_mode=render_mode)
        try:
            self.robot = self.raw_env.scene["robot"]
            self.device, self.num_envs = self.robot.data.joint_pos.device, self.cfg.num_envs
            self.action_term = self.raw_env.action_manager.get_term("joint_pos")
            self.joint_ids = torch.tensor([self.robot.joint_names.index(name) for name in JOINT_NAMES], device=self.device)
            self.action_order = torch.tensor([JOINT_NAMES.index(name) for name in self.action_term.target_names], device=self.device)
            self.torque_order = torch.tensor([self.action_term.target_names.index(name) for name in JOINT_NAMES], device=self.device)
            self.surface = MgdpTerrainQuery(terrain_data, self.device)
            self.task = MGDPTask(self.cfg, self.robot.data.default_joint_pos[0, self.joint_ids],
                                 self.robot.data.soft_joint_pos_limits[0, self.joint_ids], self.surface,
                                 device=self.device, latent_dim=latent_dim)
            self.terrain_data = terrain_data
            family = terrain_data.families[0]
            self.terrain_origins = torch.as_tensor(family.origins, device=self.device, dtype=torch.float32)
            self.terrain_classes = torch.as_tensor(family.classes, device=self.device)
            self.max_level = family.cfg.num_rows
            self.terrain_length = family.cfg.terrain_length
            self.terrain_levels = torch.randint(0, 6, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), self.num_envs / 10, rounding_mode="floor").long()
            self.env_origins = self.terrain_origins[self.terrain_levels, self.terrain_types].clone()
            self.env_class = self.terrain_classes[self.terrain_levels, self.terrain_types].clone()
            self._sync_scene_origins()
            self.episodes = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            self.frames, self.steps = torch.zeros_like(self.episodes), torch.zeros_like(self.episodes)
            self.camera_frames, self.context_versions = torch.zeros_like(self.episodes), torch.zeros_like(self.episodes)
            self.global_control_step = 0
            self.memory = torch.zeros(self.num_envs, slots.MEMORY_DIM, device=self.device)
            self.previous_action = self.memory[:, slots.PREVIOUS_ACTION]
            self.command = torch.zeros(self.num_envs, 3, device=self.device)
            self.target_heading = torch.zeros(self.num_envs, device=self.device)
            self.foot_forces = torch.zeros(self.num_envs, 4, 3, device=self.device)
            self.body_forces = torch.zeros(self.num_envs, 9, 3, device=self.device)
            self.reward_linear_body = torch.zeros(self.num_envs, 3, device=self.device)
            self.depth = torch.zeros(self.num_envs, 2, 16, 16, device=self.device)
            self.depth_clean = torch.zeros_like(self.depth)
            self.image_height = torch.zeros(self.num_envs, 1, 17, 11, device=self.device)
            self.max_steps = math.ceil(self.cfg.episode_length_s / self.task.dt)
            self.command_interval = int(10.0 / self.task.dt)
            self.command_forward_range = [-1.0, 1.0]
            self.tracking_episode_sums = torch.zeros(self.num_envs, device=self.device)
            self.current = None
            self.last_metrics = {}
        except BaseException:
            self.raw_env.close()
            raise

    def _sync_scene_origins(self):
        terrain = self.raw_env.scene.terrain
        terrain.terrain_levels.copy_(self.terrain_levels)
        terrain.terrain_types.copy_(self.terrain_types)
        terrain.env_origins.copy_(self.env_origins)

    def _read_forces(self, ids):
        # MuJoCo contact sensor 返回 primary 对 secondary 的力；MGDP 需要其反作用力。
        self.foot_forces[ids] = -self.raw_env.scene["feet_contact"].data.force[ids].reshape(-1, 4, 3)
        base = -self.raw_env.scene["base_contact"].data.force[ids].reshape(-1, 1, 3)
        legs = -self.raw_env.scene["leg_contact"].data.force[ids].reshape(-1, 8, 3)
        self.body_forces[ids] = torch.cat((base, legs), 1)

    def _physical(self):
        data = self.robot.data
        return torch.cat((data.root_link_pos_w, data.root_link_quat_w, data.root_link_lin_vel_w, data.root_link_ang_vel_b,
                          data.joint_pos[:, self.joint_ids], data.joint_vel[:, self.joint_ids],
                          self.action_term.last_torques[:, self.torque_order], self.foot_forces.flatten(1), self.body_forces.flatten(1),
                          self.reward_linear_body), -1).clone()

    def _state(self):
        context = torch.cat((self.env_origins, self.command.new_full((self.num_envs, 1), self.cfg.stage - 1),
                             self.env_class[:, None].float(), self.terrain_levels[:, None].float(),
                             self.command.new_full((self.num_envs, 1), self.global_control_step),
                             self.raw_env.mgdp_dynamics.context(), self.target_heading[:, None]), -1)
        return TaskState(self._physical(), self.memory.clone(), self.command.clone(), context)

    def _capture(self, ids, *, reset=False):
        self.raw_env.sim.sense()
        noisy, clean = process_mgdp_depth(self.raw_env.scene["depth"].data.depth[ids], noisy=self.cfg.depth_noise)
        if reset:
            self.depth[ids] = noisy.expand(-1, 2, -1, -1)
            self.depth_clean[ids] = clean.expand(-1, 2, -1, -1)
            self.camera_frames[ids] = 0
        else:
            self.depth[ids] = torch.cat((self.depth[ids, 1:], noisy), 1)
            self.depth_clean[ids] = torch.cat((self.depth_clean[ids, 1:], clean), 1)
            self.camera_frames[ids] += 1
        self.image_height[ids] = capture_height_label(self.task.terrain_height(self._state().index(ids)), noisy=self.cfg.height_noise)

    def _decision(self, ids=None):
        ids = slice(None) if ids is None else ids
        state = self._state().index(ids)
        observation = VisualObservation(self.task.observe(state, noisy=True), self.depth[ids].clone(), self.depth_clean[ids].clone(),
                                        self.episodes[ids].clone(), self.frames[ids].clone(), self.camera_frames[ids].clone(),
                                        self.image_height[ids].clone())
        return Decision(state, observation, self.context_versions[ids].clone())

    def _resample_commands(self, ids):
        easy_limit = 4 if self.cfg.stage == 1 else 2
        easy = self.env_class[ids] <= easy_limit
        for selected, hard in ((ids[easy], False), (ids[~easy], True)):
            count = selected.numel()
            if count == 0:
                continue
            if hard and self.cfg.stage == 2:
                self.command[selected, 0] = torch.rand(count, device=self.device) * 1.5
                # 原版零宽范围仍调用随机采样；保持相同的随机调用职责。
                self.command[selected, 1] = torch.rand(count, device=self.device) * 0
                self.target_heading[selected] = torch.rand(count, device=self.device) * 0
            else:
                low, high = self.command_forward_range if not hard else (-1.0, 1.0)
                self.command[selected, 0] = low + (high - low) * torch.rand(count, device=self.device)
                self.command[selected, 1] = 2 * torch.rand(count, device=self.device) - 1
                self.target_heading[selected] = 6.28 * torch.rand(count, device=self.device) - 3.14
        self.command[ids, :2] *= (torch.linalg.vector_norm(self.command[ids, :2], dim=-1) > 0.2)[:, None]

    def _update_heading(self, ids=None):
        ids = slice(None) if ids is None else ids
        error = self.target_heading[ids] - self.task.heading(self._state().index(ids))
        wrapped = torch.remainder(error, 2 * torch.pi)
        wrapped = torch.where(wrapped > torch.pi, wrapped - 2 * torch.pi, wrapped)
        self.command[ids, 2] = (0.5 * wrapped).clamp(-1, 1)

    def _update_curriculum(self, ids, final_state):
        relative = self.task.world_position(final_state)[ids, :2] - self.env_origins[ids, :2]
        if self.cfg.stage == 2:
            relative = relative - relative.new_tensor((-4.0, 0.0))
        distance = torch.linalg.vector_norm(relative, dim=-1)
        advance = distance > self.terrain_length / 2
        retreat = (distance < torch.linalg.vector_norm(final_state.command[ids, :2], dim=-1) * self.cfg.episode_length_s * 0.5) & ~advance
        levels = self.terrain_levels[ids] + advance.long() - retreat.long()
        levels = torch.where(levels >= self.max_level, torch.randint_like(levels, high=self.max_level), levels.clamp_min(0))
        self.terrain_levels[ids] = levels
        self.env_origins[ids] = self.terrain_origins[levels, self.terrain_types[ids]]
        self.env_class[ids] = self.terrain_classes[levels, self.terrain_types[ids]]
        self._sync_scene_origins()
        if self.global_control_step % self.max_steps == 0:
            if self.tracking_episode_sums[ids].mean() / self.max_steps > 0.8 * self.task.dt:
                self.command_forward_range[0] = max(-1.0, min(0.0, self.command_forward_range[0] - 0.5))
                self.command_forward_range[1] = min(1.0, max(0.0, self.command_forward_range[1] + 0.5))

    def _reset_rows(self, ids, *, final_state=None):
        if final_state is not None:
            self._update_curriculum(ids, final_state)
        self.raw_env.extras["log"] = {}
        self.raw_env._reset_idx(ids)
        root = torch.zeros(ids.numel(), 13, device=self.device)
        root[:, 3], root[:, 2] = 1, 0.45
        root[:, :2] += self.env_origins[ids, :2]
        classes = self.env_class[ids]
        if self.cfg.stage == 2:
            root[:, 0] -= 4
            varied = classes != 17
            root[varied, :2] += 0.4 * torch.rand(int(varied.sum()), 2, device=self.device) - 0.2
        else:
            root[:, 2] += self.env_origins[ids, 2]
            hurdle, gap = classes == 5, classes == 6
            other = ~hurdle & ~gap
            root[hurdle, 0] -= 2 + torch.rand(int(hurdle.sum()), device=self.device)
            root[gap, 0] -= 0.4 * torch.rand(int(gap.sum()), device=self.device)
            root[other, :2] += torch.rand(int(other.sum()), 2, device=self.device) - 0.5
        joints = self.task.nominal * (0.5 + torch.rand(ids.numel(), 12, device=self.device))
        self.robot.write_root_state_to_sim(root, env_ids=ids)
        self.robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), joint_ids=self.joint_ids, env_ids=ids)
        self.raw_env.scene.write_data_to_sim()
        self.raw_env.sim.forward()
        self.raw_env.scene.update(dt=0.0)
        self._read_forces(ids)
        self.reward_linear_body[ids] = quat_apply_inverse(self.robot.data.root_link_quat_w[ids], self.robot.data.root_link_lin_vel_w[ids])
        self.episodes[ids] += 1
        self.context_versions[ids] += 1
        self.frames[ids], self.steps[ids] = 0, 0
        self.memory[ids, slots.PREVIOUS_ACTION] = 0
        self.memory[ids, slots.AIR_TIME] = 0
        self.memory[ids, slots.CAMERA_PHASE] = self.global_control_step % 5
        self.tracking_episode_sums[ids] = 0
        self._resample_commands(ids)
        self._update_heading(ids)
        self._capture(ids, reset=True)

    @torch.no_grad()
    def reset(self):
        ids = torch.arange(self.num_envs, device=self.device)
        self._reset_rows(ids, final_state=None if self.current is None else self.current.state)
        self.current = self._decision()
        self.last_metrics = {}
        return self.current.detached_clone()

    def _step_physics(self, action):
        raw = self.raw_env
        raw.extras["log"] = {}
        raw.action_manager.process_action(action[:, self.action_order])
        for _ in range(self.cfg.decimation):
            raw._sim_step_counter += 1
            raw.action_manager.apply_action()
            raw.scene.write_data_to_sim()
            # 获准的 PhysX 限速迁移：PD 先读取步末速度，再限积分初速；不直接改根速度。
            self.robot.write_joint_velocity_to_sim(self.robot.data.joint_vel.clamp(-JOINT_VELOCITY_LIMIT, JOINT_VELOCITY_LIMIT))
            raw.sim.step()
            raw.scene.update(dt=self.cfg.timestep)
        raw.episode_length_buf += 1
        raw.common_step_counter += 1
        raw.sim.forward()
        self._read_forces(slice(None))

    @torch.no_grad()
    def step(self, action, *, auto_reset=True):
        if self.current is None:
            raise RuntimeError("Reset MGDPGo1DepthEnv before stepping.")
        if action.shape != (self.num_envs, 12) or action.dtype != torch.float32 or action.device != self.device:
            raise ValueError("MGDP action must be float32 [num_envs,12] on the environment device.")
        if not torch.isfinite(action).all() or (action.abs() > self.cfg.action_clip).any():
            raise ValueError("MGDP action must be finite and within the configured clip.")
        before = self.current.state
        self._step_physics(action)
        # 该官方缓存必须进入 final，使真实奖励可以由保存的状态完整重算。
        self.reward_linear_body.copy_(quat_apply_inverse(self.robot.data.root_link_quat_w, self.robot.data.root_link_lin_vel_w))
        self.global_control_step += 1
        self.frames += 1
        self.steps += 1
        # 先保存物理步末接触力；速度替换后 forward 只刷新根状态，不覆盖这次接触读数。
        if apply_mgdp_push(self.raw_env, self.global_control_step) is not None:
            self.raw_env.sim.forward()
        resample = (self.steps % self.command_interval == 0).nonzero().flatten()
        self._resample_commands(resample)
        self._update_heading()
        self.memory[:, slots.PREVIOUS_ACTION] = action
        self.memory[:, slots.CAMERA_PHASE] = self.global_control_step % 5
        provisional = self._state()
        reward, terms, memory = self.task.reward_values(before, action, provisional)
        self.memory[:, slots.AIR_TIME] = memory.air_time
        self.memory[:, slots.LAST_CONTACT] = memory.last_contact
        self.tracking_episode_sums += terms["tracking_lin_vel"]
        if self.global_control_step % 5 == 0:
            self._capture(torch.arange(self.num_envs, device=self.device))
        final = self._decision()
        if not torch.isfinite(final.state.physical).all():
            raise FloatingPointError("MGDP simulator produced non-finite physical state.")
        terminated = self.task.hard_termination(final.state)
        truncated = self.steps > self.max_steps
        contacts = (self.foot_forces[..., 2] > 1).float().clone()
        velocity = self.task.body_linear_velocity(final.state)
        self.last_metrics = {f"reward/{name}": value.mean().item() for name, value in terms.items()}
        self.last_metrics.update(
            reward_total=reward.mean().item(), tracking_linear_rmse=(velocity[:, :2] - final.state.command[:, :2]).square().mean().sqrt().item(),
            tracking_yaw_rmse=(final.state.physical[:, 12] - final.state.command[:, 2]).square().mean().sqrt().item(),
            failure_rate=terminated.float().mean().item(), timeout_rate=truncated.float().mean().item(),
            base_height=self.task.base_height(final.state).mean().item(), terrain_level=self.terrain_levels.float().mean().item(),
            action_saturation=(action.abs() >= self.cfg.action_clip).float().mean().item(),
            command_forward_min=self.command_forward_range[0], command_forward_max=self.command_forward_range[1],
        )
        done = terminated | truncated
        self.current = final.detached_clone()
        if auto_reset and done.any():
            ids = done.nonzero().flatten()
            self._reset_rows(ids, final_state=final.state)
            reset = self._decision(ids)
            # 仅替换重置行；继续行的观测噪声、图像、标签与 final 逐位相同。
            for field in fields(TaskState):
                getattr(self.current.state, field.name)[ids] = getattr(reset.state, field.name)
            for field in fields(VisualObservation):
                getattr(self.current.observation, field.name)[ids] = getattr(reset.observation, field.name)
            self.current.context_version[ids] = reset.context_version
        return EnvStep(final, self.current.detached_clone(), reward.clone(), terminated.clone(), truncated.clone(), contacts)

    def close(self):
        self.raw_env.close()
