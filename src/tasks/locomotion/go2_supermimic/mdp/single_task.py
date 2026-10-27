# Copyright (c) 2026-2027 zh
"""
内容：
    为单一盲MoE策略提供解析任务、可微动力学残差和地形特权接口。
    策略观测不含真实线速度、地形扫描、接触标签或教师表征。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import torch
from torch import Tensor

from agents.supermimic.interfaces import AlgorithmSpec, TaskState

from .kinematics import foot_positions_world
from .single_rewards import single_reward_terms
from .state_spec import (
    ANGULAR_VELOCITY, COMMAND_TIMER, CONTEXT_ORIGIN, CONTEXT_RANGES, CONTEXT_ZERO_PROBABILITY,
    JOINT_POSITION, JOINT_VELOCITY, LINEAR_VELOCITY, PHASE_CLOCK, POSITION, PREVIOUS_ACTION,
    QUATERNION, heading, projected_gravity, quat_apply, quat_apply_inverse, quat_conjugate,
    quat_from_rotvec, quat_multiply, quat_normalize, quat_to_rotvec, rotation_features,
)


class Go2SingleTaskModel:
    """模型和critic可用地形特权，执行actor仅接收45维本体帧。"""

    model_input_dim = 151

    def __init__(self, cfg, default_joint_pos: Tensor, joint_limits: Tensor, surface, device="cpu"):
        self.cfg = deepcopy(cfg)
        self.device = torch.device(device)
        self.surface = surface
        self.dt = self.cfg.sim.timestep * self.cfg.control.decimation
        self.default_joint_pos = torch.as_tensor(default_joint_pos, dtype=torch.float32, device=self.device).clone()
        self.joint_limits = torch.as_tensor(joint_limits, dtype=torch.float32, device=self.device).clone()
        self._validate_config()
        control = self.cfg.control
        names = ("hip", "thigh", "calf") * 4
        self.kp = self.default_joint_pos.new_tensor([control.stiffness[name] for name in names])
        self.kd = self.default_joint_pos.new_tensor([control.damping[name] for name in names])
        self.effort_limit = self.default_joint_pos.new_tensor([control.effort_limit[name] for name in names])
        self.action_scale = self.default_joint_pos.new_tensor([
            control.action_scale * (control.hip_reduction if name == "hip" else 1.0) for name in names])
        half_x, half_y = (value / 2 for value in self.cfg.terrain.scan_size_m)
        resolution = self.cfg.terrain.scan_resolution_m
        xs = torch.linspace(-half_x, half_x, round(2 * half_x / resolution) + 1, device=self.device)
        ys = torch.linspace(-half_y, half_y, round(2 * half_y / resolution) + 1, device=self.device)
        self.scan_points = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        if self.scan_points.shape != (77, 2):
            raise ValueError("Single-actor Go2 requires an 11 by 7 terrain scan")
        noise = self.cfg.noise.noise_scales
        self.noise_amplitude = self.default_joint_pos.new_tensor(
            (noise.base_ang_vel,) * 3 + (noise.projected_gravity,) * 3 + (0.,) * 3
            + (noise.joint_pos,) * 12 + (noise.joint_vel,) * 12 + (0.,) * 12) * self.cfg.noise.noise_level
        payload = {"task": self.cfg.to_dict(), "geometry": self.surface.geometry_hash,
                   "default_joint_pos": self.default_joint_pos.cpu().tolist(),
                   "joint_limits": self.joint_limits.cpu().tolist(),
                   "layout": "go2_single_moe_analytic_v1_wxyz_frame_major"}
        self.semantic_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.spec = AlgorithmSpec(37, 114, 36, 12, 45, 130, 26, 14, 3,
                                  "go2_single_moe_analytic_v1:" + self.semantic_hash,
                                  model_id="supermimic_single_moe_v1")

    def _validate_config(self):
        if self.default_joint_pos.shape != (12,) or self.joint_limits.shape != (12, 2):
            raise ValueError("Go2 joint defaults and limits must have shapes [12] and [12,2]")
        if not torch.isfinite(self.default_joint_pos).all() or not torch.isfinite(self.joint_limits).all():
            raise ValueError("Go2 joint definitions must be finite")
        if (self.joint_limits[:, 0] >= self.joint_limits[:, 1]).any():
            raise ValueError("Go2 joint limits must be strictly ordered")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Control timestep must be positive and finite")
        if self.cfg.control.delay_min_lag != 0 or self.cfg.control.delay_max_lag != 0 or self.cfg.commands.heading_command:
            raise ValueError("Single-actor Go2 requires zero action delay and direct yaw commands")
        reward = self.cfg.reward
        for name in ("tracking_std", "upright_std", "base_height_min", "fall_height", "fall_angle",
                     "fall_softness_deg", "height_softness_m"):
            value = getattr(reward, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Reward setting must be positive and finite: {name}")
        if reward.fall_angle >= math.pi:
            raise ValueError("Fall angle must be less than pi radians")
        lower, upper = self.cfg.commands.resampling_time
        if not all(math.isfinite(value) for value in (lower, upper)) or not 0 < lower <= upper:
            raise ValueError("Command resampling times must be positive and ordered")

    # ---------- 状态特征与地形 ----------

    def world_position(self, state: TaskState) -> Tensor:
        return state.physical[:, POSITION] + state.context[:, CONTEXT_ORIGIN]

    def surface_height(self, xy: Tensor) -> Tensor:
        """无支持区域使用有限占位；硬／软失败会阻断该区域的后续回报。"""
        valid = self.surface.contains(xy)
        safe = torch.where(valid.unsqueeze(-1), xy, torch.zeros_like(xy))
        height = self.surface.height_unchecked(safe)
        return torch.where(valid, height, torch.zeros_like(height))

    def base_height(self, state: TaskState) -> Tensor:
        position = self.world_position(state)
        return position[:, 2] - self.surface_height(position[:, :2])

    def terrain_scan(self, state: TaskState) -> Tensor:
        position = self.world_position(state)
        yaw = heading(state.physical[:, QUATERNION])
        x, y = self.scan_points.unbind(-1)
        points = torch.stack((yaw.cos()[:, None] * x - yaw.sin()[:, None] * y,
                              yaw.sin()[:, None] * x + yaw.cos()[:, None] * y), -1) + position[:, None, :2]
        return position[:, 2:3] - self.surface_height(points)

    def physical_features(self, state: TaskState) -> Tensor:
        physical = state.physical
        quaternion = physical[:, QUATERNION]
        return torch.cat((self.base_height(state).unsqueeze(-1), rotation_features(quaternion),
                          quat_apply_inverse(quaternion, physical[:, LINEAR_VELOCITY]),
                          physical[:, ANGULAR_VELOCITY], physical[:, JOINT_POSITION] - self.default_joint_pos,
                          physical[:, JOINT_VELOCITY]), dim=-1)

    def dynamics_features(self, state: TaskState) -> Tensor:
        return torch.cat((self.physical_features(state), self.terrain_scan(state)), dim=-1)

    def model_input(self, state: TaskState, action: Tensor) -> Tensor:
        return torch.cat((self.dynamics_features(state), action, state.memory[:, PREVIOUS_ACTION], state.context[:, :13]), dim=-1)

    def privileged(self, state: TaskState) -> Tensor:
        return torch.cat((self.physical_features(state), state.memory[:, PREVIOUS_ACTION], state.command,
                          state.memory[:, COMMAND_TIMER, None], self.terrain_scan(state)), dim=-1)

    def observe(self, state: TaskState, *, noisy: bool = False) -> Tensor:
        """仅本体传感器、命令及实际上一动作；不读取特权状态。"""
        physical = state.physical
        frame = torch.cat((physical[:, ANGULAR_VELOCITY], projected_gravity(physical[:, QUATERNION]), state.command,
                           physical[:, JOINT_POSITION] - self.default_joint_pos, physical[:, JOINT_VELOCITY],
                           state.memory[:, PREVIOUS_ACTION]), dim=-1)
        if noisy and self.cfg.noise.add_noise:
            frame = frame + (2 * torch.rand_like(frame) - 1) * self.noise_amplitude
        return frame

    # ---------- 可逆残差与任务推进 ----------

    def model_target(self, state: TaskState, next_state: TaskState) -> Tensor:
        """返回与半隐式积分精确对应的机体系残差，不将姿态四元数直接相减。"""
        old, new = state.physical, next_state.physical
        quaternion = quat_normalize(old[:, QUATERNION])
        orientation_delta = quat_to_rotvec(quat_multiply(quat_conjugate(quaternion), new[:, QUATERNION]))
        return torch.cat((quat_apply_inverse(quaternion, new[:, LINEAR_VELOCITY] - old[:, LINEAR_VELOCITY]),
                          new[:, ANGULAR_VELOCITY] - old[:, ANGULAR_VELOCITY],
                          new[:, JOINT_VELOCITY] - old[:, JOINT_VELOCITY],
                          quat_apply_inverse(quaternion, new[:, POSITION] - old[:, POSITION] - self.dt * new[:, LINEAR_VELOCITY]),
                          orientation_delta - self.dt * new[:, ANGULAR_VELOCITY],
                          new[:, JOINT_POSITION] - old[:, JOINT_POSITION] - self.dt * new[:, JOINT_VELOCITY]), dim=-1)

    def integrate(self, state: TaskState, action: Tensor, delta: Tensor) -> Tensor:
        del action
        physical = state.physical
        quaternion = quat_normalize(physical[:, QUATERNION])
        dv, dw, dqd, dp, dr, dq = delta.split((3, 3, 12, 3, 3, 12), dim=-1)
        velocity = physical[:, LINEAR_VELOCITY] + quat_apply(quaternion, dv)
        angular = physical[:, ANGULAR_VELOCITY] + dw
        joint_velocity = physical[:, JOINT_VELOCITY] + dqd
        position = physical[:, POSITION] + self.dt * velocity + quat_apply(quaternion, dp)
        orientation = quat_normalize(quat_multiply(quaternion, quat_from_rotvec(self.dt * angular + dr)))
        joint_position = physical[:, JOINT_POSITION] + self.dt * joint_velocity + dq
        return torch.cat((position, orientation, joint_position, velocity, angular, joint_velocity), dim=-1)

    def sample_commands(self, context: Tensor) -> tuple[Tensor, Tensor]:
        bounds = context[:, CONTEXT_RANGES].reshape(-1, 3, 2)
        command = bounds[..., 0] + torch.rand_like(bounds[..., 0]) * (bounds[..., 1] - bounds[..., 0])
        standing = torch.rand_like(context[:, CONTEXT_ZERO_PROBABILITY]) < context[:, CONTEXT_ZERO_PROBABILITY]
        command = torch.where(standing.unsqueeze(-1), torch.zeros_like(command), command)
        lower, upper = self.cfg.commands.resampling_time
        timer = lower + (upper - lower) * torch.rand_like(context[:, CONTEXT_ZERO_PROBABILITY])
        return command, timer

    def advance(self, state: TaskState, action: Tensor, next_physical: Tensor) -> TaskState:
        """全批量外生命令采样与where选择可编译，不按过期行产生动态形状。"""
        clock = state.memory[:, PHASE_CLOCK] + self.dt
        timer = state.memory[:, COMMAND_TIMER] - self.dt
        new_command, new_timer = self.sample_commands(state.context)
        expired = timer <= 1e-7
        command = torch.where(expired.unsqueeze(-1), new_command, state.command)
        timer = torch.where(expired, new_timer, timer)
        return TaskState(next_physical, torch.cat((action, clock.unsqueeze(-1), timer.unsqueeze(-1)), -1), command, state.context)

    # ---------- 奖励与失败 ----------

    def reward_terms(self, state: TaskState, action: Tensor, next_state: TaskState, failure: Tensor) -> dict[str, Tensor]:
        return single_reward_terms(self, state, action, next_state, failure)

    def reward(self, state: TaskState, action: Tensor, next_state: TaskState, failure: Tensor) -> Tensor:
        return self.dt * sum(getattr(self.cfg.reward.scales, name) * value
                             for name, value in self.reward_terms(state, action, next_state, failure).items())

    def hard_termination(self, state: TaskState) -> Tensor:
        supported = self.surface.contains(self.world_position(state)[:, :2], margin=self.cfg.terrain.boundary_margin_m)
        gravity = projected_gravity(state.physical[:, QUATERNION])
        return (self.base_height(state) < self.cfg.reward.fall_height) | (gravity[:, 2] > -math.cos(self.cfg.reward.fall_angle)) | ~supported

    def soft_termination(self, state: TaskState) -> Tensor:
        """高度与倾角阈值的平滑并集；支持域外明确失效，不能继续bootstrap。"""
        cfg = self.cfg.reward
        gravity = projected_gravity(state.physical[:, QUATERNION])
        tilt = torch.acos((-gravity[:, 2]).clamp(-1 + 1e-5, 1 - 1e-5))
        angle_failure = torch.sigmoid((tilt - cfg.fall_angle) / math.radians(cfg.fall_softness_deg))
        height_failure = torch.sigmoid((cfg.fall_height - self.base_height(state)) / cfg.height_softness_m)
        probability = 1 - (1 - angle_failure) * (1 - height_failure)
        supported = self.surface.contains(self.world_position(state)[:, :2], margin=self.cfg.terrain.boundary_margin_m)
        return torch.where(supported, probability, torch.ones_like(probability))

    # ---------- 真实回放事件 ----------

    def _replay_events(self, window):
        """按参考实现用运动学估计触地；粗糙地形改用足端相对当地表面的高度。"""
        physical = torch.cat((window.initial.physical[:, None], window.targets.physical), dim=1)
        context = torch.cat((window.initial.context[:, None], window.targets.context), dim=1)
        batch, length = physical.shape[:2]
        flat = physical.reshape(-1, self.spec.state_dim)
        origin = context[..., CONTEXT_ORIGIN].reshape(-1, 3)
        feet = foot_positions_world(flat[:, POSITION] + origin, flat[:, QUATERNION], flat[:, JOINT_POSITION])
        height = feet[..., 2] - self.surface_height(feet[..., :2])
        contact = (height < .035).reshape(batch, length, 4)
        valid = torch.cat((torch.ones_like(window.valid[:, :1]), window.valid), dim=1)
        flight = ~contact.any(-1) & valid
        changes = (contact[:, 1:] != contact[:, :-1]).sum(-1) * window.valid
        hard_motion = (physical[..., LINEAR_VELOCITY][..., 2].abs() > .6) & valid
        falls = (window.terminated & window.valid).any(-1)
        return changes, flight, hard_motion, falls

    def replay_priority(self, window) -> Tensor:
        """优先拟合触地切换、飞行和真实失败；重置后的无效数据不计分。"""
        changes, flight, _, falls = self._replay_events(window)
        return changes.sum(-1).float() + flight.sum(-1).float() + 4.0 * falls.float()

    def archive_worthy(self, window) -> Tensor:
        """归档至少两帧飞行、明显垂向运动或真实跌倒，普通触地切换仅优先采样。"""
        _, flight, hard_motion, falls = self._replay_events(window)
        return (flight.sum(-1) >= 2) | hard_motion.any(-1) | falls
