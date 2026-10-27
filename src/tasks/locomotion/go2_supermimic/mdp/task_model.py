# Copyright (c) 2026-2027 zh
"""内容：组合Go2可微物理积分、任务推进、本体观测、地形特权与奖励。"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

import torch
from torch import Tensor

from agents.supermimic.interfaces import AlgorithmSpec, TaskState

from .commands import advance_task, sample_commands
from .observations import actor_observation, phase_features
from .rewards import reward_terms
from .terminations import hard_termination
from .state_spec import (
    ANGULAR_VELOCITY, COMMAND_TIMER, CONTEXT_ORIGIN, JOINT_POSITION, JOINT_VELOCITY,
    LINEAR_VELOCITY, MEMORY_DIM, POSITION, PREVIOUS_ACTION, QUATERNION,
    heading, quat_apply, quat_apply_inverse, quat_conjugate, quat_from_rotvec,
    quat_multiply, quat_normalize, quat_to_rotvec, rotation_features,
)


class Go2TaskModel:
    """任务函数使用固定张量和静态地形，不反向依赖仿真adapter。"""

    def __init__(self, cfg, default_joint_pos: Tensor, joint_limits: Tensor, surface, device="cpu"):
        self.cfg = deepcopy(cfg)
        self.device = torch.device(device)
        self.default_joint_pos = torch.as_tensor(default_joint_pos, dtype=torch.float32, device=self.device).clone()
        self.joint_limits = torch.as_tensor(joint_limits, dtype=torch.float32, device=self.device).clone()
        self.surface = surface
        self.dt = self.cfg.sim.timestep * self.cfg.control.decimation
        self.output_scales = tuple(getattr(self.cfg.model, "output_scales", (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)))
        self.auxiliary_scales = torch.tensor(self.cfg.model.auxiliary_scales, dtype=torch.float32, device=self.device)
        self.privileged_memory_scales = torch.cat((torch.ones(12, device=self.device), self.auxiliary_scales))
        self._validate_config()
        half_x, half_y = (value / 2 for value in self.cfg.terrain.scan_size_m)
        resolution = self.cfg.terrain.scan_resolution_m
        xs = torch.linspace(-half_x, half_x, round(2 * half_x / resolution) + 1, device=self.device)
        ys = torch.linspace(-half_y, half_y, round(2 * half_y / resolution) + 1, device=self.device)
        self.scan_points = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
        if self.scan_points.shape != (187, 2):
            raise ValueError("Go2 task requires the configured 17 by 11 terrain scan")
        configuration = self.cfg.to_dict()
        # 旧完整快照显式恢复原行为时，哈希保持新增保护字段出现前的语义。
        if configuration["policy"].get("action_limit") is None:
            configuration["policy"].pop("action_limit", None)
        if configuration["learner"].get("actor_ramp_updates") == 0:
            configuration["learner"].pop("actor_ramp_updates", None)
        payload = {"task": configuration, "geometry": surface.geometry_hash,
                   "default_joint_pos": self.default_joint_pos.cpu().tolist(), "joint_limits": self.joint_limits.cpu().tolist(),
                   "output_scales": self.output_scales, "layout": "go2_supermimic_him_rewards_v3_wxyz_time_major"}
        self.semantic_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        self.spec = AlgorithmSpec(37, 224, 36, 12, 47, 298, 26, MEMORY_DIM, 3,
                                  "go2_supermimic_him_rewards_v3:" + self.semantic_hash)

    def _validate_config(self):
        if self.default_joint_pos.shape != (12,) or self.joint_limits.shape != (12, 2):
            raise ValueError("Go2 joint defaults and soft limits must have the configured ordered dimensions")
        if not torch.isfinite(self.default_joint_pos).all() or not torch.isfinite(self.joint_limits).all():
            raise ValueError("Go2 joint definitions must be finite")
        if (self.joint_limits[:, 0] >= self.joint_limits[:, 1]).any():
            raise ValueError("Go2 joint limits must be strictly ordered")
        if self.cfg.control.delay_min_lag != 0 or self.cfg.control.delay_max_lag != 0:
            raise ValueError("Go2 SuperMimic v2 requires zero actuator delay")
        if self.cfg.commands.heading_command:
            raise ValueError("Go2 SuperMimic v2 requires direct yaw-rate commands")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Task control timestep must be positive and finite")
        settings = self.output_scales + tuple(self.cfg.reward.state_error_scales)
        if len(self.output_scales) != 6 or len(self.cfg.reward.state_error_scales) != 6 or any(not math.isfinite(x) or x <= 0 for x in settings):
            raise ValueError("State integration and error scales must contain six positive finite values")
        if self.auxiliary_scales.shape != (18,) or not torch.isfinite(self.auxiliary_scales).all() or (self.auxiliary_scales <= 0).any():
            raise ValueError("Reward auxiliary scales must contain eighteen positive finite values")
        if not math.isfinite(self.cfg.model.auxiliary_loss_weight) or self.cfg.model.auxiliary_loss_weight < 0:
            raise ValueError("Reward auxiliary loss weight must be finite and nonnegative")
        for value in (self.cfg.reward.gait_period, self.cfg.reward.tracking_linear_std, self.cfg.reward.tracking_angular_std):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Reward periods and tracking standard deviations must be positive")
        reward = self.cfg.reward
        for values in (reward.pose_std_standing, reward.pose_std_walking, reward.pose_std_running):
            if len(values) != 3 or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError("Pose standard deviations must contain three positive finite values")
        if len(reward.pose_terrain_tolerance) != 3 or any(not math.isfinite(x) or x < 0 for x in reward.pose_terrain_tolerance):
            raise ValueError("Pose terrain tolerances must contain three nonnegative finite values")
        low, high = reward.pose_terrain_height_std
        if not all(math.isfinite(x) for x in (low, high)) or not 0 <= low < high:
            raise ValueError("Pose terrain height thresholds must be finite and ordered")
        for value in (reward.pose_terrain_radius, reward.pose_scan_max_distance, reward.stumble_ratio):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Pose scan dimensions and stumble ratio must be positive and finite")
        if not 0 <= reward.pose_walking_threshold < reward.pose_running_threshold or not math.isfinite(reward.pose_running_threshold):
            raise ValueError("Pose speed thresholds must be finite and ordered")
        if not math.isfinite(reward.gait_stance_fraction) or not 0 < reward.gait_stance_fraction < 1:
            raise ValueError("Gait stance fraction must be finite and between zero and one")
        if not math.isfinite(reward.collision_force_threshold) or reward.collision_force_threshold < 0:
            raise ValueError("Collision force threshold must be finite and nonnegative")
        lower, upper = self.cfg.commands.resampling_time
        if not all(math.isfinite(x) for x in (lower, upper)) or not 0 < lower <= upper:
            raise ValueError("Command resampling times must be ordered and positive")

    def world_position(self, state: TaskState) -> Tensor:
        return state.physical[:, POSITION] + state.context[:, CONTEXT_ORIGIN]

    def surface_height(self, xy: Tensor) -> Tensor:
        """仅有效查询进入表面；无效终态的占位值不代表存在可行地面。"""
        shape = xy.shape[:-1]
        flat = xy.reshape(-1, 2)
        valid = self.surface.contains(flat)
        result = flat[..., 0] * 0
        if valid.any():
            rows = valid.nonzero().squeeze(-1)
            result = result.index_copy(0, rows, self.surface.height(flat[valid]))
        return result.reshape(shape)

    def terrain_scan(self, state: TaskState) -> Tensor:
        position = self.world_position(state)
        yaw = heading(state.physical[:, QUATERNION])
        x, y = self.scan_points.unbind(-1)
        rotated_x = yaw.cos()[:, None] * x - yaw.sin()[:, None] * y
        rotated_y = yaw.sin()[:, None] * x + yaw.cos()[:, None] * y
        points = torch.stack((rotated_x, rotated_y), -1) + position[:, None, :2]
        return position[:, 2:3] - self.surface_height(points)

    def physical_features(self, state: TaskState) -> Tensor:
        physical = state.physical
        position = self.world_position(state)
        height = position[:, 2:3] - self.surface_height(position[:, :2]).unsqueeze(-1)
        quaternion = physical[:, QUATERNION]
        velocity = quat_apply_inverse(quaternion, physical[:, LINEAR_VELOCITY])
        return torch.cat((height, rotation_features(quaternion), velocity,
                          physical[:, ANGULAR_VELOCITY] / 2, physical[:, JOINT_POSITION],
                          physical[:, JOINT_VELOCITY] / 5), dim=-1)

    def dynamics_features(self, state: TaskState) -> Tensor:
        return torch.cat((self.physical_features(state), self.terrain_scan(state)), dim=-1)

    def integrate(self, state: TaskState, action: Tensor, delta: Tensor) -> Tensor:
        del action  # 动作经动力学网络产生delta，不在这里重复施加关节目标缩放。
        if delta.shape != (state.physical.shape[0], 36):
            raise ValueError("Go2 dynamics delta must have shape [B,36]")
        physical = state.physical
        quaternion = quat_normalize(physical[:, QUATERNION])
        groups = delta.split((3, 3, 12, 3, 3, 12), dim=-1)
        dv, dw, dqd, dp, dr, dq = (group * scale for group, scale in zip(groups, self.output_scales))
        velocity = physical[:, LINEAR_VELOCITY] + quat_apply(quaternion, dv)
        angular = physical[:, ANGULAR_VELOCITY] + dw
        joint_velocity = physical[:, JOINT_VELOCITY] + dqd
        position = physical[:, POSITION] + self.dt * velocity + quat_apply(quaternion, dp)
        orientation = quat_normalize(quat_multiply(quaternion, quat_from_rotvec(self.dt * angular + dr)))
        joint_position = physical[:, JOINT_POSITION] + self.dt * joint_velocity + dq
        return torch.cat((position, orientation, joint_position, velocity, angular, joint_velocity), dim=-1)

    def sample_commands(self, context: Tensor) -> tuple[Tensor, Tensor]:
        return sample_commands(context, tuple(self.cfg.commands.resampling_time))

    def advance(self, state: TaskState, action: Tensor, next_physical: Tensor) -> TaskState:
        return advance_task(state, action, next_physical, self.dt, tuple(self.cfg.commands.resampling_time))

    def observe(self, state: TaskState, *, noisy: bool = False) -> Tensor:
        return actor_observation(state, self.default_joint_pos, self.cfg, noisy=noisy)

    def privileged(self, state: TaskState) -> Tensor:
        return torch.cat((self.physical_features(state), state.memory[:, PREVIOUS_ACTION],
                          phase_features(state, self.cfg.reward.gait_period, self.cfg.reward.command_threshold, gated=False),
                          state.command, state.memory[:, COMMAND_TIMER, None], state.context, self.terrain_scan(state),
                          state.memory[:, 14:] / self.privileged_memory_scales), dim=-1)

    def reward_terms(self, state: TaskState, action: Tensor, next_state: TaskState, failure: Tensor) -> dict[str, Tensor]:
        return reward_terms(self, state, action, next_state, failure)

    def reward(self, state: TaskState, action: Tensor, next_state: TaskState, failure: Tensor) -> Tensor:
        terms = self.reward_terms(state, action, next_state, failure)
        return self.dt * sum(getattr(self.cfg.reward.scales, name) * value for name, value in terms.items())

    def state_error(self, predicted: TaskState, target: TaskState) -> Tensor:
        left, right = predicted.physical, target.physical
        rotation = quat_to_rotvec(quat_multiply(quat_conjugate(quat_normalize(right[:, QUATERNION])),
                                               quat_normalize(left[:, QUATERNION])))
        errors = (left[:, POSITION] - right[:, POSITION], rotation,
                  left[:, LINEAR_VELOCITY] - right[:, LINEAR_VELOCITY],
                  left[:, ANGULAR_VELOCITY] - right[:, ANGULAR_VELOCITY],
                  left[:, JOINT_POSITION] - right[:, JOINT_POSITION],
                  left[:, JOINT_VELOCITY] - right[:, JOINT_VELOCITY])
        return torch.stack([(error / scale).square().mean(-1)
                            for error, scale in zip(errors, self.cfg.reward.state_error_scales)], dim=-1).mean(-1)

    def hard_termination(self, state: TaskState) -> Tensor:
        return hard_termination(self, state)
