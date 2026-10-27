# Copyright (c) 2026-2027 zh
"""内容：采集原生物理观测，并由状态与记忆生成47维盲本体帧。"""

from __future__ import annotations

import torch
from torch import Tensor

from agents.supermimic.interfaces import TaskState

from .state_spec import ANGULAR_VELOCITY, JOINT_POSITION, JOINT_VELOCITY, PHASE_CLOCK, PREVIOUS_ACTION, QUATERNION, projected_gravity


def physical_observation(env) -> Tensor:
    """原生manager输出37维无历史状态，策略帧另由actor_observation生成。"""
    data = env.scene["robot"].data
    return torch.cat((data.root_link_pos_w - env.scene.env_origins, data.root_link_quat_w, data.joint_pos,
                      data.root_link_lin_vel_w, data.root_link_ang_vel_b, data.joint_vel), dim=-1)


def phase_features(state: TaskState, period: float, threshold: float, *, gated: bool = True) -> Tensor:
    angle = state.memory[:, PHASE_CLOCK] * (2 * torch.pi / period)
    phase = torch.stack((angle.sin(), angle.cos()), dim=-1)
    if gated:
        active = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs() > threshold
        phase = phase * active.unsqueeze(-1)
    return phase


def actor_observation(state: TaskState, default_joint_position: Tensor, cfg, *, noisy: bool = False) -> Tensor:
    """不访问真实线速度、地形、context和任何仿真传感器缓存。"""
    physical = state.physical
    frame = torch.cat((physical[:, ANGULAR_VELOCITY], projected_gravity(physical[:, QUATERNION]), state.command,
                       phase_features(state, cfg.reward.gait_period, cfg.reward.command_threshold),
                       physical[:, JOINT_POSITION] - default_joint_position, physical[:, JOINT_VELOCITY],
                       state.memory[:, PREVIOUS_ACTION]), dim=-1)
    if noisy and cfg.noise.add_noise:
        scales = cfg.noise.noise_scales
        amplitude = frame.new_tensor((scales.base_ang_vel,) * 3 + (scales.projected_gravity,) * 3 + (0.0,) * 5
                                     + (scales.joint_pos,) * 12 + (scales.joint_vel,) * 12 + (0.0,) * 12)
        frame = frame + (2 * torch.rand_like(frame) - 1) * amplitude * cfg.noise.noise_level
    return frame
