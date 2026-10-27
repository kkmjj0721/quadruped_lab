# Copyright (c) 2026-2027 zh
"""
内容：
    按RI-4438 HIM的22项奖励公式聚合Go2真实或预测转移的原始奖励率。
    来源为legged_wbc_mjlab提交e6209ec3d910b57673d5cbba1d3ca798a7fa3113的
    ri_4438_him/mdp/rewards.py及环境配置；仅适配Tensor状态和Go2关节顺序。
"""

from __future__ import annotations

import torch
from torch import Tensor

from agents.supermimic.interfaces import TaskState

from .state_spec import (
    ANGULAR_VELOCITY, JOINT_POSITION, LINEAR_VELOCITY, PREVIOUS_ACTION,
    PREVIOUS_PREVIOUS_ACTION, QUATERNION, SENSOR_REWARD_NAMES, SENSOR_REWARDS,
    heading, projected_gravity, quat_apply_inverse,
)


def variable_posture(task, state: TaskState, next_state: TaskState) -> Tensor:
    """照参考公式按速度选择姿态标准差，并随附近地形起伏放宽关节容差。"""
    cfg = task.cfg.reward
    physical = next_state.physical
    speed = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs()
    mode = (speed >= cfg.pose_walking_threshold).long() + (speed >= cfg.pose_running_threshold).long()
    values = physical.new_tensor((cfg.pose_std_standing, cfg.pose_std_walking,
                                  cfg.pose_std_running, cfg.pose_terrain_tolerance)).repeat(1, 4)
    # 扫描网格绕yaw旋转，距离权重对该旋转不变；高度由当前预测位姿重新查询。
    position = task.world_position(next_state)
    yaw = heading(physical[:, QUATERNION])
    x, y = task.scan_points.unbind(-1)
    world_xy = torch.stack((yaw.cos()[:, None] * x - yaw.sin()[:, None] * y,
                            yaw.sin()[:, None] * x + yaw.cos()[:, None] * y), -1) + position[:, None, :2]
    distances = task.terrain_scan(next_state)
    valid = task.surface.contains(world_xy) & torch.isfinite(distances) & (distances >= 0)
    valid = valid & (distances <= cfg.pose_scan_max_distance)
    weights = (1 - task.scan_points.square().sum(-1) / cfg.pose_terrain_radius ** 2).clamp_min(0).square()
    weights = weights.unsqueeze(0) * valid
    count = weights.sum(-1).clamp_min(1e-6)
    z = torch.where(valid, -distances, 0.0)
    mean = (weights * z).sum(-1) / count
    variance = (weights * (z - mean[:, None]).square()).sum(-1) / count
    # 原公式sqrt(0)的导数发散；零方差处保持同样的0值并显式使用有限导数。
    roughness = torch.where(variance > 0, variance.clamp_min(torch.finfo(variance.dtype).tiny).sqrt(), 0.0)
    low, high = cfg.pose_terrain_height_std
    alpha = ((roughness - low) / (high - low)).clamp(0, 1)
    alpha = alpha.square() * (3 - 2 * alpha)
    alpha = torch.where((mode > 0) & ((weights > 0).sum(-1) >= 4), alpha, 0.0)
    error = physical[:, JOINT_POSITION] - task.default_joint_pos
    error = (error.abs() - alpha[:, None] * values[3]).clamp_min(0)
    return torch.exp(-(error.square() / values[mode].square()).mean(-1))


def reward_terms(task, state: TaskState, action: Tensor, next_state: TaskState, failure: Tensor) -> dict[str, Tensor]:
    """目标取旧命令；传感器项从已复制或预测的终态记忆读取，不在这里乘dt。"""
    cfg = task.cfg.reward
    physical = next_state.physical
    quaternion = physical[:, QUATERNION]
    velocity = quat_apply_inverse(quaternion, physical[:, LINEAR_VELOCITY])
    angular = physical[:, ANGULAR_VELOCITY]
    joint = physical[:, JOINT_POSITION]
    gravity = projected_gravity(quaternion)
    active = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs() > cfg.command_threshold
    lower, upper = task.joint_limits.unbind(-1)
    joint_error = joint - task.default_joint_pos
    terms = {
        "track_linear_velocity": torch.exp(-(velocity[:, :2] - state.command[:, :2]).square().sum(-1) / cfg.tracking_linear_std ** 2),
        "track_angular_velocity": torch.exp(-(angular[:, 2] - state.command[:, 2]).square() / cfg.tracking_angular_std ** 2),
        "lin_vel_z": velocity[:, 2].square(),
        "body_ang_vel": angular[:, :2].square().sum(-1),
        "body_orientation_l2": gravity[:, :2].square().sum(-1),
        "pose": variable_posture(task, state, next_state),
        "action_rate_l2": (action - state.memory[:, PREVIOUS_ACTION]).square().sum(-1),
        "smoothness": (action - 2 * state.memory[:, PREVIOUS_ACTION]
                       + state.memory[:, PREVIOUS_PREVIOUS_ACTION]).square().sum(-1),
        "joint_pos_limits": ((lower - joint).clamp_min(0) + (joint - upper).clamp_min(0)).sum(-1),
        "hip_pos": joint_error[:, ::3].square().sum(-1) * ((state.command[:, 1].abs() <= cfg.command_threshold)
                                                          & (state.command[:, 2].abs() <= cfg.command_threshold)),
        "stand_still": joint_error.square().sum(-1) * ~active,
        "is_terminated": failure,
    }
    gated = {"foot_gait", "feet_air_time", "foot_slip", "soft_landing"}
    # 真实标签已经门控；此处保持相同边界，避免想象预测误差给零命令增添步态项。
    terms.update((name, value * active if name in gated else value)
                 for name, value in zip(SENSOR_REWARD_NAMES, next_state.memory[:, SENSOR_REWARDS].unbind(-1)))
    return terms
