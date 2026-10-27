# Copyright (c) 2026-2027 zh
"""
内容：
    用状态解析计算单策略盲走奖励，真实与想象共用相同公式。
    PD力矩和控制步关节加速度是奖励代理，不等同物理子步传感器读数。
"""

from __future__ import annotations

import torch
from torch import Tensor

from agents.supermimic.interfaces import TaskState

from .state_spec import (
    ANGULAR_VELOCITY, JOINT_POSITION, JOINT_VELOCITY, LINEAR_VELOCITY,
    PREVIOUS_ACTION, QUATERNION, projected_gravity, quat_apply_inverse,
)


def single_reward_terms(task, state: TaskState, action: Tensor, next_state: TaskState,
                        failure: Tensor) -> dict[str, Tensor]:
    """速度目标取转移起点；返回奖励率，统一由聚合调用乘一次控制周期。"""
    cfg = task.cfg.reward
    physical = next_state.physical
    joint, joint_velocity = physical[:, JOINT_POSITION], physical[:, JOINT_VELOCITY]
    velocity = quat_apply_inverse(physical[:, QUATERNION], physical[:, LINEAR_VELOCITY])
    gravity = projected_gravity(physical[:, QUATERNION])
    target = task.default_joint_pos + task.action_scale * action
    torque = (task.kp * (target - joint) - task.kd * joint_velocity).clamp(-task.effort_limit, task.effort_limit)
    acceleration = (joint_velocity - state.physical[:, JOINT_VELOCITY]) / task.dt
    lower, upper = task.joint_limits.unbind(-1)
    height = task.base_height(next_state)
    return {
        "track_linear_velocity": torch.exp(-(velocity[:, :2] - state.command[:, :2]).square().sum(-1) / cfg.tracking_std ** 2),
        "track_angular_velocity": torch.exp(-(physical[:, ANGULAR_VELOCITY][:, 2] - state.command[:, 2]).square() / cfg.tracking_std ** 2),
        "upright": torch.exp(-gravity[:, :2].square().sum(-1) / cfg.upright_std ** 2),
        "lin_vel_z": velocity[:, 2].square(),
        "body_ang_vel": physical[:, ANGULAR_VELOCITY][:, :2].square().sum(-1),
        "joint_deviation_l2": (joint - task.default_joint_pos).square().sum(-1),
        "base_height_low": (cfg.base_height_min - height).clamp_min(0),
        "action_rate_l2": (action - state.memory[:, PREVIOUS_ACTION]).square().sum(-1),
        "joint_pos_limits": ((lower - joint).clamp_min(0) + (joint - upper).clamp_min(0)).sum(-1),
        "joint_torques_l2": torque.square().sum(-1),
        "joint_acc_l2": acceleration.square().sum(-1),
        "is_terminated": failure,
    }
