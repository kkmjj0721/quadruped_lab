# Copyright (c) 2026-2027 zh
"""内容：实现表A.6、指定HIM及站立速度共十六项奖励；返回未乘控制周期的奖励率。"""

import torch

from .reference_state import (
    ANGULAR_BODY, CONTACT_FOUND, FOOT_VELOCITY_XY, GRAVITY, JOINT_ACCELERATION, JOINT_OFFSET, JOINT_VELOCITY,
    body_vertical_velocity, normalized_gravity, tracking_velocity,
)
from .state_spec import PREVIOUS_ACTION, PREVIOUS_PREVIOUS_ACTION
from .gait_rewards import gait_reward_terms, posture_reward


def reference_reward_terms(task, state, action, next_state, failure):
    """起点命令评价终点状态；表A.6没有额外跌倒、机身高度或力矩奖励。"""
    del failure
    cfg = task.cfg.reward
    physical = next_state.physical
    velocity = tracking_velocity(physical)
    gravity = normalized_gravity(physical)
    joint = physical[:, JOINT_OFFSET] + task.default_joint_pos
    lower, upper = task.joint_limits.unbind(-1)
    # heading系躯干z轴的y分量为-sin(roll)，由机体系重力恢复；纯前倾不受惩罚。
    sideways_lean = gravity[:, 1].square() / gravity[:, 1:3].square().sum(-1).clamp_min(1e-8)
    # heading系与世界系仅相差yaw，因此roll/pitch角速模平方不变。
    tilt_rate = (physical[:, ANGULAR_BODY].square().sum(-1) - velocity[:, 2].square()).clamp_min(0)
    command_magnitude = state.command[:, :2].norm(dim=-1) + state.command[:, 2].abs()
    standing = command_magnitude <= cfg.stand_still_command_threshold
    foot_speed_squared = physical[:, FOOT_VELOCITY_XY].reshape(-1, 4, 2).square().sum(-1)
    terms = {
        "track_linear_velocity": torch.exp(-(velocity[:, :2] - state.command[:, :2]).square().sum(-1)
                                            / cfg.tracking_linear_variance),
        "track_angular_velocity": torch.exp(-(velocity[:, 2] - state.command[:, 2]).square()
                                             / cfg.tracking_angular_variance),
        "action_rate_l2": (action - state.memory[:, PREVIOUS_ACTION]).square().sum(-1),
        "joint_pos_limits": ((lower - joint).clamp_min(0) + (joint - upper).clamp_min(0)).sum(-1),
        "torso_sideways_lean": sideways_lean,
        "body_ang_vel": tilt_rate,
        "lin_vel_z": body_vertical_velocity(physical).square(),
        "body_orientation_l2": physical[:, GRAVITY][:, :2].square().sum(-1),
        "smoothness": (action - 2 * state.memory[:, PREVIOUS_ACTION]
                       + state.memory[:, PREVIOUS_PREVIOUS_ACTION]).square().sum(-1),
        "stand_still": physical[:, JOINT_OFFSET].square().sum(-1) * standing,
        "stand_still_joint_vel_l2": physical[:, JOINT_VELOCITY].square().sum(-1) * standing,
        "foot_slip": (foot_speed_squared * physical[:, CONTACT_FOUND]).sum(-1)
                     * (command_magnitude > cfg.foot_slip_command_threshold),
        "joint_acc_l2": physical[:, JOINT_ACCELERATION].square().sum(-1),
    }
    terms.update(gait_reward_terms(task, state, next_state))
    terms["pose"] = posture_reward(task, state, next_state)
    return terms
