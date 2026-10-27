# Copyright (c) 2026-2027 zh
"""内容：定义Go2物理布局、参考系和可微四元数运算。"""

from __future__ import annotations

import torch
from torch import Tensor


JOINT_NAMES = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf"))
FOOT_NAMES = ("FL", "FR", "RL", "RR")
ACTION_DIM = len(JOINT_NAMES)
# 物理状态：相对出生原点的位置、wxyz姿态、关节位置、世界系线速度、机体系角速度、关节速度。
POSITION = slice(0, 3)
QUATERNION = slice(3, 7)
JOINT_POSITION = slice(7, 19)
LINEAR_VELOCITY = slice(19, 22)
ANGULAR_VELOCITY = slice(22, 25)
JOINT_VELOCITY = slice(25, 37)
PREVIOUS_ACTION = slice(0, 12)
PHASE_CLOCK = 12
COMMAND_TIMER = 13
PREVIOUS_PREVIOUS_ACTION = slice(14, 26)
SENSOR_REWARDS = slice(26, 36)
CONTACT_AIR_TIME = slice(36, 40)
CONTACT_TIME = slice(40, 44)
REWARD_AUXILIARY = slice(26, 44)
# 以下顺序同时约束真实传感器采集、奖励计算与世界模型辅助监督。
SENSOR_REWARD_NAMES = (
    "angular_momentum", "joint_acc_l2", "joint_torques_l2", "foot_gait", "feet_air_time",
    "foot_slip", "soft_landing", "leg_collision", "self_collision", "stumble",
)
CONTEXT_FRICTION = 0
CONTEXT_ENCODER_BIAS = slice(1, 13)
# 13:16保留名义IMU偏置；当前任务不启用物理域随机化。
CONTEXT_ORIGIN = slice(16, 19)
CONTEXT_RANGES = slice(19, 25)
CONTEXT_ZERO_PROBABILITY = 25
STATE_DIM, MEMORY_DIM, CONTEXT_DIM = 37, 44, 26


def quat_normalize(quaternion: Tensor) -> Tensor:
    """采用wxyz约定，退化输入回退单位四元数以保持数值有限。"""
    norm = quaternion.norm(dim=-1, keepdim=True)
    identity = torch.cat((torch.ones_like(quaternion[..., :1]), torch.zeros_like(quaternion[..., 1:])), -1)
    return torch.where(norm > 1e-8, quaternion / norm.clamp_min(1e-8), identity)


def quat_conjugate(quaternion: Tensor) -> Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quat_multiply(left: Tensor, right: Tensor) -> Tensor:
    scalar = left[..., :1] * right[..., :1] - (left[..., 1:] * right[..., 1:]).sum(-1, keepdim=True)
    vector = left[..., :1] * right[..., 1:] + right[..., :1] * left[..., 1:] + torch.cross(left[..., 1:], right[..., 1:], dim=-1)
    return torch.cat((scalar, vector), dim=-1)


def quat_apply(quaternion: Tensor, vector: Tensor) -> Tensor:
    quaternion = quat_normalize(quaternion)
    first = 2 * torch.cross(quaternion[..., 1:], vector, dim=-1)
    return vector + quaternion[..., :1] * first + torch.cross(quaternion[..., 1:], first, dim=-1)


def quat_apply_inverse(quaternion: Tensor, vector: Tensor) -> Tensor:
    return quat_apply(quat_conjugate(quaternion), vector)


def quat_from_rotvec(rotation: Tensor) -> Tensor:
    angle = rotation.norm(dim=-1, keepdim=True)
    # sinc在零旋转处有有限、正确的导数，避免除以零。
    return torch.cat((torch.cos(angle / 2), rotation * (0.5 * torch.sinc(angle / (2 * torch.pi)))), -1)


def quat_to_rotvec(quaternion: Tensor) -> Tensor:
    quaternion = quat_normalize(quaternion)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    norm = vector.norm(dim=-1, keepdim=True)
    scale = 2 * torch.atan2(norm, quaternion[..., :1].clamp_min(0)) / norm.clamp_min(1e-8)
    return vector * torch.where(norm > 1e-6, scale, torch.full_like(scale, 2.0))


def heading(quaternion: Tensor) -> Tensor:
    w, x, y, z = quat_normalize(quaternion).unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def heading_quaternion(quaternion: Tensor) -> Tensor:
    half = heading(quaternion) / 2
    zero = torch.zeros_like(half)
    return torch.stack((half.cos(), zero, zero, half.sin()), -1)


def rotation_features(quaternion: Tensor) -> Tensor:
    local = quat_multiply(quat_conjugate(heading_quaternion(quaternion)), quat_normalize(quaternion))
    unit_x = torch.zeros_like(local[..., 1:])
    unit_y = torch.zeros_like(unit_x)
    unit_x[..., 0] = 1
    unit_y[..., 1] = 1
    return torch.cat((quat_apply(local, unit_x), quat_apply(local, unit_y)), dim=-1)


def projected_gravity(quaternion: Tensor) -> Tensor:
    gravity = torch.zeros_like(quaternion[..., 1:])
    gravity[..., 2] = -1
    return quat_apply_inverse(quaternion, gravity)
