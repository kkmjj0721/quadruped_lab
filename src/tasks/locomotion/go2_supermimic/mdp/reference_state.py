# Copyright (c) 2026-2027 zh
"""内容：定义Go2原73维状态、52维瞬时辅助量及旧航向系平面位姿增量。"""

import torch
from torch import Tensor


HEIGHT = slice(0, 1)
GRAVITY = slice(1, 4)
VELOCITY_HEADING = slice(4, 7)
ANGULAR_BODY = slice(7, 10)
JOINT_OFFSET = slice(10, 22)
JOINT_VELOCITY = slice(22, 34)
HIPS = slice(34, 46)
KNEES = slice(46, 58)
FEET = slice(58, 70)
ROBOT_STATE = slice(0, 70)
POSE = slice(70, 73)
JOINT_ACCELERATION = slice(73, 85)
FOOT_VELOCITY_XY = slice(85, 93)
CONTACT_FOUND = slice(93, 97)
FOOT_CONTACT_FORCE = slice(97, 109)
JOINT_TORQUE = slice(109, 121)
LEG_CONTACT_FORCE = slice(121, 122)
ANGULAR_MOMENTUM = slice(122, 125)  # heading系角动量；平方和与来源world系严格等价。
AUXILIARY = slice(73, 125)
PHYSICAL_DIM = 125
LEG_PHYSICAL_DIM = 122
FORCE_PHYSICAL_DIM = 121
LEGACY_PHYSICAL_DIM = 97
CONTEXT_COM = slice(13, 16)


def yaw_rotate_inverse(vector: Tensor, yaw: Tensor) -> Tensor:
    """绕世界z轴逆旋转；支持[B,3]及[B,K,3]向量。"""
    cosine, sine = yaw.cos(), yaw.sin()
    return torch.stack((cosine * vector[..., 0] + sine * vector[..., 1],
                        -sine * vector[..., 0] + cosine * vector[..., 1], vector[..., 2]), -1)


def pose_delta(before: Tensor, after: Tensor) -> Tensor:
    """位置增量在起点yaw系表达，角增量映射到[-pi,pi]。"""
    difference = after - before
    yaw = before[..., 2]
    return torch.stack((yaw.cos() * difference[..., 0] + yaw.sin() * difference[..., 1],
                        -yaw.sin() * difference[..., 0] + yaw.cos() * difference[..., 1],
                        torch.atan2(difference[..., 2].sin(), difference[..., 2].cos())), -1)


def apply_pose_delta(pose: Tensor, delta: Tensor) -> Tensor:
    yaw = pose[..., 2]
    return torch.stack((pose[..., 0] + yaw.cos() * delta[..., 0] - yaw.sin() * delta[..., 1],
                        pose[..., 1] + yaw.sin() * delta[..., 0] + yaw.cos() * delta[..., 1],
                        yaw + delta[..., 2]), -1)


def normalized_gravity(physical: Tensor) -> Tensor:
    """仅供姿态坐标计算；不投影或改写世界模型的直接状态预测。"""
    gravity = physical[..., GRAVITY]
    return gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def tracking_velocity(physical: Tensor) -> Tensor:
    """论文A.6命令坐标：heading平面速度和世界角速度z。"""
    world_yaw_rate = -(normalized_gravity(physical) * physical[..., ANGULAR_BODY]).sum(-1, keepdim=True)
    return torch.cat((physical[..., VELOCITY_HEADING][..., :2], world_yaw_rate), -1)


def body_planar_velocity(physical: Tensor) -> Tensor:
    """由heading速度及单位重力恢复body-xy；保留机身倾斜引起的速度投影。"""
    gravity = normalized_gravity(physical)
    gx, gy, gz = gravity.unbind(-1)
    cosine_pitch = gravity[..., 1:].norm(dim=-1).clamp_min(1e-6)
    velocity = physical[..., VELOCITY_HEADING]
    body_x = cosine_pitch * velocity[..., 0] - gx * velocity[..., 2]
    body_y = (-gx * gy / cosine_pitch * velocity[..., 0]
              - gz / cosine_pitch * velocity[..., 1] - gy * velocity[..., 2])
    return torch.stack((body_x, body_y), -1)


def body_vertical_velocity(physical: Tensor) -> Tensor:
    """由heading速度与机身重力恢复body-z速度；倾斜时不能直接取world-z。"""
    gravity = normalized_gravity(physical)
    gx, gy, gz = gravity.unbind(-1)
    # heading以机身x轴的水平投影定向，norm(g_yz)对应非负cos(pitch)。
    cosine_pitch = gravity[..., 1:].norm(dim=-1).clamp_min(1e-6)
    body_z_heading = torch.stack((-gx * gz / cosine_pitch, gy / cosine_pitch, -gz), -1)
    return (physical[..., VELOCITY_HEADING] * body_z_heading).sum(-1)
