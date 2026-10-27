# Copyright (c) 2026-2027 zh
"""内容：按项目Go2 MJCF的关节轴与连杆尺寸计算四足site位置。"""

from __future__ import annotations

import torch
from torch import Tensor

from .state_spec import quat_apply


FOOT_RADIUS = 0.022
HIP_X = 0.1934
HIP_Y = 0.0465
LATERAL_LINK = 0.0955
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213


def foot_positions_body(joint_position: Tensor) -> Tensor:
    """关节及输出均按FL、FR、RL、RR排列，输出为球形足site中心。"""
    if joint_position.shape[-1] != 12:
        raise ValueError("Go2 kinematics requires twelve ordered joints")
    angles = joint_position.reshape(*joint_position.shape[:-1], 4, 3)
    hip, thigh, calf = angles.unbind(-1)
    side = joint_position.new_tensor((1, -1, 1, -1))
    front = joint_position.new_tensor((1, 1, -1, -1))
    x = -THIGH_LENGTH * thigh.sin() - CALF_LENGTH * (thigh + calf).sin()
    y = side * LATERAL_LINK
    z = -THIGH_LENGTH * thigh.cos() - CALF_LENGTH * (thigh + calf).cos()
    rotated_y = hip.cos() * y - hip.sin() * z
    rotated_z = hip.sin() * y + hip.cos() * z
    return torch.stack((x + front * HIP_X, rotated_y + side * HIP_Y, rotated_z), dim=-1)


def foot_positions_world(position: Tensor, quaternion: Tensor, joint_position: Tensor) -> Tensor:
    body = foot_positions_body(joint_position)
    orientation = quaternion.unsqueeze(-2).expand(*body.shape[:-1], 4)
    return position.unsqueeze(-2) + quat_apply(orientation, body)
