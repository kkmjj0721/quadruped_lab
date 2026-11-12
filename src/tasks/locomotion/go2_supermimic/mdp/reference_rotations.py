# Copyright (c) 2026-2027 zh
"""内容：迁入 BlinkMimic d264658 的批量旋转工具，四元数采用 wxyz。

来源：src/blinkmimic/rotations.py；原作者及 MIT 许可见同目录 LICENSE_blinkmimic。
保留参考实现的小角度展开、四元数分支及航向正则项。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mm3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """批量 3×3 矩阵乘法，沿用参考实现默认的 matmul 路径。"""
    return A @ B


def mv3(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """批量矩阵乘列向量，支持前置维度广播。"""
    return (A @ v.unsqueeze(-1)).squeeze(-1)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """将归一化后的 wxyz 四元数变为作用于列向量的旋转矩阵。"""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ), dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """以最大分量候选恢复单位四元数，并固定 w 非负。"""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    candidates = torch.stack(
        (
            torch.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            torch.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            torch.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            torch.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ), dim=-2,
    )
    index = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(candidates, -2, index[..., None, None].expand(index.shape + (1, 4))).squeeze(-2)
    q = F.normalize(q, dim=-1)
    return q * torch.where(q[..., :1] < 0, -1.0, 1.0)


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """固定单位轴与转角的 Rodrigues 旋转，供单轴关节 FK 使用。"""
    x, y, z = axis.unbind(-1)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    complement = 1 - cosine
    return torch.stack(
        (
            cosine + x * x * complement, x * y * complement - z * sine, x * z * complement + y * sine,
            y * x * complement + z * sine, cosine + y * y * complement, y * z * complement - x * sine,
            z * x * complement - y * sine, z * y * complement + x * sine, cosine + z * z * complement,
        ), dim=-1,
    ).reshape(angle.shape + (3, 3))


def skew(v: torch.Tensor) -> torch.Tensor:
    """构造满足 skew(v) @ x = v × x 的反对称矩阵。"""
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(v.shape[:-1] + (3, 3))


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """旋转向量到旋转矩阵，零点采用参考实现的小角度展开。"""
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    K = skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * mm3(K, K)


def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """四元数到最短旋转向量，保持来源的零转角极限。"""
    q = q * torch.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = torch.linalg.norm(v, dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., :1])
    scale = torch.where(sin_half < 1e-6, 2.0 / torch.clamp(q[..., :1], min=1e-6), 2 * half / (sin_half + 1e-12))
    return v * scale


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """经四元数计算旋转矩阵的对数，覆盖零角和接近 pi 的情况。"""
    return quat_to_rotvec(mat_to_quat(R))


def yaw_mat(R: torch.Tensor) -> torch.Tensor:
    """由机身 x 轴的水平投影得到航向旋转，保留来源的 1e-9 正则项。"""
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    norm = torch.sqrt(fx * fx + fy * fy + 1e-9)
    cosine, sine = fx / norm, fy / norm
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    return torch.stack((cosine, -sine, zero, sine, cosine, zero, zero, zero, one), -1).reshape(R.shape[:-2] + (3, 3))
