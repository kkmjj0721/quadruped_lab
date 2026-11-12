# Copyright (c) 2026-2027 zh
"""内容：按 BlinkMimic 的速度增量和位置残差推进广义状态。

来源：BlinkMimic d264658 的 src/blinkmimic/networks.py:113-209；
与 InstantTrack b9512a9、XMimic 54860e1 的同一结构交叉核对。
子步权重采用 SuperMimic abc124c 的 (n+1)/(2n)，四子步时为 0.625。
上游版权与 MIT 许可保留于同目录 LICENSE_blinkmimic。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .reference_rotations import mm3, mv3, so3_exp, so3_log, yaw_mat


GeneralizedState = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class ResidualDynamics:
    """状态为世界系 (p, R, v, w) 与真实 (q, qd)，支持任意前置批次维。

    输出依次为旧航向系位置残差、旧机身系旋转残差、旧机身系线／角速度
    增量、关节位置残差和关节速度增量。Go2 的 12 个关节对应 36 维输出。
    """

    def __init__(self, dt: float, substeps: int):
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Dynamics dt must be positive and finite")
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("Dynamics substeps must be a positive integer")
        self.dt = float(dt)
        self.drift_weight = (substeps + 1) / (2 * substeps)

    def _drift(self, state: GeneralizedState, next_v: Tensor, next_w: Tensor, next_qd: Tensor):
        """用旧／新速度构造旧航向系平移、旧机身系旋转向量和关节漂移。"""
        _, rotation, velocity, angular_velocity, _, joint_velocity = state
        old_weight = self.dt * (1 - self.drift_weight)
        new_weight = self.dt * self.drift_weight
        return (
            mv3(yaw_mat(rotation).transpose(-1, -2), old_weight * velocity + new_weight * next_v),
            mv3(rotation.transpose(-1, -2), old_weight * angular_velocity + new_weight * next_w),
            old_weight * joint_velocity + new_weight * next_qd,
        )

    @staticmethod
    def increments(state: GeneralizedState, next_state: GeneralizedState) -> Tensor:
        """普通物理增量用于误差归一化，旋转取相对 SO(3) 对数。"""
        p, rotation, velocity, angular_velocity, q, qd = state
        next_p, next_rotation, next_v, next_w, next_q, next_qd = next_state
        inverse = rotation.transpose(-1, -2)
        return torch.cat((
            mv3(yaw_mat(rotation).transpose(-1, -2), next_p - p),
            so3_log(mm3(inverse, next_rotation)),
            mv3(inverse, next_v - velocity),
            mv3(inverse, next_w - angular_velocity),
            next_q - q,
            next_qd - qd,
        ), -1)

    def target(self, state: GeneralizedState, next_state: GeneralizedState) -> Tensor:
        """监督位置修正而非完整位置增量；速度增量保持原定义。"""
        joints = state[4].shape[-1]
        dp, dr, dv, dw, dq, dqd = self.increments(state, next_state).split((3, 3, 3, 3, joints, joints), -1)
        drift_p, drift_r, drift_q = self._drift(state, next_state[2], next_state[3], next_state[5])
        return torch.cat((dp - drift_p, dr - drift_r, dv, dw, dq - drift_q, dqd), -1)

    def integrate(self, state: GeneralizedState, delta: Tensor) -> GeneralizedState:
        """先更新速度，再将位置残差加到漂移中；不单独修正派生身体量。"""
        p, rotation, velocity, angular_velocity, q, qd = state
        joints = q.shape[-1]
        dp, dr, dv, dw, dq, dqd = delta.split((3, 3, 3, 3, joints, joints), -1)
        next_v = velocity + mv3(rotation, dv)
        next_w = angular_velocity + mv3(rotation, dw)
        next_qd = qd + dqd
        drift_p, drift_r, drift_q = self._drift(state, next_v, next_w, next_qd)
        next_p = p + mv3(yaw_mat(rotation), dp + drift_p)
        next_rotation = mm3(rotation, so3_exp(dr + drift_r))
        return next_p, next_rotation, next_v, next_w, q + (dq + drift_q), next_qd

    @staticmethod
    def error(predicted: GeneralizedState, target: GeneralizedState) -> Tensor:
        """误差在目标状态的局部坐标系表达，与来源 state_error 等价。"""
        return ResidualDynamics.increments(target, predicted)
