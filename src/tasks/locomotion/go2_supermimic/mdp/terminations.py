# Copyright (c) 2026-2027 zh
"""内容：按相对地面高度、机身倾斜和地形支持域判断Go2真实失败。"""

import math

from torch import Tensor

from agents.supermimic.interfaces import TaskState

from .state_spec import QUATERNION, projected_gravity


def hard_termination(task, state: TaskState) -> Tensor:
    """人工超时由原生时间项单独处理，不纳入物理失败标签。"""
    position = task.world_position(state)
    supported = task.surface.contains(position[:, :2], margin=task.cfg.terrain.boundary_margin_m)
    height = position[:, 2] - task.surface_height(position[:, :2])
    gravity = projected_gravity(state.physical[:, QUATERNION])
    fallen = (height < task.cfg.reward.fall_height) | (gravity[:, 2] > -math.cos(task.cfg.reward.fall_angle))
    return fallen | ~supported
