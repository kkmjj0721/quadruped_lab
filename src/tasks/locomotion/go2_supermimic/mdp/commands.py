# Copyright (c) 2026-2027 zh
"""内容：从回放中记录的命令条件采样外生速度目标并推进任务记忆。"""

from __future__ import annotations

import torch
from torch import Tensor

from agents.supermimic.interfaces import TaskState

from .state_spec import COMMAND_TIMER, CONTEXT_RANGES, CONTEXT_ZERO_PROBABILITY, PHASE_CLOCK, PREVIOUS_ACTION, REWARD_AUXILIARY


def sample_commands(context: Tensor, resampling_time: tuple[float, float]) -> tuple[Tensor, Tensor]:
    """context的三轴范围与零命令概率在回合内固定，采样不依赖全局课程。"""
    bounds = context[:, CONTEXT_RANGES].reshape(-1, 3, 2)
    probability = context[:, CONTEXT_ZERO_PROBABILITY]
    if not torch.isfinite(bounds).all() or (bounds[..., 0] > bounds[..., 1]).any():
        raise ValueError("Invalid recorded command ranges")
    if not torch.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid recorded standing probability")
    command = bounds[..., 0] + torch.rand_like(bounds[..., 0]) * (bounds[..., 1] - bounds[..., 0])
    standing = torch.rand_like(probability) < probability
    command = torch.where(standing.unsqueeze(-1), torch.zeros_like(command), command)
    lower, upper = resampling_time
    timer = lower + (upper - lower) * torch.rand_like(probability)
    return command, timer


def advance_task(state: TaskState, action: Tensor, next_physical: Tensor, dt: float,
                 resampling_time: tuple[float, float]) -> TaskState:
    clock = state.memory[:, PHASE_CLOCK] + dt
    timer = state.memory[:, COMMAND_TIMER] - dt
    command = state.command
    expired = timer <= 1e-7
    if expired.any():
        next_command, next_timer = sample_commands(state.context[expired], resampling_time)
        rows = expired.nonzero().squeeze(-1)
        command = command.index_copy(0, rows, next_command)
        timer = timer.index_copy(0, rows, next_timer)
    # 辅助观测由真实adapter或想象预测头替换，此处只推进确定性的动作历史。
    memory = torch.cat((action, clock.unsqueeze(-1), timer.unsqueeze(-1),
                        state.memory[:, PREVIOUS_ACTION], state.memory[:, REWARD_AUXILIARY]), dim=-1)
    return TaskState(next_physical, memory, command, state.context)
