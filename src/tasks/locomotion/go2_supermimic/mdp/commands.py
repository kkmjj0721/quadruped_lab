# Copyright (c) 2026-2027 zh
"""内容：实现HIM速度采样与heading反馈，真实和想象使用同一命令状态。"""

import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from .state_spec import COMMAND_OVERRIDE, COMMAND_TIMER, FORWARD, HEADING_TARGET, STANDING


def sample_commands(cfg, context, yaw):
    """保留来源先forward覆盖、后heading反馈、最终standing清零的顺序。"""
    bounds = context[:, 19:25].reshape(-1, 3, 2)
    random = torch.rand_like(bounds[..., 0])
    command = bounds[..., 0] + random * (bounds[..., 1] - bounds[..., 0])
    heading = cfg.heading_range[0] + torch.rand_like(yaw) * (cfg.heading_range[1] - cfg.heading_range[0])
    standing = torch.rand_like(yaw) <= cfg.rel_standing_envs
    forward = torch.rand_like(yaw) <= cfg.rel_forward_envs
    command = torch.stack((torch.where(forward, command[:, 0].abs().clamp_min(0.3), command[:, 0]),
                           torch.where(forward, torch.zeros_like(yaw), command[:, 1]), command[:, 2]), -1)
    lower, upper = cfg.resampling_time
    timer = lower + (upper - lower) * torch.rand_like(yaw)
    return command, timer, heading, standing, forward


def heading_command(cfg, command, yaw, heading, standing):
    """heading比例固定为1；前进样本的wz同样会被反馈覆盖。"""
    error = wrap_to_pi(heading - yaw)
    angular = (cfg.heading_control_stiffness * error).clamp(*cfg.initial_ranges[2])
    result = torch.cat((command[:, :2], angular[:, None]), -1)
    return torch.where(standing[:, None].bool(), torch.zeros_like(result), result)


def advance_commands(cfg, state, physical, dt):
    """在终点yaw刷新下一决策命令；固定覆盖命令不受heading重写。"""
    memory = state.memory
    sampled, timer, heading, standing, forward = sample_commands(cfg, state.context, physical[:, 72])
    expired = memory[:, COMMAND_TIMER] - dt <= 0.0
    command = torch.where(expired[:, None], sampled, state.command)
    timer = torch.where(expired, timer, memory[:, COMMAND_TIMER] - dt)
    heading = torch.where(expired, heading, memory[:, HEADING_TARGET])
    standing = torch.where(expired, standing, memory[:, STANDING].bool())
    forward = torch.where(expired, forward, memory[:, FORWARD].bool())
    command = heading_command(cfg, command, physical[:, 72], heading, standing)
    command = torch.where(memory[:, COMMAND_OVERRIDE, None].bool(), state.command, command)
    return command, timer, heading, standing, forward
