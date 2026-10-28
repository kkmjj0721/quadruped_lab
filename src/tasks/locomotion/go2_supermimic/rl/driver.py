# Copyright (c) 2026-2027 zh
"""内容：为真实采集提供外部策略驱动，包含MoE-CTS策略和历史MLP策略及追逐转向。"""

from __future__ import annotations

import math
from pathlib import Path

import torch

from ..mdp.state_spec import JOINT_NAMES

# go2_rl_gym 30e74dc 的观测缩放与45维布局：角速度、重力、命令、关节位置、关节速度、上一步动作。
_ANGULAR_SCALE = 0.25
_JOINT_VELOCITY_SCALE = 0.05
_COMMAND_SCALE = (2.0, 2.0, 0.25)
_TERM_DIMS = (3, 3, 3, 12, 12, 12)
_HISTORY_LENGTH = 5
_PRETRAINED_ACTION_SCALE = 0.25
# 预训练策略的默认关节角与其训练配置一致，与本项目 init_state 不同，必须分别保留。
# 顺序必须与 _joint_order() 同为逐腿的髋、大腿、小腿。
_PRETRAINED_HIP = {"FL": 0.1, "FR": -0.1, "RL": 0.1, "RR": -0.1}
_PRETRAINED_THIGH = {"FL": 0.8, "FR": 0.8, "RL": 1.0, "RR": 1.0}
_PRETRAINED_CALF = -1.5
_PRETRAINED_DEFAULT = tuple(value for leg in ("FL", "FR", "RL", "RR")
                            for value in (_PRETRAINED_HIP[leg], _PRETRAINED_THIGH[leg], _PRETRAINED_CALF))


def _joint_order():
    """预训练策略的关节顺序为四腿乘髋、大腿、小腿，与本项目一致。"""
    return [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]


class MoeCtsDriver:
    """把预训练MoE-CTS策略当作地形数据来源，输出本项目的未缩放动作。

    导出文件自带 batch=1 环形历史，无法批量驱动；这里绕过其 forward，直接调用
    子模块并自行维护批量历史，避免逐环境拷贝策略状态。
    """

    def __init__(self, path, num_envs, default_joint_pos, action_scales, device):
        # ScriptModule 不支持 requires_grad_；torch.jit.load 的推理模块本身不累计梯度。
        policy = torch.jit.load(str(Path(path)), map_location=device).eval()
        self.policy, self.device, self.num_envs = policy, torch.device(device), num_envs
        width = sum(_TERM_DIMS)
        self.history = torch.zeros(num_envs, _HISTORY_LENGTH, width, device=device)
        # 预训练策略接受的是它自己的原始输出动作，不是本项目换算后的动作，必须单独保存。
        self.last = torch.zeros(num_envs, len(_joint_order()), device=device)
        order = _joint_order()
        self.index = [JOINT_NAMES.index(name) for name in order]
        self.pretrained_default = torch.tensor(_PRETRAINED_DEFAULT, device=device)
        self.our_default = default_joint_pos.detach().reshape(-1)[self.index].clone()
        self.our_scale = torch.tensor([action_scales[name] for _ in range(4) for name in ("hip", "thigh", "calf")],
                                      device=device)
        self.command_scale = torch.tensor(_COMMAND_SCALE, device=device)
        # 预训练导出的 normalizer 是恒等映射，但仍保留调用以避免假设未来导出不变。
        self.normalizer = policy.normalizer

    def _ordered(self, values):
        return values.index_select(-1, torch.tensor(self.index, device=self.device))

    @torch.no_grad()
    def act(self, observation):
        """observation 为本项目45维本体观测，顺序角速度、重力、关节位置、关节速度、动作、命令。"""
        angular, gravity = observation[:, 0:3], observation[:, 3:6]
        joint_pos, joint_vel = observation[:, 6:18], observation[:, 18:30]
        command = observation[:, 42:45]
        # 本项目关节位置相对自身默认角；预训练策略期望相对它自己的默认角。
        relative_joint_pos = self._ordered(joint_pos) + self.our_default - self.pretrained_default
        frame = torch.cat((angular * _ANGULAR_SCALE, gravity, command * self.command_scale, relative_joint_pos,
                           self._ordered(joint_vel) * _JOINT_VELOCITY_SCALE, self.last), -1)
        self.history = torch.cat((self.history[:, 1:], frame.unsqueeze(1)), dim=1)
        history = self.history.flatten(1)
        history, frame = self.normalizer(history), self.normalizer(frame)
        # MoE-CTS 编码器忽略第二路无命令输入，但导出签名要求显式提供。
        latent, _ = self.policy.student_moe_encoder(history, history)
        raw = self.policy.actor(torch.cat((latent, frame), -1))
        self.last = raw
        target = self.pretrained_default + _PRETRAINED_ACTION_SCALE * raw
        return (target - self.our_default) / self.our_scale

    @torch.no_grad()
    def reset(self, done):
        """回合结束时清空历史，使新回合从当前帧重新累积。"""
        if done.any():
            self.history[done] = 0
            self.last[done] = 0


class ActorDriver:
    """把已有盲速度策略作为数据来源，保存专家自己的历史而不参与策略更新。"""

    def __init__(self, path, observation_dim, action_dim, device):
        from .deployment import checkpoint_actor

        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        self.actor, _ = checkpoint_actor(checkpoint)
        self.actor = self.actor.to(device)
        if self.actor.observation_dim != observation_dim or self.actor.action_dim != action_dim:
            raise ValueError("Expert policy observation and action dimensions differ from the task")
        self.history = None
        self.fresh = None

    @torch.no_grad()
    def act(self, observation):
        if self.history is None:
            self.history = observation[:, None].repeat(1, self.actor.history_length, 1)
            self.fresh = torch.zeros(observation.shape[0], dtype=torch.bool, device=observation.device)
        else:
            history = torch.cat((self.history[:, 1:], observation[:, None]), 1)
            self.history = torch.where(self.fresh[:, None, None], observation[:, None].expand_as(history), history)
            self.fresh.zero_()
        return self.actor(self.history)

    def reset(self, done):
        """专家回合结束后，用新回合首帧填满其历史。"""
        if self.fresh is not None:
            self.fresh.logical_or_(done)


class ChaseSteer:
    """把追逐目标转成速度命令，让外部策略在场地内持续产生行进数据。"""

    def __init__(self, num_envs, device, speed_range, radius):
        self.device, self.speed_range, self.radius = torch.device(device), speed_range, radius
        self.speed = torch.zeros(num_envs, device=device)
        self.goal = torch.zeros(num_envs, 2, device=device)
        self.timer = torch.full((num_envs,), 2.0, device=device)

    @torch.no_grad()
    def resample(self, mask, center):
        count = int(mask.sum())
        if not count:
            return
        low, high = self.speed_range
        self.speed[mask] = low + (high - low) * torch.rand(count, device=self.device)
        angle = 2 * math.pi * torch.rand(count, device=self.device)
        distance = self.radius * torch.sqrt(torch.rand(count, device=self.device))
        self.goal[mask] = center[mask] + torch.stack((angle.cos(), angle.sin()), -1) * distance[:, None]
        self.timer[mask] = 2 + torch.rand(count, device=self.device)

    @torch.no_grad()
    def __call__(self, dt, position, yaw, center):
        """返回 [线速度, 0, 偏航角速度]；朝目标误差超过90度时不再前进。"""
        self.timer -= dt
        self.resample(self.timer <= 0, center)
        direction = self.goal - position
        distance = direction.norm(dim=-1)
        error = torch.atan2(direction[:, 1], direction[:, 0]) - yaw
        error = torch.atan2(error.sin(), error.cos())
        speed = torch.minimum(self.speed, 1.5 * distance)
        return torch.stack((speed * error.cos().clamp(min=0), torch.zeros_like(speed),
                            (1.5 * error).clamp(-1.5, 1.5)), -1)

    def reset(self, done, center):
        """回合结束时重采样目标与速度。"""
        self.resample(done, center)
