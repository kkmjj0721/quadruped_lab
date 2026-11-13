# Copyright (c) 2026-2027 zh
"""内容：保留 MGDP f43139d 的奖励前向，并为想象接触事件提供显式代理梯度。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


MIX = 0
PARKOUR = 1
FOOT_ORDER = ("FL", "FR", "RL", "RR")
BODY_FORCE_NAMES = (
    "base_link", "FL_thigh", "FL_calf", "FR_thigh", "FR_calf",
    "RL_thigh", "RL_calf", "RR_thigh", "RR_calf",
)
REWARD_CONTEXT_NAMES = ("family", "terrain_class", "level", "training_step")
REWARD_SCALES = {
    "action_rate": -0.01,
    "ang_vel_xy": -0.05,
    "collision": -1.0,
    "dof_acc": -2.5e-7,
    "dof_pos_limits": -10.0,
    "feet_air_time": 1.0,
    "feet_edge": -1.0,
    "feet_stumble": -1.0,
    "lin_vel_z": -1.0,
    "motion_trot": -0.1,
    "orientation": -0.2,
    "stand_still": -0.1,
    "torques": -1e-5,
    "tracking_ang_vel": 0.5,
    "tracking_lin_vel": 1.0,
}


@dataclass(frozen=True)
class RewardGradientCfg:
    """本次一阶段适配的代理温度，单位均为 N；不属于官方配置。"""

    contact_temperature: float = 0.1
    collision_temperature: float = 0.01
    stumble_temperature: float = 1.0


@dataclass
class RewardInputs:
    """关节按 FL/FR/RL/RR，每腿 hip/thigh/calf；力均为世界坐标净力。

    body_forces 为 BODY_FORCE_NAMES 对应的九个部件，calf 不包含 foot 几何体。
    joint_limits 是已经乘过 0.9 软限位系数的 [12, 2] 范围。
    foot_edge 由地形模块查询，想象路径的空间代理梯度也由该查询负责。
    """

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    previous_joint_vel: torch.Tensor
    body_linear_velocity: torch.Tensor
    body_angular_velocity: torch.Tensor
    projected_gravity: torch.Tensor
    command: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    torques: torch.Tensor
    foot_forces: torch.Tensor
    body_forces: torch.Tensor
    foot_edge: torch.Tensor
    nominal_joint_pos: torch.Tensor
    joint_limits: torch.Tensor


@dataclass
class RewardContext:
    """每项为 [B]；family 使用 MIX/PARKOUR，training_step 为全局控制步。"""

    family: torch.Tensor
    terrain_class: torch.Tensor
    level: torch.Tensor
    training_step: torch.Tensor


@dataclass
class RewardMemory:
    """[B,4] 事件记忆；重置 air_time，但按官方保留 last_contact。"""

    air_time: torch.Tensor
    last_contact: torch.Tensor


def _indicator(value: torch.Tensor, temperature: float, imagined: bool) -> torch.Tensor:
    hard = (value > 0).to(value.dtype)
    if not imagined:
        return hard
    soft = torch.sigmoid(value / temperature)
    return hard + (soft - soft.detach())


def _class_scale(context: RewardContext, mix: dict[int, float], parkour: dict[int, float], dtype: torch.dtype) -> torch.Tensor:
    scale = torch.ones_like(context.level, dtype=dtype)
    for family, mapping in ((MIX, mix), (PARKOUR, parkour)):
        for terrain_class, value in mapping.items():
            scale = torch.where((context.family == family) & (context.terrain_class == terrain_class), value, scale)
    return scale


def compute_rewards(
    inputs: RewardInputs,
    context: RewardContext,
    memory: RewardMemory,
    *,
    imagined: bool = False,
    dt: float = 0.02,
    gradients: RewardGradientCfg = RewardGradientCfg(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor], RewardMemory]:
    """返回裁剪后的总奖励、已乘权重及 dt 的分项和下一步事件记忆。

    对照 random_dog.py:803–1052、base/legged_robot.py:226–243,1015–1101。
    想象中的阈值事件与总奖励截零反向使用 STE；全部前向保留官方行为。
    """
    q, qd = inputs.joint_pos, inputs.joint_vel
    command_xy = inputs.command[:, :2]
    command_speed = torch.linalg.vector_norm(command_xy, dim=-1)
    parkour = context.family == PARKOUR
    dtype = q.dtype

    # 官方 class_to_dict 按字母序调用：air_time 先更新 last_contact，edge 再读取。
    contact = _indicator(inputs.foot_forces[..., 2] - 1.0, gradients.contact_temperature, imagined)
    previous_contact = memory.last_contact.to(dtype)
    contact_filtered = contact + previous_contact - contact * previous_contact
    first_contact = (memory.air_time > 0).to(dtype) * contact_filtered
    elapsed_air_time = memory.air_time + dt
    air_time_reward = ((elapsed_air_time - 0.5) * first_contact).sum(-1) * (command_speed > 0.1)
    next_memory = RewardMemory(elapsed_air_time * (1.0 - contact_filtered), contact)

    collision_contact = _indicator(torch.linalg.vector_norm(inputs.body_forces, dim=-1) - 0.1, gradients.collision_temperature, imagined)
    collision = torch.where(parkour, collision_contact[:, 0], collision_contact.sum(-1))
    collision *= _class_scale(context, {5: 0, 6: 0}, {1: 0.2, 2: 0.2, 3: 0.1, 13: 0.2, 14: 10, 15: 10, 16: 0.2, 17: 0}, dtype)

    stumble_margin = torch.linalg.vector_norm(inputs.foot_forces[..., :2], dim=-1) - 4.0 * inputs.foot_forces[..., 2].abs()
    stumble_contact = _indicator(stumble_margin, gradients.stumble_temperature, imagined)
    stumble = 1.0 - (1.0 - stumble_contact).prod(-1)

    trot = (q[:, :3] - q[:, 9:12]).abs().sum(-1) + (q[:, 3:6] - q[:, 6:9]).abs().sum(-1)
    trot *= (context.level <= torch.where(parkour, 1, 3)) & (context.training_step <= 20000)
    trot *= _class_scale(context, {5: 0, 6: 0}, {3: 0.1, 13: 0, 14: 0, 15: 0}, dtype)

    # 超过指令同向 0.1 m/s 后误差保持平台，这是官方实际速度跟踪公式。
    lower = torch.where(command_xy > 0, -1e5, command_xy - 0.1)
    upper = torch.where(command_xy < 0, 1e5, command_xy + 0.1)
    clipped_velocity = torch.clamp(inputs.body_linear_velocity[:, :2], min=lower, max=upper)
    edge = (contact * inputs.foot_edge).sum(-1) * (context.level > 0) * parkour
    edge *= _class_scale(context, {}, {0: 0, 1: 0, 2: 0, 15: 0, 16: 0, 17: 0}, dtype)

    raw = {
        "action_rate": (inputs.previous_action - inputs.action).square().sum(-1),
        "ang_vel_xy": inputs.body_angular_velocity[:, :2].square().sum(-1),
        "collision": collision,
        "dof_acc": ((inputs.previous_joint_vel - qd) / dt).square().sum(-1),
        "dof_pos_limits": ((inputs.joint_limits[:, 0] - q).clamp_min(0) + (q - inputs.joint_limits[:, 1]).clamp_min(0)).sum(-1) * parkour,
        "feet_air_time": air_time_reward,
        "feet_edge": edge,
        "feet_stumble": stumble,
        "lin_vel_z": inputs.body_linear_velocity[:, 2].square() * _class_scale(context, {5: 0.25, 6: 0.25}, {1: 0.25, 2: 0.25, 3: 0.05, 4: 0.05, 7: 0.05, 13: 0.1, 15: 0.01}, dtype),
        "motion_trot": trot,
        "orientation": inputs.projected_gravity[:, :2].square().sum(-1) * _class_scale(context, {5: 0.25}, {1: 0.25, 2: 0.25, 13: 0.01, 15: 0.01, 17: 0}, dtype),
        "stand_still": (q - inputs.nominal_joint_pos).abs().sum(-1) * (command_speed < 0.1),
        "torques": inputs.torques.square().sum(-1),
        "tracking_ang_vel": torch.exp(-(inputs.command[:, 2] - inputs.body_angular_velocity[:, 2]).square() / 0.25),
        "tracking_lin_vel": torch.exp(-(command_xy - clipped_velocity).square().sum(-1) / 0.25),
    }
    weighted = {name: raw[name] * (scale * dt) for name, scale in REWARD_SCALES.items()}
    # 保留官方逐项累加顺序和 only_positive_rewards；无额外死亡惩罚。
    total = torch.zeros_like(command_speed)
    for term in weighted.values():
        total = total + term
    positive = total.clamp_min(0)
    if imagined:
        # 保留官方截零前向；想象以截零前总奖励的梯度提供脱离负奖励区的方向。
        positive = positive.detach() + (total - total.detach())
    return positive, weighted, next_memory
