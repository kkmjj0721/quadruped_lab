# Copyright (c) 2026-2027 zh
"""
内容：
    从Go2真实仿真采集RI-4438 HIM的接触、力矩和角动量奖励及四足计时。
    公式来源为legged_wbc_mjlab提交e6209ec3d910b57673d5cbba1d3ca798a7fa3113的
    ri_4438_him/mdp/rewards.py和其使用的MjLab 1.6.0奖励；保留参考门控与子步语义。
"""

from __future__ import annotations

import torch
from torch import Tensor

from agents.supermimic.interfaces import TaskState

from .state_spec import FOOT_NAMES, PHASE_CLOCK, SENSOR_REWARD_NAMES


def feet_gait(contact_time: Tensor, phase: Tensor, offset, stance_fraction: float, active: Tensor) -> Tensor:
    """以真实接触持续时间匹配四足期望接触模式，逐足平均。"""
    offsets = phase.new_tensor(offset).view(1, -1)
    desired = (phase[:, None] + offsets).remainder(1.0) < stance_fraction
    return (desired == (contact_time > 0)).float().mean(-1) * active


def feet_air_time(air_time: Tensor, contact_time: Tensor, phase: Tensor, offset,
                  period: float, stance_fraction: float, active: Tensor) -> Tensor:
    """复制参考的逐足归一化支撑／腾空时长得分，不要求恰好两脚接触。"""
    contact = contact_time > 0
    desired = (phase[:, None] + phase.new_tensor(offset).view(1, -1)).remainder(1.0) < stance_fraction
    mode_time = torch.where(contact, contact_time, air_time)
    target = torch.where(contact, period * stance_fraction, period * (1 - stance_fraction))
    duration_score = (1 - (mode_time / target - 1).abs()).clamp_min(0)
    return (duration_score * (contact == desired)).mean(-1) * active


def _collision_cost(sensor_data, threshold: float, *, count_substeps: bool) -> Tensor:
    """腿碰地取历史任一命中，自碰撞累计发生命中的物理子步。"""
    history = sensor_data.force_history
    if history is not None:
        hit = (history.norm(dim=-1) > threshold).any(dim=1)
        return hit.sum(-1).float() if count_substeps else hit.any(-1).float()
    if sensor_data.found is None:
        raise ValueError("Collision reward requires force history or contact flags")
    found = sensor_data.found
    return found.sum(-1).float() if count_substeps else found.bool().any(-1).float()


def collect_reward_auxiliary(env, before: TaskState, final: TaskState, task) -> Tensor:
    """返回10项真实原始率、四足腾空时长、四足接触时长，顺序与任务记忆一致。"""
    cfg = task.cfg.reward
    robot = env.scene["robot"]
    feet_sensor = env.scene["feet_ground_contact"]
    feet = feet_sensor.data
    if feet.current_air_time is None or feet.current_contact_time is None or feet.force is None or feet.found is None:
        raise ValueError("HIM rewards require foot force, contact flags and tracked contact/air time")
    air_time, contact_time = feet.current_air_time, feet.current_contact_time
    if air_time.shape != (before.command.shape[0], 4) or contact_time.shape != air_time.shape:
        raise ValueError("HIM foot reward sensors must follow the four-foot task order")
    site_ids, site_names = robot.find_sites(FOOT_NAMES, preserve_order=True)
    if tuple(site_names) != FOOT_NAMES:
        raise ValueError("HIM foot reward site order must be FL, FR, RL, RR")
    active = before.command[:, :2].norm(dim=-1) + before.command[:, 2].abs() > cfg.command_threshold
    # 用控制步计数先取余，避免累加秒数在整周期边界落到错误的一侧。
    steps_per_period = cfg.gait_period / env.step_dt
    phase = (final.memory[:, PHASE_CLOCK] / env.step_dt).round().remainder(steps_per_period) / steps_per_period
    in_contact = (feet.found > 0).float()
    foot_velocity = robot.data.site_lin_vel_w[:, site_ids, :2]
    speed = foot_velocity.norm(dim=-1)
    force_magnitude = feet.force.norm(dim=-1)
    first_contact = feet_sensor.compute_first_contact(dt=env.step_dt)
    landing_force = force_magnitude * first_contact.float()
    terms = {
        "angular_momentum": env.scene["robot/root_angmom"].data.square().sum(-1),
        "joint_acc_l2": robot.data.joint_acc.square().sum(-1),
        "joint_torques_l2": robot.data.actuator_force.square().sum(-1),
        "foot_gait": feet_gait(contact_time, phase, cfg.gait_offset, cfg.gait_stance_fraction, active),
        "feet_air_time": feet_air_time(air_time, contact_time, phase, cfg.gait_offset,
                                        cfg.gait_period, cfg.gait_stance_fraction, active),
        "foot_slip": (speed.square() * in_contact).sum(-1) * active,
        "soft_landing": landing_force.sum(-1) * active,
        "leg_collision": _collision_cost(env.scene["leg_ground_contact"].data, cfg.collision_force_threshold,
                                          count_substeps=False),
        "self_collision": _collision_cost(env.scene["self_collision_contact"].data, cfg.collision_force_threshold,
                                           count_substeps=True),
        "stumble": (feet.force[..., :2].norm(dim=-1) > cfg.stumble_ratio * feet.force[..., 2].abs()).any(-1).float(),
    }
    diagnostics = env.extras.setdefault("log", {})
    diagnostics["Metrics/angular_momentum_mean"] = terms["angular_momentum"].sqrt().mean()
    diagnostics["Metrics/slip_velocity_mean"] = (speed * in_contact).sum() / in_contact.sum().clamp_min(1)
    diagnostics["Metrics/landing_force_mean"] = landing_force.sum() / first_contact.float().sum().clamp_min(1)
    return torch.cat((torch.stack([terms[name] for name in SENSOR_REWARD_NAMES], -1), air_time, contact_time), -1)
