# Copyright (c) 2026-2027 zh
"""内容：移植 MGDP f43139d 的静态动力学采样、全局子步动作延迟和推扰。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


DYNAMICS_CONTEXT_NAMES = (
    "base_mass_delta", "link_mass_scale_0", "link_mass_scale_1", "link_mass_scale_2",
    "com_x", "com_y", "friction", "motor_strength", "motor_offset",
)


@dataclass
class DynamicsSamples:
    """创建环境时采样，重置回合时保留；所有标量为 [B,1]。"""

    base_mass_delta: torch.Tensor
    link_mass_scale: torch.Tensor
    com_xy: torch.Tensor
    friction: torch.Tensor
    motor_strength: torch.Tensor
    motor_offset: torch.Tensor

    def context(self) -> torch.Tensor:
        """按 DYNAMICS_CONTEXT_NAMES 生成 [B,9] 模型条件。"""
        return torch.cat((self.base_mass_delta, self.link_mass_scale, self.com_xy,
                          self.friction, self.motor_strength, self.motor_offset), dim=-1)


def sample_dynamics(num_envs: int, device: str | torch.device, generator: torch.Generator | None = None) -> DynamicsSamples:
    """分布对齐官方；不声称与上游 NumPy/PyTorch 混合随机序列逐种子相同。"""
    bucket_ids = torch.randint(64, (num_envs,), device=device, generator=generator)
    friction_buckets = torch.rand(64, 1, device=device, generator=generator) + 0.2
    return DynamicsSamples(
        base_mass_delta=3.0 * torch.rand(num_envs, 1, device=device, generator=generator) - 1.0,
        link_mass_scale=0.4 * torch.rand(num_envs, 3, device=device, generator=generator) + 0.8,
        com_xy=0.1 * torch.rand(num_envs, 2, device=device, generator=generator) - 0.05,
        friction=friction_buckets[bucket_ids],
        motor_strength=0.2 * torch.rand(num_envs, 1, device=device, generator=generator) + 0.9,
        motor_offset=0.04 * torch.rand(num_envs, 1, device=device, generator=generator) - 0.02,
    )


@requires_model_fields("body_mass", "body_inertia", "body_ipos", "geom_friction", recompute=RecomputeLevel.set_const)
def apply_mgdp_dynamics(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    *,
    link_names: tuple[str, str, str],
    scale_inertia: bool,
    base_name: str = "base_link",
    entity_name: str = "robot",
) -> None:
    """仅供 startup 使用，并把采样结果保存在 env.mgdp_dynamics。

    官方仅随机 props[1:4]；调用方必须显式提供对应三段名称。URDF 声明
    文本顺序不保证导入顺序；Preview4实际导入已确认是 FL 三段。
    按质量比例缩放惯量是获准的 MuJoCo 适配，并非 PhysX 数值等价声明。
    恢复系数虽在官方配置打开，实际函数没有应用，因此这里也不随机化。
    """
    if env_ids is not None:
        raise ValueError("MGDP dynamics must run once at startup for all environments.")
    robot = env.scene[entity_name]
    body_ids, _ = robot.find_bodies((base_name, *link_names), preserve_order=True)
    if len(body_ids) != 4 or len(set(body_ids)) != 4:
        raise ValueError("MGDP dynamics requires one base and three distinct link bodies.")
    global_ids = robot.indexing.body_ids[body_ids]
    samples = sample_dynamics(env.num_envs, env.device)
    model = env.sim.model
    original_mass = model.body_mass[:, global_ids].clone()
    randomized_mass = original_mass.clone()
    randomized_mass[:, 0:1] += samples.base_mass_delta
    randomized_mass[:, 1:4] *= samples.link_mass_scale
    model.body_mass[:, global_ids] = randomized_mass
    if scale_inertia:
        model.body_inertia[:, global_ids] *= (randomized_mass / original_mass).unsqueeze(-1)
    # 上游是绝对替换 x/y，z 不变，不能改成三轴加性扰动。
    model.body_ipos[:, global_ids[0], :2] = samples.com_xy
    model.geom_friction[:, robot.indexing.geom_ids, 0] = samples.friction
    env.mgdp_dynamics = samples


def make_mgdp_dynamics_event(*, link_names: tuple[str, str, str], scale_inertia: bool, base_name: str = "base_link", entity_name: str = "robot") -> EventTermCfg:
    """创建仅执行一次的静态随机化事件。"""
    return EventTermCfg(func=apply_mgdp_dynamics, mode="startup", params={
        "link_names": link_names, "scale_inertia": scale_inertia,
        "base_name": base_name, "entity_name": entity_name,
    })


@dataclass(kw_only=True)
class MGDPJointActionCfg(BaseActionCfg):
    """配合 BuiltinMotorActuatorCfg 使用，直接写入子步 PD 扭矩。"""

    scale: float = 0.25
    kp: float = 30.0
    kd: float = 0.8
    effort_limit: float = 23.7
    decimation: int = 4
    latency_seed: int = 0

    def build(self, env: ManagerBasedRlEnv) -> MGDPJointAction:
        return MGDPJointAction(self, env)


class MGDPJointAction(BaseAction):
    """每控制步共享一个 k∈[0,4]，前 k 个物理子步使用上一动作。

    last_torques 是最后一次写入、积分之前的裁剪后扭矩，供官方奖励使用。
    target_names/target_ids 沿用 BaseAction，供环境恢复 FL/FR/RL/RR 顺序。
    """

    cfg: MGDPJointActionCfg

    def __init__(self, cfg: MGDPJointActionCfg, env: ManagerBasedRlEnv):
        if cfg.decimation != 4 or cfg.scale != 0.25 or cfg.offset != 0.0 or cfg.clip is not None:
            raise ValueError("MGDP action requires decimation=4, scale=0.25, offset=0 and clip=None.")
        super().__init__(cfg, env)
        self.previous_action = torch.zeros_like(self._raw_actions)
        self.last_torques = torch.zeros_like(self._raw_actions)
        self.latency_steps = 0
        self._pending_latency: int | None = None
        self._substep = 0
        self._latency_rng = random.Random(cfg.latency_seed)
        self._nominal = self._entity.data.default_joint_pos[:, self._target_ids]

    def set_next_latency(self, latency_steps: int) -> None:
        """由适配器预先采样同一个全局 k，用于记录决策时的模型条件。"""
        if not isinstance(latency_steps, int) or not 0 <= latency_steps <= self.cfg.decimation:
            raise ValueError("MGDP latency must be an integer from 0 through 4.")
        self._pending_latency = latency_steps

    def process_actions(self, actions: torch.Tensor) -> None:
        self.previous_action.copy_(self._raw_actions)
        self._raw_actions.copy_(actions.clamp(-100.0, 100.0))
        self.latency_steps = self._latency_rng.randint(0, self.cfg.decimation) if self._pending_latency is None else self._pending_latency
        self._pending_latency = None
        self._substep = 0

    def apply_actions(self) -> None:
        selected = self.previous_action if self._substep < self.latency_steps else self._raw_actions
        samples = self._env.mgdp_dynamics
        q = self._entity.data.joint_pos[:, self._target_ids]
        qd = self._entity.data.joint_vel[:, self._target_ids]
        self._processed_actions = self._nominal + self.cfg.scale * selected + samples.motor_offset
        torque = samples.motor_strength * (self.cfg.kp * (self._processed_actions - q) - self.cfg.kd * qd)
        self.last_torques.copy_(torque.clamp(-self.cfg.effort_limit, self.cfg.effort_limit))
        self._entity.set_joint_effort_target(self.last_torques, joint_ids=self._target_ids)
        self._substep += 1

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0
        self.previous_action[ids] = 0
        self.last_torques[ids] = 0


def apply_mgdp_push(
    env: ManagerBasedRlEnv,
    control_step: int,
    *,
    generator: torch.Generator | None = None,
    entity_name: str = "robot",
) -> torch.Tensor | None:
    """在完成物理积分后调用；每 750 步的第 1 步替换世界系 vxy。"""
    if control_step % 750 != 1:
        return None
    robot = env.scene[entity_name]
    velocity = robot.data.root_link_vel_w.clone()
    velocity[:, :2] = 2.0 * torch.rand(env.num_envs, 2, device=env.device, generator=generator) - 1.0
    robot.write_root_link_velocity_to_sim(velocity)
    return velocity[:, :2].clone()
