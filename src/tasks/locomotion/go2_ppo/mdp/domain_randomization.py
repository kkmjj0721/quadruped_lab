# Copyright (c) 2026-2027 zh
"""
内容：
    校验 Go2 域随机化范围，提供固定 IMU 标定误差和足端接触软硬度随机化。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_from_euler_xyz

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "validate_domain_randomization",
  "randomize_imu_errors",
  "imu_ang_vel_with_errors",
  "imu_projected_gravity_with_errors",
  "randomize_foot_contact_softness",
]


def validate_domain_randomization(domain_rand_cfg, physics_dt: float) -> None:
  """在环境分配前检查采样边界，避免无效动力学或被求解器静默裁剪。"""
  positive = {
    "base_mass_range", "link_mass_range", "kp_range", "kd_range",
    "effort_limit_range", "joint_armature_range", "external_force_duration_s",
    "external_force_cooldown_s", "push_interval_s", "foot_contact_time_constant_range",
  }
  nonnegative = {"foot_friction_range", "joint_friction_range", "joint_damping_range"}
  signed = {"encoder_bias_range", "base_com_range", "imu_gyro_bias_range", "imu_mount_angle_range", "external_force_range"}
  ranges = {name: getattr(domain_rand_cfg, name) for name in positive | nonnegative | signed}
  ranges.update({f"push_velocity_range.{axis}": bounds for axis, bounds in domain_rand_cfg.push_velocity_range.items()})
  for name, bounds in ranges.items():
    if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]:
      raise ValueError(f"Invalid domain randomization range: {name}.")
    if (name in positive and bounds[0] <= 0) or (name in nonnegative and bounds[0] < 0):
      raise ValueError(f"Invalid lower bound for domain randomization range: {name}.")
  if domain_rand_cfg.foot_contact_time_constant_range[0] < 2.0 * physics_dt:
    raise ValueError("Foot contact time constants must be >= 2 * physics_dt.")



def _validate_range(bounds: tuple[float, float], name: str) -> tuple[float, float]:
  """在采样前拒绝非有限或逆序区间。"""
  low, high = bounds
  if not all(math.isfinite(value) for value in bounds) or low > high:
    raise ValueError(f"{name} must contain finite, ordered bounds")
  return low, high


def randomize_imu_errors(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  gyro_bias_range: tuple[float, float],
  mount_angle_range: tuple[float, float],
) -> None:
  """为各环境采样固定安装误差和报告坐标系中的陀螺偏置。

  安装四元数采用 wxyz，表示报告坐标系相对于标称 IMU 坐标系的朝向。
  该事件在 startup 执行，既不修改机器人真值，也不随回合重置重采样。
  """
  gyro_bounds = _validate_range(gyro_bias_range, "Gyro bias range")
  angle_bounds = _validate_range(mount_angle_range, "IMU mount angle range")
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  if not hasattr(env, "_go2_imu_mount_quat"):
    env._go2_imu_mount_quat = torch.zeros((env.num_envs, 4), device=env.device)
    env._go2_imu_mount_quat[:, 0] = 1.0
    env._go2_imu_gyro_bias = torch.zeros((env.num_envs, 3), device=env.device)
  angles = torch.empty((len(env_ids), 3), device=env.device).uniform_(*angle_bounds)
  env._go2_imu_mount_quat[env_ids] = quat_from_euler_xyz(
    angles[:, 0], angles[:, 1], angles[:, 2]
  )
  env._go2_imu_gyro_bias[env_ids] = torch.empty_like(angles).uniform_(*gyro_bounds)


def imu_ang_vel_with_errors(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """将 actor 陀螺读数转入报告坐标系，再施加固定偏置。"""
  angular_velocity = mdp.builtin_sensor(env, sensor_name)
  # 观测管理器会在 startup 之前探测形状；播放也可能没有误差事件。
  if not hasattr(env, "_go2_imu_mount_quat"):
    return angular_velocity
  return quat_apply_inverse(env._go2_imu_mount_quat, angular_velocity) + env._go2_imu_gyro_bias


def imu_projected_gravity_with_errors(env: ManagerBasedRlEnv) -> torch.Tensor:
  """使用与陀螺相同的安装误差旋转 actor 重力，保留向量模长。"""
  gravity = mdp.projected_gravity(env)
  if not hasattr(env, "_go2_imu_mount_quat"):
    return gravity
  return quat_apply_inverse(env._go2_imu_mount_quat, gravity)


@requires_model_fields("geom_solref")
def randomize_foot_contact_softness(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  time_constant_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """按环境为四足共采样正值 solref 的时间常数，保留阻尼比和 solimp。"""
  low, high = _validate_range(time_constant_range, "Contact time constant range")
  if low <= 0 or low < 2.0 * env.physics_dt:
    raise ValueError("Contact time constants must be positive and >= 2 * physics_dt")
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  asset = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
  defaults = env.sim.get_default_field("geom_solref")[geom_ids]
  if not torch.isfinite(defaults).all() or (defaults <= 0).any():
    raise ValueError("Foot softness randomization requires finite, positive-format solref")
  values = torch.empty((len(env_ids), 1), device=env.device).uniform_(low, high)
  env.sim.model.geom_solref[env_ids[:, None], geom_ids, 0] = values
