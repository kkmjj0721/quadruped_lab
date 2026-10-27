# Copyright (c) 2026-2027 zh
"""
内容：
    扩展 MjLab 官方速度命令，记录地形课程需要的出生点和世界系命令位移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand as MjlabUniformVelocityCommand,
  UniformVelocityCommandCfg as MjlabUniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class UniformVelocityCommand(MjlabUniformVelocityCommand):
  """沿用官方采样、航向与可视化，只维护回合命令统计。"""

  def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.commanded_displacement_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.episode_start_pos_w = torch.zeros_like(self.commanded_displacement_w)
    self.episode_elapsed_s = torch.zeros(self.num_envs, device=self.device)
    self._command_velocity_w = torch.zeros_like(self.commanded_displacement_w)
    self._last_statistics_step = torch.full(
      (self.num_envs,), env.common_step_counter, device=self.device, dtype=torch.long,
    )
    self._pending_start = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

  def synchronize_step_counter(self) -> None:
    """续训恢复全局计数后同步时钟，避免把检查点前的时间计入新回合。"""
    self._last_statistics_step.fill_(self._env.common_step_counter)

  def update_episode_statistics(self, env_ids: torch.Tensor | None = None) -> None:
    """积分上次命令实际保持的时间；课程读取前调用可补齐终止步且不重复累计。"""
    ids = slice(None) if env_ids is None else env_ids
    elapsed = (self._env.common_step_counter - self._last_statistics_step[ids]).clamp(min=0)
    self.commanded_displacement_w[ids] += self._command_velocity_w[ids] * elapsed.unsqueeze(-1) * self._env.step_dt
    self.episode_elapsed_s[ids] += elapsed * self._env.step_dt
    self._last_statistics_step[ids] = self._env.common_step_counter

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    self.commanded_displacement_w[env_ids] = 0.0
    self.episode_elapsed_s[env_ids] = 0.0
    self._command_velocity_w[env_ids] = 0.0
    self._last_statistics_step[env_ids] = self._env.common_step_counter
    self._pending_start[env_ids] = True
    return extras

  def compute(self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    self.update_episode_statistics(env_ids)
    # reset 事件后尚未 forward 的派生位姿不可用，延迟到此处捕获出生位置。
    fresh = self._pending_start.clone()
    if env_ids is not None:
      fresh[:] = False
      fresh[env_ids] = self._pending_start[env_ids]
    self.episode_start_pos_w[fresh] = self.robot.data.root_link_pos_w[fresh, :2]
    self._pending_start[fresh] = False
    super().compute(dt, env_ids)
    # 当前命令将保持至下一控制步；保存当前航向，避免转向后再旋转历史命令。
    heading = self.robot.data.heading_w
    cos_h, sin_h = torch.cos(heading), torch.sin(heading)
    vx, vy = self.command[:, 0], self.command[:, 1]
    self._command_velocity_w[:, 0] = cos_h * vx - sin_h * vy
    self._command_velocity_w[:, 1] = sin_h * vx + cos_h * vy


@dataclass(kw_only=True)
class UniformVelocityCommandCfg(MjlabUniformVelocityCommandCfg):
  """保留官方配置接口，构建带课程统计的速度命令。"""

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommand:
    return UniformVelocityCommand(self, env)
