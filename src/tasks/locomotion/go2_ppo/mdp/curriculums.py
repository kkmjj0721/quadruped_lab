# Copyright (c) 2026-2027 zh
"""
内容：
    按命令位移调整地形难度，并按回合跟踪表现渐进扩大速度范围。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

import torch
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")
_VELOCITY_AXES = ("lin_vel_x", "lin_vel_y", "ang_vel_z")


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
  promotion_fraction: float = 0.8,
  min_commanded_distance: float = 1.0,
) -> dict[str, torch.Tensor]:
  """以真实出生点和命令净位移评估地形；早退仍承担剩余回合预算。"""
  if not 0.5 < promotion_fraction <= 1.0 or not math.isfinite(min_commanded_distance) or min_commanded_distance < 0:
    raise ValueError("Terrain curriculum requires 0.5 < promotion_fraction <= 1 and a nonnegative minimum distance.")
  terrain = env.scene.terrain
  assert terrain is not None and terrain.cfg.terrain_generator is not None
  command_term = env.command_manager.get_term(command_name)
  if not isinstance(command_term, UniformVelocityCommand):
    raise TypeError("Terrain curriculum requires a velocity command with episode statistics.")
  command_term.update_episode_statistics(env_ids)
  distance = torch.norm(
    env.scene[asset_cfg.name].data.root_link_pos_w[env_ids, :2] - command_term.episode_start_pos_w[env_ids], dim=1,
  )
  time_left = (env.max_episode_length_s - command_term.episode_elapsed_s[env_ids]).clamp(min=0.0)
  commanded_distance = torch.norm(command_term.commanded_displacement_w[env_ids], dim=1)
  commanded_distance += torch.norm(command_term.command[env_ids, :2], dim=1) * time_left
  move_up = distance > torch.clamp(
    commanded_distance * promotion_fraction, max=terrain.cfg.terrain_generator.size[0] / 2,
  )
  move_up &= commanded_distance >= min_commanded_distance
  valid = (command_term.episode_elapsed_s[env_ids] > 0) & (env.common_step_counter > 0)
  move_up &= valid
  move_down = (distance < commanded_distance * 0.5) & ~move_up & valid
  terrain.update_env_origins(env_ids, move_up, move_down)
  levels = terrain.terrain_levels.float()
  return {"mean": levels.mean(), "max": levels.max()}


class RewardBasedVelocityCurriculum(ManagerTermBase):
  """累计已结束回合的线速度跟踪分数，达标后只扩大 vx 范围。"""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    params = cfg.params
    self._command = env.command_manager.get_term(params.get("command_name", "twist"))
    assert self._command is not None
    self._threshold = params.get("linear_threshold", 0.8)
    self._increment = params.get("linear_increment", 0.25)
    self._limit = tuple(params["lin_vel_x_limit"])
    self._reward_name = params.get("linear_reward_name", "track_linear_velocity")
    interval = params.get("check_interval_s")
    interval = env.max_episode_length_s if interval is None else interval
    if not 0 < self._threshold <= 1 or not math.isfinite(interval) or interval <= 0:
      raise ValueError("Velocity curriculum requires 0 < linear_threshold <= 1 and a positive check interval.")
    if not math.isfinite(self._increment) or self._increment < 0:
      raise ValueError("Velocity curriculum linear_increment must be finite and nonnegative.")
    lower, upper = self._limit
    if not all(math.isfinite(value) for value in self._limit) or lower > upper:
      raise ValueError("Invalid velocity curriculum lin_vel_x_limit.")
    initial = self._command.cfg.ranges.lin_vel_x
    if not lower <= initial[0] <= initial[1] <= upper:
      raise ValueError("Initial lin_vel_x must be inside its curriculum limit.")
    if self._increment > 0:
      self._tracking_weight()
    self._interval_steps = max(1, math.ceil(interval / env.step_dt))
    self._last_update_step = env.common_step_counter
    self._score_sum = torch.zeros((), device=env.device)
    self._score = torch.zeros_like(self._score_sum)
    self._episode_count = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None = None,
    **params,
  ) -> dict[str, torch.Tensor]:
    del params  # manager 重复传入构造时已解析的配置。
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    ids = env_ids[env.episode_length_buf[env_ids] > 0]
    if len(ids) > 0:
      if self._increment > 0:
        budget = env.max_episode_length * (env.step_dt if env.cfg.scale_rewards_by_dt else 1.0)
        # MjLab 1.6.0 未公开回合奖励读取接口，须在 reward_manager.reset 前取值。
        rewards = env.reward_manager._episode_sums[self._reward_name][ids]
        self._score_sum += rewards.sum() / (budget * self._tracking_weight())
      self._episode_count += len(ids)
    if self._episode_count > 0 and env.common_step_counter - self._last_update_step >= self._interval_steps:
      self._score.copy_(self._score_sum / self._episode_count)
      if self._increment > 0 and self._score > self._threshold:
        lower, upper = self._command.cfg.ranges.lin_vel_x
        self._command.cfg.ranges.lin_vel_x = (
          max(self._limit[0], lower - self._increment),
          min(self._limit[1], upper + self._increment),
        )
      self._score_sum.zero_()
      self._episode_count = 0
      self._last_update_step = env.common_step_counter
    return self._metrics()

  def _tracking_weight(self) -> float:
    weight = self._env.reward_manager.get_term_cfg(self._reward_name).weight
    if not math.isfinite(weight) or weight <= 0:
      raise ValueError(f"Velocity curriculum requires a positive tracking reward weight: {self._reward_name}.")
    return weight

  def _metrics(self) -> dict[str, torch.Tensor]:
    result = {
      f"{axis}_{bound}": torch.tensor(getattr(self._command.cfg.ranges, axis)[index], device=self.device)
      for axis in _VELOCITY_AXES for index, bound in enumerate(("min", "max"))
    }
    result["linear_tracking"] = self._score.clone()
    return result

  def state_dict(self) -> dict:
    """保存 vx 进度及窗口统计，固定轴与进行中的物理回合不随课程恢复。"""
    return {
      "ranges": {"lin_vel_x": tuple(self._command.cfg.ranges.lin_vel_x)},
      "last_update_step": self._last_update_step,
      "score_sums": [self._score_sum.item()],
      "scores": [self._score.item()],
      "episode_count": self._episode_count,
    }

  def load_state_dict(self, state: dict) -> None:
    """兼容旧多轴／双槽状态，只读取 vx 范围和首个线速度统计槽。"""
    bounds = state["ranges"]["lin_vel_x"]
    score_sum, score = state["score_sums"][0], state["scores"][0]
    last_step, episode_count = int(state["last_update_step"]), int(state["episode_count"])
    if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]:
      raise ValueError("Invalid saved curriculum range for lin_vel_x.")
    if not all(math.isfinite(value) for value in (score_sum, score)) or last_step < 0 or episode_count < 0:
      raise ValueError("Invalid saved velocity curriculum statistics.")
    if self._increment > 0:
      lower, upper = self._limit
      self._command.cfg.ranges.lin_vel_x = tuple(max(lower, min(upper, value)) for value in bounds)
      self._score_sum.fill_(score_sum)
      self._score.fill_(score)
    else:
      self._score_sum.zero_()
      self._score.zero_()
    self._last_update_step = last_step
    self._episode_count = episode_count


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """按训练步数阶段调整奖励权重。"""
  del env_ids
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight], device=env.device)
