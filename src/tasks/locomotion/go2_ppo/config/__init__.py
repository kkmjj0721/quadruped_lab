# Copyright (c) 2026-2027 zh
"""
内容：
    注册 Go2 平地与粗糙地形 PPO 任务。
"""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.locomotion.go2_ppo.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .rl_cfg import unitree_go2_ppo_runner_cfg

register_mjlab_task(
  task_id = "Unitree-Go2-Rough",
  env_cfg = unitree_go2_rough_env_cfg(),
  play_env_cfg = unitree_go2_rough_env_cfg(play=True),
  rl_cfg = unitree_go2_ppo_runner_cfg(),
  runner_cls = VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id = "Unitree-Go2-Flat",
  env_cfg = unitree_go2_flat_env_cfg(),
  play_env_cfg = unitree_go2_flat_env_cfg(play=True),
  rl_cfg = unitree_go2_ppo_runner_cfg(),
  runner_cls = VelocityOnPolicyRunner,
)
