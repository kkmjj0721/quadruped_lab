# Copyright (c) 2026-2027 zh
"""
内容：
    扩展速度任务运行器，接入有序日志、保存命令课程并导出带元数据的策略。
"""

import os
import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.utils.wandb_log_writer import WandbLogWriter

from src.tasks.locomotion.go2_ppo.mdp.curriculums import RewardBasedVelocityCurriculum
from src.tasks.locomotion.go2_ppo.mdp.velocity_command import UniformVelocityCommand

from .logger import VelocityTrainingLogger


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self.logger = VelocityTrainingLogger(
      log_dir=log_dir, cfg=self.cfg, env_cfg=env.cfg, num_envs=env.num_envs,
      is_distributed=self.is_distributed, gpu_world_size=self.gpu_world_size,
      gpu_global_rank=self.gpu_global_rank, device=self.device,
    )

  def _command_curriculum(self) -> RewardBasedVelocityCurriculum | None:
    manager = self.env.unwrapped.curriculum_manager
    if "command_vel" not in manager.active_terms:
      return None
    term = manager.get_term_cfg("command_vel").func
    return term if isinstance(term, RewardBasedVelocityCurriculum) else None

  def save(self, path: str, infos=None):
    curriculum = self._command_curriculum()
    if curriculum is not None:
      infos = {**(infos or {}), "command_curriculum": curriculum.state_dict()}
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    onnx_path = os.path.join(policy_path, filename)
    writer = self.logger.writer
    use_wandb = (
      isinstance(self.logger.writer, WandbLogWriter)
      and wandb.run is not None
    )
    run_name: str = wandb.run.name if use_wandb else "local"
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if use_wandb:
      wandb.save(onnx_path, base_path=os.path.dirname(policy_path))

  def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    curriculum = self._command_curriculum()
    if curriculum is not None and infos and "command_curriculum" in infos:
      curriculum.load_state_dict(infos["command_curriculum"])
    # 新环境已经 reset，恢复 counter 后重新校准统计时钟和当前命令范围。
    command = self.env.unwrapped.command_manager.get_term("twist")
    if isinstance(command, UniformVelocityCommand):
      command.synchronize_step_counter()
      if curriculum is not None:
        self.env.unwrapped.reset()
    return infos
