# Copyright (c) 2026-2027 zh
"""内容：装配 Go2 SuperMimic 训练器，持久化课程与任务配置快照。"""

from dataclasses import fields
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from agents.supermimic.config import SuperMimicConfig
from agents.supermimic.export import export_policy_onnx
from agents.supermimic.runner import SuperMimicRunner
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from .transition_adapter import make_go2_supermimic_env
from .deployment import deployment_metadata


def algorithm_config(cfg):
    """在JSON覆盖后重新执行dataclass验证，把任务侧驱动配置并入算法预算。"""
    driver = cfg.runner.driver
    values = {field.name: getattr(cfg.algo, field.name) for field in fields(SuperMimicConfig)}
    values.update(driver_frac=driver.frac, expert_frac=driver.expert_frac,
                  driver_fade=tuple(driver.fade_iterations),
                  driver_vmin=driver.speed_min, driver_vmax=driver.speed_max)
    return SuperMimicConfig(**values)


class Go2SuperMimicRunner(SuperMimicRunner):
    _driver_metrics = {}

    def _collection_step_metrics(self):
        metrics = {}
        for name, value in self.env.last_metrics.items():
            if name.startswith("reward/"):
                term = name.removeprefix("reward/")
                metrics[f"task/reward_raw/{term}"] = value
                metrics[f"task/reward/{term}"] = self.env.step_dt * getattr(self.env.cfg.reward.scales, term) * value
            else:
                metrics[f"task/{name}"] = value
        if self.driver is not None or self.expert is not None:
            metrics.update(self._driver_metrics)
        return metrics

    def _override_actions(self, action, history, observation):
        """按参考实现从环境尾部切出专家与驱动区，其余环境保留当前策略动作。

        驱动区由预训练策略行走，专家区由历史MLP策略行走，两者都只贡献真实数据，
        不参与策略更新；占比随迭代线性衰减到零。
        """
        cfg = self.config
        if self.driver is None and self.expert is None:
            return action
        scale = self._driver_scale()
        num_envs = self.env.num_envs
        driver_count = int(scale * cfg.driver_frac * num_envs) if self.driver is not None else 0
        expert_count = int(scale * cfg.expert_frac * num_envs) if self.expert is not None else 0
        index = torch.arange(num_envs, device=self.device)
        driven = index >= num_envs - driver_count
        expert = (index >= num_envs - driver_count - expert_count) & ~driven
        if driven.any():
            # 盲速度任务以HIM命令为目标，外部驱动只接管动作，不改变其余环境的命令。
            action = torch.where(driven[:, None], self.driver.act(observation), action)
        if expert.any():
            action = torch.where(expert[:, None], self.expert.act(observation), action)
        if driven.any() or expert.any():
            self._driver_metrics = {"driver/env_count": float(driver_count), "driver/expert_env_count": float(expert_count)}
        return action

    def _arena_center(self):
        """追逐目标以场地中心为基准；平地模式原点为零。"""
        return torch.zeros(self.env.num_envs, 2, device=self.device)

    def _reset_drivers(self, done):
        if self.driver is not None:
            self.driver.reset(done)
        if self.expert is not None:
            self.expert.reset(done)
        if self.steer is not None:
            self.steer.reset(done, self._arena_center())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 先完成两种产物的生成与校验，再发布；导出失败不覆盖上次可用模型和ONNX。
        with TemporaryDirectory(prefix=".checkpoint_", dir=path.parent) as temporary:
            temporary = Path(temporary)
            pending = temporary / path.name
            super().save(pending)
            checkpoint = torch.load(pending, map_location="cpu", weights_only=False)
            checkpoint["go2_task"] = {"spec_hash": self.spec.hash, "curriculum": self.env.curriculum_state_dict()}
            torch.save(checkpoint, pending)
            export_policy_onnx(self.actor, temporary / "policy.onnx", iteration=self.iteration,
                               metadata=deployment_metadata(self.env.cfg, spec_hash=self.spec.hash,
                                                            algorithm=self.config))
            os.replace(pending, path)
            os.replace(temporary / "policy.onnx", path.parent / "policy.onnx")

    def load(self, path, *, continue_run=False, reset_critic=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        task_state = checkpoint.get("go2_task", {})
        if "curriculum" not in task_state:
            raise ValueError("Go2 SuperMimic checkpoint task metadata mismatch")
        super().load(path, continue_run=continue_run, reset_critic=reset_critic)
        if continue_run:
            self.env.resume_progress(self.iteration)
        return checkpoint


def make_go2_supermimic_runner(cfg=None, *, device="cpu", log_dir=None, execution=None):
    """构造任务环境、可选驱动策略和任务快照，失败时释放仿真资源。"""
    cfg = cfg if cfg is not None else Go2ReferenceCfg()
    config = algorithm_config(cfg)
    torch.manual_seed(cfg.env.seed)
    env = make_go2_supermimic_env(cfg, device=device)
    try:
        runner = Go2SuperMimicRunner(env, config, log_dir=log_dir,
                                     control_enabled=cfg.runner.enable_control_updates, execution=execution)
        if cfg.runner.driver.policy:
            from .driver import MoeCtsDriver
            runner.driver = MoeCtsDriver(cfg.runner.driver.policy, env.num_envs, env.task.default_joint_pos,
                                         cfg.control.action_scales, runner.device)
        if cfg.runner.driver.expert_policy:
            from .driver import ActorDriver
            runner.expert = ActorDriver(cfg.runner.driver.expert_policy, env.task.spec.observation_dim,
                                        env.task.spec.action_dim, runner.device)
        if log_dir is not None:
            snapshot = {"config": env.cfg.to_dict(), "spec_hash": env.task.spec.hash,
                        "geometry_hash": env.surface.geometry_hash}
            destination = Path(log_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "task_config.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        return runner
    except BaseException:
        env.close()
        raise
