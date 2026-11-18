# Copyright (c) 2026-2027 zh
"""内容：装配本任务训练器，保存完整感知网络、任务课程和独立部署策略。"""

from dataclasses import asdict, fields
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import torch

from agents.depth_supermimic.config import DepthSuperMimicConfig, ExecutionConfig
from agents.depth_supermimic.export import export_policy
from agents.depth_supermimic.runner import DepthSuperMimicRunner
from src.config.go1_depth.go1_depth_config import Go1DepthCfg

from .sim_adapter import make_go1_depth_env
from .world_model import Go1WorldModel
from ..mdp.state_spec import JOINT_NAMES


def algorithm_config(cfg):
    """在JSON覆盖后统一执行算法字段验证。"""
    return DepthSuperMimicConfig(**{field.name: getattr(cfg.algo, field.name)
                                  for field in fields(DepthSuperMimicConfig)})


class Go1DepthRunner(DepthSuperMimicRunner):
    def _make_world_model(self):
        return Go1WorldModel(self.task, self.config.wm_hidden)

    def _collection_step_metrics(self):
        metrics = {}
        for name, value in self.env.last_metrics.items():
            if name.startswith("reward/"):
                term = name.removeprefix("reward/")
                weight = getattr(self.env.cfg.reward.scales, term)
                metrics[f"task/reward_raw/{term}"] = value
                metrics[f"task/reward/{term}"] = self.env.step_dt * weight * value
                metrics[f"Reward_Rate/{term}"] = weight * value
            elif name.startswith("episode_reward/"):
                metrics["Episode_Reward/" + name.removeprefix("episode_reward/")] = value
            else:
                metrics[f"task/{name}"] = value
        return metrics

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            with TemporaryDirectory(prefix=".checkpoint_", dir=path.parent) as temporary:
                temporary = Path(temporary)
                pending = temporary / path.name
                super().save(pending)
                checkpoint = torch.load(pending, map_location="cpu", weights_only=False)
                checkpoint["go1_task"] = {"spec_hash": self.spec.hash,
                                           "curriculum": self.env.curriculum_state_dict()}
                torch.save(checkpoint, pending)
                cfg = self.env.cfg
                export_policy(self.actor, self.perception, temporary, iteration=self.iteration,
                              metadata={"spec_hash": self.spec.hash, "joint_names": list(JOINT_NAMES),
                                        "action_scales": cfg.control.action_scales,
                                        "default_joint": cfg.init_state.default_joint,
                                        "camera": cfg.sensors.depth.to_dict() if hasattr(cfg.sensors.depth, "to_dict")
                                        else {key: getattr(cfg.sensors.depth, key) for key in
                                              ("width", "height", "period", "near", "far")}})
                os.replace(pending, path)
                for suffix in ("pt", "onnx", "json"):
                    os.replace(temporary / f"policy.{suffix}", path.parent / f"policy.{suffix}")
        finally:
            self.timing.record_wall("save_total", time.perf_counter() - started)

    def load(self, path, *, continue_run=False, reset_critic=False):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        task_state = checkpoint.get("go1_task", {})
        if "curriculum" not in task_state:
            raise ValueError("Go1 depth checkpoint is missing task metadata")
        if continue_run and checkpoint.get("format") == self.spec.model_id and (
            checkpoint.get("spec") != asdict(self.spec) or task_state.get("spec_hash") != self.spec.hash
        ):
            raise ValueError("Go1 task dynamics changed; use an explicit warm start instead of continue_run")
        super().load(path, continue_run=continue_run, reset_critic=reset_critic)
        if continue_run:
            self.env.load_curriculum_state_dict(task_state["curriculum"])
            self.env.resume_progress(self.iteration)
        return checkpoint


def make_runner(cfg=None, *, device="cpu", log_dir=None, execution=None):
    """训练装配失败时关闭环境，保留已有产物目录供检查。"""
    cfg = cfg if cfg is not None else Go1DepthCfg()
    config = algorithm_config(cfg)
    if execution is None:
        execution = ExecutionConfig(**{field.name: getattr(cfg.runner, field.name) for field in fields(ExecutionConfig)})
    torch.manual_seed(cfg.env.seed)
    env = make_go1_depth_env(cfg, device=device)
    try:
        runner = Go1DepthRunner(env, config, log_dir=log_dir, control_enabled=cfg.runner.enable_control_updates,
                                execution=execution, collection_groups=env.terrain_curriculum.family_ids)
        if log_dir is not None:
            destination = Path(log_dir)
            destination.mkdir(parents=True, exist_ok=True)
            snapshot = {"config": env.cfg.to_dict(), "spec_hash": env.task.spec.hash,
                        "geometry_hash": env.surface.geometry_hash}
            (destination / "task_config.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        return runner
    except BaseException:
        env.close()
        raise
