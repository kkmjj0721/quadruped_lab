# Copyright (c) 2026-2027 zh
"""内容：装配单一盲策略训练器，持久化Go2课程与任务配置，并导出部署策略。"""

from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path

import torch

from agents.supermimic.single.config import SingleConfig
from agents.supermimic.single.runner import SingleRunner
from agents.supermimic.single.export import export_policy
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg

from ..mdp.state_spec import JOINT_NAMES
from .transition_adapter import make_go2_supermimic_env


def algorithm_config(cfg):
    """在JSON覆盖后重新执行dataclass验证，不把配置方法传给算法。"""
    return SingleConfig(**{field.name: getattr(cfg.single, field.name) for field in fields(SingleConfig)})


class Go2SingleActorRunner(SingleRunner):
    def _collection_step_metrics(self):
        metrics = {}
        for name, value in self.env.last_metrics.items():
            if name.startswith("reward/"):
                term = name.removeprefix("reward/")
                metrics[f"task/reward_raw/{term}"] = value
                metrics[f"task/reward/{term}"] = self.env.step_dt * getattr(self.env.cfg.reward.scales, term) * value
            else:
                metrics[f"task/{name}"] = value
        return metrics

    def save(self, path):
        path = Path(path)
        temporary = path.with_name("." + path.name + ".go2-pending")
        try:
            super().save(temporary)
            checkpoint = torch.load(temporary, map_location="cpu", weights_only=False)
            checkpoint["go2_task"] = {"spec_hash": self.spec.hash, "curriculum": self.env.curriculum_state_dict()}
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        export_policy(self.actor, path.parent, iteration=self.iteration, metadata=self._deployment_metadata())

    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        task_state = checkpoint.get("go2_task", {})
        if task_state.get("spec_hash") != self.spec.hash or "curriculum" not in task_state:
            raise ValueError("Single-actor Go2 task metadata mismatch")
        super().load(path)
        self.env.load_curriculum_state_dict(task_state["curriculum"])

    def _deployment_metadata(self):
        control = self.env.cfg.control
        return {
            "spec_hash": self.spec.hash, "algorithm": "single_mlp",
            "observation_layout": [
                {"name": "base_angular_velocity_body", "width": 3},
                {"name": "projected_gravity_body", "width": 3},
                {"name": "joint_position_offset_with_encoder_bias", "width": 12},
                {"name": "joint_velocity", "width": 12},
                {"name": "previous_unscaled_action", "width": 12},
                {"name": "velocity_command_heading_xy_world_yaw", "width": 3},
            ],
            "joint_names": list(JOINT_NAMES), "action_joint_names": list(JOINT_NAMES),
            "default_joint_positions": self.env.task.default_joint_pos.detach().cpu().tolist(),
            "action_scales": [control.action_scales[name] for _ in range(4) for name in ("hip", "thigh", "calf")],
            "joint_target_formula": "default_joint_positions + actions * action_scales",
            "joint_target_coordinate": "encoder",
            "simulation_physical_target_formula": "default_joint_positions + actions * action_scales - encoder_bias",
            "control_dt_seconds": float(self.env.step_dt),
            "observation_semantics": "blind_proprioception_and_heading_velocity_command",
            "reference_commit": "d9eee9b3f248e5300e636c433255db2c1c3e086a",
        }


def make_single_actor_runner(cfg=None, *, device="cpu", log_dir=None, execution=None):
    cfg = cfg if cfg is not None else Go2ReferenceCfg()
    config = algorithm_config(cfg)
    torch.manual_seed(cfg.env.seed)
    env = make_go2_supermimic_env(cfg, device=device)
    try:
        runner = Go2SingleActorRunner(env, config, log_dir=log_dir,
                                     control_enabled=cfg.runner.enable_control_updates, execution=execution)
        if log_dir is not None:
            root = Path(__file__).resolve().parents[1]
            hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*.py")}
            snapshot = {"config": env.cfg.to_dict(), "spec_hash": env.task.spec.hash,
                        "geometry_hash": env.surface.geometry_hash, "source_hashes": hashes}
            destination = Path(log_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "task_config.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        return runner
    except BaseException:
        env.close()
        raise
