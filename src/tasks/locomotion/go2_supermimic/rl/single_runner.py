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
    """在JSON覆盖后重新执行dataclass验证，把任务侧驱动配置并入算法预算。"""
    driver = cfg.runner.driver
    values = {field.name: getattr(cfg.single, field.name) for field in fields(SingleConfig)}
    values.update(driver_frac=driver.frac, expert_frac=driver.expert_frac,
                  driver_fade=tuple(driver.fade_iterations),
                  driver_vmin=driver.speed_min, driver_vmax=driver.speed_max)
    return SingleConfig(**values)


class Go2SingleActorRunner(SingleRunner):
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
            position, yaw = self.env.planar_pose()
            command = self.steer(self.env.step_dt, position, yaw, self._arena_center())
            self.env.set_commands(command)
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
        if self.steer is not None:
            self.steer.reset(done, self._arena_center())

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
            "spec_hash": self.spec.hash, "algorithm": "moe_history_encoder",
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
            "history_encoder": {"type": "mixture_of_experts", "input": "five_frame_blind_history",
                                "latent_dim": int(self.config.moe_latent_dim),
                                "experts": int(self.config.moe_expert_num),
                                "gate": "softmax_over_experts",
                                "current_frame_fused_with_latent": True},
            "velocity_input": "not_observed_estimated_from_history",
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
        if cfg.runner.driver.policy:
            from .driver import ChaseSteer, MoeCtsDriver
            runner.driver = MoeCtsDriver(cfg.runner.driver.policy, env.num_envs, env.task.default_joint_pos,
                                         cfg.control.action_scales, runner.device)
            runner.steer = ChaseSteer(env.num_envs, runner.device,
                                      (config.driver_vmin, config.driver_vmax), 0.8 * cfg.env.scene_extent / 2)
        if cfg.runner.driver.expert_policy:
            from .driver import ActorDriver
            runner.expert = ActorDriver(cfg.runner.driver.expert_policy, env.task.spec.observation_dim,
                                        env.task.spec.action_dim, runner.device)
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
